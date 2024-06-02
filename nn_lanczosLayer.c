/*
 * Copyright (c) 2024 Jeff Boody
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define LOG_TAG "nn"
#include "../libcc/math/cc_pow2n.h"
#include "../libcc/cc_log.h"
#include "../libcc/cc_memory.h"
#include "../libvkk/vkk.h"
#include "nn_arch.h"
#include "nn_lanczosResampler.h"
#include "nn_lanczosLayer.h"
#include "nn_engine.h"
#include "nn_layer.h"
#include "nn_tensor.h"

/***********************************************************
* private                                                  *
***********************************************************/

static nn_tensor_t*
nn_lanczosLayer_computeFpFn(nn_layer_t* base,
                            int flags, uint32_t bs,
                            nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_lanczosLayer_t* self   = (nn_lanczosLayer_t*) base;
	nn_arch_t*         arch   = base->arch;
	nn_engine_t*       engine = arch->engine;

	nn_dim_t* dimX = nn_tensor_dim(X);
	nn_dim_t* dimY = nn_tensor_dim(self->Y);

	// sb100: bs
	// sb101: state
	// sb102: X
	vkk_uniformAttachment_t ua1_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = arch->sb100_bs,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = arch->sb101_state,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X->sb_data,
		},
	};

	vkk_compute_updateUniformSetRefs(engine->compute,
	                                 self->us1_fp, 3,
	                                 ua1_array);

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1_fp,
	};

	// nn_lanczosLayer_forwardPassT
	// dispatch(RAW, bs, xh, yw, 1, 8, 8)
	vkk_computePipeline_t* cp;
	cp = engine->cp_lanczos_forwardPassT;
	if(nn_engine_computeBind(engine, cp) == 0)
	{
		return NULL;
	}
	vkk_compute_bindUniformSets(engine->compute, 2, us_array);
	nn_engine_computeDispatch(engine, VKK_HAZARD_RAW,
	                          bs, dimX->height, dimY->width,
	                          1, 8, 8);

	// nn_lanczosLayer_forwardPassY
	// dispatch(RAW, bs, yh, yw, 1, 8, 8)
	cp = engine->cp_lanczos_forwardPassY;
	if(nn_engine_computeBind(engine, cp) == 0)
	{
		return NULL;
	}
	nn_engine_computeDispatch(engine, VKK_HAZARD_RAW,
	                          bs, dimY->height, dimY->width,
	                          1, 8, 8);

	// store reference
	self->X = X;

	return self->Y;
}

static nn_tensor_t*
nn_lanczosLayer_computeBpFn(nn_layer_t* base,
                            int flags, uint32_t bs,
                            nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,yh,yw,xd)

	nn_lanczosLayer_t* self   = (nn_lanczosLayer_t*) base;
	nn_lanczosParam_t* param  = &self->param;
	nn_arch_t*         arch   = base->arch;
	nn_engine_t*       engine = arch->engine;

	nn_dim_t* dimX = nn_tensor_dim(self->dL_dX);
	nn_dim_t* dimY = nn_tensor_dim(dL_dY);

	// clear backprop gradients
	if(nn_tensor_computeFill(self->dL_dT, VKK_HAZARD_NONE,
	                         0, bs, 0.0f) == 0)
	{
		return NULL;
	}
	if(nn_tensor_computeFill(self->dL_dX, VKK_HAZARD_NONE,
	                         0, bs, 0.0f) == 0)
	{
		return NULL;
	}

	// sb100: bs
	// sb101: state
	// sb102: dL_dY
	vkk_uniformAttachment_t ua1_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = arch->sb100_bs,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = arch->sb101_state,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY->sb_data,
		},
	};

	vkk_compute_updateUniformSetRefs(engine->compute,
	                                 self->us1_bp, 3,
	                                 ua1_array);

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1_bp,
		NULL,
	};

	// nn_lanczosLayer_backprop_dL_dT
	// dispatch required for each n
	// dispatch(RAW, bs, yh, yw, 1, 8, 8)
	uint n;
	vkk_computePipeline_t* cp;
	cp = engine->cp_lanczos_backprop_dL_dT;
	if(nn_engine_computeBind(engine, cp) == 0)
	{
		return NULL;
	}
	for(n = 0; n < param->szh; ++n)
	{
		us_array[2] = nn_engine_getLanczos3Us2(engine, n);
		if(us_array[2] == NULL)
		{
			return NULL;
		}
		vkk_compute_bindUniformSets(engine->compute, 3,
		                            us_array);
		nn_engine_computeDispatch(engine, VKK_HAZARD_RAW,
		                          bs, dimY->height, dimY->width,
		                          1, 8, 8);
	}

	// nn_lanczosLayer_backprop_dL_dX
	// dispatch required for each n
	// dispatch(RAW, bs, xh, yw, 1, 8, 8)
	cp = engine->cp_lanczos_backprop_dL_dX;
	if(nn_engine_computeBind(engine, cp) == 0)
	{
		return NULL;
	}
	for(n = 0; n < param->szw; ++n)
	{
		us_array[2] = nn_engine_getLanczos3Us2(engine, n);
		if(us_array[2] == NULL)
		{
			return NULL;
		}
		vkk_compute_bindUniformSets(engine->compute, 3,
		                            us_array);
		nn_engine_computeDispatch(engine, VKK_HAZARD_RAW,
		                          bs, dimX->height, dimY->width,
		                          1, 8, 8);
	}

	return self->dL_dX;
}

static void
nn_lanczosLayer_postFn(nn_layer_t* base,
                       int flags, uint32_t bs)
{
	// ignore
}

static nn_dim_t*
nn_lanczosLayer_dimXFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_lanczosLayer_t* self = (nn_lanczosLayer_t*) base;

	return nn_tensor_dim(self->dL_dX);
}

static nn_dim_t*
nn_lanczosLayer_dimYFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_lanczosLayer_t* self = (nn_lanczosLayer_t*) base;

	return nn_tensor_dim(self->Y);
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_lanczosUs2Data_t*
nn_lanczosUs2Data_new(nn_engine_t* engine,
                      nn_lanczosUs2Key_t* key)
{
	ASSERT(engine);
	ASSERT(key);

	nn_lanczosUs2Data_t* self;
	self = (nn_lanczosUs2Data_t*)
	       CALLOC(1, sizeof(nn_lanczosUs2Data_t));
	if(self == NULL)
	{
		LOGE("CALLOC failed");
		return NULL;
	}

	self->sb200 = vkk_buffer_new(engine->engine,
	                             VKK_UPDATE_MODE_STATIC,
	                             VKK_BUFFER_USAGE_STORAGE,
	                             sizeof(nn_lanczosUs2Key_t),
	                             key);
	if(self->sb200 == NULL)
	{
		goto failure;
	}

	self->us2 = vkk_uniformSet_new(engine->engine, 2, 0, NULL,
	                               engine->usf2_lanczos);
	if(self->us2 == NULL)
	{
		goto failure;
	}

	vkk_uniformAttachment_t ua2_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->sb200,
		},
	};

	vkk_compute_updateUniformSetRefs(engine->compute,
	                                 self->us2, 1,
	                                 ua2_array);

	// success
	return self;

	// failure
	failure:
		nn_lanczosUs2Data_delete(&self);
	return NULL;
}

void nn_lanczosUs2Data_delete(nn_lanczosUs2Data_t** _self)
{
	ASSERT(_self);

	nn_lanczosUs2Data_t* self = *_self;
	if(self)
	{
		vkk_uniformSet_delete(&self->us2);
		vkk_buffer_delete(&self->sb200);
		FREE(self);
		*_self = NULL;
	}
}

nn_lanczosLayer_t*
nn_lanczosLayer_new(nn_arch_t* arch, nn_dim_t* dimX,
                    nn_dim_t* dimY, int a)
{
	ASSERT(arch);
	ASSERT(dimX);

	nn_engine_t* engine = arch->engine;

	// compute Lanczos param and kernel
	nn_lanczosResampler_t* lanczos;
	lanczos = nn_lanczosResampler_new(engine, dimX, dimY, a);
	if(lanczos == NULL)
	{
		return NULL;
	}

	nn_layerInfo_t info =
	{
		.arch          = arch,
		.compute_fp_fn = nn_lanczosLayer_computeFpFn,
		.compute_bp_fn = nn_lanczosLayer_computeBpFn,
		.post_fn       = nn_lanczosLayer_postFn,
		.dimX_fn       = nn_lanczosLayer_dimXFn,
		.dimY_fn       = nn_lanczosLayer_dimYFn,
	};

	nn_lanczosLayer_t* self;
	self = (nn_lanczosLayer_t*)
	       nn_layer_new(sizeof(nn_lanczosLayer_t), &info);
	if(self == NULL)
	{
		goto failure;
	}

	nn_lanczosParam_copy(&lanczos->param, &self->param);

	nn_dim_t dimT =
	{
		.count  = dimX->count,
		.height = dimX->height,
		.width  = dimY->width,
		.depth  = dimX->depth,
	};

	self->T = nn_tensor_new(engine, &dimT,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_COMPUTE);
	if(self->T == NULL)
	{
		goto failure;
	}

	self->Y = nn_tensor_new(engine, dimY,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_COMPUTE);
	if(self->Y == NULL)
	{
		goto failure;
	}

	nn_dim_t* dimLw = nn_tensor_dim(lanczos->Lw);

	self->Lw = nn_tensor_new(engine, dimLw,
	                         NN_TENSOR_INIT_ZERO,
	                         NN_TENSOR_MODE_COMPUTE);
	if(self->Lw == NULL)
	{
		goto failure;
	}

	// copy IO to COMPUTE
	if(nn_tensor_copy(lanczos->Lw, self->Lw, 0, 0,
	                  dimLw->count) == 0)
	{
		goto failure;
	}

	nn_dim_t* dimLh = nn_tensor_dim(lanczos->Lh);

	self->Lh = nn_tensor_new(engine, dimLh,
	                         NN_TENSOR_INIT_ZERO,
	                         NN_TENSOR_MODE_COMPUTE);
	if(self->Lh == NULL)
	{
		goto failure;
	}

	// copy IO to COMPUTE
	if(nn_tensor_copy(lanczos->Lh, self->Lh, 0, 0,
	                  dimLh->count) == 0)
	{
		goto failure;
	}

	self->dL_dT = nn_tensor_new(engine, &dimT,
	                            NN_TENSOR_INIT_ZERO,
	                            NN_TENSOR_MODE_COMPUTE);
	if(self->dL_dT == NULL)
	{
		goto failure;
	}

	self->dL_dX = nn_tensor_new(engine, dimX,
	                            NN_TENSOR_INIT_ZERO,
	                            NN_TENSOR_MODE_COMPUTE);
	if(self->dL_dX == NULL)
	{
		goto failure;
	}

	self->sb008_param = vkk_buffer_new(engine->engine,
	                                   VKK_UPDATE_MODE_STATIC,
	                                   VKK_BUFFER_USAGE_STORAGE,
	                                   sizeof(nn_lanczosParam_t),
	                                   &self->param);
	if(self->sb008_param == NULL)
	{
		goto failure;
	}

	self->us0 = vkk_uniformSet_new(engine->engine, 0, 0, NULL,
	                               engine->usf0_lanczos);
	if(self->us0 == NULL)
	{
		goto failure;
	}

	self->us1_fp = vkk_uniformSet_new(engine->engine, 1, 0, NULL,
	                                  engine->usf1_lanczos_fp);
	if(self->us1_fp == NULL)
	{
		goto failure;
	}

	self->us1_bp = vkk_uniformSet_new(engine->engine, 1, 0, NULL,
	                                  engine->usf1_lanczos_bp);
	if(self->us1_bp == NULL)
	{
		goto failure;
	}

	// sb000: dimX (bs,xh,xw,xd)
	// sb001: T    (bs,xh,yw,xd)
	// sb002: dimY (bs,yh,yw,xd)
	// sb003: Y
	// sb004: Lw
	// sb005: Lh
	// sb006: dL_dW
	// sb007: dL_dX
	// sb008: param (a, fsw, fsh, fcw, fch, szw, szh)
	vkk_uniformAttachment_t ua0_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->dL_dX->sb_dim,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->T->sb_data,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->Y->sb_dim,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->Y->sb_data,
		},
		{
			.binding = 4,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->Lw->sb_data,
		},
		{
			.binding = 5,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->Lh->sb_data,
		},
		{
			.binding = 6,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->dL_dT->sb_data,
		},
		{
			.binding = 7,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->dL_dX->sb_data,
		},
		{
			.binding = 8,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->sb008_param,
		},
	};

	vkk_compute_updateUniformSetRefs(engine->compute,
	                                 self->us0, 9,
	                                 ua0_array);

	nn_lanczosResampler_delete(&lanczos);

	// success
	return self;

	// failure
	failure:
	{
		nn_layer_delete((nn_layer_t**) &self);
		nn_lanczosResampler_delete(&lanczos);
	}
	return NULL;
}

void nn_lanczosLayer_delete(nn_lanczosLayer_t** _self)
{
	ASSERT(_self);

	nn_lanczosLayer_t* self = *_self;
	if(self)
	{
		vkk_uniformSet_delete(&self->us1_bp);
		vkk_uniformSet_delete(&self->us1_fp);
		vkk_uniformSet_delete(&self->us0);
		vkk_buffer_delete(&self->sb008_param);
		nn_tensor_delete(&self->dL_dX);
		nn_tensor_delete(&self->dL_dT);
		nn_tensor_delete(&self->Lh);
		nn_tensor_delete(&self->Lw);
		nn_tensor_delete(&self->Y);
		nn_tensor_delete(&self->T);
		nn_layer_delete((nn_layer_t**) _self);
	}
}

nn_lanczosLayer_t*
nn_lanczosLayer_import(nn_arch_t* arch, jsmn_val_t* val)
{
	ASSERT(arch);
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	jsmn_val_t* val_a    = NULL;
	jsmn_val_t* val_dimX = NULL;
	jsmn_val_t* val_dimY = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		jsmn_keyval_t* kv;
		kv = (jsmn_keyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == JSMN_TYPE_PRIMITIVE)
		{
			if(strcmp(kv->key, "a") == 0)
			{
				val_a = kv->val;
			}
		}
		else if(kv->val->type == JSMN_TYPE_OBJECT)
		{
			if(strcmp(kv->key, "dimX") == 0)
			{
				val_dimX = kv->val;
			}
			else if(strcmp(kv->key, "dimY") == 0)
			{
				val_dimY = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_a    == NULL) ||
	   (val_dimX == NULL) ||
	   (val_dimY == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	int a = (int) strtol(val_a->data, NULL, 0);

	nn_dim_t dimX;
	if(nn_dim_import(&dimX, val_dimX) == 0)
	{
		return NULL;
	}

	nn_dim_t dimY;
	if(nn_dim_import(&dimY, val_dimY) == 0)
	{
		return NULL;
	}

	nn_lanczosLayer_t* self;
	self = nn_lanczosLayer_new(arch, &dimX, &dimY, a);
	if(self == NULL)
	{
		return NULL;
	}

	return self;
}

int nn_lanczosLayer_export(nn_lanczosLayer_t* self,
                           jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	nn_dim_t* dimX = nn_tensor_dim(self->dL_dX);
	nn_dim_t* dimY = nn_tensor_dim(self->Y);

	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	ret &= jsmn_stream_key(stream, "%s", "dimX");
	ret &= nn_dim_export(dimX, stream);
	ret &= jsmn_stream_key(stream, "%s", "dimY");
	ret &= nn_dim_export(dimY, stream);
	ret &= jsmn_stream_end(stream);

	return ret;
}
