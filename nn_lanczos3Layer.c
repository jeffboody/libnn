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
#include "../texgz/pil_lanczos.h"
#include "nn_arch.h"
#include "nn_lanczos3Layer.h"
#include "nn_engine.h"
#include "nn_layer.h"
#include "nn_tensor.h"

/***********************************************************
* private                                                  *
***********************************************************/

typedef struct
{
	uint32_t stride;
} nn_lanczos3LayerParam_t;

static nn_tensor_t*
nn_lanczos3Layer_computeFpFn(nn_layer_t* base,
                             int flags, uint32_t bs,
                             nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_lanczos3Layer_t* self   = (nn_lanczos3Layer_t*) base;
	nn_arch_t*          arch   = base->arch;
	nn_engine_t*        engine = arch->engine;

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

	// nn_lanczos3Layer_forwardPassH
	// dispatch(RAW, bs, xh, yw, 1, 8, 8)
	vkk_computePipeline_t* cp;
	cp = engine->cp_lanczos3_forwardPassH;
	if(nn_engine_computeBind(engine, cp) == 0)
	{
		return NULL;
	}
	vkk_compute_bindUniformSets(engine->compute, 2, us_array);
	nn_engine_computeDispatch(engine, VKK_HAZARD_RAW,
	                          bs, dimX->height, dimY->width,
	                          1, 8, 8);

	// nn_lanczos3Layer_forwardPassY
	// dispatch(RAW, bs, yh, yw, 1, 8, 8)
	cp = engine->cp_lanczos3_forwardPassY;
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
nn_lanczos3Layer_computeBpFn(nn_layer_t* base,
                             int flags, uint32_t bs,
                             nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,yh,yw,xd)

	nn_lanczos3Layer_t* self   = (nn_lanczos3Layer_t*) base;
	nn_arch_t*          arch   = base->arch;
	nn_engine_t*        engine = arch->engine;

	nn_dim_t* dimX = nn_tensor_dim(self->dL_dX);
	nn_dim_t* dimW = nn_tensor_dim(self->W);
	nn_dim_t* dimY = nn_tensor_dim(dL_dY);
	uint32_t  sz   = dimW->depth;

	// clear backprop gradients
	if(nn_tensor_computeFill(self->dL_dH, VKK_HAZARD_NONE,
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

	// nn_lanczos3Layer_backprop_dL_dH
	// dispatch required for each n
	// dispatch(RAW, bs, yh, yw, 1, 8, 8)
	uint n;
	vkk_computePipeline_t* cp;
	cp = engine->cp_lanczos3_backprop_dL_dH;
	if(nn_engine_computeBind(engine, cp) == 0)
	{
		return NULL;
	}
	for(n = 0; n < sz; ++n)
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

	// nn_lanczos3Layer_backprop_dL_dX
	// dispatch required for each n
	// dispatch(RAW, bs, xh, yw, 1, 8, 8)
	cp = engine->cp_lanczos3_backprop_dL_dX;
	if(nn_engine_computeBind(engine, cp) == 0)
	{
		return NULL;
	}
	for(n = 0; n < sz; ++n)
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
nn_lanczos3Layer_postFn(nn_layer_t* base,
                        int flags, uint32_t bs)
{
	// ignore
}

static nn_dim_t*
nn_lanczos3Layer_dimXFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_lanczos3Layer_t* self = (nn_lanczos3Layer_t*) base;

	return nn_tensor_dim(self->dL_dX);
}

static nn_dim_t*
nn_lanczos3Layer_dimYFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_lanczos3Layer_t* self = (nn_lanczos3Layer_t*) base;

	return nn_tensor_dim(self->Y);
}

static int
nn_lanczos3Layer_newW(nn_lanczos3Layer_t* self,
                      nn_engine_t* engine)
{
	ASSERT(self);
	ASSERT(engine);

	// lanczos3 properties
	int   scale   = cc_pow2n(self->level);
	float support = 3.0f;
	float scalef  = (float) scale;
	int   n       = (int) (scalef*support + 0.01f);
	int   sz      = 2*n;

	nn_dim_t dimW =
	{
		.count  = 1,
		.height = 1,
		.width  = 1,
		.depth  = sz,
	};

	nn_tensor_t* Wio;
	Wio = nn_tensor_new(engine, &dimW, NN_TENSOR_INIT_ZERO,
	                    NN_TENSOR_MODE_IO);
	if(Wio == NULL)
	{
		return 0;
	}

	self->W = nn_tensor_new(engine, &dimW,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_COMPUTE);
	if(self->W == NULL)
	{
		goto fail_W;
	}

	// generate masks (weights)
	// for example
	// 1: 0.007,  0.030,
	//   -0.068, -0.133,
	//    0.270,  0.890,
	// 2: 0.002,  0.016,  0.030,  0.020,
	//   -0.031, -0.105, -0.147, -0.085,
	//    0.121,  0.437,  0.764,  0.971,
	float step = 1.0f/scalef;
	float x    = support - step/2.0f;
	float y;
	int   i0;
	int   i1 = sz - 1;
	for(i0 = 0; i0 < n; ++i0)
	{
		y = pil_lanczos3_filter(x)/scale;
		nn_tensor_ioSet(Wio, 0, 0, 0, i0, y);
		nn_tensor_ioSet(Wio, 0, 0, 0, i1, y);
		x -= step;
		--i1;
	}

	if(nn_tensor_copy(Wio, self->W, 0, 0, 1) == 0)
	{
		goto fail_copy;
	}

	nn_tensor_delete(&Wio);

	// success
	return 1;

	// failure
	fail_copy:
	fail_W:
		nn_tensor_delete(&Wio);
	return 0;
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_lanczos3Us2Data_t*
nn_lanczos3Us2Data_new(nn_engine_t* engine,
                       nn_lanczos3Us2Key_t* key)
{
	ASSERT(engine);
	ASSERT(key);

	nn_lanczos3Us2Data_t* self;
	self = (nn_lanczos3Us2Data_t*)
	       CALLOC(1, sizeof(nn_lanczos3Us2Data_t));
	if(self == NULL)
	{
		LOGE("CALLOC failed");
		return NULL;
	}

	self->sb200 = vkk_buffer_new(engine->engine,
	                             VKK_UPDATE_MODE_STATIC,
	                             VKK_BUFFER_USAGE_STORAGE,
	                             sizeof(nn_lanczos3Us2Key_t),
	                             key);
	if(self->sb200 == NULL)
	{
		goto failure;
	}

	self->us2 = vkk_uniformSet_new(engine->engine, 2, 0, NULL,
	                               engine->usf2_lanczos3);
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
		nn_lanczos3Us2Data_delete(&self);
	return NULL;
}

void nn_lanczos3Us2Data_delete(nn_lanczos3Us2Data_t** _self)
{
	ASSERT(_self);

	nn_lanczos3Us2Data_t* self = *_self;
	if(self)
	{
		vkk_uniformSet_delete(&self->us2);
		vkk_buffer_delete(&self->sb200);
		FREE(self);
		*_self = NULL;
	}
}

nn_lanczos3Layer_t*
nn_lanczos3Layer_new(nn_arch_t* arch, nn_dim_t* dimX,
                     int level)
{
	ASSERT(arch);
	ASSERT(dimX);

	nn_engine_t* engine = arch->engine;

	uint32_t stride = cc_pow2n(level);
	uint32_t bs     = dimX->count;
	uint32_t xh     = dimX->height;
	uint32_t xw     = dimX->width;
	uint32_t xd     = dimX->depth;
	uint32_t yh     = xh/stride;
	uint32_t yw     = xw/stride;

	nn_dim_t dimH =
	{
		.count  = bs,
		.height = xh,
		.width  = yw,
		.depth  = xd,
	};

	nn_dim_t dimY =
	{
		.count  = bs,
		.height = yh,
		.width  = yw,
		.depth  = xd,
	};

	nn_layerInfo_t info =
	{
		.arch          = arch,
		.compute_fp_fn = nn_lanczos3Layer_computeFpFn,
		.compute_bp_fn = nn_lanczos3Layer_computeBpFn,
		.post_fn       = nn_lanczos3Layer_postFn,
		.dimX_fn       = nn_lanczos3Layer_dimXFn,
		.dimY_fn       = nn_lanczos3Layer_dimYFn,
	};

	nn_lanczos3Layer_t* self;
	self = (nn_lanczos3Layer_t*)
	       nn_layer_new(sizeof(nn_lanczos3Layer_t), &info);
	if(self == NULL)
	{
		return NULL;
	}

	self->level = level;

	self->H = nn_tensor_new(engine, &dimH,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_COMPUTE);
	if(self->H == NULL)
	{
		goto failure;
	}

	if(nn_lanczos3Layer_newW(self, engine) == 0)
	{
		goto failure;
	}

	self->Y = nn_tensor_new(engine, &dimY,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_COMPUTE);
	if(self->Y == NULL)
	{
		goto failure;
	}

	self->dL_dH = nn_tensor_new(engine, &dimH,
	                            NN_TENSOR_INIT_ZERO,
	                            NN_TENSOR_MODE_COMPUTE);
	if(self->dL_dH == NULL)
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

	nn_lanczos3LayerParam_t param =
	{
		.stride = stride,
	};
	self->sb008_param = vkk_buffer_new(engine->engine,
	                                   VKK_UPDATE_MODE_STATIC,
	                                   VKK_BUFFER_USAGE_STORAGE,
	                                   sizeof(nn_lanczos3LayerParam_t),
	                                   &param);
	if(self->sb008_param == NULL)
	{
		goto failure;
	}

	self->us0 = vkk_uniformSet_new(engine->engine, 0, 0, NULL,
	                               engine->usf0_lanczos3);
	if(self->us0 == NULL)
	{
		goto failure;
	}

	self->us1_fp = vkk_uniformSet_new(engine->engine, 1, 0, NULL,
	                                  engine->usf1_lanczos3_fp);
	if(self->us1_fp == NULL)
	{
		goto failure;
	}

	self->us1_bp = vkk_uniformSet_new(engine->engine, 1, 0, NULL,
	                                  engine->usf1_lanczos3_bp);
	if(self->us1_bp == NULL)
	{
		goto failure;
	}

	// sb000: dimX (bs,xh,xw,xd)
	// sb001: H    (bs,xh,yw,xd)
	// sb002: dimW (1,1,1,sz)
	// sb003: W
	// sb004: dimY (bs,yh,yw,xd)
	// sb005: Y
	// sb006: dL_dH
	// sb007: dL_dX
	// sb008: param (stride)
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
			.buffer  = self->H->sb_data,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->W->sb_dim,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->W->sb_data,
		},
		{
			.binding = 4,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->Y->sb_dim,
		},
		{
			.binding = 5,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->Y->sb_data,
		},
		{
			.binding = 6,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->dL_dH->sb_data,
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

	// success
	return self;

	// failure
	failure:
		nn_layer_delete((nn_layer_t**) &self);
	return NULL;
}

void nn_lanczos3Layer_delete(nn_lanczos3Layer_t** _self)
{
	ASSERT(_self);

	nn_lanczos3Layer_t* self = *_self;
	if(self)
	{
		vkk_uniformSet_delete(&self->us1_bp);
		vkk_uniformSet_delete(&self->us1_fp);
		vkk_uniformSet_delete(&self->us0);
		vkk_buffer_delete(&self->sb008_param);
		nn_tensor_delete(&self->dL_dX);
		nn_tensor_delete(&self->dL_dH);
		nn_tensor_delete(&self->Y);
		nn_tensor_delete(&self->W);
		nn_tensor_delete(&self->H);
		nn_layer_delete((nn_layer_t**) _self);
	}
}

nn_lanczos3Layer_t*
nn_lanczos3Layer_import(nn_arch_t* arch, jsmn_val_t* val)
{
	ASSERT(arch);
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	jsmn_val_t* val_dimX  = NULL;
	jsmn_val_t* val_level = NULL;
	jsmn_val_t* val_W     = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		jsmn_keyval_t* kv;
		kv = (jsmn_keyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == JSMN_TYPE_PRIMITIVE)
		{
			if(strcmp(kv->key, "level") == 0)
			{
				val_level = kv->val;
			}
		}
		else if(kv->val->type == JSMN_TYPE_OBJECT)
		{
			if(strcmp(kv->key, "dimX") == 0)
			{
				val_dimX = kv->val;
			}
			else if(strcmp(kv->key, "W") == 0)
			{
				val_W = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_dimX  == NULL) ||
	   (val_level == NULL) ||
	   (val_W     == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	int level = strtol(val_level->data, NULL, 0);

	nn_dim_t dimX;
	if(nn_dim_import(&dimX, val_dimX) == 0)
	{
		return NULL;
	}

	nn_lanczos3Layer_t* self;
	self = nn_lanczos3Layer_new(arch, &dimX, level);
	if(self == NULL)
	{
		return NULL;
	}

	if(nn_tensor_import(self->W,  val_W)  == 0)
	{
		goto fail_tensor;
	}

	// success
	return self;

	// failure
	fail_tensor:
		nn_lanczos3Layer_delete(&self);
	return NULL;
}

int nn_lanczos3Layer_export(nn_lanczos3Layer_t* self,
                            jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	nn_dim_t* dimX = nn_tensor_dim(self->dL_dX);

	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	ret &= jsmn_stream_key(stream, "%s", "dimX");
	ret &= nn_dim_export(dimX, stream);
	ret &= jsmn_stream_key(stream, "%s", "level");
	ret &= jsmn_stream_int(stream, self->level);
	ret &= jsmn_stream_key(stream, "%s", "W");
	ret &= nn_tensor_export(self->W, stream);
	ret &= jsmn_stream_end(stream);

	return ret;
}
