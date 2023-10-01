/*
 * Copyright (c) 2023 Jeff Boody
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
#include "../libcc/cc_log.h"
#include "../libcc/cc_memory.h"
#include "nn_arch.h"
#include "nn_poolingLayer.h"
#include "nn_layer.h"
#include "nn_tensor.h"

const char* NN_POOLING_LAYER_STRING_MAX     = "max";
const char* NN_POOLING_LAYER_STRING_AVERAGE = "average";

/***********************************************************
* private                                                  *
***********************************************************/

// protected
extern void
nn_arch_dispatch(nn_arch_t* self,
                 vkk_hazzard_e hazzard,
                 uint32_t count_x,
                 uint32_t count_y,
                 uint32_t count_z,
                 uint32_t local_size_x,
                 uint32_t local_size_y,
                 uint32_t local_size_z);
extern int
nn_arch_bind(nn_arch_t* self,
             vkk_computePipeline_t* cp);

typedef struct
{
	uint32_t stride;
} nn_poolingLayerParam_t;

static nn_tensor_t*
nn_poolingLayer_forwardPassFn(nn_layer_t* base,
                              nn_layerMode_e mode,
                              uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_poolingLayer_t* self  = (nn_poolingLayer_t*) base;
	nn_arch_t*         arch  = base->arch;
	nn_tensor_t*       Y     = self->Y;
	nn_tensor_t*       dY_dX = self->dY_dX;
	nn_dim_t*          dimY  = nn_tensor_dim(Y);

	vkk_computePipeline_t* cp[NN_POOLING_LAYER_MODE_COUNT] =
	{
		arch->cp_pooling_forwardPassMax,
		arch->cp_pooling_forwardPassAvg,
	};

	// clear forward gradients
	if(self->mode == NN_POOLING_LAYER_MODE_MAX)
	{
		nn_tensor_clear(dY_dX, NN_TENSOR_HAZZARD_NONE);
	}

	// sb00: state
	// sb01: param (stride)
	// sb02: dim_dY_dX
	// sb03: dY_dX
	vkk_uniformAttachment_t ua0_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = arch->sb_state,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->sb01_param,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dY_dX->sb_dim,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dY_dX->sb_data,
		},
	};

	// sb10: dimX
	// sb11: X
	// sb12: dimY
	// sb13: Y
	vkk_uniformAttachment_t ua1_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X->sb_dim,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X->sb_data,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Y->sb_dim,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Y->sb_data,
		},
	};

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1,
	};

	// nn_poolingLayer_forwardPass
	// dispatch(RAW, bs, yh, yw, 1, 8, 8)
	if(nn_arch_bind(arch, cp[self->mode]) == 0)
	{
		return NULL;
	}
	vkk_compute_updateUniformSetRefs(arch->compute, self->us0,
	                                 4, ua0_array);
	vkk_compute_updateUniformSetRefs(arch->compute, self->us1,
	                                 4, ua1_array);
	vkk_compute_bindUniformSets(arch->compute, 2, us_array);
	nn_arch_dispatch(arch, VKK_HAZZARD_RAW,
	                 bs, dimY->height, dimY->width,
	                 1, 8, 8);

	return Y;
}

static nn_tensor_t*
nn_poolingLayer_backpropFn(nn_layer_t* base, uint32_t bs,
                           nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,yh,yw,xd)

	nn_poolingLayer_t* self  = (nn_poolingLayer_t*) base;
	nn_arch_t*         arch  = base->arch;
	nn_tensor_t*       dL_dX = self->dL_dX;
	nn_dim_t*          dimX  = nn_tensor_dim(dL_dX);

	// sb20: dim_dL_dY
	// sb21: dL_dY
	// sb22: dim_dL_dX
	// sb23: dL_dX
	vkk_uniformAttachment_t ua2_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY->sb_dim,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY->sb_data,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dX->sb_dim,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dX->sb_data,
		},
	};

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1,
		self->us2,
	};

	// nn_poolingLayer_backprop
	// dispatch(RAW, bs, xh, xw, 1, 8, 8)
	vkk_computePipeline_t* cp;
	cp = arch->cp_pooling_backprop;
	if(nn_arch_bind(arch, cp) == 0)
	{
		return NULL;
	}
	vkk_compute_updateUniformSetRefs(arch->compute, self->us2,
	                                 4, ua2_array);
	vkk_compute_bindUniformSets(arch->compute, 3, us_array);
	nn_arch_dispatch(arch, VKK_HAZZARD_RAW,
	                 bs, dimX->height, dimX->width,
	                 1, 8, 8);

	return dL_dX;
}

static int
nn_poolingLayer_newCompute(nn_poolingLayer_t* self)
{
	ASSERT(self);

	nn_arch_t* arch = self->base.arch;

	self->us0 = vkk_uniformSet_new(arch->engine, 0, 0, NULL,
	                               arch->usf0_pooling);
	if(self->us0 == NULL)
	{
		return 0;
	}

	self->us1 = vkk_uniformSet_new(arch->engine, 1, 0, NULL,
	                               arch->usf1_pooling);
	if(self->us1 == NULL)
	{
		goto fail_us1;
	}

	self->us2 = vkk_uniformSet_new(arch->engine, 2, 0, NULL,
	                               arch->usf2_pooling);
	if(self->us2 == NULL)
	{
		goto fail_us2;
	}

	nn_poolingLayerParam_t param =
	{
		.stride = self->stride,
	};
	self->sb01_param = vkk_buffer_new(arch->engine,
	                                  VKK_UPDATE_MODE_STATIC,
	                                  VKK_BUFFER_USAGE_STORAGE,
	                                  sizeof(nn_poolingLayerParam_t),
	                                  &param);
	if(self->sb01_param == NULL)
	{
		goto fail_sb01_param;
	}

	// success
	return 1;

	// failure
	fail_sb01_param:
		vkk_uniformSet_delete(&self->us2);
	fail_us2:
		vkk_uniformSet_delete(&self->us1);
	fail_us1:
		vkk_uniformSet_delete(&self->us0);
	return 0;
}

static void
nn_poolingLayer_deleteCompute(nn_poolingLayer_t* self)
{
	ASSERT(self);

	vkk_buffer_delete(&self->sb01_param);
	vkk_uniformSet_delete(&self->us2);
	vkk_uniformSet_delete(&self->us1);
	vkk_uniformSet_delete(&self->us0);
}

static nn_dim_t*
nn_poolingLayer_dimXFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_poolingLayer_t* self = (nn_poolingLayer_t*) base;

	return nn_tensor_dim(self->dL_dX);
}

static nn_dim_t*
nn_poolingLayer_dimYFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_poolingLayer_t* self = (nn_poolingLayer_t*) base;

	return nn_tensor_dim(self->Y);
}

static nn_poolingLayerMode_e
nn_poolingLayer_mode(const char* str)
{
	ASSERT(str);

	const char* mode_fn[NN_POOLING_LAYER_MODE_COUNT] =
	{
		NN_POOLING_LAYER_STRING_MAX,
		NN_POOLING_LAYER_STRING_AVERAGE,
	};

	int i;
	for(i = 0; i < NN_POOLING_LAYER_MODE_COUNT; ++i)
	{
		if(strcmp(str, mode_fn[i]) == 0)
		{
			return (nn_poolingLayerMode_e) i;
		}
	}

	LOGE("invalid %s", str);
	return NN_POOLING_LAYER_MODE_ERROR;
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_poolingLayer_t*
nn_poolingLayer_new(nn_arch_t* arch, nn_dim_t* dimX,
                    uint32_t stride,
                    nn_poolingLayerMode_e mode)
{
	ASSERT(arch);
	ASSERT(dimX);

	if(((int) mode < 0) ||
	   ((int) mode >= NN_POOLING_LAYER_MODE_COUNT))
	{
		LOGE("invalid mode=%i", (int) mode);
	}

	nn_layerInfo_t info =
	{
		.arch            = arch,
		.forward_pass_fn = nn_poolingLayer_forwardPassFn,
		.backprop_fn     = nn_poolingLayer_backpropFn,
		.dimX_fn         = nn_poolingLayer_dimXFn,
		.dimY_fn         = nn_poolingLayer_dimYFn,
	};

	nn_poolingLayer_t* self;
	self = (nn_poolingLayer_t*)
	       nn_layer_new(sizeof(nn_poolingLayer_t), &info);
	if(self == NULL)
	{
		return NULL;
	}

	self->stride = stride;
	self->mode   = mode;

	nn_dim_t dimY =
	{
		.count  = dimX->count,
		.height = dimX->height/stride,
		.width  = dimX->width/stride,
		.depth  = dimX->depth,
	};

	self->Y = nn_tensor_new(arch, &dimY,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_COMPUTE);
	if(self->Y == NULL)
	{
		goto fail_Y;
	}

	self->dY_dX = nn_tensor_new(arch, dimX,
	                            NN_TENSOR_INIT_ZERO,
	                            NN_TENSOR_MODE_COMPUTE);
	if(self->dY_dX == NULL)
	{
		goto fail_dY_dX;
	}

	self->dL_dX = nn_tensor_new(arch, dimX,
	                            NN_TENSOR_INIT_ZERO,
	                            NN_TENSOR_MODE_COMPUTE);
	if(self->dL_dX == NULL)
	{
		goto fail_dL_dX;
	}

	if(nn_poolingLayer_newCompute(self) == 0)
	{
		goto fail_compute;
	}

	// success
	return self;

	// failure
	fail_compute:
		nn_tensor_delete(&self->dL_dX);
	fail_dL_dX:
		nn_tensor_delete(&self->dY_dX);
	fail_dY_dX:
		nn_tensor_delete(&self->Y);
	fail_Y:
		nn_layer_delete((nn_layer_t**) &self);
	return NULL;
}

nn_poolingLayer_t*
nn_poolingLayer_import(nn_arch_t* arch, jsmn_val_t* val)
{
	ASSERT(arch);
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	jsmn_val_t* val_dimX   = NULL;
	jsmn_val_t* val_stride = NULL;
	jsmn_val_t* val_mode   = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		jsmn_keyval_t* kv;
		kv = (jsmn_keyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == JSMN_TYPE_OBJECT)
		{
			if(strcmp(kv->key, "dimX") == 0)
			{
				val_dimX = kv->val;
			}
		}
		else if(kv->val->type == JSMN_TYPE_PRIMITIVE)
		{
			if(strcmp(kv->key, "stride") == 0)
			{
				val_stride = kv->val;
			}
		}
		else if(kv->val->type == JSMN_TYPE_STRING)
		{
			if(strcmp(kv->key, "mode") == 0)
			{
				val_mode = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_dimX   == NULL) ||
	   (val_stride == NULL) ||
	   (val_mode   == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_dim_t dimX;
	if(nn_dim_load(&dimX, val_dimX) == 0)
	{
		return NULL;
	}

	uint32_t stride;
	stride = (uint32_t) strtol(val_stride->data, NULL, 0);

	nn_poolingLayerMode_e mode;
	mode = nn_poolingLayer_mode(val_mode->data);
	return nn_poolingLayer_new(arch, &dimX, stride, mode);
}

int nn_poolingLayer_export(nn_poolingLayer_t* self,
                           jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	nn_dim_t* dimX = nn_tensor_dim(self->dL_dX);

	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	ret &= jsmn_stream_key(stream, "%s", "dimX");
	ret &= nn_dim_store(dimX, stream);
	ret &= jsmn_stream_key(stream, "%s", "stride");
	ret &= jsmn_stream_int(stream, (int) self->stride);
	ret &= jsmn_stream_key(stream, "%s", "mode");
	if(self->mode == NN_POOLING_LAYER_MODE_AVERAGE)
	{
		ret &= jsmn_stream_string(stream, "%s", "average");
	}
	else
	{
		ret &= jsmn_stream_string(stream, "%s", "max");
	}
	ret &= jsmn_stream_end(stream);

	return ret;
}

void nn_poolingLayer_delete(nn_poolingLayer_t** _self)
{
	ASSERT(_self);

	nn_poolingLayer_t* self = *_self;
	if(self)
	{
		nn_poolingLayer_deleteCompute(self);
		nn_tensor_delete(&self->dL_dX);
		nn_tensor_delete(&self->dY_dX);
		nn_tensor_delete(&self->Y);
		nn_layer_delete((nn_layer_t**) _self);
	}
}
