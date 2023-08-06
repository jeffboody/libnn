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

/***********************************************************
* private                                                  *
***********************************************************/

#ifdef NN_USE_COMPUTE

static nn_tensor_t*
nn_poolingLayer_forwardPassFn(nn_layer_t* base, int mode,
                              uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_poolingLayer_t* self  = (nn_poolingLayer_t*) base;
	nn_arch_t*         arch  = base->arch;
	nn_tensor_t*       Y     = self->Y;
	nn_tensor_t*       dY_dX = self->dY_dX;
	nn_dim_t*          dimY  = nn_tensor_dim(Y);

	vkk_computePipeline_t* cp;
	if(self->mode == NN_POOLING_LAYER_MODE_AVERAGE)
	{
		cp = arch->cp_pooling_forwardPassAvg;
	}
	else if(self->mode == NN_POOLING_LAYER_MODE_MAX)
	{
		cp = arch->cp_pooling_forwardPassMax;
	}
	else
	{
		LOGE("invalid");
		return NULL;
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

	vkk_compute_bindComputePipeline(arch->compute, cp);
	vkk_compute_updateUniformSetRefs(arch->compute, self->us0,
	                                 4, ua0_array);
	vkk_compute_updateUniformSetRefs(arch->compute, self->us1,
	                                 4, ua1_array);
	vkk_compute_bindUniformSets(arch->compute, 2, us_array);
	vkk_compute_dispatch(arch->compute, VKK_HAZZARD_RAW,
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

	vkk_compute_bindComputePipeline(arch->compute,
	                                arch->cp_pooling_backprop);
	vkk_compute_updateUniformSetRefs(arch->compute, self->us2,
	                                 4, ua2_array);
	vkk_compute_bindUniformSets(arch->compute, 3, us_array);
	vkk_compute_dispatch(arch->compute, VKK_HAZZARD_RAW,
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

	self->sb01_param = vkk_buffer_new(arch->engine,
	                                  VKK_UPDATE_MODE_STATIC,
	                                  VKK_BUFFER_USAGE_STORAGE,
	                                  sizeof(float),
	                                  &self->stride);
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

#else // NN_USE_COMPUTE not defined

typedef void (*nn_poolingLayer_fn)(nn_poolingLayer_t* self,
                                   nn_tensor_t* X,
                                   uint32_t m, uint32_t i,
                                   uint32_t j, uint32_t k);

static void
nn_poolingLayer_max(nn_poolingLayer_t* self,
                    nn_tensor_t* X,
                    uint32_t m, uint32_t i,
                    uint32_t j, uint32_t k)
{
	ASSERT(self);
	ASSERT(X);

	nn_tensor_t* Y     = self->Y;
	nn_tensor_t* dY_dX = self->dY_dX;
	nn_dim_t*    dimX  = nn_tensor_dim(X);

	// compute range
	uint32_t stride = self->stride;
	uint32_t xh     = dimX->height;
	uint32_t xw     = dimX->width;
	uint32_t ii0    = stride*i;
	uint32_t jj0    = stride*j;
	uint32_t ii1    = ii0 + stride;
	uint32_t jj1    = jj0 + stride;
	if(ii1 > xh)
	{
		ii1 = xh;
	}

	if(jj1 > xw)
	{
		jj1 = xw;
	}

	// initialize max value
	float    x;
	float    xmax  = nn_tensor_get(X, m, ii0, jj0, k);
	uint32_t iimax = ii0;
	uint32_t jjmax = jj0;

	// find max value in tile
	uint32_t ii;
	uint32_t jj;
	for(ii = ii0; ii < ii1; ++ii)
	{
		for(jj = jj0; jj < jj1; ++jj)
		{
			x = nn_tensor_get(X, m, ii, jj, k);
			if(x > xmax)
			{
				xmax  = x;
				iimax = ii;
				jjmax = jj;
			}
		}
	}

	// output
	nn_tensor_set(Y, m, i, j, k, xmax);

	// forward gradients
	nn_tensor_set(dY_dX, m, iimax, jjmax, k, 1.0f);
}

static void
nn_poolingLayer_avg(nn_poolingLayer_t* self,
                    nn_tensor_t* X,
                    uint32_t m, uint32_t i,
                    uint32_t j, uint32_t k)
{
	ASSERT(self);
	ASSERT(X);

	nn_tensor_t* Y     = self->Y;
	nn_tensor_t* dY_dX = self->dY_dX;
	nn_dim_t*    dimX  = nn_tensor_dim(X);

	// compute range
	uint32_t stride = self->stride;
	uint32_t xh     = dimX->height;
	uint32_t xw     = dimX->width;
	uint32_t ii0    = stride*i;
	uint32_t jj0    = stride*j;
	uint32_t ii1    = ii0 + stride;
	uint32_t jj1    = jj0 + stride;
	if(ii1 > xh)
	{
		ii1 = xh;
	}

	if(jj1 > xw)
	{
		jj1 = xw;
	}

	// initalize average
	float di  = (float) (ii1 - ii0);
	float dj  = (float) (jj1 - jj0);
	float s   = 1.0f/(di*dj);
	float avg = 0.0f;

	// compute average
	uint32_t ii;
	uint32_t jj;
	for(ii = ii0; ii < ii1; ++ii)
	{
		for(jj = jj0; jj < jj1; ++jj)
		{
			// update sum
			avg += nn_tensor_get(X, m, ii, jj, k);

			// forward gradients
			nn_tensor_set(dY_dX, m, ii, jj, k, s);
		}
	}
	avg *= s;

	// output
	nn_tensor_set(Y, m, i, j, k, avg);
}

static nn_tensor_t*
nn_poolingLayer_forwardPassFn(nn_layer_t* base, int mode,
                              uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_poolingLayer_t* self = (nn_poolingLayer_t*) base;

	nn_tensor_t* Y      = self->Y;
	nn_tensor_t* dY_dX  = self->dY_dX;
	uint32_t     stride = self->stride;
	nn_dim_t*    dimY   = nn_tensor_dim(Y);
	uint32_t     yh     = dimY->height;
	uint32_t     yw     = dimY->width;
	uint32_t     xd     = dimY->depth;

	nn_poolingLayer_fn fn = nn_poolingLayer_avg;
	if(self->mode == NN_POOLING_LAYER_MODE_MAX)
	{
		// clear forward gradients
		nn_tensor_clear(dY_dX);

		fn = nn_poolingLayer_max;
	}

	// output and forward gradients
	uint32_t m;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(m = 0; m < bs; ++m)
	{
		for(i = 0; i < yh; i += stride)
		{
			for(j = 0; j < yw; j += stride)
			{
				for(k = 0; k < xd; ++k)
				{
					(*fn)(self, X, m, i, j, k);
				}
			}
		}
	}

	return Y;
}

static nn_tensor_t*
nn_poolingLayer_backpropFn(nn_layer_t* base, uint32_t bs,
                           nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,yh,yw,xd)

	nn_poolingLayer_t* self = (nn_poolingLayer_t*) base;

	nn_tensor_t* dY_dX  = self->dY_dX;
	nn_tensor_t* dL_dX  = self->dL_dX;
	uint32_t     stride = self->stride;
	nn_dim_t*    dimX   = nn_tensor_dim(dL_dX);
	uint32_t     xh     = dimX->height;
	uint32_t     xw     = dimX->width;
	uint32_t     xd     = dimX->depth;

	// backpropagate loss
	float    dy_dx;
	float    dl_dx;
	float    dl_dy;
	uint32_t m;
	uint32_t ii;
	uint32_t jj;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(m = 0; m < bs; ++m)
	{
		for(ii = 0; ii < xh; ++ii)
		{
			i = ii/stride;
			for(jj = 0; jj < xw; ++jj)
			{
				j = jj/stride;
				for(k = 0; k < xd; ++k)
				{
					dl_dy = nn_tensor_get(dL_dY, m, i, j, k);
					dy_dx = nn_tensor_get(dY_dX, m, ii, jj, k);
					dl_dx = dl_dy*dy_dx;
					nn_tensor_set(dL_dX, m, ii, jj, k, dl_dx);
				}
			}
		}
	}
	return dL_dX;
}

static int
nn_poolingLayer_newCompute(nn_poolingLayer_t* self)
{
	return 1;
}

static void
nn_poolingLayer_deleteCompute(nn_poolingLayer_t* self)
{
}

#endif

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

/***********************************************************
* public                                                   *
***********************************************************/

nn_poolingLayer_t*
nn_poolingLayer_new(nn_arch_t* arch, nn_dim_t* dimX,
                    uint32_t stride, int mode)
{
	ASSERT(arch);
	ASSERT(dimX);

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

	int mode = NN_POOLING_LAYER_MODE_MAX;
	if(strcmp(val_mode->data, "average") == 0)
	{
		mode = NN_POOLING_LAYER_MODE_AVERAGE;
	}

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
