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
#include "nn_factLayer.h"
#include "nn_layer.h"
#include "nn_tensor.h"

const char* NN_FACT_LAYER_STRING_LINEAR    = "linear";
const char* NN_FACT_LAYER_STRING_LOGISTIC  = "logistic";
const char* NN_FACT_LAYER_STRING_RELU      = "ReLU";
const char* NN_FACT_LAYER_STRING_PRELU     = "PReLU";
const char* NN_FACT_LAYER_STRING_TANH      = "tanh";
const char* NN_FACT_LAYER_STRING_DLINEAR   = "dlinear";
const char* NN_FACT_LAYER_STRING_DLOGISTIC = "dlogistic";
const char* NN_FACT_LAYER_STRING_DRELU     = "dReLU";
const char* NN_FACT_LAYER_STRING_DPRELU    = "dPReLU";
const char* NN_FACT_LAYER_STRING_DTANH     = "dtanh";

/***********************************************************
* private                                                  *
***********************************************************/

#ifdef NN_USE_COMPUTE

static nn_tensor_t*
nn_factLayer_forwardPassFn(nn_layer_t* base, int mode,
                           uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_factLayer_t* self = (nn_factLayer_t*) base;
	nn_arch_t*      arch = base->arch;
	nn_tensor_t*    Y    = self->Y;
	nn_dim_t*       dimX = nn_tensor_dim(X);

	vkk_computePipeline_t* cp[NN_FACT_LAYER_FN_COUNT] =
	{
		arch->cp_fact_forwardPassLinear,
		arch->cp_fact_forwardPassLogistic,
		arch->cp_fact_forwardPassReLU,
		arch->cp_fact_forwardPassPReLU,
		arch->cp_fact_forwardPassTanh,
	};

	// sb00: dimX
	// sb01: X
	vkk_uniformAttachment_t ua0_array[] =
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
	};

	// sb10: dimY
	// sb11: Y
	vkk_uniformAttachment_t ua1_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Y->sb_dim,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Y->sb_data,
		},
	};

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1,
	};

	// nn_factLayer_forwardPass
	// dispatch(RAW, bs, xh, xw, 1, 8, 8)
	vkk_compute_bindComputePipeline(arch->compute,
	                                cp[self->fn]);
	vkk_compute_updateUniformSetRefs(arch->compute, self->us0,
	                                 2, ua0_array);
	vkk_compute_updateUniformSetRefs(arch->compute, self->us1,
	                                 2, ua1_array);
	vkk_compute_bindUniformSets(arch->compute, 2, us_array);
	vkk_compute_dispatch(arch->compute, VKK_HAZZARD_RAW,
	                     bs, dimX->height, dimX->width,
	                     1, 8, 8);

	// reference for backprop
	self->X = X;

	return Y;
}

static nn_tensor_t*
nn_factLayer_backpropFn(nn_layer_t* base, uint32_t bs,
                        nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY);

	nn_factLayer_t* self = (nn_factLayer_t*) base;
	nn_arch_t*      arch = base->arch;
	nn_dim_t*       dimX = nn_tensor_dim(self->X);

	vkk_computePipeline_t* cp[NN_FACT_LAYER_FN_COUNT] =
	{
		arch->cp_fact_backpropLinear,
		arch->cp_fact_backpropLogistic,
		arch->cp_fact_backpropReLU,
		arch->cp_fact_backpropPReLU,
		arch->cp_fact_backpropTanh,
	};

	// sb20: dim_dL_dY
	// sb21: dL_dY
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
	};

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1,
		self->us2,
	};

	// nn_factLayer_backprop
	// dispatch(RAW, bs, xh, xw, 1, 8, 8)
	vkk_compute_bindComputePipeline(arch->compute,
	                                cp[self->fn]);
	vkk_compute_updateUniformSetRefs(arch->compute, self->us2,
	                                 2, ua2_array);
	vkk_compute_bindUniformSets(arch->compute, 3, us_array);
	vkk_compute_dispatch(arch->compute, VKK_HAZZARD_RAW,
	                     bs, dimX->height, dimX->width,
	                     1, 8, 8);

	// dL_dY replaced by dL_dX
	return dL_dY;
}

static int nn_factLayer_newCompute(nn_factLayer_t* self)
{
	ASSERT(self);

	nn_arch_t* arch = self->base.arch;

	self->us0 = vkk_uniformSet_new(arch->engine, 0, 0, NULL,
	                               arch->usf0_fact);
	if(self->us0 == NULL)
	{
		return 0;
	}

	self->us1 = vkk_uniformSet_new(arch->engine, 1, 0, NULL,
	                               arch->usf1_fact);
	if(self->us1 == NULL)
	{
		goto fail_us1;
	}

	self->us2 = vkk_uniformSet_new(arch->engine, 2, 0, NULL,
	                               arch->usf2_fact);
	if(self->us2 == NULL)
	{
		goto fail_us2;
	}

	// success
	return 1;

	// failure
	fail_us2:
		vkk_uniformSet_delete(&self->us1);
	fail_us1:
		vkk_uniformSet_delete(&self->us0);
	return 0;
}

static void
nn_factLayer_deleteCompute(nn_factLayer_t* self)
{
	ASSERT(self);

	vkk_uniformSet_delete(&self->us2);
	vkk_uniformSet_delete(&self->us1);
	vkk_uniformSet_delete(&self->us0);
}

#else // NN_USE_COMPUTE not defined

typedef float (*nn_factLayer_fn)(float x);

static float nn_factLayer_linear(float x)
{
	return x;
}

static float nn_factLayer_logistic(float x)
{
	return 1.0f/(1.0f + exp(-x));
}

static float nn_factLayer_ReLU(float x)
{
	if(x < 0.0f)
	{
		return 0.0f;
	}

	return x;
}

static float nn_factLayer_PReLU(float x)
{
	if(x < 0.0f)
	{
		return 0.01f*x;
	}

	return x;
}

static float nn_factLayer_tanh(float x)
{
	return tanhf(x);
}

static float nn_factLayer_dlinear(float x)
{
	return 1.0f;
}

static float nn_factLayer_dlogistic(float x)
{
	float fx = nn_factLayer_logistic(x);
	return fx*(1.0f - fx);
}

static float nn_factLayer_dReLU(float x)
{
	if(x < 0.0f)
	{
		return 0.0f;
	}

	return 1.0f;
}

static float nn_factLayer_dPReLU(float x)
{
	if(x < 0.0f)
	{
		return 0.01f;
	}

	return 1.0f;
}

static float nn_factLayer_dtanh(float x)
{
	float tanhfx = tanhf(x);
	return 1.0f - tanhfx*tanhfx;
}

static nn_tensor_t*
nn_factLayer_forwardPassFn(nn_layer_t* base, int mode,
                           uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_factLayer_t* self = (nn_factLayer_t*) base;

	nn_tensor_t* Y     = self->Y;
	nn_dim_t*    dimX  = nn_tensor_dim(X);
	uint32_t     xh    = dimX->height;
	uint32_t     xw    = dimX->width;
	uint32_t     xd    = dimX->depth;

	nn_factLayer_fn fact_fn[NN_FACT_LAYER_FN_COUNT] =
	{
		nn_factLayer_linear,
		nn_factLayer_logistic,
		nn_factLayer_ReLU,
		nn_factLayer_PReLU,
		nn_factLayer_tanh,
	};

	// output and forward gradients
	float    x;
	uint32_t m;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(m = 0; m < bs; ++m)
	{
		for(i = 0; i < xh; ++i)
		{
			for(j = 0; j < xw; ++j)
			{
				for(k = 0; k < xd; ++k)
				{
					// output
					x = nn_tensor_get(X, m, i, j, k);
					nn_tensor_set(Y, m, i, j, k,
					              (*fact_fn[self->fn])(x));
				}
			}
		}
	}

	// reference for backprop
	self->X = X;

	return Y;
}

static nn_tensor_t*
nn_factLayer_backpropFn(nn_layer_t* base, uint32_t bs,
                        nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,xh,xw,xd)

	nn_factLayer_t* self = (nn_factLayer_t*) base;

	nn_tensor_t* X    = self->X;
	nn_dim_t*    dimX = nn_tensor_dim(X);
	uint32_t     xh   = dimX->height;
	uint32_t     xw   = dimX->width;
	uint32_t     xd   = dimX->depth;

	nn_factLayer_fn dfact_fn[NN_FACT_LAYER_FN_COUNT] =
	{
		nn_factLayer_dlinear,
		nn_factLayer_dlogistic,
		nn_factLayer_dReLU,
		nn_factLayer_dPReLU,
		nn_factLayer_dtanh,
	};

	// backpropagate loss
	float    dy_dx;
	float    x;
	uint32_t m;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(m = 0; m < bs; ++m)
	{
		for(i = 0; i < xh; ++i)
		{
			for(j = 0; j < xw; ++j)
			{
				for(k = 0; k < xd; ++k)
				{
					// forward gradient
					x     = nn_tensor_get(X, m, i, j, k);
					dy_dx = (*dfact_fn[self->fn])(x);

					// dL_dY replaced by dL_dX
					// dl_dx = dl_dy*dy_dx;
					nn_tensor_mul(dL_dY, m, i, j, k, dy_dx);
				}
			}
		}
	}

	// dL_dY replaced by dL_dX
	return dL_dY;
}

static int nn_factLayer_newCompute(nn_factLayer_t* self)
{
	return 1;
}

static void
nn_factLayer_deleteCompute(nn_factLayer_t* self)
{
}

#endif

static nn_dim_t*
nn_factLayer_dimXFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_factLayer_t* self = (nn_factLayer_t*) base;

	// Y and X are the same dimensions
	// but X is a reference
	return nn_tensor_dim(self->Y);
}

static nn_dim_t*
nn_factLayer_dimYFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_factLayer_t* self = (nn_factLayer_t*) base;

	return nn_tensor_dim(self->Y);
}

static const char* nn_factLayer_string(nn_factLayerFn_e fn)
{
	ASSERT(fn >= 0);
	ASSERT(fn < NN_FACT_LAYER_FN_COUNT);

	const char* str_array[NN_FACT_LAYER_FN_COUNT] =
	{
		NN_FACT_LAYER_STRING_LINEAR,
		NN_FACT_LAYER_STRING_LOGISTIC,
		NN_FACT_LAYER_STRING_RELU,
		NN_FACT_LAYER_STRING_PRELU,
		NN_FACT_LAYER_STRING_TANH,
	};

	return str_array[fn];
}

static nn_factLayerFn_e nn_factLayer_function(const char* str)
{
	ASSERT(str);

	const char* str_fn[NN_FACT_LAYER_FN_COUNT] =
	{
		NN_FACT_LAYER_STRING_LINEAR,
		NN_FACT_LAYER_STRING_LOGISTIC,
		NN_FACT_LAYER_STRING_RELU,
		NN_FACT_LAYER_STRING_PRELU,
		NN_FACT_LAYER_STRING_TANH,
	};

	int i;
	for(i = 0; i < NN_FACT_LAYER_FN_COUNT; ++i)
	{
		if(strcmp(str, str_fn[i]) == 0)
		{
			return (nn_factLayerFn_e) i;
		}
	}

	LOGE("invalid %s", str);
	return NN_FACT_LAYER_FN_ERROR;
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_factLayer_t*
nn_factLayer_new(nn_arch_t* arch, nn_dim_t* dimX,
                 nn_factLayerFn_e fn)
{
	ASSERT(arch);
	ASSERT(dimX);

	if(((int) fn < 0) || ((int) fn >= NN_FACT_LAYER_FN_COUNT))
	{
		LOGE("invalid fn=%i", (int) fn);
		return NULL;
	}

	nn_layerInfo_t info =
	{
		.arch            = arch,
		.forward_pass_fn = nn_factLayer_forwardPassFn,
		.backprop_fn     = nn_factLayer_backpropFn,
		.dimX_fn         = nn_factLayer_dimXFn,
		.dimY_fn         = nn_factLayer_dimYFn,
	};

	nn_factLayer_t* self;
	self = (nn_factLayer_t*)
	       nn_layer_new(sizeof(nn_factLayer_t), &info);
	if(self == NULL)
	{
		return NULL;
	}

	self->fn = fn;

	self->Y = nn_tensor_new(arch, dimX,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_COMPUTE);
	if(self->Y == NULL)
	{
		goto fail_Y;
	}

	if(nn_factLayer_newCompute(self) == 0)
	{
		goto fail_compute;
	}

	// success
	return self;

	// failure
	fail_compute:
		nn_tensor_delete(&self->Y);
	fail_Y:
		nn_layer_delete((nn_layer_t**) &self);
	return NULL;
}

nn_factLayer_t*
nn_factLayer_import(nn_arch_t* arch, jsmn_val_t* val)
{
	ASSERT(arch);
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	jsmn_val_t* val_dimX = NULL;
	jsmn_val_t* val_fn   = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		jsmn_keyval_t* kv;
		kv = (jsmn_keyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == JSMN_TYPE_STRING)
		{
			if(strcmp(kv->key, "fn") == 0)
			{
				val_fn = kv->val;
			}
		}
		else if(kv->val->type == JSMN_TYPE_OBJECT)
		{
			if(strcmp(kv->key, "dimX") == 0)
			{
				val_dimX = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_dimX == NULL) || (val_fn == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_dim_t dimX;
	if(nn_dim_load(&dimX, val_dimX) == 0)
	{
		return NULL;
	}

	nn_factLayerFn_e fn = nn_factLayer_function(val_fn->data);
	return nn_factLayer_new(arch, &dimX, fn);
}

int nn_factLayer_export(nn_factLayer_t* self,
                        jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	nn_dim_t* dimX = nn_factLayer_dimXFn(&self->base);

	const char* str_fn = nn_factLayer_string(self->fn);
	if(str_fn == NULL)
	{
		LOGE("invalid");
		return 0;
	}

	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	ret &= jsmn_stream_key(stream, "%s", "dimX");
	ret &= nn_dim_store(dimX, stream);
	ret &= jsmn_stream_key(stream, "%s", "fn");
	ret &= jsmn_stream_string(stream, "%s", str_fn);
	ret &= jsmn_stream_end(stream);

	return ret;
}

void nn_factLayer_delete(nn_factLayer_t** _self)
{
	ASSERT(_self);

	nn_factLayer_t* self = *_self;
	if(self)
	{
		nn_factLayer_deleteCompute(self);
		nn_tensor_delete(&self->Y);
		nn_layer_delete((nn_layer_t**) _self);
	}
}
