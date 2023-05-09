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

static nn_tensor_t*
nn_factLayer_forwardPassFn(nn_layer_t* base, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_factLayer_t* self = (nn_factLayer_t*) base;

	nn_tensor_t* Y     = self->Y;
	nn_tensor_t* dY_dX = self->dY_dX;
	nn_dim_t*    dim   = nn_tensor_dim(Y);
	uint32_t     bs    = base->arch->batch_size;

	nn_factLayer_fn fact  = self->fact;
	nn_factLayer_fn dfact = self->dfact;

	// output and forward gradients
	float    x;
	uint32_t m;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(m = 0; m < bs; ++m)
	{
		for(i = 0; i < dim->height; ++i)
		{
			for(j = 0; j < dim->width; ++j)
			{
				for(k = 0; k < dim->depth; ++k)
				{
					// output
					x = nn_tensor_get(X, m, i, j, k);
					nn_tensor_set(Y, m, i, j, k,
					              (*fact)(x));

					// forward gradients
					nn_tensor_set(dY_dX, m, i, j, k,
					              (*dfact)(x));
				}
			}
		}
	}

	return Y;
}

static nn_tensor_t*
nn_factLayer_backpropFn(nn_layer_t* base, nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,xh,xw,xd)

	nn_factLayer_t* self = (nn_factLayer_t*) base;

	nn_tensor_t* dY_dX = self->dY_dX;
	nn_tensor_t* dL_dX = self->dL_dX;
	nn_dim_t*    dim   = nn_tensor_dim(dL_dY);
	uint32_t     bs    = base->arch->batch_size;

	// backpropagate loss
	float    dy_dx;
	float    dl_dx;
	float    dl_dy;
	uint32_t m;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(m = 0; m < bs; ++m)
	{
		for(i = 0; i < dim->height; ++i)
		{
			for(j = 0; j < dim->width; ++j)
			{
				for(k = 0; k < dim->depth; ++k)
				{
					dl_dy = nn_tensor_get(dL_dY, m, i, j, k);
					dy_dx = nn_tensor_get(dY_dX, m, i, j, k);
					dl_dx = dl_dy*dy_dx;
					nn_tensor_set(dL_dX, m, i, j, k, dl_dx);
				}
			}
		}
	}
	return dL_dX;
}

static nn_dim_t*
nn_factLayer_dimXFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_factLayer_t* self = (nn_factLayer_t*) base;

	return nn_tensor_dim(self->dL_dX);
}

static nn_dim_t*
nn_factLayer_dimYFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_factLayer_t* self = (nn_factLayer_t*) base;

	return nn_tensor_dim(self->Y);
}

/***********************************************************
* public - activation functions                            *
***********************************************************/

float nn_factLayer_linear(float x)
{
	return x;
}

float nn_factLayer_logistic(float x)
{
	return 1.0f/(1.0f + exp(-x));
}

float nn_factLayer_ReLU(float x)
{
	if(x < 0.0f)
	{
		return 0.0f;
	}

	return x;
}

float nn_factLayer_PReLU(float x)
{
	if(x < 0.0f)
	{
		return 0.01f*x;
	}

	return x;
}

float nn_factLayer_tanh(float x)
{
	return tanhf(x);
}

float nn_factLayer_dlinear(float x)
{
	return 1.0f;
}

float nn_factLayer_dlogistic(float x)
{
	float fx = nn_factLayer_logistic(x);
	return fx*(1.0f - fx);
}

float nn_factLayer_dReLU(float x)
{
	if(x < 0.0f)
	{
		return 0.0f;
	}

	return 1.0f;
}

float nn_factLayer_dPReLU(float x)
{
	if(x < 0.0f)
	{
		return 0.01f;
	}

	return 1.0f;
}

float nn_factLayer_dtanh(float x)
{
	float tanhfx = tanhf(x);
	return 1.0f - tanhfx*tanhfx;
}

const char* nn_factLayer_string(nn_factLayer_fn fact)
{
	ASSERT(fact)

	if(fact == nn_factLayer_linear)
	{
		return NN_FACT_LAYER_STRING_LINEAR;
	}
	else if(fact == nn_factLayer_logistic)
	{
		return NN_FACT_LAYER_STRING_LOGISTIC;
	}
	else if(fact == nn_factLayer_ReLU)
	{
		return NN_FACT_LAYER_STRING_RELU;
	}
	else if(fact == nn_factLayer_PReLU)
	{
		return NN_FACT_LAYER_STRING_PRELU;
	}
	else if(fact == nn_factLayer_tanh)
	{
		return NN_FACT_LAYER_STRING_TANH;
	}
	else if(fact == nn_factLayer_dlinear)
	{
		return NN_FACT_LAYER_STRING_DLINEAR;
	}
	else if(fact == nn_factLayer_dlogistic)
	{
		return NN_FACT_LAYER_STRING_DLOGISTIC;
	}
	else if(fact == nn_factLayer_dReLU)
	{
		return NN_FACT_LAYER_STRING_DRELU;
	}
	else if(fact == nn_factLayer_dPReLU)
	{
		return NN_FACT_LAYER_STRING_DPRELU;
	}
	else if(fact == nn_factLayer_dtanh)
	{
		return NN_FACT_LAYER_STRING_DTANH;
	}

	LOGE("invalid");
	return NULL;
}

nn_factLayer_fn nn_factLayer_function(const char* str)
{
	ASSERT(str);

	if(strcmp(str, NN_FACT_LAYER_STRING_LINEAR) == 0)
	{
		return nn_factLayer_linear;
	}
	else if(strcmp(str, NN_FACT_LAYER_STRING_LOGISTIC) == 0)
	{
		return nn_factLayer_logistic;
	}
	else if(strcmp(str, NN_FACT_LAYER_STRING_RELU) == 0)
	{
		return nn_factLayer_ReLU;
	}
	else if(strcmp(str, NN_FACT_LAYER_STRING_PRELU) == 0)
	{
		return nn_factLayer_PReLU;
	}
	else if(strcmp(str, NN_FACT_LAYER_STRING_TANH) == 0)
	{
		return nn_factLayer_tanh;
	}
	else if(strcmp(str, NN_FACT_LAYER_STRING_DLINEAR) == 0)
	{
		return nn_factLayer_dlinear;
	}
	else if(strcmp(str, NN_FACT_LAYER_STRING_DLOGISTIC) == 0)
	{
		return nn_factLayer_dlogistic;
	}
	else if(strcmp(str, NN_FACT_LAYER_STRING_DRELU) == 0)
	{
		return nn_factLayer_dReLU;
	}
	else if(strcmp(str, NN_FACT_LAYER_STRING_DPRELU) == 0)
	{
		return nn_factLayer_dPReLU;
	}
	else if(strcmp(str, NN_FACT_LAYER_STRING_DTANH) == 0)
	{
		return nn_factLayer_dtanh;
	}

	LOGE("invalid %s", str);
	return NULL;
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_factLayer_t*
nn_factLayer_new(nn_arch_t* arch, nn_dim_t* dimX,
                 nn_factLayer_fn fact,
                 nn_factLayer_fn dfact)
{
	ASSERT(arch);
	ASSERT(dimX);
	ASSERT(fact);
	ASSERT(dfact);

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

	self->Y = nn_tensor_new(dimX);
	if(self->Y == NULL)
	{
		goto fail_Y;
	}

	self->dY_dX = nn_tensor_new(dimX);
	if(self->dY_dX == NULL)
	{
		goto fail_dY_dX;
	}

	self->dL_dX = nn_tensor_new(dimX);
	if(self->dL_dX == NULL)
	{
		goto fail_dL_dX;
	}

	self->fact  = fact;
	self->dfact = dfact;

	// success
	return self;

	// failure
	fail_dL_dX:
		nn_tensor_delete(&self->dY_dX);
	fail_dY_dX:
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

	jsmn_val_t* val_dimX  = NULL;
	jsmn_val_t* val_fact  = NULL;
	jsmn_val_t* val_dfact = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		jsmn_keyval_t* kv;
		kv = (jsmn_keyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == JSMN_TYPE_STRING)
		{
			if(strcmp(kv->key, "fact") == 0)
			{
				val_fact = kv->val;
			}
			else if(strcmp(kv->key, "dfact") == 0)
			{
				val_dfact = kv->val;
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
	if((val_dimX  == NULL) ||
	   (val_fact  == NULL) ||
	   (val_dfact == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_dim_t dimX;
	if(nn_dim_load(&dimX, val_dimX) == 0)
	{
		return NULL;
	}

	nn_factLayer_fn fact;
	nn_factLayer_fn dfact;
	fact  = nn_factLayer_function(val_fact->data);
	dfact = nn_factLayer_function(val_dfact->data);
	if((fact == NULL) || (dfact == NULL))
	{
		return NULL;
	}

	return nn_factLayer_new(arch, &dimX, fact, dfact);
}

int nn_factLayer_export(nn_factLayer_t* self,
                        jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	nn_dim_t* dimX = nn_tensor_dim(self->dL_dX);

	const char* str_fact  = nn_factLayer_string(self->fact);
	const char* str_dfact = nn_factLayer_string(self->dfact);
	if((str_fact == NULL) || (str_dfact == NULL))
	{
		LOGE("invalid");
		return 0;
	}

	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	ret &= jsmn_stream_key(stream, "%s", "dimX");
	ret &= nn_dim_store(dimX, stream);
	ret &= jsmn_stream_key(stream, "%s", "fact");
	ret &= jsmn_stream_string(stream, "%s", str_fact);
	ret &= jsmn_stream_key(stream, "%s", "dfact");
	ret &= jsmn_stream_string(stream, "%s", str_dfact);
	ret &= jsmn_stream_end(stream);

	return ret;
}

void nn_factLayer_delete(nn_factLayer_t** _self)
{
	ASSERT(_self);

	nn_factLayer_t* self = *_self;
	if(self)
	{
		nn_tensor_delete(&self->dL_dX);
		nn_tensor_delete(&self->dY_dX);
		nn_tensor_delete(&self->Y);
		nn_layer_delete((nn_layer_t**) _self);
	}
}
