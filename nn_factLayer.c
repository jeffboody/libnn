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

#define LOG_TAG "nn"
#include "../libcc/cc_log.h"
#include "../libcc/cc_memory.h"
#include "nn_arch.h"
#include "nn_factLayer.h"
#include "nn_layer.h"
#include "nn_tensor.h"

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
nn_factLayer_dimFn(nn_layer_t* base)
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
		.dim_fn          = nn_factLayer_dimFn,
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
