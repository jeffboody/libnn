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

	// clear forward gradients
	nn_tensor_t* dY_dX = self->dY_dX;
	nn_tensor_clear(dY_dX);

	// output and forward gradients (sum)
	nn_factLayer_fn fact  = self->fact;
	nn_factLayer_fn dfact = self->dfact;
	nn_tensor_t*    Y     = self->Y;
	nn_dim_t*       dim   = nn_tensor_dim(Y);
	float           in;
	uint32_t        i;
	uint32_t        x;
	uint32_t        y;
	uint32_t        z;
	uint32_t        bs = base->arch->batch_size;
	for(i = 0; i < bs; ++i)
	{
		for(y = 0; y < dim->h; ++y)
		{
			for(x = 0; x < dim->w; ++x)
			{
				for(z = 0; z < dim->d; ++z)
				{
					// output
					in = nn_tensor_get(X, i, x, y, z);
					nn_tensor_set(Y, i, x, y, z,
					              (*fact)(in));

					// forward gradients (sum)
					nn_tensor_add(dY_dX, 0, x, y, z,
					              (*dfact)(in));
				}
			}
		}
	}

	// forward gradients (batch mean)
	float s = 1.0f/((float) bs);
	for(y = 0; y < dim->h; ++y)
	{
		for(x = 0; x < dim->w; ++x)
		{
			for(z = 0; z < dim->d; ++z)
			{
				nn_tensor_mul(dY_dX, 0, x, y, z, s);
			}
		}
	}

	return Y;
}

static nn_tensor_t*
nn_factLayer_backpropFn(nn_layer_t* base, nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(1,X.w,X.h,X.d)

	nn_factLayer_t* self  = (nn_factLayer_t*) base;
	nn_tensor_t*    dY_dX = self->dY_dX;
	nn_tensor_t*    dL_dX = self->dL_dX;
	nn_dim_t*       dim   = nn_tensor_dim(dL_dY);

	// backpropagate loss
	uint32_t x;
	uint32_t y;
	uint32_t z;
	float    dy_dx;
	float    dl_dx;
	float    dl_dy;
	for(y = 0; y < dim->h; ++y)
	{
		for(x = 0; x < dim->w; ++x)
		{
			for(z = 0; z < dim->d; ++z)
			{
				dl_dy = nn_tensor_get(dL_dY, 0, x, y, z);
				dy_dx = nn_tensor_get(dY_dX, 0, x, y, z);
				dl_dx = dl_dy*dy_dx;
				nn_tensor_set(dL_dX, 0, x, y, z, dl_dx);
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
nn_factLayer_new(nn_arch_t* arch, nn_dim_t* dim,
                 nn_factLayer_fn fact,
                 nn_factLayer_fn dfact)
{
	ASSERT(arch);
	ASSERT(dim);
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

	self->Y = nn_tensor_new(dim);
	if(self->Y == NULL)
	{
		goto fail_Y;
	}

	nn_dim_t dim1 =
	{
		.n = 1,
		.w = dim->w,
		.h = dim->h,
		.d = dim->d,
	};

	self->dY_dX = nn_tensor_new(&dim1);
	if(self->dY_dX == NULL)
	{
		goto fail_dY_dX;
	}

	self->dL_dX = nn_tensor_new(&dim1);
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
