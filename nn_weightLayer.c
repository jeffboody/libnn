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
#include "../libcc/rng/cc_rngNormal.h"
#include "../libcc/rng/cc_rngUniform.h"
#include "../libcc/cc_log.h"
#include "../libcc/cc_memory.h"
#include "nn_arch.h"
#include "nn_layer.h"
#include "nn_tensor.h"
#include "nn_weightLayer.h"

/***********************************************************
* private                                                  *
***********************************************************/

static nn_tensor_t*
nn_weightLayer_forwardPassFn(nn_layer_t* base,
                             nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_weightLayer_t* self  = (nn_weightLayer_t*) base;

	// clear forward gradients
	nn_tensor_t* dY_dW = self->dY_dW;
	nn_tensor_clear(dY_dW);

	// compute weighted sum and forward gradients (sum)
	nn_tensor_t* Y     = self->Y;
	nn_tensor_t* W     = self->W;
	nn_tensor_t* B     = self->B;
	float        wm;
	float        xm;
	float        yn;
	uint32_t     i;
	uint32_t     n;
	uint32_t     z;
	uint32_t     nc = self->nc;
	uint32_t     xd = nn_tensor_dim(X)->d;
	uint32_t     bs = base->arch->batch_size;
	for(i = 0; i < bs; ++i)
	{
		for(n = 0; n < nc; ++n)
		{
			yn = nn_tensor_get(B, n, 0, 0, 0);
			for(z = 0; z < xd; ++z)
			{
				// compute weighted sum
				xm = nn_tensor_get(X, i, 0, 0, z);
				wm = nn_tensor_get(W, n, 0, 0, z);
				yn += wm*xm;

				// forward gradients (sum)
				nn_tensor_add(dY_dW, n, 0, 0, z, xm);
			}
			nn_tensor_set(Y, i, 0, 0, n, yn);
		}
	}

	// forward gradients (batch mean)
	float s = 1.0f/((float) bs);
	for(z = 0; z < xd; ++z)
	{
		nn_tensor_mul(dY_dW, 0, 0, 0, z, s);
	}

	return Y;
}

static nn_tensor_t*
nn_weightLayer_backpropFn(nn_layer_t* base,
                          nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(1,1,1,nc)

	nn_weightLayer_t* self = (nn_weightLayer_t*) base;
	nn_arch_t*        arch = base->arch;

	nn_tensor_t* W     = self->W;
	nn_tensor_t* B     = self->B;
	nn_dim_t*    dim   = nn_tensor_dim(W);
	float        lr    = arch->learning_rate;
	nn_tensor_t* dY_dX = W;
	nn_tensor_t* dY_dW = self->dY_dW;
	nn_tensor_t* dL_dX = self->dL_dX;
	float        dy_dx;
	float        dy_dw;
	float        dy_db = 1.0f;
	float        dl_dy;
	float        dl_dx;

	// update parameters
	uint32_t i;
	uint32_t z;
	for(i = 0; i < dim->n; ++i)
	{
		dl_dy = nn_tensor_get(dL_dY, 0, 0, 0, i);

		for(z = 0; z < dim->d; ++z)
		{
			dy_dw = nn_tensor_get(dY_dW, 0, 0, 0, z);
			nn_tensor_add(W, i, 0, 0, z, -lr*dl_dy*dy_dw);
		}

		nn_tensor_add(B, i, 0, 0, 0, -lr*dl_dy*dy_db);
	}

	// backpropagate loss
	for(z = 0; z < dim->d; ++z)
	{
		dl_dx = 0.0f;
		for(i = 0; i < dim->n; ++i)
		{
			dl_dy  = nn_tensor_get(dL_dY, 0, 0, 0, i);
			dy_dx  = nn_tensor_get(dY_dX, i, 0, 0, z);
			dl_dx += dl_dy*dy_dx;
		}
		nn_tensor_set(dL_dX, 0, 0, 0, z, dl_dx);
	}

	return dL_dX;
}

static nn_dim_t*
nn_weightLayer_dimFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_weightLayer_t* self = (nn_weightLayer_t*) base;

	return nn_tensor_dim(self->Y);
}

static void
nn_weightLayer_initXavierWeights(nn_weightLayer_t* self)
{
	ASSERT(self);

	nn_arch_t* arch = self->base.arch;

	nn_dim_t* dim = nn_tensor_dim(self->W);

	float min = -1.0/sqrt((double) dim->d);
	float max = 1.0/sqrt((double) dim->d);

	uint32_t m;
	uint32_t n;
	float    w;
	for(n = 0; n < dim->n; ++n)
	{
		for(m = 0; m < dim->d; ++m)
		{
			w = cc_rngUniform_rand2F(&arch->rng_uniform,
			                         min, max);
			nn_tensor_set(self->W, n, 0, 0, m, w);
		}
	}
}

static void
nn_weightLayer_initHeWeights(nn_weightLayer_t* self)
{
	ASSERT(self);

	nn_arch_t* arch = self->base.arch;

	nn_dim_t* dim = nn_tensor_dim(self->W);

	double mu    = 0.0;
	double sigma = sqrt(2.0/((double) dim->d));
	cc_rngNormal_reset(&arch->rng_normal, mu, sigma);

	uint32_t m;
	uint32_t n;
	float    w;
	for(n = 0; n < dim->n; ++n)
	{
		for(m = 0; m < dim->d; ++m)
		{
			w = cc_rngNormal_rand1F(&arch->rng_normal);
			nn_tensor_set(self->W, n, 0, 0, m, w);
		}
	}
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_weightLayer_t*
nn_weightLayer_new(nn_arch_t* arch, uint32_t nc,
                   uint32_t xd, int init_mode)
{
	ASSERT(arch);

	nn_layerInfo_t info =
	{
		.arch            = arch,
		.forward_pass_fn = nn_weightLayer_forwardPassFn,
		.backprop_fn     = nn_weightLayer_backpropFn,
		.dim_fn          = nn_weightLayer_dimFn,
	};

	nn_weightLayer_t* self;
	self = (nn_weightLayer_t*)
	       nn_layer_new(sizeof(nn_weightLayer_t), &info);
	if(self == NULL)
	{
		return NULL;
	}

	nn_dim_t dimW =
	{
		.n = nc,
		.w = 1,
		.h = 1,
		.d = xd,
	};
	self->W = nn_tensor_new(&dimW);
	if(self->W == NULL)
	{
		goto fail_W;
	}

	if(init_mode == NN_WEIGHT_LAYER_INITMODE_HE)
	{
		nn_weightLayer_initHeWeights(self);
	}
	else
	{
		nn_weightLayer_initXavierWeights(self);
	}

	nn_dim_t dimB =
	{
		.n = nc,
		.w = 1,
		.h = 1,
		.d = 1,
	};
	self->B = nn_tensor_new(&dimB);
	if(self->B == NULL)
	{
		goto fail_B;
	}

	nn_dim_t dimY =
	{
		.n = arch->max_batch_size,
		.w = 1,
		.h = 1,
		.d = nc,
	};
	self->Y = nn_tensor_new(&dimY);
	if(self->Y == NULL)
	{
		goto fail_Y;
	}

	self->dY_dW = nn_tensor_new(nn_tensor_dim(self->W));
	if(self->dY_dW == NULL)
	{
		goto fail_dY_dW;
	}

	nn_dim_t dim_dL_dX =
	{
		.n = 1,
		.w = 1,
		.h = 1,
		.d = xd,
	};
	self->dL_dX = nn_tensor_new(&dim_dL_dX);
	if(self->dL_dX == NULL)
	{
		goto fail_dL_dX;
	}

	// success
	return self;

	// failure
	fail_dL_dX:
		nn_tensor_delete(&self->dY_dW);
	fail_dY_dW:
		nn_tensor_delete(&self->Y);
	fail_Y:
		nn_tensor_delete(&self->B);
	fail_B:
		nn_tensor_delete(&self->W);
	fail_W:
		nn_layer_delete((nn_layer_t**) &self);
	return NULL;
}

void nn_weightLayer_delete(nn_weightLayer_t** _self)
{
	ASSERT(_self);

	nn_weightLayer_t* self = *_self;
	if(self)
	{
		nn_tensor_delete(&self->dL_dX);
		nn_tensor_delete(&self->dY_dW);
		nn_tensor_delete(&self->Y);
		nn_tensor_delete(&self->B);
		nn_tensor_delete(&self->W);
		nn_layer_delete((nn_layer_t**) _self);
	}
}
