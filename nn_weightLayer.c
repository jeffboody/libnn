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

	// flattened tensors
	nn_tensor_t  Xf;
	nn_tensor_t  Yf;
	nn_tensor_flatten(X, &Xf);
	nn_tensor_flatten(self->Y, &Yf);

	nn_tensor_t* W = self->W;
	nn_tensor_t* B = self->B;

	// compute weighted sum and forward gradients (sum)
	float     w;
	float     x;
	float     y;
	uint32_t  m;
	uint32_t  n;
	uint32_t  k;
	nn_dim_t* dimX = nn_tensor_dim(&Xf);
	nn_dim_t* dimY = nn_tensor_dim(&Yf);
	uint32_t  xd   = dimX->depth;
	uint32_t  nc   = dimY->depth;
	uint32_t  bs   = base->arch->batch_size;
	for(m = 0; m < bs; ++m)
	{
		for(n = 0; n < nc; ++n)
		{
			if(self->flags & NN_WEIGHT_LAYER_FLAG_DISABLE_BIAS)
			{
				y = 0.0f;
			}
			else
			{
				y = nn_tensor_get(B, n, 0, 0, 0);
			}

			for(k = 0; k < xd; ++k)
			{
				// compute weighted sum
				x = nn_tensor_get(&Xf, m, 0, 0, k);
				w = nn_tensor_get(W, n, 0, 0, k);
				y += w*x;

				// forward gradients (sum)
				nn_tensor_add(dY_dW, n, 0, 0, k, x);
			}
			nn_tensor_set(&Yf, m, 0, 0, n, y);
		}
	}

	// forward gradients (batch mean)
	float s = 1.0f/((float) bs);
	for(k = 0; k < xd; ++k)
	{
		nn_tensor_mul(dY_dW, 0, 0, 0, k, s);
	}

	return self->Y;
}

static nn_tensor_t*
nn_weightLayer_backpropFn(nn_layer_t* base,
                          nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(1,1,1,nc)

	nn_weightLayer_t* self = (nn_weightLayer_t*) base;
	nn_arch_t*        arch = base->arch;

	nn_tensor_t* W      = self->W;
	nn_tensor_t* B      = self->B;
	nn_tensor_t* VW     = self->VW;
	nn_tensor_t* VB     = self->VB;
	nn_dim_t*    dim    = nn_tensor_dim(W);
	float        lr     = arch->learning_rate;
	float        mu     = arch->momentum_decay;
	float        lambda = arch->l2_lambda;
	nn_tensor_t* dY_dX  = W;
	nn_tensor_t* dY_dW  = self->dY_dW;
	nn_tensor_t* dL_dX  = self->dL_dX;
	float        dy_dx;
	float        dy_dw;
	float        dy_db  = 1.0f;
	float        dl_dy;
	float        dl_dx;

	// update parameters
	uint32_t n;
	uint32_t k;
	float    v0;
	float    v1;
	float    w;
	for(n = 0; n < dim->count; ++n)
	{
		dl_dy = nn_tensor_get(dL_dY, 0, 0, 0, n);

		for(k = 0; k < dim->depth; ++k)
		{
			dy_dw = nn_tensor_get(dY_dW, 0, 0, 0, k);
			w     = nn_tensor_get(W, n, 0, 0, k);

			// Nesterov Momentum Update and L2 Regularization
			// (weights)
			v0 = nn_tensor_get(VW, n, 0, 0, k);
			v1 = mu*v0 - lr*(dl_dy*dy_dw + 2*lambda*w);
			nn_tensor_set(VW, n, 0, 0, k, v1);
			nn_tensor_add(W, n, 0, 0, k, -mu*v0 + (1 - mu)*v1);
		}

		// Nesterov Momentum Update (bias)
		if((self->flags & NN_WEIGHT_LAYER_FLAG_DISABLE_BIAS) == 0)
		{
			v0 = nn_tensor_get(VB, n, 0, 0, k);
			v1 = mu*v0 - lr*dl_dy*dy_db;
			nn_tensor_set(VB, n, 0, 0, k, v1);
			nn_tensor_add(B, n, 0, 0, k, -mu*v0 + (1 - mu)*v1);
		}
	}

	// backpropagate loss
	for(k = 0; k < dim->depth; ++k)
	{
		dl_dx = 0.0f;
		for(n = 0; n < dim->count; ++n)
		{
			dl_dy  = nn_tensor_get(dL_dY, 0, 0, 0, n);
			dy_dx  = nn_tensor_get(dY_dX, n, 0, 0, k);
			dl_dx += dl_dy*dy_dx;
		}
		nn_tensor_set(dL_dX, 0, 0, 0, k, dl_dx);
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

	float min = -1.0/sqrt((double) dim->depth);
	float max = 1.0/sqrt((double) dim->depth);

	uint32_t k;
	uint32_t n;
	float    w;
	for(n = 0; n < dim->count; ++n)
	{
		for(k = 0; k < dim->depth; ++k)
		{
			w = cc_rngUniform_rand2F(&arch->rng_uniform,
			                         min, max);
			nn_tensor_set(self->W, n, 0, 0, k, w);
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
	double sigma = sqrt(2.0/((double) dim->depth));
	cc_rngNormal_reset(&arch->rng_normal, mu, sigma);

	uint32_t k;
	uint32_t n;
	float    w;
	for(n = 0; n < dim->count; ++n)
	{
		for(k = 0; k < dim->depth; ++k)
		{
			w = cc_rngNormal_rand1F(&arch->rng_normal);
			nn_tensor_set(self->W, n, 0, 0, k, w);
		}
	}
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_weightLayer_t*
nn_weightLayer_new(nn_arch_t* arch, nn_dim_t* dimX,
                   nn_dim_t* dimY, int flags)
{
	ASSERT(arch);
	ASSERT(dimX);
	ASSERT(dimY);

	uint32_t xd = dimX->height*dimX->width*dimX->depth;
	uint32_t nc = dimY->height*dimY->width*dimY->depth;

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

	self->flags = flags;

	nn_dim_t dimW =
	{
		.count  = nc,
		.height = 1,
		.width  = 1,
		.depth  = xd,
	};
	self->W = nn_tensor_new(&dimW);
	if(self->W == NULL)
	{
		goto fail_W;
	}

	if(flags & NN_WEIGHT_LAYER_FLAG_HE)
	{
		nn_weightLayer_initHeWeights(self);
	}
	else
	{
		nn_weightLayer_initXavierWeights(self);
	}

	nn_dim_t dimB =
	{
		.count  = nc,
		.height = 1,
		.width  = 1,
		.depth  = 1,
	};
	self->B = nn_tensor_new(&dimB);
	if(self->B == NULL)
	{
		goto fail_B;
	}

	self->Y = nn_tensor_new(dimY);
	if(self->Y == NULL)
	{
		goto fail_Y;
	}

	self->VW = nn_tensor_new(&dimW);
	if(self->VW == NULL)
	{
		goto fail_VW;
	}

	self->VB = nn_tensor_new(&dimB);
	if(self->VB == NULL)
	{
		goto fail_VB;
	}

	self->dY_dW = nn_tensor_new(nn_tensor_dim(self->W));
	if(self->dY_dW == NULL)
	{
		goto fail_dY_dW;
	}

	nn_dim_t dim_dL_dX =
	{
		.count  = 1,
		.height = 1,
		.width  = 1,
		.depth  = xd,
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
		nn_tensor_delete(&self->VB);
	fail_VB:
		nn_tensor_delete(&self->VW);
	fail_VW:
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
		nn_tensor_delete(&self->VB);
		nn_tensor_delete(&self->VW);
		nn_tensor_delete(&self->Y);
		nn_tensor_delete(&self->B);
		nn_tensor_delete(&self->W);
		nn_layer_delete((nn_layer_t**) _self);
	}
}
