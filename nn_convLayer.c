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
#include "nn_convLayer.h"
#include "nn_layer.h"
#include "nn_tensor.h"

/***********************************************************
* private                                                  *
***********************************************************/

static void
nn_convLayer_forwardPass(nn_convLayer_t* self, nn_tensor_t* X,
                         uint32_t m, uint32_t i,
                         uint32_t j, uint32_t f)
{
	ASSERT(self);

	nn_dim_t* dimX = nn_tensor_dim(X);
	nn_dim_t* dimW = nn_tensor_dim(self->W);

	uint32_t xd = dimX->depth;
	uint32_t fh = dimW->height;
	uint32_t fw = dimW->width;

	nn_tensor_t* W = self->W;
	nn_tensor_t* B = self->B;
	nn_tensor_t* Y = self->Y;

	// initialize Y
	float y;
	if(self->flags & NN_CONV_LAYER_FLAG_DISABLE_BIAS)
	{
		y = 0.0f;
	}
	else
	{
		y = nn_tensor_get(B, f, 0, 0, 0);
	}

	// compute Y
	uint32_t fi;
	uint32_t fj;
	uint32_t k;
	float    x;
	float    w;
	for(fi = 0; fi < fh; ++fi)
	{
		for(fj = 0; fj < fw; ++fj)
		{
			for(k = 0; k < xd; ++k)
			{
				w  = nn_tensor_get(W, f, fi, fj, k);
				x  = nn_tensor_get(X, m, i + fi, j + fj, k);
				y += w*x;
			}
		}
	}
	nn_tensor_set(Y, m, i, j, f, y);
}

static nn_tensor_t*
nn_convLayer_forwardPassFn(nn_layer_t* base,
                           nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_convLayer_t* self = (nn_convLayer_t*) base;

	// clear forward gradients
	nn_tensor_t* dY_dW = self->dY_dW;
	nn_tensor_clear(dY_dW);

	nn_dim_t* dimW = nn_tensor_dim(self->W);
	nn_dim_t* dimX = nn_tensor_dim(X);
	nn_dim_t* dimY = nn_tensor_dim(self->Y);

	uint32_t bs = base->arch->batch_size;
	uint32_t fc = dimW->count;
	uint32_t xh = dimX->height;
	uint32_t xw = dimX->width;
	uint32_t xd = dimX->depth;
	uint32_t yh = dimY->height;
	uint32_t yw = dimY->width;

	// forward pass Y
	uint32_t m;
	uint32_t i;
	uint32_t j;
	uint32_t f;
	for(m = 0; m < bs; ++m)
	{
		for(i = 0; i < yh; ++i)
		{
			for(j = 0; j < yw; ++j)
			{
				for(f = 0; f < fc; ++f)
				{
					nn_convLayer_forwardPass(self, X, m, i, j, f);
				}
			}
		}
	}

	// forward gradient dY/dW
	uint32_t k;
	float    s = 1.0f/((float) bs);
	float    x;
	float    dy_dw;
	for(i = 0; i < xh; ++i)
	{
		for(j = 0; j < xw; ++j)
		{
			for(k = 0; k < xd; ++k)
			{
				dy_dw = 0.0f;
				for(m = 0; m < bs; ++m)
				{
					x = nn_tensor_get(X, m, i, j, k);
					dy_dw += x;
				}
				nn_tensor_set(dY_dW, 0, i, j, k, s*dy_dw);
			}
		}
	}

	return self->Y;
}

static void
nn_convLayer_backprop(nn_convLayer_t* self,
                      nn_tensor_t* dL_dY,
                      uint32_t i, uint32_t j, uint32_t f)
{
	ASSERT(self);
	ASSERT(dL_dY);

	nn_dim_t* dimW = nn_tensor_dim(self->W);
	uint32_t  fh   = dimW->height;
	uint32_t  fw   = dimW->width;
	uint32_t  xd   = dimW->depth;

	nn_tensor_t* dY_dX = self->W;
	nn_tensor_t* dY_dW = self->dY_dW;
	nn_tensor_t* dL_dW = self->dL_dW;
	nn_tensor_t* dL_dB = self->dL_dB;
	nn_tensor_t* dL_dX = self->dL_dX;

	float dl_dy;
	dl_dy = nn_tensor_get(dL_dY, 0, i, j, f);

	// sum dL_dB
	float dy_db = 1.0f;
	nn_tensor_add(dL_dB, f, 0, 0, 0, dl_dy*dy_db);

	// backpropagate loss
	float    dy_dx;
	float    dy_dw;
	uint32_t fi;
	uint32_t fj;
	uint32_t k;
	for(fi = 0; fi < fh; ++fi)
	{
		for(fj = 0; fj < fw; ++fj)
		{
			for(k = 0; k < xd; ++k)
			{
				// backpropagate loss
				dy_dx = nn_tensor_get(dY_dX, f, fi, fj, k);
				nn_tensor_add(dL_dX, 0, i + fi, j + fj, k, dl_dy*dy_dx);

				// sum dL_dW
				dy_dw = nn_tensor_get(dY_dW, 0, i + fi, j + fj, k);
				nn_tensor_add(dL_dW, f, fi, fj, k, dl_dy*dy_dw);
			}
		}
	}
}

static nn_tensor_t*
nn_convLayer_backpropFn(nn_layer_t* base,
                        nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(1,1,1,nc)

	nn_convLayer_t* self = (nn_convLayer_t*) base;

	nn_dim_t* dim = nn_tensor_dim(dL_dY);
	uint32_t  yh  = dim->height;
	uint32_t  yw  = dim->width;
	uint32_t  fc  = dim->count;
	uint32_t  fh  = dim->height;
	uint32_t  fw  = dim->width;
	uint32_t  xd  = dim->depth;

	// clear backprop gradients
	nn_tensor_t* W     = self->W;
	nn_tensor_t* B     = self->B;
	nn_tensor_t* dL_dW = self->dL_dW;
	nn_tensor_t* dL_dB = self->dL_dB;
	nn_tensor_t* dL_dX = self->dL_dX;
	nn_tensor_clear(dL_dW);
	nn_tensor_clear(dL_dB);
	nn_tensor_clear(dL_dX);

	// backpropagate loss
	uint32_t i;
	uint32_t j;
	uint32_t f;
	for(i = 0; i < yh; ++i)
	{
		for(j = 0; j < yw; ++j)
		{
			for(f = 0; f < fc; ++f)
			{
				nn_convLayer_backprop(self, dL_dY, i, j, f);
			}
		}
	}

	// update parameters
	float    dl_db;
	float    dl_dw;
	float    lr = self->base.arch->learning_rate;
	uint32_t fi;
	uint32_t fj;
	uint32_t k;
	for(f = 0; f < fc; ++f)
	{
		if((self->flags & NN_CONV_LAYER_FLAG_DISABLE_BIAS) == 0)
		{
			dl_db = nn_tensor_get(dL_dB, f, 0, 0, 0);
			nn_tensor_add(B, f, 0, 0, 0, -lr*dl_db);
		}

		for(fi = 0; fi < fh; ++fi)
		{
			for(fj = 0; fj < fw; ++fj)
			{
				for(k = 0; k < xd; ++k)
				{
					dl_dw = nn_tensor_get(dL_dW, f, fi, fj, k);
					nn_tensor_add(W, f, fi, fj, k, -lr*dl_dw);
				}
			}
		}
	}

	return self->dL_dX;
}

static nn_dim_t*
nn_convLayer_dimFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_convLayer_t* self = (nn_convLayer_t*) base;

	return nn_tensor_dim(self->Y);
}

static void
nn_convLayer_initXavierWeights(nn_convLayer_t* self)
{
	ASSERT(self);

	nn_arch_t* arch = self->base.arch;

	nn_dim_t* dim = nn_tensor_dim(self->W);
	uint32_t  hwd = dim->height*dim->width*dim->depth;
	float     min = -1.0/sqrt((double) hwd);
	float     max = 1.0/sqrt((double) hwd);

	uint32_t n;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	float    w;
	for(n = 0; n < dim->count; ++n)
	{
		for(i = 0; i < dim->height; ++i)
		{
			for(j = 0; j < dim->width; ++j)
			{
				for(k = 0; k < dim->depth; ++k)
				{
					w = cc_rngUniform_rand2F(&arch->rng_uniform,
					                         min, max);
					nn_tensor_set(self->W, n, i, j, k, w);
				}
			}
		}
	}
}

static void
nn_convLayer_initHeWeights(nn_convLayer_t* self)
{
	ASSERT(self);

	nn_arch_t* arch = self->base.arch;

	nn_dim_t* dim = nn_tensor_dim(self->W);
	uint32_t  hwd = dim->height*dim->width*dim->depth;

	double mu    = 0.0;
	double sigma = sqrt(2.0/((double) hwd));
	cc_rngNormal_reset(&arch->rng_normal, mu, sigma);

	uint32_t n;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	float    w;
	for(n = 0; n < dim->count; ++n)
	{
		for(i = 0; i < dim->height; ++i)
		{
			for(j = 0; j < dim->width; ++j)
			{
				for(k = 0; k < dim->depth; ++k)
				{
					w = cc_rngNormal_rand1F(&arch->rng_normal);
					nn_tensor_set(self->W, n, i, j, k, w);
				}
			}
		}
	}
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_convLayer_t*
nn_convLayer_new(nn_arch_t* arch, nn_dim_t* dimX,
                 nn_dim_t* dimW, int flags)
{
	ASSERT(arch);
	ASSERT(dimX);
	ASSERT(dimW);

	uint32_t fc = dimW->count;
	uint32_t fh = dimW->height;
	uint32_t fw = dimW->width;
	uint32_t xh = dimX->height;
	uint32_t xw = dimX->width;
	uint32_t xd = dimX->depth;
	uint32_t bs = arch->max_batch_size;

	if(dimX->depth != dimW->depth)
	{
		LOGE("invalid depth=%u:%u",
		     dimX->depth, dimW->depth);
		return NULL;
	}

	// TODO - add same padding
	if(flags & NN_CONV_LAYER_FLAG_PAD_SAME)
	{
		LOGE("unsupported");
		return NULL;
	}

	nn_layerInfo_t info =
	{
		.arch            = arch,
		.forward_pass_fn = nn_convLayer_forwardPassFn,
		.backprop_fn     = nn_convLayer_backpropFn,
		.dim_fn          = nn_convLayer_dimFn,
	};

	nn_convLayer_t* self;
	self = (nn_convLayer_t*)
	       nn_layer_new(sizeof(nn_convLayer_t), &info);
	if(self == NULL)
	{
		return NULL;
	}

	self->flags = flags;

	self->W = nn_tensor_new(dimW);
	if(self->W == NULL)
	{
		goto fail_W;
	}

	if(flags & NN_CONV_LAYER_FLAG_HE)
	{
		nn_convLayer_initHeWeights(self);
	}
	else
	{
		nn_convLayer_initXavierWeights(self);
	}

	nn_dim_t dimB =
	{
		.count  = fc,
		.height = 1,
		.width  = 1,
		.depth  = 1,
	};

	self->B = nn_tensor_new(&dimB);
	if(self->B == NULL)
	{
		goto fail_B;
	}

	uint32_t yh = xh - fh + 1;
	uint32_t yw = xw - fw + 1;

	nn_dim_t dimY =
	{
		.count  = bs,
		.height = yh,
		.width  = yw,
		.depth  = fc,
	};

	self->Y = nn_tensor_new(&dimY);
	if(self->Y == NULL)
	{
		goto fail_Y;
	}

	nn_dim_t dim_1hwd =
	{
		.count  = 1,
		.height = xh,
		.width  = xw,
		.depth  = xd,
	};

	self->dY_dW = nn_tensor_new(&dim_1hwd);
	if(self->dY_dW == NULL)
	{
		goto fail_dY_dW;
	}

	self->dL_dW = nn_tensor_new(dimW);
	if(self->dL_dW == NULL)
	{
		goto fail_dL_dW;
	}

	self->dL_dB = nn_tensor_new(&dimB);
	if(self->dL_dB == NULL)
	{
		goto fail_dL_dB;
	}

	self->dL_dX = nn_tensor_new(&dim_1hwd);
	if(self->dL_dX == NULL)
	{
		goto fail_dL_dX;
	}

	// success
	return self;

	// failure
	fail_dL_dX:
		nn_tensor_delete(&self->dL_dB);
	fail_dL_dB:
		nn_tensor_delete(&self->dL_dW);
	fail_dL_dW:
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

void nn_convLayer_delete(nn_convLayer_t** _self)
{
	ASSERT(_self);

	nn_convLayer_t* self = *_self;
	if(self)
	{
		nn_tensor_delete(&self->dL_dB);
		nn_tensor_delete(&self->dL_dW);
		nn_tensor_delete(&self->dL_dX);
		nn_tensor_delete(&self->dY_dW);
		nn_tensor_delete(&self->Y);
		nn_tensor_delete(&self->B);
		nn_tensor_delete(&self->W);
		nn_layer_delete((nn_layer_t**) _self);
	}
}
