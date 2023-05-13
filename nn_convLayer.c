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
nn_convLayer_forwardPass(nn_convLayer_t* self,
                         nn_tensor_t* X,
                         uint32_t m, uint32_t i,
                         uint32_t j, uint32_t f)
{
	ASSERT(self);

	nn_tensor_t* W    = self->W;
	nn_tensor_t* B    = self->B;
	nn_tensor_t* Y    = self->Y;
	nn_dim_t*    dimX = nn_tensor_dim(X);
	nn_dim_t*    dimW = nn_tensor_dim(W);
	uint32_t     xd   = dimX->depth;
	uint32_t     fh   = dimW->height;
	uint32_t     fw   = dimW->width;

	// initialize y
	float y;
	if(self->flags & NN_CONV_LAYER_FLAG_DISABLE_BIAS)
	{
		y = 0.0f;
	}
	else
	{
		y = nn_tensor_get(B, f, 0, 0, 0);
	}

	// compute weighted sum
	float    w;
	float    x;
	uint32_t fi;
	uint32_t fj;
	uint32_t k;
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
nn_convLayer_forwardPassFn(nn_layer_t* base, int mode,
                           uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_convLayer_t* self = (nn_convLayer_t*) base;

	nn_dim_t* dimW = nn_tensor_dim(self->W);
	nn_dim_t* dimY = nn_tensor_dim(self->Y);
	uint32_t  fc   = dimW->count;
	uint32_t  yh   = dimY->height;
	uint32_t  yw   = dimY->width;

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

	// store reference
	self->X = X;

	return self->Y;
}

static void
nn_convLayer_backprop(nn_convLayer_t* self,
                      nn_tensor_t* dL_dY,
                      uint32_t m, uint32_t i,
                      uint32_t j, uint32_t f)
{
	ASSERT(self);
	ASSERT(dL_dY); // dim(bs, yh, yw, fc)

	nn_tensor_t* W     = self->W;
	nn_tensor_t* dY_dX = W;
	nn_tensor_t* dY_dW = self->X;
	nn_tensor_t* dL_dW = self->dL_dW;
	nn_tensor_t* dL_dB = self->dL_dB;
	nn_tensor_t* dL_dX = self->dL_dX;
	nn_dim_t*    dimW  = nn_tensor_dim(W);
	uint32_t     fh    = dimW->height;
	uint32_t     fw    = dimW->width;
	uint32_t     xd    = dimW->depth;

	float    dl_dy = nn_tensor_get(dL_dY, m, i, j, f);
	float    dy_dx;
	float    dy_dw;
	float    dy_db = 1.0f;
	uint32_t fi;
	uint32_t fj;
	uint32_t k;
	for(fi = 0; fi < fh; ++fi)
	{
		for(fj = 0; fj < fw; ++fj)
		{
			for(k = 0; k < xd; ++k)
			{
				// backpropagate dL_dX
				dy_dx = nn_tensor_get(dY_dX, f, fi, fj, k);
				nn_tensor_add(dL_dX, m, i + fi, j + fj, k, dl_dy*dy_dx);

				// sum dL_dW
				dy_dw = nn_tensor_get(dY_dW, m, i + fi, j + fj, k);
				nn_tensor_add(dL_dW, f, fi, fj, k, dl_dy*dy_dw);
			}
		}
	}

	// sum dL_dB
	if((self->flags & NN_CONV_LAYER_FLAG_DISABLE_BIAS) == 0)
	{
		nn_tensor_add(dL_dB, f, 0, 0, 0, dl_dy*dy_db);
	}
}

static nn_tensor_t*
nn_convLayer_backpropFn(nn_layer_t* base, uint32_t bs,
                        nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,yh,yw,fc)

	nn_convLayer_t* self = (nn_convLayer_t*) base;
	nn_arch_t*      arch = base->arch;

	nn_tensor_t* W      = self->W;
	nn_tensor_t* B      = self->B;
	nn_tensor_t* VW     = self->VW;
	nn_tensor_t* VB     = self->VB;
	nn_dim_t*    dim    = nn_tensor_dim(dL_dY);
	uint32_t     yh     = dim->height;
	uint32_t     yw     = dim->width;
	uint32_t     fc     = dim->count;
	uint32_t     fh     = dim->height;
	uint32_t     fw     = dim->width;
	uint32_t     xd     = dim->depth;
	float        lr     = arch->learning_rate;
	float        mu     = arch->momentum_decay;
	float        lambda = arch->l2_lambda;

	// clear backprop gradients
	nn_tensor_t* dL_dW = self->dL_dW;
	nn_tensor_t* dL_dB = self->dL_dB;
	nn_tensor_t* dL_dX = self->dL_dX;
	nn_tensor_clear(dL_dW);
	if((self->flags & NN_CONV_LAYER_FLAG_DISABLE_BIAS) == 0)
	{
		nn_tensor_clear(dL_dB);
	}
	nn_tensor_clear(dL_dX);

	// sum gradients and backpropagate loss
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
					nn_convLayer_backprop(self, dL_dY, m, i, j, f);
				}
			}
		}
	}

	// update parameters
	float    v0;
	float    v1;
	float    w;
	float    dl_dw;
	float    dl_db;
	float    s = 1.0f/((float) bs);
	uint32_t fi;
	uint32_t fj;
	uint32_t k;
	for(f = 0; f < fc; ++f)
	{
		// weights
		for(fi = 0; fi < fh; ++fi)
		{
			for(fj = 0; fj < fw; ++fj)
			{
				for(k = 0; k < xd; ++k)
				{
					dl_dw = s*nn_tensor_get(dL_dW, f, fi, fj, k);
					w     = nn_tensor_get(W, f, fi, fj, k);

					// Nesterov Momentum Update and L2 Regularization
					v0 = nn_tensor_get(VW, f, fi, fj, k);
					v1 = mu*v0 - lr*(dl_dw + 2*lambda*w);
					nn_tensor_set(VW, f, fi, fj, k, v1);
					nn_tensor_add(W, f, fi, fj, k, -mu*v0 + (1 + mu)*v1);
				}
			}
		}

		// bias
		if((self->flags & NN_CONV_LAYER_FLAG_DISABLE_BIAS) == 0)
		{
			dl_db = s*nn_tensor_get(dL_dB, f, 0, 0, 0);

			// Nesterov Momentum Update
			v0    = nn_tensor_get(VB, f, 0, 0, 0);
			v1    = mu*v0 - lr*dl_db;
			nn_tensor_set(VB, f, 0, 0, 0, v1);
			nn_tensor_add(B, f, 0, 0, 0, -mu*v0 + (1 + mu)*v1);
		}
	}

	return dL_dX;
}

static nn_dim_t*
nn_convLayer_dimXFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_convLayer_t* self = (nn_convLayer_t*) base;

	return nn_tensor_dim(self->dL_dX);
}

static nn_dim_t*
nn_convLayer_dimYFn(nn_layer_t* base)
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

	nn_dim_t* dimW = nn_tensor_dim(self->W);
	uint32_t  fc   = dimW->count;
	uint32_t  fh   = dimW->height;
	uint32_t  fw   = dimW->width;
	uint32_t  xd   = dimW->depth;
	uint32_t  hwd  = fh*fw*xd;
	float     min  = -1.0/sqrt((double) hwd);
	float     max  = 1.0/sqrt((double) hwd);

	float    w;
	uint32_t n;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(n = 0; n < fc; ++n)
	{
		for(i = 0; i < fh; ++i)
		{
			for(j = 0; j < fw; ++j)
			{
				for(k = 0; k < xd; ++k)
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

	nn_dim_t* dimW = nn_tensor_dim(self->W);
	uint32_t  fc   = dimW->count;
	uint32_t  fh   = dimW->height;
	uint32_t  fw   = dimW->width;
	uint32_t  xd   = dimW->depth;
	uint32_t  hwd  = fh*fw*xd;

	double mu    = 0.0;
	double sigma = sqrt(2.0/((double) hwd));
	cc_rngNormal_reset(&arch->rng_normal, mu, sigma);

	float    w;
	uint32_t n;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(n = 0; n < fc; ++n)
	{
		for(i = 0; i < fh; ++i)
		{
			for(j = 0; j < fw; ++j)
			{
				for(k = 0; k < xd; ++k)
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
	uint32_t bs = dimX->count;
	uint32_t xh = dimX->height;
	uint32_t xw = dimX->width;

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
		.dimX_fn         = nn_convLayer_dimXFn,
		.dimY_fn         = nn_convLayer_dimYFn,
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

	self->VW = nn_tensor_new(dimW);
	if(self->VW == NULL)
	{
		goto fail_VW;
	}

	self->VB = nn_tensor_new(&dimB);
	if(self->VB == NULL)
	{
		goto fail_VB;
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

	self->dL_dX = nn_tensor_new(dimX);
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

nn_convLayer_t*
nn_convLayer_import(nn_arch_t* arch, jsmn_val_t* val)
{
	ASSERT(arch);
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	jsmn_val_t* val_dimX  = NULL;
	jsmn_val_t* val_dimW  = NULL;
	jsmn_val_t* val_flags = NULL;
	jsmn_val_t* val_W     = NULL;
	jsmn_val_t* val_B     = NULL;
	jsmn_val_t* val_VW    = NULL;
	jsmn_val_t* val_VB    = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		jsmn_keyval_t* kv;
		kv = (jsmn_keyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == JSMN_TYPE_PRIMITIVE)
		{
			if(strcmp(kv->key, "flags") == 0)
			{
				val_flags = kv->val;
			}
		}
		else if(kv->val->type == JSMN_TYPE_OBJECT)
		{
			if(strcmp(kv->key, "dimX") == 0)
			{
				val_dimX = kv->val;
			}
			else if(strcmp(kv->key, "dimW") == 0)
			{
				val_dimW = kv->val;
			}
			else if(strcmp(kv->key, "W") == 0)
			{
				val_W = kv->val;
			}
			else if(strcmp(kv->key, "B") == 0)
			{
				val_B = kv->val;
			}
			else if(strcmp(kv->key, "VW") == 0)
			{
				val_VW = kv->val;
			}
			else if(strcmp(kv->key, "VB") == 0)
			{
				val_VB = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_dimX  == NULL) ||
	   (val_dimW  == NULL) ||
	   (val_flags == NULL) ||
	   (val_W     == NULL) ||
	   (val_B     == NULL) ||
	   (val_VW    == NULL) ||
	   (val_VB    == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	int flags = strtol(val_flags->data, NULL, 0);

	nn_dim_t dimX;
	nn_dim_t dimW;
	if((nn_dim_load(&dimX, val_dimX) == 0) ||
	   (nn_dim_load(&dimW, val_dimW) == 0))
	{
		return NULL;
	}

	nn_convLayer_t* self;
	self = nn_convLayer_new(arch, &dimX, &dimW, flags);
	if(self == NULL)
	{
		return NULL;
	}

	// load tensors
	if((nn_tensor_load(self->W,  val_W)  == 0) ||
	   (nn_tensor_load(self->B,  val_B)  == 0) ||
	   (nn_tensor_load(self->VW, val_VW) == 0) ||
	   (nn_tensor_load(self->VB, val_VB) == 0))
	{
		goto fail_tensor;
	}

	// success
	return self;

	// failure
	fail_tensor:
		nn_convLayer_delete(&self);
	return NULL;
}

int nn_convLayer_export(nn_convLayer_t* self,
                        jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	nn_dim_t* dimX = nn_tensor_dim(self->dL_dX);
	nn_dim_t* dimW = nn_tensor_dim(self->W);

	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	ret &= jsmn_stream_key(stream, "%s", "dimX");
	ret &= nn_dim_store(dimX, stream);
	ret &= jsmn_stream_key(stream, "%s", "dimW");
	ret &= nn_dim_store(dimW, stream);
	ret &= jsmn_stream_key(stream, "%s", "flags");
	ret &= jsmn_stream_int(stream, self->flags);
	ret &= jsmn_stream_key(stream, "%s", "W");
	ret &= nn_tensor_store(self->W, stream);
	ret &= jsmn_stream_key(stream, "%s", "B");
	ret &= nn_tensor_store(self->B, stream);
	ret &= jsmn_stream_key(stream, "%s", "VW");
	ret &= nn_tensor_store(self->VW, stream);
	ret &= jsmn_stream_key(stream, "%s", "VB");
	ret &= nn_tensor_store(self->VB, stream);
	ret &= jsmn_stream_end(stream);

	return ret;
}

void nn_convLayer_delete(nn_convLayer_t** _self)
{
	ASSERT(_self);

	nn_convLayer_t* self = *_self;
	if(self)
	{
		nn_tensor_delete(&self->dL_dX);
		nn_tensor_delete(&self->dL_dB);
		nn_tensor_delete(&self->dL_dW);
		nn_tensor_delete(&self->VB);
		nn_tensor_delete(&self->VW);
		nn_tensor_delete(&self->Y);
		nn_tensor_delete(&self->B);
		nn_tensor_delete(&self->W);
		nn_layer_delete((nn_layer_t**) _self);
	}
}