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

	nn_weightLayer_t* self = (nn_weightLayer_t*) base;
	nn_arch_t*        arch = base->arch;

	nn_tensor_t* W    = self->W;
	nn_tensor_t* B    = self->B;
	nn_tensor_t* Y    = self->Y;
	nn_dim_t*    dimX = nn_tensor_dim(X);
	nn_dim_t*    dimY = nn_tensor_dim(Y);
	uint32_t     xd   = dimX->depth;
	uint32_t     nc   = dimY->depth;
	uint32_t     bs   = arch->batch_size;

	float    w;
	float    x;
	float    y;
	uint32_t m;
	uint32_t n;
	uint32_t k;
	for(m = 0; m < bs; ++m)
	{
		for(n = 0; n < nc; ++n)
		{
			// initialize y
			if(self->flags & NN_WEIGHT_LAYER_FLAG_DISABLE_BIAS)
			{
				y = 0.0f;
			}
			else
			{
				y = nn_tensor_get(B, n, 0, 0, 0);
			}

			// compute weighted sum
			for(k = 0; k < xd; ++k)
			{
				// compute weighted sum
				x = nn_tensor_get(X, m, 0, 0, k);
				w = nn_tensor_get(W, n, 0, 0, k);
				y += w*x;
			}
			nn_tensor_set(Y, m, 0, 0, n, y);
		}
	}

	// store reference
	self->X = X;

	return Y;
}

static nn_tensor_t*
nn_weightLayer_backpropFn(nn_layer_t* base,
                          nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,1,1,nc)

	nn_weightLayer_t* self = (nn_weightLayer_t*) base;
	nn_arch_t*        arch = base->arch;

	nn_tensor_t* W      = self->W;
	nn_tensor_t* B      = self->B;
	nn_tensor_t* VW     = self->VW;
	nn_tensor_t* VB     = self->VB;
	nn_tensor_t* dY_dX  = W;
	nn_tensor_t* dY_dW  = self->X;
	nn_dim_t*    dim    = nn_tensor_dim(W);
	uint32_t     bs     = arch->batch_size;
	uint32_t     nc     = dim->count;
	uint32_t     xd     = dim->depth;
	float        lr     = arch->learning_rate;
	float        mu     = arch->momentum_decay;
	float        lambda = arch->l2_lambda;

	// clear backprop gradients
	nn_tensor_t* dL_dW = self->dL_dW;
	nn_tensor_t* dL_dB = self->dL_dB;
	nn_tensor_t* dL_dX = self->dL_dX;
	nn_tensor_clear(dL_dW);
	if((self->flags & NN_WEIGHT_LAYER_FLAG_DISABLE_BIAS) == 0)
	{
		nn_tensor_clear(dL_dB);
	}
	nn_tensor_clear(dL_dX);

	// sum gradients and backpropagate loss
	float    dy_dx;
	float    dy_dw;
	float    dy_db  = 1.0f;
	float    dl_dy;
	uint32_t m;
	uint32_t n;
	uint32_t k;
	for(m = 0; m < bs; ++m)
	{
		for(n = 0; n < nc; ++n)
		{
			dl_dy = nn_tensor_get(dL_dY, m, 0, 0, n);

			for(k = 0; k < xd; ++k)
			{
				// backpropagate dL_dX
				dy_dx = nn_tensor_get(dY_dX, n, 0, 0, k);
				nn_tensor_add(dL_dX, m, 0, 0, k, dl_dy*dy_dx);

				// sum dL_dW
				dy_dw = nn_tensor_get(dY_dW, n, 0, 0, k);
				nn_tensor_add(dL_dW, n, 0, 0, k, dl_dy*dy_dw);
			}

			// sum dL_dB
			if((self->flags & NN_WEIGHT_LAYER_FLAG_DISABLE_BIAS) == 0)
			{
				nn_tensor_add(dL_dB, n, 0, 0, 0, dl_dy*dy_db);
			}
		}
	}

	// update parameters
	float v0;
	float v1;
	float w;
	float dl_dw;
	float dl_db;
	float s = 1.0f/((float) bs);
	for(n = 0; n < nc; ++n)
	{
		// weights
		for(k = 0; k < xd; ++k)
		{
			dl_dw = s*nn_tensor_get(dL_dW, n, 0, 0, k);
			w     = nn_tensor_get(W, n, 0, 0, k);

			// Nesterov Momentum Update and L2 Regularization
			v0 = nn_tensor_get(VW, n, 0, 0, k);
			v1 = mu*v0 - lr*(dl_dw + 2*lambda*w);
			nn_tensor_set(VW, n, 0, 0, k, v1);
			nn_tensor_add(W, n, 0, 0, k, -mu*v0 + (1 - mu)*v1);
		}

		// bias
		if((self->flags & NN_WEIGHT_LAYER_FLAG_DISABLE_BIAS) == 0)
		{
			dl_db = s*nn_tensor_get(dL_dB, n, 0, 0, 0);

			// Nesterov Momentum Update
			v0    = nn_tensor_get(VB, n, 0, 0, 0);
			v1    = mu*v0 - lr*dl_db;
			nn_tensor_set(VB, n, 0, 0, 0, v1);
			nn_tensor_add(B, n, 0, 0, 0, -mu*v0 + (1 - mu)*v1);
		}
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

	float    w;
	uint32_t k;
	uint32_t n;
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

	float    w;
	uint32_t k;
	uint32_t n;
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

	// X and Y must be flattened
	if((dimX->height != 1) || (dimX->width != 1) ||
	   (dimY->height != 1) || (dimY->width != 1))
	{
		LOGE("invalid dimX=%u:%u, dimY=%u:%u",
		     dimX->height, dimX->width,
		     dimY->height, dimY->width);
		return NULL;
	}

	uint32_t xd = dimX->depth;
	uint32_t nc = dimY->depth;

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

	self->dL_dW = nn_tensor_new(&dimW);
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

nn_weightLayer_t*
nn_weightLayer_import(nn_arch_t* arch, jsmn_val_t* val)
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
			if(strcmp(kv->key, "val_dimX") == 0)
			{
				val_dimX = kv->val;
			}
			else if(strcmp(kv->key, "val_dimW") == 0)
			{
				val_dimW = kv->val;
			}
			else if(strcmp(kv->key, "val_W") == 0)
			{
				val_W = kv->val;
			}
			else if(strcmp(kv->key, "val_B") == 0)
			{
				val_B = kv->val;
			}
			else if(strcmp(kv->key, "val_VW") == 0)
			{
				val_VW = kv->val;
			}
			else if(strcmp(kv->key, "val_VB") == 0)
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

	nn_weightLayer_t* self;
	self = nn_weightLayer_new(arch, &dimX, &dimW, flags);
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
		nn_weightLayer_delete(&self);
	return NULL;
}

int nn_weightLayer_export(nn_weightLayer_t* self,
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

void nn_weightLayer_delete(nn_weightLayer_t** _self)
{
	ASSERT(_self);

	nn_weightLayer_t* self = *_self;
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
