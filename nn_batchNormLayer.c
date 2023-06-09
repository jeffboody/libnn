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

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define LOG_TAG "nn"
#include "../libcc/cc_log.h"
#include "../libcc/cc_memory.h"
#include "nn_arch.h"
#include "nn_batchNormLayer.h"
#include "nn_layer.h"
#include "nn_tensor.h"

/***********************************************************
* private                                                  *
***********************************************************/

static nn_tensor_t*
nn_batchNormLayer_forwardPassFn(nn_layer_t* base, int mode,
                                uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_batchNormLayer_t* self = (nn_batchNormLayer_t*) base;
	nn_arch_t*           arch = base->arch;

	nn_tensor_t* G        = self->G;
	nn_tensor_t* B        = self->B;
	nn_tensor_t* Xhat     = self->Xhat;
	nn_tensor_t* Y        = self->Y;
	nn_tensor_t* Xmean_mb = self->Xmean_mb;
	nn_tensor_t* Xvar_mb  = self->Xvar_mb;
	nn_tensor_t* Xmean_ra = self->Xmean_ra;
	nn_tensor_t* Xvar_ra  = self->Xvar_ra;
	nn_dim_t*    dim      = nn_tensor_dim(X);
	uint32_t     xh       = dim->height;
	uint32_t     xw       = dim->width;
	uint32_t     xd       = dim->depth;

	// prediction (running average) or
	// training (mini-batch)
	nn_tensor_t* Xmean = self->Xmean_ra;
	nn_tensor_t* Xvar  = self->Xvar_ra;

	uint32_t m;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	if(mode == NN_LAYER_MODE_TRAIN)
	{
		Xmean = self->Xmean_mb;
		Xvar  = self->Xvar_mb;

		// compute mini-batch mean
		// update running mean
		float xmean_ra;
		float xmean_mb;
		float momentum = arch->batch_momentum;
		float M = (float) (bs*xh*xw);
		for(k = 0; k < xd; ++k)
		{
			// compute mini-batch mean
			xmean_mb = 0.0f;
			for(m = 0; m < bs; ++m)
			{
				for(i = 0; i < xh; ++i)
				{
					for(j = 0; j < xw; ++j)
					{
						xmean_mb += nn_tensor_get(X, m, i, j, k);
					}
				}
			}
			xmean_mb /= M;
			nn_tensor_set(Xmean_mb, 0, 0, 0, k, xmean_mb);

			// update running mean
			xmean_ra = nn_tensor_get(Xmean_ra, 0, 0, 0, k);
			xmean_ra = momentum*xmean_ra + (1 - momentum)*xmean_mb;
			nn_tensor_set(Xmean_ra, 0, 0, 0, k, xmean_ra);
		}

		// compute mini-batch variance
		// update running variance
		float xvar_ra;
		float xvar_mb;
		float dx;
		for(k = 0; k < xd; ++k)
		{
			// compute mini-batch variance
			xvar_mb  = 0.0f;
			xmean_mb = nn_tensor_get(Xmean_mb, 0, 0, 0, k);
			for(m = 0; m < bs; ++m)
			{
				for(i = 0; i < xh; ++i)
				{
					for(j = 0; j < xw; ++j)
					{
						dx       = nn_tensor_get(X, m, i, j, k) - xmean_mb;
						xvar_mb += dx*dx;
					}
				}
			}
			xvar_mb /= M;
			nn_tensor_set(Xvar_mb, 0, 0, 0, k, xvar_mb);

			// update running variance
			xvar_ra = nn_tensor_get(Xvar_ra, 0, 0, 0, k);
			xvar_ra = momentum*xvar_ra + (1 - momentum)*xvar_mb;
			nn_tensor_set(Xvar_ra, 0, 0, 0, k, xvar_ra);
		}
	}

	// compute Xhat
	float xhat;
	float x;
	float xmean;
	float xvar;
	for(k = 0; k < xd; ++k)
	{
		xmean = nn_tensor_get(Xmean, 0, 0, 0, k);
		xvar  = nn_tensor_get(Xvar,  0, 0, 0, k);
		for(m = 0; m < bs; ++m)
		{
			for(i = 0; i < xh; ++i)
			{
				for(j = 0; j < xw; ++j)
				{
					x    = nn_tensor_get(X, m, i, j, k);
					xhat = (x - xmean)/(sqrtf(xvar) + FLT_EPSILON);
					nn_tensor_set(Xhat, m, i, j, k, xhat);
				}
			}
		}
	}

	// compute Y
	float y;
	float gamma;
	float beta;
	for(k = 0; k < xd; ++k)
	{
		gamma = nn_tensor_get(G, 0, 0, 0, k);
		beta  = nn_tensor_get(B, 0, 0, 0, k);
		for(m = 0; m < bs; ++m)
		{
			for(i = 0; i < xh; ++i)
			{
				for(j = 0; j < xw; ++j)
				{
					xhat = nn_tensor_get(Xhat, m, i, j, k);
					y    = gamma*xhat + beta;
					nn_tensor_set(Y, m, i, j, k, y);
				}
			}
		}
	}

	return Y;
}

static void
nn_batchNormLayer_backpropSum(nn_batchNormLayer_t* self,
                              uint32_t bs, uint32_t k,
                              float* _b, float* _c)
{
	ASSERT(self);
	ASSERT(_b);
	ASSERT(_c);

	nn_tensor_t* Xhat     = self->Xhat;
	nn_tensor_t* dL_dXhat = self->dL_dXhat;
	nn_dim_t*    dim      = nn_tensor_dim(Xhat);
	uint32_t     xh       = dim->height;
	uint32_t     xw       = dim->width;

	float b = 0.0f;
	float c = 0.0f;

	float    dl_dxhat;
	float    xhat;
	uint32_t m;
	uint32_t i;
	uint32_t j;
	for(m = 0; m < bs; ++m)
	{
		for(i = 0; i < xh; ++i)
		{
			for(j = 0; j < xw; ++j)
			{
				dl_dxhat = nn_tensor_get(dL_dXhat, m, i, j, k);
				xhat     = nn_tensor_get(Xhat, m, i, j, k);
				b       += dl_dxhat;
				c       += dl_dxhat*xhat;
			}
		}
	}

	*_b = b;
	*_c = c;
}

static nn_tensor_t*
nn_batchNormLayer_backpropFn(nn_layer_t* base, uint32_t bs,
                             nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,xh,xw,xd)

	nn_batchNormLayer_t* self = (nn_batchNormLayer_t*) base;
	nn_arch_t*           arch = base->arch;

	nn_tensor_t* G        = self->G;
	nn_tensor_t* B        = self->B;
	nn_tensor_t* Xhat     = self->Xhat;
	nn_tensor_t* Xvar_mb  = self->Xvar_mb;
	nn_tensor_t* dL_dXhat = self->dL_dXhat;
	nn_dim_t*    dim      = nn_tensor_dim(dL_dY);
	float        lr       = arch->learning_rate;
	uint32_t     xh       = dim->height;
	uint32_t     xw       = dim->width;
	uint32_t     xd       = dim->depth;

	// compute dL_dXhat
	float dl_dy;
	float gamma;
	uint32_t m;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(k = 0; k < xd; ++k)
	{
		gamma = nn_tensor_get(G, 0, 0, 0, k);
		for(m = 0; m < bs; ++m)
		{
			for(i = 0; i < xh; ++i)
			{
				for(j = 0; j < xw; ++j)
				{
					dl_dy = nn_tensor_get(dL_dY, m, i, j, k);
					nn_tensor_set(dL_dXhat, m, i, j, k, dl_dy*gamma);
				}
			}
		}
	}

	// update G and B
	// compute dL_dX
	float dl_db;
	float dl_dg;
	float xhat;
	float a;
	float b;
	float c;
	float d;
	float xvar;
	float dl_dxhat;
	float M = (float) (bs*xh*xw);
	for(k = 0; k < xd; ++k)
	{
		dl_dg = 0.0f;
		dl_db = 0.0f;
		xvar  = nn_tensor_get(Xvar_mb, 0, 0, 0, k);
		d     = M*sqrtf(xvar + FLT_EPSILON);
		nn_batchNormLayer_backpropSum(self, bs, k, &b, &c);
		for(m = 0; m < bs; ++m)
		{
			for(i = 0; i < xh; ++i)
			{
				for(j = 0; j < xw; ++j)
				{
					// compute dl_dg and dl_db
					dl_dy  = nn_tensor_get(dL_dY, m, i, j, k);
					xhat   = nn_tensor_get(Xhat, m, i, j, k);
					dl_dg += dl_dy*xhat;
					dl_db += dl_dy;

					// compute dL_dX
					// dL_dY replaced by dL_dX
					dl_dxhat = nn_tensor_get(dL_dXhat, m, i, j, k);
					a        = M*dl_dxhat;
					nn_tensor_set(dL_dY, m, i, j, k, (a - b - xhat*c)/d);
				}
			}
		}

		// update G and B
		nn_tensor_add(G, 0, 0, 0, k, -lr*dl_dg);
		nn_tensor_add(B, 0, 0, 0, k, -lr*dl_db);
	}

	// dL_dY replaced by dL_dX
	return dL_dY;
}

static nn_dim_t*
nn_batchNormLayer_dimXFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_batchNormLayer_t* self = (nn_batchNormLayer_t*) base;

	return nn_tensor_dim(self->Xhat);
}

static nn_dim_t*
nn_batchNormLayer_dimYFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_batchNormLayer_t* self = (nn_batchNormLayer_t*) base;

	return nn_tensor_dim(self->Y);
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_batchNormLayer_t*
nn_batchNormLayer_new(nn_arch_t* arch, nn_dim_t* dimX)
{
	ASSERT(arch);
	ASSERT(dimX);

	uint32_t xd = dimX->depth;

	nn_dim_t dim_111d =
	{
		.count  = 1,
		.height = 1,
		.width  = 1,
		.depth  = xd,
	};

	nn_layerInfo_t info =
	{
		.arch            = arch,
		.forward_pass_fn = nn_batchNormLayer_forwardPassFn,
		.backprop_fn     = nn_batchNormLayer_backpropFn,
		.dimX_fn         = nn_batchNormLayer_dimXFn,
		.dimY_fn         = nn_batchNormLayer_dimYFn,
	};

	nn_batchNormLayer_t* self;
	self = (nn_batchNormLayer_t*)
	       nn_layer_new(sizeof(nn_batchNormLayer_t), &info);
	if(self == NULL)
	{
		return NULL;
	}

	self->G = nn_tensor_new(&dim_111d);
	if(self->G == NULL)
	{
		goto fail_G;
	}

	// initialize G to 1.0f
	uint32_t k;
	for(k = 0; k < xd; ++k)
	{
		nn_tensor_set(self->G, 0, 0, 0, k, 1.0f);
	}

	self->B = nn_tensor_new(&dim_111d);
	if(self->B == NULL)
	{
		goto fail_B;
	}

	self->Xhat = nn_tensor_new(dimX);
	if(self->Xhat == NULL)
	{
		goto fail_Xhat;
	}

	self->Y = nn_tensor_new(dimX);
	if(self->Y == NULL)
	{
		goto fail_Y;
	}

	self->Xmean_mb = nn_tensor_new(&dim_111d);
	if(self->Xmean_mb == NULL)
	{
		goto fail_Xmean_mb;
	}

	self->Xvar_mb = nn_tensor_new(&dim_111d);
	if(self->Xvar_mb == NULL)
	{
		goto fail_Xvar_mb;
	}

	self->Xmean_ra = nn_tensor_new(&dim_111d);
	if(self->Xmean_ra == NULL)
	{
		goto fail_Xmean_ra;
	}

	self->Xvar_ra = nn_tensor_new(&dim_111d);
	if(self->Xvar_ra == NULL)
	{
		goto fail_Xvar_ra;
	}

	self->dL_dXhat = nn_tensor_new(dimX);
	if(self->dL_dXhat == NULL)
	{
		goto fail_dL_dXhat;
	}

	// success
	return self;

	// failure
	fail_dL_dXhat:
		nn_tensor_delete(&self->Xvar_ra);
	fail_Xvar_ra:
		nn_tensor_delete(&self->Xmean_ra);
	fail_Xmean_ra:
		nn_tensor_delete(&self->Xvar_mb);
	fail_Xvar_mb:
		nn_tensor_delete(&self->Xmean_mb);
	fail_Xmean_mb:
		nn_tensor_delete(&self->Y);
	fail_Y:
		nn_tensor_delete(&self->Xhat);
	fail_Xhat:
		nn_tensor_delete(&self->B);
	fail_B:
		nn_tensor_delete(&self->G);
	fail_G:
		nn_layer_delete((nn_layer_t**) &self);
	return NULL;
}

nn_batchNormLayer_t*
nn_batchNormLayer_import(nn_arch_t* arch, jsmn_val_t* val)
{
	ASSERT(arch);
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	jsmn_val_t* val_dimX     = NULL;
	jsmn_val_t* val_G        = NULL;
	jsmn_val_t* val_B        = NULL;
	jsmn_val_t* val_Xmean_ra = NULL;
	jsmn_val_t* val_Xvar_ra  = NULL;

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
			else if(strcmp(kv->key, "G") == 0)
			{
				val_G = kv->val;
			}
			else if(strcmp(kv->key, "B") == 0)
			{
				val_B = kv->val;
			}
			else if(strcmp(kv->key, "Xmean_ra") == 0)
			{
				val_Xmean_ra = kv->val;
			}
			else if(strcmp(kv->key, "Xvar_ra") == 0)
			{
				val_Xvar_ra = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_dimX     == NULL) ||
	   (val_G        == NULL) ||
	   (val_B        == NULL) ||
	   (val_Xmean_ra == NULL) ||
	   (val_Xvar_ra  == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_dim_t dimX;
	if(nn_dim_load(&dimX, val_dimX) == 0)
	{
		return NULL;
	}

	nn_batchNormLayer_t* self;
	self = nn_batchNormLayer_new(arch, &dimX);
	if(self == NULL)
	{
		return NULL;
	}

	// load tensors
	if((nn_tensor_load(self->G,        val_G) == 0)        ||
	   (nn_tensor_load(self->B,        val_B) == 0)        ||
	   (nn_tensor_load(self->Xmean_ra, val_Xmean_ra) == 0) ||
	   (nn_tensor_load(self->Xvar_ra,  val_Xvar_ra) == 0))
	{
		goto fail_tensor;
	}

	// success
	return self;

	// failure
	fail_tensor:
		nn_batchNormLayer_delete(&self);
	return NULL;
}

int nn_batchNormLayer_export(nn_batchNormLayer_t* self,
                             jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	nn_dim_t* dimX = nn_tensor_dim(self->Xhat);

	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	ret &= jsmn_stream_key(stream, "%s", "dimX");
	ret &= nn_dim_store(dimX, stream);
	ret &= jsmn_stream_key(stream, "%s", "G");
	ret &= nn_tensor_store(self->G, stream);
	ret &= jsmn_stream_key(stream, "%s", "B");
	ret &= nn_tensor_store(self->B, stream);
	ret &= jsmn_stream_key(stream, "%s", "Xmean_ra");
	ret &= nn_tensor_store(self->Xmean_ra, stream);
	ret &= jsmn_stream_key(stream, "%s", "Xvar_ra");
	ret &= nn_tensor_store(self->Xvar_ra, stream);
	ret &= jsmn_stream_end(stream);

	return ret;
}

void nn_batchNormLayer_delete(nn_batchNormLayer_t** _self)
{
	ASSERT(_self);

	nn_batchNormLayer_t* self = *_self;
	if(self)
	{
		nn_tensor_delete(&self->dL_dXhat);
		nn_tensor_delete(&self->Xvar_ra);
		nn_tensor_delete(&self->Xmean_ra);
		nn_tensor_delete(&self->Xvar_mb);
		nn_tensor_delete(&self->Xmean_mb);
		nn_tensor_delete(&self->Y);
		nn_tensor_delete(&self->Xhat);
		nn_tensor_delete(&self->B);
		nn_tensor_delete(&self->G);
		nn_layer_delete((nn_layer_t**) &self);
	}
}
