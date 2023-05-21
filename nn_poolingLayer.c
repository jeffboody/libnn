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
#include "nn_poolingLayer.h"
#include "nn_layer.h"
#include "nn_tensor.h"

/***********************************************************
* private                                                  *
***********************************************************/

static void
nn_poolingLayer_max(nn_poolingLayer_t* self,
                    nn_tensor_t* X,
                    uint32_t m, uint32_t i,
                    uint32_t j, uint32_t k)
{
	ASSERT(self);
	ASSERT(X);

	nn_tensor_t* Y     = self->Y;
	nn_tensor_t* dY_dX = self->dY_dX;
	nn_dim_t*    dimX  = nn_tensor_dim(X);

	// initialize max value
	float    x;
	float    xmax = nn_tensor_get(X, m, i, j, k);
	uint32_t imax = i;
	uint32_t jmax = j;

	// compute range
	uint32_t h  = self->h;
	uint32_t w  = self->w;
	uint32_t xh = dimX->height;
	uint32_t xw = dimX->width;
	uint32_t i1 = i + h;
	uint32_t j1 = j + w;
	if(i1 > xh)
	{
		i1 = xh;
	}

	if(j1 > xw)
	{
		j1 = xw;
	}

	// find max value in tile
	uint32_t ii;
	uint32_t jj;
	for(ii = i; ii < i1; ++ii)
	{
		for(jj = j; jj < j1; ++jj)
		{
			x = nn_tensor_get(X, m, ii, jj, k);
			if(x > xmax)
			{
				xmax = x;
				imax = ii;
				jmax = jj;
			}
		}
	}

	// output
	nn_tensor_set(Y, m, i/h, j/w, k, xmax);

	// forward gradients
	nn_tensor_set(dY_dX, m, imax, jmax, k, 1.0f);
}

static void
nn_poolingLayer_avg(nn_poolingLayer_t* self,
                    nn_tensor_t* X,
                    uint32_t m, uint32_t i,
                    uint32_t j, uint32_t k)
{
	ASSERT(self);
	ASSERT(X);

	nn_tensor_t* Y     = self->Y;
	nn_tensor_t* dY_dX = self->dY_dX;
	nn_dim_t*    dimX  = nn_tensor_dim(X);

	// compute range
	uint32_t h  = self->h;
	uint32_t w  = self->w;
	uint32_t xh = dimX->height;
	uint32_t xw = dimX->width;
	uint32_t i1 = i + h;
	uint32_t j1 = j + w;
	if(i1 > xh)
	{
		i1 = xh;
	}

	if(j1 > xw)
	{
		j1 = xw;
	}

	// initalize average
	float di  = (float) (i1 - i);
	float dj  = (float) (j1 - j);
	float s   = 1.0f/(di*dj);
	float avg = 0.0f;

	// compute average
	uint32_t ii;
	uint32_t jj;
	for(ii = i; ii < i1; ++ii)
	{
		for(jj = j; jj < j1; ++jj)
		{
			// update sum
			avg += nn_tensor_get(X, m, ii, jj, k);

			// forward gradients
			nn_tensor_set(dY_dX, m, ii, jj, k, s);
		}
	}
	avg *= s;

	// output
	nn_tensor_set(Y, m, i/h, j/w, k, avg);
}

static nn_tensor_t*
nn_poolingLayer_forwardPassMaxFn(nn_layer_t* base, int mode,
                                 uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_poolingLayer_t* self = (nn_poolingLayer_t*) base;

	nn_tensor_t* Y     = self->Y;
	nn_tensor_t* dY_dX = self->dY_dX;
	nn_dim_t*    dimX  = nn_tensor_dim(X);
	uint32_t     xh    = dimX->height;
	uint32_t     xw    = dimX->width;
	uint32_t     xd    = dimX->depth;
	uint32_t     h     = self->h;
	uint32_t     w     = self->w;

	// clear forward gradients
	nn_tensor_clear(dY_dX);

	// output and forward gradients
	uint32_t m;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(m = 0; m < bs; ++m)
	{
		for(i = 0; i < xh; i += h)
		{
			for(j = 0; j < xw; j += w)
			{
				for(k = 0; k < xd; ++k)
				{
					nn_poolingLayer_max(self, X, m, i, j, k);
				}
			}
		}
	}

	return Y;
}

static nn_tensor_t*
nn_poolingLayer_forwardPassAvgFn(nn_layer_t* base, int mode,
                                 uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_poolingLayer_t* self = (nn_poolingLayer_t*) base;

	nn_tensor_t* Y     = self->Y;
	nn_dim_t*    dimX  = nn_tensor_dim(X);
	uint32_t     xh    = dimX->height;
	uint32_t     xw    = dimX->width;
	uint32_t     xd    = dimX->depth;
	uint32_t     h     = self->h;
	uint32_t     w     = self->w;

	// output and forward gradients
	uint32_t m;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(m = 0; m < bs; ++m)
	{
		for(i = 0; i < xh; i += h)
		{
			for(j = 0; j < xw; j += w)
			{
				for(k = 0; k < xd; ++k)
				{
					nn_poolingLayer_avg(self, X, m, i, j, k);
				}
			}
		}
	}

	return Y;
}

static nn_tensor_t*
nn_poolingLayer_backpropFn(nn_layer_t* base, uint32_t bs,
                        nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,yh,yw,xd)

	nn_poolingLayer_t* self = (nn_poolingLayer_t*) base;

	nn_tensor_t* dY_dX = self->dY_dX;
	nn_tensor_t* dL_dX = self->dL_dX;
	nn_dim_t*    dimX  = nn_tensor_dim(dL_dX);
	uint32_t     xh    = dimX->height;
	uint32_t     xw    = dimX->width;
	uint32_t     xd    = dimX->depth;
	uint32_t     h     = self->h;
	uint32_t     w     = self->w;

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
		for(i = 0; i < xh; ++i)
		{
			for(j = 0; j < xw; ++j)
			{
				for(k = 0; k < xd; ++k)
				{
					dl_dy = nn_tensor_get(dL_dY, m, i/h, j/w, k);
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
nn_poolingLayer_dimXFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_poolingLayer_t* self = (nn_poolingLayer_t*) base;

	return nn_tensor_dim(self->dL_dX);
}

static nn_dim_t*
nn_poolingLayer_dimYFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_poolingLayer_t* self = (nn_poolingLayer_t*) base;

	return nn_tensor_dim(self->Y);
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_poolingLayer_t*
nn_poolingLayer_new(nn_arch_t* arch,
                    nn_dim_t* dimX,
                    uint32_t h, uint32_t w,
                    int mode)
{
	ASSERT(arch);
	ASSERT(dimX);

	nn_layerInfo_t info =
	{
		.arch            = arch,
		.forward_pass_fn = nn_poolingLayer_forwardPassMaxFn,
		.backprop_fn     = nn_poolingLayer_backpropFn,
		.dimX_fn         = nn_poolingLayer_dimXFn,
		.dimY_fn         = nn_poolingLayer_dimYFn,
	};

	if(mode == NN_POOLING_LAYER_MODE_AVERAGE)
	{
		info.forward_pass_fn = nn_poolingLayer_forwardPassAvgFn;
	}

	nn_poolingLayer_t* self;
	self = (nn_poolingLayer_t*)
	       nn_layer_new(sizeof(nn_poolingLayer_t), &info);
	if(self == NULL)
	{
		return NULL;
	}

	self->h    = h;
	self->w    = w;
	self->mode = mode;

	nn_dim_t dimY =
	{
		.count  = dimX->count,
		.height = dimX->height/h,
		.width  = dimX->width/w,
		.depth  = dimX->depth,
	};

	self->Y = nn_tensor_new(&dimY);
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

nn_poolingLayer_t*
nn_poolingLayer_import(nn_arch_t* arch, jsmn_val_t* val)
{
	ASSERT(arch);
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	jsmn_val_t* val_dimX = NULL;
	jsmn_val_t* val_h    = NULL;
	jsmn_val_t* val_w    = NULL;
	jsmn_val_t* val_mode = NULL;

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
		}
		else if(kv->val->type == JSMN_TYPE_PRIMITIVE)
		{
			if(strcmp(kv->key, "h") == 0)
			{
				val_h = kv->val;
			}
			else if(strcmp(kv->key, "w") == 0)
			{
				val_w = kv->val;
			}
		}
		else if(kv->val->type == JSMN_TYPE_STRING)
		{
			if(strcmp(kv->key, "mode") == 0)
			{
				val_mode = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_dimX  == NULL) ||
	   (val_h     == NULL) ||
	   (val_w     == NULL) ||
	   (val_mode  == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_dim_t dimX;
	if(nn_dim_load(&dimX, val_dimX) == 0)
	{
		return NULL;
	}

	uint32_t h = (uint32_t) strtol(val_h->data, NULL, 0);
	uint32_t w = (uint32_t) strtol(val_w->data, NULL, 0);

	int mode = NN_POOLING_LAYER_MODE_MAX;
	if(strcmp(val_mode->data, "average") == 0)
	{
		mode = NN_POOLING_LAYER_MODE_AVERAGE;
	}

	return nn_poolingLayer_new(arch, &dimX, h, w, mode);
}

int nn_poolingLayer_export(nn_poolingLayer_t* self,
                           jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	nn_dim_t* dimX = nn_tensor_dim(self->dL_dX);

	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	ret &= jsmn_stream_key(stream, "%s", "dimX");
	ret &= nn_dim_store(dimX, stream);
	ret &= jsmn_stream_key(stream, "%s", "h");
	ret &= jsmn_stream_int(stream, (int) self->h);
	ret &= jsmn_stream_key(stream, "%s", "w");
	ret &= jsmn_stream_int(stream, (int) self->w);
	ret &= jsmn_stream_key(stream, "%s", "mode");
	if(self->mode == NN_POOLING_LAYER_MODE_AVERAGE)
	{
		ret &= jsmn_stream_string(stream, "%s", "average");
	}
	else
	{
		ret &= jsmn_stream_string(stream, "%s", "max");
	}
	ret &= jsmn_stream_end(stream);

	return ret;
}

void nn_poolingLayer_delete(nn_poolingLayer_t** _self)
{
	ASSERT(_self);

	nn_poolingLayer_t* self = *_self;
	if(self)
	{
		nn_tensor_delete(&self->dL_dX);
		nn_tensor_delete(&self->dY_dX);
		nn_tensor_delete(&self->Y);
		nn_layer_delete((nn_layer_t**) _self);
	}
}
