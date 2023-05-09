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

#include <stdlib.h>
#include <string.h>

#define LOG_TAG "nn"
#include "../libcc/cc_log.h"
#include "../libcc/cc_memory.h"
#include "nn_arch.h"
#include "nn_mseLoss.h"
#include "nn_tensor.h"

/***********************************************************
* private                                                  *
***********************************************************/

static nn_tensor_t*
nn_mseLoss_backpropFn(nn_loss_t* base, nn_tensor_t* Y,
                      nn_tensor_t* Yt)
{
	ASSERT(base);
	ASSERT(Y);
	ASSERT(Yt);

	nn_mseLoss_t* self = (nn_mseLoss_t*) base;

	nn_tensor_t* dL_dY = self->dL_dY;
	nn_dim_t*    dim   = nn_tensor_dim(Y);
	uint32_t     bs    = base->arch->batch_size;
	uint32_t     yh    = dim->height;
	uint32_t     yw    = dim->width;
	uint32_t     yd    = dim->depth;

	float    y;
	float    yt;
	float    dy;
	float    M    = (float) (bs*yh*yw*yd);
	float    loss = 0.0f;
	uint32_t m;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(m = 0; m < bs; ++m)
	{
		for(i = 0; i < yh; ++i)
		{
			for(j = 0; j < yw; ++j)
			{
				for(k = 0; k < yd; ++k)
				{
					y     = nn_tensor_get(Y, m, i, j, k);
					yt    = nn_tensor_get(Yt, m, i, j, k);
					dy    = y - yt;
					loss += dy*dy;
					nn_tensor_set(dL_dY, m, i, j, k, 2.0f*dy);
				}
			}
		}
	}
	base->loss = loss/M;

	return dL_dY;
}

static nn_dim_t*
nn_mseLoss_dimYFn(nn_loss_t* base)
{
	ASSERT(base);

	nn_mseLoss_t* self = (nn_mseLoss_t*) base;

	return nn_tensor_dim(self->dL_dY);
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_mseLoss_t*
nn_mseLoss_new(nn_arch_t* arch, nn_dim_t* dimY)
{
	ASSERT(arch);

	nn_lossInfo_t info =
	{
		.arch        = arch,
		.backprop_fn = nn_mseLoss_backpropFn,
		.dimY_fn     = nn_mseLoss_dimYFn,
	};

	nn_mseLoss_t* self;
	self = (nn_mseLoss_t*)
	       nn_loss_new(sizeof(nn_mseLoss_t), &info);
	if(self == NULL)
	{
		return NULL;
	}

	self->dL_dY = nn_tensor_new(dimY);
	if(self->dL_dY == NULL)
	{
		goto fail_dL_dY;
	}

	// success
	return self;

	// failure
	fail_dL_dY:
		nn_loss_delete((nn_loss_t**) &self);
	return NULL;
}

nn_mseLoss_t*
nn_mseLoss_import(nn_arch_t* arch, jsmn_val_t* val)
{
	ASSERT(arch);
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	jsmn_val_t* val_dimY = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		jsmn_keyval_t* kv;
		kv = (jsmn_keyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == JSMN_TYPE_OBJECT)
		{
			if(strcmp(kv->key, "dimY") == 0)
			{
				val_dimY = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if(val_dimY == NULL)
	{
		LOGE("invalid");
		return NULL;
	}

	nn_dim_t dimY;
	if(nn_dim_load(&dimY, val_dimY) == 0)
	{
		return NULL;
	}

	nn_mseLoss_t* self;
	self = nn_mseLoss_new(arch, &dimY);
	if(self == NULL)
	{
		return NULL;
	}

	return self;
}

int nn_mseLoss_export(nn_mseLoss_t* self,
                      jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	nn_dim_t* dimY = nn_tensor_dim(self->dL_dY);

	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	ret &= jsmn_stream_key(stream, "%s", "dimY");
	ret &= nn_dim_store(dimY, stream);
	ret &= jsmn_stream_end(stream);

	return ret;
}

void nn_mseLoss_delete(nn_mseLoss_t** _self)
{
	ASSERT(_self);

	nn_mseLoss_t* self = *_self;
	if(self)
	{
		nn_tensor_delete(&self->dL_dY);
		nn_loss_delete((nn_loss_t**) _self);
	}
}
