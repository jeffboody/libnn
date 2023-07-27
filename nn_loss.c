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
#include "../libcc/math/cc_float.h"
#include "../libcc/cc_log.h"
#include "../libcc/cc_memory.h"
#include "nn_arch.h"
#include "nn_loss.h"
#include "nn_tensor.h"

const char* NN_LOSS_STRING_MSE = "mse";
const char* NN_LOSS_STRING_MAE = "mae";
const char* NN_LOSS_STRING_BCE = "bce";

/***********************************************************
* public - loss functions                                  *
***********************************************************/

nn_tensor_t*
nn_loss_mse(nn_loss_t* self, uint32_t bs,
            nn_tensor_t* Y, nn_tensor_t* Yt)
{
	ASSERT(self);
	ASSERT(Y);
	ASSERT(Yt);

	nn_tensor_t* dL_dY = self->dL_dY;
	nn_dim_t*    dim   = nn_tensor_dim(Y);
	uint32_t     yh    = dim->height;
	uint32_t     yw    = dim->width;
	uint32_t     yd    = dim->depth;

	float    y;
	float    yt;
	float    dy;
	float    dl_dy;
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
					dl_dy = 2.0f*dy;
					loss += dy*dy;
					nn_tensor_set(dL_dY, m, i, j, k, dl_dy);
				}
			}
		}
	}
	self->loss = loss/M;

	return dL_dY;
}

nn_tensor_t*
nn_loss_mae(nn_loss_t* self, uint32_t bs,
            nn_tensor_t* Y, nn_tensor_t* Yt)
{
	ASSERT(self);
	ASSERT(Y);
	ASSERT(Yt);

	nn_tensor_t* dL_dY = self->dL_dY;
	nn_dim_t*    dim   = nn_tensor_dim(Y);
	uint32_t     yh    = dim->height;
	uint32_t     yw    = dim->width;
	uint32_t     yd    = dim->depth;

	float    y;
	float    yt;
	float    dy;
	float    ady;
	float    dl_dy;
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
					ady   = fabs(dy);
					dl_dy = dy/(ady + FLT_EPSILON);
					loss += ady;
					nn_tensor_set(dL_dY, m, i, j, k, dl_dy);
				}
			}
		}
	}
	self->loss = loss/M;

	return dL_dY;
}

nn_tensor_t*
nn_loss_bce(nn_loss_t* self, uint32_t bs,
            nn_tensor_t* Y, nn_tensor_t* Yt)
{
	ASSERT(self);
	ASSERT(Y);
	ASSERT(Yt);

	nn_tensor_t* dL_dY = self->dL_dY;
	nn_dim_t*    dim   = nn_tensor_dim(Y);
	uint32_t     yh    = dim->height;
	uint32_t     yw    = dim->width;
	uint32_t     yd    = dim->depth;

	float    y;
	float    yt;
	float    dl_dy;
	float    M       = (float) (bs*yh*yw*yd);
	float    loss    = 0.0f;
	float    epsilon = FLT_EPSILON;
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
					y     = cc_clamp(y, epsilon, 1.0f - epsilon);
					yt    = nn_tensor_get(Yt, m, i, j, k);
					dl_dy = -(y - yt)/(logf(10.0f)*(y - 1.0f)*y + epsilon);
					loss += -(yt*log10f(y + epsilon) +
					          (1.0f - yt)*log10f(1.0f - y + epsilon));
					nn_tensor_set(dL_dY, m, i, j, k, dl_dy);
				}
			}
		}
	}
	self->loss = loss/M;

	return dL_dY;
}

const char* nn_loss_string(nn_loss_fn loss_fn)
{
	ASSERT(loss_fn);

	if(loss_fn == nn_loss_mse)
	{
		return NN_LOSS_STRING_MSE;
	}
	else if(loss_fn == nn_loss_mae)
	{
		return NN_LOSS_STRING_MAE;
	}
	else if(loss_fn == nn_loss_bce)
	{
		return NN_LOSS_STRING_BCE;
	}

	LOGE("invalid");
	return NULL;
}

nn_loss_fn nn_loss_function(const char* str)
{
	ASSERT(str);

	if(strcmp(str, NN_LOSS_STRING_MSE) == 0)
	{
		return nn_loss_mse;
	}
	else if(strcmp(str, NN_LOSS_STRING_MAE) == 0)
	{
		return nn_loss_mae;
	}
	else if(strcmp(str, NN_LOSS_STRING_BCE) == 0)
	{
		return nn_loss_bce;
	}

	LOGE("invalid %s", str);
	return NULL;
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_loss_t*
nn_loss_new(nn_arch_t* arch, nn_dim_t* dimY,
            nn_loss_fn loss_fn)
{
	ASSERT(arch);
	ASSERT(dimY);
	ASSERT(loss_fn);

	nn_loss_t* self;
	self = (nn_loss_t*) CALLOC(1, sizeof(nn_loss_t));
	if(self == NULL)
	{
		LOGE("CALLOC failed");
		return NULL;
	}

	self->arch    = arch;
	self->loss_fn = loss_fn;

	self->dL_dY = nn_tensor_new(dimY);
	if(self->dL_dY == NULL)
	{
		goto fail_dL_dY;
	}

	// success
	return self;

	// failure
	fail_dL_dY:
		FREE(self);
	return NULL;
}

nn_loss_t*
nn_loss_import(nn_arch_t* arch, jsmn_val_t* val)
{
	ASSERT(arch);
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	jsmn_val_t* val_dimY    = NULL;
	jsmn_val_t* val_loss_fn = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		jsmn_keyval_t* kv;
		kv = (jsmn_keyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == JSMN_TYPE_STRING)
		{
			if(strcmp(kv->key, "loss_fn") == 0)
			{
				val_loss_fn = kv->val;
			}
		}
		else if(kv->val->type == JSMN_TYPE_OBJECT)
		{
			if(strcmp(kv->key, "dimY") == 0)
			{
				val_dimY = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_dimY    == NULL) ||
	   (val_loss_fn == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_loss_fn loss_fn;
	loss_fn = nn_loss_function(val_loss_fn->data);
	if(loss_fn == NULL)
	{
		LOGE("invalid %s", val_loss_fn->data);
		return NULL;
	}

	nn_dim_t dimY;
	if(nn_dim_load(&dimY, val_dimY) == 0)
	{
		return NULL;
	}

	return nn_loss_new(arch, &dimY, loss_fn);
}

int nn_loss_export(nn_loss_t* self, jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	nn_dim_t* dimY = nn_tensor_dim(self->dL_dY);

	const char* str_loss_fn = nn_loss_string(self->loss_fn);
	if(str_loss_fn == NULL)
	{
		LOGE("invalid");
		return 0;
	}

	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	ret &= jsmn_stream_key(stream, "%s", "loss_fn");
	ret &= jsmn_stream_string(stream, "%s", str_loss_fn);
	ret &= jsmn_stream_key(stream, "%s", "dimY");
	ret &= nn_dim_store(dimY, stream);
	ret &= jsmn_stream_end(stream);

	return ret;
}

void nn_loss_delete(nn_loss_t** _self)
{
	ASSERT(_self);

	nn_loss_t* self = *_self;
	if(self)
	{
		nn_tensor_delete(&self->dL_dY);
		FREE(self);
		*_self = self;
	}
}

nn_tensor_t*
nn_loss_loss(nn_loss_t* self, uint32_t bs,
             nn_tensor_t* Y, nn_tensor_t* Yt)
{
	ASSERT(self);
	ASSERT(Y);
	ASSERT(Yt);

	nn_loss_fn loss_fn = self->loss_fn;
	return (*loss_fn)(self, bs, Y, Yt);
}

nn_dim_t* nn_loss_dimY(nn_loss_t* self)
{
	ASSERT(self);

	return nn_tensor_dim(self->dL_dY);
}
