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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LOG_TAG "nn"
#include "../libcc/cc_log.h"
#include "../libcc/cc_memory.h"
#include "nn_tensor.h"

/***********************************************************
* private                                                  *
***********************************************************/

static float*
nn_tensor_data(nn_tensor_t* self, uint32_t n)
{
	ASSERT(self);

	if(n >= self->dim.count)
	{
		LOGE("invalid n=%u, count=%u", n, self->dim.count);
		return NULL;
	}

	nn_dim_t* dim = &self->dim;
	return &self->data[n*dim->height*dim->width*dim->depth];
}

static size_t nn_tensor_stride(nn_tensor_t* self)
{
	ASSERT(self);

	nn_dim_t* dim = &self->dim;
	return dim->height*dim->width*dim->depth*sizeof(float);
}

static int
nn_tensor_loadData(nn_tensor_t* self, jsmn_val_t* val)
{
	ASSERT(self);
	ASSERT(val);

	nn_dim_t* dim   = &self->dim;
	uint32_t  count = dim->count*dim->height*
	                  dim->width*dim->depth;

	uint32_t       i;
	cc_listIter_t* iter = cc_list_head(val->array->list);
	for(i = 0; i < count; ++i)
	{
		if(iter == NULL)
		{
			LOGE("invalid");
			return 0;
		}

		jsmn_val_t* elem;
		elem = (jsmn_val_t*) cc_list_peekIter(iter);
		if(elem->type != JSMN_TYPE_PRIMITIVE)
		{
			LOGE("invalid");
			return 0;
		}

		self->data[i] = strtof(elem->data, NULL);

		iter = cc_list_next(iter);
	}

	return 1;
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_tensor_t* nn_tensor_new(nn_dim_t* dim)
{
	ASSERT(dim);

	nn_tensor_t* self;
	self = (nn_tensor_t*)
	       CALLOC(1, sizeof(nn_tensor_t));
	if(self == NULL)
	{
		LOGE("CALLOC failed");
		return NULL;
	}

	nn_dim_copy(dim, &self->dim);

	self->data = (float*)
	             CALLOC(1, nn_dim_sizeof(dim));
	if(self->data == NULL)
	{
		LOGE("CALLOC failed");
		goto fail_data;
	}

	// success
	return self;

	// failure
	fail_data:
		FREE(self);
	return NULL;
}

void nn_tensor_delete(nn_tensor_t** _self)
{
	ASSERT(_self);

	nn_tensor_t* self = *_self;
	if(self)
	{
		FREE(self->data);
		FREE(self);
		*_self = NULL;
	}
}

void nn_tensor_print(nn_tensor_t* self, const char* name)
{
	ASSERT(self);

	jsmn_stream_t* stream = jsmn_stream_new();
	if(stream == NULL)
	{
		return;
	}

	nn_tensor_store(self, stream);

	size_t size = 0;
	const char* buffer = jsmn_stream_buffer(stream, &size);
	if(buffer)
	{
		printf("%s: %s\n", name, buffer);
	}

	jsmn_stream_delete(&stream);
}

int nn_tensor_load(nn_tensor_t* self, jsmn_val_t* val)
{
	ASSERT(self);
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid type=%i", (int) val->type);
		return 0;
	}

	jsmn_val_t* val_dim  = NULL;
	jsmn_val_t* val_data = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		jsmn_keyval_t* kv;
		kv = (jsmn_keyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == JSMN_TYPE_OBJECT)
		{
			if(strcmp(kv->key, "dim") == 0)
			{
				val_dim = kv->val;
			}
		}
		else if(kv->val->type == JSMN_TYPE_ARRAY)
		{
			if(strcmp(kv->key, "data") == 0)
			{
				val_data = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_dim  == NULL) ||
	   (val_data == NULL))
	{
		LOGE("invalid");
		return 0;
	}

	nn_dim_t dim;
	if((nn_dim_load(&dim, val_dim)         == 0) ||
	   (nn_dim_equals(&self->dim, &dim)    == 0) ||
	   (nn_tensor_loadData(self, val_data) == 0))
	{
		return 0;
	}

	return 1;
}

int nn_tensor_store(nn_tensor_t* self,
                    jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	nn_dim_t* dim   = &self->dim;
	uint32_t  count = dim->count*dim->height*
	                  dim->width*dim->depth;

	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	ret &= jsmn_stream_key(stream, "%s", "dim");
	ret &= nn_dim_store(dim, stream);
	ret &= jsmn_stream_key(stream, "%s", "data");
	ret &= jsmn_stream_beginArray(stream);

	uint32_t i;
	for(i = 0; i < count; ++i)
	{
		ret &= jsmn_stream_float(stream, self->data[i]);
	}

	ret &= jsmn_stream_end(stream);
	ret &= jsmn_stream_end(stream);

	return ret;
}

void nn_tensor_flatten(nn_tensor_t* self,
                       nn_tensor_t* flat)
{
	ASSERT(self);
	ASSERT(flat);

	flat->dim.count  = self->dim.count;
	flat->dim.height = 1;
	flat->dim.width  = 1;
	flat->dim.depth  = self->dim.height*self->dim.width*
	                   self->dim.depth;
	flat->data  = self->data;
}

void nn_tensor_clear(nn_tensor_t* self)
{
	ASSERT(self);

	nn_dim_t* dim = &self->dim;

	memset(self->data, 0, dim->count*dim->height*dim->width*
	                      dim->depth*sizeof(float));
}

float nn_tensor_get(nn_tensor_t* self,
                    uint32_t n, uint32_t i,
                    uint32_t j, uint32_t k)
{
	ASSERT(self);

	uint32_t sn = self->dim.height*self->dim.width*
	              self->dim.depth;
	uint32_t sy = self->dim.width*self->dim.depth;
	uint32_t sx = self->dim.depth;
	return self->data[n*sn + i*sy + j*sx + k];
}

void nn_tensor_set(nn_tensor_t* self,
                   uint32_t n, uint32_t i,
                   uint32_t j, uint32_t k,
                   float val)
{
	ASSERT(self);

	uint32_t sn = self->dim.height*self->dim.width*
	              self->dim.depth;
	uint32_t sy = self->dim.width*self->dim.depth;
	uint32_t sx = self->dim.depth;
	self->data[n*sn + i*sy + j*sx + k] = val;
}

void nn_tensor_add(nn_tensor_t* self,
                   uint32_t n, uint32_t i,
                   uint32_t j, uint32_t k,
                   float val)
{
	ASSERT(self);

	uint32_t sn = self->dim.height*self->dim.width*
	              self->dim.depth;
	uint32_t sy = self->dim.width*self->dim.depth;
	uint32_t sx = self->dim.depth;
	self->data[n*sn + i*sy + j*sx + k] += val;
}

void nn_tensor_mul(nn_tensor_t* self,
                   uint32_t n, uint32_t i,
                   uint32_t j, uint32_t k,
                   float val)
{
	ASSERT(self);

	uint32_t sn = self->dim.height*self->dim.width*
	              self->dim.depth;
	uint32_t sy = self->dim.width*self->dim.depth;
	uint32_t sx = self->dim.depth;
	self->data[n*sn + i*sy + j*sx + k] *= val;
}

float nn_tensor_norm(nn_tensor_t* self, uint32_t count)
{
	ASSERT(self);

	nn_dim_t* dim = nn_tensor_dim(self);

	float    xx = 0.0f;
	float    x;
	uint32_t n;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(n = 0; n < count; ++n)
	{
		for(i = 0; i < dim->height; ++i)
		{
			for(j = 0; j < dim->width; ++j)
			{
				for(k = 0; k < dim->depth; ++k)
				{
					x = nn_tensor_get(self, n, i, j, k);
					xx += x*x;
				}
			}
		}
	}
	return sqrtf(xx);
}

float nn_tensor_min(nn_tensor_t* self, uint32_t count)
{
	ASSERT(self);

	nn_dim_t* dim = nn_tensor_dim(self);

	float    min = nn_tensor_get(self, 0, 0, 0, 0);
	float    x;
	uint32_t n;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(n = 0; n < count; ++n)
	{
		for(i = 0; i < dim->height; ++i)
		{
			for(j = 0; j < dim->width; ++j)
			{
				for(k = 0; k < dim->depth; ++k)
				{
					x = nn_tensor_get(self, n, i, j, k);
					if(x < min)
					{
						min = x;
					}
				}
			}
		}
	}
	return min;
}

float nn_tensor_max(nn_tensor_t* self, uint32_t count)
{
	ASSERT(self);

	nn_dim_t* dim = nn_tensor_dim(self);

	float    max = nn_tensor_get(self, 0, 0, 0, 0);
	float    x;
	uint32_t n;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(n = 0; n < count; ++n)
	{
		for(i = 0; i < dim->height; ++i)
		{
			for(j = 0; j < dim->width; ++j)
			{
				for(k = 0; k < dim->depth; ++k)
				{
					x = nn_tensor_get(self, n, i, j, k);
					if(x > max)
					{
						max = x;
					}
				}
			}
		}
	}
	return max;
}

float nn_tensor_avg(nn_tensor_t* self, uint32_t count)
{
	ASSERT(self);

	nn_dim_t* dim = nn_tensor_dim(self);

	float    sum = 0.0f;
	uint32_t n;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(n = 0; n < count; ++n)
	{
		for(i = 0; i < dim->height; ++i)
		{
			for(j = 0; j < dim->width; ++j)
			{
				for(k = 0; k < dim->depth; ++k)
				{
					sum += nn_tensor_get(self, n, i, j, k);
				}
			}
		}
	}
	return sum/((float) (count*dim->height*dim->width));
}

nn_dim_t* nn_tensor_dim(nn_tensor_t* self)
{
	ASSERT(self);

	return &self->dim;
}

int nn_tensor_blit(nn_tensor_t* src,
                   nn_tensor_t* dst,
                   uint32_t count,
                   uint32_t src_offset,
                   uint32_t dst_offset)
{
	ASSERT(src);
	ASSERT(dst);

	float* src_data   = nn_tensor_data(src, src_offset);
	float* dst_data   = nn_tensor_data(dst, dst_offset);
	size_t src_stride = nn_tensor_stride(src);
	size_t dst_stride = nn_tensor_stride(dst);

	if((count == 0)                          ||
	   (src_stride != dst_stride)            ||
	   (src_offset + count > src->dim.count) ||
	   (dst_offset + count > dst->dim.count) ||
	   (src_data == NULL) || (dst_data == NULL))
	{
		LOGE("invalid count=%u, offset=%u:%u, data=%p:%p, stride=%u:%u",
		     count, src_offset, dst_offset,
		     src_data, dst_data,
		     (uint32_t) src_stride,
		     (uint32_t) dst_stride);
		return 0;
	}

	memcpy(dst_data, src_data, count*dst_stride);

	return 1;
}
