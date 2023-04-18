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
#include "nn_tensor.h"

/***********************************************************
* private                                                  *
***********************************************************/

static float*
nn_tensor_data(nn_tensor_t* self, uint32_t i)
{
	ASSERT(self);

	if(i >= self->dim.n)
	{
		LOGE("invalid i=%u, n=%u", i, self->dim.n);
		return NULL;
	}

	nn_dim_t* dim = &self->dim;
	return &self->data[i*dim->w*dim->h*dim->d];
}

size_t nn_tensor_stride(nn_tensor_t* self)
{
	ASSERT(self);

	nn_dim_t* dim = &self->dim;
	return dim->w*dim->h*dim->d*sizeof(float);
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

void nn_tensor_clear(nn_tensor_t* self)
{
	ASSERT(self);

	nn_dim_t* dim = &self->dim;

	uint32_t count = dim->n*dim->w*dim->h*dim->d;
	memset(self->data, 0, count*sizeof(float));
}

float nn_tensor_get(nn_tensor_t* self,
                    uint32_t i, uint32_t x,
                    uint32_t y, uint32_t z)
{
	ASSERT(self);

	uint32_t sn = self->dim.w*self->dim.h*self->dim.d;
	uint32_t sy = self->dim.w*self->dim.d;
	uint32_t sx = self->dim.d;
	return self->data[i*sn + y*sy + x*sx + z];
}

void nn_tensor_set(nn_tensor_t* self,
                   uint32_t i, uint32_t x,
                   uint32_t y, uint32_t z,
                   float val)
{
	ASSERT(self);

	uint32_t sn = self->dim.w*self->dim.h*self->dim.d;
	uint32_t sy = self->dim.w*self->dim.d;
	uint32_t sx = self->dim.d;
	self->data[i*sn + y*sy + x*sx + z] = val;
}

void nn_tensor_add(nn_tensor_t* self,
                   uint32_t i, uint32_t x,
                   uint32_t y, uint32_t z,
                   float val)
{
	ASSERT(self);

	uint32_t sn = self->dim.w*self->dim.h*self->dim.d;
	uint32_t sy = self->dim.w*self->dim.d;
	uint32_t sx = self->dim.d;
	self->data[i*sn + y*sy + x*sx + z] += val;
}

void nn_tensor_mul(nn_tensor_t* self,
                   uint32_t i, uint32_t x,
                   uint32_t y, uint32_t z,
                   float val)
{
	ASSERT(self);

	uint32_t sn = self->dim.w*self->dim.h*self->dim.d;
	uint32_t sy = self->dim.w*self->dim.d;
	uint32_t sx = self->dim.d;
	self->data[i*sn + y*sy + x*sx + z] *= val;
}

nn_dim_t* nn_tensor_dim(nn_tensor_t* self)
{
	ASSERT(self);

	return &self->dim;
}

int nn_tensor_blit(nn_tensor_t* src,
                   nn_tensor_t* dst,
                   uint32_t srci, uint32_t dsti)
{
	ASSERT(src);
	ASSERT(dst);

	float* src_data   = nn_tensor_data(src, srci);
	float* dst_data   = nn_tensor_data(dst, dsti);
	size_t src_stride = nn_tensor_stride(src);
	size_t dst_stride = nn_tensor_stride(dst);

	if((src_data == NULL) || (dst_data == NULL) ||
	   (src_stride != dst_stride))
	{
		LOGE("invalid data=%p:%p, stride=%u:%u",
		     src_data, dst_data,
		     (uint32_t) src_stride,
		     (uint32_t) dst_stride);
		return 0;
	}

	memcpy(dst_data, src_data, dst_stride);

	return 1;
}
