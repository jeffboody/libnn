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
#include "nn_arch.h"
#include "nn_tensor.h"

/***********************************************************
* private                                                  *
***********************************************************/

// protected
extern void
nn_arch_dispatch(nn_arch_t* self,
                 vkk_hazzard_e hazzard,
                 uint32_t count_x,
                 uint32_t count_y,
                 uint32_t count_z,
                 uint32_t local_size_x,
                 uint32_t local_size_y,
                 uint32_t local_size_z);
extern int
nn_arch_bind(nn_arch_t* self,
             vkk_computePipeline_t* cp);

static int
nn_tensor_isModeIO(nn_tensor_t* self)
{
	ASSERT(self);

	// ignore mode when compute is disabled
	if(self->mode == NN_TENSOR_MODE_COMPUTE)
	{
		return 0;
	}

	return 1;
}

static float*
nn_tensor_data(nn_tensor_t* self, uint32_t n)
{
	ASSERT(self);

	if(nn_tensor_isModeIO(self) == 0)
	{
		return NULL;
	}

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

	nn_dim_t* dim = nn_tensor_dim(self);

	nn_tensor_t* tmp = NULL;
	if(self->mode == NN_TENSOR_MODE_COMPUTE)
	{
		tmp = nn_tensor_new(self->arch, dim,
		                    NN_TENSOR_INIT_ZERO,
		                    NN_TENSOR_MODE_IO);
		if(tmp == NULL)
		{
			return 0;
		}

		if(nn_tensor_loadData(tmp, val) == 0)
		{
			goto fail_loadData;
		}

		if(nn_tensor_blit(tmp, self, dim->count, 0, 0) == 0)
		{
			goto fail_blit;
		}

		nn_tensor_delete(&tmp);
	}
	else
	{
		uint32_t count = dim->count*dim->height*
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
	}

	// success
	return 1;

	// failure
	fail_blit:
	fail_loadData:
		nn_tensor_delete(&tmp);
	return 0;
}

static void
nn_tensor_initXavierWeights(nn_tensor_t* self)
{
	ASSERT(self);

	nn_arch_t* arch = self->arch;

	nn_dim_t* dim = nn_tensor_dim(self);
	uint32_t  fc  = dim->count;
	uint32_t  fh  = dim->height;
	uint32_t  fw  = dim->width;
	uint32_t  xd  = dim->depth;
	uint32_t  hwd = fh*fw*xd;
	float     min = -1.0/sqrt((double) hwd);
	float     max = 1.0/sqrt((double) hwd);

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
					nn_tensor_set(self, n, i, j, k, w);
				}
			}
		}
	}
}

static void
nn_tensor_initHeWeights(nn_tensor_t* self)
{
	ASSERT(self);

	nn_arch_t* arch = self->arch;

	nn_dim_t* dim = nn_tensor_dim(self);
	uint32_t  fc  = dim->count;
	uint32_t  fh  = dim->height;
	uint32_t  fw  = dim->width;
	uint32_t  xd  = dim->depth;
	uint32_t  hwd = fh*fw*xd;

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
					nn_tensor_set(self, n, i, j, k, w);
				}
			}
		}
	}
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_tensor_t*
nn_tensor_new(nn_arch_t* arch, nn_dim_t* dim,
              nn_tensorInit_e init,
              nn_tensorMode_e mode)
{
	ASSERT(arch);
	ASSERT(dim);

	nn_tensor_t* self;
	self = (nn_tensor_t*)
	       CALLOC(1, sizeof(nn_tensor_t));
	if(self == NULL)
	{
		LOGE("CALLOC failed");
		return NULL;
	}

	self->arch = arch;
	self->mode = mode;

	nn_dim_copy(dim, &self->dim);

	vkk_updateMode_e um;
	um = vkk_compute_updateMode(arch->compute);

	if(mode == NN_TENSOR_MODE_COMPUTE)
	{
		nn_tensor_t* tmp;
		tmp = nn_tensor_new(arch, dim, init, NN_TENSOR_MODE_IO);
		if(tmp == NULL)
		{
			goto fail_data;
		}

		self->us0_clear = vkk_uniformSet_new(arch->engine,
		                                     0, 0, NULL,
		                                     arch->usf0_tensor);
		if(self->us0_clear == NULL)
		{
			nn_tensor_delete(&tmp);
			goto fail_data;
		}

		self->sb_dim = vkk_buffer_new(arch->engine, um,
		                              VKK_BUFFER_USAGE_STORAGE,
		                              sizeof(nn_dim_t),
		                              dim);
		if(self->sb_dim == NULL)
		{
			vkk_uniformSet_delete(&self->us0_clear);
			nn_tensor_delete(&tmp);
			goto fail_data;
		}

		self->sb_data = vkk_buffer_new(arch->engine, um,
		                               VKK_BUFFER_USAGE_STORAGE,
		                               nn_dim_sizeof(dim),
		                               tmp->data);
		if(self->sb_data == NULL)
		{
			vkk_buffer_delete(&self->sb_dim);
			vkk_uniformSet_delete(&self->us0_clear);
			nn_tensor_delete(&tmp);
			goto fail_data;
		}

		nn_tensor_delete(&tmp);
	}
	else
	{
		self->data = (float*)
		             CALLOC(1, nn_dim_sizeof(dim));
		if(self->data == NULL)
		{
			LOGE("CALLOC failed");
			goto fail_data;
		}

		if(init == NN_TENSOR_INIT_XAVIER)
		{
			nn_tensor_initXavierWeights(self);
		}
		else if(init == NN_TENSOR_INIT_HE)
		{
			nn_tensor_initHeWeights(self);
		}
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
		vkk_buffer_delete(&self->sb_data);
		vkk_buffer_delete(&self->sb_dim);
		vkk_uniformSet_delete(&self->us0_clear);
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

	nn_dim_t* dim = nn_tensor_dim(self);

	int ret = 1;
	nn_tensor_t* tmp = NULL;
	if(self->mode == NN_TENSOR_MODE_COMPUTE)
	{
		tmp = nn_tensor_new(self->arch, dim,
		                    NN_TENSOR_INIT_ZERO,
		                    NN_TENSOR_MODE_IO);
		if(tmp == NULL)
		{
			return 0;
		}

		if(nn_tensor_blit(self, tmp, dim->count, 0, 0) == 0)
		{
			goto fail_blit;
		}

		if(nn_tensor_store(tmp, stream) == 0)
		{
			goto fail_store;
		}

		nn_tensor_delete(&tmp);
	}
	else
	{
		uint32_t count = dim->count*dim->height*
		                 dim->width*dim->depth;

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
	}

	// success
	return ret;

	// failure
	fail_store:
	fail_blit:
		nn_tensor_delete(&tmp);
	return 0;
}

void nn_tensor_clear(nn_tensor_t* self, int hazzard)
{
	ASSERT(self);

	nn_dim_t* dim = &self->dim;

	uint32_t count;
	count = dim->count*dim->height*
	        dim->width*dim->depth;

	if(nn_tensor_isModeIO(self) == 0)
	{
		nn_arch_t* arch = self->arch;

		// compute tensors may only be cleared while computing
		if(arch->computing == 0)
		{
			LOGE("invalid");
			return;
		}

		// sb00: dimX
		// sb01: X
		vkk_uniformAttachment_t ua0_array[] =
		{
			{
				.binding = 0,
				.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
				.buffer  = self->sb_dim,
			},
			{
				.binding = 1,
				.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
				.buffer  = self->sb_data,
			},
		};

		// dispatch(NONE, xn*xh*xw*xd, 1, 1, 64, 1, 1)
		vkk_computePipeline_t* cp;
		if(count%64 == 0)
		{
			cp = arch->cp_tensor_clearAligned;
		}
		else
		{
			cp = arch->cp_tensor_clear;
		}
		nn_arch_bind(arch, cp);
		vkk_compute_updateUniformSetRefs(arch->compute,
		                                 self->us0_clear,
		                                 2, ua0_array);
		vkk_compute_bindUniformSets(arch->compute, 1,
		                            &self->us0_clear);
		nn_arch_dispatch(arch,
		                 (vkk_hazzard_e) hazzard,
		                 count, 1, 1, 64, 1, 1);
	}
	else
	{
		memset(self->data, 0, count*sizeof(float));
	}
}

float nn_tensor_get(nn_tensor_t* self,
                    uint32_t n, uint32_t i,
                    uint32_t j, uint32_t k)
{
	ASSERT(self);

	if(nn_tensor_isModeIO(self) == 0)
	{
		LOGE("invalid mode=%i", (int) self->mode);
		return 0.0f;
	}

	uint32_t sn = self->dim.height*self->dim.width*
	              self->dim.depth;
	uint32_t sy = self->dim.width*self->dim.depth;
	uint32_t sx = self->dim.depth;
	return self->data[n*sn + i*sy + j*sx + k];
}

void nn_tensor_set(nn_tensor_t* self,
                   uint32_t n, uint32_t i,
                   uint32_t j, uint32_t k,
                   float v)
{
	ASSERT(self);

	if(nn_tensor_isModeIO(self) == 0)
	{
		LOGE("invalid mode=%i", (int) self->mode);
		return;
	}

	uint32_t sn = self->dim.height*self->dim.width*
	              self->dim.depth;
	uint32_t sy = self->dim.width*self->dim.depth;
	uint32_t sx = self->dim.depth;
	self->data[n*sn + i*sy + j*sx + k] = v;
}

void nn_tensor_add(nn_tensor_t* self,
                   uint32_t n, uint32_t i,
                   uint32_t j, uint32_t k,
                   float v)
{
	ASSERT(self);

	if(nn_tensor_isModeIO(self) == 0)
	{
		LOGE("invalid mode=%i", (int) self->mode);
		return;
	}

	uint32_t sn = self->dim.height*self->dim.width*
	              self->dim.depth;
	uint32_t sy = self->dim.width*self->dim.depth;
	uint32_t sx = self->dim.depth;
	self->data[n*sn + i*sy + j*sx + k] += v;
}

void nn_tensor_mul(nn_tensor_t* self,
                   uint32_t n, uint32_t i,
                   uint32_t j, uint32_t k,
                   float v)
{
	ASSERT(self);

	if(nn_tensor_isModeIO(self) == 0)
	{
		LOGE("invalid mode=%i", (int) self->mode);
		return;
	}

	uint32_t sn = self->dim.height*self->dim.width*
	              self->dim.depth;
	uint32_t sy = self->dim.width*self->dim.depth;
	uint32_t sx = self->dim.depth;
	self->data[n*sn + i*sy + j*sx + k] *= v;
}

float nn_tensor_getv(nn_tensor_t* self, uint32_t n)
{
	ASSERT(self);

	if(nn_tensor_isModeIO(self) == 0)
	{
		LOGE("invalid mode=%i", (int) self->mode);
		return 0.0f;
	}

	return self->data[n];
}

void
nn_tensor_setv(nn_tensor_t* self, uint32_t n, float v)
{
	ASSERT(self);

	if(nn_tensor_isModeIO(self) == 0)
	{
		LOGE("invalid mode=%i", (int) self->mode);
		return;
	}

	self->data[n] = v;
}

void
nn_tensor_addv(nn_tensor_t* self, uint32_t n, float v)
{
	ASSERT(self);

	if(nn_tensor_isModeIO(self) == 0)
	{
		LOGE("invalid mode=%i", (int) self->mode);
		return;
	}

	self->data[n] += v;
}

void
nn_tensor_mulv(nn_tensor_t* self, uint32_t n, float v)
{
	ASSERT(self);

	if(nn_tensor_isModeIO(self) == 0)
	{
		LOGE("invalid mode=%i", (int) self->mode);
		return;
	}

	self->data[n] *= v;
}

float nn_tensor_norm(nn_tensor_t* self, uint32_t count)
{
	ASSERT(self);

	nn_dim_t* dim = nn_tensor_dim(self);

	if(nn_tensor_isModeIO(self) == 0)
	{
		LOGE("invalid mode=%i", (int) self->mode);
		return 0.0f;
	}

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

	if(nn_tensor_isModeIO(self) == 0)
	{
		LOGE("invalid mode=%i", (int) self->mode);
		return 0.0f;
	}

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

	if(nn_tensor_isModeIO(self) == 0)
	{
		LOGE("invalid mode=%i", (int) self->mode);
		return 0.0f;
	}

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

	if(nn_tensor_isModeIO(self) == 0)
	{
		LOGE("invalid mode=%i", (int) self->mode);
		return 0.0f;
	}

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

	size_t src_stride = nn_tensor_stride(src);
	size_t dst_stride = nn_tensor_stride(dst);
	size_t size       = count*src_stride;
	if((count == 0)                          ||
	   (src->arch  != dst->arch)             ||
	   (src_stride != dst_stride)            ||
	   (src_offset + count > src->dim.count) ||
	   (dst_offset + count > dst->dim.count))
	{
		LOGE("invalid count=%u, arch=%p:%p, "
		     "offset=%u:%u, stride=%u:%u",
		     count, src->arch, dst->arch,
		     src_offset, dst_offset,
		     (uint32_t) src_stride,
		     (uint32_t) dst_stride);
		return 0;
	}

	float* src_data = nn_tensor_data(src, src_offset);
	float* dst_data = nn_tensor_data(dst, dst_offset);

	vkk_compute_t* compute = src->arch->compute;
	if((src->mode == NN_TENSOR_MODE_IO) &&
	   (dst->mode == NN_TENSOR_MODE_COMPUTE))
	{
		vkk_compute_writeBuffer(compute, dst->sb_data, size,
		                        dst_offset, src_data);
	}
	else if((src->mode == NN_TENSOR_MODE_COMPUTE) &&
	        (dst->mode == NN_TENSOR_MODE_IO))
	{
		vkk_compute_readBuffer(compute, src->sb_data, size,
		                       src_offset, dst_data);
	}
	else if((src->mode == NN_TENSOR_MODE_COMPUTE) &&
	        (dst->mode == NN_TENSOR_MODE_COMPUTE))
	{
		LOGE("unsupported");
		return 0;
	}
	else
	{
		memcpy(dst_data, src_data, size);
	}

	return 1;
}
