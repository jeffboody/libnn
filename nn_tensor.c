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
#include "../libcc/math/cc_float.h"
#include "../libcc/cc_log.h"
#include "../libcc/cc_memory.h"
#include "../texgz/texgz_png.h"
#include "nn_arch.h"
#include "nn_engine.h"
#include "nn_tensorStats.h"
#include "nn_tensor.h"

typedef union
{
	float    f32;
	uint32_t u32;
} nn_tensorValue_t;

/***********************************************************
* private                                                  *
***********************************************************/

static float*
nn_tensor_data(nn_tensor_t* self, uint32_t n)
{
	ASSERT(self);

	if(self->mode != NN_TENSOR_MODE_IO)
	{
		LOGE("invalid mode=%u", self->mode);
		return NULL;
	}

	nn_dim_t* dim = &self->dim;
	if(n >= dim->count)
	{
		LOGE("invalid n=%u, count=%u", n, dim->count);
		return NULL;
	}

	size_t stride = nn_dim_strideElements(dim);
	return &self->data[n*stride];
}

static int
nn_tensor_importData(nn_tensor_t* self, jsmn_val_t* val)
{
	ASSERT(self);
	ASSERT(val);

	nn_dim_t* dim = nn_tensor_dim(self);

	nn_tensor_t* tmp = NULL;
	if(self->mode == NN_TENSOR_MODE_COMPUTE)
	{
		tmp = nn_tensor_new(self->engine, dim,
		                    NN_TENSOR_INIT_ZERO,
		                    NN_TENSOR_MODE_IO);
		if(tmp == NULL)
		{
			return 0;
		}

		if(nn_tensor_importData(tmp, val) == 0)
		{
			goto fail_importData;
		}

		if(nn_tensor_copy(tmp, self, 0, 0, dim->count) == 0)
		{
			goto fail_copy;
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
	fail_copy:
	fail_importData:
		nn_tensor_delete(&tmp);
	return 0;
}

static void
nn_tensor_initXavierWeights(nn_tensor_t* self)
{
	ASSERT(self);

	nn_engine_t* engine = self->engine;

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
					w = cc_rngUniform_rand2F(&engine->rng_uniform,
					                         min, max);
					nn_tensor_ioSet(self, n, i, j, k, w);
				}
			}
		}
	}
}

static void
nn_tensor_initHeWeights(nn_tensor_t* self)
{
	ASSERT(self);

	nn_engine_t* engine = self->engine;

	nn_dim_t* dim = nn_tensor_dim(self);
	uint32_t  fc  = dim->count;
	uint32_t  fh  = dim->height;
	uint32_t  fw  = dim->width;
	uint32_t  xd  = dim->depth;
	uint32_t  hwd = fh*fw*xd;

	double mu    = 0.0;
	double sigma = sqrt(2.0/((double) hwd));
	cc_rngNormal_reset(&engine->rng_normal, mu, sigma);

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
					w = cc_rngNormal_rand1F(&engine->rng_normal);
					nn_tensor_ioSet(self, n, i, j, k, w);
				}
			}
		}
	}
}

static void
nn_tensor_initSN(nn_engine_t* engine, float* buf,
                 uint32_t n)
{
	ASSERT(engine);
	ASSERT(buf);

	cc_rngNormal_reset(&engine->rng_normal, 0.0, 1.0);

	float mag = 0.0f;

	// initialize buf and compute magnitude
	uint32_t i;
	for(i = 0; i < n; ++i)
	{
		buf[i]  = cc_rngNormal_rand1F(&engine->rng_normal);
		mag    += buf[i]*buf[i];
	}
	mag = sqrtf(mag);

	// normalize buf
	for(i = 0; i < n; ++i)
	{
		buf[i] /= mag;
	}
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_tensor_t*
nn_tensor_new(nn_engine_t* engine, nn_dim_t* dim,
              nn_tensorInit_e init,
              nn_tensorMode_e mode)
{
	ASSERT(engine);
	ASSERT(dim);

	nn_tensor_t* self;
	self = (nn_tensor_t*)
	       CALLOC(1, sizeof(nn_tensor_t));
	if(self == NULL)
	{
		LOGE("CALLOC failed");
		return NULL;
	}

	self->engine = engine;
	self->mode   = mode;

	nn_dim_copy(dim, &self->dim);

	vkk_updateMode_e um;
	um = vkk_compute_updateMode(engine->compute);

	if(mode == NN_TENSOR_MODE_COMPUTE)
	{
		nn_tensor_t* tmp;
		tmp = nn_tensor_new(engine, dim, init,
		                    NN_TENSOR_MODE_IO);
		if(tmp == NULL)
		{
			goto fail_data;
		}

		self->sb_dim = vkk_buffer_new(engine->engine, um,
		                              VKK_BUFFER_USAGE_STORAGE,
		                              sizeof(nn_dim_t),
		                              dim);
		if(self->sb_dim == NULL)
		{
			nn_tensor_delete(&tmp);
			goto fail_data;
		}

		self->sb_data = vkk_buffer_new(engine->engine, um,
		                               VKK_BUFFER_USAGE_STORAGE,
		                               nn_dim_sizeBytes(dim),
		                               tmp->data);
		if(self->sb_data == NULL)
		{
			vkk_buffer_delete(&self->sb_dim);
			nn_tensor_delete(&tmp);
			goto fail_data;
		}

		self->us0 = vkk_uniformSet_new(engine->engine,
		                               0, 0, NULL,
		                               engine->usf0_tensor);
		if(self->us0 == NULL)
		{
			vkk_buffer_delete(&self->sb_data);
			vkk_buffer_delete(&self->sb_dim);
			nn_tensor_delete(&tmp);
			goto fail_data;
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

		vkk_compute_updateUniformSetRefs(engine->compute,
		                                 self->us0,
		                                 2, ua0_array);

		nn_tensor_delete(&tmp);
	}
	else
	{
		self->data = (float*)
		             CALLOC(1, nn_dim_sizeBytes(dim));
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
		vkk_uniformSet_delete(&self->us1_norm);
		vkk_buffer_delete(&self->sb14_c);
		vkk_buffer_delete(&self->sb13_data_v2);
		vkk_buffer_delete(&self->sb12_data_u2);
		vkk_buffer_delete(&self->sb11_data_v1);
		vkk_buffer_delete(&self->sb10_data_u1);
		vkk_uniformSet_delete(&self->us0);
		vkk_buffer_delete(&self->sb_data);
		vkk_buffer_delete(&self->sb_dim);
		FREE(self->data);
		FREE(self);
		*_self = NULL;
	}
}

int nn_tensor_import(nn_tensor_t* self, jsmn_val_t* val)
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
	if((nn_dim_import(&dim, val_dim)         == 0) ||
	   (nn_dim_sizeEquals(&self->dim, &dim)  == 0) ||
	   (nn_tensor_importData(self, val_data) == 0))
	{
		return 0;
	}

	return 1;
}

int nn_tensor_export(nn_tensor_t* self,
                    jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	nn_dim_t* dim = nn_tensor_dim(self);

	int ret = 1;
	nn_tensor_t* tmp = NULL;
	if(self->mode == NN_TENSOR_MODE_COMPUTE)
	{
		tmp = nn_tensor_new(self->engine, dim,
		                    NN_TENSOR_INIT_ZERO,
		                    NN_TENSOR_MODE_IO);
		if(tmp == NULL)
		{
			return 0;
		}

		if(nn_tensor_copy(self, tmp, 0, 0, dim->count) == 0)
		{
			goto fail_copy;
		}

		if(nn_tensor_export(tmp, stream) == 0)
		{
			goto fail_export;
		}

		nn_tensor_delete(&tmp);
	}
	else
	{
		uint32_t count = dim->count*dim->height*
		                 dim->width*dim->depth;

		ret &= jsmn_stream_beginObject(stream);
		ret &= jsmn_stream_key(stream, "%s", "dim");
		ret &= nn_dim_export(dim, stream);
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
	fail_export:
	fail_copy:
		nn_tensor_delete(&tmp);
	return 0;
}

nn_dim_t* nn_tensor_dim(nn_tensor_t* self)
{
	ASSERT(self);

	return &self->dim;
}

int nn_tensor_copy(nn_tensor_t* src,
                   nn_tensor_t* dst,
                   uint32_t src_n,
                   uint32_t dst_n,
                   uint32_t count)
{
	ASSERT(src);
	ASSERT(dst);

	nn_dim_t* dim_src = nn_tensor_dim(src);
	nn_dim_t* dim_dst = nn_tensor_dim(dst);

	size_t src_stride = nn_dim_strideBytes(dim_src);
	size_t dst_stride = nn_dim_strideBytes(dim_dst);
	if((count == 0)                     ||
	   (src_stride != dst_stride)       ||
	   (src_n + count > src->dim.count) ||
	   (dst_n + count > dst->dim.count))
	{
		LOGE("invalid count=%u:%u:%u, n=%u:%u, stride=%u:%u",
		     count, src->dim.count, dst->dim.count,
		     src_n, dst_n,
		     (uint32_t) src_stride,
		     (uint32_t) dst_stride);
		return 0;
	}

	float* src_data = NULL;
	if(src->mode == NN_TENSOR_MODE_IO)
	{
		src_data = nn_tensor_data(src, src_n);
		if(src_data == NULL)
		{
			return 0;
		}
	}

	float* dst_data = NULL;
	if(dst->mode == NN_TENSOR_MODE_IO)
	{
		dst_data = nn_tensor_data(dst, dst_n);
		if(dst_data == NULL)
		{
			return 0;
		}
	}

	size_t size = count*src_stride;
	if((src->mode == NN_TENSOR_MODE_IO) &&
	   (dst->mode == NN_TENSOR_MODE_COMPUTE))
	{
		vkk_buffer_writeStorage(dst->sb_data, dst_n, size,
		                        src_data);
	}
	else if((src->mode == NN_TENSOR_MODE_COMPUTE) &&
	        (dst->mode == NN_TENSOR_MODE_IO))
	{
		vkk_buffer_readStorage(src->sb_data, src_n, size,
		                       dst_data);
	}
	else if((src->mode == NN_TENSOR_MODE_COMPUTE) &&
	        (dst->mode == NN_TENSOR_MODE_COMPUTE))
	{
		vkk_buffer_copyStorage(src->sb_data, dst->sb_data,
		                       src_n, dst_n, size);
	}
	else
	{
		memcpy(dst_data, src_data, size);
	}

	return 1;
}

int nn_tensor_ioClear(nn_tensor_t* self,
                      uint32_t n,
                      uint32_t count)
{
	ASSERT(self);

	nn_dim_t* dim = nn_tensor_dim(self);
	if((count + n) > dim->count)
	{
		LOGE("invalid n=%u, count=%u:%u",
		     n, count, dim->count);
		return 0;
	}

	float* data = nn_tensor_data(self, n);
	if(data == NULL)
	{
		return 0;
	}

	memset(data, 0, count*nn_dim_strideBytes(dim));

	return 1;
}

int nn_tensor_ioCopy(nn_tensor_t* src,
                     nn_tensor_t* dst,
                     uint32_t src_n,
                     uint32_t dst_n,
                     uint32_t count)
{
	ASSERT(src);
	ASSERT(dst);

	nn_dim_t* dim_src = nn_tensor_dim(src);
	nn_dim_t* dim_dst = nn_tensor_dim(dst);
	if(nn_dim_strideEquals(dim_src, dim_dst) == 0)
	{
		LOGE("invalid height=%u:%u, width=%u:%u, depth=%u:%u",
		     dim_src->height, dim_dst->height,
		     dim_src->width,  dim_dst->width,
		     dim_src->depth,  dim_dst->depth);
		return 0;
	}

	if(((count + src_n) > dim_src->count) ||
	   ((count + dst_n) > dim_dst->count))
	{
		LOGE("invalid n=%u:%u, count=%u:%u:%u",
		     src_n, dst_n,
		     count, dim_src->count, dim_dst->count);
		return 0;
	}

	float* src_data = nn_tensor_data(src, src_n);
	float* dst_data = nn_tensor_data(dst, dst_n);
	if((dst_data == NULL) || (dst_data == NULL))
	{
		return 0;
	}

	size_t bytes = nn_dim_strideBytes(dim_src);
	memcpy(dst_data, src_data, count*bytes);

	return 1;
}

float nn_tensor_ioGet(nn_tensor_t* self,
                      uint32_t n, uint32_t i,
                      uint32_t j, uint32_t k)
{
	ASSERT(self);
	ASSERT(self->mode & NN_TENSOR_MODE_IO);
	ASSERT(nn_dim_valid(self, n, i, j, k));

	uint32_t sn = self->dim.height*self->dim.width*
	              self->dim.depth;
	uint32_t sy = self->dim.width*self->dim.depth;
	uint32_t sx = self->dim.depth;
	return self->data[n*sn + i*sy + j*sx + k];
}

void nn_tensor_ioSet(nn_tensor_t* self,
                     uint32_t n, uint32_t i,
                     uint32_t j, uint32_t k,
                     float v)
{
	ASSERT(self);
	ASSERT(self->mode & NN_TENSOR_MODE_IO);
	ASSERT(nn_dim_valid(self, n, i, j, k));

	uint32_t sn = self->dim.height*self->dim.width*
	              self->dim.depth;
	uint32_t sy = self->dim.width*self->dim.depth;
	uint32_t sx = self->dim.depth;
	self->data[n*sn + i*sy + j*sx + k] = v;
}

int
nn_tensor_ioExportPng(nn_tensor_t* self, const char* fname,
                      uint32_t n, uint32_t k, uint32_t depth,
                      float min, float max)
{
	ASSERT(self);
	ASSERT(fname);

	nn_dim_t* dim = nn_tensor_dim(self);
	uint32_t  h   = dim->height;
	uint32_t  w   = dim->width;

	if(self->mode != NN_TENSOR_MODE_IO)
	{
		LOGE("invalid");
		return 0;
	}

	if((n >= dim->count) || (depth > 4) ||
	   ((k + depth) > dim->depth))
	{
		LOGE("invalid n=%u, k=%u, depth=%u, dim=%u,%u",
		     n, k, depth, dim->count, dim->depth);
		return 0;
	}

	int format = (depth == 1) ? TEXGZ_LUMINANCE : TEXGZ_RGBA;

	texgz_tex_t* tex;
	tex = texgz_tex_new(w, h, w, h, TEXGZ_UNSIGNED_BYTE,
	                    format, NULL);
	if(tex == NULL)
	{
		return 0;
	}

	float    t;
	uint32_t i;
	uint32_t j;
	uint32_t kd;
	unsigned char pixel[4] =
	{
		0x00, 0x00, 0x00, 0xFF,
	};
	for(i = 0; i < h; ++i)
	{
		for(j = 0; j < w; ++j)
		{
			for(kd = k; kd < k + depth; ++kd)
			{
				t = nn_tensor_ioGet(self, n, i, j, kd);
				pixel[kd - k] = (unsigned char)
				                cc_clamp(255.0f*(t - min)/(max - min),
				                         0.0f, 255.0f);
			}
			texgz_tex_setPixel(tex, j, i, pixel);
		}
	}

	if(texgz_png_export(tex, fname) == 0)
	{
		goto fail_export;
	}
	texgz_tex_delete(&tex);

	// success
	return 1;

	// failure
	fail_export:
		texgz_tex_delete(&tex);
	return 0;
}

int nn_tensor_computeFill(nn_tensor_t* self,
                          vkk_hazard_e hazard,
                          uint32_t n,
                          uint32_t count,
                          float value)
{
	ASSERT(self);

	nn_engine_t* engine = self->engine;

	if(self->mode != NN_TENSOR_MODE_COMPUTE)
	{
		LOGE("invalid mode=%i", self->mode);
		return 0;
	}

	if(vkk_compute_active(engine->compute) == 0)
	{
		LOGE("invalid");
		return 0;
	}

	nn_dim_t* dim = nn_tensor_dim(self);
	if((count + n) > dim->count)
	{
		LOGE("invalid count=%u:%u, n=%u",
		     count, dim->count, n);
		return 0;
	}

	nn_tensorValue_t data =
	{
		.f32 = value,
	};

	size_t bytes = nn_dim_strideBytes(dim);
	vkk_compute_fillStorage(engine->compute, hazard,
	                        self->sb_data, n*bytes,
	                        count*bytes, data.u32);

	return 1;
}

int nn_tensor_computeCopy(nn_tensor_t* src,
                          nn_tensor_t* dst,
                          vkk_hazard_e hazard,
                          uint32_t src_n,
                          uint32_t dst_n,
                          uint32_t count)
{
	ASSERT(src);
	ASSERT(dst);

	nn_engine_t* engine = src->engine;

	if((src->mode != NN_TENSOR_MODE_COMPUTE) ||
	   (dst->mode != NN_TENSOR_MODE_COMPUTE))
	{
		LOGE("invalid mode=%i:%i",
		     src->mode, dst->mode);
		return 0;
	}

	if(vkk_compute_active(engine->compute) == 0)
	{
		LOGE("invalid");
		return 0;
	}

	nn_dim_t* dim_src = nn_tensor_dim(src);
	nn_dim_t* dim_dst = nn_tensor_dim(dst);

	size_t src_stride = nn_dim_strideBytes(dim_src);
	size_t dst_stride = nn_dim_strideBytes(dim_dst);
	if((count == 0)                     ||
	   (src_stride != dst_stride)       ||
	   (src_n + count > src->dim.count) ||
	   (dst_n + count > dst->dim.count))
	{
		LOGE("invalid count=%u:%u:%u, n=%u:%u, stride=%u:%u",
		     count, src->dim.count, dst->dim.count,
		     src_n, dst_n,
		     (uint32_t) src_stride,
		     (uint32_t) dst_stride);
		return 0;
	}

	size_t bytes = nn_dim_strideBytes(dim_src);
	vkk_compute_copyStorage(engine->compute, hazard,
	                        src->sb_data, dst->sb_data,
	                        src_n*bytes,
	                        dst_n*bytes,
	                        count*bytes);

	return 1;
}

int nn_tensor_computeNormalize(nn_tensor_t* self,
                               vkk_hazard_e hazard,
                               nn_tensorNormMode_e norm,
                               float c)
{
	ASSERT(self);

	nn_engine_t* engine = self->engine;

	nn_dim_t* dim = &self->dim;

	if(self->mode != NN_TENSOR_MODE_COMPUTE)
	{
		LOGE("invalid");
		return 0;
	}

	if(vkk_compute_active(engine->compute) == 0)
	{
		LOGE("invalid");
		return 0;
	}

	// create sb10_data_u1 on demand
	vkk_updateMode_e um;
	um = vkk_compute_updateMode(engine->compute);
	if(self->sb10_data_u1 == NULL)
	{
		// dim(fc)
		uint32_t n = dim->count;

		float* buf = CALLOC(n, sizeof(float));
		if(buf == NULL)
		{
			LOGE("CALLOC failed");
			return 0;
		}
		nn_tensor_initSN(engine, buf, n);

		self->sb10_data_u1 = vkk_buffer_new(engine->engine, um,
		                                    VKK_BUFFER_USAGE_STORAGE,
		                                    n*sizeof(float),
		                                    buf);
		if(self->sb10_data_u1 == NULL)
		{
			FREE(buf);
			return 0;
		}

		FREE(buf);
	}

	// create sb11_data_v1 on demand
	if(self->sb11_data_v1 == NULL)
	{
		// dim(fh*fw*xd)
		uint32_t n = dim->height*dim->width*dim->depth;

		self->sb11_data_v1 = vkk_buffer_new(engine->engine, um,
		                                    VKK_BUFFER_USAGE_STORAGE,
		                                    n*sizeof(float),
		                                    NULL);
		if(self->sb11_data_v1 == NULL)
		{
			return 0;
		}
	}

	// create sb12_data_u2 and sb13_data_v2 on demand
	if(norm == NN_TENSOR_NORM_MODE_BSSN)
	{
		if(self->sb12_data_u2 == NULL)
		{
			// dim(xd)
			uint32_t n = dim->depth;

			float* buf = CALLOC(n, sizeof(float));
			if(buf == NULL)
			{
				LOGE("CALLOC failed");
				return 0;
			}
			nn_tensor_initSN(engine, buf, n);

			self->sb12_data_u2 = vkk_buffer_new(engine->engine, um,
			                                    VKK_BUFFER_USAGE_STORAGE,
			                                    n*sizeof(float),
			                                    buf);
			if(self->sb12_data_u2 == NULL)
			{
				FREE(buf);
				return 0;
			}

			FREE(buf);
		}

		if(self->sb13_data_v2 == NULL)
		{
			// dim(fc*fh*fw)
			uint32_t n = dim->count*dim->height*dim->width;

			self->sb13_data_v2 = vkk_buffer_new(engine->engine, um,
			                                    VKK_BUFFER_USAGE_STORAGE,
			                                    n*sizeof(float),
			                                    NULL);
			if(self->sb13_data_v2 == NULL)
			{
				return 0;
			}
		}
	}

	// create sb14_c on demand
	// c is only used by BSSN but is still bound for SN
	if(self->sb14_c == NULL)
	{
		self->sb14_c = vkk_buffer_new(engine->engine, um,
		                              VKK_BUFFER_USAGE_STORAGE,
		                              sizeof(float),
		                              &c);
		if(self->sb14_c == NULL)
		{
			return 0;
		}
	}
	else
	{
		vkk_buffer_writeStorage(self->sb14_c, 0, sizeof(float),
		                        &c);
	}

	// create us1_norm on demand
	if(self->us1_norm == NULL)
	{
		self->us1_norm = vkk_uniformSet_new(engine->engine,
		                                    1, 0, NULL,
		                                    engine->usf1_tensor_norm);
		if(self->us1_norm == NULL)
		{
			return 0;
		}
	}

	// sb10: u1
	// sb11: v1
	// sb12: u2 (optional)
	// sb13: v2 (optional)
	// sb14: c
	vkk_uniformAttachment_t ua1_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->sb10_data_u1,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->sb11_data_v1,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->sb12_data_u2 ?
			           self->sb12_data_u2 : self->sb10_data_u1,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->sb13_data_v2 ?
			           self->sb13_data_v2 : self->sb11_data_v1,
		},
		{
			.binding = 4,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->sb14_c,
		},
	};

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1_norm,
	};

	// dispatch(hazard, 1, 1, 1, 64, 1, 1)
	vkk_computePipeline_t* cp = engine->cp_tensor_sn;
	if(norm == NN_TENSOR_NORM_MODE_BSSN)
	{
		cp = engine->cp_tensor_bssn;
	}

	if(nn_engine_bind(engine, cp) == 0)
	{
		return 0;
	}
	vkk_compute_updateUniformSetRefs(engine->compute,
	                                 self->us1_norm,
	                                 5, ua1_array);
	vkk_compute_bindUniformSets(engine->compute, 2,
	                            us_array);
	nn_engine_dispatch(engine, hazard,
	                   1, 1, 1, 64, 1, 1);

	return 1;
}

int nn_tensor_computeStats(nn_tensor_t* self,
                           vkk_hazard_e hazard,
                           uint32_t count,
                           nn_tensorStats_t* stats)
{
	ASSERT(self);
	ASSERT(stats);

	nn_engine_t* engine = self->engine;

	if(self->mode != NN_TENSOR_MODE_COMPUTE)
	{
		LOGE("invalid");
		return 0;
	}

	if(vkk_compute_active(engine->compute) == 0)
	{
		LOGE("invalid");
		return 0;
	}

	nn_dim_t* dim = nn_tensor_dim(self);
	if((count == 0) || (count > dim->count))
	{
		LOGE("invalid count=%u:%u", count, dim->count);
		return 0;
	}

	stats->data.count = count;
	vkk_buffer_writeStorage(stats->sb10_stats, 0,
	                        sizeof(nn_tensorStatsData_t),
	                        &stats->data);

	// sb10: stats
	vkk_uniformAttachment_t ua1_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = stats->sb10_stats,
		},
	};

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		stats->us1,
	};

	// dispatch(hazard, 1, 1, 1, 8, 8, 1)
	vkk_computePipeline_t* cp = engine->cp_tensor_stats;
	if(nn_engine_bind(engine, cp) == 0)
	{
		return 0;
	}
	vkk_compute_updateUniformSetRefs(engine->compute,
	                                 stats->us1,
	                                 1, ua1_array);
	vkk_compute_bindUniformSets(engine->compute, 2,
	                            us_array);
	nn_engine_dispatch(engine, hazard,
	                   1, 1, 1, 8, 8, 1);

	stats->dirty = 1;

	return 1;
}
