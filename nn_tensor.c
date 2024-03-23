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

const char* NN_TENSOR_NORM_STRING_NONE = "none";
const char* NN_TENSOR_NORM_STRING_SN   = "sn";
const char* NN_TENSOR_NORM_STRING_BSSN = "bssn";

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
nn_tensor_importStorage(nn_tensor_t* self, jsmn_val_t* val,
                        vkk_buffer_t* buf)
{
	ASSERT(self);
	ASSERT(val);
	ASSERT(buf);

	if(val->type != JSMN_TYPE_ARRAY)
	{
		LOGE("invalid type=%i", val->type);
		return 0;
	}

	size_t   size  = vkk_buffer_size(buf);
	uint32_t count = (uint32_t) (size/sizeof(float));
	float*   tmp   = (float*) CALLOC(1, size);
	if(tmp == NULL)
	{
		LOGE("CALLOC failed");
		return 0;
	}

	// fill tmp
	uint32_t       i;
	cc_listIter_t* iter  = cc_list_head(val->array->list);
	for(i = 0; i < count; ++i)
	{
		if(iter == NULL)
		{
			LOGE("invalid");
			goto fail_array;
		}

		jsmn_val_t* elem;
		elem = (jsmn_val_t*) cc_list_peekIter(iter);
		if(elem->type != JSMN_TYPE_PRIMITIVE)
		{
			LOGE("invalid");
			goto fail_array;
		}

		tmp[i] = strtof(elem->data, NULL);

		iter = cc_list_next(iter);
	}

	if(vkk_buffer_writeStorage(buf, 0, size, tmp) == 0)
	{
		goto fail_write;
	}

	FREE(tmp);

	// success
	return 1;

	// failure
	fail_write:
	fail_array:
		FREE(tmp);
	return 0;
}

static int
nn_tensor_exportStorage(nn_tensor_t* self,
                        jsmn_stream_t* stream,
                        const char* name,
                        vkk_buffer_t* buf)
{
	ASSERT(self);
	ASSERT(stream);
	ASSERT(name);
	ASSERT(buf);

	size_t   size  = vkk_buffer_size(buf);
	uint32_t count = (uint32_t) (size/sizeof(float));
	float*   tmp   = (float*) CALLOC(1, size);
	if(tmp == NULL)
	{
		LOGE("CALLOC failed");
		return 0;
	}

	if(vkk_buffer_readStorage(buf, 0, size, tmp) == 0)
	{
		goto fail_read;
	}

	int ret = 0;
	ret &= jsmn_stream_key(stream, "%s", name);
	ret &= jsmn_stream_beginArray(stream);

	uint32_t i;
	for(i = 0; i < count; ++i)
	{
		ret &= jsmn_stream_float(stream, tmp[i]);
	}
	ret &= jsmn_stream_end(stream);

	FREE(tmp);

	// success
	return ret;

	// failure
	fail_read:
		FREE(tmp);
	return 0;
}

static int
nn_tensor_importData(nn_tensor_t* self, jsmn_val_t* val)
{
	ASSERT(self);
	ASSERT(val);

	if(val->type != JSMN_TYPE_ARRAY)
	{
		LOGE("invalid type=%i", val->type);
		return 0;
	}

	if(self->mode == NN_TENSOR_MODE_COMPUTE)
	{
		return nn_tensor_importStorage(self, val,
		                               self->sb_data);
	}

	// fill data
	uint32_t       i;
	nn_dim_t*      dim   = nn_tensor_dim(self);
	uint32_t       count = nn_dim_sizeElements(dim);
	cc_listIter_t* iter  = cc_list_head(val->array->list);
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

static int
nn_tensor_initNormMode(nn_tensor_t* self,
                       nn_tensorNorm_e norm,
                       float c, int init_sn)
{
	ASSERT(self);

	nn_engine_t* engine = self->engine;

	nn_dim_t* dim = &self->dim;
	uint32_t  fc  = dim->count;
	uint32_t  fh  = dim->height;
	uint32_t  fw  = dim->width;
	uint32_t  xd  = dim->depth;

	// check the norm mode
	if(norm == NN_TENSOR_NORM_NONE)
	{
		LOGE("invalid");
		return 0;
	}
	else if(norm == self->norm)
	{
		vkk_buffer_writeStorage(self->sb14_c, 0,
		                        sizeof(float), &c);
		return 1;
	}

	// reset norm state
	self->norm = NN_TENSOR_NORM_NONE;
	vkk_buffer_delete(&self->sb10_data_u1);
	vkk_buffer_delete(&self->sb11_data_v1);
	vkk_buffer_delete(&self->sb12_data_u2);
	vkk_buffer_delete(&self->sb13_data_v2);
	vkk_buffer_delete(&self->sb14_c);
	vkk_uniformSet_delete(&self->us1_norm);

	float* tmp_u1 = NULL;
	if(init_sn)
	{
		tmp_u1 = CALLOC(fc, sizeof(float));
		if(tmp_u1 == NULL)
		{
			LOGE("CALLOC failed");
			return 0;
		}
		nn_tensor_initSN(engine, tmp_u1, fc);
	}

	vkk_updateMode_e um;
	um = vkk_compute_updateMode(engine->compute);
	self->sb10_data_u1 = vkk_buffer_new(engine->engine, um,
	                                    VKK_BUFFER_USAGE_STORAGE,
	                                    fc*sizeof(float),
	                                    tmp_u1);
	if(self->sb10_data_u1 == NULL)
	{
		goto fail_sb10_data_u1;
	}

	self->sb11_data_v1 = vkk_buffer_new(engine->engine, um,
	                                    VKK_BUFFER_USAGE_STORAGE,
	                                    fh*fw*xd*sizeof(float),
	                                    NULL);
	if(self->sb11_data_v1 == NULL)
	{
		goto fail_sb11_data_v1;
	}

	float* tmp_u2 = NULL;
	if(norm == NN_TENSOR_NORM_BSSN)
	{
		if(init_sn)
		{
			tmp_u2 = CALLOC(xd, sizeof(float));
			if(tmp_u2 == NULL)
			{
				goto fail_tmp_u2;
			}
			nn_tensor_initSN(engine, tmp_u2, xd);
		}

		self->sb12_data_u2 = vkk_buffer_new(engine->engine, um,
		                                    VKK_BUFFER_USAGE_STORAGE,
		                                    xd*sizeof(float),
		                                    tmp_u2);
		if(self->sb12_data_u2 == NULL)
		{
			goto fail_sb12_data_u2;
		}

		self->sb13_data_v2 = vkk_buffer_new(engine->engine, um,
		                                    VKK_BUFFER_USAGE_STORAGE,
		                                    fc*fw*fh*sizeof(float),
		                                    NULL);
		if(self->sb13_data_v2 == NULL)
		{
			goto fail_sb13_data_v2;
		}
	}

	// c is only used by BSSN but is still bound for SN
	self->sb14_c = vkk_buffer_new(engine->engine, um,
	                              VKK_BUFFER_USAGE_STORAGE,
	                              sizeof(float),
	                              &c);
	if(self->sb14_c == NULL)
	{
		goto fail_sb14_c;
	}

	self->us1_norm = vkk_uniformSet_new(engine->engine,
	                                    1, 0, NULL,
	                                    engine->usf1_tensor_norm);
	if(self->us1_norm == NULL)
	{
		goto fail_us1_norm;
	}

	// free tmp buffers
	FREE(tmp_u2);
	FREE(tmp_u1);

	self->norm = norm;

	// success
	return 1;

	// failure
	fail_us1_norm:
		vkk_buffer_delete(&self->sb14_c);
	fail_sb14_c:
		vkk_buffer_delete(&self->sb13_data_v2);
	fail_sb13_data_v2:
		vkk_buffer_delete(&self->sb12_data_u2);
	fail_sb12_data_u2:
		FREE(tmp_u2);
	fail_tmp_u2:
		vkk_buffer_delete(&self->sb11_data_v1);
	fail_sb11_data_v1:
		vkk_buffer_delete(&self->sb10_data_u1);
	fail_sb10_data_u1:
		FREE(tmp_u1);
	return 0;
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_tensorOpKUs0Data_t*
nn_tensorOpKUs0Data_new(nn_tensor_t* X1,
                        nn_tensor_t* X2,
                        nn_tensor_t* Y,
                        nn_tensorOpKUs0Idx_t* idx)
{
	// X2 and Y may be NULL
	ASSERT(X1);
	ASSERT(idx);

	nn_engine_t* engine = X1->engine;

	nn_tensorOpKUs0Data_t* self;
	self = (nn_tensorOpKUs0Data_t*)
	       CALLOC(1, sizeof(nn_tensorOpKUs0Data_t));
	if(self == NULL)
	{
		LOGE("CALLOC failed");
		return NULL;
	}

	vkk_updateMode_e um;
	um = vkk_compute_updateMode(engine->compute);
	self->sb006_idx = vkk_buffer_new(engine->engine, um,
	                                 VKK_BUFFER_USAGE_STORAGE,
	                                 sizeof(nn_tensorOpKUs0Idx_t),
	                                 idx);
	if(self->sb006_idx == NULL)
	{
		goto fail_sb006_idx;
	}

	// optionally replace X2 and Y with the Null tensor
	if(X2 == NULL)
	{
		X2 = engine->Null;
	}
	if(Y == NULL)
	{
		Y = engine->Null;
	}

	self->us0 = vkk_uniformSet_new(engine->engine,
	                               0, 0, NULL,
	                               engine->usf0_tensor_opk);
	if(self->us0 == NULL)
	{
		goto fail_us0;
	}

	// sb000: dimX1
	// sb001: X1
	// sb002: dimX2
	// sb003: X2
	// sb004: dimY
	// sb005: Y
	// sb006: idx (x1n,x2n,yn,count,x1k,x2k,yk,depth,value)
	vkk_uniformAttachment_t ua0_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X1->sb_dim,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X1->sb_data,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X2->sb_dim,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X2->sb_data,
		},
		{
			.binding = 4,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Y->sb_dim,
		},
		{
			.binding = 5,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Y->sb_data,
		},
		{
			.binding = 6,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->sb006_idx,
		},
	};

	vkk_compute_updateUniformSetRefs(engine->compute,
	                                 self->us0, 7,
	                                 ua0_array);

	// success
	return self;

	// failure:
	fail_us0:
		vkk_buffer_delete(&self->sb006_idx);
	fail_sb006_idx:
		FREE(self);
	return NULL;
}

void
nn_tensorOpKUs0Data_delete(nn_tensorOpKUs0Data_t** _self)
{
	ASSERT(_self);

	nn_tensorOpKUs0Data_t* self = *_self;
	if(self)
	{
		vkk_uniformSet_delete(&self->us0);
		vkk_buffer_delete(&self->sb006_idx);
		FREE(self);
		*_self = NULL;
	}
}

int
nn_tensorOpKUs0Data_update(nn_tensorOpKUs0Data_t* self,
                           nn_tensor_t* X1,
                           nn_tensor_t* X2,
                           nn_tensor_t* Y,
                           nn_tensorOpKUs0Idx_t* idx)
{
	// X2 and Y may be NULL
	ASSERT(self);
	ASSERT(X1);
	ASSERT(idx);

	nn_engine_t* engine = X1->engine;

	if(vkk_buffer_writeStorage(self->sb006_idx, 0,
	                           sizeof(nn_tensorOpKUs0Idx_t),
	                           idx) == 0)
	{
		return 0;
	}

	// optionally replace X2 and Y with the Null tensor
	if(X2 == NULL)
	{
		X2 = engine->Null;
	}
	if(Y == NULL)
	{
		Y = engine->Null;
	}

	// sb000: dimX1
	// sb001: X1
	// sb002: dimX2
	// sb003: X2
	// sb004: dimY
	// sb005: Y
	// sb006: idx (x1n,x2n,yn,count,x1k,x2k,yk,depth,value)
	vkk_uniformAttachment_t ua0_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X1->sb_dim,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X1->sb_data,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X2->sb_dim,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X2->sb_data,
		},
		{
			.binding = 4,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Y->sb_dim,
		},
		{
			.binding = 5,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Y->sb_data,
		},
		{
			.binding = 6,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->sb006_idx,
		},
	};

	vkk_compute_updateUniformSetRefs(engine->compute,
	                                 self->us0, 7,
	                                 ua0_array);
	return 1;
}

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
		                                 self->us0, 2,
		                                 ua0_array);

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
	jsmn_val_t* val_norm = NULL;
	jsmn_val_t* val_u1   = NULL;
	jsmn_val_t* val_v1   = NULL;
	jsmn_val_t* val_u2   = NULL;
	jsmn_val_t* val_v2   = NULL;

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
			else if(strcmp(kv->key, "u1") == 0)
			{
				val_u1 = kv->val;
			}
			else if(strcmp(kv->key, "v1") == 0)
			{
				val_v1 = kv->val;
			}
			else if(strcmp(kv->key, "u2") == 0)
			{
				val_u2 = kv->val;
			}
			else if(strcmp(kv->key, "v2") == 0)
			{
				val_v2 = kv->val;
			}
		}
		else if(kv->val->type == JSMN_TYPE_STRING)
		{
			if(strcmp(kv->key, "norm") == 0)
			{
				val_norm = kv->val;
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

	nn_tensorNorm_e norm = NN_TENSOR_NORM_NONE;
	if(val_norm)
	{
		if(strcmp(val_norm->data,
		          NN_TENSOR_NORM_STRING_SN) == 0)
		{
			norm = NN_TENSOR_NORM_SN;
		}
		else if(strcmp(val_norm->data,
		               NN_TENSOR_NORM_STRING_BSSN) == 0)
		{
			norm = NN_TENSOR_NORM_BSSN;
		}
	}

	// optional norm working buffers
	if(norm && (self->mode == NN_TENSOR_MODE_COMPUTE))
	{
		if((val_u1 == NULL) || (val_v1 == NULL) ||
		   (val_u2 == NULL) || (val_v2 == NULL))
		{
			LOGE("invalid");
			return 0;
		}

		if(nn_tensor_initNormMode(self, norm, 1.0, 0) == 0)
		{
			return 0;
		}

		if((nn_tensor_importStorage(self, val_u1,
		                            self->sb10_data_u1) == 0) ||
		   (nn_tensor_importStorage(self, val_v1,
		                            self->sb11_data_v1) == 0) ||
		   (nn_tensor_importStorage(self, val_u2,
		                            self->sb12_data_u2) == 0) ||
		   (nn_tensor_importStorage(self, val_v2,
		                            self->sb13_data_v2) == 0))
		{
			return 0;
		}
	}

	return 1;
}

int nn_tensor_export(nn_tensor_t* self,
                    jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	nn_dim_t* dim = nn_tensor_dim(self);

	const char* norm_array[NN_TENSOR_NORM_COUNT] =
	{
		NN_TENSOR_NORM_STRING_NONE,
		NN_TENSOR_NORM_STRING_SN,
		NN_TENSOR_NORM_STRING_BSSN,
	};

	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	ret &= jsmn_stream_key(stream, "%s", "dim");
	ret &= nn_dim_export(dim, stream);
	if(self->mode == NN_TENSOR_MODE_IO)
	{
		ret &= jsmn_stream_key(stream, "%s", "data");
		ret &= jsmn_stream_beginArray(stream);

		uint32_t i;
		for(i = 0; i < nn_dim_sizeElements(dim); ++i)
		{
			ret &= jsmn_stream_float(stream, self->data[i]);
		}
		ret &= jsmn_stream_end(stream);
	}
	else
	{
		ret &= nn_tensor_exportStorage(self, stream, "data",
		                               self->sb_data);

		if(self->norm)
		{
			ret &= jsmn_stream_key(stream, "%s", "norm");
			ret &= jsmn_stream_string(stream, "%s",
			                          norm_array[self->norm]);
			ret &= nn_tensor_exportStorage(self, stream, "u1",
			                               self->sb10_data_u1);
			ret &= nn_tensor_exportStorage(self, stream, "v1",
			                               self->sb11_data_v1);
			ret &= nn_tensor_exportStorage(self, stream, "u2",
			                               self->sb12_data_u2);
			ret &= nn_tensor_exportStorage(self, stream, "v2",
			                               self->sb13_data_v2);
		}
	}
	ret &= jsmn_stream_end(stream);

	return ret;
}

nn_dim_t* nn_tensor_dim(nn_tensor_t* self)
{
	ASSERT(self);

	return &self->dim;
}

nn_tensorMode_e nn_tensor_mode(nn_tensor_t* self)
{
	ASSERT(self);

	return self->mode;
}

int nn_tensor_copy(nn_tensor_t* X,
                   nn_tensor_t* Y,
                   uint32_t xn,
                   uint32_t yn,
                   uint32_t count)
{
	ASSERT(X);
	ASSERT(Y);

	nn_dim_t* dimX = nn_tensor_dim(X);
	nn_dim_t* dimY = nn_tensor_dim(Y);

	size_t x_stride = nn_dim_strideBytes(dimX);
	size_t y_stride = nn_dim_strideBytes(dimY);
	if((count == 0)                ||
	   (x_stride != y_stride)      ||
	   (xn + count > X->dim.count) ||
	   (yn + count > Y->dim.count))
	{
		LOGE("invalid count=%u:%u:%u, n=%u:%u, stride=%u:%u",
		     count, X->dim.count, Y->dim.count,
		     xn, yn,
		     (uint32_t) x_stride,
		     (uint32_t) y_stride);
		return 0;
	}

	float* x_data = NULL;
	if(X->mode == NN_TENSOR_MODE_IO)
	{
		x_data = nn_tensor_data(X, xn);
		if(x_data == NULL)
		{
			return 0;
		}
	}

	float* y_data = NULL;
	if(Y->mode == NN_TENSOR_MODE_IO)
	{
		y_data = nn_tensor_data(Y, yn);
		if(y_data == NULL)
		{
			return 0;
		}
	}

	size_t size = count*x_stride;
	if((X->mode == NN_TENSOR_MODE_IO) &&
	   (Y->mode == NN_TENSOR_MODE_COMPUTE))
	{
		vkk_buffer_writeStorage(Y->sb_data, yn, size,
		                        x_data);
	}
	else if((X->mode == NN_TENSOR_MODE_COMPUTE) &&
	        (Y->mode == NN_TENSOR_MODE_IO))
	{
		vkk_buffer_readStorage(X->sb_data, xn, size,
		                       y_data);
	}
	else if((X->mode == NN_TENSOR_MODE_COMPUTE) &&
	        (Y->mode == NN_TENSOR_MODE_COMPUTE))
	{
		vkk_buffer_copyStorage(X->sb_data, Y->sb_data,
		                       xn, yn, size);
	}
	else
	{
		memcpy(y_data, x_data, size);
	}

	return 1;
}

int nn_tensor_ioClear(nn_tensor_t* self,
                      uint32_t n,
                      uint32_t count)
{
	ASSERT(self);

	if(self->mode != NN_TENSOR_MODE_IO)
	{
		LOGE("invalid");
		return 0;
	}

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

int nn_tensor_ioCopy(nn_tensor_t* X,
                     nn_tensor_t* Y,
                     uint32_t xn,
                     uint32_t yn,
                     uint32_t count)
{
	ASSERT(X);
	ASSERT(Y);

	if((X->mode != NN_TENSOR_MODE_IO) ||
	   (Y->mode != NN_TENSOR_MODE_IO))
	{
		LOGE("invalid");
		return 0;
	}

	nn_dim_t* dimX = nn_tensor_dim(X);
	nn_dim_t* dimY = nn_tensor_dim(Y);
	if(nn_dim_strideEquals(dimX, dimY) == 0)
	{
		LOGE("invalid height=%u:%u, width=%u:%u, depth=%u:%u",
		     dimX->height, dimY->height,
		     dimX->width,  dimY->width,
		     dimX->depth,  dimY->depth);
		return 0;
	}

	if(((count + xn) > dimX->count) ||
	   ((count + yn) > dimY->count))
	{
		LOGE("invalid n=%u:%u, count=%u:%u:%u",
		     xn, yn, count, dimX->count, dimY->count);
		return 0;
	}

	float* x_data = nn_tensor_data(X, xn);
	float* y_data = nn_tensor_data(Y, yn);
	if((x_data == NULL) || (y_data == NULL))
	{
		return 0;
	}

	size_t bytes = nn_dim_strideBytes(dimX);
	memcpy(y_data, x_data, count*bytes);

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

int nn_tensor_computeCopy(nn_tensor_t* X,
                          nn_tensor_t* Y,
                          vkk_hazard_e hazard,
                          uint32_t xn,
                          uint32_t yn,
                          uint32_t count)
{
	ASSERT(X);
	ASSERT(Y);

	nn_engine_t* engine = X->engine;

	if((X->mode != NN_TENSOR_MODE_COMPUTE) ||
	   (Y->mode != NN_TENSOR_MODE_COMPUTE))
	{
		LOGE("invalid mode=%i:%i",
		     X->mode, Y->mode);
		return 0;
	}

	if(vkk_compute_active(engine->compute) == 0)
	{
		LOGE("invalid");
		return 0;
	}

	nn_dim_t* dimX = nn_tensor_dim(X);
	nn_dim_t* dimY = nn_tensor_dim(Y);

	size_t x_stride = nn_dim_strideBytes(dimX);
	size_t y_stride = nn_dim_strideBytes(dimY);
	if((count == 0)                ||
	   (x_stride != y_stride)      ||
	   (xn + count > X->dim.count) ||
	   (yn + count > Y->dim.count))
	{
		LOGE("invalid count=%u:%u:%u, n=%u:%u, stride=%u:%u",
		     count, X->dim.count, Y->dim.count,
		     xn, yn,
		     (uint32_t) x_stride,
		     (uint32_t) y_stride);
		return 0;
	}

	size_t bytes = nn_dim_strideBytes(dimX);
	vkk_compute_copyStorage(engine->compute, hazard,
	                        X->sb_data, Y->sb_data,
	                        xn*bytes, yn*bytes,
	                        count*bytes);

	return 1;
}

int nn_tensor_computeFillK(nn_tensor_t* self,
                           vkk_hazard_e hazard,
                           uint32_t n,
                           uint32_t count,
                           uint32_t k,
                           uint32_t depth,
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
	if((count == 0) || ((n + count) > dim->count))
	{
		LOGE("invalid n=%u, count=%u:%u", n, count, dim->count);
		return 0;
	}
	if((depth == 0) || ((k + depth) > dim->depth))
	{
		LOGE("invalid k=%u, depth=%u:%u", k, depth, dim->depth);
		return 0;
	}

	vkk_uniformSet_t* us0;
	us0 = nn_engine_getTensorOpKUs0(engine, self, NULL, NULL,
	                                n, 0, 0, count,
	                                k, 0, 0, depth, value);
	if(us0 == NULL)
	{
		return 0;
	}

	vkk_uniformSet_t* us_array[] =
	{
		us0,
	};

	// nn_tensor_fillK.comp
	// dispatch(hazard, count, xh, xw, 1, 8, 8)
	vkk_computePipeline_t* cp = engine->cp_tensor_fillk;
	if(nn_engine_computeBind(engine, cp) == 0)
	{
		return 0;
	}
	vkk_compute_bindUniformSets(engine->compute, 1,
	                            us_array);
	nn_engine_computeDispatch(engine, hazard,
	                          count, dim->height, dim->width,
	                          1, 8, 8);

	return 1;
}

int nn_tensor_computeCopyK(nn_tensor_t* X,
                           nn_tensor_t* Y,
                           vkk_hazard_e hazard,
                           uint32_t xn,
                           uint32_t yn,
                           uint32_t count,
                           uint32_t xk,
                           uint32_t yk,
                           uint32_t depth)
{
	ASSERT(X);
	ASSERT(Y);

	nn_engine_t* engine = X->engine;

	if((X->mode != NN_TENSOR_MODE_COMPUTE) ||
	   (Y->mode != NN_TENSOR_MODE_COMPUTE))
	{
		LOGE("invalid mode=%i:%i", X->mode, Y->mode);
		return 0;
	}

	if(vkk_compute_active(engine->compute) == 0)
	{
		LOGE("invalid");
		return 0;
	}

	nn_dim_t* dimX = nn_tensor_dim(X);
	nn_dim_t* dimY = nn_tensor_dim(Y);
	if((count == 0)                 ||
	   ((xn + count) > dimX->count) ||
	   ((yn + count) > dimY->count))
	{
		LOGE("invalid n=%u:%u, count=%u:%u:%u",
		     xn, yn, count, dimX->count, dimY->count);
		return 0;
	}
	if((depth == 0)                 ||
	   ((xk + depth) > dimX->depth) ||
	   ((yk + depth) > dimY->depth))
	{
		LOGE("invalid k=%u:%u, depth=%u:%u:%u",
		     xk, yk, depth, dimX->depth, dimY->depth);
		return 0;
	}
	if((dimX->height != dimY->height) ||
	   (dimX->width  != dimY->width))
	{
		LOGE("invalid height=%u:%u, width=%u:%u",
		     dimX->height, dimY->height,
		     dimX->width,  dimY->width);
		return 0;
	}

	vkk_uniformSet_t* us0;
	us0 = nn_engine_getTensorOpKUs0(engine, X, NULL, Y,
	                                xn, 0, yn, count,
	                                xk, 0, yk, depth,
	                                0.0f);
	if(us0 == NULL)
	{
		return 0;
	}

	vkk_uniformSet_t* us_array[] =
	{
		us0,
	};

	// nn_tensor_copyK.comp
	// dispatch(hazard, count, xh, xw, 1, 8, 8)
	vkk_computePipeline_t* cp = engine->cp_tensor_copyk;
	if(nn_engine_computeBind(engine, cp) == 0)
	{
		return 0;
	}
	vkk_compute_bindUniformSets(engine->compute, 1,
	                            us_array);
	nn_engine_computeDispatch(engine, hazard, count,
	                          dimX->height, dimX->width,
	                          1, 8, 8);

	return 1;
}

int nn_tensor_computeAddK(nn_tensor_t* X1,
                          nn_tensor_t* X2,
                          nn_tensor_t* Y,
                          vkk_hazard_e hazard,
                          uint32_t x1n,
                          uint32_t x2n,
                          uint32_t yn,
                          uint32_t count,
                          uint32_t x1k,
                          uint32_t x2k,
                          uint32_t yk,
                          uint32_t depth)
{
	ASSERT(X1);
	ASSERT(X2);
	ASSERT(Y);

	nn_engine_t* engine = X1->engine;

	if((X1->mode != NN_TENSOR_MODE_COMPUTE) ||
	   (X2->mode != NN_TENSOR_MODE_COMPUTE) ||
	   (Y->mode  != NN_TENSOR_MODE_COMPUTE))
	{
		LOGE("invalid mode=%i:%i:%i",
		     X1->mode, X2->mode, Y->mode);
		return 0;
	}

	if(vkk_compute_active(engine->compute) == 0)
	{
		LOGE("invalid");
		return 0;
	}

	nn_dim_t* dimX1 = nn_tensor_dim(X1);
	nn_dim_t* dimX2 = nn_tensor_dim(X2);
	nn_dim_t* dimY  = nn_tensor_dim(Y);
	if((count == 0)                   ||
	   ((x1n + count) > dimX1->count) ||
	   ((x2n + count) > dimX2->count) ||
	   ((yn  + count) > dimY->count))
	{
		LOGE("invalid n=%u:%u:%u, count=%u:%u:%u:%u",
		     x1n, x2n, yn, count, dimX1->count,
		     dimX2->count, dimY->count);
		return 0;
	}
	if((depth == 0)                   ||
	   ((x1k + depth) > dimX1->depth) ||
	   ((x2k + depth) > dimX2->depth) ||
	   ((yk  + depth) > dimY->depth))
	{
		LOGE("invalid k=%u:%u:%u, depth=%u:%u:%u:%u",
		     x1k, x2k, yk, depth, dimX1->depth,
		     dimX2->depth, dimY->depth);
		return 0;
	}
	if((dimX1->height != dimX2->height) ||
	   (dimX1->height != dimY->height)  ||
	   (dimX1->width  != dimX2->width)  ||
	   (dimX1->width  != dimY->width))
	{
		LOGE("invalid height=%u:%u:%u, width=%u:%u:%u",
		     dimX1->height, dimX2->height,
		     dimY->height, dimX1->width,
		     dimX2->width, dimY->width);
		return 0;
	}

	vkk_uniformSet_t* us0;
	us0 = nn_engine_getTensorOpKUs0(engine, X1, X2, Y,
	                                x1n, x2n, yn, count,
	                                x1k, x2k, yk, depth,
	                                0.0f);
	if(us0 == NULL)
	{
		return 0;
	}

	vkk_uniformSet_t* us_array[] =
	{
		us0,
	};

	// nn_tensor_addK.comp
	// dispatch(hazard, count, xh, xw, 1, 8, 8)
	vkk_computePipeline_t* cp = engine->cp_tensor_addk;
	if(nn_engine_computeBind(engine, cp) == 0)
	{
		return 0;
	}
	vkk_compute_bindUniformSets(engine->compute, 1,
	                            us_array);
	nn_engine_computeDispatch(engine, hazard, count,
	                          dimX1->height,
	                          dimX1->width,
	                          1, 8, 8);

	return 1;
}

int nn_tensor_computeMixK(nn_tensor_t* X1,
                          nn_tensor_t* X2,
                          nn_tensor_t* Y,
                          vkk_hazard_e hazard,
                          uint32_t x1n,
                          uint32_t x2n,
                          uint32_t yn,
                          uint32_t count,
                          uint32_t x1k,
                          uint32_t x2k,
                          uint32_t yk,
                          uint32_t depth,
                          float value)
{
	ASSERT(X1);
	ASSERT(X2);
	ASSERT(Y);

	nn_engine_t* engine = X1->engine;

	if((X1->mode != NN_TENSOR_MODE_COMPUTE) ||
	   (X2->mode != NN_TENSOR_MODE_COMPUTE) ||
	   (Y->mode  != NN_TENSOR_MODE_COMPUTE))
	{
		LOGE("invalid mode=%i:%i:%i",
		     X1->mode, X2->mode, Y->mode);
		return 0;
	}

	if(vkk_compute_active(engine->compute) == 0)
	{
		LOGE("invalid");
		return 0;
	}

	nn_dim_t* dimX1 = nn_tensor_dim(X1);
	nn_dim_t* dimX2 = nn_tensor_dim(X2);
	nn_dim_t* dimY  = nn_tensor_dim(Y);
	if((count == 0)                   ||
	   ((x1n + count) > dimX1->count) ||
	   ((x2n + count) > dimX2->count) ||
	   ((yn  + count) > dimY->count))
	{
		LOGE("invalid n=%u:%u:%u, count=%u:%u:%u:%u",
		     x1n, x2n, yn,
		     count, dimX1->count,
		     dimX2->count, dimY->count);
		return 0;
	}
	if((depth == 0)                   ||
	   ((x1k + depth) > dimX1->depth) ||
	   ((x2k + depth) > dimX2->depth) ||
	   ((yk  + depth) > dimY->depth))
	{
		LOGE("invalid k=%u:%u:%u, depth=%u:%u:%u:%u",
		     x1k, x2k, yk,
		     depth, dimX1->depth,
		     dimX2->depth, dimY->depth);
		return 0;
	}
	if((dimX1->height != dimX2->height) ||
	   (dimX1->height != dimY->height)  ||
	   (dimX1->width  != dimX2->width)  ||
	   (dimX1->width  != dimY->width))
	{
		LOGE("invalid height=%u:%u:%u, width=%u:%u:%u",
		     dimX1->height, dimX2->height, dimY->height,
		     dimX1->width, dimX2->width, dimY->width);
		return 0;
	}

	vkk_uniformSet_t* us0;
	us0 = nn_engine_getTensorOpKUs0(engine, X1, X2, Y,
	                                x1n, x2n, yn, count,
	                                x1k, x2k, yk, depth,
	                                value);
	if(us0 == NULL)
	{
		return 0;
	}

	vkk_uniformSet_t* us_array[] =
	{
		us0,
	};

	// nn_tensor_mixK.comp
	// dispatch(hazard, count, xh, xw, 1, 8, 8)
	vkk_computePipeline_t* cp = engine->cp_tensor_mixk;
	if(nn_engine_computeBind(engine, cp) == 0)
	{
		return 0;
	}
	vkk_compute_bindUniformSets(engine->compute, 1,
	                            us_array);
	nn_engine_computeDispatch(engine, hazard, count,
	                          dimX1->height,
	                          dimX1->width,
	                          1, 8, 8);

	return 1;
}

int nn_tensor_computeNormalize(nn_tensor_t* self,
                               vkk_hazard_e hazard,
                               nn_tensorNorm_e norm,
                               float c)
{
	ASSERT(self);

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

	if(nn_tensor_initNormMode(self, norm, c, 1) == 0)
	{
		return 0;
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
	if(norm == NN_TENSOR_NORM_BSSN)
	{
		cp = engine->cp_tensor_bssn;
	}

	if(nn_engine_computeBind(engine, cp) == 0)
	{
		return 0;
	}
	vkk_compute_updateUniformSetRefs(engine->compute,
	                                 self->us1_norm, 5,
	                                 ua1_array);
	vkk_compute_bindUniformSets(engine->compute, 2,
	                            us_array);
	nn_engine_computeDispatch(engine, hazard,
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

	nn_tensorStats_update(stats, count);

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		stats->us1,
	};

	// dispatch(hazard, 1, 1, 1, 8, 8, 1)
	vkk_computePipeline_t* cp = engine->cp_tensor_stats;
	if(nn_engine_computeBind(engine, cp) == 0)
	{
		return 0;
	}
	vkk_compute_bindUniformSets(engine->compute, 2,
	                            us_array);
	nn_engine_computeDispatch(engine, hazard,
	                          1, 1, 1, 8, 8, 1);

	return 1;
}
