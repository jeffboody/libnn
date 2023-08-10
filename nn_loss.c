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
* private                                                  *
***********************************************************/

#ifdef NN_USE_COMPUTE

static int nn_loss_newCompute(nn_loss_t* self)
{
	ASSERT(self);

	nn_arch_t* arch = self->arch;

	self->us0 = vkk_uniformSet_new(arch->engine, 0, 0, NULL,
	                               arch->usf0_loss);
	if(self->us0 == NULL)
	{
		return 0;
	}

	vkk_updateMode_e um;
	um = vkk_compute_updateMode(arch->compute);

	float loss = 0.0f;
	self->sb07_loss = vkk_buffer_new(arch->engine, um,
	                                 VKK_BUFFER_USAGE_STORAGE,
	                                 sizeof(float), &loss);
	if(self->sb07_loss == NULL)
	{
		goto fail_sb07_loss;
	}

	// success
	return 1;

	// failure
	fail_sb07_loss:
		vkk_uniformSet_delete(&self->us0);
	return 0;
}

static void nn_loss_deleteCompute(nn_loss_t* self)
{
	ASSERT(self);

	vkk_buffer_delete(&self->sb07_loss);
	vkk_uniformSet_delete(&self->us0);
}

#else // NN_USE_COMPUTE not defined

static int nn_loss_newCompute(nn_loss_t* self)
{
	return 1;
}

static void nn_loss_deleteCompute(nn_loss_t* self)
{
}

#endif


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

	self->dL_dY = nn_tensor_new(arch, dimY,
	                            NN_TENSOR_INIT_ZERO,
	                            NN_TENSOR_MODE_COMPUTE);
	if(self->dL_dY == NULL)
	{
		goto fail_dL_dY;
	}

	if(nn_loss_newCompute(self) == 0)
	{
		goto fail_compute;
	}

	// success
	return self;

	// failure
	fail_compute:
		nn_tensor_delete(&self->dL_dY);
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
		nn_loss_deleteCompute(self);
		nn_tensor_delete(&self->dL_dY);
		FREE(self);
		*_self = self;
	}
}

#ifdef NN_USE_COMPUTE

nn_tensor_t*
nn_loss_loss(nn_loss_t* self, uint32_t bs,
             nn_tensor_t* Y, nn_tensor_t* Yt)
{
	ASSERT(self);
	ASSERT(Y);
	ASSERT(Yt);

	nn_arch_t*   arch  = self->arch;
	nn_tensor_t* dL_dY = self->dL_dY;
	nn_dim_t*    dimY  = nn_tensor_dim(Y);

	vkk_computePipeline_t* cp;
	vkk_computePipeline_t* cp_dL_dY;
	if(self->loss_fn == nn_loss_mse)
	{
		cp       = arch->cp_loss_mse;
		cp_dL_dY = arch->cp_loss_dL_dY_mse;
	}
	else if(self->loss_fn == nn_loss_mae)
	{
		cp       = arch->cp_loss_mae;
		cp_dL_dY = arch->cp_loss_dL_dY_mae;
	}
	else if(self->loss_fn == nn_loss_bce)
	{
		cp       = arch->cp_loss_bce;
		cp_dL_dY = arch->cp_loss_dL_dY_bce;
	}
	else
	{
		LOGE("invalid");
		return NULL;
	}

	// sb00: state
	// sb01: dimY
	// sb02: Y
	// sb03: dimYt
	// sb04: Yt
	// sb05: dim_dL_dY
	// sb06: dL_dY
	// sb07: loss
	vkk_uniformAttachment_t ua0_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = arch->sb_state,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Y->sb_dim,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Y->sb_data,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Yt->sb_dim,
		},
		{
			.binding = 4,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Yt->sb_data,
		},
		{
			.binding = 5,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY->sb_dim,
		},
		{
			.binding = 6,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY->sb_data,
		},
		{
			.binding = 7,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->sb07_loss,
		},
	};

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
	};

	// nn_loss
	// dispatch(RAW, 1, 1, 1, 8, 8, 1)
	vkk_compute_bindComputePipeline(arch->compute, cp);
	vkk_compute_updateUniformSetRefs(arch->compute, self->us0,
	                                 8, ua0_array);
	vkk_compute_bindUniformSets(arch->compute, 1, us_array);
	vkk_compute_dispatch(arch->compute, VKK_HAZZARD_RAW,
	                     1, 1, 1, 8, 8, 1);

	// nn_loss_dL_dY
	// RAW hazzard handled by nn_loss
	// dispatch(NONE, bs, yh, yw, 1, 8, 8)
	vkk_compute_bindComputePipeline(arch->compute, cp_dL_dY);
	vkk_compute_dispatch(arch->compute, VKK_HAZZARD_NONE,
	                     bs, dimY->height, dimY->width,
	                     1, 8, 8);

	// loss is read by nn_arch_endCompute

	return dL_dY;
}

#else // NN_USE_COMPUTE not defined

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

#endif

nn_dim_t* nn_loss_dimY(nn_loss_t* self)
{
	ASSERT(self);

	return nn_tensor_dim(self->dL_dY);
}
