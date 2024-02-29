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
#include "nn_engine.h"
#include "nn_layer.h"
#include "nn_loss.h"
#include "nn_tensorStats.h"
#include "nn_tensor.h"

const char* NN_LOSS_STRING_MSE = "mse";
const char* NN_LOSS_STRING_MAE = "mae";
const char* NN_LOSS_STRING_BCE = "bce";

/***********************************************************
* private                                                  *
***********************************************************/

static const char* nn_loss_string(nn_lossFn_e fn)
{
	ASSERT(fn >= 0);
	ASSERT(fn < NN_LOSS_FN_COUNT);

	const char* str_array[NN_LOSS_FN_COUNT] =
	{
		NN_LOSS_STRING_MSE,
		NN_LOSS_STRING_MAE,
		NN_LOSS_STRING_BCE,
	};

	return str_array[fn];
}

static nn_lossFn_e nn_loss_function(const char* str)
{
	ASSERT(str);

	const char* str_fn[NN_LOSS_FN_COUNT] =
	{
		NN_LOSS_STRING_MSE,
		NN_LOSS_STRING_MAE,
		NN_LOSS_STRING_BCE,
	};

	int i;
	for(i = 0; i < NN_LOSS_FN_COUNT; ++i)
	{
		if(strcmp(str, str_fn[i]) == 0)
		{
			return (nn_lossFn_e) i;
		}
	}

	LOGE("invalid %s", str);
	return NN_LOSS_FN_ERROR;
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_loss_t*
nn_loss_new(nn_arch_t* arch, nn_dim_t* dimY,
            nn_lossFn_e loss_fn)
{
	ASSERT(arch);
	ASSERT(dimY);

	nn_engine_t* engine = arch->engine;

	if(((int) loss_fn < 0) ||
	   ((int) loss_fn >= NN_LOSS_FN_COUNT))
	{
		LOGE("invalid loss_fn=%i", (int) loss_fn);
		return NULL;
	}

	nn_loss_t* self;
	self = (nn_loss_t*) CALLOC(1, sizeof(nn_loss_t));
	if(self == NULL)
	{
		LOGE("CALLOC failed");
		return NULL;
	}

	self->arch    = arch;
	self->loss_fn = loss_fn;

	self->dL_dY = nn_tensor_new(engine, dimY,
	                            NN_TENSOR_INIT_ZERO,
	                            NN_TENSOR_MODE_COMPUTE);
	if(self->dL_dY == NULL)
	{
		goto fail_dL_dY;
	}

	self->stats_dL_dY = nn_tensorStats_new(engine);
	if(self->stats_dL_dY == NULL)
	{
		goto fail_stats_dL_dY;
	}

	self->us0 = vkk_uniformSet_new(engine->engine, 0, 0, NULL,
	                               engine->usf0_loss);
	if(self->us0 == NULL)
	{
		goto fail_us0;
	}

	vkk_updateMode_e um;
	um = vkk_compute_updateMode(engine->compute);

	float loss = 0.0f;
	self->sb07_loss = vkk_buffer_new(engine->engine, um,
	                                 VKK_BUFFER_USAGE_STORAGE,
	                                 sizeof(float), &loss);
	if(self->sb07_loss == NULL)
	{
		goto fail_sb07_loss;
	}

	// success
	return self;

	// failure
	fail_sb07_loss:
		vkk_uniformSet_delete(&self->us0);
	fail_us0:
		nn_tensorStats_delete(&self->stats_dL_dY);
	fail_stats_dL_dY:
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

	nn_dim_t dimY;
	if(nn_dim_load(&dimY, val_dimY) == 0)
	{
		return NULL;
	}

	nn_lossFn_e loss_fn;
	loss_fn = nn_loss_function(val_loss_fn->data);
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
		vkk_buffer_delete(&self->sb07_loss);
		vkk_uniformSet_delete(&self->us0);
		nn_tensorStats_delete(&self->stats_dL_dY);
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

	nn_arch_t*   arch   = self->arch;
	nn_engine_t* engine = arch->engine;
	nn_tensor_t* dL_dY  = self->dL_dY;
	nn_dim_t*    dimY   = nn_tensor_dim(Y);

	vkk_computePipeline_t* cp;
	vkk_computePipeline_t* cp_dL_dY;
	if(self->loss_fn == NN_LOSS_FN_MSE)
	{
		cp       = engine->cp_loss_mse;
		cp_dL_dY = engine->cp_loss_dL_dY_mse;
	}
	else if(self->loss_fn == NN_LOSS_FN_MAE)
	{
		cp       = engine->cp_loss_mae;
		cp_dL_dY = engine->cp_loss_dL_dY_mae;
	}
	else if(self->loss_fn == NN_LOSS_FN_BCE)
	{
		cp       = engine->cp_loss_bce;
		cp_dL_dY = engine->cp_loss_dL_dY_bce;
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
			.buffer  = arch->sb00_state,
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
	if(nn_engine_bind(engine, cp) == 0)
	{
		return NULL;
	}
	vkk_compute_updateUniformSetRefs(engine->compute, self->us0,
	                                 8, ua0_array);
	vkk_compute_bindUniformSets(engine->compute, 1, us_array);
	nn_engine_dispatch(engine, VKK_HAZARD_RAW,
	                   1, 1, 1, 8, 8, 1);

	// nn_loss_dL_dY
	// RAW hazard handled by nn_loss
	// dispatch(NONE, bs, yh, yw, 1, 8, 8)
	if(nn_engine_bind(engine, cp_dL_dY) == 0)
	{
		return NULL;
	}
	nn_engine_dispatch(engine, VKK_HAZARD_NONE,
	                   bs, dimY->height, dimY->width,
	                   1, 8, 8);

	if(nn_tensor_computeStats(dL_dY, bs,
	                          VKK_HAZARD_RAW,
	                          self->stats_dL_dY) == 0)
	{
		return NULL;
	}

	return dL_dY;
}

void nn_loss_post(nn_loss_t* self, int flags)
{
	ASSERT(self);

	vkk_buffer_readStorage(self->sb07_loss, sizeof(float), 0,
	                       &self->loss);

	if(flags & NN_LAYER_FLAG_BACKPROP)
	{
		LOGI("dL_dY min=%f, max=%f, mean=%f, stddev=%f, norm=%f",
		     nn_tensorStats_min(self->stats_dL_dY),
		     nn_tensorStats_max(self->stats_dL_dY),
		     nn_tensorStats_mean(self->stats_dL_dY),
		     nn_tensorStats_stddev(self->stats_dL_dY),
		     nn_tensorStats_norm(self->stats_dL_dY));
	}
}

nn_dim_t* nn_loss_dimY(nn_loss_t* self)
{
	ASSERT(self);

	return nn_tensor_dim(self->dL_dY);
}
