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

static int
nn_loss_function(const char* str, nn_lossFn_e* _loss_fn)
{
	ASSERT(str);
	ASSERT(_loss_fn);

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
			*_loss_fn = (nn_lossFn_e) i;
			return 1;
		}
	}

	LOGE("invalid %s", str);
	return 0;
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_loss_t*
nn_loss_new(nn_engine_t* engine, nn_dim_t* dimY,
            nn_lossFn_e loss_fn)
{
	ASSERT(engine);
	ASSERT(dimY);

	nn_loss_t* self;
	self = (nn_loss_t*) CALLOC(1, sizeof(nn_loss_t));
	if(self == NULL)
	{
		LOGE("CALLOC failed");
		return NULL;
	}

	self->engine  = engine;
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

	vkk_updateMode_e um;
	um = vkk_compute_updateMode(engine->compute);

	self->sb000_bs = vkk_buffer_new(engine->engine, um,
	                                VKK_BUFFER_USAGE_STORAGE,
	                                sizeof(uint32_t), NULL);
	if(self->sb000_bs == NULL)
	{
		goto fail_sb000_bs;
	}

	self->sb001_loss = vkk_buffer_new(engine->engine, um,
	                                  VKK_BUFFER_USAGE_STORAGE,
	                                  sizeof(float), NULL);
	if(self->sb001_loss == NULL)
	{
		goto fail_sb001_loss;
	}

	self->us0 = vkk_uniformSet_new(engine->engine, 0, 0, NULL,
	                               engine->usf0_loss);
	if(self->us0 == NULL)
	{
		goto fail_us0;
	}

	self->us1 = vkk_uniformSet_new(engine->engine, 1, 0, NULL,
	                               engine->usf1_loss);
	if(self->us1 == NULL)
	{
		goto fail_us1;
	}

	// sb000: bs
	// sb001: loss
	// sb002: dimY
	// sb003: dL_dY
	vkk_uniformAttachment_t ua0_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->sb000_bs,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->sb001_loss,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->dL_dY->sb_dim,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->dL_dY->sb_data,
		},
	};

	vkk_compute_updateUniformSetRefs(engine->compute,
	                                 self->us0, 4,
	                                 ua0_array);

	// success
	return self;

	// failure
	fail_us1:
		vkk_uniformSet_delete(&self->us0);
	fail_us0:
		vkk_buffer_delete(&self->sb001_loss);
	fail_sb001_loss:
		vkk_buffer_delete(&self->sb000_bs);
	fail_sb000_bs:
		nn_tensorStats_delete(&self->stats_dL_dY);
	fail_stats_dL_dY:
		nn_tensor_delete(&self->dL_dY);
	fail_dL_dY:
		FREE(self);
	return NULL;
}

void nn_loss_delete(nn_loss_t** _self)
{
	ASSERT(_self);

	nn_loss_t* self = *_self;
	if(self)
	{
		vkk_uniformSet_delete(&self->us1);
		vkk_uniformSet_delete(&self->us0);
		vkk_buffer_delete(&self->sb001_loss);
		vkk_buffer_delete(&self->sb000_bs);
		nn_tensorStats_delete(&self->stats_dL_dY);
		nn_tensor_delete(&self->dL_dY);
		FREE(self);
		*_self = self;
	}
}

nn_loss_t*
nn_loss_import(nn_engine_t* engine, jsmn_val_t* val)
{
	ASSERT(engine);
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
	if(nn_dim_import(&dimY, val_dimY) == 0)
	{
		return NULL;
	}

	nn_lossFn_e loss_fn;
	if(nn_loss_function(val_loss_fn->data, &loss_fn) == 0)
	{
		return NULL;
	}

	return nn_loss_new(engine, &dimY, loss_fn);
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
	ret &= nn_dim_export(dimY, stream);
	ret &= jsmn_stream_end(stream);

	return ret;
}

nn_tensor_t*
nn_loss_loss(nn_loss_t* self, uint32_t bs,
             nn_tensor_t* Y, nn_tensor_t* Yt)
{
	ASSERT(self);
	ASSERT(Y);
	ASSERT(Yt);

	nn_engine_t* engine = self->engine;
	nn_tensor_t* dL_dY  = self->dL_dY;
	nn_dim_t*    dimY   = nn_tensor_dim(Y);

	nn_dim_t* dimY1 = nn_loss_dimY(self);
	nn_dim_t* dimY2 = nn_tensor_dim(Y);
	nn_dim_t* dimY3 = nn_tensor_dim(Yt);
	if((nn_dim_sizeEquals(dimY1, dimY2) == 0) ||
	   (nn_dim_sizeEquals(dimY1, dimY3) == 0))
	{
		LOGE("invalid count=%u:%u:%u, height=%u:%u:%u, width=%u:%u:%u, depth=%u:%u:%u",
		     dimY1->count,  dimY2->count,  dimY3->count,
		     dimY1->height, dimY2->height, dimY3->height,
		     dimY1->width,  dimY2->width,  dimY3->width,
		     dimY1->depth,  dimY2->depth,  dimY3->depth);
		return NULL;
	}

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

	vkk_buffer_writeStorage(self->sb000_bs, 0,
	                        sizeof(uint32_t), &bs);

	// sb100: Y
	// sb101: Yt
	vkk_uniformAttachment_t ua1_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Y->sb_data,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Yt->sb_data,
		},
	};

	vkk_compute_updateUniformSetRefs(engine->compute,
	                                 self->us1, 2,
	                                 ua1_array);

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1,
	};

	// nn_loss
	// dispatch(RAW, 1, 1, 1, 8, 8, 1)
	if(nn_engine_computeBind(engine, cp) == 0)
	{
		return NULL;
	}
	vkk_compute_bindUniformSets(engine->compute, 2, us_array);
	nn_engine_computeDispatch(engine, VKK_HAZARD_RAW,
	                          1, 1, 1, 8, 8, 1);

	// nn_loss_dL_dY
	// RAW hazard handled by nn_loss
	// dispatch(NONE, bs, yh, yw, 1, 8, 8)
	if(nn_engine_computeBind(engine, cp_dL_dY) == 0)
	{
		return NULL;
	}
	nn_engine_computeDispatch(engine, VKK_HAZARD_NONE,
	                          bs, dimY->height, dimY->width,
	                          1, 8, 8);

	if(nn_tensor_computeStats(dL_dY, VKK_HAZARD_RAW, bs,
	                          self->stats_dL_dY) == 0)
	{
		return NULL;
	}

	return dL_dY;
}

void nn_loss_post(nn_loss_t* self, int flags)
{
	ASSERT(self);

	vkk_buffer_readStorage(self->sb001_loss, 0,
	                       sizeof(float), &self->loss);

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
