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
#include "../libvkk/vkk.h"
#include "nn_arch.h"
#include "nn_engine.h"
#include "nn_layer.h"
#include "nn_loss.h"
#include "nn_tensor.h"

/***********************************************************
* private                                                  *
***********************************************************/

static void
nn_arch_post(nn_arch_t* self, nn_layerMode_e layer_mode)
{
	ASSERT(self);

	cc_listIter_t* iter = cc_list_head(self->layers);
	while(iter)
	{
		nn_layer_t* layer;
		layer = (nn_layer_t*) cc_list_peekIter(iter);

		nn_layer_post(layer, layer_mode);

		iter = cc_list_next(iter);
	}

	if(self->loss)
	{
		nn_loss_post(self->loss);
	}
}

static int
nn_arch_beginCompute(nn_arch_t* self,
                     uint32_t bs,
                     nn_tensor_t* X,
                     nn_tensor_t* Yt)
{
	// Yt may be NULL
	ASSERT(self);
	ASSERT(X);

	nn_engine_t* engine = self->engine;

	// optionally resize X
	if(X->tensor_mode == NN_TENSOR_MODE_IO)
	{
		if(self->X)
		{
			if(nn_dim_equals(nn_tensor_dim(self->X),
			                 nn_tensor_dim(X)) == 0)
			{
				nn_tensor_delete(&self->X);
			}
		}

		// allocate X on demand
		if(self->X == NULL)
		{
			self->X = nn_tensor_new(engine, nn_tensor_dim(X),
			                        NN_TENSOR_INIT_ZERO,
			                        NN_TENSOR_MODE_COMPUTE);
			if(self->X == NULL)
			{
				return 0;
			}
		}
	}

	// optionally resize Yt
	if(Yt && (Yt->tensor_mode == NN_TENSOR_MODE_IO))
	{
		if(self->Yt)
		{
			if(nn_dim_equals(nn_tensor_dim(self->Yt),
			                 nn_tensor_dim(Yt)) == 0)
			{
				nn_tensor_delete(&self->Yt);
			}
		}

		// allocate Yt on demand
		if(self->Yt == NULL)
		{
			self->Yt = nn_tensor_new(engine, nn_tensor_dim(Yt),
			                         NN_TENSOR_INIT_ZERO,
			                         NN_TENSOR_MODE_COMPUTE);
			if(self->Yt == NULL)
			{
				return 0;
			}
		}
	}

	if(vkk_compute_begin(engine->compute) == 0)
	{
		return 0;
	}

	// optionally blit X
	if(X->tensor_mode == NN_TENSOR_MODE_IO)
	{
		if(nn_tensor_blit(X, self->X, bs, 0, 0) == 0)
		{
			goto fail_blit;
		}
	}

	// optionally blit Yt
	if(Yt && (Yt->tensor_mode == NN_TENSOR_MODE_IO))
	{
		if(nn_tensor_blit(Yt, self->Yt, bs, 0, 0) == 0)
		{
			goto fail_blit;
		}
	}

	// update global state
	self->state.bs = bs;
	vkk_compute_writeBuffer(engine->compute,
	                        self->sb_state,
	                        sizeof(nn_archState_t),
	                        0, &self->state);

	engine->computing = 1;

	// success
	return 1;

	// failure
	fail_blit:
		vkk_compute_end(engine->compute);
	return 0;
}

static void nn_arch_endCompute(nn_arch_t* self)
{
	ASSERT(self);

	nn_engine_t* engine = self->engine;

	if(engine->computing)
	{
		LOGD("DISPATCH %i", engine->dispatch);

		engine->computing = 0;
		engine->dispatch  = 0;

		vkk_compute_end(engine->compute);
	}
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_arch_t* nn_arch_new(nn_engine_t* engine,
                       size_t base_size,
                       nn_archState_t* state)
{
	ASSERT(engine);
	ASSERT(state);

	if(base_size == 0)
	{
		base_size = sizeof(nn_arch_t);
	}

	nn_arch_t* self;
	self = (nn_arch_t*) CALLOC(1, base_size);
	if(self == NULL)
	{
		LOGE("CALLOC failed");
		return NULL;
	}

	self->engine = engine;

	memcpy(&self->state, state, sizeof(nn_archState_t));

	vkk_updateMode_e um;
	um = vkk_compute_updateMode(engine->compute);

	self->sb_state = vkk_buffer_new(engine->engine, um,
	                                VKK_BUFFER_USAGE_STORAGE,
	                                sizeof(nn_archState_t),
	                                NULL);
	if(self->sb_state == NULL)
	{
		goto fail_sb_state;
	}

	self->layers = cc_list_new();
	if(self->layers == NULL)
	{
		goto fail_layers;
	}

	// success
	return self;

	// failure
	fail_layers:
		vkk_buffer_delete(&self->sb_state);
	fail_sb_state:
		FREE(self);
	return NULL;
}

nn_arch_t*
nn_arch_import(nn_engine_t* engine,
               size_t base_size, jsmn_val_t* val)
{
	ASSERT(engine);
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	// bs not required
	jsmn_val_t* val_learning_rate   = NULL;
	jsmn_val_t* val_momentum_decay  = NULL;
	jsmn_val_t* val_batch_momentum  = NULL;
	jsmn_val_t* val_l2_lambda       = NULL;
	jsmn_val_t* val_clip_max_weight = NULL;
	jsmn_val_t* val_clip_max_bias   = NULL;
	jsmn_val_t* val_clip_mu_inc     = NULL;
	jsmn_val_t* val_clip_mu_dec     = NULL;
	jsmn_val_t* val_clip_scale      = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		jsmn_keyval_t* kv;
		kv = (jsmn_keyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == JSMN_TYPE_PRIMITIVE)
		{
			if(strcmp(kv->key, "learning_rate") == 0)
			{
				val_learning_rate = kv->val;
			}
			else if(strcmp(kv->key, "momentum_decay") == 0)
			{
				val_momentum_decay = kv->val;
			}
			else if(strcmp(kv->key, "batch_momentum") == 0)
			{
				val_batch_momentum = kv->val;
			}
			else if(strcmp(kv->key, "l2_lambda") == 0)
			{
				val_l2_lambda = kv->val;
			}
			else if(strcmp(kv->key, "clip_max_weight") == 0)
			{
				val_clip_max_weight = kv->val;
			}
			else if(strcmp(kv->key, "clip_max_bias") == 0)
			{
				val_clip_max_bias = kv->val;
			}
			else if(strcmp(kv->key, "clip_mu_inc") == 0)
			{
				val_clip_mu_inc = kv->val;
			}
			else if(strcmp(kv->key, "clip_mu_dec") == 0)
			{
				val_clip_mu_dec = kv->val;
			}
			else if(strcmp(kv->key, "clip_scale") == 0)
			{
				val_clip_scale = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_learning_rate   == NULL) ||
	   (val_momentum_decay  == NULL) ||
	   (val_batch_momentum  == NULL) ||
	   (val_l2_lambda       == NULL) ||
	   (val_clip_max_weight == NULL) ||
	   (val_clip_max_bias   == NULL) ||
	   (val_clip_mu_inc     == NULL) ||
	   (val_clip_mu_dec     == NULL) ||
	   (val_clip_scale      == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_archState_t state =
	{
		.learning_rate   = strtof(val_learning_rate->data,   NULL),
		.momentum_decay  = strtof(val_momentum_decay->data,  NULL),
		.batch_momentum  = strtof(val_batch_momentum->data,  NULL),
		.l2_lambda       = strtof(val_l2_lambda->data,       NULL),
		.clip_max_weight = strtof(val_clip_max_weight->data, NULL),
		.clip_max_bias   = strtof(val_clip_max_bias->data,   NULL),
		.clip_mu_inc     = strtof(val_clip_mu_inc->data,     NULL),
		.clip_mu_dec     = strtof(val_clip_mu_dec->data,     NULL),
		.clip_scale      = strtof(val_clip_scale->data,     NULL),
	};

	return nn_arch_new(engine, base_size, &state);
}

int nn_arch_export(nn_arch_t* self, jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	nn_archState_t* state = &self->state;

	// bs not required
	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	ret &= jsmn_stream_key(stream, "%s", "learning_rate");
	ret &= jsmn_stream_float(stream, state->learning_rate);
	ret &= jsmn_stream_key(stream, "%s", "momentum_decay");
	ret &= jsmn_stream_float(stream, state->momentum_decay);
	ret &= jsmn_stream_key(stream, "%s", "batch_momentum");
	ret &= jsmn_stream_float(stream, state->batch_momentum);
	ret &= jsmn_stream_key(stream, "%s", "l2_lambda");
	ret &= jsmn_stream_float(stream, state->l2_lambda);
	ret &= jsmn_stream_key(stream, "%s", "clip_max_weight");
	ret &= jsmn_stream_float(stream, state->clip_max_weight);
	ret &= jsmn_stream_key(stream, "%s", "clip_max_bias");
	ret &= jsmn_stream_float(stream, state->clip_max_bias);
	ret &= jsmn_stream_key(stream, "%s", "clip_mu_inc");
	ret &= jsmn_stream_float(stream, state->clip_mu_inc);
	ret &= jsmn_stream_key(stream, "%s", "clip_mu_dec");
	ret &= jsmn_stream_float(stream, state->clip_mu_dec);
	ret &= jsmn_stream_key(stream, "%s", "clip_scale");
	ret &= jsmn_stream_float(stream, state->clip_scale);
	ret &= jsmn_stream_end(stream);

	return ret;
}

void nn_arch_delete(nn_arch_t** _self)
{
	ASSERT(_self);

	nn_arch_t* self = *_self;
	if(self)
	{
		nn_tensor_delete(&self->Yt);
		nn_tensor_delete(&self->X);
		cc_list_discard(self->layers);
		cc_list_delete(&self->layers);
		vkk_buffer_delete(&self->sb_state);
		FREE(self);
		*_self = NULL;
	}
}

int nn_arch_attachLayer(nn_arch_t* self,
                        nn_layer_t* layer)
{
	ASSERT(self);
	ASSERT(layer);

	if(self->loss)
	{
		LOGE("invalid");
		return 0;
	}

	// validate dimensions
	nn_layer_t* tail;
	tail = (nn_layer_t*) cc_list_peekTail(self->layers);
	if(tail)
	{
		if(nn_dim_equals(nn_layer_dimY(tail),
		                 nn_layer_dimX(layer)) == 0)
		{
			nn_dim_t* dimY = nn_layer_dimY(tail);
			nn_dim_t* dimX = nn_layer_dimX(layer);
			LOGE("invalid count=%u:%u, height=%u:%u, width=%u:%u, depth=%u:%u",
			     dimX->count,  dimY->count,
			     dimX->height, dimY->height,
			     dimX->width,  dimY->width,
			     dimX->depth,  dimY->depth);
			return 0;
		}
	}

	if(cc_list_append(self->layers, NULL, layer) == NULL)
	{
		return 0;
	}

	return 1;
}

int nn_arch_attachLoss(nn_arch_t* self,
                       nn_loss_t* loss)
{
	ASSERT(self);
	ASSERT(loss);

	if(self->loss)
	{
		LOGE("invalid");
		return 0;
	}

	// validate dimensions
	nn_layer_t* tail;
	tail = (nn_layer_t*) cc_list_peekTail(self->layers);
	if((tail == NULL) ||
	   (nn_dim_equals(nn_layer_dimY(tail),
	                  nn_loss_dimY(loss)) == 0))
	{
		LOGE("invalid");
		return 0;
	}

	self->loss = loss;

	return 1;
}

nn_tensor_t*
nn_arch_train(nn_arch_t* self, nn_layerMode_e layer_mode,
              uint32_t bs, nn_tensor_t* X,
              nn_tensor_t* Yt, nn_tensor_t* Y)
{
	// Y may be NULL
	ASSERT(self);
	ASSERT(X);
	ASSERT(Yt);

	// check if a loss function was specified
	if(self->loss == NULL)
	{
		LOGE("invalid");
		return 0;
	}

	if(nn_arch_beginCompute(self, bs, X, Yt) == 0)
	{
		return NULL;
	}

	// optionally replace X and Yt with compute tensors
	if(X->tensor_mode == NN_TENSOR_MODE_IO)
	{
		X = self->X;
	}
	if(Yt->tensor_mode == NN_TENSOR_MODE_IO)
	{
		Yt = self->Yt;
	}

	// perform forward pass for each batch
	cc_listIter_t* iter = cc_list_head(self->layers);
	while(iter)
	{
		nn_layer_t* layer;
		layer = (nn_layer_t*) cc_list_peekIter(iter);

		X = nn_layer_forwardPass(layer, layer_mode, bs, X);
		if(X == NULL)
		{
			goto fail_forwardPass;
		}

		iter = cc_list_next(iter);
	}

	// backpropagate loss
	nn_tensor_t* dL_dY = NULL;
	if(self->loss)
	{
		dL_dY = nn_loss_loss(self->loss, bs, X, Yt);
	}

	if(dL_dY == NULL)
	{
		goto fail_loss;
	}

	// perform backpropagation
	iter = cc_list_tail(self->layers);
	while(iter)
	{
		nn_layer_t* layer;
		layer = (nn_layer_t*) cc_list_peekIter(iter);

		dL_dY = nn_layer_backprop(layer, layer_mode, bs, dL_dY);
		if(dL_dY == NULL)
		{
			goto fail_backprop;
		}

		iter = cc_list_prev(iter);
	}

	nn_arch_endCompute(self);
	nn_arch_post(self, layer_mode);

	// optionally blit Y
	if(Y)
	{
		if(nn_tensor_blit(X, Y, bs, 0, 0) == 0)
		{
			return NULL;
		}
	}

	// success
	return dL_dY;

	// failure
	fail_backprop:
	fail_loss:
	fail_forwardPass:
		nn_arch_endCompute(self);
	return NULL;
}

float nn_arch_loss(nn_arch_t* self)
{
	ASSERT(self);

	if(self->loss)
	{
		return self->loss->loss;
	}

	return 0.0f;
}

int nn_arch_predict(nn_arch_t* self,
                    uint32_t bs,
                    nn_tensor_t* X,
                    nn_tensor_t* Y)
{
	ASSERT(self);
	ASSERT(X);
	ASSERT(Y);

	if(nn_arch_beginCompute(self, bs, X, NULL) == 0)
	{
		return 0;
	}

	// replace X with compute tensor
	if(X->tensor_mode == NN_TENSOR_MODE_IO)
	{
		X = self->X;
	}

	cc_listIter_t* iter = cc_list_head(self->layers);
	while(iter)
	{
		nn_layer_t* layer;
		layer = (nn_layer_t*) cc_list_peekIter(iter);

		X = nn_layer_forwardPass(layer,
		                         NN_LAYER_MODE_PREDICT,
		                         bs, X);
		if(X == NULL)
		{
			goto fail_forwardPass;
		}

		iter = cc_list_next(iter);
	}

	nn_arch_endCompute(self);
	nn_arch_post(self, NN_LAYER_MODE_PREDICT);

	// success
	return nn_tensor_blit(X, Y, bs, 0, 0);

	// failure
	fail_forwardPass:
		nn_arch_endCompute(self);
	return 0;
}
