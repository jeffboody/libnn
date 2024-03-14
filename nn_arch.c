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
#include "../libcc/math/cc_float.h"
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
nn_arch_post(nn_arch_t* self, int flags)
{
	ASSERT(self);

	cc_listIter_t* iter = cc_list_head(self->layers);
	while(iter)
	{
		nn_layer_t* layer;
		layer = (nn_layer_t*) cc_list_peekIter(iter);

		nn_layer_post(layer, flags);

		iter = cc_list_next(iter);
	}

	if(self->loss)
	{
		nn_loss_post(self->loss, flags);
	}
}

static int
nn_arch_init(nn_arch_t* self,
             uint32_t bs,
             nn_tensor_t* X,
             nn_tensor_t* Yt)
{
	// X and Yt may be NULL
	ASSERT(self);

	nn_engine_t* engine = self->engine;

	// optionally create X
	if(X && (X->mode == NN_TENSOR_MODE_IO))
	{
		if(self->X)
		{
			if(nn_dim_sizeEquals(nn_tensor_dim(self->X),
			                     nn_tensor_dim(X)) == 0)
			{
				nn_tensor_delete(&self->X);
			}
		}

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

		if(nn_tensor_copy(X, self->X, 0, 0, bs) == 0)
		{
			return 0;
		}
	}

	// optionally create Yt
	if(Yt && (Yt->mode == NN_TENSOR_MODE_IO))
	{
		if(self->Yt)
		{
			if(nn_dim_sizeEquals(nn_tensor_dim(self->Yt),
			                     nn_tensor_dim(Yt)) == 0)
			{
				nn_tensor_delete(&self->Yt);
			}
		}

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

		if(nn_tensor_copy(Yt, self->Yt, 0, 0, bs) == 0)
		{
			return 0;
		}
	}

	// update global state
	nn_archState_t* state = &self->state;
	if(Yt)
	{
		state->adam_beta1t *= state->adam_beta1;
		state->adam_beta2t *= state->adam_beta2;
	}
	vkk_buffer_writeStorage(self->sb100_bs, 0,
	                        sizeof(uint32_t), &bs);
	vkk_buffer_writeStorage(self->sb101_state, 0,
	                        sizeof(nn_archState_t), state);

	return nn_engine_begin(engine);
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

	self->sb100_bs = vkk_buffer_new(engine->engine, um,
	                                VKK_BUFFER_USAGE_STORAGE,
	                                sizeof(uint32_t),
	                                NULL);
	if(self->sb100_bs == NULL)
	{
		goto fail_sb100_bs;
	}

	self->sb101_state = vkk_buffer_new(engine->engine, um,
	                                   VKK_BUFFER_USAGE_STORAGE,
	                                   sizeof(nn_archState_t),
	                                   NULL);
	if(self->sb101_state == NULL)
	{
		goto fail_sb101_state;
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
		vkk_buffer_delete(&self->sb101_state);
	fail_sb101_state:
		vkk_buffer_delete(&self->sb100_bs);
	fail_sb100_bs:
		FREE(self);
	return NULL;
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
		vkk_buffer_delete(&self->sb101_state);
		vkk_buffer_delete(&self->sb100_bs);
		FREE(self);
		*_self = NULL;
	}
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
	jsmn_val_t* val_adam_alpha  = NULL;
	jsmn_val_t* val_adam_beta1  = NULL;
	jsmn_val_t* val_adam_beta2  = NULL;
	jsmn_val_t* val_adam_beta1t = NULL;
	jsmn_val_t* val_adam_beta2t = NULL;
	jsmn_val_t* val_adam_lambda = NULL;
	jsmn_val_t* val_adam_nu     = NULL;
	jsmn_val_t* val_bn_momentum = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		jsmn_keyval_t* kv;
		kv = (jsmn_keyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == JSMN_TYPE_PRIMITIVE)
		{
			if(strcmp(kv->key, "adam_alpha") == 0)
			{
				val_adam_alpha = kv->val;
			}
			else if(strcmp(kv->key, "adam_beta1") == 0)
			{
				val_adam_beta1 = kv->val;
			}
			else if(strcmp(kv->key, "adam_beta2") == 0)
			{
				val_adam_beta2 = kv->val;
			}
			else if(strcmp(kv->key, "adam_beta1t") == 0)
			{
				val_adam_beta1t = kv->val;
			}
			else if(strcmp(kv->key, "adam_beta2t") == 0)
			{
				val_adam_beta2t = kv->val;
			}
			else if(strcmp(kv->key, "adam_lambda") == 0)
			{
				val_adam_lambda = kv->val;
			}
			else if(strcmp(kv->key, "adam_nu") == 0)
			{
				val_adam_nu = kv->val;
			}
			else if(strcmp(kv->key, "bn_momentum") == 0)
			{
				val_bn_momentum = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_adam_alpha  == NULL) ||
	   (val_adam_beta1  == NULL) ||
	   (val_adam_beta2  == NULL) ||
	   (val_adam_beta1t == NULL) ||
	   (val_adam_beta2t == NULL) ||
	   (val_adam_lambda == NULL) ||
	   (val_adam_nu     == NULL) ||
	   (val_bn_momentum == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_archState_t state =
	{
		.adam_alpha  = strtof(val_adam_alpha->data,  NULL),
		.adam_beta1  = strtof(val_adam_beta1->data,  NULL),
		.adam_beta2  = strtof(val_adam_beta2->data,  NULL),
		.adam_beta1t = strtof(val_adam_beta1t->data, NULL),
		.adam_beta2t = strtof(val_adam_beta2t->data, NULL),
		.adam_lambda = strtof(val_adam_lambda->data, NULL),
		.adam_nu     = strtof(val_adam_nu->data,     NULL),
		.bn_momentum = strtof(val_bn_momentum->data, NULL),
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
	ret &= jsmn_stream_key(stream, "%s", "adam_alpha");
	ret &= jsmn_stream_float(stream, state->adam_alpha);
	ret &= jsmn_stream_key(stream, "%s", "adam_beta1");
	ret &= jsmn_stream_float(stream, state->adam_beta1);
	ret &= jsmn_stream_key(stream, "%s", "adam_beta2");
	ret &= jsmn_stream_float(stream, state->adam_beta2);
	ret &= jsmn_stream_key(stream, "%s", "adam_beta1t");
	ret &= jsmn_stream_float(stream, state->adam_beta1t);
	ret &= jsmn_stream_key(stream, "%s", "adam_beta2t");
	ret &= jsmn_stream_float(stream, state->adam_beta2t);
	ret &= jsmn_stream_key(stream, "%s", "adam_lambda");
	ret &= jsmn_stream_float(stream, state->adam_lambda);
	ret &= jsmn_stream_key(stream, "%s", "adam_nu");
	ret &= jsmn_stream_float(stream, state->adam_nu);
	ret &= jsmn_stream_key(stream, "%s", "bn_momentum");
	ret &= jsmn_stream_float(stream, state->bn_momentum);
	ret &= jsmn_stream_end(stream);

	return ret;
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
		nn_dim_t* dimY = nn_layer_dimY(tail);
		nn_dim_t* dimX = nn_layer_dimX(layer);
		if(nn_dim_sizeEquals(dimY, dimX) == 0)
		{
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
	   (nn_dim_sizeEquals(nn_layer_dimY(tail),
	                      nn_loss_dimY(loss)) == 0))
	{
		LOGE("invalid");
		return 0;
	}

	self->loss = loss;

	return 1;
}

nn_tensor_t*
nn_arch_train(nn_arch_t* self, int flags,
              uint32_t bs, nn_tensor_t* X,
              nn_tensor_t* Yt, nn_tensor_t* Y)
{
	// X and Y may be NULL
	ASSERT(self);
	ASSERT(flags & NN_LAYER_FLAG_BACKPROP);
	ASSERT(Yt);

	if(nn_arch_init(self, bs, X, Yt) == 0)
	{
		return NULL;
	}

	cc_listIter_t* iter;
	if(flags & NN_LAYER_FLAG_FORWARD_PASS)
	{
		ASSERT(X);

		// optionally replace X with compute tensor
		if(X->mode == NN_TENSOR_MODE_IO)
		{
			X = self->X;
		}

		// perform forward pass
		iter = cc_list_head(self->layers);
		while(iter)
		{
			nn_layer_t* layer;
			layer = (nn_layer_t*) cc_list_peekIter(iter);

			X = nn_layer_forwardPass(layer, flags, bs, X);
			if(X == NULL)
			{
				goto fail_forwardPass;
			}

			iter = cc_list_next(iter);
		}
		self->O = X;
	}
	else
	{
		ASSERT(self->O);

		// see NN_LAYER_FLAG_BACKPROP_NOP
		X = self->O;
	}

	// optionally replace Yt with compute tensor
	if(Yt->mode == NN_TENSOR_MODE_IO)
	{
		Yt = self->Yt;
	}

	// compute loss
	nn_tensor_t* dL_dY;
	dL_dY = nn_loss_loss(self->loss, bs, X, Yt);
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

		dL_dY = nn_layer_backprop(layer, flags, bs, dL_dY);
		if(dL_dY == NULL)
		{
			goto fail_backprop;
		}

		iter = cc_list_prev(iter);
	}

	nn_engine_end(self->engine);
	nn_arch_post(self, flags);

	// optionally copy Y
	if(Y)
	{
		if(nn_tensor_copy(X, Y, 0, 0, bs) == 0)
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
		nn_engine_end(self->engine);
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

	if(nn_arch_init(self, bs, X, NULL) == 0)
	{
		return 0;
	}

	// replace X with compute tensor
	if(X->mode == NN_TENSOR_MODE_IO)
	{
		X = self->X;
	}

	cc_listIter_t* iter = cc_list_head(self->layers);
	while(iter)
	{
		nn_layer_t* layer;
		layer = (nn_layer_t*) cc_list_peekIter(iter);

		X = nn_layer_forwardPass(layer,
		                         NN_LAYER_FLAG_FORWARD_PASS,
		                         bs, X);
		if(X == NULL)
		{
			goto fail_forwardPass;
		}

		iter = cc_list_next(iter);
	}
	self->O = X;

	nn_engine_end(self->engine);
	nn_arch_post(self, NN_LAYER_FLAG_FORWARD_PASS);

	// success
	return nn_tensor_copy(X, Y, 0, 0, bs);

	// failure
	fail_forwardPass:
		nn_engine_end(self->engine);
	return 0;
}
