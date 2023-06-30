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
#include "nn_arch.h"
#include "nn_layer.h"
#include "nn_loss.h"
#include "nn_tensor.h"

#define NN_ARCH_THREADS 4

/***********************************************************
* public                                                   *
***********************************************************/

nn_arch_t* nn_arch_new(size_t base_size,
                       nn_archInfo_t* info)
{
	ASSERT(info);

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

	self->learning_rate  = info->learning_rate;
	self->momentum_decay = info->momentum_decay;
	self->batch_momentum = info->batch_momentum;
	self->l2_lambda      = info->l2_lambda;
	self->clip_max       = info->clip_max;
	self->clip_momentum  = info->clip_momentum;

	self->layers = cc_list_new();
	if(self->layers == NULL)
	{
		goto fail_layers;
	}

	cc_rngUniform_init(&self->rng_uniform);
	cc_rngNormal_init(&self->rng_normal, 0.0, 1.0);

	// success
	return self;

	// failure
	fail_layers:
		FREE(self);
	return NULL;
}

nn_arch_t*
nn_arch_import(size_t base_size, jsmn_val_t* val)
{
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	jsmn_val_t* val_learning_rate  = NULL;
	jsmn_val_t* val_momentum_decay = NULL;
	jsmn_val_t* val_batch_momentum = NULL;
	jsmn_val_t* val_l2_lambda      = NULL;
	jsmn_val_t* val_clip_max       = NULL;
	jsmn_val_t* val_clip_momentum  = NULL;

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
			else if(strcmp(kv->key, "clip_max") == 0)
			{
				val_clip_max = kv->val;
			}
			else if(strcmp(kv->key, "clip_momentum") == 0)
			{
				val_clip_momentum = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_learning_rate  == NULL) ||
	   (val_momentum_decay == NULL) ||
	   (val_batch_momentum == NULL) ||
	   (val_l2_lambda      == NULL) ||
	   (val_clip_max       == NULL) ||
	   (val_clip_momentum  == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_archInfo_t info =
	{
		.learning_rate  = strtof(val_learning_rate->data,  NULL),
		.momentum_decay = strtof(val_momentum_decay->data, NULL),
		.batch_momentum = strtof(val_batch_momentum->data, NULL),
		.l2_lambda      = strtof(val_l2_lambda->data,      NULL),
		.clip_max       = strtof(val_clip_max->data,       NULL),
		.clip_momentum  = strtof(val_clip_momentum->data,  NULL),
	};

	return nn_arch_new(base_size, &info);
}

int nn_arch_export(nn_arch_t* self, jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	ret &= jsmn_stream_key(stream, "%s", "learning_rate");
	ret &= jsmn_stream_float(stream, self->learning_rate);
	ret &= jsmn_stream_key(stream, "%s", "momentum_decay");
	ret &= jsmn_stream_float(stream, self->momentum_decay);
	ret &= jsmn_stream_key(stream, "%s", "batch_momentum");
	ret &= jsmn_stream_float(stream, self->batch_momentum);
	ret &= jsmn_stream_key(stream, "%s", "l2_lambda");
	ret &= jsmn_stream_float(stream, self->l2_lambda);
	ret &= jsmn_stream_key(stream, "%s", "clip_max");
	ret &= jsmn_stream_float(stream, self->clip_max);
	ret &= jsmn_stream_key(stream, "%s", "clip_momentum");
	ret &= jsmn_stream_float(stream, self->clip_momentum);
	ret &= jsmn_stream_end(stream);

	return ret;
}

void nn_arch_delete(nn_arch_t** _self)
{
	ASSERT(_self);

	nn_arch_t* self = *_self;
	if(self)
	{
		cc_list_discard(self->layers);
		cc_list_delete(&self->layers);
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

int nn_arch_train(nn_arch_t* self,
                  uint32_t bs,
                  nn_tensor_t* X,
                  nn_tensor_t* Yt)
{
	ASSERT(self);
	ASSERT(X);
	ASSERT(Yt);

	// perform forward pass for each batch
	nn_tensor_t*   Yi = X;
	cc_listIter_t* iter = cc_list_head(self->layers);
	while(iter)
	{
		nn_layer_t* layer;
		layer = (nn_layer_t*) cc_list_peekIter(iter);

		Yi = nn_layer_forwardPass(layer,
		                          NN_LAYER_MODE_TRAIN,
		                          bs, Yi);
		if(Yi == NULL)
		{
			return 0;
		}

		iter = cc_list_next(iter);
	}

	// backpropagate loss
	nn_tensor_t* dL_dY = NULL;
	{
		nn_loss_fn loss_fn;
		loss_fn = self->loss->loss_fn;

		dL_dY = (*loss_fn)(self->loss, bs, Yi, Yt);
		if(dL_dY == NULL)
		{
			return 0;
		}
	}

	// perform backpropagation
	iter = cc_list_tail(self->layers);
	while(iter)
	{
		nn_layer_t* layer;
		layer = (nn_layer_t*) cc_list_peekIter(iter);

		dL_dY = nn_layer_backprop(layer, bs, dL_dY);
		if(dL_dY == NULL)
		{
			return 0;
		}

		iter = cc_list_prev(iter);
	}

	return 1;
}

float nn_arch_loss(nn_arch_t* self)
{
	ASSERT(self);

	return self->loss->loss;
}

int nn_arch_predict(nn_arch_t* self,
                    nn_tensor_t* X,
                    nn_tensor_t* Y)
{
	ASSERT(self);
	ASSERT(X);
	ASSERT(Y);

	nn_tensor_t*   Yi   = X;
	cc_listIter_t* iter = cc_list_head(self->layers);
	while(iter)
	{
		nn_layer_t* layer;
		layer = (nn_layer_t*) cc_list_peekIter(iter);

		Yi = nn_layer_forwardPass(layer,
		                          NN_LAYER_MODE_PREDICT,
		                          1, Yi);
		if(Yi == NULL)
		{
			return 0;
		}

		iter = cc_list_next(iter);
	}

	return nn_tensor_blit(Yi, Y, 0, 0);
}
