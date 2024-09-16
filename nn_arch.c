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
nn_arch_post(nn_arch_t* self, int flags, uint32_t bs)
{
	ASSERT(self);

	cc_listIter_t* iter = cc_list_head(self->layers);
	while(iter)
	{
		nn_layer_t* layer;
		layer = (nn_layer_t*) cc_list_peekIter(iter);

		nn_layer_post(layer, flags, bs);

		iter = cc_list_next(iter);
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

	self->layers = cc_list_new();
	if(self->layers == NULL)
	{
		goto fail_layers;
	}

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

	// success
	return self;

	// failure
	fail_sb101_state:
		vkk_buffer_delete(&self->sb100_bs);
	fail_sb100_bs:
		cc_list_delete(&self->layers);
	fail_layers:
		FREE(self);
	return NULL;
}

void nn_arch_delete(nn_arch_t** _self)
{
	ASSERT(_self);

	nn_arch_t* self = *_self;
	if(self)
	{
		vkk_buffer_delete(&self->sb101_state);
		vkk_buffer_delete(&self->sb100_bs);
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

nn_arch_t*
nn_arch_import(nn_engine_t* engine,
               size_t base_size, cc_jsmnVal_t* val)
{
	ASSERT(engine);
	ASSERT(val);

	if(val->type != CC_JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	// bs not required
	cc_jsmnVal_t* val_adam_alpha  = NULL;
	cc_jsmnVal_t* val_adam_beta1  = NULL;
	cc_jsmnVal_t* val_adam_beta2  = NULL;
	cc_jsmnVal_t* val_adam_beta1t = NULL;
	cc_jsmnVal_t* val_adam_beta2t = NULL;
	cc_jsmnVal_t* val_bn_momentum = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		cc_jsmnKeyval_t* kv;
		kv = (cc_jsmnKeyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == CC_JSMN_TYPE_PRIMITIVE)
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
		.bn_momentum = strtof(val_bn_momentum->data, NULL),
	};

	return nn_arch_new(engine, base_size, &state);
}

int nn_arch_export(nn_arch_t* self,
                   cc_jsmnStream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	nn_archState_t* state = &self->state;

	// bs not required
	int ret = 1;
	ret &= cc_jsmnStream_beginObject(stream);
	ret &= cc_jsmnStream_key(stream, "%s", "adam_alpha");
	ret &= cc_jsmnStream_float(stream, state->adam_alpha);
	ret &= cc_jsmnStream_key(stream, "%s", "adam_beta1");
	ret &= cc_jsmnStream_float(stream, state->adam_beta1);
	ret &= cc_jsmnStream_key(stream, "%s", "adam_beta2");
	ret &= cc_jsmnStream_float(stream, state->adam_beta2);
	ret &= cc_jsmnStream_key(stream, "%s", "adam_beta1t");
	ret &= cc_jsmnStream_float(stream, state->adam_beta1t);
	ret &= cc_jsmnStream_key(stream, "%s", "adam_beta2t");
	ret &= cc_jsmnStream_float(stream, state->adam_beta2t);
	ret &= cc_jsmnStream_key(stream, "%s", "bn_momentum");
	ret &= cc_jsmnStream_float(stream, state->bn_momentum);
	ret &= cc_jsmnStream_end(stream);

	return ret;
}

nn_dim_t* nn_arch_dimX(nn_arch_t* self)
{
	ASSERT(self);

	nn_layer_t* layer;
	layer = (nn_layer_t*) cc_list_peekHead(self->layers);
	if(layer == NULL)
	{
		LOGE("invalid");
		return NULL;
	}

	return nn_layer_dimX(layer);
}

nn_dim_t* nn_arch_dimY(nn_arch_t* self)
{
	ASSERT(self);

	nn_layer_t* layer;
	layer = (nn_layer_t*) cc_list_peekTail(self->layers);
	if(layer == NULL)
	{
		LOGE("invalid");
		return NULL;
	}

	return nn_layer_dimY(layer);
}

nn_archState_t* nn_arch_state(nn_arch_t* self)
{
	ASSERT(self);

	return &self->state;
}

nn_tensor_t*
nn_arch_forwardPass(nn_arch_t* self,
                    int flags, uint32_t bs,
                    nn_tensor_t* X)
{
	ASSERT(self);
	ASSERT(X);

	if(nn_tensor_mode(X) != NN_TENSOR_MODE_COMPUTE)
	{
		LOGE("invalid");
		return NULL;
	}

	nn_archState_t* state = &self->state;
	vkk_buffer_writeStorage(self->sb100_bs, 0,
	                        sizeof(uint32_t), &bs);
	vkk_buffer_writeStorage(self->sb101_state, 0,
	                        sizeof(nn_archState_t), state);

	if(nn_engine_computeBegin(self->engine) == 0)
	{
		return NULL;
	}

	// perform forward pass
	cc_listIter_t* iter = cc_list_head(self->layers);
	while(iter)
	{
		nn_layer_t* layer;
		layer = (nn_layer_t*) cc_list_peekIter(iter);

		X = nn_layer_computeFp(layer, flags, bs, X);
		if(X == NULL)
		{
			goto fail_forwardPass;
		}

		iter = cc_list_next(iter);
	}

	nn_engine_computeEnd(self->engine);
	nn_arch_post(self, flags, bs);

	// success
	return X;

	// failure
	fail_forwardPass:
		nn_engine_computeEnd(self->engine);
	return NULL;
}

nn_tensor_t*
nn_arch_backprop(nn_arch_t* self,
                 int flags, uint32_t bs,
                 nn_tensor_t* dL_dY)
{
	ASSERT(self);
	ASSERT(dL_dY);

	if(nn_tensor_mode(dL_dY) != NN_TENSOR_MODE_COMPUTE)
	{
		LOGE("invalid");
		return NULL;
	}

	nn_archState_t* state = &self->state;
	if((flags & NN_ARCH_FLAG_BP_NOP) == 0)
	{
		state->adam_beta1t *= state->adam_beta1;
		state->adam_beta2t *= state->adam_beta2;
	}
	vkk_buffer_writeStorage(self->sb100_bs, 0,
	                        sizeof(uint32_t), &bs);
	vkk_buffer_writeStorage(self->sb101_state, 0,
	                        sizeof(nn_archState_t), state);

	if(nn_engine_computeBegin(self->engine) == 0)
	{
		return NULL;
	}

	// perform backprop
	cc_listIter_t* iter = cc_list_tail(self->layers);
	while(iter)
	{
		nn_layer_t* layer;
		layer = (nn_layer_t*) cc_list_peekIter(iter);

		dL_dY = nn_layer_computeBp(layer, flags, bs, dL_dY);
		if(dL_dY == NULL)
		{
			goto fail_backprop;
		}

		iter = cc_list_prev(iter);
	}

	nn_engine_computeEnd(self->engine);
	nn_arch_post(self, flags, bs);

	// success
	return dL_dY;

	// failure
	fail_backprop:
		nn_engine_computeEnd(self->engine);
	return NULL;
}
