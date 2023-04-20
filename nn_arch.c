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

	self->max_batch_size = info->max_batch_size;
	self->learning_rate  = info->learning_rate;
	self->momentum_decay = info->momentum_decay;
	self->batch_momentum = info->batch_momentum;
	self->l2_lambda      = info->l2_lambda;

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

	self->loss = loss;

	return 1;
}

int nn_arch_train(nn_arch_t* self,
                  uint32_t batch_size,
                  nn_tensor_t* X,
                  nn_tensor_t* Yt)
{
	ASSERT(self);
	ASSERT(X);
	ASSERT(Yt);

	self->batch_size = batch_size;

	// perform forward pass for each batch
	nn_tensor_t*   Yi = X;
	cc_listIter_t* iter = cc_list_head(self->layers);
	while(iter)
	{
		nn_layer_t* layer;
		layer = (nn_layer_t*) cc_list_peekIter(iter);

		nn_layer_forwardPassFn forward_pass_fn;
		forward_pass_fn = layer->forward_pass_fn;

		Yi = (*forward_pass_fn)(layer, Yi);
		if(Yi == NULL)
		{
			return 0;
		}

		iter = cc_list_next(iter);
	}

	// backpropagate loss
	nn_tensor_t* dL_dY = NULL;
	{
		nn_loss_backpropFn backprop_fn;
		backprop_fn = self->loss->backprop_fn;

		dL_dY = (*backprop_fn)(self->loss, Yi, Yt);
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

		nn_layer_backpropFn backprop_fn;
		backprop_fn = layer->backprop_fn;

		dL_dY = (*backprop_fn)(layer, dL_dY);
		if(dL_dY == NULL)
		{
			return 0;
		}

		iter = cc_list_prev(iter);
	}

	return 1;
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

		nn_layer_forwardPassFn forward_pass_fn;
		forward_pass_fn = layer->forward_pass_fn;

		Yi = (*forward_pass_fn)(layer, Yi);
		if(Yi == NULL)
		{
			return 0;
		}

		iter = cc_list_next(iter);
	}

	return nn_tensor_blit(Yi, Y, 0, 0);
}