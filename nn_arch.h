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

#ifndef nn_arch_H
#define nn_arch_H

#include "../libcc/rng/cc_rngNormal.h"
#include "../libcc/rng/cc_rngUniform.h"
#include "../libcc/cc_list.h"
#include "nn.h"

typedef struct nn_archInfo_s
{
	// hyperparameters
	uint32_t max_batch_size;
	float    learning_rate;
	float    momentum_decay;
	float    batch_momentum;
	float    l2_lambda;
} nn_archInfo_t;

typedef struct nn_arch_s
{
	// hyperparameters
	uint32_t max_batch_size;
	uint32_t batch_size;
	float    learning_rate;
	float    momentum_decay;
	float    batch_momentum;
	float    l2_lambda;

	// neural network (references)
	cc_list_t* layers;
	nn_loss_t* loss;

	// random number generators
	cc_rngUniform_t rng_uniform;
	cc_rngNormal_t  rng_normal;
} nn_arch_t;

nn_arch_t* nn_arch_new(size_t base_size,
                       nn_archInfo_t* info);
void       nn_arch_delete(nn_arch_t** _self);
int        nn_arch_attachLayer(nn_arch_t* self,
                               nn_layer_t* layer);
int        nn_arch_attachLoss(nn_arch_t* self,
                              nn_loss_t* loss);
int        nn_arch_train(nn_arch_t* self,
                         uint32_t batch_size,
                         nn_tensor_t* X,
                         nn_tensor_t* Yt);
int        nn_arch_predict(nn_arch_t* self,
                           nn_tensor_t* X,
                           nn_tensor_t* Y);

#endif
