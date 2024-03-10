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

#include "../jsmn/wrapper/jsmn_stream.h"
#include "../libcc/cc_list.h"
#include "../libvkk/vkk.h"
#include "nn.h"

// Recommended Defaults
// adam_alpha:  0.0001f
// adam_beta1:  0.9f
// adam_beta2:  0.999f
// adam_beta1t: 1.0f
// adam_beta2t: 1.0f
// adam_lambda: s*0.001f, s=0.25f=[0.0f...2.0f]
// adam_nu:     1.0f
// bn_momentum: 0.99f
//
// See "Decoupled Weight Decay Regularization"
// for the definition of Adam parameters
// https://arxiv.org/pdf/1711.05101.pdf
typedef struct nn_archState_s
{
	// TODO - remove bs
	uint32_t bs;
	float    adam_alpha;    // learning rate
	float    adam_beta1;    // first moment decay rate
	float    adam_beta2;    // second moment decay rate
	float    adam_beta1t;   // beta1^t
	float    adam_beta2t;   // beta2^t
	float    adam_lambda;   // AdamW decoupled weight decay
	float    adam_nu;       // AdamW schedule multiplier
	float    bn_momentum;
} nn_archState_t;

typedef struct nn_arch_s
{
	nn_engine_t* engine;

	// TODO - remove sb00_state
	nn_archState_t state;
	vkk_buffer_t*  sb100_bs;
	vkk_buffer_t*  sb101_state;
	vkk_buffer_t*  sb00_state;

	// references
	cc_list_t* layers;
	nn_loss_t* loss;

	// cached output reference
	// see NN_LAYER_FLAG_BACKPROP_NOP
	nn_tensor_t* O;

	// compute tensors below are allocated on demand

	// default NN
	nn_tensor_t* X;
	nn_tensor_t* Yt;
} nn_arch_t;

nn_arch_t*   nn_arch_new(nn_engine_t* engine,
                         size_t base_size,
                         nn_archState_t* state);
nn_arch_t*   nn_arch_import(nn_engine_t* engine,
                            size_t base_size,
                            jsmn_val_t* val);
int          nn_arch_export(nn_arch_t* self,
                            jsmn_stream_t* stream);
void         nn_arch_delete(nn_arch_t** _self);
int          nn_arch_attachLayer(nn_arch_t* self,
                                 nn_layer_t* layer);
int          nn_arch_attachLoss(nn_arch_t* self,
                                nn_loss_t* loss);
nn_tensor_t* nn_arch_train(nn_arch_t* self,
                           int flags,
                           uint32_t bs,
                           nn_tensor_t* X,
                           nn_tensor_t* Yt,
                           nn_tensor_t* Y);
float        nn_arch_loss(nn_arch_t* self);
int          nn_arch_predict(nn_arch_t* self,
                             uint32_t bs,
                             nn_tensor_t* X,
                             nn_tensor_t* Y);

#endif
