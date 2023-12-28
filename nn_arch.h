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

// See "Decoupled Weight Decay Regularization" for the
// definition of SGD and ADAM parameters
// https://arxiv.org/pdf/1711.05101.pdf
typedef struct nn_archState_s
{
	uint32_t bs;
	float    sgd_alpha;     // learning rate
	float    sgd_beta1;     // momentum factor
	float    sgd_l2_lambda; // L2 regularization
	float    bn_momentum;
	float    gan_blend_factor;
	float    gan_blend_scalar;
	float    gan_blend_min;
	float    gan_blend_max;
	float    lerp_s;
	float    lerp_min;
	float    lerp_max;
} nn_archState_t;

typedef struct nn_arch_s
{
	nn_engine_t* engine;

	nn_archState_t state;
	vkk_buffer_t*  sb_state;

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

	// Fair cGAN
	nn_tensor_t* Xd;
	nn_tensor_t* Cg0;
	nn_tensor_t* Cg1;
	nn_tensor_t* Cr0;
	nn_tensor_t* Cr1;
	nn_tensor_t* Ytg;
	nn_tensor_t* Ytr;
	nn_tensor_t* dL_dYb;

	vkk_uniformSet_t* us0;
	vkk_uniformSet_t* us1;
	vkk_uniformSet_t* us2;
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
nn_tensor_t* nn_arch_trainLERP(nn_arch_t* self,
                               nn_arch_t* lerp,
                               uint32_t bs,
                               nn_tensor_t* Xt,
                               nn_tensor_t* Yt,
                               nn_tensor_t* X,
                               nn_tensor_t* Y);
nn_tensor_t* nn_arch_trainFairCGAN(nn_arch_t* G,
                                   nn_arch_t* D,
                                   uint32_t bs,
                                   nn_tensor_t* Cg0,
                                   nn_tensor_t* Cg1,
                                   nn_tensor_t* Cr0,
                                   nn_tensor_t* Cr1,
                                   nn_tensor_t* Ytg,
                                   nn_tensor_t* Ytr,
                                   nn_tensor_t* Yt11,
                                   nn_tensor_t* Yt10,
                                   nn_tensor_t* dL_dYb,
                                   nn_tensor_t* dL_dYg,
                                   nn_tensor_t* dL_dYd,
                                   nn_tensor_t* Yg,
                                   nn_tensor_t* Yd,
                                   float* loss,
                                   float* g_loss,
                                   float* d_loss);
float        nn_arch_loss(nn_arch_t* self);
int          nn_arch_predict(nn_arch_t* self,
                             uint32_t bs,
                             nn_tensor_t* X,
                             nn_tensor_t* Y);

#endif
