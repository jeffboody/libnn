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

// Flag Usage
//
// NN_ARCH_FLAG_FP_BN (Forward Pass Batch Normalization)
// * AVERAGE: Inference pass uses running averages for
//            mean/variance and does not update the running
//            average.
// * INSTANCE: Inference pass performs instance
//             normalization which uses the test-batch for
//             mean/variance and does not update the running
//             average.
// Only one of AVERAGE/INSTANCE should be set. If neither
// flag is set then batch normalization defaults to training
// which updates running average but uses mini-batch for
// mean and variance.
//
// NN_ARCH_FLAG_BP_NOP (Backprop No Parameter Update)
// * Disable beta1t and beta2t Update
// * Disable Parameter Update
//
// NN_ARCH_FLAG_BP_STATS (Backprop Statistics)
// * Compute and log statistics during backprop
#define NN_ARCH_FLAG_FP_BN_RUNNING   0x0001
#define NN_ARCH_FLAG_FP_BN_INSTANCE  0x0002
#define NN_ARCH_FLAG_BP_NOP          0x0010
#define NN_ARCH_FLAG_BP_STATS        0x0020

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
	float adam_alpha;    // learning rate
	float adam_beta1;    // first moment decay rate
	float adam_beta2;    // second moment decay rate
	float adam_beta1t;   // beta1^t
	float adam_beta2t;   // beta2^t
	float adam_lambda;   // AdamW decoupled weight decay
	float adam_nu;       // AdamW schedule multiplier
	float bn_momentum;
} nn_archState_t;

typedef struct nn_arch_s
{
	nn_engine_t* engine;

	nn_archState_t state;

	// references
	cc_list_t* layers;

	vkk_buffer_t* sb100_bs;
	vkk_buffer_t* sb101_state;
} nn_arch_t;

nn_arch_t*      nn_arch_new(nn_engine_t* engine,
                            size_t base_size,
                            nn_archState_t* state);
void            nn_arch_delete(nn_arch_t** _self);
int             nn_arch_attachLayer(nn_arch_t* self,
                                    nn_layer_t* layer);
nn_arch_t*      nn_arch_import(nn_engine_t* engine,
                               size_t base_size,
                               jsmn_val_t* val);
int             nn_arch_export(nn_arch_t* self,
                               jsmn_stream_t* stream);
nn_dim_t*       nn_arch_dimX(nn_arch_t* self);
nn_dim_t*       nn_arch_dimY(nn_arch_t* self);
nn_archState_t* nn_arch_state(nn_arch_t* self);
nn_tensor_t*    nn_arch_forwardPass(nn_arch_t* self,
                                    int flags,
                                    uint32_t bs,
                                    nn_tensor_t* X);
nn_tensor_t*    nn_arch_backprop(nn_arch_t* self,
                                 int flags,
                                 uint32_t bs,
                                 nn_tensor_t* dL_dY);

#endif
