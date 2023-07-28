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
#include "../jsmn/wrapper/jsmn_wrapper.h"
#include "../libcc/rng/cc_rngNormal.h"
#include "../libcc/rng/cc_rngUniform.h"
#include "../libcc/cc_list.h"
#include "nn.h"

#ifdef NN_USE_COMPUTE
#include "../libvkk/vkk.h"
#else
// dummy type for vkk_engine_t
typedef int vkk_engine_t;
#endif

typedef struct nn_archState_s
{
	uint32_t bs;
	float    learning_rate;
	float    momentum_decay;
	float    batch_momentum;
	float    l2_lambda;
	float    clip_max;
	float    clip_momentum;
} nn_archState_t;

typedef struct nn_arch_s
{
	nn_archState_t state;

	// neural network (references)
	cc_list_t* layers;
	nn_loss_t* loss;

	// random number generators
	cc_rngUniform_t rng_uniform;
	cc_rngNormal_t  rng_normal;

	#ifdef NN_USE_COMPUTE
	vkk_engine_t* engine;

	vkk_compute_t* compute;

	vkk_uniformSetFactory_t* usf0_batchNorm;
	vkk_uniformSetFactory_t* usf1_batchNorm;
	vkk_uniformSetFactory_t* usf2_batchNorm;
	vkk_uniformSetFactory_t* usf3_batchNorm;
	vkk_uniformSetFactory_t* usf0_conv;
	vkk_uniformSetFactory_t* usf1_conv;
	vkk_uniformSetFactory_t* usf2_conv;
	vkk_uniformSetFactory_t* usf3_conv;
	vkk_uniformSetFactory_t* usf0_fact;
	vkk_uniformSetFactory_t* usf1_fact;
	vkk_uniformSetFactory_t* usf2_fact;
	vkk_uniformSetFactory_t* usf0_pooling;
	vkk_uniformSetFactory_t* usf1_pooling;
	vkk_uniformSetFactory_t* usf2_pooling;
	vkk_uniformSetFactory_t* usf0_skip;
	vkk_uniformSetFactory_t* usf1_skip;
	vkk_uniformSetFactory_t* usf0_weight;
	vkk_uniformSetFactory_t* usf1_weight;
	vkk_uniformSetFactory_t* usf2_weight;
	vkk_uniformSetFactory_t* usf0_loss;
	vkk_uniformSetFactory_t* usf0_tensor;

	vkk_pipelineLayout_t* pl_batchNorm;
	vkk_pipelineLayout_t* pl_conv;
	vkk_pipelineLayout_t* pl_fact;
	vkk_pipelineLayout_t* pl_pooling;
	vkk_pipelineLayout_t* pl_skip;
	vkk_pipelineLayout_t* pl_weight;
	vkk_pipelineLayout_t* pl_loss;
	vkk_pipelineLayout_t* pl_tensor;

	vkk_computePipeline_t* cp_batchNorm_forwardPassXmean;
	vkk_computePipeline_t* cp_batchNorm_forwardPassXvar;
	vkk_computePipeline_t* cp_batchNorm_forwardPassXhat;
	vkk_computePipeline_t* cp_batchNorm_forwardPassY;
	vkk_computePipeline_t* cp_batchNorm_backprop_dL_dX;
	vkk_computePipeline_t* cp_batchNorm_backprop_dL_dXhat;
	vkk_computePipeline_t* cp_batchNorm_backpropSum;
	vkk_computePipeline_t* cp_conv_forwardPass;
	vkk_computePipeline_t* cp_conv_forwardPassT;
	vkk_computePipeline_t* cp_conv_backprop_dL_dX;
	vkk_computePipeline_t* cp_conv_backprop_dL_dW;
	vkk_computePipeline_t* cp_conv_backprop_dL_dB;
	vkk_computePipeline_t* cp_conv_backpropT_dL_dX;
	vkk_computePipeline_t* cp_conv_backpropT_dL_dW;
	vkk_computePipeline_t* cp_conv_backpropGradientClipping;
	vkk_computePipeline_t* cp_conv_backpropUpdateW;
	vkk_computePipeline_t* cp_conv_backpropUpdateB;
	vkk_computePipeline_t* cp_fact_forwardPassLinear;
	vkk_computePipeline_t* cp_fact_forwardPassLogistic;
	vkk_computePipeline_t* cp_fact_forwardPassReLU;
	vkk_computePipeline_t* cp_fact_forwardPassPReLU;
	vkk_computePipeline_t* cp_fact_forwardPassTanh;
	vkk_computePipeline_t* cp_fact_backpropLinear;
	vkk_computePipeline_t* cp_fact_backpropLogistic;
	vkk_computePipeline_t* cp_fact_backpropReLU;
	vkk_computePipeline_t* cp_fact_backpropPReLU;
	vkk_computePipeline_t* cp_fact_backpropTanh;
	vkk_computePipeline_t* cp_pooling_forwardPassAvg;
	vkk_computePipeline_t* cp_pooling_forwardPassMax;
	vkk_computePipeline_t* cp_pooling_backprop;
	vkk_computePipeline_t* cp_skip_forwardPassAdd;
	vkk_computePipeline_t* cp_skip_forwardPassCat;
	vkk_computePipeline_t* cp_skip_backpropCat;
	vkk_computePipeline_t* cp_skip_backpropFork;
	vkk_computePipeline_t* cp_weight_forwardPass;
	vkk_computePipeline_t* cp_weight_backpropGradientClipping;
	vkk_computePipeline_t* cp_weight_backpropUpdateW;
	vkk_computePipeline_t* cp_weight_backpropUpdateB;
	vkk_computePipeline_t* cp_weight_backprop_dL_dX;
	vkk_computePipeline_t* cp_weight_backprop_dL_dW;
	vkk_computePipeline_t* cp_weight_backprop_dL_dB;
	vkk_computePipeline_t* cp_loss_dL_dY_mse;
	vkk_computePipeline_t* cp_loss_dL_dY_mae;
	vkk_computePipeline_t* cp_loss_dL_dY_bce;
	vkk_computePipeline_t* cp_loss_mse;
	vkk_computePipeline_t* cp_loss_mae;
	vkk_computePipeline_t* cp_loss_bce;
	vkk_computePipeline_t* cp_tensor_clear;
	vkk_computePipeline_t* cp_tensor_clearAligned;

	vkk_buffer_t* sb00_state;

	nn_tensor_t* X;
	nn_tensor_t* Yt;
	#endif
} nn_arch_t;

nn_arch_t* nn_arch_new(vkk_engine_t* engine,
                       size_t base_size,
                       nn_archState_t* state);
nn_arch_t* nn_arch_import(vkk_engine_t* engine,
                          size_t base_size,
                          jsmn_val_t* val);
int        nn_arch_export(nn_arch_t* self,
                          jsmn_stream_t* stream);
void       nn_arch_delete(nn_arch_t** _self);
int        nn_arch_attachLayer(nn_arch_t* self,
                               nn_layer_t* layer);
int        nn_arch_attachLoss(nn_arch_t* self,
                              nn_loss_t* loss);
int        nn_arch_train(nn_arch_t* self,
                         uint32_t bs,
                         nn_tensor_t* X,
                         nn_tensor_t* Yt);
float      nn_arch_loss(nn_arch_t* self);
int        nn_arch_predict(nn_arch_t* self,
                           nn_tensor_t* X,
                           nn_tensor_t* Y);

#endif
