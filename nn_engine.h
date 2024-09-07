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

#ifndef nn_engine_H
#define nn_engine_H

#include "../libcc/rng/cc_rngNormal.h"
#include "../libcc/rng/cc_rngUniform.h"
#include "../libcc/cc_list.h"
#include "../libcc/cc_map.h"
#include "../libvkk/vkk.h"
#include "nn.h"

typedef struct nn_engine_s
{
	vkk_engine_t* engine;

	int dispatch;

	cc_rngUniform_t rng_uniform;
	cc_rngNormal_t  rng_normal;

	vkk_compute_t* compute;

	vkk_uniformSetFactory_t* usf0_batchNorm;
	vkk_uniformSetFactory_t* usf1_batchNorm_fp;
	vkk_uniformSetFactory_t* usf1_batchNorm_bp;
	vkk_uniformSetFactory_t* usf2_batchNorm;
	vkk_uniformSetFactory_t* usf0_conv;
	vkk_uniformSetFactory_t* usf1_conv_fp;
	vkk_uniformSetFactory_t* usf1_conv_bp;
	vkk_uniformSetFactory_t* usf2_conv;
	vkk_uniformSetFactory_t* usf0_fact;
	vkk_uniformSetFactory_t* usf1_fact_fp;
	vkk_uniformSetFactory_t* usf1_fact_bp;
	vkk_uniformSetFactory_t* usf0_lanczos;
	vkk_uniformSetFactory_t* usf1_lanczos_fp;
	vkk_uniformSetFactory_t* usf1_lanczos_bp;
	vkk_uniformSetFactory_t* usf2_lanczos;
	vkk_uniformSetFactory_t* usf0_skip;
	vkk_uniformSetFactory_t* usf1_skip_fp;
	vkk_uniformSetFactory_t* usf1_skip_bp;
	vkk_uniformSetFactory_t* usf0_weight;
	vkk_uniformSetFactory_t* usf1_weight_fp;
	vkk_uniformSetFactory_t* usf1_weight_bp;
	vkk_uniformSetFactory_t* usf0_loss;
	vkk_uniformSetFactory_t* usf1_loss;
	vkk_uniformSetFactory_t* usf0_tensor;
	vkk_uniformSetFactory_t* usf1_tensor_stats;
	vkk_uniformSetFactory_t* usf1_tensor_norm;
	vkk_uniformSetFactory_t* usf0_tensor_op;

	vkk_pipelineLayout_t* pl_batchNorm_fp;
	vkk_pipelineLayout_t* pl_batchNorm_bp;
	vkk_pipelineLayout_t* pl_conv_fp;
	vkk_pipelineLayout_t* pl_conv_bp;
	vkk_pipelineLayout_t* pl_fact_fp;
	vkk_pipelineLayout_t* pl_fact_bp;
	vkk_pipelineLayout_t* pl_lanczos_fp;
	vkk_pipelineLayout_t* pl_lanczos_bp;
	vkk_pipelineLayout_t* pl_skip_fp;
	vkk_pipelineLayout_t* pl_skip_bp;
	vkk_pipelineLayout_t* pl_weight_fp;
	vkk_pipelineLayout_t* pl_weight_bp;
	vkk_pipelineLayout_t* pl_loss;
	vkk_pipelineLayout_t* pl_tensor_stats;
	vkk_pipelineLayout_t* pl_tensor_norm;
	vkk_pipelineLayout_t* pl_tensor_op;

	vkk_computePipeline_t* cp_batchNorm_forwardPassXmeanTrain;
	vkk_computePipeline_t* cp_batchNorm_forwardPassXvarTrain;
	vkk_computePipeline_t* cp_batchNorm_forwardPassXmeanCompute;
	vkk_computePipeline_t* cp_batchNorm_forwardPassXvarCompute;
	vkk_computePipeline_t* cp_batchNorm_forwardPassXhat;
	vkk_computePipeline_t* cp_batchNorm_forwardPassY;
	vkk_computePipeline_t* cp_batchNorm_backprop_dL_dX;
	vkk_computePipeline_t* cp_batchNorm_backprop_dL_dXhat;
	vkk_computePipeline_t* cp_batchNorm_backpropSum;
	vkk_computePipeline_t* cp_batchNorm_backpropSumNOP;
	vkk_computePipeline_t* cp_conv_forwardPass;
	vkk_computePipeline_t* cp_conv_forwardPassT;
	vkk_computePipeline_t* cp_conv_backprop_dL_dX;
	vkk_computePipeline_t* cp_conv_backprop_dL_dW;
	vkk_computePipeline_t* cp_conv_backprop_dL_dW_dB;
	vkk_computePipeline_t* cp_conv_backprop_dL_dB;
	vkk_computePipeline_t* cp_conv_backpropT_dL_dX;
	vkk_computePipeline_t* cp_conv_backpropT_dL_dW;
	vkk_computePipeline_t* cp_conv_backpropUpdateW;
	vkk_computePipeline_t* cp_conv_backpropUpdateB;
	vkk_computePipeline_t* cp_fact_forwardPassLinear;
	vkk_computePipeline_t* cp_fact_forwardPassLogistic;
	vkk_computePipeline_t* cp_fact_forwardPassReLU;
	vkk_computePipeline_t* cp_fact_forwardPassPReLU;
	vkk_computePipeline_t* cp_fact_forwardPassLReLU;
	vkk_computePipeline_t* cp_fact_forwardPassTanh;
	vkk_computePipeline_t* cp_fact_forwardPassSink;
	vkk_computePipeline_t* cp_fact_backpropLinear;
	vkk_computePipeline_t* cp_fact_backpropLogistic;
	vkk_computePipeline_t* cp_fact_backpropReLU;
	vkk_computePipeline_t* cp_fact_backpropPReLU;
	vkk_computePipeline_t* cp_fact_backpropLReLU;
	vkk_computePipeline_t* cp_fact_backpropTanh;
	vkk_computePipeline_t* cp_fact_backpropSink;
	vkk_computePipeline_t* cp_lanczos_forwardPassT;
	vkk_computePipeline_t* cp_lanczos_forwardPassY;
	vkk_computePipeline_t* cp_lanczos_backprop_dL_dT;
	vkk_computePipeline_t* cp_lanczos_backprop_dL_dX;
	vkk_computePipeline_t* cp_skip_forwardPassAdd;
	vkk_computePipeline_t* cp_skip_forwardPassCat;
	vkk_computePipeline_t* cp_skip_backpropAdd;
	vkk_computePipeline_t* cp_skip_backpropCat;
	vkk_computePipeline_t* cp_skip_backpropFork;
	vkk_computePipeline_t* cp_weight_forwardPass;
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
	vkk_computePipeline_t* cp_tensor_stats;
	vkk_computePipeline_t* cp_tensor_sn;
	vkk_computePipeline_t* cp_tensor_bssn;
	vkk_computePipeline_t* cp_tensor_computeFillOp;
	vkk_computePipeline_t* cp_tensor_computeCopyOp;
	vkk_computePipeline_t* cp_tensor_computeAddOp;
	vkk_computePipeline_t* cp_tensor_computeMixOp;
	vkk_computePipeline_t* cp_tensor_computeMulOp;
	vkk_computePipeline_t* cp_tensor_computeScaleOp;
	vkk_computePipeline_t* cp_tensor_computeScaleAddOp;

	nn_tensor_t* Null;

	cc_map_t*  map_bn_us2;
	cc_map_t*  map_conv_us2;
	cc_map_t*  map_lanczos_us2;
	cc_list_t* list_tensorOp_us0[2];
} nn_engine_t;

nn_engine_t*      nn_engine_new(vkk_engine_t* engine);
void              nn_engine_delete(nn_engine_t** _self);
vkk_uniformSet_t* nn_engine_getBatchNormUs2(nn_engine_t* self,
                                            uint32_t k);
vkk_uniformSet_t* nn_engine_getConvUs2(nn_engine_t* self,
                                       uint32_t f, uint32_t fi,
                                       uint32_t fj, uint32_t k);
vkk_uniformSet_t* nn_engine_getLanczos3Us2(nn_engine_t* self,
                                           uint32_t n);
vkk_uniformSet_t* nn_engine_getTensorOpUs0(nn_engine_t* self,
                                           nn_tensor_t* X1,
                                           nn_tensor_t* X2,
                                           nn_tensor_t* Y,
                                           nn_tensorOpUs0Idx_t* idx);
int               nn_engine_computeBegin(nn_engine_t* self);
void              nn_engine_computeEnd(nn_engine_t* self);
void              nn_engine_computeDispatch(nn_engine_t* self,
                                            vkk_hazard_e hazard,
                                            uint32_t count_x,
                                            uint32_t count_y,
                                            uint32_t count_z,
                                            uint32_t local_size_x,
                                            uint32_t local_size_y,
                                            uint32_t local_size_z);
int               nn_engine_computeBind(nn_engine_t* self,
                                        vkk_computePipeline_t* cp);

#endif
