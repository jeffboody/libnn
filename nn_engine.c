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
#include "../libvkk/vkk.h"
#include "nn_batchNormLayer.h"
#include "nn_convLayer.h"
#include "nn_engine.h"
#include "nn_lanczosLayer.h"
#include "nn_layer.h"
#include "nn_loss.h"
#include "nn_tensor.h"

// split dispatch to improve UI responsiveness
// 1) the actual number of dispatches issued may vary
//    depending on NN layer design
// 2) the dispatch amount to achieve good UI performance may
//    depend on the hardware used and the neural network
//    architecure for a particular problem
#define NN_ENGINE_DISPATCH_HINT 100

/***********************************************************
* private                                                  *
***********************************************************/

static void
nn_engine_initUbArray(vkk_uniformBinding_t* ub_array,
                      uint32_t count)
{
	ASSERT(ub_array);

	uint32_t i;
	for(i = 0; i < count; ++i)
	{
		ub_array[i].binding = i;
		ub_array[i].type    = VKK_UNIFORM_TYPE_STORAGE_REF;
		ub_array[i].stage   = VKK_STAGE_COMPUTE;
	}
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_engine_t*
nn_engine_new(vkk_engine_t* engine)
{
	ASSERT(engine);

	nn_engine_t* self;
	self = (nn_engine_t*)
	       CALLOC(1, sizeof(nn_engine_t));
	if(self == NULL)
	{
		LOGE("CALLOC failed");
		return NULL;
	}

	self->engine = engine;

	cc_rngUniform_init(&self->rng_uniform);
	cc_rngNormal_init(&self->rng_normal, 0.0, 1.0);

	self->compute = vkk_compute_new(engine);
	if(self->compute == NULL)
	{
		goto failure;
	}

	vkk_updateMode_e um;
	um = vkk_compute_updateMode(self->compute);

	// all ub_arrays will contain storage buffer references
	// but each usf may have a different count
	// see readme.md for more details
	vkk_uniformBinding_t ub_array[20] = { 0 };
	nn_engine_initUbArray(ub_array, 20);

	// sb000: dimX (xbs,xh,xw,xd)
	// ...
	// sb015: Csum
	self->usf0_batchNorm = vkk_uniformSetFactory_new(engine, um,
	                                                 16, ub_array);

	// sb100: bs
	// ...
	// sb104: Xvar
	self->usf1_batchNorm_fp = vkk_uniformSetFactory_new(engine, um,
	                                                    5, ub_array);

	// sb100: bs
	// ...
	// sb102: dL_dY
	self->usf1_batchNorm_bp = vkk_uniformSetFactory_new(engine, um,
	                                                    3, ub_array);

	// sb200: idx (k)
	self->usf2_batchNorm = vkk_uniformSetFactory_new(engine, um,
	                                                 1, ub_array);

	// sb000: dimX (xbs,xh,xw,xd)
	// ...
	// sb013: param (disable_bias,stride)
	self->usf0_conv = vkk_uniformSetFactory_new(engine, um,
	                                            14, ub_array);

	// sb100: bs
	// ...
	// sb102: X
	self->usf1_conv_fp = vkk_uniformSetFactory_new(engine, um,
	                                               3, ub_array);

	// sb100: bs
	// ...
	// sb103: dL_dY
	self->usf1_conv_bp = vkk_uniformSetFactory_new(engine, um,
	                                               4, ub_array);

	// sb200: idx (f,fi,fj,k)
	self->usf2_conv = vkk_uniformSetFactory_new(engine, um,
	                                            1, ub_array);

	// sb000: dimX
	// sb001: Y
	self->usf0_fact = vkk_uniformSetFactory_new(engine, um,
	                                            2, ub_array);

	// sb100: bs
	// ...
	// sb102: X
	self->usf1_fact_fp = vkk_uniformSetFactory_new(engine, um,
	                                               3, ub_array);

	// sb100: bs
	// ...
	// sb103: dL_dY
	self->usf1_fact_bp = vkk_uniformSetFactory_new(engine, um,
	                                               4, ub_array);

	// sb000: dimX (bs,xh,xw,xd)
	// ...
	// sb008: param (stride)
	self->usf0_lanczos = vkk_uniformSetFactory_new(engine, um,
	                                               9, ub_array);

	// sb100: bs
	// ...
	// sb102: X
	self->usf1_lanczos_fp = vkk_uniformSetFactory_new(engine, um,
	                                                  3, ub_array);

	// sb100: bs
	// ...
	// sb102: dL_dY
	self->usf1_lanczos_bp = vkk_uniformSetFactory_new(engine, um,
	                                                  3, ub_array);

	// sb200: idx (n)
	self->usf2_lanczos = vkk_uniformSetFactory_new(engine, um,
	                                               1, ub_array);

	// sb000: param (beta)
	self->usf0_skip = vkk_uniformSetFactory_new(engine, um,
	                                            1, ub_array);

	// sb100: bs
	// ...
	// sb107: Y
	self->usf1_skip_fp = vkk_uniformSetFactory_new(engine, um,
	                                               8, ub_array);

	// sb100: bs
	// ...
	// sb109: dL_dY2
	self->usf1_skip_bp = vkk_uniformSetFactory_new(engine, um,
	                                               10, ub_array);

	// sb000: dimX
	// ...
	// sb013: param (disable_bias)
	self->usf0_weight = vkk_uniformSetFactory_new(engine, um,
	                                              14, ub_array);

	// sb100: bs
	// ...
	// sb102: X
	self->usf1_weight_fp = vkk_uniformSetFactory_new(engine, um,
	                                                 3, ub_array);

	// sb100: bs
	// ...
	// sb103: dL_dY
	self->usf1_weight_bp = vkk_uniformSetFactory_new(engine, um,
	                                                 4, ub_array);

	// sb000: bs
	// ...
	// sb003: dL_dY
	self->usf0_loss = vkk_uniformSetFactory_new(engine, um,
	                                            4, ub_array);

	// sb100: Y
	// sb101: Yt
	self->usf1_loss = vkk_uniformSetFactory_new(engine, um,
	                                            2, ub_array);

	// sb00: dimX
	// ...
	// sb02: stats
	self->usf0_tensor = vkk_uniformSetFactory_new(engine, um,
	                                              3, ub_array);

	// sb10: stats
	self->usf1_tensor_stats = vkk_uniformSetFactory_new(engine,
	                                                    um, 1,
	                                                    ub_array);

	// sb20: u1
	// ...
	// sb24: c
	self->usf1_tensor_norm = vkk_uniformSetFactory_new(engine,
	                                                   um, 5,
	                                                   ub_array);

	// sb000: dimX1
	// ...
	// sb006: idx (x1n,...,value)
	self->usf0_tensor_op = vkk_uniformSetFactory_new(engine,
	                                                 um, 7,
	                                                 ub_array);

	if((self->usf0_batchNorm    == NULL) ||
	   (self->usf1_batchNorm_fp == NULL) ||
	   (self->usf1_batchNorm_bp == NULL) ||
	   (self->usf2_batchNorm    == NULL) ||
	   (self->usf0_conv         == NULL) ||
	   (self->usf1_conv_fp      == NULL) ||
	   (self->usf1_conv_bp      == NULL) ||
	   (self->usf2_conv         == NULL) ||
	   (self->usf0_fact         == NULL) ||
	   (self->usf1_fact_fp      == NULL) ||
	   (self->usf1_fact_bp      == NULL) ||
	   (self->usf0_lanczos      == NULL) ||
	   (self->usf1_lanczos_fp   == NULL) ||
	   (self->usf1_lanczos_bp   == NULL) ||
	   (self->usf2_lanczos      == NULL) ||
	   (self->usf0_skip         == NULL) ||
	   (self->usf1_skip_fp      == NULL) ||
	   (self->usf1_skip_bp      == NULL) ||
	   (self->usf0_weight       == NULL) ||
	   (self->usf1_weight_fp    == NULL) ||
	   (self->usf1_weight_bp    == NULL) ||
	   (self->usf0_loss         == NULL) ||
	   (self->usf1_loss         == NULL) ||
	   (self->usf0_tensor       == NULL) ||
	   (self->usf1_tensor_stats == NULL) ||
	   (self->usf1_tensor_norm  == NULL) ||
	   (self->usf0_tensor_op    == NULL))
	{
		goto failure;
	}

	vkk_uniformSetFactory_t* usf_array_batchNorm_fp[] =
	{
		self->usf0_batchNorm,
		self->usf1_batchNorm_fp,
		self->usf2_batchNorm,
	};
	self->pl_batchNorm_fp = vkk_pipelineLayout_new(engine, 3,
	                                               usf_array_batchNorm_fp);

	vkk_uniformSetFactory_t* usf_array_batchNorm_bp[] =
	{
		self->usf0_batchNorm,
		self->usf1_batchNorm_bp,
		self->usf2_batchNorm,
	};
	self->pl_batchNorm_bp = vkk_pipelineLayout_new(engine, 3,
	                                               usf_array_batchNorm_bp);

	vkk_uniformSetFactory_t* usf_array_conv_fp[] =
	{
		self->usf0_conv,
		self->usf1_conv_fp,
	};
	self->pl_conv_fp = vkk_pipelineLayout_new(engine, 2,
	                                          usf_array_conv_fp);

	vkk_uniformSetFactory_t* usf_array_conv_bp[] =
	{
		self->usf0_conv,
		self->usf1_conv_bp,
		self->usf2_conv,
	};
	self->pl_conv_bp = vkk_pipelineLayout_new(engine, 3,
	                                          usf_array_conv_bp);

	vkk_uniformSetFactory_t* usf_array_fact_fp[] =
	{
		self->usf0_fact,
		self->usf1_fact_fp,
	};
	self->pl_fact_fp = vkk_pipelineLayout_new(engine, 2,
	                                          usf_array_fact_fp);

	vkk_uniformSetFactory_t* usf_array_fact_bp[] =
	{
		self->usf0_fact,
		self->usf1_fact_bp,
	};
	self->pl_fact_bp = vkk_pipelineLayout_new(engine, 2,
	                                          usf_array_fact_bp);

	vkk_uniformSetFactory_t* usf_array_lanczos_fp[] =
	{
		self->usf0_lanczos,
		self->usf1_lanczos_fp,
	};
	self->pl_lanczos_fp = vkk_pipelineLayout_new(engine, 2,
	                                             usf_array_lanczos_fp);

	vkk_uniformSetFactory_t* usf_array_lanczos_bp[] =
	{
		self->usf0_lanczos,
		self->usf1_lanczos_bp,
		self->usf2_lanczos,
	};
	self->pl_lanczos_bp = vkk_pipelineLayout_new(engine, 3,
	                                             usf_array_lanczos_bp);

	vkk_uniformSetFactory_t* usf_array_skip_fp[] =
	{
		self->usf0_skip,
		self->usf1_skip_fp,
	};
	self->pl_skip_fp = vkk_pipelineLayout_new(engine, 2,
	                                          usf_array_skip_fp);

	vkk_uniformSetFactory_t* usf_array_skip_bp[] =
	{
		self->usf0_skip,
		self->usf1_skip_bp,
	};
	self->pl_skip_bp = vkk_pipelineLayout_new(engine, 2,
	                                          usf_array_skip_bp);

	vkk_uniformSetFactory_t* usf_array_weight_fp[] =
	{
		self->usf0_weight,
		self->usf1_weight_fp,
	};
	self->pl_weight_fp = vkk_pipelineLayout_new(engine, 2,
	                                            usf_array_weight_fp);

	vkk_uniformSetFactory_t* usf_array_weight_bp[] =
	{
		self->usf0_weight,
		self->usf1_weight_bp,
	};
	self->pl_weight_bp = vkk_pipelineLayout_new(engine, 2,
	                                            usf_array_weight_bp);

	vkk_uniformSetFactory_t* usf_array_loss[] =
	{
		self->usf0_loss,
		self->usf1_loss,
	};
	self->pl_loss = vkk_pipelineLayout_new(engine, 2,
	                                       usf_array_loss);

	vkk_uniformSetFactory_t* usf_array_tensor_stats[] =
	{
		self->usf0_tensor,
		self->usf1_tensor_stats,
	};
	self->pl_tensor_stats = vkk_pipelineLayout_new(engine, 2,
	                                               usf_array_tensor_stats);

	vkk_uniformSetFactory_t* usf_array_tensor_norm[] =
	{
		self->usf0_tensor,
		self->usf1_tensor_norm,
	};
	self->pl_tensor_norm = vkk_pipelineLayout_new(engine, 2,
	                                              usf_array_tensor_norm);

	vkk_uniformSetFactory_t* usf_array_tensor_op[] =
	{
		self->usf0_tensor_op,
	};
	self->pl_tensor_op = vkk_pipelineLayout_new(engine, 1,
	                                            usf_array_tensor_op);

	if((self->pl_batchNorm_fp == NULL) ||
	   (self->pl_batchNorm_bp == NULL) ||
	   (self->pl_conv_fp      == NULL) ||
	   (self->pl_conv_bp      == NULL) ||
	   (self->pl_fact_fp      == NULL) ||
	   (self->pl_fact_bp      == NULL) ||
	   (self->pl_lanczos_fp   == NULL) ||
	   (self->pl_lanczos_bp   == NULL) ||
	   (self->pl_skip_fp      == NULL) ||
	   (self->pl_skip_bp      == NULL) ||
	   (self->pl_weight_fp    == NULL) ||
	   (self->pl_weight_bp    == NULL) ||
	   (self->pl_loss         == NULL) ||
	   (self->pl_tensor_stats == NULL) ||
	   (self->pl_tensor_norm  == NULL) ||
	   (self->pl_tensor_op    == NULL))
	{
		goto failure;
	}

	vkk_computePipelineInfo_t cpi_batchNorm_forwardPassXmeanTrain =
	{
		.compute = self->compute,
		.pl      = self->pl_batchNorm_fp,
		.cs      = "nn/shaders/nn_batchNormLayer_forwardPassXmeanTrain_comp.spv",
	};

	self->cp_batchNorm_forwardPassXmeanTrain =
		vkk_computePipeline_new(engine,
		                        &cpi_batchNorm_forwardPassXmeanTrain);

	vkk_computePipelineInfo_t cpi_batchNorm_forwardPassXvarTrain =
	{
		.compute = self->compute,
		.pl      = self->pl_batchNorm_fp,
		.cs      = "nn/shaders/nn_batchNormLayer_forwardPassXvarTrain_comp.spv",
	};

	self->cp_batchNorm_forwardPassXvarTrain =
		vkk_computePipeline_new(engine,
		                        &cpi_batchNorm_forwardPassXvarTrain);

	vkk_computePipelineInfo_t cpi_batchNorm_forwardPassXmeanCompute =
	{
		.compute = self->compute,
		.pl      = self->pl_batchNorm_fp,
		.cs      = "nn/shaders/nn_batchNormLayer_forwardPassXmeanCompute_comp.spv",
	};

	self->cp_batchNorm_forwardPassXmeanCompute =
		vkk_computePipeline_new(engine,
		                        &cpi_batchNorm_forwardPassXmeanCompute);

	vkk_computePipelineInfo_t cpi_batchNorm_forwardPassXvarCompute =
	{
		.compute = self->compute,
		.pl      = self->pl_batchNorm_fp,
		.cs      = "nn/shaders/nn_batchNormLayer_forwardPassXvarCompute_comp.spv",
	};

	self->cp_batchNorm_forwardPassXvarCompute =
		vkk_computePipeline_new(engine,
		                        &cpi_batchNorm_forwardPassXvarCompute);

	vkk_computePipelineInfo_t cpi_batchNorm_forwardPassXhat =
	{
		.compute = self->compute,
		.pl      = self->pl_batchNorm_fp,
		.cs      = "nn/shaders/nn_batchNormLayer_forwardPassXhat_comp.spv",
	};

	self->cp_batchNorm_forwardPassXhat =
		vkk_computePipeline_new(engine,
		                        &cpi_batchNorm_forwardPassXhat);

	vkk_computePipelineInfo_t cpi_batchNorm_forwardPassY =
	{
		.compute = self->compute,
		.pl      = self->pl_batchNorm_fp,
		.cs      = "nn/shaders/nn_batchNormLayer_forwardPassY_comp.spv",
	};

	self->cp_batchNorm_forwardPassY =
		vkk_computePipeline_new(engine,
		                        &cpi_batchNorm_forwardPassY);

	vkk_computePipelineInfo_t cpi_batchNorm_backprop_dL_dX =
	{
		.compute = self->compute,
		.pl      = self->pl_batchNorm_bp,
		.cs      = "nn/shaders/nn_batchNormLayer_backprop_dL_dX_comp.spv",
	};

	self->cp_batchNorm_backprop_dL_dX =
		vkk_computePipeline_new(engine,
		                        &cpi_batchNorm_backprop_dL_dX);

	vkk_computePipelineInfo_t cpi_batchNorm_backprop_dL_dXhat =
	{
		.compute = self->compute,
		.pl      = self->pl_batchNorm_bp,
		.cs      = "nn/shaders/nn_batchNormLayer_backprop_dL_dXhat_comp.spv",
	};

	self->cp_batchNorm_backprop_dL_dXhat =
		vkk_computePipeline_new(engine,
		                        &cpi_batchNorm_backprop_dL_dXhat);

	vkk_computePipelineInfo_t cpi_batchNorm_backpropSum =
	{
		.compute = self->compute,
		.pl      = self->pl_batchNorm_bp,
		.cs      = "nn/shaders/nn_batchNormLayer_backpropSum_comp.spv",
	};

	self->cp_batchNorm_backpropSum =
		vkk_computePipeline_new(engine,
		                        &cpi_batchNorm_backpropSum);

	vkk_computePipelineInfo_t cpi_batchNorm_backpropSumNOP =
	{
		.compute = self->compute,
		.pl      = self->pl_batchNorm_bp,
		.cs      = "nn/shaders/nn_batchNormLayer_backpropSumNOP_comp.spv",
	};

	self->cp_batchNorm_backpropSumNOP =
		vkk_computePipeline_new(engine,
		                        &cpi_batchNorm_backpropSumNOP);

	vkk_computePipelineInfo_t cpi_conv_forwardPass =
	{
		.compute = self->compute,
		.pl      = self->pl_conv_fp,
		.cs      = "nn/shaders/nn_convLayer_forwardPass_comp.spv",
	};

	self->cp_conv_forwardPass =
		vkk_computePipeline_new(engine,
		                        &cpi_conv_forwardPass);

	vkk_computePipelineInfo_t cpi_conv_forwardPassT =
	{
		.compute = self->compute,
		.pl      = self->pl_conv_fp,
		.cs      = "nn/shaders/nn_convLayer_forwardPassT_comp.spv",
	};

	self->cp_conv_forwardPassT =
		vkk_computePipeline_new(engine,
		                        &cpi_conv_forwardPassT);

	vkk_computePipelineInfo_t cpi_conv_backprop_dL_dX =
	{
		.compute = self->compute,
		.pl      = self->pl_conv_bp,
		.cs      = "nn/shaders/nn_convLayer_backprop_dL_dX_comp.spv",
	};

	self->cp_conv_backprop_dL_dX =
		vkk_computePipeline_new(engine,
		                        &cpi_conv_backprop_dL_dX);

	vkk_computePipelineInfo_t cpi_conv_backprop_dL_dW =
	{
		.compute = self->compute,
		.pl      = self->pl_conv_bp,
		.cs      = "nn/shaders/nn_convLayer_backprop_dL_dW_comp.spv",
	};

	self->cp_conv_backprop_dL_dW =
		vkk_computePipeline_new(engine,
		                        &cpi_conv_backprop_dL_dW);

	vkk_computePipelineInfo_t cpi_conv_backprop_dL_dB =
	{
		.compute = self->compute,
		.pl      = self->pl_conv_bp,
		.cs      = "nn/shaders/nn_convLayer_backprop_dL_dB_comp.spv",
	};

	self->cp_conv_backprop_dL_dB =
		vkk_computePipeline_new(engine,
		                        &cpi_conv_backprop_dL_dB);

	vkk_computePipelineInfo_t cpi_conv_backpropT_dL_dX =
	{
		.compute = self->compute,
		.pl      = self->pl_conv_bp,
		.cs      = "nn/shaders/nn_convLayer_backpropT_dL_dX_comp.spv",
	};

	self->cp_conv_backpropT_dL_dX =
		vkk_computePipeline_new(engine,
		                        &cpi_conv_backpropT_dL_dX);

	vkk_computePipelineInfo_t cpi_conv_backpropT_dL_dW =
	{
		.compute = self->compute,
		.pl      = self->pl_conv_bp,
		.cs      = "nn/shaders/nn_convLayer_backpropT_dL_dW_comp.spv",
	};

	self->cp_conv_backpropT_dL_dW =
		vkk_computePipeline_new(engine,
		                        &cpi_conv_backpropT_dL_dW);

	vkk_computePipelineInfo_t cpi_conv_backpropUpdateW =
	{
		.compute = self->compute,
		.pl      = self->pl_conv_bp,
		.cs      = "nn/shaders/nn_convLayer_backpropUpdateW_comp.spv",
	};

	self->cp_conv_backpropUpdateW =
		vkk_computePipeline_new(engine,
		                        &cpi_conv_backpropUpdateW);

	vkk_computePipelineInfo_t cpi_conv_backpropUpdateB =
	{
		.compute = self->compute,
		.pl      = self->pl_conv_bp,
		.cs      = "nn/shaders/nn_convLayer_backpropUpdateB_comp.spv",
	};

	self->cp_conv_backpropUpdateB =
		vkk_computePipeline_new(engine,
		                        &cpi_conv_backpropUpdateB);

	vkk_computePipelineInfo_t cpi_fact_forwardPassLinear =
	{
		.compute = self->compute,
		.pl      = self->pl_fact_fp,
		.cs      = "nn/shaders/nn_factLayer_forwardPassLinear_comp.spv",
	};

	self->cp_fact_forwardPassLinear =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_forwardPassLinear);

	vkk_computePipelineInfo_t cpi_fact_forwardPassLogistic =
	{
		.compute = self->compute,
		.pl      = self->pl_fact_fp,
		.cs      = "nn/shaders/nn_factLayer_forwardPassLogistic_comp.spv",
	};

	self->cp_fact_forwardPassLogistic =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_forwardPassLogistic);

	vkk_computePipelineInfo_t cpi_fact_forwardPassReLU =
	{
		.compute = self->compute,
		.pl      = self->pl_fact_fp,
		.cs      = "nn/shaders/nn_factLayer_forwardPassReLU_comp.spv",
	};

	self->cp_fact_forwardPassReLU =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_forwardPassReLU);

	vkk_computePipelineInfo_t cpi_fact_forwardPassPReLU =
	{
		.compute = self->compute,
		.pl      = self->pl_fact_fp,
		.cs      = "nn/shaders/nn_factLayer_forwardPassPReLU_comp.spv",
	};

	self->cp_fact_forwardPassPReLU =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_forwardPassPReLU);

	vkk_computePipelineInfo_t cpi_fact_forwardPassLReLU =
	{
		.compute = self->compute,
		.pl      = self->pl_fact_fp,
		.cs      = "nn/shaders/nn_factLayer_forwardPassLReLU_comp.spv",
	};

	self->cp_fact_forwardPassLReLU =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_forwardPassLReLU);

	vkk_computePipelineInfo_t cpi_fact_forwardPassTanh =
	{
		.compute = self->compute,
		.pl      = self->pl_fact_fp,
		.cs      = "nn/shaders/nn_factLayer_forwardPassTanh_comp.spv",
	};

	self->cp_fact_forwardPassTanh =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_forwardPassTanh);

	vkk_computePipelineInfo_t cpi_fact_forwardPassSink =
	{
		.compute = self->compute,
		.pl      = self->pl_fact_fp,
		.cs      = "nn/shaders/nn_factLayer_forwardPassSink_comp.spv",
	};

	self->cp_fact_forwardPassSink =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_forwardPassSink);

	vkk_computePipelineInfo_t cpi_fact_backpropLinear =
	{
		.compute = self->compute,
		.pl      = self->pl_fact_bp,
		.cs      = "nn/shaders/nn_factLayer_backpropLinear_comp.spv",
	};

	self->cp_fact_backpropLinear =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_backpropLinear);

	vkk_computePipelineInfo_t cpi_fact_backpropLogistic =
	{
		.compute = self->compute,
		.pl      = self->pl_fact_bp,
		.cs      = "nn/shaders/nn_factLayer_backpropLogistic_comp.spv",
	};

	self->cp_fact_backpropLogistic =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_backpropLogistic);

	vkk_computePipelineInfo_t cpi_fact_backpropReLU =
	{
		.compute = self->compute,
		.pl      = self->pl_fact_bp,
		.cs      = "nn/shaders/nn_factLayer_backpropReLU_comp.spv",
	};

	self->cp_fact_backpropReLU =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_backpropReLU);

	vkk_computePipelineInfo_t cpi_fact_backpropPReLU =
	{
		.compute = self->compute,
		.pl      = self->pl_fact_bp,
		.cs      = "nn/shaders/nn_factLayer_backpropPReLU_comp.spv",
	};

	self->cp_fact_backpropPReLU =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_backpropPReLU);

	vkk_computePipelineInfo_t cpi_fact_backpropLReLU =
	{
		.compute = self->compute,
		.pl      = self->pl_fact_bp,
		.cs      = "nn/shaders/nn_factLayer_backpropLReLU_comp.spv",
	};

	self->cp_fact_backpropLReLU =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_backpropLReLU);

	vkk_computePipelineInfo_t cpi_fact_backpropTanh =
	{
		.compute = self->compute,
		.pl      = self->pl_fact_bp,
		.cs      = "nn/shaders/nn_factLayer_backpropTanh_comp.spv",
	};

	self->cp_fact_backpropTanh =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_backpropTanh);

	vkk_computePipelineInfo_t cpi_fact_backpropSink =
	{
		.compute = self->compute,
		.pl      = self->pl_fact_bp,
		.cs      = "nn/shaders/nn_factLayer_backpropSink_comp.spv",
	};

	self->cp_fact_backpropSink =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_backpropSink);

	vkk_computePipelineInfo_t cpi_lanczos_forwardPassT =
	{
		.compute = self->compute,
		.pl      = self->pl_lanczos_fp,
		.cs      = "nn/shaders/nn_lanczosLayer_forwardPassT_comp.spv",
	};

	self->cp_lanczos_forwardPassT =
		vkk_computePipeline_new(engine,
		                        &cpi_lanczos_forwardPassT);

	vkk_computePipelineInfo_t cpi_lanczos_forwardPassY =
	{
		.compute = self->compute,
		.pl      = self->pl_lanczos_fp,
		.cs      = "nn/shaders/nn_lanczosLayer_forwardPassY_comp.spv",
	};

	self->cp_lanczos_forwardPassY =
		vkk_computePipeline_new(engine,
		                        &cpi_lanczos_forwardPassY);

	vkk_computePipelineInfo_t cpi_lanczos_backprop_dL_dT =
	{
		.compute = self->compute,
		.pl      = self->pl_lanczos_bp,
		.cs      = "nn/shaders/nn_lanczosLayer_backprop_dL_dT_comp.spv",
	};

	self->cp_lanczos_backprop_dL_dT =
		vkk_computePipeline_new(engine,
		                        &cpi_lanczos_backprop_dL_dT);

	vkk_computePipelineInfo_t cpi_lanczos_backprop_dL_dX =
	{
		.compute = self->compute,
		.pl      = self->pl_lanczos_bp,
		.cs      = "nn/shaders/nn_lanczosLayer_backprop_dL_dX_comp.spv",
	};

	self->cp_lanczos_backprop_dL_dX =
		vkk_computePipeline_new(engine,
		                        &cpi_lanczos_backprop_dL_dX);

	vkk_computePipelineInfo_t cpi_skip_forwardPassAdd =
	{
		.compute = self->compute,
		.pl      = self->pl_skip_fp,
		.cs      = "nn/shaders/nn_skipLayer_forwardPassAdd_comp.spv",
	};

	self->cp_skip_forwardPassAdd =
		vkk_computePipeline_new(engine,
		                        &cpi_skip_forwardPassAdd);

	vkk_computePipelineInfo_t cpi_skip_forwardPassCat =
	{
		.compute = self->compute,
		.pl      = self->pl_skip_fp,
		.cs      = "nn/shaders/nn_skipLayer_forwardPassCat_comp.spv",
	};

	self->cp_skip_forwardPassCat =
		vkk_computePipeline_new(engine,
		                        &cpi_skip_forwardPassCat);

	vkk_computePipelineInfo_t cpi_skip_backpropAdd =
	{
		.compute = self->compute,
		.pl      = self->pl_skip_bp,
		.cs      = "nn/shaders/nn_skipLayer_backpropAdd_comp.spv",
	};

	self->cp_skip_backpropAdd =
		vkk_computePipeline_new(engine,
		                        &cpi_skip_backpropAdd);

	vkk_computePipelineInfo_t cpi_skip_backpropCat =
	{
		.compute = self->compute,
		.pl      = self->pl_skip_bp,
		.cs      = "nn/shaders/nn_skipLayer_backpropCat_comp.spv",
	};

	self->cp_skip_backpropCat =
		vkk_computePipeline_new(engine,
		                        &cpi_skip_backpropCat);

	vkk_computePipelineInfo_t cpi_skip_backpropFork =
	{
		.compute = self->compute,
		.pl      = self->pl_skip_bp,
		.cs      = "nn/shaders/nn_skipLayer_backpropFork_comp.spv",
	};

	self->cp_skip_backpropFork =
		vkk_computePipeline_new(engine,
		                        &cpi_skip_backpropFork);

	vkk_computePipelineInfo_t cpi_weight_forwardPass =
	{
		.compute = self->compute,
		.pl      = self->pl_weight_fp,
		.cs      = "nn/shaders/nn_weightLayer_forwardPass_comp.spv",
	};

	self->cp_weight_forwardPass =
		vkk_computePipeline_new(engine,
		                        &cpi_weight_forwardPass);

	vkk_computePipelineInfo_t cpi_weight_backpropUpdateW =
	{
		.compute = self->compute,
		.pl      = self->pl_weight_bp,
		.cs      = "nn/shaders/nn_weightLayer_backpropUpdateW_comp.spv",
	};

	self->cp_weight_backpropUpdateW =
		vkk_computePipeline_new(engine,
		                        &cpi_weight_backpropUpdateW);

	vkk_computePipelineInfo_t cpi_weight_backpropUpdateB =
	{
		.compute = self->compute,
		.pl      = self->pl_weight_bp,
		.cs      = "nn/shaders/nn_weightLayer_backpropUpdateB_comp.spv",
	};

	self->cp_weight_backpropUpdateB =
		vkk_computePipeline_new(engine,
		                        &cpi_weight_backpropUpdateB);

	vkk_computePipelineInfo_t cpi_weight_backprop_dL_dX =
	{
		.compute = self->compute,
		.pl      = self->pl_weight_bp,
		.cs      = "nn/shaders/nn_weightLayer_backprop_dL_dX_comp.spv",
	};

	self->cp_weight_backprop_dL_dX =
		vkk_computePipeline_new(engine,
		                        &cpi_weight_backprop_dL_dX);

	vkk_computePipelineInfo_t cpi_weight_backprop_dL_dW =
	{
		.compute = self->compute,
		.pl      = self->pl_weight_bp,
		.cs      = "nn/shaders/nn_weightLayer_backprop_dL_dW_comp.spv",
	};

	self->cp_weight_backprop_dL_dW =
		vkk_computePipeline_new(engine,
		                        &cpi_weight_backprop_dL_dW);

	vkk_computePipelineInfo_t cpi_weight_backprop_dL_dB =
	{
		.compute = self->compute,
		.pl      = self->pl_weight_bp,
		.cs      = "nn/shaders/nn_weightLayer_backprop_dL_dB_comp.spv",
	};

	self->cp_weight_backprop_dL_dB =
		vkk_computePipeline_new(engine,
		                        &cpi_weight_backprop_dL_dB);

	vkk_computePipelineInfo_t cpi_loss_dL_dY_mse =
	{
		.compute = self->compute,
		.pl      = self->pl_loss,
		.cs      = "nn/shaders/nn_loss_dL_dY_mse_comp.spv",
	};

	self->cp_loss_dL_dY_mse =
		vkk_computePipeline_new(engine,
		                        &cpi_loss_dL_dY_mse);

	vkk_computePipelineInfo_t cpi_loss_dL_dY_mae =
	{
		.compute = self->compute,
		.pl      = self->pl_loss,
		.cs      = "nn/shaders/nn_loss_dL_dY_mae_comp.spv",
	};

	self->cp_loss_dL_dY_mae =
		vkk_computePipeline_new(engine,
		                        &cpi_loss_dL_dY_mae);

	vkk_computePipelineInfo_t cpi_loss_dL_dY_bce =
	{
		.compute = self->compute,
		.pl      = self->pl_loss,
		.cs      = "nn/shaders/nn_loss_dL_dY_bce_comp.spv",
	};

	self->cp_loss_dL_dY_bce =
		vkk_computePipeline_new(engine,
		                        &cpi_loss_dL_dY_bce);

	vkk_computePipelineInfo_t cpi_loss_mse =
	{
		.compute = self->compute,
		.pl      = self->pl_loss,
		.cs      = "nn/shaders/nn_loss_mse_comp.spv",
	};

	self->cp_loss_mse =
		vkk_computePipeline_new(engine,
		                        &cpi_loss_mse);

	vkk_computePipelineInfo_t cpi_loss_mae =
	{
		.compute = self->compute,
		.pl      = self->pl_loss,
		.cs      = "nn/shaders/nn_loss_mae_comp.spv",
	};

	self->cp_loss_mae =
		vkk_computePipeline_new(engine,
		                        &cpi_loss_mae);

	vkk_computePipelineInfo_t cpi_loss_bce =
	{
		.compute = self->compute,
		.pl      = self->pl_loss,
		.cs      = "nn/shaders/nn_loss_bce_comp.spv",
	};

	self->cp_loss_bce =
		vkk_computePipeline_new(engine,
		                        &cpi_loss_bce);

	vkk_computePipelineInfo_t cpi_tensor_stats =
	{
		.compute = self->compute,
		.pl      = self->pl_tensor_stats,
		.cs      = "nn/shaders/nn_tensor_stats_comp.spv",
	};

	self->cp_tensor_stats =
		vkk_computePipeline_new(engine,
		                        &cpi_tensor_stats);

	vkk_computePipelineInfo_t cpi_tensor_sn =
	{
		.compute = self->compute,
		.pl      = self->pl_tensor_norm,
		.cs      = "nn/shaders/nn_tensor_sn_comp.spv",
	};

	self->cp_tensor_sn =
		vkk_computePipeline_new(engine,
		                        &cpi_tensor_sn);

	vkk_computePipelineInfo_t cpi_tensor_bssn =
	{
		.compute = self->compute,
		.pl      = self->pl_tensor_norm,
		.cs      = "nn/shaders/nn_tensor_bssn_comp.spv",
	};

	self->cp_tensor_bssn =
		vkk_computePipeline_new(engine,
		                        &cpi_tensor_bssn);

	vkk_computePipelineInfo_t cpi_tensor_computeFillOp =
	{
		.compute = self->compute,
		.pl      = self->pl_tensor_op,
		.cs      = "nn/shaders/nn_tensor_computeFillOp_comp.spv",
	};

	self->cp_tensor_computeFillOp =
		vkk_computePipeline_new(engine,
		                        &cpi_tensor_computeFillOp);

	vkk_computePipelineInfo_t cpi_tensor_computeCopyOp =
	{
		.compute = self->compute,
		.pl      = self->pl_tensor_op,
		.cs      = "nn/shaders/nn_tensor_computeCopyOp_comp.spv",
	};

	self->cp_tensor_computeCopyOp =
		vkk_computePipeline_new(engine,
		                        &cpi_tensor_computeCopyOp);

	vkk_computePipelineInfo_t cpi_tensor_computeAddOp =
	{
		.compute = self->compute,
		.pl      = self->pl_tensor_op,
		.cs      = "nn/shaders/nn_tensor_computeAddOp_comp.spv",
	};

	self->cp_tensor_computeAddOp =
		vkk_computePipeline_new(engine,
		                        &cpi_tensor_computeAddOp);

	vkk_computePipelineInfo_t cpi_tensor_computeMixOp =
	{
		.compute = self->compute,
		.pl      = self->pl_tensor_op,
		.cs      = "nn/shaders/nn_tensor_computeMixOp_comp.spv",
	};

	self->cp_tensor_computeMixOp =
		vkk_computePipeline_new(engine,
		                        &cpi_tensor_computeMixOp);

	vkk_computePipelineInfo_t cpi_tensor_computeScaleOp =
	{
		.compute = self->compute,
		.pl      = self->pl_tensor_op,
		.cs      = "nn/shaders/nn_tensor_computeScaleOp_comp.spv",
	};

	self->cp_tensor_computeScaleOp =
		vkk_computePipeline_new(engine,
		                        &cpi_tensor_computeScaleOp);

	vkk_computePipelineInfo_t cpi_tensor_computeScaleAddOp =
	{
		.compute = self->compute,
		.pl      = self->pl_tensor_op,
		.cs      = "nn/shaders/nn_tensor_computeScaleAddOp_comp.spv",
	};

	self->cp_tensor_computeScaleAddOp =
		vkk_computePipeline_new(engine,
		                        &cpi_tensor_computeScaleAddOp);

	if((self->cp_batchNorm_forwardPassXmeanTrain   == NULL) ||
	   (self->cp_batchNorm_forwardPassXvarTrain    == NULL) ||
	   (self->cp_batchNorm_forwardPassXmeanCompute == NULL) ||
	   (self->cp_batchNorm_forwardPassXvarCompute  == NULL) ||
	   (self->cp_batchNorm_forwardPassXhat         == NULL) ||
	   (self->cp_batchNorm_forwardPassY            == NULL) ||
	   (self->cp_batchNorm_backprop_dL_dX          == NULL) ||
	   (self->cp_batchNorm_backprop_dL_dXhat       == NULL) ||
	   (self->cp_batchNorm_backpropSum             == NULL) ||
	   (self->cp_batchNorm_backpropSumNOP          == NULL) ||
	   (self->cp_conv_forwardPass                  == NULL) ||
	   (self->cp_conv_forwardPassT                 == NULL) ||
	   (self->cp_conv_backprop_dL_dX               == NULL) ||
	   (self->cp_conv_backprop_dL_dW               == NULL) ||
	   (self->cp_conv_backprop_dL_dB               == NULL) ||
	   (self->cp_conv_backpropT_dL_dX              == NULL) ||
	   (self->cp_conv_backpropT_dL_dW              == NULL) ||
	   (self->cp_conv_backpropUpdateW              == NULL) ||
	   (self->cp_conv_backpropUpdateB              == NULL) ||
	   (self->cp_fact_forwardPassLinear            == NULL) ||
	   (self->cp_fact_forwardPassLogistic          == NULL) ||
	   (self->cp_fact_forwardPassReLU              == NULL) ||
	   (self->cp_fact_forwardPassPReLU             == NULL) ||
	   (self->cp_fact_forwardPassLReLU             == NULL) ||
	   (self->cp_fact_forwardPassTanh              == NULL) ||
	   (self->cp_fact_forwardPassSink              == NULL) ||
	   (self->cp_fact_backpropLinear               == NULL) ||
	   (self->cp_fact_backpropLogistic             == NULL) ||
	   (self->cp_fact_backpropReLU                 == NULL) ||
	   (self->cp_fact_backpropPReLU                == NULL) ||
	   (self->cp_fact_backpropLReLU                == NULL) ||
	   (self->cp_fact_backpropTanh                 == NULL) ||
	   (self->cp_fact_backpropSink                 == NULL) ||
	   (self->cp_lanczos_forwardPassT              == NULL) ||
	   (self->cp_lanczos_forwardPassY              == NULL) ||
	   (self->cp_lanczos_backprop_dL_dT            == NULL) ||
	   (self->cp_lanczos_backprop_dL_dX            == NULL) ||
	   (self->cp_skip_forwardPassAdd               == NULL) ||
	   (self->cp_skip_forwardPassCat               == NULL) ||
	   (self->cp_skip_backpropAdd                  == NULL) ||
	   (self->cp_skip_backpropCat                  == NULL) ||
	   (self->cp_skip_backpropFork                 == NULL) ||
	   (self->cp_weight_forwardPass                == NULL) ||
	   (self->cp_weight_backpropUpdateW            == NULL) ||
	   (self->cp_weight_backpropUpdateB            == NULL) ||
	   (self->cp_weight_backprop_dL_dX             == NULL) ||
	   (self->cp_weight_backprop_dL_dW             == NULL) ||
	   (self->cp_weight_backprop_dL_dB             == NULL) ||
	   (self->cp_loss_dL_dY_mse                    == NULL) ||
	   (self->cp_loss_dL_dY_mae                    == NULL) ||
	   (self->cp_loss_dL_dY_bce                    == NULL) ||
	   (self->cp_loss_mse                          == NULL) ||
	   (self->cp_loss_mae                          == NULL) ||
	   (self->cp_loss_bce                          == NULL) ||
	   (self->cp_tensor_stats                      == NULL) ||
	   (self->cp_tensor_sn                         == NULL) ||
	   (self->cp_tensor_bssn                       == NULL) ||
	   (self->cp_tensor_computeFillOp              == NULL) ||
	   (self->cp_tensor_computeCopyOp              == NULL) ||
	   (self->cp_tensor_computeAddOp               == NULL) ||
	   (self->cp_tensor_computeMixOp               == NULL) ||
	   (self->cp_tensor_computeScaleOp             == NULL) ||
	   (self->cp_tensor_computeScaleAddOp          == NULL))
	{
		goto failure;
	}

	nn_dim_t dimNull =
	{
		.count  = 1,
		.height = 1,
		.width  = 1,
		.depth  = 1,
	};
	self->Null = nn_tensor_new(self, &dimNull,
	                           NN_TENSOR_INIT_ZERO,
	                           NN_TENSOR_MODE_COMPUTE);
	if(self->Null == NULL)
	{
		goto failure;
	}

	self->map_bn_us2 = cc_map_new();
	if(self->map_bn_us2 == NULL)
	{
		goto failure;
	}

	self->map_conv_us2 = cc_map_new();
	if(self->map_conv_us2 == NULL)
	{
		goto failure;
	}

	self->map_lanczos_us2 = cc_map_new();
	if(self->map_lanczos_us2 == NULL)
	{
		goto failure;
	}

	self->list_tensorOp_us0[0] = cc_list_new();
	if(self->list_tensorOp_us0[0] == NULL)
	{
		goto failure;
	}

	self->list_tensorOp_us0[1] = cc_list_new();
	if(self->list_tensorOp_us0[1] == NULL)
	{
		goto failure;
	}

	// success
	return self;

	// failure
	failure:
		nn_engine_delete(&self);
	return NULL;
}

void nn_engine_delete(nn_engine_t** _self)
{
	ASSERT(_self);

	nn_engine_t* self = *_self;
	if(self)
	{
		if(self->list_tensorOp_us0[0] &&
		   self->list_tensorOp_us0[1])
		{
			cc_list_appendList(self->list_tensorOp_us0[0],
			                   self->list_tensorOp_us0[1]);
			cc_list_delete(&self->list_tensorOp_us0[1]);
		}

		nn_tensorOpUs0Data_t* data;
		cc_listIter_t*        iter;
		if(self->list_tensorOp_us0[0])
		{
			iter = cc_list_head(self->list_tensorOp_us0[0]);
			while(iter)
			{
				data = (nn_tensorOpUs0Data_t*)
				       cc_list_remove(self->list_tensorOp_us0[0],
				                      &iter);
				nn_tensorOpUs0Data_delete(&data);
			}
			cc_list_delete(&self->list_tensorOp_us0[0]);
		}

		cc_mapIter_t* miter;
		if(self->map_bn_us2)
		{
			miter = cc_map_head(self->map_bn_us2);
			while(miter)
			{
				nn_batchNormUs2Data_t* data;
				data = (nn_batchNormUs2Data_t*)
				       cc_map_remove(self->map_bn_us2,
				                     &miter);
				nn_batchNormUs2Data_delete(&data);
			}
			cc_map_delete(&self->map_bn_us2);
		}

		if(self->map_conv_us2)
		{
			miter = cc_map_head(self->map_conv_us2);
			while(miter)
			{
				nn_convUs2Data_t* data;
				data = (nn_convUs2Data_t*)
				       cc_map_remove(self->map_conv_us2, &miter);
				nn_convUs2Data_delete(&data);
			}
			cc_map_delete(&self->map_conv_us2);
		}

		if(self->map_lanczos_us2)
		{
			miter = cc_map_head(self->map_lanczos_us2);
			while(miter)
			{
				nn_lanczosUs2Data_t* data;
				data = (nn_lanczosUs2Data_t*)
				       cc_map_remove(self->map_lanczos_us2, &miter);
				nn_lanczosUs2Data_delete(&data);
			}
			cc_map_delete(&self->map_lanczos_us2);
		}

		nn_tensor_delete(&self->Null);
		vkk_computePipeline_delete(&self->cp_tensor_computeScaleAddOp);
		vkk_computePipeline_delete(&self->cp_tensor_computeScaleOp);
		vkk_computePipeline_delete(&self->cp_tensor_computeMixOp);
		vkk_computePipeline_delete(&self->cp_tensor_computeAddOp);
		vkk_computePipeline_delete(&self->cp_tensor_computeCopyOp);
		vkk_computePipeline_delete(&self->cp_tensor_computeFillOp);
		vkk_computePipeline_delete(&self->cp_tensor_bssn);
		vkk_computePipeline_delete(&self->cp_tensor_sn);
		vkk_computePipeline_delete(&self->cp_tensor_stats);
		vkk_computePipeline_delete(&self->cp_loss_bce);
		vkk_computePipeline_delete(&self->cp_loss_mae);
		vkk_computePipeline_delete(&self->cp_loss_mse);
		vkk_computePipeline_delete(&self->cp_loss_dL_dY_bce);
		vkk_computePipeline_delete(&self->cp_loss_dL_dY_mae);
		vkk_computePipeline_delete(&self->cp_loss_dL_dY_mse);
		vkk_computePipeline_delete(&self->cp_weight_backprop_dL_dB);
		vkk_computePipeline_delete(&self->cp_weight_backprop_dL_dW);
		vkk_computePipeline_delete(&self->cp_weight_backprop_dL_dX);
		vkk_computePipeline_delete(&self->cp_weight_backpropUpdateB);
		vkk_computePipeline_delete(&self->cp_weight_backpropUpdateW);
		vkk_computePipeline_delete(&self->cp_weight_forwardPass);
		vkk_computePipeline_delete(&self->cp_skip_backpropFork);
		vkk_computePipeline_delete(&self->cp_skip_backpropCat);
		vkk_computePipeline_delete(&self->cp_skip_backpropAdd);
		vkk_computePipeline_delete(&self->cp_skip_forwardPassCat);
		vkk_computePipeline_delete(&self->cp_skip_forwardPassAdd);
		vkk_computePipeline_delete(&self->cp_lanczos_backprop_dL_dX);
		vkk_computePipeline_delete(&self->cp_lanczos_backprop_dL_dT);
		vkk_computePipeline_delete(&self->cp_lanczos_forwardPassY);
		vkk_computePipeline_delete(&self->cp_lanczos_forwardPassT);
		vkk_computePipeline_delete(&self->cp_fact_backpropSink);
		vkk_computePipeline_delete(&self->cp_fact_backpropTanh);
		vkk_computePipeline_delete(&self->cp_fact_backpropLReLU);
		vkk_computePipeline_delete(&self->cp_fact_backpropPReLU);
		vkk_computePipeline_delete(&self->cp_fact_backpropReLU);
		vkk_computePipeline_delete(&self->cp_fact_backpropLogistic);
		vkk_computePipeline_delete(&self->cp_fact_backpropLinear);
		vkk_computePipeline_delete(&self->cp_fact_forwardPassSink);
		vkk_computePipeline_delete(&self->cp_fact_forwardPassTanh);
		vkk_computePipeline_delete(&self->cp_fact_forwardPassLReLU);
		vkk_computePipeline_delete(&self->cp_fact_forwardPassPReLU);
		vkk_computePipeline_delete(&self->cp_fact_forwardPassReLU);
		vkk_computePipeline_delete(&self->cp_fact_forwardPassLogistic);
		vkk_computePipeline_delete(&self->cp_fact_forwardPassLinear);
		vkk_computePipeline_delete(&self->cp_conv_backpropUpdateB);
		vkk_computePipeline_delete(&self->cp_conv_backpropUpdateW);
		vkk_computePipeline_delete(&self->cp_conv_backpropT_dL_dW);
		vkk_computePipeline_delete(&self->cp_conv_backpropT_dL_dX);
		vkk_computePipeline_delete(&self->cp_conv_backprop_dL_dB);
		vkk_computePipeline_delete(&self->cp_conv_backprop_dL_dW);
		vkk_computePipeline_delete(&self->cp_conv_backprop_dL_dX);
		vkk_computePipeline_delete(&self->cp_conv_forwardPassT);
		vkk_computePipeline_delete(&self->cp_conv_forwardPass);
		vkk_computePipeline_delete(&self->cp_batchNorm_backpropSum);
		vkk_computePipeline_delete(&self->cp_batchNorm_backpropSumNOP);
		vkk_computePipeline_delete(&self->cp_batchNorm_backprop_dL_dXhat);
		vkk_computePipeline_delete(&self->cp_batchNorm_backprop_dL_dX);
		vkk_computePipeline_delete(&self->cp_batchNorm_forwardPassY);
		vkk_computePipeline_delete(&self->cp_batchNorm_forwardPassXhat);
		vkk_computePipeline_delete(&self->cp_batchNorm_forwardPassXvarCompute);
		vkk_computePipeline_delete(&self->cp_batchNorm_forwardPassXmeanCompute);
		vkk_computePipeline_delete(&self->cp_batchNorm_forwardPassXvarTrain);
		vkk_computePipeline_delete(&self->cp_batchNorm_forwardPassXmeanTrain);
		vkk_pipelineLayout_delete(&self->pl_tensor_op);
		vkk_pipelineLayout_delete(&self->pl_tensor_norm);
		vkk_pipelineLayout_delete(&self->pl_tensor_stats);
		vkk_pipelineLayout_delete(&self->pl_loss);
		vkk_pipelineLayout_delete(&self->pl_weight_bp);
		vkk_pipelineLayout_delete(&self->pl_weight_fp);
		vkk_pipelineLayout_delete(&self->pl_skip_bp);
		vkk_pipelineLayout_delete(&self->pl_skip_fp);
		vkk_pipelineLayout_delete(&self->pl_lanczos_bp);
		vkk_pipelineLayout_delete(&self->pl_lanczos_fp);
		vkk_pipelineLayout_delete(&self->pl_fact_bp);
		vkk_pipelineLayout_delete(&self->pl_fact_fp);
		vkk_pipelineLayout_delete(&self->pl_conv_bp);
		vkk_pipelineLayout_delete(&self->pl_conv_fp);
		vkk_pipelineLayout_delete(&self->pl_batchNorm_bp);
		vkk_pipelineLayout_delete(&self->pl_batchNorm_fp);
		vkk_uniformSetFactory_delete(&self->usf0_tensor_op);
		vkk_uniformSetFactory_delete(&self->usf1_tensor_norm);
		vkk_uniformSetFactory_delete(&self->usf1_tensor_stats);
		vkk_uniformSetFactory_delete(&self->usf0_tensor);
		vkk_uniformSetFactory_delete(&self->usf1_loss);
		vkk_uniformSetFactory_delete(&self->usf0_loss);
		vkk_uniformSetFactory_delete(&self->usf1_weight_bp);
		vkk_uniformSetFactory_delete(&self->usf1_weight_fp);
		vkk_uniformSetFactory_delete(&self->usf0_weight);
		vkk_uniformSetFactory_delete(&self->usf1_skip_bp);
		vkk_uniformSetFactory_delete(&self->usf1_skip_fp);
		vkk_uniformSetFactory_delete(&self->usf0_skip);
		vkk_uniformSetFactory_delete(&self->usf2_lanczos);
		vkk_uniformSetFactory_delete(&self->usf1_lanczos_bp);
		vkk_uniformSetFactory_delete(&self->usf1_lanczos_fp);
		vkk_uniformSetFactory_delete(&self->usf0_lanczos);
		vkk_uniformSetFactory_delete(&self->usf1_fact_bp);
		vkk_uniformSetFactory_delete(&self->usf1_fact_fp);
		vkk_uniformSetFactory_delete(&self->usf0_fact);
		vkk_uniformSetFactory_delete(&self->usf2_conv);
		vkk_uniformSetFactory_delete(&self->usf1_conv_bp);
		vkk_uniformSetFactory_delete(&self->usf1_conv_fp);
		vkk_uniformSetFactory_delete(&self->usf0_conv);
		vkk_uniformSetFactory_delete(&self->usf2_batchNorm);
		vkk_uniformSetFactory_delete(&self->usf1_batchNorm_bp);
		vkk_uniformSetFactory_delete(&self->usf1_batchNorm_fp);
		vkk_uniformSetFactory_delete(&self->usf0_batchNorm);
		vkk_compute_delete(&self->compute);
		FREE(self);
		*_self = NULL;
	}
}

vkk_uniformSet_t*
nn_engine_getBatchNormUs2(nn_engine_t* self, uint32_t k)
{
	ASSERT(self);

	nn_batchNormUs2Data_t* data;

	nn_batchNormUs2Key_t key =
	{
		.k = k,
	};

	// find existing data
	cc_mapIter_t* miter;
	miter = cc_map_findp(self->map_bn_us2,
	                     sizeof(nn_batchNormUs2Key_t),
	                     &key);
	if(miter)
	{
		data = (nn_batchNormUs2Data_t*) cc_map_val(miter);
		return data->us2;
	}

	data = nn_batchNormUs2Data_new(self, &key);
	if(data == NULL)
	{
		return NULL;
	}

	if(cc_map_addp(self->map_bn_us2,
	               data, sizeof(nn_batchNormUs2Key_t),
	               &key) == NULL)
	{
		goto fail_add;
	}

	// success
	return data->us2;

	// failure
	fail_add:
		nn_batchNormUs2Data_delete(&data);
	return NULL;
}

vkk_uniformSet_t*
nn_engine_getConvUs2(nn_engine_t* self,
                     uint32_t f, uint32_t fi,
                     uint32_t fj, uint32_t k)
{
	ASSERT(self);

	nn_convUs2Data_t* data;

	nn_convUs2Key_t key =
	{
		.f  = f,
		.fi = fi,
		.fj = fj,
		.k  = k,
	};

	// find existing data
	cc_mapIter_t* miter;
	miter = cc_map_findp(self->map_conv_us2,
	                     sizeof(nn_convUs2Key_t),
	                     &key);
	if(miter)
	{
		data = (nn_convUs2Data_t*) cc_map_val(miter);
		return data->us2;
	}

	data = nn_convUs2Data_new(self, &key);
	if(data == NULL)
	{
		return NULL;
	}

	if(cc_map_addp(self->map_conv_us2,
	               data, sizeof(nn_convUs2Key_t),
	               &key) == NULL)
	{
		goto fail_add;
	}

	// success
	return data->us2;

	// failure
	fail_add:
		nn_convUs2Data_delete(&data);
	return NULL;
}

vkk_uniformSet_t*
nn_engine_getLanczos3Us2(nn_engine_t* self, uint32_t n)
{
	ASSERT(self);

	nn_lanczosUs2Data_t* data;

	nn_lanczosUs2Key_t key =
	{
		.n = n,
	};

	// find existing data
	cc_mapIter_t* miter;
	miter = cc_map_findp(self->map_lanczos_us2,
	                     sizeof(nn_lanczosUs2Key_t),
	                     &key);
	if(miter)
	{
		data = (nn_lanczosUs2Data_t*) cc_map_val(miter);
		return data->us2;
	}

	data = nn_lanczosUs2Data_new(self, &key);
	if(data == NULL)
	{
		return NULL;
	}

	if(cc_map_addp(self->map_lanczos_us2,
	               data, sizeof(nn_lanczosUs2Key_t),
	               &key) == NULL)
	{
		goto fail_add;
	}

	// success
	return data->us2;

	// failure
	fail_add:
		nn_lanczosUs2Data_delete(&data);
	return NULL;
}

vkk_uniformSet_t*
nn_engine_getTensorOpUs0(nn_engine_t* self,
                         nn_tensor_t* X1,
                         nn_tensor_t* X2,
                         nn_tensor_t* Y,
                         nn_tensorOpUs0Idx_t* idx)
{
	// X2 and Y may be NULL
	ASSERT(self);
	ASSERT(X1);
	ASSERT(idx);

	nn_tensorOpUs0Data_t* data;
	cc_listIter_t*        iter;
	iter = cc_list_head(self->list_tensorOp_us0[0]);
	if(iter)
	{
		data = (nn_tensorOpUs0Data_t*)
		       cc_list_peekIter(iter);

		if(nn_tensorOpUs0Data_update(data, X1, X2, Y, idx) == 0)
		{
			return NULL;
		}

		cc_list_swapn(self->list_tensorOp_us0[0],
		              self->list_tensorOp_us0[1],
		              iter, NULL);
	}
	else
	{
		data = nn_tensorOpUs0Data_new(X1, X2, Y, idx);
		if(data == NULL)
		{
			return NULL;
		}

		if(cc_list_append(self->list_tensorOp_us0[1], NULL,
		                  data) == NULL)
		{
			goto fail_append;
		}
	}

	// success
	return data->us0;

	// failure
	fail_append:
		nn_tensorOpUs0Data_delete(&data);
	return NULL;
}

int nn_engine_computeBegin(nn_engine_t* self)
{
	ASSERT(self);

	return vkk_compute_begin(self->compute);
}

void nn_engine_computeEnd(nn_engine_t* self)
{
	ASSERT(self);

	if(self->dispatch)
	{
		LOGD("DISPATCH %i", self->dispatch);
		self->dispatch = 0;
	}

	vkk_compute_end(self->compute);

	// make data available for next pass
	cc_list_appendList(self->list_tensorOp_us0[0],
	                   self->list_tensorOp_us0[1]);
}

void nn_engine_computeDispatch(nn_engine_t* self,
                               vkk_hazard_e hazard,
                               uint32_t count_x,
                               uint32_t count_y,
                               uint32_t count_z,
                               uint32_t local_size_x,
                               uint32_t local_size_y,
                               uint32_t local_size_z)
{
	ASSERT(self);

	vkk_compute_dispatch(self->compute, hazard,
	                     count_x, count_y, count_z,
	                     local_size_x, local_size_y,
	                     local_size_z);
	++self->dispatch;
}

int
nn_engine_computeBind(nn_engine_t* self,
                      vkk_computePipeline_t* cp)
{
	ASSERT(self);
	ASSERT(cp);

	// split dispatch to improve UI responsiveness
	if(self->dispatch >= NN_ENGINE_DISPATCH_HINT)
	{
		LOGD("DISPATCH %i", self->dispatch);

		self->dispatch = 0;

		vkk_compute_end(self->compute);
		if(vkk_compute_begin(self->compute) == 0)
		{
			return 0;
		}
	}

	vkk_compute_bindComputePipeline(self->compute, cp);

	return 1;
}
