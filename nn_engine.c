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
#include "nn_engine.h"
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

typedef struct
{
	uint32_t k;
} nn_batchNormIdxKey_t;

typedef struct
{
	vkk_buffer_t*     sb30;
	vkk_uniformSet_t* us3;
} nn_batchNormIdxData_t;

typedef struct
{
	uint32_t f;
	uint32_t fi;
	uint32_t fj;
	uint32_t k;
} nn_convIdxKey_t;

typedef struct
{
	vkk_buffer_t*     sb30;
	vkk_uniformSet_t* us3;
} nn_convIdxData_t;

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

	// sb00: state
	self->usf0_arch = vkk_uniformSetFactory_new(engine, um,
	                                            1, ub_array);

	// sb10:  dimXd
	// ...
	// sb19:  Yr
	self->usf1_arch = vkk_uniformSetFactory_new(engine, um,
	                                            10, ub_array);

	// sb20: dim_dL_dYg
	// ...
	// sb25: dL_dYb
	self->usf2_arch = vkk_uniformSetFactory_new(engine, um,
	                                            6, ub_array);

	// sb00: state
	// ...
	// sb08: Xvar_mb
	self->usf0_batchNorm = vkk_uniformSetFactory_new(engine, um,
	                                                 9, ub_array);

	// sb10:  dimX
	// ...
	// sb113: Xvar_ra
	self->usf1_batchNorm = vkk_uniformSetFactory_new(engine, um,
	                                                 14, ub_array);

	// sb20: dim_dL_dXhat
	// ...
	// sb211: VB
	self->usf2_batchNorm = vkk_uniformSetFactory_new(engine, um,
	                                                 12, ub_array);

	// sb30: idx
	self->usf3_batchNorm = vkk_uniformSetFactory_new(engine, um,
	                                                 1, ub_array);

	// sb00: state
	// ...
	// sb07: B
	self->usf0_conv = vkk_uniformSetFactory_new(engine, um,
	                                            8, ub_array);

	// sb10: dimY
	// sb11: Y
	self->usf1_conv = vkk_uniformSetFactory_new(engine, um,
	                                            2, ub_array);

	// sb20:  dim_dL_dY
	// ...
	// sb211: VB
	self->usf2_conv = vkk_uniformSetFactory_new(engine, um,
	                                            12, ub_array);

	// sb30:  idx
	self->usf3_conv = vkk_uniformSetFactory_new(engine, um,
	                                            1, ub_array);

	// sb00: dimX
	// sb01: X
	self->usf0_fact = vkk_uniformSetFactory_new(engine, um,
	                                            2, ub_array);

	// sb10: dimY
	// sb11: Y
	self->usf1_fact = vkk_uniformSetFactory_new(engine, um,
	                                            2, ub_array);

	// sb20: dim_dL_dY
	// sb21: dL_dY
	self->usf2_fact = vkk_uniformSetFactory_new(engine, um,
	                                            2, ub_array);

	// sb00: state
	// sb01: param (beta)
	self->usf0_skip = vkk_uniformSetFactory_new(engine, um,
	                                            2, ub_array);

	// sb10: dimX/dimX1
	// ...
	// sb15: X2
	self->usf1_skip = vkk_uniformSetFactory_new(engine, um,
	                                            6, ub_array);

	// sb20: dim_dL_dY
	// ...
	// sb27: dL_dY2
	self->usf2_skip = vkk_uniformSetFactory_new(engine, um,
	                                            8, ub_array);

	// sb00: state
	// ...
	// sb07: B
	self->usf0_weight = vkk_uniformSetFactory_new(engine, um,
	                                              8, ub_array);

	// sb10: dimY
	// sb11: Y
	self->usf1_weight = vkk_uniformSetFactory_new(engine, um,
	                                              2, ub_array);

	// sb20:  dim_dL_dY
	// ...
	// sb211: VB
	self->usf2_weight = vkk_uniformSetFactory_new(engine, um,
	                                              12, ub_array);

	// sb00: state
	// ...
	// sb07: loss
	self->usf0_loss = vkk_uniformSetFactory_new(engine, um,
	                                            8, ub_array);

	// sb00: dimX
	// ...
	// sb02: stats
	self->usf0_tensor = vkk_uniformSetFactory_new(engine, um,
	                                              3, ub_array);

	// sb10: stats
	self->usf1_tensor = vkk_uniformSetFactory_new(engine, um,
	                                              1, ub_array);

	// sb20: u1
	// ...
	// sb24: c
	self->usf2_tensor = vkk_uniformSetFactory_new(engine, um,
	                                              5, ub_array);

	if((self->usf0_arch      == NULL) ||
	   (self->usf1_arch      == NULL) ||
	   (self->usf2_arch      == NULL) ||
	   (self->usf0_batchNorm == NULL) ||
	   (self->usf1_batchNorm == NULL) ||
	   (self->usf2_batchNorm == NULL) ||
	   (self->usf3_batchNorm == NULL) ||
	   (self->usf0_conv      == NULL) ||
	   (self->usf1_conv      == NULL) ||
	   (self->usf2_conv      == NULL) ||
	   (self->usf3_conv      == NULL) ||
	   (self->usf0_fact      == NULL) ||
	   (self->usf1_fact      == NULL) ||
	   (self->usf2_fact      == NULL) ||
	   (self->usf0_skip      == NULL) ||
	   (self->usf1_skip      == NULL) ||
	   (self->usf2_skip      == NULL) ||
	   (self->usf0_weight    == NULL) ||
	   (self->usf1_weight    == NULL) ||
	   (self->usf2_weight    == NULL) ||
	   (self->usf0_loss      == NULL) ||
	   (self->usf0_tensor    == NULL) ||
	   (self->usf1_tensor    == NULL) ||
	   (self->usf2_tensor    == NULL))
	{
		goto failure;
	}

	vkk_uniformSetFactory_t* usf_array_arch[] =
	{
		self->usf0_arch,
		self->usf1_arch,
		self->usf2_arch,
	};
	self->pl_arch = vkk_pipelineLayout_new(engine, 3,
	                                       usf_array_arch);

	vkk_uniformSetFactory_t* usf_array_batchNorm[] =
	{
		self->usf0_batchNorm,
		self->usf1_batchNorm,
		self->usf2_batchNorm,
		self->usf3_batchNorm,
	};
	self->pl_batchNorm = vkk_pipelineLayout_new(engine, 4,
	                                            usf_array_batchNorm);

	vkk_uniformSetFactory_t* usf_array_conv[] =
	{
		self->usf0_conv,
		self->usf1_conv,
		self->usf2_conv,
		self->usf3_conv,
	};
	self->pl_conv = vkk_pipelineLayout_new(engine, 4,
	                                       usf_array_conv);

	vkk_uniformSetFactory_t* usf_array_fact[] =
	{
		self->usf0_fact,
		self->usf1_fact,
		self->usf2_fact,
	};
	self->pl_fact = vkk_pipelineLayout_new(engine, 3,
	                                       usf_array_fact);

	vkk_uniformSetFactory_t* usf_array_skip[] =
	{
		self->usf0_skip,
		self->usf1_skip,
		self->usf2_skip,
	};
	self->pl_skip = vkk_pipelineLayout_new(engine, 3,
	                                       usf_array_skip);

	vkk_uniformSetFactory_t* usf_array_weight[] =
	{
		self->usf0_weight,
		self->usf1_weight,
		self->usf2_weight,
	};
	self->pl_weight = vkk_pipelineLayout_new(engine, 3,
	                                         usf_array_weight);

	vkk_uniformSetFactory_t* usf_array_loss[] =
	{
		self->usf0_loss,
	};
	self->pl_loss = vkk_pipelineLayout_new(engine, 1,
	                                       usf_array_loss);

	vkk_uniformSetFactory_t* usf_array_tensor[] =
	{
		self->usf0_tensor,
		self->usf1_tensor,
		self->usf2_tensor,
	};
	self->pl_tensor = vkk_pipelineLayout_new(engine, 3,
	                                         usf_array_tensor);

	if((self->pl_arch      == NULL) ||
	   (self->pl_batchNorm == NULL) ||
	   (self->pl_conv      == NULL) ||
	   (self->pl_fact      == NULL) ||
	   (self->pl_skip      == NULL) ||
	   (self->pl_weight    == NULL) ||
	   (self->pl_loss      == NULL) ||
	   (self->pl_tensor    == NULL))
	{
		goto failure;
	}

	vkk_computePipelineInfo_t cpi_arch_forwardPassFairCGAN =
	{
		.compute = self->compute,
		.pl      = self->pl_arch,
		.cs      = "nn/shaders/nn_arch_forwardPassFairCGAN_comp.spv",
	};

	self->cp_arch_forwardPassFairCGAN =
		vkk_computePipeline_new(engine,
		                        &cpi_arch_forwardPassFairCGAN);

	vkk_computePipelineInfo_t cpi_arch_backpropFairCGAN =
	{
		.compute = self->compute,
		.pl      = self->pl_arch,
		.cs      = "nn/shaders/nn_arch_backpropFairCGAN_comp.spv",
	};

	self->cp_arch_backpropFairCGAN =
		vkk_computePipeline_new(engine,
		                        &cpi_arch_backpropFairCGAN);

	vkk_computePipelineInfo_t cpi_batchNorm_forwardPassXmean =
	{
		.compute = self->compute,
		.pl      = self->pl_batchNorm,
		.cs      = "nn/shaders/nn_batchNormLayer_forwardPassXmean_comp.spv",
	};

	self->cp_batchNorm_forwardPassXmean =
		vkk_computePipeline_new(engine,
		                        &cpi_batchNorm_forwardPassXmean);

	vkk_computePipelineInfo_t cpi_batchNorm_forwardPassXvar =
	{
		.compute = self->compute,
		.pl      = self->pl_batchNorm,
		.cs      = "nn/shaders/nn_batchNormLayer_forwardPassXvar_comp.spv",
	};

	self->cp_batchNorm_forwardPassXvar =
		vkk_computePipeline_new(engine,
		                        &cpi_batchNorm_forwardPassXvar);

	vkk_computePipelineInfo_t cpi_batchNorm_forwardPassXhat =
	{
		.compute = self->compute,
		.pl      = self->pl_batchNorm,
		.cs      = "nn/shaders/nn_batchNormLayer_forwardPassXhat_comp.spv",
	};

	self->cp_batchNorm_forwardPassXhat =
		vkk_computePipeline_new(engine,
		                        &cpi_batchNorm_forwardPassXhat);

	vkk_computePipelineInfo_t cpi_batchNorm_forwardPassY =
	{
		.compute = self->compute,
		.pl      = self->pl_batchNorm,
		.cs      = "nn/shaders/nn_batchNormLayer_forwardPassY_comp.spv",
	};

	self->cp_batchNorm_forwardPassY =
		vkk_computePipeline_new(engine,
		                        &cpi_batchNorm_forwardPassY);

	vkk_computePipelineInfo_t cpi_batchNorm_backprop_dL_dX =
	{
		.compute = self->compute,
		.pl      = self->pl_batchNorm,
		.cs      = "nn/shaders/nn_batchNormLayer_backprop_dL_dX_comp.spv",
	};

	self->cp_batchNorm_backprop_dL_dX =
		vkk_computePipeline_new(engine,
		                        &cpi_batchNorm_backprop_dL_dX);

	vkk_computePipelineInfo_t cpi_batchNorm_backprop_dL_dXhat =
	{
		.compute = self->compute,
		.pl      = self->pl_batchNorm,
		.cs      = "nn/shaders/nn_batchNormLayer_backprop_dL_dXhat_comp.spv",
	};

	self->cp_batchNorm_backprop_dL_dXhat =
		vkk_computePipeline_new(engine,
		                        &cpi_batchNorm_backprop_dL_dXhat);

	vkk_computePipelineInfo_t cpi_batchNorm_backpropSum =
	{
		.compute = self->compute,
		.pl      = self->pl_batchNorm,
		.cs      = "nn/shaders/nn_batchNormLayer_backpropSum_comp.spv",
	};

	self->cp_batchNorm_backpropSum =
		vkk_computePipeline_new(engine,
		                        &cpi_batchNorm_backpropSum);

	vkk_computePipelineInfo_t cpi_batchNorm_backpropSumNOP =
	{
		.compute = self->compute,
		.pl      = self->pl_batchNorm,
		.cs      = "nn/shaders/nn_batchNormLayer_backpropSumNOP_comp.spv",
	};

	self->cp_batchNorm_backpropSumNOP =
		vkk_computePipeline_new(engine,
		                        &cpi_batchNorm_backpropSumNOP);

	vkk_computePipelineInfo_t cpi_conv_forwardPass =
	{
		.compute = self->compute,
		.pl      = self->pl_conv,
		.cs      = "nn/shaders/nn_convLayer_forwardPass_comp.spv",
	};

	self->cp_conv_forwardPass =
		vkk_computePipeline_new(engine,
		                        &cpi_conv_forwardPass);

	vkk_computePipelineInfo_t cpi_conv_forwardPassT =
	{
		.compute = self->compute,
		.pl      = self->pl_conv,
		.cs      = "nn/shaders/nn_convLayer_forwardPassT_comp.spv",
	};

	self->cp_conv_forwardPassT =
		vkk_computePipeline_new(engine,
		                        &cpi_conv_forwardPassT);

	vkk_computePipelineInfo_t cpi_conv_backprop_dL_dX =
	{
		.compute = self->compute,
		.pl      = self->pl_conv,
		.cs      = "nn/shaders/nn_convLayer_backprop_dL_dX_comp.spv",
	};

	self->cp_conv_backprop_dL_dX =
		vkk_computePipeline_new(engine,
		                        &cpi_conv_backprop_dL_dX);

	vkk_computePipelineInfo_t cpi_conv_backprop_dL_dW =
	{
		.compute = self->compute,
		.pl      = self->pl_conv,
		.cs      = "nn/shaders/nn_convLayer_backprop_dL_dW_comp.spv",
	};

	self->cp_conv_backprop_dL_dW =
		vkk_computePipeline_new(engine,
		                        &cpi_conv_backprop_dL_dW);

	vkk_computePipelineInfo_t cpi_conv_backprop_dL_dB =
	{
		.compute = self->compute,
		.pl      = self->pl_conv,
		.cs      = "nn/shaders/nn_convLayer_backprop_dL_dB_comp.spv",
	};

	self->cp_conv_backprop_dL_dB =
		vkk_computePipeline_new(engine,
		                        &cpi_conv_backprop_dL_dB);

	vkk_computePipelineInfo_t cpi_conv_backpropT_dL_dX =
	{
		.compute = self->compute,
		.pl      = self->pl_conv,
		.cs      = "nn/shaders/nn_convLayer_backpropT_dL_dX_comp.spv",
	};

	self->cp_conv_backpropT_dL_dX =
		vkk_computePipeline_new(engine,
		                        &cpi_conv_backpropT_dL_dX);

	vkk_computePipelineInfo_t cpi_conv_backpropT_dL_dW =
	{
		.compute = self->compute,
		.pl      = self->pl_conv,
		.cs      = "nn/shaders/nn_convLayer_backpropT_dL_dW_comp.spv",
	};

	self->cp_conv_backpropT_dL_dW =
		vkk_computePipeline_new(engine,
		                        &cpi_conv_backpropT_dL_dW);

	vkk_computePipelineInfo_t cpi_conv_backpropUpdateW =
	{
		.compute = self->compute,
		.pl      = self->pl_conv,
		.cs      = "nn/shaders/nn_convLayer_backpropUpdateW_comp.spv",
	};

	self->cp_conv_backpropUpdateW =
		vkk_computePipeline_new(engine,
		                        &cpi_conv_backpropUpdateW);

	vkk_computePipelineInfo_t cpi_conv_backpropUpdateB =
	{
		.compute = self->compute,
		.pl      = self->pl_conv,
		.cs      = "nn/shaders/nn_convLayer_backpropUpdateB_comp.spv",
	};

	self->cp_conv_backpropUpdateB =
		vkk_computePipeline_new(engine,
		                        &cpi_conv_backpropUpdateB);

	vkk_computePipelineInfo_t cpi_fact_forwardPassLinear =
	{
		.compute = self->compute,
		.pl      = self->pl_fact,
		.cs      = "nn/shaders/nn_factLayer_forwardPassLinear_comp.spv",
	};

	self->cp_fact_forwardPassLinear =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_forwardPassLinear);

	vkk_computePipelineInfo_t cpi_fact_forwardPassLogistic =
	{
		.compute = self->compute,
		.pl      = self->pl_fact,
		.cs      = "nn/shaders/nn_factLayer_forwardPassLogistic_comp.spv",
	};

	self->cp_fact_forwardPassLogistic =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_forwardPassLogistic);

	vkk_computePipelineInfo_t cpi_fact_forwardPassReLU =
	{
		.compute = self->compute,
		.pl      = self->pl_fact,
		.cs      = "nn/shaders/nn_factLayer_forwardPassReLU_comp.spv",
	};

	self->cp_fact_forwardPassReLU =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_forwardPassReLU);

	vkk_computePipelineInfo_t cpi_fact_forwardPassPReLU =
	{
		.compute = self->compute,
		.pl      = self->pl_fact,
		.cs      = "nn/shaders/nn_factLayer_forwardPassPReLU_comp.spv",
	};

	self->cp_fact_forwardPassPReLU =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_forwardPassPReLU);

	vkk_computePipelineInfo_t cpi_fact_forwardPassTanh =
	{
		.compute = self->compute,
		.pl      = self->pl_fact,
		.cs      = "nn/shaders/nn_factLayer_forwardPassTanh_comp.spv",
	};

	self->cp_fact_forwardPassTanh =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_forwardPassTanh);

	vkk_computePipelineInfo_t cpi_fact_forwardPassSink =
	{
		.compute = self->compute,
		.pl      = self->pl_fact,
		.cs      = "nn/shaders/nn_factLayer_forwardPassSink_comp.spv",
	};

	self->cp_fact_forwardPassSink =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_forwardPassSink);

	vkk_computePipelineInfo_t cpi_fact_backpropLinear =
	{
		.compute = self->compute,
		.pl      = self->pl_fact,
		.cs      = "nn/shaders/nn_factLayer_backpropLinear_comp.spv",
	};

	self->cp_fact_backpropLinear =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_backpropLinear);

	vkk_computePipelineInfo_t cpi_fact_backpropLogistic =
	{
		.compute = self->compute,
		.pl      = self->pl_fact,
		.cs      = "nn/shaders/nn_factLayer_backpropLogistic_comp.spv",
	};

	self->cp_fact_backpropLogistic =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_backpropLogistic);

	vkk_computePipelineInfo_t cpi_fact_backpropReLU =
	{
		.compute = self->compute,
		.pl      = self->pl_fact,
		.cs      = "nn/shaders/nn_factLayer_backpropReLU_comp.spv",
	};

	self->cp_fact_backpropReLU =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_backpropReLU);

	vkk_computePipelineInfo_t cpi_fact_backpropPReLU =
	{
		.compute = self->compute,
		.pl      = self->pl_fact,
		.cs      = "nn/shaders/nn_factLayer_backpropPReLU_comp.spv",
	};

	self->cp_fact_backpropPReLU =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_backpropPReLU);

	vkk_computePipelineInfo_t cpi_fact_backpropTanh =
	{
		.compute = self->compute,
		.pl      = self->pl_fact,
		.cs      = "nn/shaders/nn_factLayer_backpropTanh_comp.spv",
	};

	self->cp_fact_backpropTanh =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_backpropTanh);

	vkk_computePipelineInfo_t cpi_fact_backpropSink =
	{
		.compute = self->compute,
		.pl      = self->pl_fact,
		.cs      = "nn/shaders/nn_factLayer_backpropSink_comp.spv",
	};

	self->cp_fact_backpropSink =
		vkk_computePipeline_new(engine,
		                        &cpi_fact_backpropSink);

	vkk_computePipelineInfo_t cpi_skip_forwardPassAdd =
	{
		.compute = self->compute,
		.pl      = self->pl_skip,
		.cs      = "nn/shaders/nn_skipLayer_forwardPassAdd_comp.spv",
	};

	self->cp_skip_forwardPassAdd =
		vkk_computePipeline_new(engine,
		                        &cpi_skip_forwardPassAdd);

	vkk_computePipelineInfo_t cpi_skip_forwardPassCat =
	{
		.compute = self->compute,
		.pl      = self->pl_skip,
		.cs      = "nn/shaders/nn_skipLayer_forwardPassCat_comp.spv",
	};

	self->cp_skip_forwardPassCat =
		vkk_computePipeline_new(engine,
		                        &cpi_skip_forwardPassCat);

	vkk_computePipelineInfo_t cpi_skip_backpropAdd =
	{
		.compute = self->compute,
		.pl      = self->pl_skip,
		.cs      = "nn/shaders/nn_skipLayer_backpropAdd_comp.spv",
	};

	self->cp_skip_backpropAdd =
		vkk_computePipeline_new(engine,
		                        &cpi_skip_backpropAdd);

	vkk_computePipelineInfo_t cpi_skip_backpropCat =
	{
		.compute = self->compute,
		.pl      = self->pl_skip,
		.cs      = "nn/shaders/nn_skipLayer_backpropCat_comp.spv",
	};

	self->cp_skip_backpropCat =
		vkk_computePipeline_new(engine,
		                        &cpi_skip_backpropCat);

	vkk_computePipelineInfo_t cpi_skip_backpropFork =
	{
		.compute = self->compute,
		.pl      = self->pl_skip,
		.cs      = "nn/shaders/nn_skipLayer_backpropFork_comp.spv",
	};

	self->cp_skip_backpropFork =
		vkk_computePipeline_new(engine,
		                        &cpi_skip_backpropFork);

	vkk_computePipelineInfo_t cpi_weight_forwardPass =
	{
		.compute = self->compute,
		.pl      = self->pl_weight,
		.cs      = "nn/shaders/nn_weightLayer_forwardPass_comp.spv",
	};

	self->cp_weight_forwardPass =
		vkk_computePipeline_new(engine,
		                        &cpi_weight_forwardPass);

	vkk_computePipelineInfo_t cpi_weight_backpropUpdateW =
	{
		.compute = self->compute,
		.pl      = self->pl_weight,
		.cs      = "nn/shaders/nn_weightLayer_backpropUpdateW_comp.spv",
	};

	self->cp_weight_backpropUpdateW =
		vkk_computePipeline_new(engine,
		                        &cpi_weight_backpropUpdateW);

	vkk_computePipelineInfo_t cpi_weight_backpropUpdateB =
	{
		.compute = self->compute,
		.pl      = self->pl_weight,
		.cs      = "nn/shaders/nn_weightLayer_backpropUpdateB_comp.spv",
	};

	self->cp_weight_backpropUpdateB =
		vkk_computePipeline_new(engine,
		                        &cpi_weight_backpropUpdateB);

	vkk_computePipelineInfo_t cpi_weight_backprop_dL_dX =
	{
		.compute = self->compute,
		.pl      = self->pl_weight,
		.cs      = "nn/shaders/nn_weightLayer_backprop_dL_dX_comp.spv",
	};

	self->cp_weight_backprop_dL_dX =
		vkk_computePipeline_new(engine,
		                        &cpi_weight_backprop_dL_dX);

	vkk_computePipelineInfo_t cpi_weight_backprop_dL_dW =
	{
		.compute = self->compute,
		.pl      = self->pl_weight,
		.cs      = "nn/shaders/nn_weightLayer_backprop_dL_dW_comp.spv",
	};

	self->cp_weight_backprop_dL_dW =
		vkk_computePipeline_new(engine,
		                        &cpi_weight_backprop_dL_dW);

	vkk_computePipelineInfo_t cpi_weight_backprop_dL_dB =
	{
		.compute = self->compute,
		.pl      = self->pl_weight,
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

	vkk_computePipelineInfo_t cpi_tensor_clear =
	{
		.compute = self->compute,
		.pl      = self->pl_tensor,
		.cs      = "nn/shaders/nn_tensor_clear_comp.spv",
	};

	self->cp_tensor_clear =
		vkk_computePipeline_new(engine,
		                        &cpi_tensor_clear);

	vkk_computePipelineInfo_t cpi_tensor_clearAligned =
	{
		.compute = self->compute,
		.pl      = self->pl_tensor,
		.cs      = "nn/shaders/nn_tensor_clearAligned_comp.spv",
	};

	self->cp_tensor_clearAligned =
		vkk_computePipeline_new(engine,
		                        &cpi_tensor_clearAligned);

	vkk_computePipelineInfo_t cpi_tensor_stats =
	{
		.compute = self->compute,
		.pl      = self->pl_tensor,
		.cs      = "nn/shaders/nn_tensor_stats_comp.spv",
	};

	self->cp_tensor_stats =
		vkk_computePipeline_new(engine,
		                        &cpi_tensor_stats);

	vkk_computePipelineInfo_t cpi_tensor_sn =
	{
		.compute = self->compute,
		.pl      = self->pl_tensor,
		.cs      = "nn/shaders/nn_tensor_sn_comp.spv",
	};

	self->cp_tensor_sn =
		vkk_computePipeline_new(engine,
		                        &cpi_tensor_sn);

	vkk_computePipelineInfo_t cpi_tensor_bssn =
	{
		.compute = self->compute,
		.pl      = self->pl_tensor,
		.cs      = "nn/shaders/nn_tensor_bssn_comp.spv",
	};

	self->cp_tensor_bssn =
		vkk_computePipeline_new(engine,
		                        &cpi_tensor_bssn);

	if((self->cp_arch_forwardPassFairCGAN        == NULL) ||
	   (self->cp_arch_backpropFairCGAN           == NULL) ||
	   (self->cp_batchNorm_forwardPassXmean      == NULL) ||
	   (self->cp_batchNorm_forwardPassXvar       == NULL) ||
	   (self->cp_batchNorm_forwardPassXhat       == NULL) ||
	   (self->cp_batchNorm_forwardPassY          == NULL) ||
	   (self->cp_batchNorm_backprop_dL_dX        == NULL) ||
	   (self->cp_batchNorm_backprop_dL_dXhat     == NULL) ||
	   (self->cp_batchNorm_backpropSum           == NULL) ||
	   (self->cp_batchNorm_backpropSumNOP        == NULL) ||
	   (self->cp_conv_forwardPass                == NULL) ||
	   (self->cp_conv_forwardPassT               == NULL) ||
	   (self->cp_conv_backprop_dL_dX             == NULL) ||
	   (self->cp_conv_backprop_dL_dW             == NULL) ||
	   (self->cp_conv_backprop_dL_dB             == NULL) ||
	   (self->cp_conv_backpropT_dL_dX            == NULL) ||
	   (self->cp_conv_backpropT_dL_dW            == NULL) ||
	   (self->cp_conv_backpropUpdateW            == NULL) ||
	   (self->cp_conv_backpropUpdateB            == NULL) ||
	   (self->cp_fact_forwardPassLinear          == NULL) ||
	   (self->cp_fact_forwardPassLogistic        == NULL) ||
	   (self->cp_fact_forwardPassReLU            == NULL) ||
	   (self->cp_fact_forwardPassPReLU           == NULL) ||
	   (self->cp_fact_forwardPassTanh            == NULL) ||
	   (self->cp_fact_forwardPassSink            == NULL) ||
	   (self->cp_fact_backpropLinear             == NULL) ||
	   (self->cp_fact_backpropLogistic           == NULL) ||
	   (self->cp_fact_backpropReLU               == NULL) ||
	   (self->cp_fact_backpropPReLU              == NULL) ||
	   (self->cp_fact_backpropTanh               == NULL) ||
	   (self->cp_fact_backpropSink               == NULL) ||
	   (self->cp_skip_forwardPassAdd             == NULL) ||
	   (self->cp_skip_forwardPassCat             == NULL) ||
	   (self->cp_skip_backpropAdd                == NULL) ||
	   (self->cp_skip_backpropCat                == NULL) ||
	   (self->cp_skip_backpropFork               == NULL) ||
	   (self->cp_weight_forwardPass              == NULL) ||
	   (self->cp_weight_backpropUpdateW          == NULL) ||
	   (self->cp_weight_backpropUpdateB          == NULL) ||
	   (self->cp_weight_backprop_dL_dX           == NULL) ||
	   (self->cp_weight_backprop_dL_dW           == NULL) ||
	   (self->cp_weight_backprop_dL_dB           == NULL) ||
	   (self->cp_loss_dL_dY_mse                  == NULL) ||
	   (self->cp_loss_dL_dY_mae                  == NULL) ||
	   (self->cp_loss_dL_dY_bce                  == NULL) ||
	   (self->cp_loss_mse                        == NULL) ||
	   (self->cp_loss_mae                        == NULL) ||
	   (self->cp_loss_bce                        == NULL) ||
	   (self->cp_tensor_clear                    == NULL) ||
	   (self->cp_tensor_clearAligned             == NULL) ||
	   (self->cp_tensor_stats                    == NULL) ||
	   (self->cp_tensor_sn                       == NULL) ||
	   (self->cp_tensor_bssn                     == NULL))
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

	self->map_batchNormIdx = cc_map_new();
	if(self->map_batchNormIdx == NULL)
	{
		goto failure;
	}

	self->map_convIdx = cc_map_new();
	if(self->map_convIdx == NULL)
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
		cc_mapIter_t* miter;
		if(self->map_batchNormIdx)
		{
			miter = cc_map_head(self->map_batchNormIdx);
			while(miter)
			{
				nn_batchNormIdxData_t* data;
				data = (nn_batchNormIdxData_t*)
				       cc_map_remove(self->map_batchNormIdx, &miter);
				vkk_uniformSet_delete(&data->us3);
				vkk_buffer_delete(&data->sb30);
				FREE(data);
			}
			cc_map_delete(&self->map_batchNormIdx);
		}

		if(self->map_convIdx)
		{
			miter = cc_map_head(self->map_convIdx);
			while(miter)
			{
				nn_convIdxData_t* data;
				data = (nn_convIdxData_t*)
				       cc_map_remove(self->map_convIdx, &miter);
				vkk_uniformSet_delete(&data->us3);
				vkk_buffer_delete(&data->sb30);
				FREE(data);
			}
			cc_map_delete(&self->map_convIdx);
		}

		nn_tensor_delete(&self->Null);
		vkk_computePipeline_delete(&self->cp_tensor_bssn);
		vkk_computePipeline_delete(&self->cp_tensor_sn);
		vkk_computePipeline_delete(&self->cp_tensor_stats);
		vkk_computePipeline_delete(&self->cp_tensor_clearAligned);
		vkk_computePipeline_delete(&self->cp_tensor_clear);
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
		vkk_computePipeline_delete(&self->cp_fact_backpropSink);
		vkk_computePipeline_delete(&self->cp_fact_backpropTanh);
		vkk_computePipeline_delete(&self->cp_fact_backpropPReLU);
		vkk_computePipeline_delete(&self->cp_fact_backpropReLU);
		vkk_computePipeline_delete(&self->cp_fact_backpropLogistic);
		vkk_computePipeline_delete(&self->cp_fact_backpropLinear);
		vkk_computePipeline_delete(&self->cp_fact_forwardPassSink);
		vkk_computePipeline_delete(&self->cp_fact_forwardPassTanh);
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
		vkk_computePipeline_delete(&self->cp_batchNorm_forwardPassXvar);
		vkk_computePipeline_delete(&self->cp_batchNorm_forwardPassXmean);
		vkk_computePipeline_delete(&self->cp_arch_backpropFairCGAN);
		vkk_computePipeline_delete(&self->cp_arch_forwardPassFairCGAN);
		vkk_pipelineLayout_delete(&self->pl_tensor);
		vkk_pipelineLayout_delete(&self->pl_loss);
		vkk_pipelineLayout_delete(&self->pl_weight);
		vkk_pipelineLayout_delete(&self->pl_skip);
		vkk_pipelineLayout_delete(&self->pl_fact);
		vkk_pipelineLayout_delete(&self->pl_conv);
		vkk_pipelineLayout_delete(&self->pl_batchNorm);
		vkk_pipelineLayout_delete(&self->pl_arch);
		vkk_uniformSetFactory_delete(&self->usf2_tensor);
		vkk_uniformSetFactory_delete(&self->usf1_tensor);
		vkk_uniformSetFactory_delete(&self->usf0_tensor);
		vkk_uniformSetFactory_delete(&self->usf0_loss);
		vkk_uniformSetFactory_delete(&self->usf2_weight);
		vkk_uniformSetFactory_delete(&self->usf1_weight);
		vkk_uniformSetFactory_delete(&self->usf0_weight);
		vkk_uniformSetFactory_delete(&self->usf2_skip);
		vkk_uniformSetFactory_delete(&self->usf1_skip);
		vkk_uniformSetFactory_delete(&self->usf0_skip);
		vkk_uniformSetFactory_delete(&self->usf2_fact);
		vkk_uniformSetFactory_delete(&self->usf1_fact);
		vkk_uniformSetFactory_delete(&self->usf0_fact);
		vkk_uniformSetFactory_delete(&self->usf3_conv);
		vkk_uniformSetFactory_delete(&self->usf2_conv);
		vkk_uniformSetFactory_delete(&self->usf1_conv);
		vkk_uniformSetFactory_delete(&self->usf0_conv);
		vkk_uniformSetFactory_delete(&self->usf3_batchNorm);
		vkk_uniformSetFactory_delete(&self->usf2_batchNorm);
		vkk_uniformSetFactory_delete(&self->usf1_batchNorm);
		vkk_uniformSetFactory_delete(&self->usf0_batchNorm);
		vkk_uniformSetFactory_delete(&self->usf2_arch);
		vkk_uniformSetFactory_delete(&self->usf1_arch);
		vkk_uniformSetFactory_delete(&self->usf0_arch);
		vkk_compute_delete(&self->compute);
		FREE(self);
		*_self = NULL;
	}
}

vkk_uniformSet_t*
nn_engine_getBatchNormIdx(nn_engine_t* self, uint32_t k)
{
	ASSERT(self);

	nn_batchNormIdxData_t* data;

	nn_batchNormIdxKey_t key =
	{
		.k = k,
	};

	// find an existing idx
	cc_mapIter_t* miter;
	miter = cc_map_findp(self->map_batchNormIdx,
	                     sizeof(nn_batchNormIdxKey_t),
	                     &key);
	if(miter)
	{
		data = (nn_batchNormIdxData_t*) cc_map_val(miter);
		return data->us3;
	}

	data = (nn_batchNormIdxData_t*)
	       CALLOC(1, sizeof(nn_batchNormIdxData_t));
	if(data == NULL)
	{
		LOGE("CALLOC failed");
		return NULL;
	}

	data->sb30 = vkk_buffer_new(self->engine,
	                            VKK_UPDATE_MODE_STATIC,
	                            VKK_BUFFER_USAGE_STORAGE,
	                            sizeof(nn_batchNormIdxKey_t),
	                            &key);
	if(data->sb30 == NULL)
	{
		goto fail_sb30;
	}

	data->us3 = vkk_uniformSet_new(self->engine, 3, 0, NULL,
	                               self->usf3_batchNorm);
	if(data->us3 == NULL)
	{
		goto fail_us3;
	}

	if(cc_map_addp(self->map_batchNormIdx,
	               data, sizeof(nn_batchNormIdxKey_t),
	               &key) == NULL)
	{
		goto fail_add;
	}

	vkk_uniformAttachment_t ua3_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = data->sb30,
		},
	};

	vkk_compute_updateUniformSetRefs(self->compute, data->us3,
	                                 1, ua3_array);

	// success
	return data->us3;

	// failure
	fail_add:
		vkk_uniformSet_delete(&data->us3);
	fail_us3:
		vkk_buffer_delete(&data->sb30);
	fail_sb30:
		FREE(data);
	return NULL;
}

vkk_uniformSet_t*
nn_engine_getConvIdx(nn_engine_t* self,
                     uint32_t f, uint32_t fi,
                     uint32_t fj, uint32_t k)
{
	ASSERT(self);

	nn_convIdxData_t* data;

	nn_convIdxKey_t key =
	{
		.f  = f,
		.fi = fi,
		.fj = fj,
		.k  = k,
	};

	// find an existing idx
	cc_mapIter_t* miter;
	miter = cc_map_findp(self->map_convIdx,
	                     sizeof(nn_convIdxKey_t),
	                     &key);
	if(miter)
	{
		data = (nn_convIdxData_t*) cc_map_val(miter);
		return data->us3;
	}

	data = (nn_convIdxData_t*)
	       CALLOC(1, sizeof(nn_convIdxData_t));
	if(data == NULL)
	{
		LOGE("CALLOC failed");
		return NULL;
	}

	data->sb30 = vkk_buffer_new(self->engine,
	                            VKK_UPDATE_MODE_STATIC,
	                            VKK_BUFFER_USAGE_STORAGE,
	                            sizeof(nn_convIdxKey_t),
	                            &key);
	if(data->sb30 == NULL)
	{
		goto fail_sb30;
	}

	data->us3 = vkk_uniformSet_new(self->engine, 3, 0, NULL,
	                               self->usf3_conv);
	if(data->us3 == NULL)
	{
		goto fail_us3;
	}

	if(cc_map_addp(self->map_convIdx,
	               data, sizeof(nn_convIdxKey_t),
	               &key) == NULL)
	{
		goto fail_add;
	}

	vkk_uniformAttachment_t ua3_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = data->sb30,
		},
	};

	vkk_compute_updateUniformSetRefs(self->compute, data->us3,
	                                 1, ua3_array);

	// success
	return data->us3;

	// failure
	fail_add:
		vkk_uniformSet_delete(&data->us3);
	fail_us3:
		vkk_buffer_delete(&data->sb30);
	fail_sb30:
		FREE(data);
	return NULL;
}

int nn_engine_begin(nn_engine_t* self)
{
	ASSERT(self);

	return vkk_compute_begin(self->compute);
}

void nn_engine_end(nn_engine_t* self)
{
	ASSERT(self);

	if(self->dispatch)
	{
		LOGD("DISPATCH %i", self->dispatch);
		self->dispatch = 0;
	}

	vkk_compute_end(self->compute);
}

void nn_engine_dispatch(nn_engine_t* self,
                        vkk_hazzard_e hazzard,
                        uint32_t count_x,
                        uint32_t count_y,
                        uint32_t count_z,
                        uint32_t local_size_x,
                        uint32_t local_size_y,
                        uint32_t local_size_z)
{
	ASSERT(self);

	vkk_compute_dispatch(self->compute, hazzard,
	                     count_x, count_y, count_z,
	                     local_size_x, local_size_y,
	                     local_size_z);
	++self->dispatch;
}

int
nn_engine_bind(nn_engine_t* self,
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
