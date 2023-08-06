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
#include "nn_arch.h"
#include "nn_layer.h"
#include "nn_loss.h"
#include "nn_tensor.h"

#define NN_ARCH_THREADS 4

/***********************************************************
* private                                                  *
***********************************************************/

#ifdef NN_USE_COMPUTE

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

static void nn_arch_deleteCompute(nn_arch_t* self)
{
	ASSERT(self);

	cc_mapIter_t* miter;
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

	nn_tensor_delete(&self->Yt);
	nn_tensor_delete(&self->X);
	nn_tensor_delete(&self->Null);
	vkk_buffer_delete(&self->sb_state);
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
	vkk_computePipeline_delete(&self->cp_weight_backpropGradientClipping);
	vkk_computePipeline_delete(&self->cp_weight_forwardPass);
	vkk_computePipeline_delete(&self->cp_skip_backpropFork);
	vkk_computePipeline_delete(&self->cp_skip_backpropCat);
	vkk_computePipeline_delete(&self->cp_skip_forwardPassCat);
	vkk_computePipeline_delete(&self->cp_skip_forwardPassAdd);
	vkk_computePipeline_delete(&self->cp_pooling_backprop);
	vkk_computePipeline_delete(&self->cp_pooling_forwardPassMax);
	vkk_computePipeline_delete(&self->cp_pooling_forwardPassAvg);
	vkk_computePipeline_delete(&self->cp_fact_backpropTanh);
	vkk_computePipeline_delete(&self->cp_fact_backpropPReLU);
	vkk_computePipeline_delete(&self->cp_fact_backpropReLU);
	vkk_computePipeline_delete(&self->cp_fact_backpropLogistic);
	vkk_computePipeline_delete(&self->cp_fact_backpropLinear);
	vkk_computePipeline_delete(&self->cp_fact_forwardPassTanh);
	vkk_computePipeline_delete(&self->cp_fact_forwardPassPReLU);
	vkk_computePipeline_delete(&self->cp_fact_forwardPassReLU);
	vkk_computePipeline_delete(&self->cp_fact_forwardPassLogistic);
	vkk_computePipeline_delete(&self->cp_fact_forwardPassLinear);
	vkk_computePipeline_delete(&self->cp_conv_backpropUpdateB);
	vkk_computePipeline_delete(&self->cp_conv_backpropUpdateW);
	vkk_computePipeline_delete(&self->cp_conv_backpropGradientClipping);
	vkk_computePipeline_delete(&self->cp_conv_backpropT_dL_dW);
	vkk_computePipeline_delete(&self->cp_conv_backpropT_dL_dX);
	vkk_computePipeline_delete(&self->cp_conv_backprop_dL_dB);
	vkk_computePipeline_delete(&self->cp_conv_backprop_dL_dW);
	vkk_computePipeline_delete(&self->cp_conv_backprop_dL_dX);
	vkk_computePipeline_delete(&self->cp_conv_forwardPassT);
	vkk_computePipeline_delete(&self->cp_conv_forwardPass);
	vkk_computePipeline_delete(&self->cp_batchNorm_backpropSum);
	vkk_computePipeline_delete(&self->cp_batchNorm_backprop_dL_dXhat);
	vkk_computePipeline_delete(&self->cp_batchNorm_backprop_dL_dX);
	vkk_computePipeline_delete(&self->cp_batchNorm_forwardPassY);
	vkk_computePipeline_delete(&self->cp_batchNorm_forwardPassXhat);
	vkk_computePipeline_delete(&self->cp_batchNorm_forwardPassXvar);
	vkk_computePipeline_delete(&self->cp_batchNorm_forwardPassXmean);
	vkk_pipelineLayout_delete(&self->pl_tensor);
	vkk_pipelineLayout_delete(&self->pl_loss);
	vkk_pipelineLayout_delete(&self->pl_weight);
	vkk_pipelineLayout_delete(&self->pl_skip);
	vkk_pipelineLayout_delete(&self->pl_pooling);
	vkk_pipelineLayout_delete(&self->pl_fact);
	vkk_pipelineLayout_delete(&self->pl_conv);
	vkk_pipelineLayout_delete(&self->pl_batchNorm);
	vkk_uniformSetFactory_delete(&self->usf0_tensor);
	vkk_uniformSetFactory_delete(&self->usf0_loss);
	vkk_uniformSetFactory_delete(&self->usf2_weight);
	vkk_uniformSetFactory_delete(&self->usf1_weight);
	vkk_uniformSetFactory_delete(&self->usf0_weight);
	vkk_uniformSetFactory_delete(&self->usf1_skip);
	vkk_uniformSetFactory_delete(&self->usf0_skip);
	vkk_uniformSetFactory_delete(&self->usf2_pooling);
	vkk_uniformSetFactory_delete(&self->usf1_pooling);
	vkk_uniformSetFactory_delete(&self->usf0_pooling);
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
	vkk_compute_delete(&self->compute);
}

static void
nn_arch_initUbArray(vkk_uniformBinding_t* ub_array,
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

static int
nn_arch_newCompute(nn_arch_t* self, vkk_engine_t* engine)
{
	ASSERT(self);
	ASSERT(engine);

	// call nn_arch_deleteCompute to handle errors

	self->compute = vkk_compute_new(engine);
	if(self->compute == NULL)
	{
		return 0;
	}

	vkk_updateMode_e um;
	um = vkk_compute_updateMode(self->compute);

	// all ub_arrays will contain storage buffer references
	// but each usf may have a different count
	// see readme.md for more details
	vkk_uniformBinding_t ub_array[20] = { 0 };
	nn_arch_initUbArray(ub_array, 20);

	// sb00: state
	// ...
	// sb09: Xvar_mb
	self->usf0_batchNorm = vkk_uniformSetFactory_new(engine, um,
	                                                 10, ub_array);

	// sb10:  dimX
	// ...
	// sb113: Xvar_ra
	self->usf1_batchNorm = vkk_uniformSetFactory_new(engine, um,
	                                                 14, ub_array);

	// sb20: dim_dL_dXhat
	// ...
	// sb27: Csum
	self->usf2_batchNorm = vkk_uniformSetFactory_new(engine, um,
	                                                 8, ub_array);

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

	// sb20:  gc
	// sb21:  dim_dL_dY
	// sb22:  dL_dY
	// sb23:  dim_dL_dW
	// sb24:  dL_dW
	// sb25:  dim_dL_dB
	// sb26:  dL_dB
	// sb27:  dim_dL_dX
	// sb28:  dL_dX
	// sb29:  dimVW
	// sb210: VW
	// sb211: dimVB
	// sb212: VB
	self->usf2_conv = vkk_uniformSetFactory_new(engine, um,
	                                            13, ub_array);

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
	// ...
	// sb03: dY_dX
	self->usf0_pooling = vkk_uniformSetFactory_new(engine, um,
	                                               4, ub_array);

	// sb10: dimX
	// ...
	// sb13: Y
	self->usf1_pooling = vkk_uniformSetFactory_new(engine, um,
	                                               4, ub_array);

	// sb20: dim_dL_dY
	// ...
	// sb23: dL_dX
	self->usf2_pooling = vkk_uniformSetFactory_new(engine, um,
	                                               4, ub_array);

	// sb00: dimX/dimX1
	// ...
	// sb05: X2
	self->usf0_skip = vkk_uniformSetFactory_new(engine, um,
	                                            6, ub_array);

	// sb10: dim_dL_dY
	// ...
	// sb17: dL_dY2
	self->usf1_skip = vkk_uniformSetFactory_new(engine, um,
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

	// sb20:  gc
	// ...
	// sb212: VB
	self->usf2_weight = vkk_uniformSetFactory_new(engine, um,
	                                              13, ub_array);

	// sb00: state
	// sb01: dimY
	// sb02: Y
	// sb03: dimYt
	// sb04: Yt
	// sb05: dim_dL_dY
	// sb06: dL_dY
	// sb07: loss
	self->usf0_loss = vkk_uniformSetFactory_new(engine, um,
	                                            8, ub_array);

	// sb00: dimX
	// sb01: X
	self->usf0_tensor = vkk_uniformSetFactory_new(engine, um,
	                                              2, ub_array);

	if((self->usf0_batchNorm == NULL) ||
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
	   (self->usf0_pooling   == NULL) ||
	   (self->usf1_pooling   == NULL) ||
	   (self->usf2_pooling   == NULL) ||
	   (self->usf0_skip      == NULL) ||
	   (self->usf1_skip      == NULL) ||
	   (self->usf0_weight    == NULL) ||
	   (self->usf1_weight    == NULL) ||
	   (self->usf2_weight    == NULL) ||
	   (self->usf0_loss      == NULL) ||
	   (self->usf0_tensor    == NULL))
	{
		nn_arch_deleteCompute(self);
		return 0;
	}

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

	vkk_uniformSetFactory_t* usf_array_pooling[] =
	{
		self->usf0_pooling,
		self->usf1_pooling,
		self->usf2_pooling,
	};
	self->pl_pooling = vkk_pipelineLayout_new(engine, 3,
	                                          usf_array_pooling);

	vkk_uniformSetFactory_t* usf_array_skip[] =
	{
		self->usf0_skip,
		self->usf1_skip,
	};
	self->pl_skip = vkk_pipelineLayout_new(engine, 2,
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
	};
	self->pl_tensor = vkk_pipelineLayout_new(engine, 1,
	                                         usf_array_tensor);

	if((self->pl_batchNorm == NULL) ||
	   (self->pl_conv      == NULL) ||
	   (self->pl_fact      == NULL) ||
	   (self->pl_pooling   == NULL) ||
	   (self->pl_skip      == NULL) ||
	   (self->pl_weight    == NULL) ||
	   (self->pl_loss      == NULL) ||
	   (self->pl_tensor    == NULL))
	{
		nn_arch_deleteCompute(self);
		return 0;
	}

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

	vkk_computePipelineInfo_t cpi_conv_backpropGradientClipping =
	{
		.compute = self->compute,
		.pl      = self->pl_conv,
		.cs      = "nn/shaders/nn_convLayer_backpropGradientClipping_comp.spv",
	};

	self->cp_conv_backpropGradientClipping =
		vkk_computePipeline_new(engine,
		                        &cpi_conv_backpropGradientClipping);

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

	vkk_computePipelineInfo_t cpi_pooling_forwardPassAvg =
	{
		.compute = self->compute,
		.pl      = self->pl_pooling,
		.cs      = "nn/shaders/nn_poolingLayer_forwardPassAvg_comp.spv",
	};

	self->cp_pooling_forwardPassAvg =
		vkk_computePipeline_new(engine,
		                        &cpi_pooling_forwardPassAvg);

	vkk_computePipelineInfo_t cpi_pooling_forwardPassMax =
	{
		.compute = self->compute,
		.pl      = self->pl_pooling,
		.cs      = "nn/shaders/nn_poolingLayer_forwardPassMax_comp.spv",
	};

	self->cp_pooling_forwardPassMax =
		vkk_computePipeline_new(engine,
		                        &cpi_pooling_forwardPassMax);

	vkk_computePipelineInfo_t cpi_pooling_backprop =
	{
		.compute = self->compute,
		.pl      = self->pl_pooling,
		.cs      = "nn/shaders/nn_poolingLayer_backprop_comp.spv",
	};

	self->cp_pooling_backprop =
		vkk_computePipeline_new(engine,
		                        &cpi_pooling_backprop);

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

	vkk_computePipelineInfo_t cpi_weight_backpropGradientClipping =
	{
		.compute = self->compute,
		.pl      = self->pl_weight,
		.cs      = "nn/shaders/nn_weightLayer_backpropGradientClipping_comp.spv",
	};

	self->cp_weight_backpropGradientClipping =
		vkk_computePipeline_new(engine,
		                        &cpi_weight_backpropGradientClipping);

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

	if((self->cp_batchNorm_forwardPassXmean      == NULL) ||
	   (self->cp_batchNorm_forwardPassXvar       == NULL) ||
	   (self->cp_batchNorm_forwardPassXhat       == NULL) ||
	   (self->cp_batchNorm_forwardPassY          == NULL) ||
	   (self->cp_batchNorm_backprop_dL_dX        == NULL) ||
	   (self->cp_batchNorm_backprop_dL_dXhat     == NULL) ||
	   (self->cp_batchNorm_backpropSum           == NULL) ||
	   (self->cp_conv_forwardPass                == NULL) ||
	   (self->cp_conv_forwardPassT               == NULL) ||
	   (self->cp_conv_backprop_dL_dX             == NULL) ||
	   (self->cp_conv_backprop_dL_dW             == NULL) ||
	   (self->cp_conv_backprop_dL_dB             == NULL) ||
	   (self->cp_conv_backpropT_dL_dX            == NULL) ||
	   (self->cp_conv_backpropT_dL_dW            == NULL) ||
	   (self->cp_conv_backpropGradientClipping   == NULL) ||
	   (self->cp_conv_backpropUpdateW            == NULL) ||
	   (self->cp_conv_backpropUpdateB            == NULL) ||
	   (self->cp_fact_forwardPassLinear          == NULL) ||
	   (self->cp_fact_forwardPassLogistic        == NULL) ||
	   (self->cp_fact_forwardPassReLU            == NULL) ||
	   (self->cp_fact_forwardPassPReLU           == NULL) ||
	   (self->cp_fact_forwardPassTanh            == NULL) ||
	   (self->cp_fact_backpropLinear             == NULL) ||
	   (self->cp_fact_backpropLogistic           == NULL) ||
	   (self->cp_fact_backpropReLU               == NULL) ||
	   (self->cp_fact_backpropPReLU              == NULL) ||
	   (self->cp_fact_backpropTanh               == NULL) ||
	   (self->cp_pooling_forwardPassAvg          == NULL) ||
	   (self->cp_pooling_forwardPassMax          == NULL) ||
	   (self->cp_pooling_backprop                == NULL) ||
	   (self->cp_skip_forwardPassAdd             == NULL) ||
	   (self->cp_skip_forwardPassCat             == NULL) ||
	   (self->cp_skip_backpropCat                == NULL) ||
	   (self->cp_skip_backpropFork               == NULL) ||
	   (self->cp_weight_forwardPass              == NULL) ||
	   (self->cp_weight_backpropGradientClipping == NULL) ||
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
	   (self->cp_tensor_clearAligned             == NULL))
	{
		nn_arch_deleteCompute(self);
		return 0;
	}

	self->sb_state = vkk_buffer_new(engine, um,
	                                VKK_BUFFER_USAGE_STORAGE,
	                                sizeof(nn_archState_t),
	                                NULL);
	if(self->sb_state == NULL)
	{
		nn_arch_deleteCompute(self);
		return 0;
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
		nn_arch_deleteCompute(self);
		return 0;
	}

	self->map_convIdx = cc_map_new();
	if(self->map_convIdx == NULL)
	{
		nn_arch_deleteCompute(self);
		return 0;
	}

	self->engine = engine;

	return 1;
}

static int
nn_arch_beginCompute(nn_arch_t* self,
                     uint32_t bs,
                     nn_tensor_t* X,
                     nn_tensor_t* Yt)
{
	// Yt may be NULL
	ASSERT(self);
	ASSERT(X);

	// resize X
	if(self->X)
	{
		if(nn_dim_equals(nn_tensor_dim(self->X),
		                 nn_tensor_dim(X)) == 0)
		{
			nn_tensor_delete(&self->X);
		}
	}

	// resize Yt
	if(Yt && self->Yt)
	{
		if(nn_dim_equals(nn_tensor_dim(self->Yt),
		                 nn_tensor_dim(Yt)) == 0)
		{
			nn_tensor_delete(&self->Yt);
		}
	}

	// allocate X on demand
	if(self->X == NULL)
	{
		self->X = nn_tensor_new(self, nn_tensor_dim(X),
		                        NN_TENSOR_INIT_ZERO,
		                        NN_TENSOR_MODE_COMPUTE);
		if(self->X == NULL)
		{
			return 0;
		}
	}

	// allocate Yt on demand
	if(Yt && (self->Yt == NULL))
	{
		self->Yt = nn_tensor_new(self, nn_tensor_dim(Yt),
		                         NN_TENSOR_INIT_ZERO,
		                         NN_TENSOR_MODE_COMPUTE);
		if(self->Yt == NULL)
		{
			return 0;
		}
	}

	if(vkk_compute_begin(self->compute) == 0)
	{
		return 0;
	}

	if(nn_tensor_blit(X, self->X, bs, 0, 0) == 0)
	{
		goto fail_blit;
	}

	if(Yt && (nn_tensor_blit(Yt, self->Yt, bs, 0, 0) == 0))
	{
		goto fail_blit;
	}

	// update global state
	self->state.bs = bs;
	vkk_compute_writeBuffer(self->compute,
	                        self->sb_state,
	                        sizeof(nn_archState_t),
	                        0, &self->state);

	// success
	return 1;

	// failure
	fail_blit:
		vkk_compute_end(self->compute);
	return 0;
}

static void nn_arch_endCompute(nn_arch_t* self)
{
	ASSERT(self);

	vkk_compute_end(self->compute);

	nn_loss_t* loss = self->loss;
	vkk_compute_readBuffer(self->compute, loss->sb07_loss,
	                       sizeof(float), 0, &loss->loss);
}

#else // NN_USE_COMPUTE not defined

static int
nn_arch_newCompute(nn_arch_t* self, vkk_engine_t* engine)
{
	return 1;
}

static void nn_arch_deleteCompute(nn_arch_t* self)
{
}

static int
nn_arch_beginCompute(nn_arch_t* self,
                     uint32_t bs,
                     nn_tensor_t* X,
                     nn_tensor_t* Yt)
{
	return 1;
}

static void nn_arch_endCompute(nn_arch_t* self)
{
}

#endif

/***********************************************************
* protected                                                *
***********************************************************/

#ifdef NN_USE_COMPUTE

vkk_uniformSet_t*
nn_arch_getConvIdx(nn_arch_t* self,
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

#endif

/***********************************************************
* public                                                   *
***********************************************************/

nn_arch_t* nn_arch_new(vkk_engine_t* engine,
                       size_t base_size,
                       nn_archState_t* state)
{
	// engine may be null
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

	memcpy(&self->state, state, sizeof(nn_archState_t));

	self->layers = cc_list_new();
	if(self->layers == NULL)
	{
		goto fail_layers;
	}

	cc_rngUniform_init(&self->rng_uniform);
	cc_rngNormal_init(&self->rng_normal, 0.0, 1.0);

	if(nn_arch_newCompute(self, engine) == 0)
	{
		goto fail_compute;
	}

	// success
	return self;

	// failure
	fail_compute:
		cc_list_delete(&self->layers);
	fail_layers:
		FREE(self);
	return NULL;
}

nn_arch_t*
nn_arch_import(vkk_engine_t* engine,
               size_t base_size, jsmn_val_t* val)
{
	// engine may be null
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	// bs not required
	jsmn_val_t* val_learning_rate  = NULL;
	jsmn_val_t* val_momentum_decay = NULL;
	jsmn_val_t* val_batch_momentum = NULL;
	jsmn_val_t* val_l2_lambda      = NULL;
	jsmn_val_t* val_clip_max       = NULL;
	jsmn_val_t* val_clip_momentum  = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		jsmn_keyval_t* kv;
		kv = (jsmn_keyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == JSMN_TYPE_PRIMITIVE)
		{
			if(strcmp(kv->key, "learning_rate") == 0)
			{
				val_learning_rate = kv->val;
			}
			else if(strcmp(kv->key, "momentum_decay") == 0)
			{
				val_momentum_decay = kv->val;
			}
			else if(strcmp(kv->key, "batch_momentum") == 0)
			{
				val_batch_momentum = kv->val;
			}
			else if(strcmp(kv->key, "l2_lambda") == 0)
			{
				val_l2_lambda = kv->val;
			}
			else if(strcmp(kv->key, "clip_max") == 0)
			{
				val_clip_max = kv->val;
			}
			else if(strcmp(kv->key, "clip_momentum") == 0)
			{
				val_clip_momentum = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_learning_rate  == NULL) ||
	   (val_momentum_decay == NULL) ||
	   (val_batch_momentum == NULL) ||
	   (val_l2_lambda      == NULL) ||
	   (val_clip_max       == NULL) ||
	   (val_clip_momentum  == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_archState_t state =
	{
		.learning_rate  = strtof(val_learning_rate->data,  NULL),
		.momentum_decay = strtof(val_momentum_decay->data, NULL),
		.batch_momentum = strtof(val_batch_momentum->data, NULL),
		.l2_lambda      = strtof(val_l2_lambda->data,      NULL),
		.clip_max       = strtof(val_clip_max->data,       NULL),
		.clip_momentum  = strtof(val_clip_momentum->data,  NULL),
	};

	return nn_arch_new(engine, base_size, &state);
}

int nn_arch_export(nn_arch_t* self, jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	nn_archState_t* state = &self->state;

	// bs not required
	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	ret &= jsmn_stream_key(stream, "%s", "learning_rate");
	ret &= jsmn_stream_float(stream, state->learning_rate);
	ret &= jsmn_stream_key(stream, "%s", "momentum_decay");
	ret &= jsmn_stream_float(stream, state->momentum_decay);
	ret &= jsmn_stream_key(stream, "%s", "batch_momentum");
	ret &= jsmn_stream_float(stream, state->batch_momentum);
	ret &= jsmn_stream_key(stream, "%s", "l2_lambda");
	ret &= jsmn_stream_float(stream, state->l2_lambda);
	ret &= jsmn_stream_key(stream, "%s", "clip_max");
	ret &= jsmn_stream_float(stream, state->clip_max);
	ret &= jsmn_stream_key(stream, "%s", "clip_momentum");
	ret &= jsmn_stream_float(stream, state->clip_momentum);
	ret &= jsmn_stream_end(stream);

	return ret;
}

void nn_arch_delete(nn_arch_t** _self)
{
	ASSERT(_self);

	nn_arch_t* self = *_self;
	if(self)
	{
		nn_arch_deleteCompute(self);
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

	// validate dimensions
	nn_layer_t* tail;
	tail = (nn_layer_t*) cc_list_peekTail(self->layers);
	if(tail)
	{
		if(nn_dim_equals(nn_layer_dimY(tail),
		                 nn_layer_dimX(layer)) == 0)
		{
			nn_dim_t* dimY = nn_layer_dimY(tail);
			nn_dim_t* dimX = nn_layer_dimX(layer);
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

	// validate dimensions
	nn_layer_t* tail;
	tail = (nn_layer_t*) cc_list_peekTail(self->layers);
	if((tail == NULL) ||
	   (nn_dim_equals(nn_layer_dimY(tail),
	                  nn_loss_dimY(loss)) == 0))
	{
		LOGE("invalid");
		return 0;
	}

	self->loss = loss;

	return 1;
}

int nn_arch_train(nn_arch_t* self,
                  uint32_t bs,
                  nn_tensor_t* X,
                  nn_tensor_t* Yt)
{
	ASSERT(self);
	ASSERT(X);
	ASSERT(Yt);

	if(nn_arch_beginCompute(self, bs, X, Yt) == 0)
	{
		return 0;
	}

	// replace X and Yt with compute tensors
	#ifdef NN_USE_COMPUTE
	X  = self->X;
	Yt = self->Yt;
	#endif

	// perform forward pass for each batch
	cc_listIter_t* iter = cc_list_head(self->layers);
	while(iter)
	{
		nn_layer_t* layer;
		layer = (nn_layer_t*) cc_list_peekIter(iter);

		X = nn_layer_forwardPass(layer,
		                         NN_LAYER_MODE_TRAIN,
		                         bs, X);
		if(X == NULL)
		{
			goto fail_forwardPass;
		}

		iter = cc_list_next(iter);
	}

	// backpropagate loss
	nn_tensor_t* dL_dY = NULL;
	{
		nn_loss_fn loss_fn;
		loss_fn = self->loss->loss_fn;

		dL_dY = (*loss_fn)(self->loss, bs, X, Yt);
		if(dL_dY == NULL)
		{
			goto fail_loss;
		}
	}

	// perform backpropagation
	iter = cc_list_tail(self->layers);
	while(iter)
	{
		nn_layer_t* layer;
		layer = (nn_layer_t*) cc_list_peekIter(iter);

		dL_dY = nn_layer_backprop(layer, bs, dL_dY);
		if(dL_dY == NULL)
		{
			goto fail_backprop;
		}

		iter = cc_list_prev(iter);
	}

	nn_arch_endCompute(self);

	// success
	return 1;

	// failure
	fail_backprop:
	fail_loss:
	fail_forwardPass:
		nn_arch_endCompute(self);
	return 0;
}

float nn_arch_loss(nn_arch_t* self)
{
	ASSERT(self);

	return self->loss->loss;
}

int nn_arch_predict(nn_arch_t* self,
                    nn_tensor_t* X,
                    nn_tensor_t* Y)
{
	ASSERT(self);
	ASSERT(X);
	ASSERT(Y);

	if(nn_arch_beginCompute(self, 1, X, NULL) == 0)
	{
		return 0;
	}

	// replace X with compute tensor
	#ifdef NN_USE_COMPUTE
	X = self->X;
	#endif

	cc_listIter_t* iter = cc_list_head(self->layers);
	while(iter)
	{
		nn_layer_t* layer;
		layer = (nn_layer_t*) cc_list_peekIter(iter);

		X = nn_layer_forwardPass(layer,
		                         NN_LAYER_MODE_PREDICT,
		                         1, X);
		if(X == NULL)
		{
			goto fail_forwardPass;
		}

		iter = cc_list_next(iter);
	}

	nn_arch_endCompute(self);

	// success
	return nn_tensor_blit(X, Y, 1, 0, 0);

	// failure
	fail_forwardPass:
		nn_arch_endCompute(self);
	return 0;
}
