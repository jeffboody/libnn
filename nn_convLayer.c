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

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define LOG_TAG "nn"
#include "../libcc/rng/cc_rngNormal.h"
#include "../libcc/rng/cc_rngUniform.h"
#include "../libcc/cc_log.h"
#include "../libcc/cc_memory.h"
#include "../libvkk/vkk.h"
#include "nn_arch.h"
#include "nn_convLayer.h"
#include "nn_engine.h"
#include "nn_layer.h"
#include "nn_tensorStats.h"
#include "nn_tensor.h"

/***********************************************************
* private                                                  *
***********************************************************/

typedef struct
{
	uint32_t disable_bias;
	uint32_t stride;
} nn_convLayerParam_t;

static nn_tensor_t*
nn_convLayer_forwardPassFn(nn_layer_t* base, int flags,
                           uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_convLayer_t* self   = (nn_convLayer_t*) base;
	nn_arch_t*      arch   = base->arch;
	nn_engine_t*    engine = arch->engine;

	nn_dim_t* dimY = nn_tensor_dim(self->Y);

	// optionally perform Spectral Normalization
	if(self->flags & NN_CONV_LAYER_FLAG_NORM_SN)
	{
		if(nn_tensor_normalize(self->W,
		                       VKK_HAZARD_NONE,
		                       NN_TENSOR_NORM_MODE_SN,
		                       1.0f) == 0)
		{
			return NULL;
		}
	}
	else if(self->flags & NN_CONV_LAYER_FLAG_NORM_BSSN)
	{
		if(nn_tensor_normalize(self->W,
		                       VKK_HAZARD_NONE,
		                       NN_TENSOR_NORM_MODE_BSSN,
		                       1.2f) == 0)
		{
			return NULL;
		}
	}

	// sb100: bs
	// sb101: state
	// sb102: X
	vkk_uniformAttachment_t ua1_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = arch->sb100_bs,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = arch->sb101_state,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X->sb_data,
		},
	};

	vkk_compute_updateUniformSetRefs(engine->compute,
	                                 self->us1_fp,
	                                 3, ua1_array);

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1_fp,
	};

	// nn_convLayer_forwardPass
	// dispatch(RAW, bs, yh, yw, 1, 8, 8)
	vkk_computePipeline_t* cp;
	cp = engine->cp_conv_forwardPass;
	if(nn_engine_bind(engine, cp) == 0)
	{
		return NULL;
	}
	vkk_compute_bindUniformSets(engine->compute, 2, us_array);
	nn_engine_dispatch(engine, VKK_HAZARD_RAW,
	                   bs, dimY->height, dimY->width,
	                   1, 8, 8);

	// store reference
	self->X = X;

	return self->Y;
}

static nn_tensor_t*
nn_convLayer_backpropFn(nn_layer_t* base,
                        int flags,
                        uint32_t bs,
                        nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,yh,yw,fc)

	nn_convLayer_t* self   = (nn_convLayer_t*) base;
	nn_arch_t*      arch   = base->arch;
	nn_engine_t*    engine = arch->engine;

	nn_dim_t* dimW = nn_tensor_dim(self->W);
	nn_dim_t* dimY = nn_tensor_dim(dL_dY);
	uint32_t  fc   = dimW->count;
	uint32_t  fh   = dimW->height;
	uint32_t  fw   = dimW->width;
	uint32_t  xd   = dimW->depth;

	// clear backprop gradients
	if(nn_tensor_clear(self->dL_dW, VKK_HAZARD_NONE) == 0)
	{
		return NULL;
	}

	if((self->flags & NN_CONV_LAYER_FLAG_DISABLE_BIAS) == 0)
	{
		if(nn_tensor_clear(self->dL_dB, VKK_HAZARD_NONE) == 0)
		{
			return NULL;
		}
	}

	if(nn_tensor_clear(self->dL_dX, VKK_HAZARD_NONE) == 0)
	{
		return NULL;
	}

	// sb100: bs
	// sb101: state
	// sb102: X
	// sb103: dL_dY
	vkk_uniformAttachment_t ua1_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = arch->sb100_bs,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = arch->sb101_state,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->X->sb_data,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY->sb_data,
		},
	};

	vkk_compute_updateUniformSetRefs(engine->compute,
	                                 self->us1_bp,
	                                 4, ua1_array);

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1_bp,
		NULL,
	};

	// nn_convLayer_backprop_dL_dX
	// dispatch required for each fi,fj
	// dispatch(RAW, bs, yh, yw, 1, 8, 8)
	uint fi;
	uint fj;
	vkk_computePipeline_t* cp;
	cp = engine->cp_conv_backprop_dL_dX;
	if(nn_engine_bind(engine, cp) == 0)
	{
		return NULL;
	}
	for(fi = 0; fi < fh; ++fi)
	{
		for(fj = 0; fj < fw; ++fj)
		{
			us_array[2] = nn_engine_getConvUs2(engine,
			                                   0, fi, fj, 0);
			if(us_array[2] == NULL)
			{
				return NULL;
			}
			vkk_compute_bindUniformSets(engine->compute, 3,
			                            us_array);
			nn_engine_dispatch(engine, VKK_HAZARD_RAW,
			                   bs, dimY->height, dimY->width,
			                   1, 8, 8);
		}
	}

	// nn_convLayer_backprop_dL_dW
	// dispatch required for each f,fi,fj,k
	// RAW hazard handled by nn_convLayer_backprop_dL_dX
	// dispatch(NONE, 1, 1, 1, 8, 8, 1)
	uint32_t f;
	uint32_t k;
	cp = engine->cp_conv_backprop_dL_dW;
	for(f = 0; f < fc; ++f)
	{
		if(nn_engine_bind(engine, cp) == 0)
		{
			return NULL;
		}

		for(fi = 0; fi < fh; ++fi)
		{
			for(fj = 0; fj < fw; ++fj)
			{
				for(k = 0; k < xd; ++k)
				{
					us_array[2] = nn_engine_getConvUs2(engine,
					                                   f, fi, fj, k);
					if(us_array[2] == NULL)
					{
						return NULL;
					}
					vkk_compute_bindUniformSets(engine->compute, 3,
					                            us_array);
					nn_engine_dispatch(engine, VKK_HAZARD_NONE,
					                   1, 1, 1, 8, 8, 1);
				}
			}
		}
	}

	// nn_convLayer_backprop_dL_dB
	// dispatch required for each f
	// RAW hazard handled by nn_convLayer_backprop_dL_dX
	// dispatch(NONE, 1, 1, 1, 8, 8, 1)
	if((self->flags & NN_CONV_LAYER_FLAG_DISABLE_BIAS) == 0)
	{
		cp = engine->cp_conv_backprop_dL_dB;
		if(nn_engine_bind(engine, cp) == 0)
		{
			return NULL;
		}

		for(f = 0; f < fc; ++f)
		{
			us_array[2] = nn_engine_getConvUs2(engine, f, 0, 0, 0);
			if(us_array[2] == NULL)
			{
				return NULL;
			}
			vkk_compute_bindUniformSets(engine->compute, 3,
			                            us_array);
			nn_engine_dispatch(engine, VKK_HAZARD_NONE,
			                   1, 1, 1, 8, 8, 1);
		}
	}

	// optionally skip parameter update
	if(flags & NN_LAYER_FLAG_NOP)
	{
		return self->dL_dX;
	}

	// nn_convLayer_backpropUpdateW
	// dispatch(RAW, fc, fh, fw, 4, 4, 4)
	cp = engine->cp_conv_backpropUpdateW;
	if(nn_engine_bind(engine, cp) == 0)
	{
		return NULL;
	}
	vkk_compute_bindUniformSets(engine->compute, 2, us_array);
	nn_engine_dispatch(engine, VKK_HAZARD_RAW,
	                   fc, fh, fw, 4, 4, 4);

	// nn_convLayer_backpropUpdateB
	// RAW hazard handled by nn_convLayer_backpropUpdateW
	// dispatch(NONE, fc, 1, 1, 64, 1, 1)
	cp = engine->cp_conv_backpropUpdateB;
	if(nn_engine_bind(engine, cp) == 0)
	{
		return NULL;
	}
	nn_engine_dispatch(engine, VKK_HAZARD_NONE,
	                   fc, 1, 1, 64, 1, 1);

	if(nn_tensor_computeStats(self->dL_dX, bs,
	                          VKK_HAZARD_RAW,
	                          self->stats_dL_dX) == 0)
	{
		return NULL;
	}

	return self->dL_dX;
}

static nn_tensor_t*
nn_convLayer_forwardPassTFn(nn_layer_t* base, int flags,
                            uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_convLayer_t* self   = (nn_convLayer_t*) base;
	nn_arch_t*      arch   = base->arch;
	nn_engine_t*    engine = arch->engine;

	nn_dim_t* dimY = nn_tensor_dim(self->Y);

	// optionally perform Spectral Normalization
	if(self->flags & NN_CONV_LAYER_FLAG_NORM_SN)
	{
		if(nn_tensor_normalize(self->W,
		                       VKK_HAZARD_NONE,
		                       NN_TENSOR_NORM_MODE_SN,
		                       1.0f) == 0)
		{
			return NULL;
		}
	}
	else if(self->flags & NN_CONV_LAYER_FLAG_NORM_BSSN)
	{
		if(nn_tensor_normalize(self->W,
		                       VKK_HAZARD_NONE,
		                       NN_TENSOR_NORM_MODE_BSSN,
		                       1.2f) == 0)
		{
			return NULL;
		}
	}

	// sb100: bs
	// sb101: state
	// sb102: X
	vkk_uniformAttachment_t ua1_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = arch->sb100_bs,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = arch->sb101_state,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X->sb_data,
		},
	};

	vkk_compute_updateUniformSetRefs(engine->compute,
	                                 self->us1_fp,
	                                 3, ua1_array);

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1_fp,
	};

	// nn_convLayer_forwardPassT
	// dispatch(RAW, bs, yh, yw, 1, 8, 8)
	vkk_computePipeline_t* cp;
	cp = engine->cp_conv_forwardPassT;
	if(nn_engine_bind(engine, cp) == 0)
	{
		return NULL;
	}
	vkk_compute_bindUniformSets(engine->compute, 2, us_array);
	nn_engine_dispatch(engine, VKK_HAZARD_RAW,
	                   bs, dimY->height, dimY->width,
	                   1, 8, 8);

	// store reference
	self->X = X;

	return self->Y;
}

static nn_tensor_t*
nn_convLayer_backpropTFn(nn_layer_t* base, int flags,
                         uint32_t bs, nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,yh,yw,fc)

	nn_convLayer_t* self   = (nn_convLayer_t*) base;
	nn_arch_t*      arch   = base->arch;
	nn_engine_t*    engine = arch->engine;

	nn_dim_t* dimW = nn_tensor_dim(self->W);
	nn_dim_t* dimY = nn_tensor_dim(dL_dY);
	uint32_t  fc   = dimW->count;
	uint32_t  fh   = dimW->height;
	uint32_t  fw   = dimW->width;
	uint32_t  xd   = dimW->depth;

	// clear backprop gradients
	if(nn_tensor_clear(self->dL_dW, VKK_HAZARD_NONE) == 0)
	{
		return NULL;
	}

	if((self->flags & NN_CONV_LAYER_FLAG_DISABLE_BIAS) == 0)
	{
		if(nn_tensor_clear(self->dL_dB, VKK_HAZARD_NONE) == 0)
		{
			return NULL;
		}
	}

	if(nn_tensor_clear(self->dL_dX, VKK_HAZARD_NONE) == 0)
	{
		return NULL;
	}

	// sb100: bs
	// sb101: state
	// sb102: X
	// sb103: dL_dY
	vkk_uniformAttachment_t ua1_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = arch->sb100_bs,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = arch->sb101_state,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->X->sb_data,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY->sb_data,
		},
	};

	vkk_compute_updateUniformSetRefs(engine->compute,
	                                 self->us1_bp,
	                                 4, ua1_array);

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1_bp,
		NULL,
	};

	// nn_convLayer_backpropT_dL_dX
	// dispatch required for each fi,fj
	// dispatch(RAW, bs, yh, yw, 1, 8, 8)
	uint fi;
	uint fj;
	vkk_computePipeline_t* cp;
	cp = engine->cp_conv_backpropT_dL_dX;
	if(nn_engine_bind(engine, cp) == 0)
	{
		return NULL;
	}
	for(fi = 0; fi < fh; ++fi)
	{
		for(fj = 0; fj < fw; ++fj)
		{
			us_array[2] = nn_engine_getConvUs2(engine,
			                                   0, fi, fj, 0);
			if(us_array[2] == NULL)
			{
				return NULL;
			}
			vkk_compute_bindUniformSets(engine->compute, 3,
			                            us_array);
			nn_engine_dispatch(engine, VKK_HAZARD_RAW,
			                   bs, dimY->height, dimY->width,
			                   1, 8, 8);
		}
	}

	// nn_convLayer_backpropT_dL_dW
	// dispatch required for each f,fi,fj,k
	// RAW hazard handled by nn_convLayer_backpropT_dL_dX
	// dispatch(NONE, 1, 1, 1, 8, 8, 1)
	uint32_t f;
	uint32_t k;
	cp = engine->cp_conv_backpropT_dL_dW;
	for(f = 0; f < fc; ++f)
	{
		if(nn_engine_bind(engine, cp) == 0)
		{
			return NULL;
		}

		for(fi = 0; fi < fh; ++fi)
		{
			for(fj = 0; fj < fw; ++fj)
			{
				for(k = 0; k < xd; ++k)
				{
					us_array[2] = nn_engine_getConvUs2(engine,
					                                   f, fi, fj, k);
					if(us_array[2] == NULL)
					{
						return NULL;
					}
					vkk_compute_bindUniformSets(engine->compute, 3,
					                            us_array);
					nn_engine_dispatch(engine, VKK_HAZARD_NONE,
					                   1, 1, 1, 8, 8, 1);
				}
			}
		}
	}

	// nn_convLayer_backprop_dL_dB
	// dispatch required for each f
	// RAW hazard handled by nn_convLayer_backprop_dL_dX
	// dispatch(NONE, 1, 1, 1, 8, 8, 1)
	if((self->flags & NN_CONV_LAYER_FLAG_DISABLE_BIAS) == 0)
	{
		cp = engine->cp_conv_backprop_dL_dB;
		if(nn_engine_bind(engine, cp) == 0)
		{
			return NULL;
		}

		for(f = 0; f < fc; ++f)
		{
			us_array[2] = nn_engine_getConvUs2(engine,
			                                   f, 0, 0, 0);
			if(us_array[2] == NULL)
			{
				return NULL;
			}
			vkk_compute_bindUniformSets(engine->compute, 3,
			                            us_array);
			nn_engine_dispatch(engine, VKK_HAZARD_NONE,
			                   1, 1, 1, 8, 8, 1);
		}
	}

	// optionally skip parameter update
	if(flags & NN_LAYER_FLAG_NOP)
	{
		return self->dL_dX;
	}

	// nn_convLayer_backpropUpdateW
	// dispatch(RAW, fc, fh, fw, 4, 4, 4)
	cp = engine->cp_conv_backpropUpdateW;
	if(nn_engine_bind(engine, cp) == 0)
	{
		return NULL;
	}
	vkk_compute_bindUniformSets(engine->compute, 2, us_array);
	nn_engine_dispatch(engine, VKK_HAZARD_RAW,
	                   fc, fh, fw, 4, 4, 4);

	// nn_convLayer_backpropUpdateB
	// RAW hazard handled by nn_convLayer_backpropUpdateW
	// dispatch(NONE, fc, 1, 1, 64, 1, 1)
	cp = engine->cp_conv_backpropUpdateB;
	if(nn_engine_bind(engine, cp) == 0)
	{
		return NULL;
	}
	nn_engine_dispatch(engine, VKK_HAZARD_NONE,
	                   fc, 1, 1, 64, 1, 1);

	if(nn_tensor_computeStats(self->dL_dX, bs,
	                          VKK_HAZARD_RAW,
	                          self->stats_dL_dX) == 0)
	{
		return NULL;
	}

	return self->dL_dX;
}

static void
nn_convLayer_postFn(nn_layer_t* base, int flags)
{
	ASSERT(base);

	nn_convLayer_t* self = (nn_convLayer_t*) base;

	if(flags & NN_LAYER_FLAG_BACKPROP)
	{
		LOGI("dL_dX min=%f, max=%f, mean=%f, stddev=%f, norm=%f",
		     nn_tensorStats_min(self->stats_dL_dX),
		     nn_tensorStats_max(self->stats_dL_dX),
		     nn_tensorStats_mean(self->stats_dL_dX),
		     nn_tensorStats_stddev(self->stats_dL_dX),
		     nn_tensorStats_norm(self->stats_dL_dX));
	}
}

static nn_dim_t*
nn_convLayer_dimXFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_convLayer_t* self = (nn_convLayer_t*) base;

	return nn_tensor_dim(self->dL_dX);
}

static nn_dim_t*
nn_convLayer_dimYFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_convLayer_t* self = (nn_convLayer_t*) base;

	return nn_tensor_dim(self->Y);
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_convUs2Data_t*
nn_convUs2Data_new(nn_engine_t* engine,
                   nn_convUs2Key_t* key)
{
	ASSERT(engine);
	ASSERT(key);

	nn_convUs2Data_t* self;
	self = (nn_convUs2Data_t*)
	       CALLOC(1, sizeof(nn_convUs2Data_t));
	if(self == NULL)
	{
		LOGE("CALLOC failed");
		return NULL;
	}

	self->sb200 = vkk_buffer_new(engine->engine,
	                             VKK_UPDATE_MODE_STATIC,
	                             VKK_BUFFER_USAGE_STORAGE,
	                             sizeof(nn_convUs2Key_t),
	                             key);
	if(self->sb200 == NULL)
	{
		goto fail_sb200;
	}

	self->us2 = vkk_uniformSet_new(engine->engine, 2, 0, NULL,
	                               engine->usf2_conv);
	if(self->us2 == NULL)
	{
		goto fail_us2;
	}

	vkk_uniformAttachment_t ua2_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->sb200,
		},
	};

	vkk_compute_updateUniformSetRefs(engine->compute,
	                                 self->us2,
	                                 1, ua2_array);

	// success
	return self;

	// failure
	fail_us2:
		vkk_buffer_delete(&self->sb200);
	fail_sb200:
		FREE(self);
	return NULL;
}

void nn_convUs2Data_delete(nn_convUs2Data_t** _self)
{
	ASSERT(_self);

	nn_convUs2Data_t* self = *_self;
	if(self)
	{
		vkk_uniformSet_delete(&self->us2);
		vkk_buffer_delete(&self->sb200);
		FREE(self);
		*_self = NULL;
	}
}

nn_convLayer_t*
nn_convLayer_new(nn_arch_t* arch, nn_dim_t* dimX,
                 nn_dim_t* dimW, uint32_t stride,
                 int flags)
{
	ASSERT(arch);
	ASSERT(dimX);
	ASSERT(dimW);

	nn_engine_t* engine = arch->engine;

	uint32_t fc = dimW->count;
	uint32_t bs = dimX->count;
	uint32_t xh = dimX->height;
	uint32_t xw = dimX->width;

	if(dimX->depth != dimW->depth)
	{
		LOGE("invalid depth=%u:%u",
		     dimX->depth, dimW->depth);
		return NULL;
	}

	if(stride < 1)
	{
		LOGE("invalid stride=%u", stride);
		return NULL;
	}

	nn_layerInfo_t info =
	{
		.arch            = arch,
		.forward_pass_fn = nn_convLayer_forwardPassFn,
		.backprop_fn     = nn_convLayer_backpropFn,
		.post_fn         = nn_convLayer_postFn,
		.dimX_fn         = nn_convLayer_dimXFn,
		.dimY_fn         = nn_convLayer_dimYFn,
	};

	if(flags & NN_CONV_LAYER_FLAG_TRANSPOSE)
	{
		info.forward_pass_fn = nn_convLayer_forwardPassTFn;
		info.backprop_fn     = nn_convLayer_backpropTFn;
	}

	nn_convLayer_t* self;
	self = (nn_convLayer_t*)
	       nn_layer_new(sizeof(nn_convLayer_t), &info);
	if(self == NULL)
	{
		return NULL;
	}

	self->flags  = flags;
	self->stride = stride;

	// XAVIER is default
	if(flags & NN_CONV_LAYER_FLAG_HE)
	{
		self->W = nn_tensor_new(engine, dimW,
		                        NN_TENSOR_INIT_HE,
		                        NN_TENSOR_MODE_COMPUTE);
	}
	else
	{
		self->W = nn_tensor_new(engine, dimW,
		                        NN_TENSOR_INIT_XAVIER,
		                        NN_TENSOR_MODE_COMPUTE);
	}

	if(self->W == NULL)
	{
		goto fail_W;
	}

	nn_dim_t dimB =
	{
		.count  = fc,
		.height = 1,
		.width  = 1,
		.depth  = 1,
	};

	self->B = nn_tensor_new(engine, &dimB,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_COMPUTE);
	if(self->B == NULL)
	{
		goto fail_B;
	}

	uint32_t yh;
	uint32_t yw;
	if(flags & NN_CONV_LAYER_FLAG_TRANSPOSE)
	{
		yh = stride*xh;
		yw = stride*xw;
	}
	else
	{
		yh = xh/stride;
		yw = xw/stride;
	}

	nn_dim_t dimY =
	{
		.count  = bs,
		.height = yh,
		.width  = yw,
		.depth  = fc,
	};

	self->Y = nn_tensor_new(engine, &dimY,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_COMPUTE);
	if(self->Y == NULL)
	{
		goto fail_Y;
	}

	self->MW = nn_tensor_new(engine, dimW,
	                         NN_TENSOR_INIT_ZERO,
	                         NN_TENSOR_MODE_COMPUTE);
	if(self->MW == NULL)
	{
		goto fail_MW;
	}

	self->VW = nn_tensor_new(engine, dimW,
	                         NN_TENSOR_INIT_ZERO,
	                         NN_TENSOR_MODE_COMPUTE);
	if(self->VW == NULL)
	{
		goto fail_VW;
	}

	self->MB = nn_tensor_new(engine, &dimB,
	                         NN_TENSOR_INIT_ZERO,
	                         NN_TENSOR_MODE_COMPUTE);
	if(self->MB == NULL)
	{
		goto fail_MB;
	}

	self->VB = nn_tensor_new(engine, &dimB,
	                         NN_TENSOR_INIT_ZERO,
	                         NN_TENSOR_MODE_COMPUTE);
	if(self->VB == NULL)
	{
		goto fail_VB;
	}

	self->dL_dW = nn_tensor_new(engine, dimW,
	                            NN_TENSOR_INIT_ZERO,
	                            NN_TENSOR_MODE_COMPUTE);
	if(self->dL_dW == NULL)
	{
		goto fail_dL_dW;
	}

	self->dL_dB = nn_tensor_new(engine, &dimB,
	                            NN_TENSOR_INIT_ZERO,
	                            NN_TENSOR_MODE_COMPUTE);
	if(self->dL_dB == NULL)
	{
		goto fail_dL_dB;
	}

	self->dL_dX = nn_tensor_new(engine, dimX,
	                            NN_TENSOR_INIT_ZERO,
	                            NN_TENSOR_MODE_COMPUTE);
	if(self->dL_dX == NULL)
	{
		goto fail_dL_dX;
	}

	self->stats_dL_dX = nn_tensorStats_new(engine);
	if(self->stats_dL_dX == NULL)
	{
		goto fail_stats_dL_dX;
	}

	nn_convLayerParam_t param =
	{
		.disable_bias = (self->flags & NN_CONV_LAYER_FLAG_DISABLE_BIAS) ? 1 : 0,
		.stride       = self->stride,
	};
	self->sb013_param = vkk_buffer_new(engine->engine,
	                                   VKK_UPDATE_MODE_STATIC,
	                                   VKK_BUFFER_USAGE_STORAGE,
	                                   sizeof(nn_convLayerParam_t),
	                                   &param);
	if(self->sb013_param == NULL)
	{
		goto fail_sb013_param;
	}

	self->us0 = vkk_uniformSet_new(engine->engine, 0, 0, NULL,
	                               engine->usf0_conv);
	if(self->us0 == NULL)
	{
		goto fail_us0;
	}

	self->us1_fp = vkk_uniformSet_new(engine->engine, 1, 0, NULL,
	                                  engine->usf1_conv_fp);
	if(self->us1_fp == NULL)
	{
		goto fail_us1_fp;
	}

	self->us1_bp = vkk_uniformSet_new(engine->engine, 1, 0, NULL,
	                                  engine->usf1_conv_bp);
	if(self->us1_bp == NULL)
	{
		goto fail_us1_bp;
	}

	// sb000: dimX (xbs,xh,xw,xd)
	// sb001: dimW (fc,fh,fw,xd)
	// sb002: W
	// sb003: B
	// sb004: dimY
	// sb005: Y
	// sb006: MW
	// sb007: VW
	// sb008: MB
	// sb019: VB
	// sb010: dL_dW
	// sb011: dL_dB
	// sb012: dL_dX
	// sb013: param (disable_bias,stride)
	vkk_uniformAttachment_t ua0_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->dL_dX->sb_dim,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->W->sb_dim,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->W->sb_data,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->B->sb_data,
		},
		{
			.binding = 4,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->Y->sb_dim,
		},
		{
			.binding = 5,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->Y->sb_data,
		},
		{
			.binding = 6,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->MW->sb_data,
		},
		{
			.binding = 7,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->VW->sb_data,
		},
		{
			.binding = 8,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->MB->sb_data,
		},
		{
			.binding = 9,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->VB->sb_data,
		},
		{
			.binding = 10,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->dL_dW->sb_data,
		},
		{
			.binding = 11,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->dL_dB->sb_data,
		},
		{
			.binding = 12,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->dL_dX->sb_data,
		},
		{
			.binding = 13,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->sb013_param,
		},
	};

	vkk_compute_updateUniformSetRefs(engine->compute, self->us0,
	                                 14, ua0_array);

	// success
	return self;

	// failure
	fail_us1_bp:
		vkk_uniformSet_delete(&self->us1_fp);
	fail_us1_fp:
		vkk_uniformSet_delete(&self->us0);
	fail_us0:
		vkk_buffer_delete(&self->sb013_param);
	fail_sb013_param:
		nn_tensorStats_delete(&self->stats_dL_dX);
	fail_stats_dL_dX:
		nn_tensor_delete(&self->dL_dX);
	fail_dL_dX:
		nn_tensor_delete(&self->dL_dB);
	fail_dL_dB:
		nn_tensor_delete(&self->dL_dW);
	fail_dL_dW:
		nn_tensor_delete(&self->VB);
	fail_VB:
		nn_tensor_delete(&self->MB);
	fail_MB:
		nn_tensor_delete(&self->VW);
	fail_VW:
		nn_tensor_delete(&self->MW);
	fail_MW:
		nn_tensor_delete(&self->Y);
	fail_Y:
		nn_tensor_delete(&self->B);
	fail_B:
		nn_tensor_delete(&self->W);
	fail_W:
		nn_layer_delete((nn_layer_t**) &self);
	return NULL;
}

void nn_convLayer_delete(nn_convLayer_t** _self)
{
	ASSERT(_self);

	nn_convLayer_t* self = *_self;
	if(self)
	{
		vkk_uniformSet_delete(&self->us1_bp);
		vkk_uniformSet_delete(&self->us1_fp);
		vkk_uniformSet_delete(&self->us0);
		vkk_buffer_delete(&self->sb013_param);
		nn_tensorStats_delete(&self->stats_dL_dX);
		nn_tensor_delete(&self->dL_dX);
		nn_tensor_delete(&self->dL_dB);
		nn_tensor_delete(&self->dL_dW);
		nn_tensor_delete(&self->VB);
		nn_tensor_delete(&self->MB);
		nn_tensor_delete(&self->VW);
		nn_tensor_delete(&self->MW);
		nn_tensor_delete(&self->Y);
		nn_tensor_delete(&self->B);
		nn_tensor_delete(&self->W);
		nn_layer_delete((nn_layer_t**) _self);
	}
}

nn_convLayer_t*
nn_convLayer_import(nn_arch_t* arch, jsmn_val_t* val)
{
	ASSERT(arch);
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	jsmn_val_t* val_dimX   = NULL;
	jsmn_val_t* val_dimW   = NULL;
	jsmn_val_t* val_flags  = NULL;
	jsmn_val_t* val_stride = NULL;
	jsmn_val_t* val_W      = NULL;
	jsmn_val_t* val_B      = NULL;
	jsmn_val_t* val_MW     = NULL;
	jsmn_val_t* val_VW     = NULL;
	jsmn_val_t* val_MB     = NULL;
	jsmn_val_t* val_VB     = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		jsmn_keyval_t* kv;
		kv = (jsmn_keyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == JSMN_TYPE_PRIMITIVE)
		{
			if(strcmp(kv->key, "flags") == 0)
			{
				val_flags = kv->val;
			}
			else if(strcmp(kv->key, "stride") == 0)
			{
				val_stride = kv->val;
			}
		}
		else if(kv->val->type == JSMN_TYPE_OBJECT)
		{
			if(strcmp(kv->key, "dimX") == 0)
			{
				val_dimX = kv->val;
			}
			else if(strcmp(kv->key, "dimW") == 0)
			{
				val_dimW = kv->val;
			}
			else if(strcmp(kv->key, "W") == 0)
			{
				val_W = kv->val;
			}
			else if(strcmp(kv->key, "B") == 0)
			{
				val_B = kv->val;
			}
			else if(strcmp(kv->key, "MW") == 0)
			{
				val_MW = kv->val;
			}
			else if(strcmp(kv->key, "VW") == 0)
			{
				val_VW = kv->val;
			}
			else if(strcmp(kv->key, "MB") == 0)
			{
				val_MB = kv->val;
			}
			else if(strcmp(kv->key, "VB") == 0)
			{
				val_VB = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_dimX          == NULL) ||
	   (val_dimW          == NULL) ||
	   (val_flags         == NULL) ||
	   (val_stride        == NULL) ||
	   (val_W             == NULL) ||
	   (val_B             == NULL) ||
	   (val_MW            == NULL) ||
	   (val_VW            == NULL) ||
	   (val_MB            == NULL) ||
	   (val_VB            == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	int      flags  = strtol(val_flags->data, NULL, 0);
	uint32_t stride = strtol(val_stride->data, NULL, 0);

	nn_dim_t dimX;
	nn_dim_t dimW;
	if((nn_dim_load(&dimX, val_dimX) == 0) ||
	   (nn_dim_load(&dimW, val_dimW) == 0))
	{
		return NULL;
	}

	nn_convLayer_t* self;
	self = nn_convLayer_new(arch, &dimX, &dimW,
	                        stride, flags);
	if(self == NULL)
	{
		return NULL;
	}

	// load tensors
	if((nn_tensor_load(self->W,  val_W)  == 0) ||
	   (nn_tensor_load(self->B,  val_B)  == 0) ||
	   (nn_tensor_load(self->MW, val_MW) == 0) ||
	   (nn_tensor_load(self->VW, val_VW) == 0) ||
	   (nn_tensor_load(self->MB, val_MB) == 0) ||
	   (nn_tensor_load(self->VB, val_VB) == 0))
	{
		goto fail_tensor;
	}

	// success
	return self;

	// failure
	fail_tensor:
		nn_convLayer_delete(&self);
	return NULL;
}

int nn_convLayer_export(nn_convLayer_t* self,
                        jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	nn_dim_t* dimX = nn_tensor_dim(self->dL_dX);
	nn_dim_t* dimW = nn_tensor_dim(self->W);

	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	ret &= jsmn_stream_key(stream, "%s", "dimX");
	ret &= nn_dim_store(dimX, stream);
	ret &= jsmn_stream_key(stream, "%s", "dimW");
	ret &= nn_dim_store(dimW, stream);
	ret &= jsmn_stream_key(stream, "%s", "flags");
	ret &= jsmn_stream_int(stream, self->flags);
	ret &= jsmn_stream_key(stream, "%s", "stride");
	ret &= jsmn_stream_int(stream, (int) self->stride);
	ret &= jsmn_stream_key(stream, "%s", "W");
	ret &= nn_tensor_store(self->W, stream);
	ret &= jsmn_stream_key(stream, "%s", "B");
	ret &= nn_tensor_store(self->B, stream);
	ret &= jsmn_stream_key(stream, "%s", "MW");
	ret &= nn_tensor_store(self->MW, stream);
	ret &= jsmn_stream_key(stream, "%s", "VW");
	ret &= nn_tensor_store(self->VW, stream);
	ret &= jsmn_stream_key(stream, "%s", "MB");
	ret &= nn_tensor_store(self->MB, stream);
	ret &= jsmn_stream_key(stream, "%s", "VB");
	ret &= nn_tensor_store(self->VB, stream);
	ret &= jsmn_stream_end(stream);

	return ret;
}
