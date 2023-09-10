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
#include "nn_arch.h"
#include "nn_convLayer.h"
#include "nn_layer.h"
#include "nn_tensor.h"

#ifdef NN_USE_COMPUTE
#include "../libvkk/vkk.h"
#endif

/***********************************************************
* private                                                  *
***********************************************************/

#ifdef NN_USE_COMPUTE

// protected
extern vkk_uniformSet_t*
nn_arch_getConvIdx(nn_arch_t* self,
                   uint32_t f, uint32_t fi,
                   uint32_t fj, uint32_t k);
extern void
nn_arch_dispatch(nn_arch_t* self,
                 vkk_hazzard_e hazzard,
                 uint32_t count_x,
                 uint32_t count_y,
                 uint32_t count_z,
                 uint32_t local_size_x,
                 uint32_t local_size_y,
                 uint32_t local_size_z);
extern int
nn_arch_bind(nn_arch_t* self,
             vkk_computePipeline_t* cp);

typedef struct
{
	uint32_t disable_bias;
	uint32_t stride;
} nn_convLayerParam_t;

static nn_tensor_t*
nn_convLayer_forwardPassFn(nn_layer_t* base,
                           nn_layerMode_e mode,
                           uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_convLayer_t* self = (nn_convLayer_t*) base;
	nn_arch_t*      arch = base->arch;

	nn_tensor_t* W    = self->W;
	nn_tensor_t* B    = self->B;
	nn_tensor_t* Y    = self->Y;
	nn_dim_t*    dimY = nn_tensor_dim(Y);

	// sb00: state
	// sb01: param (disable_bias and stride)
	// sb02: dimX
	// sb03: X
	// sb04: dimW
	// sb05: W
	// sb06: dimB
	// sb07: B
	vkk_uniformAttachment_t ua0_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = arch->sb_state,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->sb01_param,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X->sb_dim,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X->sb_data,
		},
		{
			.binding = 4,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = W->sb_dim,
		},
		{
			.binding = 5,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = W->sb_data,
		},
		{
			.binding = 6,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = B->sb_dim,
		},
		{
			.binding = 7,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = B->sb_data,
		},
	};

	// sb10: dimY
	// sb11: Y
	vkk_uniformAttachment_t ua1_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Y->sb_dim,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Y->sb_data,
		},
	};

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1,
	};

	// nn_convLayer_forwardPass
	// dispatch(RAW, bs, yh, yw, 1, 8, 8)
	vkk_computePipeline_t* cp;
	cp = arch->cp_conv_forwardPass;
	if(nn_arch_bind(arch, cp) == 0)
	{
		return NULL;
	}
	vkk_compute_updateUniformSetRefs(arch->compute, self->us0,
	                                 8, ua0_array);
	vkk_compute_updateUniformSetRefs(arch->compute, self->us1,
	                                 2, ua1_array);
	vkk_compute_bindUniformSets(arch->compute, 2, us_array);
	nn_arch_dispatch(arch, VKK_HAZZARD_RAW,
	                 bs, dimY->height, dimY->width,
	                 1, 8, 8);

	// store reference
	self->X = X;

	return Y;
}

static nn_tensor_t*
nn_convLayer_backpropFn(nn_layer_t* base, uint32_t bs,
                        nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,yh,yw,fc)

	nn_convLayer_t*   self  = (nn_convLayer_t*) base;
	nn_convLayerGc_t* gc    = &self->gc;
	nn_arch_t*        arch  = base->arch;
	nn_archState_t*   state = &arch->state;

	nn_tensor_t* VW   = self->VW;
	nn_tensor_t* VB   = self->VB;
	nn_dim_t*    dimW = nn_tensor_dim(self->W);
	nn_dim_t*    dimY = nn_tensor_dim(dL_dY);
	uint32_t     fc   = dimW->count;
	uint32_t     fh   = dimW->height;
	uint32_t     fw   = dimW->width;
	uint32_t     xd   = dimW->depth;

	// clear backprop gradients
	nn_tensor_t* dL_dW = self->dL_dW;
	nn_tensor_t* dL_dB = self->dL_dB;
	nn_tensor_t* dL_dX = self->dL_dX;
	nn_tensor_clear(dL_dW, NN_TENSOR_HAZZARD_NONE);
	if((self->flags & NN_CONV_LAYER_FLAG_DISABLE_BIAS) == 0)
	{
		nn_tensor_clear(dL_dB, NN_TENSOR_HAZZARD_NONE);
	}
	nn_tensor_clear(dL_dX, NN_TENSOR_HAZZARD_NONE);

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
	vkk_uniformAttachment_t ua2_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->sb20_gc,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY->sb_dim,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY->sb_data,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dW->sb_dim,
		},
		{
			.binding = 4,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dW->sb_data,
		},
		{
			.binding = 5,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dB->sb_dim,
		},
		{
			.binding = 6,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dB->sb_data,
		},
		{
			.binding = 7,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dX->sb_dim,
		},
		{
			.binding = 8,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dX->sb_data,
		},
		{
			.binding = 9,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = VW->sb_dim,
		},
		{
			.binding = 10,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = VW->sb_data,
		},
		{
			.binding = 11,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = VB->sb_dim,
		},
		{
			.binding = 12,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = VB->sb_data,
		},
	};

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1,
		self->us2,
	};

	// nn_convLayer_backprop_dL_dX
	// dispatch required for each fi,fj
	// dispatch(RAW, bs, yh, yw, 1, 8, 8)
	uint fi;
	uint fj;
	vkk_computePipeline_t* cp;
	vkk_uniformSet_t*      us3;
	cp = arch->cp_conv_backprop_dL_dX;
	if(nn_arch_bind(arch, cp) == 0)
	{
		return NULL;
	}
	vkk_compute_updateUniformSetRefs(arch->compute, self->us2,
	                                 13, ua2_array);
	vkk_compute_bindUniformSets(arch->compute, 3, us_array);
	for(fi = 0; fi < fh; ++fi)
	{
		for(fj = 0; fj < fw; ++fj)
		{
			us3 = nn_arch_getConvIdx(arch, 0, fi, fj, 0);
			if(us3 == NULL)
			{
				return NULL;
			}
			vkk_compute_bindUniformSets(arch->compute, 1, &us3);
			nn_arch_dispatch(arch, VKK_HAZZARD_RAW,
			                 bs, dimY->height, dimY->width,
			                 1, 8, 8);
		}
	}

	// nn_convLayer_backprop_dL_dW
	// dispatch required for each f,fi,fj,k
	// RAW hazzard handled by nn_convLayer_backprop_dL_dX
	// dispatch(NONE, 1, 1, 1, 8, 8, 1)
	uint32_t f;
	uint32_t k;
	cp = arch->cp_conv_backprop_dL_dW;
	for(f = 0; f < fc; ++f)
	{
		if(nn_arch_bind(arch, cp) == 0)
		{
			return NULL;
		}

		for(fi = 0; fi < fh; ++fi)
		{
			for(fj = 0; fj < fw; ++fj)
			{
				for(k = 0; k < xd; ++k)
				{
					us3 = nn_arch_getConvIdx(arch, f, fi, fj, k);
					if(us3 == NULL)
					{
						return NULL;
					}
					vkk_compute_bindUniformSets(arch->compute, 1, &us3);
					nn_arch_dispatch(arch, VKK_HAZZARD_NONE,
					                 1, 1, 1, 8, 8, 1);
				}
			}
		}
	}

	// nn_convLayer_backprop_dL_dB
	// dispatch required for each f
	// RAW hazzard handled by nn_convLayer_backprop_dL_dX
	// dispatch(NONE, 1, 1, 1, 8, 8, 1)
	if((self->flags & NN_CONV_LAYER_FLAG_DISABLE_BIAS) == 0)
	{
		cp = arch->cp_conv_backprop_dL_dB;
		if(nn_arch_bind(arch, cp) == 0)
		{
			return NULL;
		}
		for(f = 0; f < fc; ++f)
		{
			us3 = nn_arch_getConvIdx(arch, f, 0, 0, 0);
			if(us3 == NULL)
			{
				return NULL;
			}
			vkk_compute_bindUniformSets(arch->compute, 1, &us3);
			nn_arch_dispatch(arch, VKK_HAZZARD_NONE,
			                 1, 1, 1, 8, 8, 1);
		}
	}

	// initialize gc but keep running averages
	gc->gcw        = 1.0f;
	gc->gcb        = 1.0f;
	gc->norm_w     = 0.0f;
	gc->norm_b     = 0.0f;
	gc->norm_dl_dw = 0.0f;
	gc->norm_dl_db = 0.0f;
	vkk_compute_writeBuffer(arch->compute, self->sb20_gc,
	                        sizeof(nn_convLayerGc_t), 0, gc);

	// nn_convLayer_backpropGradientClipping
	// dispatch(RAW, 1, 1, 1, 4, 4, 4)
	if((state->clip_max_weight > 0.0f) &&
	   (state->clip_max_bias   > 0.0f) &&
	   (state->clip_mu_inc     > 0.0f) &&
	   (state->clip_mu_dec     > 0.0f))
	{
		cp = arch->cp_conv_backpropGradientClipping;
		if(nn_arch_bind(arch, cp) == 0)
		{
			return NULL;
		}
		nn_arch_dispatch(arch, VKK_HAZZARD_RAW,
		                 1, 1, 1, 4, 4, 4);
	}

	// nn_convLayer_backpropUpdateW
	// dispatch(RAW, fc, fh, fw, 4, 4, 4)
	cp = arch->cp_conv_backpropUpdateW;
	if(nn_arch_bind(arch, cp) == 0)
	{
		return NULL;
	}
	nn_arch_dispatch(arch, VKK_HAZZARD_RAW,
	                 fc, fh, fw, 4, 4, 4);

	// nn_convLayer_backpropUpdateB
	// RAW hazzard handled by nn_convLayer_backpropUpdateW
	// dispatch(NONE, fc, 1, 1, 64, 1, 1)
	cp = arch->cp_conv_backpropUpdateB;
	if(nn_arch_bind(arch, cp) == 0)
	{
		return NULL;
	}
	nn_arch_dispatch(arch, VKK_HAZZARD_NONE,
	                 fc, 1, 1, 64, 1, 1);

	return dL_dX;
}

static void
nn_convLayer_postFn(nn_layer_t* base,
                    nn_layerMode_e mode)
{
	ASSERT(base);

	nn_convLayer_t*   self  = (nn_convLayer_t*) base;
	nn_convLayerGc_t* gc    = &self->gc;
	nn_arch_t*        arch  = base->arch;
	nn_archState_t*   state = &arch->state;

	if((mode == NN_LAYER_MODE_TRAIN)   &&
	   (state->clip_max_weight > 0.0f) &&
	   (state->clip_max_bias   > 0.0f) &&
	   (state->clip_mu_inc     > 0.0f) &&
	   (state->clip_mu_dec     > 0.0f))
	{
		vkk_compute_readBuffer(arch->compute, self->sb20_gc,
		                       sizeof(nn_convLayerGc_t), 0, gc);

		#ifdef NN_GC_DEBUG
		LOGI("norm: w=%f, b=%f, dl_dw=%f, dl_dw_ra=%f, dl_db=%f, dl_db_ra=%f",
		     gc->norm_w, gc->norm_b,
		     gc->norm_dl_dw, gc->norm_dl_dw_ra,
		     gc->norm_dl_db, gc->norm_dl_db_ra);
		#endif
	}
}

static nn_tensor_t*
nn_convLayer_forwardPassTFn(nn_layer_t* base,
                            nn_layerMode_e mode,
                            uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_convLayer_t* self = (nn_convLayer_t*) base;
	nn_arch_t*      arch = base->arch;

	nn_tensor_t* W    = self->W;
	nn_tensor_t* B    = self->B;
	nn_tensor_t* Y    = self->Y;
	nn_dim_t*    dimY = nn_tensor_dim(Y);

	// sb00: state
	// sb01: param (disable_bias and stride)
	// sb02: dimX
	// sb03: X
	// sb04: dimW
	// sb05: W
	// sb06: dimB
	// sb07: B
	vkk_uniformAttachment_t ua0_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = arch->sb_state,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->sb01_param,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X->sb_dim,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X->sb_data,
		},
		{
			.binding = 4,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = W->sb_dim,
		},
		{
			.binding = 5,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = W->sb_data,
		},
		{
			.binding = 6,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = B->sb_dim,
		},
		{
			.binding = 7,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = B->sb_data,
		},
	};

	// sb10: dimY
	// sb11: Y
	vkk_uniformAttachment_t ua1_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Y->sb_dim,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Y->sb_data,
		},
	};

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1,
	};

	// nn_convLayer_forwardPassT
	// dispatch(RAW, bs, yh, yw, 1, 8, 8)
	vkk_computePipeline_t* cp;
	cp = arch->cp_conv_forwardPassT;
	if(nn_arch_bind(arch, cp) == 0)
	{
		return NULL;
	}
	vkk_compute_updateUniformSetRefs(arch->compute, self->us0,
	                                 8, ua0_array);
	vkk_compute_updateUniformSetRefs(arch->compute, self->us1,
	                                 2, ua1_array);
	vkk_compute_bindUniformSets(arch->compute, 2, us_array);
	nn_arch_dispatch(arch, VKK_HAZZARD_RAW,
	                 bs, dimY->height, dimY->width,
	                 1, 8, 8);

	// store reference
	self->X = X;

	return Y;
}

static nn_tensor_t*
nn_convLayer_backpropTFn(nn_layer_t* base, uint32_t bs,
                         nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,yh,yw,fc)

	nn_convLayer_t*   self  = (nn_convLayer_t*) base;
	nn_convLayerGc_t* gc    = &self->gc;
	nn_arch_t*        arch  = base->arch;
	nn_archState_t*   state = &arch->state;

	nn_tensor_t* VW   = self->VW;
	nn_tensor_t* VB   = self->VB;
	nn_dim_t*    dimW = nn_tensor_dim(self->W);
	nn_dim_t*    dimY = nn_tensor_dim(dL_dY);
	uint32_t     fc   = dimW->count;
	uint32_t     fh   = dimW->height;
	uint32_t     fw   = dimW->width;
	uint32_t     xd   = dimW->depth;

	// clear backprop gradients
	nn_tensor_t* dL_dW = self->dL_dW;
	nn_tensor_t* dL_dB = self->dL_dB;
	nn_tensor_t* dL_dX = self->dL_dX;
	nn_tensor_clear(dL_dW, NN_TENSOR_HAZZARD_NONE);
	if((self->flags & NN_CONV_LAYER_FLAG_DISABLE_BIAS) == 0)
	{
		nn_tensor_clear(dL_dB, NN_TENSOR_HAZZARD_NONE);
	}
	nn_tensor_clear(dL_dX, NN_TENSOR_HAZZARD_NONE);

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
	vkk_uniformAttachment_t ua2_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->sb20_gc,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY->sb_dim,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY->sb_data,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dW->sb_dim,
		},
		{
			.binding = 4,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dW->sb_data,
		},
		{
			.binding = 5,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dB->sb_dim,
		},
		{
			.binding = 6,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dB->sb_data,
		},
		{
			.binding = 7,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dX->sb_dim,
		},
		{
			.binding = 8,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dX->sb_data,
		},
		{
			.binding = 9,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = VW->sb_dim,
		},
		{
			.binding = 10,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = VW->sb_data,
		},
		{
			.binding = 11,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = VB->sb_dim,
		},
		{
			.binding = 12,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = VB->sb_data,
		},
	};

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1,
		self->us2,
	};

	// nn_convLayer_backpropT_dL_dX
	// dispatch required for each fi,fj
	// dispatch(RAW, bs, yh, yw, 1, 8, 8)
	uint fi;
	uint fj;
	vkk_computePipeline_t* cp;
	vkk_uniformSet_t*      us3;
	cp = arch->cp_conv_backpropT_dL_dX;
	if(nn_arch_bind(arch, cp) == 0)
	{
		return NULL;
	}
	vkk_compute_updateUniformSetRefs(arch->compute, self->us2,
	                                 13, ua2_array);
	vkk_compute_bindUniformSets(arch->compute, 3, us_array);
	for(fi = 0; fi < fh; ++fi)
	{
		for(fj = 0; fj < fw; ++fj)
		{
			us3 = nn_arch_getConvIdx(arch, 0, fi, fj, 0);
			if(us3 == NULL)
			{
				return NULL;
			}
			vkk_compute_bindUniformSets(arch->compute, 1, &us3);
			nn_arch_dispatch(arch, VKK_HAZZARD_RAW,
			                 bs, dimY->height, dimY->width,
			                 1, 8, 8);
		}
	}

	// nn_convLayer_backpropT_dL_dW
	// dispatch required for each f,fi,fj,k
	// RAW hazzard handled by nn_convLayer_backpropT_dL_dX
	// dispatch(NONE, 1, 1, 1, 8, 8, 1)
	uint32_t f;
	uint32_t k;
	cp = arch->cp_conv_backpropT_dL_dW;
	for(f = 0; f < fc; ++f)
	{
		if(nn_arch_bind(arch, cp) == 0)
		{
			return NULL;
		}

		for(fi = 0; fi < fh; ++fi)
		{
			for(fj = 0; fj < fw; ++fj)
			{
				for(k = 0; k < xd; ++k)
				{
					us3 = nn_arch_getConvIdx(arch, f, fi, fj, k);
					if(us3 == NULL)
					{
						return NULL;
					}
					vkk_compute_bindUniformSets(arch->compute, 1, &us3);
					nn_arch_dispatch(arch, VKK_HAZZARD_NONE,
					                 1, 1, 1, 8, 8, 1);
				}
			}
		}
	}

	// nn_convLayer_backprop_dL_dB
	// dispatch required for each f
	// RAW hazzard handled by nn_convLayer_backprop_dL_dX
	// dispatch(NONE, 1, 1, 1, 8, 8, 1)
	if((self->flags & NN_CONV_LAYER_FLAG_DISABLE_BIAS) == 0)
	{
		cp = arch->cp_conv_backprop_dL_dB;
		if(nn_arch_bind(arch, cp) == 0)
		{
			return NULL;
		}
		for(f = 0; f < fc; ++f)
		{
			us3 = nn_arch_getConvIdx(arch, f, 0, 0, 0);
			if(us3 == NULL)
			{
				return NULL;
			}
			vkk_compute_bindUniformSets(arch->compute, 1, &us3);
			nn_arch_dispatch(arch, VKK_HAZZARD_NONE,
			                 1, 1, 1, 8, 8, 1);
		}
	}

	// initialize gc but keep running averages
	gc->gcw        = 1.0f;
	gc->gcb        = 1.0f;
	gc->norm_w     = 0.0f;
	gc->norm_b     = 0.0f;
	gc->norm_dl_dw = 0.0f;
	gc->norm_dl_db = 0.0f;
	vkk_compute_writeBuffer(arch->compute, self->sb20_gc,
	                        sizeof(nn_convLayerGc_t), 0, gc);

	// nn_convLayer_backpropGradientClipping
	// dispatch(RAW, 1, 1, 1, 4, 4, 4)
	if((state->clip_max_weight > 0.0f) &&
	   (state->clip_max_bias   > 0.0f) &&
	   (state->clip_mu_inc     > 0.0f) &&
	   (state->clip_mu_dec     > 0.0f))
	{
		cp = arch->cp_conv_backpropGradientClipping;
		if(nn_arch_bind(arch, cp) == 0)
		{
			return NULL;
		}
		nn_arch_dispatch(arch, VKK_HAZZARD_RAW,
		                 1, 1, 1, 4, 4, 4);
	}

	// nn_convLayer_backpropUpdateW
	// dispatch(RAW, fc, fh, fw, 4, 4, 4)
	cp = arch->cp_conv_backpropUpdateW;
	if(nn_arch_bind(arch, cp) == 0)
	{
		return NULL;
	}
	nn_arch_dispatch(arch, VKK_HAZZARD_RAW,
	                 fc, fh, fw, 4, 4, 4);

	// nn_convLayer_backpropUpdateB
	// RAW hazzard handled by nn_convLayer_backpropUpdateW
	// dispatch(NONE, fc, 1, 1, 64, 1, 1)
	cp = arch->cp_conv_backpropUpdateB;
	if(nn_arch_bind(arch, cp) == 0)
	{
		return NULL;
	}
	nn_arch_dispatch(arch, VKK_HAZZARD_NONE,
	                 fc, 1, 1, 64, 1, 1);

	return dL_dX;
}

static int
nn_convLayer_newCompute(nn_convLayer_t* self)
{
	ASSERT(self);

	nn_arch_t* arch = self->base.arch;

	self->us0 = vkk_uniformSet_new(arch->engine, 0, 0, NULL,
	                               arch->usf0_conv);
	if(self->us0 == NULL)
	{
		return 0;
	}

	self->us1 = vkk_uniformSet_new(arch->engine, 1, 0, NULL,
	                               arch->usf1_conv);
	if(self->us1 == NULL)
	{
		goto fail_us1;
	}

	self->us2 = vkk_uniformSet_new(arch->engine, 2, 0, NULL,
	                               arch->usf2_conv);
	if(self->us2 == NULL)
	{
		goto fail_us2;
	}

	nn_convLayerParam_t param =
	{
		.disable_bias = (self->flags & NN_CONV_LAYER_FLAG_DISABLE_BIAS) ? 1 : 0,
		.stride       = self->stride,
	};
	self->sb01_param = vkk_buffer_new(arch->engine,
	                                  VKK_UPDATE_MODE_STATIC,
	                                  VKK_BUFFER_USAGE_STORAGE,
	                                  sizeof(nn_convLayerParam_t),
	                                  &param);
	if(self->sb01_param == NULL)
	{
		goto fail_sb01_param;
	}

	self->sb20_gc = vkk_buffer_new(arch->engine,
	                               VKK_UPDATE_MODE_SYNCHRONOUS,
	                               VKK_BUFFER_USAGE_STORAGE,
	                               sizeof(nn_convLayerGc_t), NULL);
	if(self->sb20_gc == NULL)
	{
		goto fail_sb20_gc;
	}

	// success
	return 1;

	// failure
	fail_sb20_gc:
		vkk_buffer_delete(&self->sb01_param);
	fail_sb01_param:
		vkk_uniformSet_delete(&self->us2);
	fail_us2:
		vkk_uniformSet_delete(&self->us1);
	fail_us1:
		vkk_uniformSet_delete(&self->us0);
	return 0;
}

static void
nn_convLayer_deleteCompute(nn_convLayer_t* self)
{
	ASSERT(self);

	vkk_buffer_delete(&self->sb20_gc);
	vkk_buffer_delete(&self->sb01_param);
	vkk_uniformSet_delete(&self->us2);
	vkk_uniformSet_delete(&self->us1);
	vkk_uniformSet_delete(&self->us0);
}

#else // NN_USE_COMPUTE not undefined

static void
nn_convLayer_forwardPass(nn_convLayer_t* self,
                         nn_tensor_t* X,
                         uint32_t m, uint32_t i,
                         uint32_t j, uint32_t f)
{
	ASSERT(self);

	nn_tensor_t* W     = self->W;
	nn_tensor_t* B     = self->B;
	nn_tensor_t* Y     = self->Y;
	nn_tensor_t* dL_dX = self->dL_dX;
	nn_dim_t*    dimW  = nn_tensor_dim(W);
	nn_dim_t*    dimX  = nn_tensor_dim(dL_dX);
	uint32_t     fh    = dimW->height;
	uint32_t     fw    = dimW->width;
	uint32_t     xh    = dimX->height;
	uint32_t     xw    = dimX->width;
	uint32_t     xd    = dimX->depth;
	uint32_t     s     = self->stride;

	// initialize y
	float y;
	if(self->flags & NN_CONV_LAYER_FLAG_DISABLE_BIAS)
	{
		y = 0.0f;
	}
	else
	{
		y = nn_tensor_getv(B, f);
	}

	// compute weighted sum
	float    w;
	float    x;
	uint32_t fi;
	uint32_t fj;
	uint32_t k;
	int      ii;
	int      jj;
	for(fi = 0; fi < fh; ++fi)
	{
		ii = ((int) (s*i + fi)) - ((int) (fh/2));
		if((ii < 0) || (ii >= xh))
		{
			continue;
		}

		for(fj = 0; fj < fw; ++fj)
		{
			jj = ((int) (s*j + fj)) - ((int) (fw/2));
			if((jj < 0) || (jj >= xw))
			{
				continue;
			}

			for(k = 0; k < xd; ++k)
			{
				w  = nn_tensor_get(W, f, fi, fj, k);
				x  = nn_tensor_get(X, m, ii, jj, k);
				y += w*x;
			}
		}
	}
	nn_tensor_set(Y, m, i, j, f, y);
}

static nn_tensor_t*
nn_convLayer_forwardPassFn(nn_layer_t* base,
                           nn_layerMode_e mode,
                           uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_convLayer_t* self = (nn_convLayer_t*) base;

	nn_dim_t* dimW = nn_tensor_dim(self->W);
	nn_dim_t* dimY = nn_tensor_dim(self->Y);
	uint32_t  fc   = dimW->count;
	uint32_t  yh   = dimY->height;
	uint32_t  yw   = dimY->width;

	// forward pass Y
	uint32_t m;
	uint32_t i;
	uint32_t j;
	uint32_t f;
	for(m = 0; m < bs; ++m)
	{
		for(i = 0; i < yh; ++i)
		{
			for(j = 0; j < yw; ++j)
			{
				for(f = 0; f < fc; ++f)
				{
					nn_convLayer_forwardPass(self, X, m, i, j, f);
				}
			}
		}
	}

	// store reference
	self->X = X;

	return self->Y;
}

static void
nn_convLayer_gradientClipping(nn_convLayer_t* self,
                              uint32_t bs)
{
	ASSERT(self);

	nn_arch_t*        arch  = self->base.arch;
	nn_convLayerGc_t* gc    = &self->gc;
	nn_archState_t*   state = &arch->state;
	nn_tensor_t*      W     = self->W;
	nn_tensor_t*      B     = self->B;
	nn_tensor_t*      dL_dW = self->dL_dW;
	nn_tensor_t*      dL_dB = self->dL_dB;
	nn_dim_t*         dimW  = nn_tensor_dim(W);
	uint32_t          fc    = dimW->count;
	uint32_t          fh    = dimW->height;
	uint32_t          fw    = dimW->width;
	uint32_t          xd    = dimW->depth;
	float             s     = 1.0f/((float) bs);

	// compute norms
	float    w;
	float    b;
	float    dl_dw;
	float    dl_db;
	uint32_t f;
	uint32_t fi;
	uint32_t fj;
	uint32_t k;
	for(f = 0; f < fc; ++f)
	{
		// weights gradient
		for(fi = 0; fi < fh; ++fi)
		{
			for(fj = 0; fj < fw; ++fj)
			{
				for(k = 0; k < xd; ++k)
				{
					w               = nn_tensor_get(W, f, fi, fj, k);
					dl_dw           = s*nn_tensor_get(dL_dW, f, fi, fj, k);
					gc->norm_w     += w*w;
					gc->norm_dl_dw += dl_dw*dl_dw;
				}
			}
		}

		// bias gradient
		if((self->flags & NN_CONV_LAYER_FLAG_DISABLE_BIAS) == 0)
		{
			b               = nn_tensor_getv(B, f);
			dl_db           = s*nn_tensor_getv(dL_dB, f);
			gc->norm_b     += b*b;
			gc->norm_dl_db += dl_db*dl_db;
		}
	}
	gc->norm_w     = state->clip_scale*sqrtf(gc->norm_w);
	gc->norm_b     = state->clip_scale*sqrtf(gc->norm_b);
	gc->norm_dl_dw = state->clip_scale*sqrtf(gc->norm_dl_dw);
	gc->norm_dl_db = state->clip_scale*sqrtf(gc->norm_dl_db);

	// compute running averages for norm_dl_dw
	float clip_mu;
	if(gc->norm_dl_dw > gc->norm_dl_dw_ra)
	{
		clip_mu = state->clip_mu_inc;
	}
	else
	{
		clip_mu = state->clip_mu_dec;
	}
	gc->norm_dl_dw_ra = clip_mu*gc->norm_dl_dw_ra +
	                    (1.0f - clip_mu)*gc->norm_dl_dw;

	// compute running averages for norm_dl_db
	if(gc->norm_dl_db > gc->norm_dl_db_ra)
	{
		clip_mu = state->clip_mu_inc;
	}
	else
	{
		clip_mu = state->clip_mu_dec;
	}
	gc->norm_dl_db_ra = clip_mu*gc->norm_dl_db_ra +
	                    (1.0f - clip_mu)*gc->norm_dl_db;

	// clamp running averages for norm_dl_dw_ra
	if(state->clip_max_weight > 0.0f)
	{
		if(gc->norm_dl_dw_ra > state->clip_max_weight)
		{
			gc->norm_dl_dw_ra = state->clip_max_weight;
		}
	}

	// clamp running averages for norm_dl_db_ra
	if(state->clip_max_bias > 0.0f)
	{
		if(gc->norm_dl_db_ra > state->clip_max_bias)
		{
			gc->norm_dl_db_ra = state->clip_max_bias;
		}
	}

	// apply gradient clipping
	if(gc->norm_dl_dw > gc->norm_dl_dw_ra)
	{
		gc->gcw = gc->norm_dl_dw_ra/gc->norm_dl_dw;
	}
	if(gc->norm_dl_db > gc->norm_dl_db_ra)
	{
		gc->gcb = gc->norm_dl_db_ra/gc->norm_dl_db;
	}
}

static void
nn_convLayer_backprop(nn_convLayer_t* self,
                      nn_tensor_t* dL_dY,
                      uint32_t m, uint32_t i,
                      uint32_t j, uint32_t f)
{
	ASSERT(self);
	ASSERT(dL_dY); // dim(bs, yh, yw, fc)

	nn_tensor_t* W     = self->W;
	nn_tensor_t* dY_dX = W;
	nn_tensor_t* dY_dW = self->X;
	nn_tensor_t* dL_dW = self->dL_dW;
	nn_tensor_t* dL_dB = self->dL_dB;
	nn_tensor_t* dL_dX = self->dL_dX;
	nn_dim_t*    dimW  = nn_tensor_dim(W);
	nn_dim_t*    dimX  = nn_tensor_dim(dL_dX);
	uint32_t     fh    = dimW->height;
	uint32_t     fw    = dimW->width;
	uint32_t     xh    = dimX->height;
	uint32_t     xw    = dimX->width;
	uint32_t     xd    = dimX->depth;
	uint32_t     s     = self->stride;

	float    dl_dy = nn_tensor_get(dL_dY, m, i, j, f);
	float    dy_dx;
	float    dy_dw;
	float    dy_db = 1.0f;
	uint32_t fi;
	uint32_t fj;
	uint32_t k;
	int      ii;
	int      jj;
	for(fi = 0; fi < fh; ++fi)
	{
		ii = ((int) (s*i + fi)) - ((int) (fh/2));
		if((ii < 0) || (ii >= xh))
		{
			continue;
		}

		for(fj = 0; fj < fw; ++fj)
		{
			jj = ((int) (s*j + fj)) - ((int) (fw/2));
			if((jj < 0) || (jj >= xw))
			{
				continue;
			}

			for(k = 0; k < xd; ++k)
			{
				// backpropagate dL_dX
				dy_dx = nn_tensor_get(dY_dX, f, fi, fj, k);
				nn_tensor_add(dL_dX, m, ii, jj, k, dl_dy*dy_dx);

				// sum dL_dW
				dy_dw = nn_tensor_get(dY_dW, m, ii, jj, k);
				nn_tensor_add(dL_dW, f, fi, fj, k, dl_dy*dy_dw);
			}
		}
	}

	// sum dL_dB
	if((self->flags & NN_CONV_LAYER_FLAG_DISABLE_BIAS) == 0)
	{
		nn_tensor_addv(dL_dB, f, dl_dy*dy_db);
	}
}

static nn_tensor_t*
nn_convLayer_backpropFn(nn_layer_t* base, uint32_t bs,
                        nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,yh,yw,fc)

	nn_convLayer_t*   self  = (nn_convLayer_t*) base;
	nn_convLayerGc_t* gc    = &self->gc;
	nn_arch_t*        arch  = base->arch;
	nn_archState_t*   state = &arch->state;

	nn_tensor_t* W      = self->W;
	nn_tensor_t* B      = self->B;
	nn_tensor_t* VW     = self->VW;
	nn_tensor_t* VB     = self->VB;
	nn_dim_t*    dimY   = nn_tensor_dim(dL_dY);
	nn_dim_t*    dimW   = nn_tensor_dim(W);
	uint32_t     yh     = dimY->height;
	uint32_t     yw     = dimY->width;
	uint32_t     fc     = dimW->count;
	uint32_t     fh     = dimW->height;
	uint32_t     fw     = dimW->width;
	uint32_t     xd     = dimW->depth;
	float        lr     = state->learning_rate;
	float        mu     = state->momentum_decay;
	float        lambda = state->l2_lambda;

	// clear backprop gradients
	nn_tensor_t* dL_dW = self->dL_dW;
	nn_tensor_t* dL_dB = self->dL_dB;
	nn_tensor_t* dL_dX = self->dL_dX;
	nn_tensor_clear(dL_dW, NN_TENSOR_HAZZARD_NONE);
	if((self->flags & NN_CONV_LAYER_FLAG_DISABLE_BIAS) == 0)
	{
		nn_tensor_clear(dL_dB, NN_TENSOR_HAZZARD_NONE);
	}
	nn_tensor_clear(dL_dX, NN_TENSOR_HAZZARD_NONE);

	// sum gradients and backpropagate loss
	uint32_t m;
	uint32_t i;
	uint32_t j;
	uint32_t f;
	for(m = 0; m < bs; ++m)
	{
		for(i = 0; i < yh; ++i)
		{
			for(j = 0; j < yw; ++j)
			{
				for(f = 0; f < fc; ++f)
				{
					nn_convLayer_backprop(self, dL_dY, m, i, j, f);
				}
			}
		}
	}

	// optionally compute gradient clipping
	// initialize gc but keep running averages
	gc->gcw        = 1.0f;
	gc->gcb        = 1.0f;
	gc->norm_w     = 0.0f;
	gc->norm_b     = 0.0f;
	gc->norm_dl_dw = 0.0f;
	gc->norm_dl_db = 0.0f;
	if((state->clip_max_weight > 0.0f) &&
	   (state->clip_max_bias   > 0.0f) &&
	   (state->clip_mu_inc     > 0.0f) &&
	   (state->clip_mu_dec     > 0.0f))
	{
		nn_convLayer_gradientClipping(self, bs);
	}

	// update parameters
	float    v0;
	float    v1;
	float    w;
	float    dl_dw;
	float    dl_db;
	float    s = 1.0f/((float) bs);
	uint32_t fi;
	uint32_t fj;
	uint32_t k;
	for(f = 0; f < fc; ++f)
	{
		// weights
		for(fi = 0; fi < fh; ++fi)
		{
			for(fj = 0; fj < fw; ++fj)
			{
				for(k = 0; k < xd; ++k)
				{
					dl_dw = s*nn_tensor_get(dL_dW, f, fi, fj, k);
					w     = nn_tensor_get(W, f, fi, fj, k);

					// Nesterov Momentum Update and L2 Regularization
					v0 = nn_tensor_get(VW, f, fi, fj, k);
					v1 = mu*v0 - lr*(gc->gcw*dl_dw + 2*lambda*w);
					nn_tensor_set(VW, f, fi, fj, k, v1);
					nn_tensor_add(W, f, fi, fj, k, -mu*v0 + (1 + mu)*v1);
				}
			}
		}

		// bias
		if((self->flags & NN_CONV_LAYER_FLAG_DISABLE_BIAS) == 0)
		{
			dl_db = s*nn_tensor_getv(dL_dB, f);

			// Nesterov Momentum Update
			v0 = nn_tensor_getv(VB, f);
			v1 = mu*v0 - lr*gc->gcb*dl_db;
			nn_tensor_setv(VB, f, v1);
			nn_tensor_addv(B, f, -mu*v0 + (1 + mu)*v1);
		}
	}

	return dL_dX;
}

static void
nn_convLayer_forwardPassT(nn_convLayer_t* self,
                          nn_tensor_t* X,
                          uint32_t m, uint32_t i,
                          uint32_t j, uint32_t f)
{
	ASSERT(self);

	nn_tensor_t* W     = self->W;
	nn_tensor_t* B     = self->B;
	nn_tensor_t* Y     = self->Y;
	nn_tensor_t* dL_dX = self->dL_dX;
	nn_dim_t*    dimW  = nn_tensor_dim(W);
	nn_dim_t*    dimX  = nn_tensor_dim(dL_dX);
	nn_dim_t*    dimY  = nn_tensor_dim(Y);
	uint32_t     fh    = dimW->height;
	uint32_t     fw    = dimW->width;
	uint32_t     xh    = dimX->height;
	uint32_t     xw    = dimX->width;
	uint32_t     xd    = dimX->depth;
	uint32_t     yh    = dimY->height;
	uint32_t     yw    = dimY->width;
	uint32_t     s     = self->stride;

	// initialize y
	float y;
	if(self->flags & NN_CONV_LAYER_FLAG_DISABLE_BIAS)
	{
		y = 0.0f;
	}
	else
	{
		y = nn_tensor_getv(B, f);
	}

	// compute weighted sum
	float    w;
	float    x;
	uint32_t fi;
	uint32_t fj;
	uint32_t k;
	uint32_t vh = yh + 2*(fh/2);         // virtual size
	uint32_t vw = yw + 2*(fw/2);
	uint32_t sh = xh + (xh - 1)*(s - 1); // strided size
	uint32_t sw = xw + (xw - 1)*(s - 1);
	uint32_t oi = (vh - sh)/2;           // strided offset
	uint32_t oj = (vw - sw)/2;
	int      ii;
	int      jj;
	for(fi = 0; fi < fh; ++fi)
	{
		// input index ii
		ii = ((int) (i + fi)) - ((int) oi);
		if((ii < 0) || (ii >= sh) || (ii%s))
		{
			continue;
		}
		ii = ii/s;

		for(fj = 0; fj < fw; ++fj)
		{
			// input index jj
			jj = ((int) (j + fj)) - ((int) oj);
			if((jj < 0) || (jj >= sw) || (jj%s))
			{
				continue;
			}
			jj = jj/s;

			for(k = 0; k < xd; ++k)
			{
				w  = nn_tensor_get(W, f, fi, fj, k);
				x  = nn_tensor_get(X, m, ii, jj, k);
				y += w*x;
			}
		}
	}
	nn_tensor_set(Y, m, i, j, f, y);
}

static void
nn_convLayer_postFn(nn_layer_t* base,
                    nn_layerMode_e mode)
{
	ASSERT(base);

	nn_arch_t*      arch  = base->arch;
	nn_archState_t* state = &arch->state;

	if((mode == NN_LAYER_MODE_TRAIN)   &&
	   (state->clip_max_weight > 0.0f) &&
	   (state->clip_max_bias   > 0.0f) &&
	   (state->clip_mu_inc     > 0.0f) &&
	   (state->clip_mu_dec     > 0.0f))
	{
		#ifdef NN_GC_DEBUG
		nn_convLayer_t*   self = (nn_convLayer_t*) base;
		nn_convLayerGc_t* gc   = &self->gc;
		LOGI("norm: w=%f, b=%f, dl_dw=%f, dl_dw_ra=%f, dl_db=%f, dl_db_ra=%f",
		     gc->norm_w, gc->norm_b,
		     gc->norm_dl_dw, gc->norm_dl_dw_ra,
		     gc->norm_dl_db, gc->norm_dl_db_ra);
		#endif
	}
}

static nn_tensor_t*
nn_convLayer_forwardPassTFn(nn_layer_t* base,
                            nn_layerMode_e mode,
                            uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_convLayer_t* self = (nn_convLayer_t*) base;

	nn_dim_t* dimW = nn_tensor_dim(self->W);
	nn_dim_t* dimY = nn_tensor_dim(self->Y);
	uint32_t  fc   = dimW->count;
	uint32_t  yh   = dimY->height;
	uint32_t  yw   = dimY->width;

	// forward pass Y
	uint32_t m;
	uint32_t i;
	uint32_t j;
	uint32_t f;
	for(m = 0; m < bs; ++m)
	{
		for(i = 0; i < yh; ++i)
		{
			for(j = 0; j < yw; ++j)
			{
				for(f = 0; f < fc; ++f)
				{
					nn_convLayer_forwardPassT(self, X, m, i, j, f);
				}
			}
		}
	}

	// store reference
	self->X = X;

	return self->Y;
}

static void
nn_convLayer_backpropT(nn_convLayer_t* self,
                       nn_tensor_t* dL_dY,
                       uint32_t m, uint32_t i,
                       uint32_t j, uint32_t f)
{
	ASSERT(self);
	ASSERT(dL_dY); // dim(bs, yh, yw, fc)

	nn_tensor_t* W     = self->W;
	nn_tensor_t* dY_dX = W;
	nn_tensor_t* dY_dW = self->X;
	nn_tensor_t* dL_dW = self->dL_dW;
	nn_tensor_t* dL_dB = self->dL_dB;
	nn_tensor_t* dL_dX = self->dL_dX;
	nn_dim_t*    dimW  = nn_tensor_dim(W);
	nn_dim_t*    dimX  = nn_tensor_dim(dL_dX);
	nn_dim_t*    dimY  = nn_tensor_dim(dL_dY);
	uint32_t     fh    = dimW->height;
	uint32_t     fw    = dimW->width;
	uint32_t     xh    = dimX->height;
	uint32_t     xw    = dimX->width;
	uint32_t     xd    = dimX->depth;
	uint32_t     yh    = dimY->height;
	uint32_t     yw    = dimY->width;
	uint32_t     s     = self->stride;

	float    dl_dy = nn_tensor_get(dL_dY, m, i, j, f);
	float    dy_dx;
	float    dy_dw;
	float    dy_db = 1.0f;
	uint32_t fi;
	uint32_t fj;
	uint32_t k;
	uint32_t vh = yh + 2*(fh/2);         // virtual size
	uint32_t vw = yw + 2*(fw/2);
	uint32_t sh = xh + (xh - 1)*(s - 1); // strided size
	uint32_t sw = xw + (xw - 1)*(s - 1);
	uint32_t oi = (vh - sh)/2;           // strided offset
	uint32_t oj = (vw - sw)/2;
	int      ii;
	int      jj;
	for(fi = 0; fi < fh; ++fi)
	{
		// input index ii
		ii = ((int) (i + fi)) - ((int) oi);
		if((ii < 0) || (ii >= sh) || (ii%s))
		{
			continue;
		}
		ii = ii/s;

		for(fj = 0; fj < fw; ++fj)
		{
			// input index jj
			jj = ((int) (j + fj)) - ((int) oj);
			if((jj < 0) || (jj >= sw) || (jj%s))
			{
				continue;
			}
			jj = jj/s;

			for(k = 0; k < xd; ++k)
			{
				// backpropagate dL_dX
				dy_dx = nn_tensor_get(dY_dX, f, fi, fj, k);
				nn_tensor_add(dL_dX, m, ii, jj, k, dl_dy*dy_dx);

				// sum dL_dW
				dy_dw = nn_tensor_get(dY_dW, m, ii, jj, k);
				nn_tensor_add(dL_dW, f, fi, fj, k, dl_dy*dy_dw);
			}
		}
	}

	// sum dL_dB
	if((self->flags & NN_CONV_LAYER_FLAG_DISABLE_BIAS) == 0)
	{
		nn_tensor_addv(dL_dB, f, dl_dy*dy_db);
	}
}

static nn_tensor_t*
nn_convLayer_backpropTFn(nn_layer_t* base, uint32_t bs,
                         nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,yh,yw,fc)

	nn_convLayer_t*   self  = (nn_convLayer_t*) base;
	nn_convLayerGc_t* gc    = &self->gc;
	nn_arch_t*        arch  = base->arch;
	nn_archState_t*   state = &arch->state;

	nn_tensor_t* W      = self->W;
	nn_tensor_t* B      = self->B;
	nn_tensor_t* VW     = self->VW;
	nn_tensor_t* VB     = self->VB;
	nn_dim_t*    dimY   = nn_tensor_dim(dL_dY);
	nn_dim_t*    dimW   = nn_tensor_dim(W);
	uint32_t     yh     = dimY->height;
	uint32_t     yw     = dimY->width;
	uint32_t     fc     = dimW->count;
	uint32_t     fh     = dimW->height;
	uint32_t     fw     = dimW->width;
	uint32_t     xd     = dimW->depth;
	float        lr     = state->learning_rate;
	float        mu     = state->momentum_decay;
	float        lambda = state->l2_lambda;

	// clear backprop gradients
	nn_tensor_t* dL_dW = self->dL_dW;
	nn_tensor_t* dL_dB = self->dL_dB;
	nn_tensor_t* dL_dX = self->dL_dX;
	nn_tensor_clear(dL_dW, NN_TENSOR_HAZZARD_NONE);
	if((self->flags & NN_CONV_LAYER_FLAG_DISABLE_BIAS) == 0)
	{
		nn_tensor_clear(dL_dB, NN_TENSOR_HAZZARD_NONE);
	}
	nn_tensor_clear(dL_dX, NN_TENSOR_HAZZARD_NONE);

	// sum gradients and backpropagate loss
	uint32_t m;
	uint32_t i;
	uint32_t j;
	uint32_t f;
	for(m = 0; m < bs; ++m)
	{
		for(i = 0; i < yh; ++i)
		{
			for(j = 0; j < yw; ++j)
			{
				for(f = 0; f < fc; ++f)
				{
					nn_convLayer_backpropT(self, dL_dY, m, i, j, f);
				}
			}
		}
	}

	// optionally compute gradient clipping
	// initialize gc but keep running averages
	gc->gcw        = 1.0f;
	gc->gcb        = 1.0f;
	gc->norm_w     = 0.0f;
	gc->norm_b     = 0.0f;
	gc->norm_dl_dw = 0.0f;
	gc->norm_dl_db = 0.0f;
	if((state->clip_max_weight > 0.0f) &&
	   (state->clip_max_bias   > 0.0f) &&
	   (state->clip_mu_inc     > 0.0f) &&
	   (state->clip_mu_dec     > 0.0f))
	{
		nn_convLayer_gradientClipping(self, bs);
	}

	// update parameters
	float    v0;
	float    v1;
	float    w;
	float    dl_dw;
	float    dl_db;
	float    s = 1.0f/((float) bs);
	uint32_t fi;
	uint32_t fj;
	uint32_t k;
	for(f = 0; f < fc; ++f)
	{
		// weights
		for(fi = 0; fi < fh; ++fi)
		{
			for(fj = 0; fj < fw; ++fj)
			{
				for(k = 0; k < xd; ++k)
				{
					dl_dw = s*nn_tensor_get(dL_dW, f, fi, fj, k);
					w     = nn_tensor_get(W, f, fi, fj, k);

					// Nesterov Momentum Update and L2 Regularization
					v0 = nn_tensor_get(VW, f, fi, fj, k);
					v1 = mu*v0 - lr*(gc->gcw*dl_dw + 2*lambda*w);
					nn_tensor_set(VW, f, fi, fj, k, v1);
					nn_tensor_add(W, f, fi, fj, k, -mu*v0 + (1 + mu)*v1);
				}
			}
		}

		// bias
		if((self->flags & NN_CONV_LAYER_FLAG_DISABLE_BIAS) == 0)
		{
			dl_db = s*nn_tensor_getv(dL_dB, f);

			// Nesterov Momentum Update
			v0 = nn_tensor_getv(VB, f);
			v1 = mu*v0 - lr*gc->gcb*dl_db;
			nn_tensor_setv(VB, f, v1);
			nn_tensor_addv(B, f, -mu*v0 + (1 + mu)*v1);
		}
	}

	return dL_dX;
}

static int
nn_convLayer_newCompute(nn_convLayer_t* self)
{
	return 1;
}

static void
nn_convLayer_deleteCompute(nn_convLayer_t* self)
{
}

#endif

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

nn_convLayer_t*
nn_convLayer_new(nn_arch_t* arch, nn_dim_t* dimX,
                 nn_dim_t* dimW, uint32_t stride,
                 int flags)
{
	ASSERT(arch);
	ASSERT(dimX);
	ASSERT(dimW);

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
		self->W = nn_tensor_new(arch, dimW,
		                        NN_TENSOR_INIT_HE,
		                        NN_TENSOR_MODE_COMPUTE);
	}
	else
	{
		self->W = nn_tensor_new(arch, dimW,
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

	self->B = nn_tensor_new(arch, &dimB,
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

	self->Y = nn_tensor_new(arch, &dimY,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_COMPUTE);
	if(self->Y == NULL)
	{
		goto fail_Y;
	}

	self->VW = nn_tensor_new(arch, dimW,
	                         NN_TENSOR_INIT_ZERO,
	                         NN_TENSOR_MODE_COMPUTE);
	if(self->VW == NULL)
	{
		goto fail_VW;
	}

	self->VB = nn_tensor_new(arch, &dimB,
	                         NN_TENSOR_INIT_ZERO,
	                         NN_TENSOR_MODE_COMPUTE);
	if(self->VB == NULL)
	{
		goto fail_VB;
	}

	self->dL_dW = nn_tensor_new(arch, dimW,
	                            NN_TENSOR_INIT_ZERO,
	                            NN_TENSOR_MODE_COMPUTE);
	if(self->dL_dW == NULL)
	{
		goto fail_dL_dW;
	}

	self->dL_dB = nn_tensor_new(arch, &dimB,
	                            NN_TENSOR_INIT_ZERO,
	                            NN_TENSOR_MODE_COMPUTE);
	if(self->dL_dB == NULL)
	{
		goto fail_dL_dB;
	}

	self->dL_dX = nn_tensor_new(arch, dimX,
	                            NN_TENSOR_INIT_ZERO,
	                            NN_TENSOR_MODE_COMPUTE);
	if(self->dL_dX == NULL)
	{
		goto fail_dL_dX;
	}

	if(nn_convLayer_newCompute(self) == 0)
	{
		goto fail_compute;
	}

	// success
	return self;

	// failure
	fail_compute:
		nn_tensor_delete(&self->dL_dX);
	fail_dL_dX:
		nn_tensor_delete(&self->dL_dB);
	fail_dL_dB:
		nn_tensor_delete(&self->dL_dW);
	fail_dL_dW:
		nn_tensor_delete(&self->VB);
	fail_VB:
		nn_tensor_delete(&self->VW);
	fail_VW:
		nn_tensor_delete(&self->Y);
	fail_Y:
		nn_tensor_delete(&self->B);
	fail_B:
		nn_tensor_delete(&self->W);
	fail_W:
		nn_layer_delete((nn_layer_t**) &self);
	return NULL;
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

	jsmn_val_t* val_dimX          = NULL;
	jsmn_val_t* val_dimW          = NULL;
	jsmn_val_t* val_flags         = NULL;
	jsmn_val_t* val_stride        = NULL;
	jsmn_val_t* val_W             = NULL;
	jsmn_val_t* val_B             = NULL;
	jsmn_val_t* val_VW            = NULL;
	jsmn_val_t* val_VB            = NULL;
	jsmn_val_t* val_norm_dl_dw_ra = NULL;
	jsmn_val_t* val_norm_dl_db_ra = NULL;

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
			else if(strcmp(kv->key, "norm_dl_dw_ra") == 0)
			{
				val_norm_dl_dw_ra = kv->val;
			}
			else if(strcmp(kv->key, "norm_dl_db_ra") == 0)
			{
				val_norm_dl_db_ra = kv->val;
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
			else if(strcmp(kv->key, "VW") == 0)
			{
				val_VW = kv->val;
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
	   (val_VW            == NULL) ||
	   (val_VB            == NULL) ||
	   (val_norm_dl_dw_ra == NULL) ||
	   (val_norm_dl_db_ra == NULL))
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

	// initialize running averages
	self->gc.norm_dl_dw_ra = strtof(val_norm_dl_dw_ra->data, NULL);
	self->gc.norm_dl_db_ra = strtof(val_norm_dl_db_ra->data, NULL);

	// load tensors
	if((nn_tensor_load(self->W,  val_W)  == 0) ||
	   (nn_tensor_load(self->B,  val_B)  == 0) ||
	   (nn_tensor_load(self->VW, val_VW) == 0) ||
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
	ret &= jsmn_stream_key(stream, "%s", "VW");
	ret &= nn_tensor_store(self->VW, stream);
	ret &= jsmn_stream_key(stream, "%s", "VB");
	ret &= nn_tensor_store(self->VB, stream);
	ret &= jsmn_stream_key(stream, "%s", "norm_dl_dw_ra");
	ret &= jsmn_stream_float(stream, self->gc.norm_dl_dw_ra);
	ret &= jsmn_stream_key(stream, "%s", "norm_dl_db_ra");
	ret &= jsmn_stream_float(stream, self->gc.norm_dl_db_ra);
	ret &= jsmn_stream_end(stream);

	return ret;
}

void nn_convLayer_delete(nn_convLayer_t** _self)
{
	ASSERT(_self);

	nn_convLayer_t* self = *_self;
	if(self)
	{
		nn_convLayer_deleteCompute(self);
		nn_tensor_delete(&self->dL_dX);
		nn_tensor_delete(&self->dL_dB);
		nn_tensor_delete(&self->dL_dW);
		nn_tensor_delete(&self->VB);
		nn_tensor_delete(&self->VW);
		nn_tensor_delete(&self->Y);
		nn_tensor_delete(&self->B);
		nn_tensor_delete(&self->W);
		nn_layer_delete((nn_layer_t**) _self);
	}
}
