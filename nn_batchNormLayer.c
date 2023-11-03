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

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define LOG_TAG "nn"
#include "../libcc/cc_log.h"
#include "../libcc/cc_memory.h"
#include "nn_arch.h"
#include "nn_engine.h"
#include "nn_batchNormLayer.h"
#include "nn_layer.h"
#include "nn_tensor.h"

/***********************************************************
* private                                                  *
***********************************************************/

static nn_tensor_t*
nn_batchNormLayer_forwardPassFn(nn_layer_t* base,
                                nn_layerMode_e layer_mode,
                                uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_batchNormLayer_t* self   = (nn_batchNormLayer_t*) base;
	nn_arch_t*           arch   = base->arch;
	nn_engine_t*         engine = arch->engine;

	nn_tensor_t* G        = self->G;
	nn_tensor_t* B        = self->B;
	nn_tensor_t* Xhat     = self->Xhat;
	nn_tensor_t* Y        = self->Y;
	nn_tensor_t* Xmean_mb = self->Xmean_mb;
	nn_tensor_t* Xvar_mb  = self->Xvar_mb;
	nn_tensor_t* Xmean_ra = self->Xmean_ra;
	nn_tensor_t* Xvar_ra  = self->Xvar_ra;
	nn_dim_t*    dim      = nn_tensor_dim(X);
	uint32_t     xh       = dim->height;
	uint32_t     xw       = dim->width;
	uint32_t     xd       = dim->depth;

	// prediction (running average) or
	// training (mini-batch) or instance normalization
	nn_tensor_t* Xmean = self->Xmean_ra;
	nn_tensor_t* Xvar  = self->Xvar_ra;
	if((layer_mode == NN_LAYER_MODE_TRAIN) ||
	   (self->bn_mode == NN_BATCH_NORM_MODE_INSTANCE))
	{
		Xmean = self->Xmean_mb;
		Xvar  = self->Xvar_mb;
	}

	// sb00: state
	// sb01: dimXhat
	// sb02: Xhat
	// sb03: dimG
	// sb04: G
	// sb05: dimB
	// sb06: B
	// sb07: dimXvar_mb
	// sb08: Xvar_mb
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
			.buffer  = Xhat->sb_dim,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Xhat->sb_data,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = G->sb_dim,
		},
		{
			.binding = 4,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = G->sb_data,
		},
		{
			.binding = 5,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = B->sb_dim,
		},
		{
			.binding = 6,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = B->sb_data,
		},
		{
			.binding = 7,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Xvar_mb->sb_dim,
		},
		{
			.binding = 8,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Xvar_mb->sb_data,
		},
	};

	// sb10:  dimX
	// sb11:  X
	// sb12:  dimY
	// sb13:  Y
	// sb14:  dimXmean
	// sb15:  Xmean
	// sb16:  dimXvar
	// sb17:  Xvar
	// sb18:  dimXmean_mb
	// sb19:  Xmean_mb
	// sb110: dimXmean_ra
	// sb111: Xmean_ra
	// sb112: dimXvar_ra
	// sb113: Xvar_ra
	vkk_uniformAttachment_t ua1_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X->sb_dim,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X->sb_data,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Y->sb_dim,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Y->sb_data,
		},
		{
			.binding = 4,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Xmean->sb_dim,
		},
		{
			.binding = 5,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Xmean->sb_data,
		},
		{
			.binding = 6,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Xvar->sb_dim,
		},
		{
			.binding = 7,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Xvar->sb_data,
		},
		{
			.binding = 8,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Xmean_mb->sb_dim,
		},
		{
			.binding = 9,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Xmean_mb->sb_data,
		},
		{
			.binding = 10,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Xmean_ra->sb_dim,
		},
		{
			.binding = 11,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Xmean_ra->sb_data,
		},
		{
			.binding = 12,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Xvar_ra->sb_dim,
		},
		{
			.binding = 13,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Xvar_ra->sb_data,
		},
	};

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1,
	};

	// update once after first pipeline is bound
	int update = 1;

	uint32_t k;
	vkk_computePipeline_t* cp;
	vkk_uniformSet_t*      us3;
	if((layer_mode == NN_LAYER_MODE_TRAIN) ||
	   (self->bn_mode == NN_BATCH_NORM_MODE_INSTANCE))
	{
		// nn_batchNormLayer_forwardPassXmean
		// dispatch required for each k
		// dispatch((k == 0) ? RAW : NONE, 1, 1, 1, 8, 8, 1)

		cp = engine->cp_batchNorm_forwardPassXmean;
		if(nn_engine_bind(engine, cp) == 0)
		{
			return NULL;
		}
		vkk_compute_updateUniformSetRefs(engine->compute, self->us0,
		                                 9, ua0_array);
		vkk_compute_updateUniformSetRefs(engine->compute, self->us1,
		                                 14, ua1_array);
		vkk_compute_bindUniformSets(engine->compute, 2, us_array);
		update = 0;

		for(k = 0; k < xd; ++k)
		{
			us3 = nn_engine_getBatchNormIdx(engine, k);
			if(us3 == NULL)
			{
				return NULL;
			}
			vkk_compute_bindUniformSets(engine->compute, 1, &us3);
			if(k == 0)
			{
				nn_engine_dispatch(engine, VKK_HAZZARD_RAW,
				                   1, 1, 1, 8, 8, 1);
			}
			else
			{
				nn_engine_dispatch(engine, VKK_HAZZARD_NONE,
				                   1, 1, 1, 8, 8, 1);
			}
		}

		// nn_batchNormLayer_forwardPassXvar
		// dispatch required for each k
		// dispatch((k == 0) ? RAW : NONE, 1, 1, 1, 8, 8, 1)
		cp = engine->cp_batchNorm_forwardPassXvar;
		if(nn_engine_bind(engine, cp) == 0)
		{
			return NULL;
		}

		for(k = 0; k < xd; ++k)
		{
			us3 = nn_engine_getBatchNormIdx(engine, k);
			if(us3 == NULL)
			{
				return NULL;
			}
			vkk_compute_bindUniformSets(engine->compute, 1, &us3);
			if(k == 0)
			{
				nn_engine_dispatch(engine, VKK_HAZZARD_RAW,
				                   1, 1, 1, 8, 8, 1);
			}
			else
			{
				nn_engine_dispatch(engine, VKK_HAZZARD_NONE,
				                   1, 1, 1, 8, 8, 1);
			}
		}
	}

	// nn_batchNormLayer_forwardPassXhat
	// dispatch(RAW, bs, xh, xw, 1, 8, 8)
	cp = engine->cp_batchNorm_forwardPassXhat;
	if(nn_engine_bind(engine, cp) == 0)
	{
		return NULL;
	}
	if(update)
	{
		vkk_compute_updateUniformSetRefs(engine->compute, self->us0,
		                                 9, ua0_array);
		vkk_compute_updateUniformSetRefs(engine->compute, self->us1,
		                                 14, ua1_array);
		vkk_compute_bindUniformSets(engine->compute, 2, us_array);
		update = 0;
	}
	nn_engine_dispatch(engine, VKK_HAZZARD_RAW,
	                   bs, xh, xw, 1, 8, 8);

	// nn_batchNormLayer_forwardPassY
	// dispatch(RAW, bs, xh, xw, 1, 8, 8)
	cp = engine->cp_batchNorm_forwardPassY;
	if(nn_engine_bind(engine, cp) == 0)
	{
		return NULL;
	}
	nn_engine_dispatch(engine, VKK_HAZZARD_RAW,
	                   bs, xh, xw, 1, 8, 8);

	return Y;
}

static nn_tensor_t*
nn_batchNormLayer_backpropFn(nn_layer_t* base,
                             nn_layerMode_e layer_mode,
                             uint32_t bs,
                             nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,xh,xw,xd)

	nn_batchNormLayer_t* self   = (nn_batchNormLayer_t*) base;
	nn_arch_t*           arch   = base->arch;
	nn_engine_t*         engine = arch->engine;

	nn_tensor_t* dL_dXhat = self->dL_dXhat;
	nn_tensor_t* Bsum     = self->Bsum;
	nn_tensor_t* Csum     = self->Csum;
	nn_dim_t*    dim      = nn_tensor_dim(dL_dY);
	uint32_t     xh       = dim->height;
	uint32_t     xw       = dim->width;
	uint32_t     xd       = dim->depth;

	// sb20: dim_dL_dXhat
	// sb21: dL_dXhat
	// sb22: dim_dL_dY
	// sb23: dL_dY
	// sb24: dimBsum
	// sb25: Bsum
	// sb26: dimCsum
	// sb27: Csum
	vkk_uniformAttachment_t ua2_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dXhat->sb_dim,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dXhat->sb_data,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY->sb_dim,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY->sb_data,
		},
		{
			.binding = 4,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Bsum->sb_dim,
		},
		{
			.binding = 5,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Bsum->sb_data,
		},
		{
			.binding = 6,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Csum->sb_dim,
		},
		{
			.binding = 7,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Csum->sb_data,
		},
	};

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1,
		self->us2,
	};

	// nn_batchNormLayer_backprop_dL_dXhat
	// dispatch(RAW, bs, xh, xw, 1, 8, 8)
	vkk_computePipeline_t* cp;
	cp = engine->cp_batchNorm_backprop_dL_dXhat;
	if(nn_engine_bind(engine, cp) == 0)
	{
		return NULL;
	}
	vkk_compute_updateUniformSetRefs(engine->compute, self->us2,
	                                 8, ua2_array);
	vkk_compute_bindUniformSets(engine->compute, 3, us_array);
	nn_engine_dispatch(engine, VKK_HAZZARD_RAW,
	                   bs, xh, xw, 1, 8, 8);

	// optionally skip parameter update
	// nn_batchNormLayer_backpropSum or
	// nn_batchNormLayer_backpropSumNOP
	// dispatch required for each k
	// dispatch((k == 0) ? RAW : NONE, 1, 1, 1, 8, 8, 1)
	uint32_t k;
	vkk_uniformSet_t* us3;
	if(layer_mode == NN_LAYER_MODE_TRAIN_NOP)
	{
		cp = engine->cp_batchNorm_backpropSumNOP;
	}
	else
	{
		cp = engine->cp_batchNorm_backpropSum;
	}
	if(nn_engine_bind(engine, cp) == 0)
	{
		return NULL;
	}

	for(k = 0; k < xd; ++k)
	{
		us3 = nn_engine_getBatchNormIdx(engine, k);
		if(us3 == NULL)
		{
			return NULL;
		}
		vkk_compute_bindUniformSets(engine->compute, 1, &us3);
		if(k == 0)
		{
			nn_engine_dispatch(engine, VKK_HAZZARD_RAW,
			                   1, 1, 1, 8, 8, 1);
		}
		else
		{
			nn_engine_dispatch(engine, VKK_HAZZARD_NONE,
			                   1, 1, 1, 8, 8, 1);
		}
	}

	// nn_batchNorm_backprop_dL_dX
	// dispatch(RAW, bs, xh, xw, 1, 8, 8)
	cp = engine->cp_batchNorm_backprop_dL_dX;
	if(nn_engine_bind(engine, cp) == 0)
	{
		return NULL;
	}
	nn_engine_dispatch(engine, VKK_HAZZARD_RAW,
	                   bs, xh, xw, 1, 8, 8);

	// dL_dY replaced by dL_dX
	return dL_dY;
}

static int
nn_batchNormLayer_newCompute(nn_batchNormLayer_t* self)
{
	ASSERT(self);

	nn_arch_t*   arch   = self->base.arch;
	nn_engine_t* engine = arch->engine;
	nn_dim_t*    dimG   = nn_tensor_dim(self->G);

	self->Bsum = nn_tensor_new(engine, dimG,
	                           NN_TENSOR_INIT_ZERO,
	                           NN_TENSOR_MODE_COMPUTE);
	if(self->Bsum == NULL)
	{
		return 0;
	}

	self->Csum = nn_tensor_new(engine, dimG,
	                           NN_TENSOR_INIT_ZERO,
	                           NN_TENSOR_MODE_COMPUTE);
	if(self->Csum == NULL)
	{
		goto fail_Csum;
	}

	self->us0 = vkk_uniformSet_new(engine->engine, 0, 0, NULL,
	                               engine->usf0_batchNorm);
	if(self->us0 == NULL)
	{
		goto fail_us0;
	}

	self->us1 = vkk_uniformSet_new(engine->engine, 1, 0, NULL,
	                               engine->usf1_batchNorm);
	if(self->us1 == NULL)
	{
		goto fail_us1;
	}

	self->us2 = vkk_uniformSet_new(engine->engine, 2, 0, NULL,
	                               engine->usf2_batchNorm);
	if(self->us2 == NULL)
	{
		goto fail_us2;
	}

	// success
	return 1;

	// failure
	fail_us2:
		vkk_uniformSet_delete(&self->us1);
	fail_us1:
		vkk_uniformSet_delete(&self->us0);
	fail_us0:
		nn_tensor_delete(&self->Csum);
	fail_Csum:
		nn_tensor_delete(&self->Bsum);
	return 0;
}

static void
nn_batchNormLayer_deleteCompute(nn_batchNormLayer_t* self)
{
	ASSERT(self);

	vkk_uniformSet_delete(&self->us2);
	vkk_uniformSet_delete(&self->us1);
	vkk_uniformSet_delete(&self->us0);
	nn_tensor_delete(&self->Csum);
	nn_tensor_delete(&self->Bsum);
}

static nn_dim_t*
nn_batchNormLayer_dimXFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_batchNormLayer_t* self = (nn_batchNormLayer_t*) base;

	return nn_tensor_dim(self->Xhat);
}

static nn_dim_t*
nn_batchNormLayer_dimYFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_batchNormLayer_t* self = (nn_batchNormLayer_t*) base;

	return nn_tensor_dim(self->Y);
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_batchNormLayer_t*
nn_batchNormLayer_new(nn_arch_t* arch,
                      nn_batchNormMode_e bn_mode,
                      nn_dim_t* dimX)
{
	ASSERT(arch);
	ASSERT(dimX);

	nn_engine_t* engine = arch->engine;

	uint32_t xd = dimX->depth;

	nn_dim_t dim_111d =
	{
		.count  = 1,
		.height = 1,
		.width  = 1,
		.depth  = xd,
	};

	nn_layerInfo_t info =
	{
		.arch            = arch,
		.forward_pass_fn = nn_batchNormLayer_forwardPassFn,
		.backprop_fn     = nn_batchNormLayer_backpropFn,
		.dimX_fn         = nn_batchNormLayer_dimXFn,
		.dimY_fn         = nn_batchNormLayer_dimYFn,
	};

	nn_batchNormLayer_t* self;
	self = (nn_batchNormLayer_t*)
	       nn_layer_new(sizeof(nn_batchNormLayer_t), &info);
	if(self == NULL)
	{
		return NULL;
	}

	self->bn_mode = bn_mode;

	self->G = nn_tensor_new(engine, &dim_111d,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_COMPUTE);
	if(self->G == NULL)
	{
		goto fail_G;
	}

	nn_tensor_t* tmpG;
	tmpG = nn_tensor_new(engine, &dim_111d,
	                     NN_TENSOR_INIT_ZERO,
	                     NN_TENSOR_MODE_IO);
	if(tmpG == NULL)
	{
		goto fail_tmpG;
	}

	// initialize G to 1.0f
	uint32_t k;
	for(k = 0; k < xd; ++k)
	{
		nn_tensor_set(tmpG, 0, 0, 0, k, 1.0f);
	}

	if(nn_tensor_blit(tmpG, self->G, 1, 0, 0) == 0)
	{
		goto fail_blitG;
	}

	self->B = nn_tensor_new(engine, &dim_111d,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_COMPUTE);
	if(self->B == NULL)
	{
		goto fail_B;
	}

	self->Xhat = nn_tensor_new(engine, dimX,
	                           NN_TENSOR_INIT_ZERO,
	                           NN_TENSOR_MODE_COMPUTE);
	if(self->Xhat == NULL)
	{
		goto fail_Xhat;
	}

	self->Y = nn_tensor_new(engine, dimX,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_COMPUTE);
	if(self->Y == NULL)
	{
		goto fail_Y;
	}

	self->Xmean_mb = nn_tensor_new(engine, &dim_111d,
	                               NN_TENSOR_INIT_ZERO,
	                               NN_TENSOR_MODE_COMPUTE);
	if(self->Xmean_mb == NULL)
	{
		goto fail_Xmean_mb;
	}

	self->Xvar_mb = nn_tensor_new(engine, &dim_111d,
	                              NN_TENSOR_INIT_ZERO,
	                              NN_TENSOR_MODE_COMPUTE);
	if(self->Xvar_mb == NULL)
	{
		goto fail_Xvar_mb;
	}

	self->Xmean_ra = nn_tensor_new(engine, &dim_111d,
	                               NN_TENSOR_INIT_ZERO,
	                               NN_TENSOR_MODE_COMPUTE);
	if(self->Xmean_ra == NULL)
	{
		goto fail_Xmean_ra;
	}

	self->Xvar_ra = nn_tensor_new(engine, &dim_111d,
	                              NN_TENSOR_INIT_ZERO,
	                              NN_TENSOR_MODE_COMPUTE);
	if(self->Xvar_ra == NULL)
	{
		goto fail_Xvar_ra;
	}

	self->dL_dXhat = nn_tensor_new(engine, dimX,
	                               NN_TENSOR_INIT_ZERO,
	                               NN_TENSOR_MODE_COMPUTE);
	if(self->dL_dXhat == NULL)
	{
		goto fail_dL_dXhat;
	}

	if(nn_batchNormLayer_newCompute(self) == 0)
	{
		goto fail_compute;
	}

	nn_tensor_delete(&tmpG);

	// success
	return self;

	// failure
	fail_compute:
		nn_tensor_delete(&self->dL_dXhat);
	fail_dL_dXhat:
		nn_tensor_delete(&self->Xvar_ra);
	fail_Xvar_ra:
		nn_tensor_delete(&self->Xmean_ra);
	fail_Xmean_ra:
		nn_tensor_delete(&self->Xvar_mb);
	fail_Xvar_mb:
		nn_tensor_delete(&self->Xmean_mb);
	fail_Xmean_mb:
		nn_tensor_delete(&self->Y);
	fail_Y:
		nn_tensor_delete(&self->Xhat);
	fail_Xhat:
		nn_tensor_delete(&self->B);
	fail_B:
	fail_blitG:
		nn_tensor_delete(&tmpG);
	fail_tmpG:
		nn_tensor_delete(&self->G);
	fail_G:
		nn_layer_delete((nn_layer_t**) &self);
	return NULL;
}

nn_batchNormLayer_t*
nn_batchNormLayer_import(nn_arch_t* arch, jsmn_val_t* val)
{
	ASSERT(arch);
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	jsmn_val_t* val_bn_mode  = NULL;
	jsmn_val_t* val_dimX     = NULL;
	jsmn_val_t* val_G        = NULL;
	jsmn_val_t* val_B        = NULL;
	jsmn_val_t* val_Xmean_ra = NULL;
	jsmn_val_t* val_Xvar_ra  = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		jsmn_keyval_t* kv;
		kv = (jsmn_keyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == JSMN_TYPE_STRING)
		{
			if(strcmp(kv->key, "bn_mode") == 0)
			{
				val_bn_mode = kv->val;
			}
		}
		else if(kv->val->type == JSMN_TYPE_OBJECT)
		{
			if(strcmp(kv->key, "dimX") == 0)
			{
				val_dimX = kv->val;
			}
			else if(strcmp(kv->key, "G") == 0)
			{
				val_G = kv->val;
			}
			else if(strcmp(kv->key, "B") == 0)
			{
				val_B = kv->val;
			}
			else if(strcmp(kv->key, "Xmean_ra") == 0)
			{
				val_Xmean_ra = kv->val;
			}
			else if(strcmp(kv->key, "Xvar_ra") == 0)
			{
				val_Xvar_ra = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_bn_mode  == NULL) ||
	   (val_dimX     == NULL) ||
	   (val_G        == NULL) ||
	   (val_B        == NULL) ||
	   (val_Xmean_ra == NULL) ||
	   (val_Xvar_ra  == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_batchNormMode_e bn_mode;
	if(strcmp(val_bn_mode->data, "RUNNING") == 0)
	{
		bn_mode = NN_BATCH_NORM_MODE_RUNNING;
	}
	else if(strcmp(val_bn_mode->data, "INSTANCE") == 0)
	{
		bn_mode = NN_BATCH_NORM_MODE_INSTANCE;
	}
	else
	{
		LOGE("invalid bn_mode=%s", val_bn_mode->data);
		return NULL;
	}

	nn_dim_t dimX;
	if(nn_dim_load(&dimX, val_dimX) == 0)
	{
		return NULL;
	}

	nn_batchNormLayer_t* self;
	self = nn_batchNormLayer_new(arch, bn_mode, &dimX);
	if(self == NULL)
	{
		return NULL;
	}

	// load tensors
	if((nn_tensor_load(self->G,        val_G) == 0)        ||
	   (nn_tensor_load(self->B,        val_B) == 0)        ||
	   (nn_tensor_load(self->Xmean_ra, val_Xmean_ra) == 0) ||
	   (nn_tensor_load(self->Xvar_ra,  val_Xvar_ra) == 0))
	{
		goto fail_tensor;
	}

	// success
	return self;

	// failure
	fail_tensor:
		nn_batchNormLayer_delete(&self);
	return NULL;
}

int nn_batchNormLayer_export(nn_batchNormLayer_t* self,
                             jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	nn_dim_t* dimX = nn_tensor_dim(self->Xhat);

	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	ret &= jsmn_stream_key(stream, "%s", "bn_mode");
	if(self->bn_mode == NN_BATCH_NORM_MODE_RUNNING)
	{
		ret &= jsmn_stream_string(stream, "%s", "RUNNING");
	}
	else if(self->bn_mode == NN_BATCH_NORM_MODE_INSTANCE)
	{
		ret &= jsmn_stream_string(stream, "%s", "INSTANCE");
	}
	else
	{
		LOGE("invalid bn_mode=%i", (int) self->bn_mode);
		return 0;
	}
	ret &= jsmn_stream_key(stream, "%s", "dimX");
	ret &= nn_dim_store(dimX, stream);
	ret &= jsmn_stream_key(stream, "%s", "G");
	ret &= nn_tensor_store(self->G, stream);
	ret &= jsmn_stream_key(stream, "%s", "B");
	ret &= nn_tensor_store(self->B, stream);
	ret &= jsmn_stream_key(stream, "%s", "Xmean_ra");
	ret &= nn_tensor_store(self->Xmean_ra, stream);
	ret &= jsmn_stream_key(stream, "%s", "Xvar_ra");
	ret &= nn_tensor_store(self->Xvar_ra, stream);
	ret &= jsmn_stream_end(stream);

	return ret;
}

void nn_batchNormLayer_delete(nn_batchNormLayer_t** _self)
{
	ASSERT(_self);

	nn_batchNormLayer_t* self = *_self;
	if(self)
	{
		nn_batchNormLayer_deleteCompute(self);
		nn_tensor_delete(&self->dL_dXhat);
		nn_tensor_delete(&self->Xvar_ra);
		nn_tensor_delete(&self->Xmean_ra);
		nn_tensor_delete(&self->Xvar_mb);
		nn_tensor_delete(&self->Xmean_mb);
		nn_tensor_delete(&self->Y);
		nn_tensor_delete(&self->Xhat);
		nn_tensor_delete(&self->B);
		nn_tensor_delete(&self->G);
		nn_layer_delete((nn_layer_t**) &self);
	}
}
