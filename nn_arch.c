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
#include "../libcc/math/cc_float.h"
#include "../libcc/cc_log.h"
#include "../libcc/cc_memory.h"
#include "../libvkk/vkk.h"
#include "nn_arch.h"
#include "nn_engine.h"
#include "nn_layer.h"
#include "nn_loss.h"
#include "nn_tensor.h"

/***********************************************************
* private                                                  *
***********************************************************/

static void
nn_arch_post(nn_arch_t* self, int flags)
{
	ASSERT(self);

	cc_listIter_t* iter = cc_list_head(self->layers);
	while(iter)
	{
		nn_layer_t* layer;
		layer = (nn_layer_t*) cc_list_peekIter(iter);

		nn_layer_post(layer, flags);

		iter = cc_list_next(iter);
	}

	if(self->loss)
	{
		nn_loss_post(self->loss, flags);
	}
}

static int
nn_arch_init(nn_arch_t* self,
             uint32_t bs,
             nn_tensor_t* X,
             nn_tensor_t* Yt)
{
	// X and Yt may be NULL
	ASSERT(self);

	nn_engine_t* engine = self->engine;

	// optionally create X
	if(X && (X->tensor_mode == NN_TENSOR_MODE_IO))
	{
		if(self->X)
		{
			if(nn_dim_equals(nn_tensor_dim(self->X),
			                 nn_tensor_dim(X)) == 0)
			{
				nn_tensor_delete(&self->X);
			}
		}

		if(self->X == NULL)
		{
			self->X = nn_tensor_new(engine, nn_tensor_dim(X),
			                        NN_TENSOR_INIT_ZERO,
			                        NN_TENSOR_MODE_COMPUTE);
			if(self->X == NULL)
			{
				return 0;
			}
		}

		if(nn_tensor_blit(X, self->X, bs, 0, 0) == 0)
		{
			return 0;
		}
	}

	// optionally create Yt
	if(Yt && (Yt->tensor_mode == NN_TENSOR_MODE_IO))
	{
		if(self->Yt)
		{
			if(nn_dim_equals(nn_tensor_dim(self->Yt),
			                 nn_tensor_dim(Yt)) == 0)
			{
				nn_tensor_delete(&self->Yt);
			}
		}

		if(self->Yt == NULL)
		{
			self->Yt = nn_tensor_new(engine, nn_tensor_dim(Yt),
			                         NN_TENSOR_INIT_ZERO,
			                         NN_TENSOR_MODE_COMPUTE);
			if(self->Yt == NULL)
			{
				return 0;
			}
		}

		if(nn_tensor_blit(Yt, self->Yt, bs, 0, 0) == 0)
		{
			return 0;
		}
	}

	// update global state
	nn_archState_t* state = &self->state;
	state->bs = bs;
	if(Yt)
	{
		state->adam_beta1t *= state->adam_beta1;
		state->adam_beta2t *= state->adam_beta2;
	}
	vkk_compute_writeBuffer(engine->compute,
	                        self->sb_state,
	                        sizeof(nn_archState_t),
	                        0, state);

	return nn_engine_begin(engine);
}

static int
nn_arch_initFairCGAN(nn_arch_t* self,
                     uint32_t bs,
                     nn_tensor_t* Cg0,
                     nn_tensor_t* Cg1,
                     nn_tensor_t* Cr0,
                     nn_tensor_t* Cr1,
                     nn_tensor_t* Ytg,
                     nn_tensor_t* Ytr)
{
	// Cg1 and Cr1 are optional
	ASSERT(self);
	ASSERT(Cg0);
	ASSERT(Cr0);
	ASSERT(Ytg);
	ASSERT(Ytr);

	nn_engine_t* engine = self->engine;
	uint32_t     bs2    = bs/2;

	// create Cg0
	if(Cg0->tensor_mode == NN_TENSOR_MODE_IO)
	{
		if(self->Cg0)
		{
			if(nn_dim_equals(nn_tensor_dim(self->Cg0),
			                 nn_tensor_dim(Cg0)) == 0)
			{
				nn_tensor_delete(&self->Cg0);
			}
		}

		if(self->Cg0 == NULL)
		{
			self->Cg0 = nn_tensor_new(engine, nn_tensor_dim(Cg0),
			                          NN_TENSOR_INIT_ZERO,
			                          NN_TENSOR_MODE_COMPUTE);
			if(self->Cg0 == NULL)
			{
				return 0;
			}
		}

		if(nn_tensor_blit(Cg0, self->Cg0, bs2, 0, 0) == 0)
		{
			return 0;
		}
	}

	// optionally create Cg1
	if(Cg1 == NULL)
	{
		Cg1 = Cg0;
	}
	else if(Cg1->tensor_mode == NN_TENSOR_MODE_IO)
	{
		if(self->Cg1)
		{
			if(nn_dim_equals(nn_tensor_dim(self->Cg1),
			                 nn_tensor_dim(Cg1)) == 0)
			{
				nn_tensor_delete(&self->Cg1);
			}
		}

		if(self->Cg1 == NULL)
		{
			self->Cg1 = nn_tensor_new(engine, nn_tensor_dim(Cg1),
			                          NN_TENSOR_INIT_ZERO,
			                          NN_TENSOR_MODE_COMPUTE);
			if(self->Cg1 == NULL)
			{
				return 0;
			}
		}

		if(nn_tensor_blit(Cg1, self->Cg1, bs2, 0, 0) == 0)
		{
			return 0;
		}
	}

	// create Cr0
	if(Cr0->tensor_mode == NN_TENSOR_MODE_IO)
	{
		if(self->Cr0)
		{
			if(nn_dim_equals(nn_tensor_dim(self->Cr0),
			                 nn_tensor_dim(Cr0)) == 0)
			{
				nn_tensor_delete(&self->Cr0);
			}
		}

		if(self->Cr0 == NULL)
		{
			self->Cr0 = nn_tensor_new(engine, nn_tensor_dim(Cr0),
			                          NN_TENSOR_INIT_ZERO,
			                          NN_TENSOR_MODE_COMPUTE);
			if(self->Cr0 == NULL)
			{
				return 0;
			}
		}

		if(nn_tensor_blit(Cr0, self->Cr0, bs2, 0, 0) == 0)
		{
			return 0;
		}
	}

	// optionally create Cr1
	if(Cr1 == NULL)
	{
		Cr1 = Cr0;
	}
	else if(Cr1->tensor_mode == NN_TENSOR_MODE_IO)
	{
		if(self->Cr1)
		{
			if(nn_dim_equals(nn_tensor_dim(self->Cr1),
			                 nn_tensor_dim(Cr1)) == 0)
			{
				nn_tensor_delete(&self->Cr1);
			}
		}

		if(self->Cr1 == NULL)
		{
			self->Cr1 = nn_tensor_new(engine, nn_tensor_dim(Cr1),
			                          NN_TENSOR_INIT_ZERO,
			                          NN_TENSOR_MODE_COMPUTE);
			if(self->Cr1 == NULL)
			{
				return 0;
			}
		}

		if(nn_tensor_blit(Cr1, self->Cr1, bs2, 0, 0) == 0)
		{
			return 0;
		}
	}

	// create Ytg
	if(Ytg->tensor_mode == NN_TENSOR_MODE_IO)
	{
		if(self->Ytg)
		{
			if(nn_dim_equals(nn_tensor_dim(self->Ytg),
			                 nn_tensor_dim(Ytg)) == 0)
			{
				nn_tensor_delete(&self->Ytg);
			}
		}

		if(self->Ytg == NULL)
		{
			self->Ytg = nn_tensor_new(engine, nn_tensor_dim(Ytg),
			                          NN_TENSOR_INIT_ZERO,
			                          NN_TENSOR_MODE_COMPUTE);
			if(self->Ytg == NULL)
			{
				return 0;
			}
		}

		if(nn_tensor_blit(Ytg, self->Ytg, bs2, 0, 0) == 0)
		{
			return 0;
		}
	}

	// create Ytr
	if(Ytr->tensor_mode == NN_TENSOR_MODE_IO)
	{
		if(self->Ytr)
		{
			if(nn_dim_equals(nn_tensor_dim(self->Ytr),
			                 nn_tensor_dim(Ytr)) == 0)
			{
				nn_tensor_delete(&self->Ytr);
			}
		}

		if(self->Ytr == NULL)
		{
			self->Ytr = nn_tensor_new(engine, nn_tensor_dim(Ytr),
			                          NN_TENSOR_INIT_ZERO,
			                          NN_TENSOR_MODE_COMPUTE);
			if(self->Ytr == NULL)
			{
				return 0;
			}
		}

		if(nn_tensor_blit(Ytr, self->Ytr, bs2, 0, 0) == 0)
		{
			return 0;
		}
	}

	// create Xd
	nn_dim_t* dimCr  = nn_tensor_dim(Cr1);
	nn_dim_t* dimYtr = nn_tensor_dim(Ytr);
	nn_dim_t dimXd =
	{
		.count  = bs,
		.height = dimCr->height,
		.width  = dimCr->width,
		.depth  = dimCr->depth + dimYtr->depth,
	};

	if(self->Xd)
	{
		if(nn_dim_equals(nn_tensor_dim(self->Xd),
		                 &dimXd) == 0)
		{
			nn_tensor_delete(&self->Xd);
		}
	}

	if(self->Xd == NULL)
	{
		self->Xd = nn_tensor_new(engine, &dimXd,
		                         NN_TENSOR_INIT_ZERO,
		                         NN_TENSOR_MODE_COMPUTE);
		if(self->Xd == NULL)
		{
			return 0;
		}
	}

	// create dL_dYb
	nn_dim_t* dim_dL_dY = nn_tensor_dim(Ytg);
	if(self->dL_dYb)
	{
		if(nn_dim_equals(nn_tensor_dim(self->dL_dYb),
		                 dim_dL_dY) == 0)
		{
			nn_tensor_delete(&self->dL_dYb);
		}
	}

	if(self->dL_dYb == NULL)
	{
		self->dL_dYb = nn_tensor_new(engine, dim_dL_dY,
		                             NN_TENSOR_INIT_ZERO,
		                             NN_TENSOR_MODE_COMPUTE);
		if(self->dL_dYb == NULL)
		{
			return 0;
		}
	}

	// update gan_blend_factor
	nn_archState_t* state = &self->state;
	state->gan_blend_factor *= state->gan_blend_scalar;
	state->gan_blend_factor  = cc_clamp(state->gan_blend_factor,
	                                    state->gan_blend_min,
	                                    state->gan_blend_max);

	// update global state
	state->bs = bs2;
	state->adam_beta1t *= state->adam_beta1;
	state->adam_beta2t *= state->adam_beta2;
	vkk_compute_writeBuffer(engine->compute,
	                        self->sb_state,
	                        sizeof(nn_archState_t),
	                        0, state);

	return nn_engine_begin(engine);
}

static int
nn_arch_initLERP(nn_arch_t* self, nn_arch_t* lerp,
                 uint32_t bs,
                 nn_tensor_t* Xt, nn_tensor_t* Yt)
{
	ASSERT(self);
	ASSERT(lerp);
	ASSERT(Xt);
	ASSERT(Yt);

	nn_engine_t* engine = self->engine;

	// create Xt
	if(Xt->tensor_mode == NN_TENSOR_MODE_IO)
	{
		// X is Xt
		if(self->X)
		{
			if(nn_dim_equals(nn_tensor_dim(self->X),
			                 nn_tensor_dim(Xt)) == 0)
			{
				nn_tensor_delete(&self->X);
			}
		}

		if(self->X == NULL)
		{
			self->X = nn_tensor_new(engine, nn_tensor_dim(Xt),
			                        NN_TENSOR_INIT_ZERO,
			                        NN_TENSOR_MODE_COMPUTE);
			if(self->X == NULL)
			{
				return 0;
			}
		}

		if(nn_tensor_blit(Xt, self->X, bs, 0, 0) == 0)
		{
			return 0;
		}
	}

	// create Yt
	if(Yt->tensor_mode == NN_TENSOR_MODE_IO)
	{
		if(self->Yt)
		{
			if(nn_dim_equals(nn_tensor_dim(self->Yt),
			                 nn_tensor_dim(Yt)) == 0)
			{
				nn_tensor_delete(&self->Yt);
			}
		}

		if(self->Yt == NULL)
		{
			self->Yt = nn_tensor_new(engine, nn_tensor_dim(Yt),
			                          NN_TENSOR_INIT_ZERO,
			                          NN_TENSOR_MODE_COMPUTE);
			if(self->Yt == NULL)
			{
				return 0;
			}
		}

		if(nn_tensor_blit(Yt, self->Yt, bs, 0, 0) == 0)
		{
			return 0;
		}
	}

	// update global state
	nn_archState_t* state1 = &self->state;
	nn_archState_t* state2 = &lerp->state;
	state1->bs = bs;
	state2->bs = bs;
	state1->adam_beta1t *= state1->adam_beta1;
	state1->adam_beta2t *= state1->adam_beta2;
	state2->adam_beta1t *= state2->adam_beta1;
	state2->adam_beta2t *= state2->adam_beta2;
	vkk_compute_writeBuffer(engine->compute,
	                        self->sb_state,
	                        sizeof(nn_archState_t),
	                        0, state1);
	vkk_compute_writeBuffer(engine->compute,
	                        lerp->sb_state,
	                        sizeof(nn_archState_t),
	                        0, state2);

	return nn_engine_begin(engine);
}

static int
nn_arch_forwardPassFairCGAN(nn_arch_t* self, uint32_t bs,
                            nn_tensor_t* Cg1,
                            nn_tensor_t* Cr1,
                            nn_tensor_t* Yg,
                            nn_tensor_t* Ytr)
{
	ASSERT(self);
	ASSERT(Cg1);
	ASSERT(Cr1);
	ASSERT(Yg);
	ASSERT(Ytr);

	nn_engine_t* engine = self->engine;
	nn_dim_t*    dim    = nn_tensor_dim(Cg1);
	uint32_t     bs2    = bs/2;

	// sb00: state
	vkk_uniformAttachment_t ua0_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->sb_state,
		},
	};

	// sb10:  dimXd
	// sb11:  Xd
	// sb12:  dimCg
	// sb13:  Cg
	// sb14:  dimCr
	// sb15:  Cr
	// sb16:  dimYg
	// sb17:  Yg
	// sb18:  dimYtr
	// sb19:  Ytr
	vkk_uniformAttachment_t ua1_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->Xd->sb_dim,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->Xd->sb_data,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Cg1->sb_dim,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Cg1->sb_data,
		},
		{
			.binding = 4,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Cr1->sb_dim,
		},
		{
			.binding = 5,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Cr1->sb_data,
		},
		{
			.binding = 6,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Yg->sb_dim,
		},
		{
			.binding = 7,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Yg->sb_data,
		},
		{
			.binding = 8,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Ytr->sb_dim,
		},
		{
			.binding = 9,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Ytr->sb_data,
		},
	};

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1,
	};

	// nn_arch_forwardPassFairCGAN
	// Xd=(Ytr|Cr,Yg|Cg)
	// dispatch(RAW, bs/2, xh, xw, 1, 8, 8)
	vkk_computePipeline_t* cp;
	cp = engine->cp_arch_forwardPassFairCGAN;
	if(nn_engine_bind(engine, cp) == 0)
	{
		return 0;
	}
	vkk_compute_updateUniformSetRefs(engine->compute, self->us0,
	                                 1, ua0_array);
	vkk_compute_updateUniformSetRefs(engine->compute, self->us1,
	                                 10, ua1_array);
	vkk_compute_bindUniformSets(engine->compute, 2, us_array);
	nn_engine_dispatch(engine, VKK_HAZZARD_RAW,
	                   bs2, dim->height, dim->width,
	                   1, 8, 8);

	return 1;
}

static nn_tensor_t*
nn_arch_backpropFairCGAN(nn_arch_t* self, uint32_t bs,
                         nn_tensor_t* dL_dYg,
                         nn_tensor_t* dL_dYd)
{
	ASSERT(self);
	ASSERT(dL_dYg);
	ASSERT(dL_dYd);

	nn_engine_t* engine = self->engine;
	nn_tensor_t* dL_dYb = self->dL_dYb;
	nn_dim_t*    dim    = nn_tensor_dim(dL_dYg);
	uint32_t     bs2    = bs/2;

	// sb20: dim_dL_dYg
	// sb21: dL_dYg
	// sb22: dim_dL_dYd
	// sb23: dL_dYd
	// sb24: dim_dL_dYb
	// sb25: dL_dYb
	vkk_uniformAttachment_t ua2_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dYg->sb_dim,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dYg->sb_data,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dYd->sb_dim,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dYd->sb_data,
		},
		{
			.binding = 4,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dYb->sb_dim,
		},
		{
			.binding = 5,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dYb->sb_data,
		},
	};

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1,
		self->us2,
	};

	// nn_arch_backpropFairCGAN
	// dL_dYb = blend(dL_dYg + filter(dL_dYd))
	// dispatch(RAW, bs/2, xh, xw, 1, 8, 8)
	vkk_computePipeline_t* cp;
	cp = engine->cp_arch_backpropFairCGAN;
	if(nn_engine_bind(engine, cp) == 0)
	{
		return NULL;
	}
	vkk_compute_updateUniformSetRefs(engine->compute, self->us2,
	                                 6, ua2_array);
	vkk_compute_bindUniformSets(engine->compute, 3, us_array);
	nn_engine_dispatch(engine, VKK_HAZZARD_RAW,
	                   bs2, dim->height, dim->width,
	                   1, 8, 8);

	return dL_dYb;
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_arch_t* nn_arch_new(nn_engine_t* engine,
                       size_t base_size,
                       nn_archState_t* state)
{
	ASSERT(engine);
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

	self->engine = engine;

	memcpy(&self->state, state, sizeof(nn_archState_t));

	vkk_updateMode_e um;
	um = vkk_compute_updateMode(engine->compute);

	self->sb_state = vkk_buffer_new(engine->engine, um,
	                                VKK_BUFFER_USAGE_STORAGE,
	                                sizeof(nn_archState_t),
	                                NULL);
	if(self->sb_state == NULL)
	{
		goto fail_sb_state;
	}

	self->layers = cc_list_new();
	if(self->layers == NULL)
	{
		goto fail_layers;
	}

	self->us0 = vkk_uniformSet_new(engine->engine, 0, 0, NULL,
	                               engine->usf0_arch);
	if(self->us0 == NULL)
	{
		goto fail_us0;
	}

	self->us1 = vkk_uniformSet_new(engine->engine, 1, 0, NULL,
	                               engine->usf1_arch);
	if(self->us1 == NULL)
	{
		goto fail_us1;
	}

	self->us2 = vkk_uniformSet_new(engine->engine, 2, 0, NULL,
	                               engine->usf2_arch);
	if(self->us2 == NULL)
	{
		goto fail_us2;
	}

	// success
	return self;

	// failure
	fail_us2:
		vkk_uniformSet_delete(&self->us1);
	fail_us1:
		vkk_uniformSet_delete(&self->us0);
	fail_us0:
		cc_list_delete(&self->layers);
	fail_layers:
		vkk_buffer_delete(&self->sb_state);
	fail_sb_state:
		FREE(self);
	return NULL;
}

nn_arch_t*
nn_arch_import(nn_engine_t* engine,
               size_t base_size, jsmn_val_t* val)
{
	ASSERT(engine);
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	// bs not required
	jsmn_val_t* val_adam_alpha       = NULL;
	jsmn_val_t* val_adam_beta1       = NULL;
	jsmn_val_t* val_adam_beta2       = NULL;
	jsmn_val_t* val_adam_beta1t      = NULL;
	jsmn_val_t* val_adam_beta2t      = NULL;
	jsmn_val_t* val_adam_lambda      = NULL;
	jsmn_val_t* val_adam_nu          = NULL;
	jsmn_val_t* val_bn_momentum      = NULL;
	jsmn_val_t* val_gan_blend_factor = NULL;
	jsmn_val_t* val_gan_blend_scalar = NULL;
	jsmn_val_t* val_gan_blend_min    = NULL;
	jsmn_val_t* val_gan_blend_max    = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		jsmn_keyval_t* kv;
		kv = (jsmn_keyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == JSMN_TYPE_PRIMITIVE)
		{
			if(strcmp(kv->key, "adam_alpha") == 0)
			{
				val_adam_alpha = kv->val;
			}
			else if(strcmp(kv->key, "adam_beta1") == 0)
			{
				val_adam_beta1 = kv->val;
			}
			else if(strcmp(kv->key, "adam_beta2") == 0)
			{
				val_adam_beta2 = kv->val;
			}
			else if(strcmp(kv->key, "adam_beta1t") == 0)
			{
				val_adam_beta1t = kv->val;
			}
			else if(strcmp(kv->key, "adam_beta2t") == 0)
			{
				val_adam_beta2t = kv->val;
			}
			else if(strcmp(kv->key, "adam_lambda") == 0)
			{
				val_adam_lambda = kv->val;
			}
			else if(strcmp(kv->key, "adam_nu") == 0)
			{
				val_adam_nu = kv->val;
			}
			else if(strcmp(kv->key, "bn_momentum") == 0)
			{
				val_bn_momentum = kv->val;
			}
			else if(strcmp(kv->key, "gan_blend_factor") == 0)
			{
				val_gan_blend_factor = kv->val;
			}
			else if(strcmp(kv->key, "gan_blend_scalar") == 0)
			{
				val_gan_blend_scalar = kv->val;
			}
			else if(strcmp(kv->key, "gan_blend_min") == 0)
			{
				val_gan_blend_min = kv->val;
			}
			else if(strcmp(kv->key, "gan_blend_max") == 0)
			{
				val_gan_blend_max = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_adam_alpha       == NULL) ||
	   (val_adam_beta1       == NULL) ||
	   (val_adam_beta2       == NULL) ||
	   (val_adam_beta1t      == NULL) ||
	   (val_adam_beta2t      == NULL) ||
	   (val_adam_lambda      == NULL) ||
	   (val_adam_nu          == NULL) ||
	   (val_bn_momentum      == NULL) ||
	   (val_gan_blend_factor == NULL) ||
	   (val_gan_blend_scalar == NULL) ||
	   (val_gan_blend_min    == NULL) ||
	   (val_gan_blend_max    == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_archState_t state =
	{
		.adam_alpha       = strtof(val_adam_alpha->data,       NULL),
		.adam_beta1       = strtof(val_adam_beta1->data,       NULL),
		.adam_beta2       = strtof(val_adam_beta2->data,       NULL),
		.adam_beta1t      = strtof(val_adam_beta1t->data,      NULL),
		.adam_beta2t      = strtof(val_adam_beta2t->data,      NULL),
		.adam_lambda      = strtof(val_adam_lambda->data,      NULL),
		.adam_nu          = strtof(val_adam_nu->data,          NULL),
		.bn_momentum      = strtof(val_bn_momentum->data,      NULL),
		.gan_blend_factor = strtof(val_gan_blend_factor->data, NULL),
		.gan_blend_scalar = strtof(val_gan_blend_scalar->data, NULL),
		.gan_blend_min    = strtof(val_gan_blend_min->data,    NULL),
		.gan_blend_max    = strtof(val_gan_blend_max->data,    NULL),
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
	ret &= jsmn_stream_key(stream, "%s", "adam_alpha");
	ret &= jsmn_stream_float(stream, state->adam_alpha);
	ret &= jsmn_stream_key(stream, "%s", "adam_beta1");
	ret &= jsmn_stream_float(stream, state->adam_beta1);
	ret &= jsmn_stream_key(stream, "%s", "adam_beta2");
	ret &= jsmn_stream_float(stream, state->adam_beta2);
	ret &= jsmn_stream_key(stream, "%s", "adam_beta1t");
	ret &= jsmn_stream_float(stream, state->adam_beta1t);
	ret &= jsmn_stream_key(stream, "%s", "adam_beta2t");
	ret &= jsmn_stream_float(stream, state->adam_beta2t);
	ret &= jsmn_stream_key(stream, "%s", "adam_lambda");
	ret &= jsmn_stream_float(stream, state->adam_lambda);
	ret &= jsmn_stream_key(stream, "%s", "adam_nu");
	ret &= jsmn_stream_float(stream, state->adam_nu);
	ret &= jsmn_stream_key(stream, "%s", "bn_momentum");
	ret &= jsmn_stream_float(stream, state->bn_momentum);
	ret &= jsmn_stream_key(stream, "%s", "gan_blend_factor");
	ret &= jsmn_stream_float(stream, state->gan_blend_factor);
	ret &= jsmn_stream_key(stream, "%s", "gan_blend_scalar");
	ret &= jsmn_stream_float(stream, state->gan_blend_scalar);
	ret &= jsmn_stream_key(stream, "%s", "gan_blend_min");
	ret &= jsmn_stream_float(stream, state->gan_blend_min);
	ret &= jsmn_stream_key(stream, "%s", "gan_blend_max");
	ret &= jsmn_stream_float(stream, state->gan_blend_max);
	ret &= jsmn_stream_end(stream);

	return ret;
}

void nn_arch_delete(nn_arch_t** _self)
{
	ASSERT(_self);

	nn_arch_t* self = *_self;
	if(self)
	{
		vkk_uniformSet_delete(&self->us2);
		vkk_uniformSet_delete(&self->us1);
		vkk_uniformSet_delete(&self->us0);
		nn_tensor_delete(&self->dL_dYb);
		nn_tensor_delete(&self->Ytr);
		nn_tensor_delete(&self->Ytg);
		nn_tensor_delete(&self->Cr1);
		nn_tensor_delete(&self->Cr0);
		nn_tensor_delete(&self->Cg1);
		nn_tensor_delete(&self->Cg0);
		nn_tensor_delete(&self->Xd);
		nn_tensor_delete(&self->Yt);
		nn_tensor_delete(&self->X);
		cc_list_discard(self->layers);
		cc_list_delete(&self->layers);
		vkk_buffer_delete(&self->sb_state);
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

nn_tensor_t*
nn_arch_train(nn_arch_t* self, int flags,
              uint32_t bs, nn_tensor_t* X,
              nn_tensor_t* Yt, nn_tensor_t* Y)
{
	// X and Y may be NULL
	ASSERT(self);
	ASSERT(flags & NN_LAYER_FLAG_BACKPROP);
	ASSERT(Yt);

	if(nn_arch_init(self, bs, X, Yt) == 0)
	{
		return NULL;
	}

	cc_listIter_t* iter;
	if(flags & NN_LAYER_FLAG_FORWARD_PASS)
	{
		ASSERT(X);

		// optionally replace X with compute tensor
		if(X->tensor_mode == NN_TENSOR_MODE_IO)
		{
			X = self->X;
		}

		// perform forward pass
		iter = cc_list_head(self->layers);
		while(iter)
		{
			nn_layer_t* layer;
			layer = (nn_layer_t*) cc_list_peekIter(iter);

			X = nn_layer_forwardPass(layer, flags, bs, X);
			if(X == NULL)
			{
				goto fail_forwardPass;
			}

			iter = cc_list_next(iter);
		}
		self->O = X;
	}
	else
	{
		ASSERT(self->O);

		// see NN_LAYER_FLAG_BACKPROP_NOP
		X = self->O;
	}

	// optionally replace Yt with compute tensor
	if(Yt->tensor_mode == NN_TENSOR_MODE_IO)
	{
		Yt = self->Yt;
	}

	// compute loss
	nn_tensor_t* dL_dY;
	dL_dY = nn_loss_loss(self->loss, bs, X, Yt);
	if(dL_dY == NULL)
	{
		goto fail_loss;
	}

	// perform backprop
	iter = cc_list_tail(self->layers);
	while(iter)
	{
		nn_layer_t* layer;
		layer = (nn_layer_t*) cc_list_peekIter(iter);

		dL_dY = nn_layer_backprop(layer, flags, bs, dL_dY);
		if(dL_dY == NULL)
		{
			goto fail_backprop;
		}

		iter = cc_list_prev(iter);
	}

	nn_engine_end(self->engine);
	nn_arch_post(self, flags);

	// optionally blit Y
	if(Y)
	{
		if(nn_tensor_blit(X, Y, bs, 0, 0) == 0)
		{
			return NULL;
		}
	}

	// success
	return dL_dY;

	// failure
	fail_backprop:
	fail_loss:
	fail_forwardPass:
		nn_engine_end(self->engine);
	return NULL;
}

nn_tensor_t*
nn_arch_trainLERP(nn_arch_t* self, nn_arch_t* lerp,
                  uint32_t bs,
                  nn_tensor_t* Xt, nn_tensor_t* Yt,
                  nn_tensor_t* X, nn_tensor_t* Y)
{
	// X and Y may be NULL
	ASSERT(self);
	ASSERT(lerp);
	ASSERT(Xt);
	ASSERT(Yt);

	int flags = NN_LAYER_FLAG_TRAIN;

	if(nn_arch_initLERP(self, lerp, bs, Xt, Yt) == 0)
	{
		return NULL;
	}

	// optionally replace Xt, Yt with compute tensors
	if(Xt->tensor_mode == NN_TENSOR_MODE_IO)
	{
		// X is Xt
		Xt = self->X;
	}

	if(Yt->tensor_mode == NN_TENSOR_MODE_IO)
	{
		Yt = self->Yt;
	}

	// perform the forward pass to compute Y1=NN1(Xt)
	nn_tensor_t*   Y1   = Xt;
	cc_listIter_t* iter = cc_list_head(self->layers);
	while(iter)
	{
		nn_layer_t* layer;
		layer = (nn_layer_t*) cc_list_peekIter(iter);

		Y1 = nn_layer_forwardPass(layer, flags, bs, Y1);
		if(Y1 == NULL)
		{
			goto fail_train;
		}

		iter = cc_list_next(iter);
	}

	// perform the forward pass to compute X2=NN2(Yt)
	nn_tensor_t* X2 = Yt;
	iter = cc_list_head(lerp->layers);
	while(iter)
	{
		nn_layer_t* layer;
		layer = (nn_layer_t*) cc_list_peekIter(iter);

		X2 = nn_layer_forwardPass(layer, flags, bs, X2);
		if(X2 == NULL)
		{
			goto fail_train;
		}

		iter = cc_list_next(iter);
	}

	// compute loss
	nn_tensor_t* dL_dY1;
	nn_tensor_t* dL_dX2;
	dL_dY1 = nn_loss_loss(self->loss, bs, Y1, Yt);
	dL_dX2 = nn_loss_loss(lerp->loss, bs, X2, Xt);
	if((dL_dY1 == NULL) || (dL_dX2 == NULL))
	{
		goto fail_train;
	}

	// perform backprop (dL_dY1)
	iter = cc_list_tail(self->layers);
	while(iter)
	{
		nn_layer_t* layer;
		layer = (nn_layer_t*) cc_list_peekIter(iter);

		dL_dY1 = nn_layer_backprop(layer, flags, bs, dL_dY1);
		if(dL_dY1 == NULL)
		{
			goto fail_train;
		}

		iter = cc_list_prev(iter);
	}

	// perform backprop (dL_dX2)
	iter = cc_list_tail(lerp->layers);
	while(iter)
	{
		nn_layer_t* layer;
		layer = (nn_layer_t*) cc_list_peekIter(iter);

		dL_dX2 = nn_layer_backprop(layer, flags, bs, dL_dX2);
		if(dL_dX2 == NULL)
		{
			goto fail_train;
		}

		iter = cc_list_prev(iter);
	}

	nn_engine_end(self->engine);
	nn_arch_post(self, flags);
	nn_arch_post(lerp, flags);

	// optionally blit Y
	if(Y)
	{
		if(nn_tensor_blit(Y1, Y, bs, 0, 0) == 0)
		{
			return NULL;
		}
	}

	// optionally blit X
	if(X)
	{
		if(nn_tensor_blit(X2, X, bs, 0, 0) == 0)
		{
			return NULL;
		}
	}

	// success
	return dL_dY1;

	// failure
	fail_train:
		nn_engine_end(self->engine);
	return NULL;
}

nn_tensor_t*
nn_arch_trainFairCGAN(nn_arch_t* G,
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
                      float* d_loss)
{
	// dL_dYb, dL_dYg, dL_dYd, Yg, Yd,
	// loss, g_loss and d_loss are optional outputs
	// Cg1 and Cr1 are optional
	ASSERT(G);
	ASSERT(D);
	ASSERT(Cg0);
	ASSERT(Cg0 != Cg1);
	ASSERT(Cr0);
	ASSERT(Cr0 != Cr1);
	ASSERT(Ytg);
	ASSERT(Ytr);
	ASSERT(Yt11);
	ASSERT(Yt10);

	int flags = NN_LAYER_FLAG_TRAIN;

	uint32_t bs2 = bs/2;

	if(nn_arch_initFairCGAN(G, bs, Cg0, Cg1, Cr0, Cr1,
	                        Ytg, Ytr) == 0)
	{
		return NULL;
	}

	// optionally replace Cg0, Cg1, Cr0, Cr1, Ytg and Ytr
	// with compute tensors
	if(Cg0->tensor_mode == NN_TENSOR_MODE_IO)
	{
		Cg0 = G->Cg0;
	}

	if(Cr0->tensor_mode == NN_TENSOR_MODE_IO)
	{
		Cr0 = G->Cr0;
	}

	if(Cg1 == NULL)
	{
		Cg1 = Cg0;
	}
	else if(Cg1->tensor_mode == NN_TENSOR_MODE_IO)
	{
		Cg1 = G->Cg1;
	}

	if(Cr1 == NULL)
	{
		Cr1 = Cr0;
	}
	else if(Cr1->tensor_mode == NN_TENSOR_MODE_IO)
	{
		Cr1 = G->Cr1;
	}

	if(Ytg->tensor_mode == NN_TENSOR_MODE_IO)
	{
		Ytg = G->Ytg;
	}

	if(Ytr->tensor_mode == NN_TENSOR_MODE_IO)
	{
		Ytr = G->Ytr;
	}

	// perform the forward pass to compute Yg=G(Cg)
	nn_tensor_t*   X    = Cg0;
	cc_listIter_t* iter = cc_list_head(G->layers);
	while(iter)
	{
		nn_layer_t* layer;
		layer = (nn_layer_t*) cc_list_peekIter(iter);

		X = nn_layer_forwardPass(layer, flags, bs2, X);
		if(X == NULL)
		{
			goto fail_train;
		}

		iter = cc_list_next(iter);
	}
	G->O = X;

	// compute loss
	// avoid duplicate naming of dL_dYg compute tensor
	nn_tensor_t* dL_dYg_ct;
	dL_dYg_ct = nn_loss_loss(G->loss, bs2, X, Ytg);
	if(dL_dYg_ct == NULL)
	{
		goto fail_train;
	}

	if(loss)
	{
		*loss = nn_arch_loss(G);
	}

	// compute Xd=(Ytr|Cr,Yg|Cg) where Yg=X
	if(nn_arch_forwardPassFairCGAN(G, bs, Cg1, Cr1,
	                               X, Ytr) == 0)
	{
		goto fail_train;
	}

	// finish generator forward pass
	nn_engine_end(G->engine);

	// train discriminator
	// avoid duplicate naming of dL_dYd compute tensor
	nn_tensor_t* dL_dYd_ct;
	dL_dYd_ct = nn_arch_train(D, flags, bs, G->Xd, Yt10, Yd);
	if(dL_dYd_ct == NULL)
	{
		goto fail_train;
	}

	if(d_loss)
	{
		*d_loss = nn_arch_loss(D);
	}

	// train generator
	dL_dYd_ct = nn_arch_train(D, NN_LAYER_FLAG_BACKPROP_NOP,
	                          bs, NULL, Yt11, NULL);
	if(dL_dYd_ct == NULL)
	{
		goto fail_train;
	}

	if(g_loss)
	{
		*g_loss = nn_arch_loss(D);
	}

	// resume generator backprop
	if(nn_engine_begin(G->engine) == 0)
	{
		// do not end
		return NULL;
	}

	nn_tensor_t* dL_dY;
	dL_dY = nn_arch_backpropFairCGAN(G, bs, dL_dYg_ct,
	                                 dL_dYd_ct);
	if(dL_dY == NULL)
	{
		goto fail_train;
	}

	// optionally blit dL_dYb
	// blit must be prior to backprop to avoid overwrite
	if(dL_dYb)
	{
		// finish nn_arch_backpropFairCGAN
		nn_engine_end(G->engine);

		if(nn_tensor_blit(G->dL_dYb, dL_dYb, bs2, 0, 0) == 0)
		{
			return NULL;
		}

		// resume generator backprop
		if(nn_engine_begin(G->engine) == 0)
		{
			// do not end
			return NULL;
		}
	}

	// perform backprop
	iter = cc_list_tail(G->layers);
	while(iter)
	{
		nn_layer_t* layer;
		layer = (nn_layer_t*) cc_list_peekIter(iter);

		dL_dY = nn_layer_backprop(layer, flags, bs2, dL_dY);
		if(dL_dY == NULL)
		{
			goto fail_train;
		}

		iter = cc_list_prev(iter);
	}

	nn_engine_end(G->engine);
	nn_arch_post(G, flags);

	// optionally blit dL_dYg
	if(dL_dYg)
	{
		if(nn_tensor_blit(dL_dYg_ct, dL_dYg, bs2, 0, 0) == 0)
		{
			return NULL;
		}
	}

	// optionally blit dL_dYd
	if(dL_dYd)
	{
		if(nn_tensor_blit(dL_dYd_ct, dL_dYd, bs, 0, 0) == 0)
		{
			return NULL;
		}
	}

	// optionally blit Yg
	if(Yg)
	{
		if(nn_tensor_blit(X, Yg, bs2, 0, 0) == 0)
		{
			return NULL;
		}
	}

	// success
	return dL_dY;

	// failure
	fail_train:
		nn_engine_end(G->engine);
	return NULL;
}

float nn_arch_loss(nn_arch_t* self)
{
	ASSERT(self);

	if(self->loss)
	{
		return self->loss->loss;
	}

	return 0.0f;
}

int nn_arch_predict(nn_arch_t* self,
                    uint32_t bs,
                    nn_tensor_t* X,
                    nn_tensor_t* Y)
{
	ASSERT(self);
	ASSERT(X);
	ASSERT(Y);

	if(nn_arch_init(self, bs, X, NULL) == 0)
	{
		return 0;
	}

	// replace X with compute tensor
	if(X->tensor_mode == NN_TENSOR_MODE_IO)
	{
		X = self->X;
	}

	cc_listIter_t* iter = cc_list_head(self->layers);
	while(iter)
	{
		nn_layer_t* layer;
		layer = (nn_layer_t*) cc_list_peekIter(iter);

		X = nn_layer_forwardPass(layer,
		                         NN_LAYER_FLAG_FORWARD_PASS,
		                         bs, X);
		if(X == NULL)
		{
			goto fail_forwardPass;
		}

		iter = cc_list_next(iter);
	}
	self->O = X;

	nn_engine_end(self->engine);
	nn_arch_post(self, NN_LAYER_FLAG_FORWARD_PASS);

	// success
	return nn_tensor_blit(X, Y, bs, 0, 0);

	// failure
	fail_forwardPass:
		nn_engine_end(self->engine);
	return 0;
}
