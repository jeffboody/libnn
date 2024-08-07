/*
 * Copyright (c) 2024 Jeff Boody
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
#include "../libcc/cc_log.h"
#include "../libcc/cc_memory.h"
#include "nn_batchNormLayer.h"
#include "nn_encdecLayer.h"
#include "nn_convLayer.h"
#include "nn_dim.h"
#include "nn_factLayer.h"
#include "nn_lanczosLayer.h"
#include "nn_skipLayer.h"
#include "nn_tensor.h"

/***********************************************************
* private - nn_encdecLayer_t                               *
***********************************************************/

static nn_tensor_t*
nn_encdecLayer_computeFpFn(nn_layer_t* base,
                           int flags, uint32_t bs,
                           nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_encdecLayer_t* self = (nn_encdecLayer_t*) base;

	X = nn_layer_computeFp(&self->enc0->base,
	                       flags, bs, X);
	if(X == NULL)
	{
		return NULL;
	}

	X = nn_layer_computeFp(self->down1.base,
	                       flags, bs, X);
	if(X == NULL)
	{
		return NULL;
	}

	X = nn_layer_computeFp(&self->enc1->base,
	                       flags, bs, X);
	if(X == NULL)
	{
		return NULL;
	}

	X = nn_layer_computeFp(self->down2.base,
	                       flags, bs, X);
	if(X == NULL)
	{
		return NULL;
	}

	X = nn_layer_computeFp(&self->node20->base,
	                       flags, bs, X);
	if(X == NULL)
	{
		return NULL;
	}

	X = nn_layer_computeFp(&self->node21->base,
	                       flags, bs, X);
	if(X == NULL)
	{
		return NULL;
	}

	X = nn_layer_computeFp(&self->node22->base,
	                       flags, bs, X);
	if(X == NULL)
	{
		return NULL;
	}

	X = nn_layer_computeFp(&self->node23->base,
	                       flags, bs, X);
	if(X == NULL)
	{
		return NULL;
	}

	X = nn_layer_computeFp(self->up1.base,
	                       flags, bs, X);
	if(X == NULL)
	{
		return NULL;
	}

	X = nn_layer_computeFp(&self->dec1->base,
	                       flags, bs, X);
	if(X == NULL)
	{
		return NULL;
	}

	X = nn_layer_computeFp(self->up0.base,
	                       flags, bs, X);
	if(X == NULL)
	{
		return NULL;
	}

	X = nn_layer_computeFp(&self->dec0->base,
	                       flags, bs, X);
	if(X == NULL)
	{
		return NULL;
	}

	return X;
}

static nn_tensor_t*
nn_encdecLayer_computeBpFn(nn_layer_t* base,
                           int flags, uint32_t bs,
                           nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,xh,xw,xd)

	nn_encdecLayer_t* self = (nn_encdecLayer_t*) base;

	dL_dY = nn_layer_computeBp(&self->dec0->base,
	                           flags, bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	dL_dY = nn_layer_computeBp(self->up0.base,
	                           flags, bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	dL_dY = nn_layer_computeBp(&self->dec1->base,
	                           flags, bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	dL_dY = nn_layer_computeBp(self->up1.base,
	                           flags, bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	dL_dY = nn_layer_computeBp(&self->node23->base,
	                           flags, bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	dL_dY = nn_layer_computeBp(&self->node22->base,
	                           flags, bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	dL_dY = nn_layer_computeBp(&self->node21->base,
	                           flags, bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	dL_dY = nn_layer_computeBp(&self->node20->base,
	                           flags, bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	dL_dY = nn_layer_computeBp(self->down2.base,
	                           flags, bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	dL_dY = nn_layer_computeBp(&self->enc1->base,
	                           flags, bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	dL_dY = nn_layer_computeBp(self->down1.base,
	                           flags, bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	dL_dY = nn_layer_computeBp(&self->enc0->base,
	                           flags, bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	return dL_dY;
}

static void
nn_encdecLayer_postFn(nn_layer_t* base,
                      int flags, uint32_t bs)
{
	ASSERT(base);

	nn_encdecLayer_t* self = (nn_encdecLayer_t*) base;

	nn_layer_post(&self->enc0->base,   flags, bs);
	nn_layer_post(self->down1.base,    flags, bs);
	nn_layer_post(&self->enc1->base,   flags, bs);
	nn_layer_post(self->down2.base,    flags, bs);
	nn_layer_post(&self->node20->base, flags, bs);
	nn_layer_post(&self->node21->base, flags, bs);
	nn_layer_post(&self->node22->base, flags, bs);
	nn_layer_post(&self->node23->base, flags, bs);
	nn_layer_post(self->up1.base,      flags, bs);
	nn_layer_post(&self->dec1->base,   flags, bs);
	nn_layer_post(self->up0.base,      flags, bs);
	nn_layer_post(&self->dec0->base,   flags, bs);
}

static nn_dim_t*
nn_encdecLayer_dimXFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_encdecLayer_t* self = (nn_encdecLayer_t*) base;

	return nn_layer_dimX(&self->enc0->base);
}

static nn_dim_t*
nn_encdecLayer_dimYFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_encdecLayer_t* self = (nn_encdecLayer_t*) base;

	return nn_layer_dimY(&self->dec0->base);
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_encdecLayer_t*
nn_encdecLayer_new(nn_encdecLayerInfo_t* info)
{
	ASSERT(info);

	nn_dim_t* dim;

	nn_layerInfo_t layer_info =
	{
		.arch          = info->arch,
		.compute_fp_fn = nn_encdecLayer_computeFpFn,
		.compute_bp_fn = nn_encdecLayer_computeBpFn,
		.post_fn       = nn_encdecLayer_postFn,
		.dimX_fn       = nn_encdecLayer_dimXFn,
		.dimY_fn       = nn_encdecLayer_dimYFn,
	};

	nn_encdecLayer_t* self;
	self = (nn_encdecLayer_t*)
	       nn_layer_new(sizeof(nn_encdecLayer_t),
	                    &layer_info);
	if(self == NULL)
	{
		return NULL;
	}

	self->sampler = info->sampler;

	nn_coderSkipMode_e enc_skip_mode;
	enc_skip_mode = NN_CODER_SKIP_MODE_NONE;
	if(info->skip_mode == NN_CODER_SKIP_MODE_ADD)
	{
		enc_skip_mode = NN_CODER_SKIP_MODE_FORK_ADD;
	}
	else if(info->skip_mode == NN_CODER_SKIP_MODE_CAT)
	{
		enc_skip_mode = NN_CODER_SKIP_MODE_FORK_CAT;
	}

	nn_coderLayerInfo_t enc0_info =
	{
		.arch        = info->arch,
		.dimX        = info->dimX,
		.fc          = info->fc,
		.conv_flags  = info->norm_flags0,
		.conv_size   = 3,
		.conv_stride = 1,

		// skip layer
		.skip_mode = enc_skip_mode,

		// bn layer
		.bn_mode = info->bn_mode0,

		// fact layer
		.fact_fn = info->fact_fn,
	};

	self->enc0 = nn_coderLayer_new(&enc0_info);
	if(self->enc0 == NULL)
	{
		goto failure;
	}
	dim = nn_layer_dimY(&self->enc0->base);

	nn_coderLayerInfo_t down_coder_info =
	{
		.arch        = info->arch,
		.dimX        = dim,
		.fc          = info->fc,
		.conv_flags  = info->norm_flags12,
		.conv_size   = 3,
		.conv_stride = 2,
		.bn_mode     = info->bn_mode12,
		.fact_fn     = info->fact_fn,
	};

	if(info->sampler == NN_ENCDEC_SAMPLER_CODER)
	{
		self->down1.coder = nn_coderLayer_new(&down_coder_info);
	}
	else if(info->sampler == NN_ENCDEC_SAMPLER_LANCZOS)
	{
		nn_dim_t dimY =
		{
			.count  = dim->count,
			.height = dim->height/2,
			.width  = dim->width/2,
			.depth  = dim->depth,
		};
		self->down1.lanczos = nn_lanczosLayer_new(info->arch,
		                                          dim,
		                                          &dimY,
		                                          info->a);
	}
	if(self->down1.base == NULL)
	{
		goto failure;
	}
	dim = nn_layer_dimY(self->down1.base);

	nn_coderLayerInfo_t enc1_info =
	{
		.arch        = info->arch,
		.dimX        = dim,
		.fc          = info->fc,
		.conv_flags  = info->norm_flags12,
		.conv_size   = 3,
		.conv_stride = 1,

		// skip layer
		.skip_mode = enc_skip_mode,

		// bn layer
		.bn_mode = info->bn_mode12,

		// fact layer
		.fact_fn = info->fact_fn,
	};

	self->enc1 = nn_coderLayer_new(&enc1_info);
	if(self->enc1 == NULL)
	{
		goto failure;
	}
	dim = nn_layer_dimY(&self->enc1->base);

	if(info->sampler == NN_ENCDEC_SAMPLER_CODER)
	{
		down_coder_info.dimX = dim;
		self->down2.coder    = nn_coderLayer_new(&down_coder_info);
	}
	else if(info->sampler == NN_ENCDEC_SAMPLER_LANCZOS)
	{
		nn_dim_t dimY =
		{
			.count  = dim->count,
			.height = dim->height/2,
			.width  = dim->width/2,
			.depth  = dim->depth,
		};
		self->down2.lanczos = nn_lanczosLayer_new(info->arch,
		                                          dim,
		                                          &dimY,
		                                          info->a);
	}
	if(self->down2.base == NULL)
	{
		goto failure;
	}
	dim = nn_layer_dimY(self->down2.base);

	nn_coderLayerInfo_t node2_info =
	{
		.arch        = info->arch,
		.dimX        = dim,
		.fc          = info->fc,
		.conv_flags  = info->norm_flags12,
		.conv_size   = 3,
		.conv_stride = 1,

		// bn layer
		.bn_mode = info->bn_mode12,

		// fact layer
		.fact_fn = info->fact_fn,
	};

	self->node20 = nn_coderLayer_new(&node2_info);
	if(self->node20 == NULL)
	{
		goto failure;
	}
	dim = nn_layer_dimY(&self->node20->base);

	self->node21 = nn_coderLayer_new(&node2_info);
	if(self->node21 == NULL)
	{
		goto failure;
	}
	dim = nn_layer_dimY(&self->node21->base);

	self->node22 = nn_coderLayer_new(&node2_info);
	if(self->node22 == NULL)
	{
		goto failure;
	}
	dim = nn_layer_dimY(&self->node22->base);

	self->node23 = nn_coderLayer_new(&node2_info);
	if(self->node23 == NULL)
	{
		goto failure;
	}
	dim = nn_layer_dimY(&self->node23->base);

	nn_coderLayerInfo_t up_coder_info =
	{
		.arch        = info->arch,
		.dimX        = dim,
		.fc          = info->fc,
		.conv_flags  = NN_CONV_LAYER_FLAG_TRANSPOSE |
		               info->norm_flags12,
		.conv_size   = 2,
		.conv_stride = 2,
		.bn_mode     = info->bn_mode12,
		.fact_fn     = info->fact_fn,
	};

	if(info->sampler == NN_ENCDEC_SAMPLER_CODER)
	{
		self->up1.coder = nn_coderLayer_new(&up_coder_info);
	}
	else if(info->sampler == NN_ENCDEC_SAMPLER_LANCZOS)
	{
		nn_dim_t dimY =
		{
			.count  = dim->count,
			.height = 2*dim->height,
			.width  = 2*dim->width,
			.depth  = dim->depth,
		};
		self->up1.lanczos = nn_lanczosLayer_new(info->arch,
		                                        dim,
		                                        &dimY,
		                                        info->a);
	}
	if(self->up1.base == NULL)
	{
		goto failure;
	}
	dim = nn_layer_dimY(self->up1.base);

	nn_coderLayerInfo_t dec1_info =
	{
		.arch        = info->arch,
		.dimX        = dim,
		.fc          = info->fc,
		.conv_flags  = info->norm_flags12,
		.conv_size   = 3,
		.conv_stride = 1,

		// skip layer
		.skip_mode  = info->skip_mode,
		.skip_coder = self->enc1,
		.skip_beta  = info->skip_beta,

		// bn layer
		.bn_mode = info->bn_mode12,

		// fact layer
		.fact_fn = info->fact_fn,
	};

	self->dec1 = nn_coderLayer_new(&dec1_info);
	if(self->dec1 == NULL)
	{
		goto failure;
	}
	dim = nn_layer_dimY(&self->dec1->base);

	if(info->sampler == NN_ENCDEC_SAMPLER_CODER)
	{
		// up0.coder uses norm_flags12 and bn_mode12
		up_coder_info.dimX = dim;
		self->up0.coder    = nn_coderLayer_new(&up_coder_info);
	}
	else if(info->sampler == NN_ENCDEC_SAMPLER_LANCZOS)
	{
		nn_dim_t dimY =
		{
			.count  = dim->count,
			.height = 2*dim->height,
			.width  = 2*dim->width,
			.depth  = dim->depth,
		};
		self->up0.lanczos = nn_lanczosLayer_new(info->arch,
		                                        dim,
		                                        &dimY,
		                                        info->a);
	}
	if(self->up0.base == NULL)
	{
		goto failure;
	}
	dim = nn_layer_dimY(self->up0.base);

	nn_coderLayerInfo_t dec0_info =
	{
		.arch        = info->arch,
		.dimX        = dim,
		.fc          = info->fc,
		.conv_flags  = info->norm_flags0,
		.conv_size   = 3,
		.conv_stride = 1,

		// skip layer
		.skip_mode  = info->skip_mode,
		.skip_coder = self->enc0,
		.skip_beta  = info->skip_beta,

		// bn layer
		.bn_mode = info->bn_mode0,

		// fact layer
		.fact_fn = info->fact_fn,
	};

	self->dec0 = nn_coderLayer_new(&dec0_info);
	if(self->dec0 == NULL)
	{
		goto failure;
	}
	dim = nn_layer_dimY(&self->dec0->base);

	// success
	return self;

	// failure
	failure:
		nn_encdecLayer_delete(&self);
	return NULL;
}

void nn_encdecLayer_delete(nn_encdecLayer_t** _self)
{
	ASSERT(_self);

	nn_encdecLayer_t* self = *_self;
	if(self)
	{
		if(self->sampler == NN_ENCDEC_SAMPLER_CODER)
		{
			nn_coderLayer_delete(&self->up0.coder);
			nn_coderLayer_delete(&self->up1.coder);
			nn_coderLayer_delete(&self->down2.coder);
			nn_coderLayer_delete(&self->down1.coder);
		}
		else if(self->sampler == NN_ENCDEC_SAMPLER_LANCZOS)
		{
			nn_lanczosLayer_delete(&self->up0.lanczos);
			nn_lanczosLayer_delete(&self->up1.lanczos);
			nn_lanczosLayer_delete(&self->down2.lanczos);
			nn_lanczosLayer_delete(&self->down1.lanczos);
		}

		nn_coderLayer_delete(&self->enc0);
		nn_coderLayer_delete(&self->enc1);
		nn_coderLayer_delete(&self->node20);
		nn_coderLayer_delete(&self->node21);
		nn_coderLayer_delete(&self->node22);
		nn_coderLayer_delete(&self->node23);
		nn_coderLayer_delete(&self->dec1);
		nn_coderLayer_delete(&self->dec0);
		nn_layer_delete((nn_layer_t**) _self);
	}
}

nn_encdecLayer_t*
nn_encdecLayer_import(nn_arch_t* arch, cc_jsmnVal_t* val)
{
	ASSERT(arch);
	ASSERT(val);

	if(val->type != CC_JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	cc_jsmnVal_t* val_sampler = NULL;
	cc_jsmnVal_t* val_enc0    = NULL;
	cc_jsmnVal_t* val_down1   = NULL;
	cc_jsmnVal_t* val_enc1    = NULL;
	cc_jsmnVal_t* val_down2   = NULL;
	cc_jsmnVal_t* val_node20  = NULL;
	cc_jsmnVal_t* val_node21  = NULL;
	cc_jsmnVal_t* val_node22  = NULL;
	cc_jsmnVal_t* val_node23  = NULL;
	cc_jsmnVal_t* val_up1     = NULL;
	cc_jsmnVal_t* val_dec1    = NULL;
	cc_jsmnVal_t* val_up0     = NULL;
	cc_jsmnVal_t* val_dec0    = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		cc_jsmnKeyval_t* kv;
		kv = (cc_jsmnKeyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == CC_JSMN_TYPE_OBJECT)
		{
			if(strcmp(kv->key, "enc0") == 0)
			{
				val_enc0 = kv->val;
			}
			else if(strcmp(kv->key, "down1") == 0)
			{
				val_down1 = kv->val;
			}
			else if(strcmp(kv->key, "enc1") == 0)
			{
				val_enc1 = kv->val;
			}
			else if(strcmp(kv->key, "down2") == 0)
			{
				val_down2 = kv->val;
			}
			else if(strcmp(kv->key, "node20") == 0)
			{
				val_node20 = kv->val;
			}
			else if(strcmp(kv->key, "node21") == 0)
			{
				val_node21 = kv->val;
			}
			else if(strcmp(kv->key, "node22") == 0)
			{
				val_node22 = kv->val;
			}
			else if(strcmp(kv->key, "node23") == 0)
			{
				val_node23 = kv->val;
			}
			else if(strcmp(kv->key, "up1") == 0)
			{
				val_up1 = kv->val;
			}
			else if(strcmp(kv->key, "dec1") == 0)
			{
				val_dec1 = kv->val;
			}
			else if(strcmp(kv->key, "up0") == 0)
			{
				val_up0 = kv->val;
			}
			else if(strcmp(kv->key, "dec0") == 0)
			{
				val_dec0 = kv->val;
			}
		}
		else if(kv->val->type == CC_JSMN_TYPE_STRING)
		{
			if(strcmp(kv->key, "sampler") == 0)
			{
				val_sampler = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_sampler == NULL) ||
	   (val_enc0    == NULL) || (val_down1   == NULL) ||
	   (val_enc1    == NULL) || (val_down2   == NULL) ||
	   (val_node20  == NULL) || (val_node21  == NULL) ||
	   (val_node22  == NULL) || (val_node23  == NULL) ||
	   (val_up1     == NULL) || (val_dec1    == NULL) ||
	   (val_up0     == NULL) || (val_dec0    == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_layerInfo_t layer_info =
	{
		.arch          = arch,
		.compute_fp_fn = nn_encdecLayer_computeFpFn,
		.compute_bp_fn = nn_encdecLayer_computeBpFn,
		.post_fn       = nn_encdecLayer_postFn,
		.dimX_fn       = nn_encdecLayer_dimXFn,
		.dimY_fn       = nn_encdecLayer_dimYFn,
	};

	nn_encdecLayer_t*  self;
	self = (nn_encdecLayer_t*)
	       nn_layer_new(sizeof(nn_encdecLayer_t),
	                    &layer_info);
	if(self == NULL)
	{
		return NULL;
	}

	// import sampler
	if(strcmp(val_sampler->data, "CODER") == 0)
	{
		self->sampler = NN_ENCDEC_SAMPLER_CODER;
	}
	else if(strcmp(val_sampler->data, "LANCZOS") == 0)
	{
		self->sampler = NN_ENCDEC_SAMPLER_LANCZOS;
	}
	else
	{
		LOGE("invalid sampler=%s", val_sampler->data);
		goto failure;
	}

	// import layers
	self->enc0 = nn_coderLayer_import(arch, val_enc0, NULL);
	if(self->enc0 == NULL)
	{
		goto failure;
	}

	nn_encdecSampler_t* down1 = &self->down1;
	if(self->sampler == NN_ENCDEC_SAMPLER_CODER)
	{
		down1->coder = nn_coderLayer_import(arch, val_down1,
		                                    NULL);
	}
	else if(self->sampler == NN_ENCDEC_SAMPLER_LANCZOS)
	{
		down1->lanczos = nn_lanczosLayer_import(arch,
		                                        val_down1);
	}
	if(down1->base == NULL)
	{
		goto failure;
	}

	self->enc1 = nn_coderLayer_import(arch, val_enc1, NULL);
	if(self->enc1 == NULL)
	{
		goto failure;
	}

	nn_encdecSampler_t* down2 = &self->down2;
	if(self->sampler == NN_ENCDEC_SAMPLER_CODER)
	{
		down2->coder = nn_coderLayer_import(arch, val_down2,
		                                    NULL);
	}
	else if(self->sampler == NN_ENCDEC_SAMPLER_LANCZOS)
	{
		down2->lanczos = nn_lanczosLayer_import(arch,
		                                        val_down2);
	}
	if(down2->base == NULL)
	{
		goto failure;
	}

	self->node20 = nn_coderLayer_import(arch, val_node20,
	                                    NULL);
	if(self->node20 == NULL)
	{
		goto failure;
	}

	self->node21 = nn_coderLayer_import(arch, val_node21,
	                                    NULL);
	if(self->node21 == NULL)
	{
		goto failure;
	}

	self->node22 = nn_coderLayer_import(arch, val_node22,
	                                    NULL);
	if(self->node22 == NULL)
	{
		goto failure;
	}

	self->node23 = nn_coderLayer_import(arch, val_node23,
	                                    NULL);
	if(self->node23 == NULL)
	{
		goto failure;
	}

	nn_encdecSampler_t* up1 = &self->up1;
	if(self->sampler == NN_ENCDEC_SAMPLER_CODER)
	{
		up1->coder = nn_coderLayer_import(arch, val_up1,
		                                  NULL);
	}
	else if(self->sampler == NN_ENCDEC_SAMPLER_LANCZOS)
	{
		up1->lanczos = nn_lanczosLayer_import(arch,
		                                      val_up1);
	}
	if(up1->base == NULL)
	{
		goto failure;
	}

	self->dec1 = nn_coderLayer_import(arch, val_dec1,
	                                  self->enc1);
	if(self->dec1 == NULL)
	{
		goto failure;
	}

	nn_encdecSampler_t* up0 = &self->up0;
	if(self->sampler == NN_ENCDEC_SAMPLER_CODER)
	{
		up0->coder = nn_coderLayer_import(arch, val_up0,
		                                  NULL);
	}
	else if(self->sampler == NN_ENCDEC_SAMPLER_LANCZOS)
	{
		up0->lanczos = nn_lanczosLayer_import(arch,
		                                      val_up0);
	}
	if(up0->base == NULL)
	{
		goto failure;
	}

	self->dec0 = nn_coderLayer_import(arch, val_dec0,
	                                  self->enc0);
	if(self->dec0 == NULL)
	{
		goto failure;
	}

	// success
	return self;

	// failure
	failure:
		nn_encdecLayer_delete(&self);
	return NULL;
}

int nn_encdecLayer_export(nn_encdecLayer_t* self,
                          cc_jsmnStream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	int ret = 1;
	ret &= cc_jsmnStream_beginObject(stream);

	// export sampler
	if(self->sampler == NN_ENCDEC_SAMPLER_CODER)
	{
		ret &= cc_jsmnStream_key(stream, "%s", "sampler");
		ret &= cc_jsmnStream_string(stream, "%s", "CODER");

		ret &= cc_jsmnStream_key(stream, "%s", "down1");
		ret &= nn_coderLayer_export(self->down1.coder, stream);
		ret &= cc_jsmnStream_key(stream, "%s", "down2");
		ret &= nn_coderLayer_export(self->down2.coder, stream);
		ret &= cc_jsmnStream_key(stream, "%s", "up1");
		ret &= nn_coderLayer_export(self->up1.coder, stream);
		ret &= cc_jsmnStream_key(stream, "%s", "up0");
		ret &= nn_coderLayer_export(self->up0.coder, stream);
	}
	else if(self->sampler == NN_ENCDEC_SAMPLER_LANCZOS)
	{
		ret &= cc_jsmnStream_key(stream, "%s", "sampler");
		ret &= cc_jsmnStream_string(stream, "%s", "LANCZOS");

		ret &= cc_jsmnStream_key(stream, "%s", "down1");
		ret &= nn_lanczosLayer_export(self->down1.lanczos, stream);
		ret &= cc_jsmnStream_key(stream, "%s", "down2");
		ret &= nn_lanczosLayer_export(self->down2.lanczos, stream);
		ret &= cc_jsmnStream_key(stream, "%s", "up1");
		ret &= nn_lanczosLayer_export(self->up1.lanczos, stream);
		ret &= cc_jsmnStream_key(stream, "%s", "up0");
		ret &= nn_lanczosLayer_export(self->up0.lanczos, stream);
	}

	// export encoder/decoder and nodes
	ret &= cc_jsmnStream_key(stream, "%s", "enc0");
	ret &= nn_coderLayer_export(self->enc0, stream);
	ret &= cc_jsmnStream_key(stream, "%s", "enc1");
	ret &= nn_coderLayer_export(self->enc1, stream);
	ret &= cc_jsmnStream_key(stream, "%s", "node20");
	ret &= nn_coderLayer_export(self->node20, stream);
	ret &= cc_jsmnStream_key(stream, "%s", "node21");
	ret &= nn_coderLayer_export(self->node21, stream);
	ret &= cc_jsmnStream_key(stream, "%s", "node22");
	ret &= nn_coderLayer_export(self->node22, stream);
	ret &= cc_jsmnStream_key(stream, "%s", "node23");
	ret &= nn_coderLayer_export(self->node23, stream);
	ret &= cc_jsmnStream_key(stream, "%s", "dec1");
	ret &= nn_coderLayer_export(self->dec1, stream);
	ret &= cc_jsmnStream_key(stream, "%s", "dec0");
	ret &= nn_coderLayer_export(self->dec0, stream);

	ret &= cc_jsmnStream_end(stream);

	return ret;
}
