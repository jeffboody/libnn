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
#include "../libcc/cc_log.h"
#include "../libcc/cc_memory.h"
#include "nn_batchNormLayer.h"
#include "nn_coderLayer.h"
#include "nn_convLayer.h"
#include "nn_dim.h"
#include "nn_factLayer.h"
#include "nn_skipLayer.h"
#include "nn_tensor.h"

/***********************************************************
* private - nn_coderLayer_t                                *
***********************************************************/

static nn_tensor_t*
nn_coderLayer_computeFpFn(nn_layer_t* base,
                          int flags, uint32_t bs,
                          nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_coderLayer_t* self = (nn_coderLayer_t*) base;

	if(self->conv)
	{
		X = nn_layer_computeFp(&self->conv->base,
		                       flags, bs, X);
		if(X == NULL)
		{
			return NULL;
		}
	}

	if(self->skip &&
	   ((self->skip->skip_mode == NN_SKIP_MODE_FORK_ADD) ||
	    (self->skip->skip_mode == NN_SKIP_MODE_ADD)))
	{
		X = nn_layer_computeFp(&self->skip->base,
		                       flags, bs, X);
		if(X == NULL)
		{
			return NULL;
		}
	}

	if(self->bn)
	{
		X = nn_layer_computeFp(&self->bn->base,
		                       flags, bs, X);
		if(X == NULL)
		{
			return NULL;
		}
	}

	if(self->fact)
	{
		X = nn_layer_computeFp(&self->fact->base,
		                       flags, bs, X);
		if(X == NULL)
		{
			return NULL;
		}
	}

	if(self->skip &&
	   ((self->skip->skip_mode == NN_SKIP_MODE_FORK_CAT) ||
	    (self->skip->skip_mode == NN_SKIP_MODE_CAT)))
	{
		X = nn_layer_computeFp(&self->skip->base,
		                       flags, bs, X);
		if(X == NULL)
		{
			return NULL;
		}
	}

	return X;
}

static nn_tensor_t*
nn_coderLayer_computeBpFn(nn_layer_t* base,
                          int flags, uint32_t bs,
                          nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,xh,xw,xd)

	nn_coderLayer_t* self = (nn_coderLayer_t*) base;

	if(self->skip &&
	   ((self->skip->skip_mode == NN_SKIP_MODE_FORK_CAT) ||
	    (self->skip->skip_mode == NN_SKIP_MODE_CAT)))
	{
		dL_dY = nn_layer_computeBp(&self->skip->base,
		                           flags, bs, dL_dY);
		if(dL_dY == NULL)
		{
			return NULL;
		}
	}

	if(self->fact)
	{
		dL_dY = nn_layer_computeBp(&self->fact->base,
		                           flags, bs, dL_dY);
		if(dL_dY == NULL)
		{
			return NULL;
		}
	}

	if(self->bn)
	{
		dL_dY = nn_layer_computeBp(&self->bn->base,
		                           flags, bs, dL_dY);
		if(dL_dY == NULL)
		{
			return NULL;
		}
	}

	if(self->skip &&
	   ((self->skip->skip_mode == NN_SKIP_MODE_FORK_ADD) ||
	    (self->skip->skip_mode == NN_SKIP_MODE_ADD)))
	{
		dL_dY = nn_layer_computeBp(&self->skip->base,
		                           flags, bs, dL_dY);
		if(dL_dY == NULL)
		{
			return NULL;
		}
	}

	if(self->conv)
	{
		dL_dY = nn_layer_computeBp(&self->conv->base,
		                           flags, bs, dL_dY);
		if(dL_dY == NULL)
		{
			return NULL;
		}
	}

	return dL_dY;
}

static void
nn_coderLayer_postFn(nn_layer_t* base,
                     int flags, uint32_t bs)
{
	ASSERT(base);

	nn_coderLayer_t* self = (nn_coderLayer_t*) base;

	if(self->conv)
	{
		nn_layer_post(&self->conv->base, flags, bs);
	}

	if(self->skip)
	{
		nn_layer_post(&self->skip->base, flags, bs);
	}

	if(self->bn)
	{
		nn_layer_post(&self->bn->base, flags, bs);
	}

	if(self->fact)
	{
		nn_layer_post(&self->fact->base, flags, bs);
	}
}

static nn_dim_t*
nn_coderLayer_dimXFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_coderLayer_t* self = (nn_coderLayer_t*) base;

	return &self->dimX;
}

static nn_dim_t*
nn_coderLayer_dimYFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_coderLayer_t* self = (nn_coderLayer_t*) base;

	return &self->dimY;
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_coderLayer_t*
nn_coderLayer_new(nn_coderLayerInfo_t* info)
{
	ASSERT(info);

	nn_layerInfo_t layer_info =
	{
		.arch          = info->arch,
		.compute_fp_fn = nn_coderLayer_computeFpFn,
		.compute_bp_fn = nn_coderLayer_computeBpFn,
		.post_fn       = nn_coderLayer_postFn,
		.dimX_fn       = nn_coderLayer_dimXFn,
		.dimY_fn       = nn_coderLayer_dimYFn,
	};

	nn_coderLayer_t* self;
	self = (nn_coderLayer_t*)
	       nn_layer_new(sizeof(nn_coderLayer_t),
	                    &layer_info);
	if(self == NULL)
	{
		return NULL;
	}

	nn_dim_t* dim = info->dimX;
	nn_dim_copy(dim, &self->dimX);

	if(info->conv_size)
	{
		uint32_t xd = dim->depth;

		nn_dim_t dimW =
		{
			.count  = info->fc,
			.width  = info->conv_size,
			.height = info->conv_size,
			.depth  = xd,
		};

		int flags = NN_CONV_LAYER_FLAG_XAVIER;
		if((info->fact_fn == NN_FACT_LAYER_FN_RELU)  ||
		   (info->fact_fn == NN_FACT_LAYER_FN_PRELU) ||
		   (info->fact_fn == NN_FACT_LAYER_FN_LRELU))
		{
			flags = NN_CONV_LAYER_FLAG_HE;
		}
		if(info->bn_mode  &&
		   (info->skip_mode != NN_CODER_SKIP_MODE_FORK_ADD) &&
		   (info->skip_mode != NN_CODER_SKIP_MODE_ADD))
		{
			flags |= NN_CONV_LAYER_FLAG_DISABLE_BIAS;
		}
		flags |= info->conv_flags;

		self->conv = nn_convLayer_new(info->arch, dim, &dimW,
		                              info->conv_stride, flags);
		if(self->conv == NULL)
		{
			goto fail_conv;
		}
		dim = nn_layer_dimY(&self->conv->base);
	}

	if(info->skip_mode == NN_CODER_SKIP_MODE_ADD)
	{
		ASSERT(info->skip_coder);
		self->skip = nn_skipLayer_newAdd(info->arch, dim,
		                                 info->skip_coder->skip,
		                                 info->skip_beta);
		if(self->skip == NULL)
		{
			LOGE("invalid");
			goto fail_skip_add;
		}
		dim = nn_layer_dimY(&self->skip->base);
	}
	else if(info->skip_mode == NN_CODER_SKIP_MODE_FORK_ADD)
	{
		self->skip = nn_skipLayer_newFork(info->arch, dim,
		                                  (nn_skipMode_e)
		                                  info->skip_mode);
		if(self->skip == NULL)
		{
			LOGE("invalid");
			goto fail_skip_add;
		}
		dim = nn_layer_dimY(&self->skip->base);
	}

	if(info->bn_mode)
	{
		self->bn = nn_batchNormLayer_new(info->arch, dim);
		if(self->bn == NULL)
		{
			goto fail_bn;
		}
	}

	if(info->fact_fn > NN_FACT_LAYER_FN_LINEAR)
	{
		self->fact = nn_factLayer_new(info->arch, dim,
		                              info->fact_fn);
		if(self->fact == NULL)
		{
			goto fail_fact;
		}
	}

	if(info->skip_mode == NN_CODER_SKIP_MODE_CAT)
	{
		ASSERT(info->skip_coder);
		self->skip = nn_skipLayer_newCat(info->arch, dim,
		                                 info->skip_coder->skip);
		if(self->skip == NULL)
		{
			LOGE("invalid");
			goto fail_skip_cat;
		}
		dim = nn_layer_dimY(&self->skip->base);
	}
	else if(info->skip_mode == NN_CODER_SKIP_MODE_FORK_CAT)
	{
		self->skip = nn_skipLayer_newFork(info->arch, dim,
		                                  (nn_skipMode_e)
		                                  info->skip_mode);
		if(self->skip == NULL)
		{
			LOGE("invalid");
			goto fail_skip_cat;
		}
		dim = nn_layer_dimY(&self->skip->base);
	}

	nn_dim_copy(dim, &self->dimY);

	// success
	return self;

	// failure
	fail_skip_cat:
		nn_factLayer_delete(&self->fact);
	fail_fact:
		nn_batchNormLayer_delete(&self->bn);
	fail_bn:
		nn_skipLayer_delete(&self->skip);
	fail_skip_add:
		nn_convLayer_delete(&self->conv);
	fail_conv:
		nn_layer_delete((nn_layer_t**) &self);
	return NULL;
}

void nn_coderLayer_delete(nn_coderLayer_t** _self)
{
	ASSERT(_self);

	nn_coderLayer_t* self = *_self;
	if(self)
	{
		nn_factLayer_delete(&self->fact);
		nn_batchNormLayer_delete(&self->bn);
		nn_skipLayer_delete(&self->skip);
		nn_convLayer_delete(&self->conv);
		nn_layer_delete((nn_layer_t**) _self);
	}
}

nn_coderLayer_t*
nn_coderLayer_import(nn_arch_t* arch, cc_jsmnVal_t* val,
                     nn_coderLayer_t* skip_coder)
{
	// skip_coder is optional
	ASSERT(arch);
	ASSERT(val);

	if(val->type != CC_JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	cc_jsmnVal_t* val_dimX = NULL;
	cc_jsmnVal_t* val_dimY = NULL;
	cc_jsmnVal_t* val_conv = NULL;
	cc_jsmnVal_t* val_skip = NULL;
	cc_jsmnVal_t* val_bn   = NULL;
	cc_jsmnVal_t* val_fact = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		cc_jsmnKeyval_t* kv;
		kv = (cc_jsmnKeyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == CC_JSMN_TYPE_OBJECT)
		{
			if(strcmp(kv->key, "dimX") == 0)
			{
				val_dimX = kv->val;
			}
			else if(strcmp(kv->key, "dimY") == 0)
			{
				val_dimY = kv->val;
			}
			else if(strcmp(kv->key, "conv") == 0)
			{
				val_conv = kv->val;
			}
			else if(strcmp(kv->key, "skip") == 0)
			{
				val_skip = kv->val;
			}
			else if(strcmp(kv->key, "bn") == 0)
			{
				val_bn = kv->val;
			}
			else if(strcmp(kv->key, "fact") == 0)
			{
				val_fact = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	// layers are optional
	if((val_dimX == NULL) || (val_dimY == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_layerInfo_t layer_info =
	{
		.arch          = arch,
		.compute_fp_fn = nn_coderLayer_computeFpFn,
		.compute_bp_fn = nn_coderLayer_computeBpFn,
		.post_fn       = nn_coderLayer_postFn,
		.dimX_fn       = nn_coderLayer_dimXFn,
		.dimY_fn       = nn_coderLayer_dimYFn,
	};

	nn_coderLayer_t*  self;
	self = (nn_coderLayer_t*)
	       nn_layer_new(sizeof(nn_coderLayer_t),
	                    &layer_info);
	if(self == NULL)
	{
		return NULL;
	}

	if(nn_dim_import(&self->dimX, val_dimX) == 0)
	{
		goto fail_dimX;
	}

	if(nn_dim_import(&self->dimY, val_dimY) == 0)
	{
		goto fail_dimY;
	}

	if(val_conv)
	{
		self->conv = nn_convLayer_import(arch, val_conv);
		if(self->conv == NULL)
		{
			goto fail_conv;
		}
	}

	if(val_skip)
	{
		nn_skipLayer_t* skip_fork = NULL;
		if(skip_coder)
		{
			skip_fork = skip_coder->skip;
		}

		self->skip = nn_skipLayer_import(arch, val_skip,
		                                 skip_fork);
		if(self->skip == NULL)
		{
			goto fail_skip;
		}
	}

	if(val_bn)
	{
		self->bn = nn_batchNormLayer_import(arch, val_bn);
		if(self->bn == NULL)
		{
			goto fail_bn;
		}
	}

	if(val_fact)
	{
		self->fact = nn_factLayer_import(arch, val_fact);
		if(self->fact == NULL)
		{
			goto fail_fact;
		}
	}

	// success
	return self;

	// failure
	fail_fact:
		nn_batchNormLayer_delete(&self->bn);
	fail_bn:
		nn_skipLayer_delete(&self->skip);
	fail_skip:
		nn_convLayer_delete(&self->conv);
	fail_conv:
	fail_dimY:
	fail_dimX:
		nn_layer_delete((nn_layer_t**) &self);
	return NULL;
}

int nn_coderLayer_export(nn_coderLayer_t* self,
                         cc_jsmnStream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	int ret = 1;
	ret &= cc_jsmnStream_beginObject(stream);
	ret &= cc_jsmnStream_key(stream, "%s", "dimX");
	ret &= nn_dim_export(&self->dimX, stream);
	ret &= cc_jsmnStream_key(stream, "%s", "dimY");
	ret &= nn_dim_export(&self->dimY, stream);

	if(self->conv)
	{
		ret &= cc_jsmnStream_key(stream, "%s", "conv");
		ret &= nn_convLayer_export(self->conv, stream);
	}

	if(self->skip)
	{
		ret &= cc_jsmnStream_key(stream, "%s", "skip");
		ret &= nn_skipLayer_export(self->skip, stream);
	}

	if(self->bn)
	{
		ret &= cc_jsmnStream_key(stream, "%s", "bn");
		ret &= nn_batchNormLayer_export(self->bn, stream);
	}

	if(self->fact)
	{
		ret &= cc_jsmnStream_key(stream, "%s", "fact");
		ret &= nn_factLayer_export(self->fact, stream);
	}
	ret &= cc_jsmnStream_end(stream);

	return ret;
}
