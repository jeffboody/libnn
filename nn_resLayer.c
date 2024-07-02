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

#include <stdlib.h>
#include <string.h>

#define LOG_TAG "nn"
#include "../libcc/cc_log.h"
#include "../libcc/cc_memory.h"
#include "nn_batchNormLayer.h"
#include "nn_convLayer.h"
#include "nn_dim.h"
#include "nn_factLayer.h"
#include "nn_resLayer.h"
#include "nn_skipLayer.h"
#include "nn_tensor.h"

/***********************************************************
* private - nn_resLayer_t                                  *
***********************************************************/

static nn_tensor_t*
nn_resLayer_computeFpFn(nn_layer_t* base,
                        int flags, uint32_t bs,
                        nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_resLayer_t* self = (nn_resLayer_t*) base;

	X = nn_layer_computeFp(&self->skip1->base,
	                       flags, bs, X);
	if(X == NULL)
	{
		return NULL;
	}

	if(self->bn1)
	{
		X = nn_layer_computeFp(&self->bn1->base,
		                       flags, bs, X);
		if(X == NULL)
		{
			return NULL;
		}
	}

	X = nn_layer_computeFp(&self->fact1->base,
	                       flags, bs, X);
	if(X == NULL)
	{
		return NULL;
	}

	X = nn_layer_computeFp(&self->conv1->base,
	                       flags, bs, X);
	if(X == NULL)
	{
		return NULL;
	}

	if(self->bn2)
	{
		X = nn_layer_computeFp(&self->bn2->base,
		                       flags, bs, X);
		if(X == NULL)
		{
			return NULL;
		}
	}

	X = nn_layer_computeFp(&self->fact2->base,
	                       flags, bs, X);
	if(X == NULL)
	{
		return NULL;
	}

	X = nn_layer_computeFp(&self->conv2->base,
	                       flags, bs, X);
	if(X == NULL)
	{
		return NULL;
	}

	X = nn_layer_computeFp(&self->skip2->base,
	                       flags, bs, X);
	if(X == NULL)
	{
		return NULL;
	}

	return X;
}

static nn_tensor_t*
nn_resLayer_computeBpFn(nn_layer_t* base,
                        int flags, uint32_t bs,
                        nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,xh,xw,xd)

	nn_resLayer_t* self = (nn_resLayer_t*) base;

	dL_dY = nn_layer_computeBp(&self->skip2->base,
	                           flags, bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	dL_dY = nn_layer_computeBp(&self->conv2->base,
	                           flags, bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	dL_dY = nn_layer_computeBp(&self->fact2->base,
	                           flags, bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	if(self->bn2)
	{
		dL_dY = nn_layer_computeBp(&self->bn2->base,
		                           flags, bs, dL_dY);
		if(dL_dY == NULL)
		{
			return NULL;
		}
	}

	dL_dY = nn_layer_computeBp(&self->conv1->base,
	                           flags, bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	dL_dY = nn_layer_computeBp(&self->fact1->base,
	                           flags, bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	if(self->bn1)
	{
		dL_dY = nn_layer_computeBp(&self->bn1->base,
		                           flags, bs, dL_dY);
		if(dL_dY == NULL)
		{
			return NULL;
		}
	}

	dL_dY = nn_layer_computeBp(&self->skip1->base,
	                           flags, bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	return dL_dY;
}

static void
nn_resLayer_postFn(nn_layer_t* base,
                   int flags, uint32_t bs)
{
	ASSERT(base);

	nn_resLayer_t* self = (nn_resLayer_t*) base;

	nn_layer_post(&self->skip1->base, flags, bs);

	if(self->bn1)
	{
		nn_layer_post(&self->bn1->base, flags, bs);
	}

	nn_layer_post(&self->fact1->base, flags, bs);
	nn_layer_post(&self->conv1->base, flags, bs);

	if(self->bn2)
	{
		nn_layer_post(&self->bn2->base, flags, bs);
	}

	nn_layer_post(&self->fact2->base, flags, bs);
	nn_layer_post(&self->conv2->base, flags, bs);
	nn_layer_post(&self->skip2->base, flags, bs);
}

static nn_dim_t*
nn_resLayer_dimXFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_resLayer_t* self = (nn_resLayer_t*) base;

	return nn_layer_dimX(&self->skip1->base);
}

static nn_dim_t*
nn_resLayer_dimYFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_resLayer_t* self = (nn_resLayer_t*) base;

	return nn_layer_dimY(&self->skip2->base);
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_resLayer_t*
nn_resLayer_new(nn_arch_t* arch,
                nn_dim_t* dimX,
                float skip_beta,
                nn_resBatchNormMode_e bn_mode,
                nn_factLayerFn_e fact_fn,
                int norm_flags)
{
	ASSERT(arch);
	ASSERT(dimX);

	// X is the output Y of the previous layer
	// where Y is dim(bs,yh,yw,fc)
	uint32_t fc = dimX->depth;

	nn_layerInfo_t layer_info =
	{
		.arch          = arch,
		.compute_fp_fn = nn_resLayer_computeFpFn,
		.compute_bp_fn = nn_resLayer_computeBpFn,
		.post_fn       = nn_resLayer_postFn,
		.dimX_fn       = nn_resLayer_dimXFn,
		.dimY_fn       = nn_resLayer_dimYFn,
	};

	nn_resLayer_t* self;
	self = (nn_resLayer_t*)
	       nn_layer_new(sizeof(nn_resLayer_t),
	                    &layer_info);
	if(self == NULL)
	{
		return NULL;
	}

	self->skip1 = nn_skipLayer_newFork(arch, dimX,
	                                   NN_SKIP_MODE_FORK_ADD);
	if(self->skip1 == NULL)
	{
		goto failure;
	}

	if(bn_mode)
	{
		self->bn1 = nn_batchNormLayer_new(arch, dimX);
		if(self->bn1 == NULL)
		{
			goto failure;
		}
	}

	self->fact1 = nn_factLayer_new(arch, dimX, fact_fn);
	if(self->fact1 == NULL)
	{
		goto failure;
	}

	nn_dim_t dimW =
	{
		.count  = fc,
		.height = 3,
		.width  = 3,
		.depth  = fc,
	};

	int flags = NN_CONV_LAYER_FLAG_XAVIER;
	if((fact_fn == NN_FACT_LAYER_FN_RELU)  ||
	   (fact_fn == NN_FACT_LAYER_FN_PRELU) ||
	   (fact_fn == NN_FACT_LAYER_FN_LRELU))
	{
		flags = NN_CONV_LAYER_FLAG_HE;
	}
	flags |= norm_flags;

	self->conv1 = nn_convLayer_new(arch, dimX, &dimW,
	                               1, flags);
	if(self->conv1 == NULL)
	{
		goto failure;
	}

	if(bn_mode)
	{
		self->bn2 = nn_batchNormLayer_new(arch, dimX);
		if(self->bn2 == NULL)
		{
			goto failure;
		}
	}

	self->fact2 = nn_factLayer_new(arch, dimX, fact_fn);
	if(self->fact2 == NULL)
	{
		goto failure;
	}

	self->conv2 = nn_convLayer_new(arch, dimX, &dimW,
	                               1, flags);
	if(self->conv2 == NULL)
	{
		goto failure;
	}

	self->skip2 = nn_skipLayer_newAdd(arch, dimX,
	                                  self->skip1,
	                                  skip_beta);
	if(self->skip2 == NULL)
	{
		goto failure;
	}

	// success
	return self;

	// failure
	failure:
		nn_resLayer_delete(&self);
	return NULL;
}

void nn_resLayer_delete(nn_resLayer_t** _self)
{
	ASSERT(_self);

	nn_resLayer_t* self = *_self;
	if(self)
	{
		nn_skipLayer_delete(&self->skip2);
		nn_convLayer_delete(&self->conv2);
		nn_factLayer_delete(&self->fact2);
		nn_batchNormLayer_delete(&self->bn2);
		nn_convLayer_delete(&self->conv1);
		nn_factLayer_delete(&self->fact1);
		nn_batchNormLayer_delete(&self->bn1);
		nn_skipLayer_delete(&self->skip1);
		nn_layer_delete((nn_layer_t**) _self);
	}
}

nn_resLayer_t*
nn_resLayer_import(nn_arch_t* arch, cc_jsmnVal_t* val)
{
	ASSERT(arch);
	ASSERT(val);

	if(val->type != CC_JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	cc_jsmnVal_t* val_skip1 = NULL;
	cc_jsmnVal_t* val_bn1   = NULL;
	cc_jsmnVal_t* val_fact1 = NULL;
	cc_jsmnVal_t* val_conv1 = NULL;
	cc_jsmnVal_t* val_bn2   = NULL;
	cc_jsmnVal_t* val_fact2 = NULL;
	cc_jsmnVal_t* val_conv2 = NULL;
	cc_jsmnVal_t* val_skip2 = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		cc_jsmnKeyval_t* kv;
		kv = (cc_jsmnKeyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == CC_JSMN_TYPE_OBJECT)
		{
			if(strcmp(kv->key, "skip1") == 0)
			{
				val_skip1 = kv->val;
			}
			else if(strcmp(kv->key, "bn1") == 0)
			{
				val_bn1 = kv->val;
			}
			else if(strcmp(kv->key, "fact1") == 0)
			{
				val_fact1 = kv->val;
			}
			else if(strcmp(kv->key, "conv1") == 0)
			{
				val_conv1 = kv->val;
			}
			else if(strcmp(kv->key, "bn2") == 0)
			{
				val_bn2 = kv->val;
			}
			else if(strcmp(kv->key, "fact2") == 0)
			{
				val_fact2 = kv->val;
			}
			else if(strcmp(kv->key, "conv2") == 0)
			{
				val_conv2 = kv->val;
			}
			else if(strcmp(kv->key, "skip2") == 0)
			{
				val_skip2 = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	// bn layers are optional
	if((val_skip1 == NULL) ||
	   (val_fact1 == NULL) ||
	   (val_conv1 == NULL) ||
	   (val_fact2 == NULL) ||
	   (val_conv2 == NULL) ||
	   (val_skip2 == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_layerInfo_t layer_info =
	{
		.arch          = arch,
		.compute_fp_fn = nn_resLayer_computeFpFn,
		.compute_bp_fn = nn_resLayer_computeBpFn,
		.post_fn       = nn_resLayer_postFn,
		.dimX_fn       = nn_resLayer_dimXFn,
		.dimY_fn       = nn_resLayer_dimYFn,
	};

	nn_resLayer_t*  self;
	self = (nn_resLayer_t*)
	       nn_layer_new(sizeof(nn_resLayer_t),
	                    &layer_info);
	if(self == NULL)
	{
		return NULL;
	}

	self->skip1 = nn_skipLayer_import(arch, val_skip1,
	                                  NULL);
	if(self->skip1 == NULL)
	{
		goto failure;
	}

	if(val_bn1)
	{
		self->bn1 = nn_batchNormLayer_import(arch, val_bn1);
		if(self->bn1 == NULL)
		{
			goto failure;
		}
	}

	self->fact1 = nn_factLayer_import(arch, val_fact1);
	if(self->fact1 == NULL)
	{
		goto failure;
	}

	self->conv1 = nn_convLayer_import(arch, val_conv1);
	if(self->conv1 == NULL)
	{
		goto failure;
	}

	if(val_bn2)
	{
		self->bn2 = nn_batchNormLayer_import(arch, val_bn2);
		if(self->bn2 == NULL)
		{
			goto failure;
		}
	}

	self->fact2 = nn_factLayer_import(arch, val_fact2);
	if(self->fact2 == NULL)
	{
		goto failure;
	}

	self->conv2 = nn_convLayer_import(arch, val_conv2);
	if(self->conv2 == NULL)
	{
		goto failure;
	}

	self->skip2 = nn_skipLayer_import(arch, val_skip2,
	                                  self->skip1);
	if(self->skip2 == NULL)
	{
		goto failure;
	}

	// success
	return self;

	// failure
	failure:
		nn_resLayer_delete(&self);
	return NULL;
}

int nn_resLayer_export(nn_resLayer_t* self,
                       cc_jsmnStream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	int ret = 1;
	ret &= cc_jsmnStream_beginObject(stream);
	ret &= cc_jsmnStream_key(stream, "%s", "skip1");
	ret &= nn_skipLayer_export(self->skip1, stream);

	if(self->bn1)
	{
		ret &= cc_jsmnStream_key(stream, "%s", "bn1");
		ret &= nn_batchNormLayer_export(self->bn1, stream);
	}

	ret &= cc_jsmnStream_key(stream, "%s", "fact1");
	ret &= nn_factLayer_export(self->fact1, stream);
	ret &= cc_jsmnStream_key(stream, "%s", "conv1");
	ret &= nn_convLayer_export(self->conv1, stream);

	if(self->bn2)
	{
		ret &= cc_jsmnStream_key(stream, "%s", "bn2");
		ret &= nn_batchNormLayer_export(self->bn2, stream);
	}

	ret &= cc_jsmnStream_key(stream, "%s", "fact2");
	ret &= nn_factLayer_export(self->fact2, stream);
	ret &= cc_jsmnStream_key(stream, "%s", "conv2");
	ret &= nn_convLayer_export(self->conv2, stream);
	ret &= cc_jsmnStream_key(stream, "%s", "skip2");
	ret &= nn_skipLayer_export(self->skip2, stream);

	ret &= cc_jsmnStream_end(stream);

	return ret;
}
