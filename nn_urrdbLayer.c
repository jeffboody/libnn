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
#include "nn_tensor.h"
#include "nn_urrdbBlockLayer.h"
#include "nn_urrdbLayer.h"

/***********************************************************
* private                                                  *
***********************************************************/

static nn_tensor_t*
nn_urrdbLayer_forwardPassFn(nn_layer_t* base, int flags,
                            uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_urrdbLayer_t* self;
	self = (nn_urrdbLayer_t*) base;

	X = nn_layer_forwardPass(&self->coder0->base,
	                         flags, bs, X);
	if(X == NULL)
	{
		return NULL;
	}

	cc_listIter_t* iter = cc_list_head(self->blocks);
	while(iter)
	{
		nn_layer_t* block;
		block = (nn_layer_t*) cc_list_peekIter(iter);

		X = nn_layer_forwardPass(block, flags, bs, X);
		if(X == NULL)
		{
			return NULL;
		}

		iter = cc_list_next(iter);
	}

	X = nn_layer_forwardPass(&self->coder1->base,
	                         flags, bs, X);
	if(X == NULL)
	{
		return NULL;
	}

	return nn_layer_forwardPass(&self->coder2->base,
	                            flags, bs, X);
}

static nn_tensor_t*
nn_urrdbLayer_backpropFn(nn_layer_t* base, int flags,
                         uint32_t bs, nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,xh,xw,xd)

	nn_urrdbLayer_t* self;
	self = (nn_urrdbLayer_t*) base;

	dL_dY = nn_layer_backprop(&self->coder2->base, flags,
	                          bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	dL_dY = nn_layer_backprop(&self->coder1->base, flags,
	                          bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	cc_listIter_t* iter = cc_list_tail(self->blocks);
	while(iter)
	{
		nn_layer_t* block;
		block = (nn_layer_t*) cc_list_peekIter(iter);

		dL_dY = nn_layer_backprop(block, flags, bs, dL_dY);
		if(dL_dY == NULL)
		{
			return NULL;
		}

		iter = cc_list_prev(iter);
	}

	return nn_layer_backprop(&self->coder0->base, flags,
	                         bs, dL_dY);
}

static void
nn_urrdbLayer_postFn(nn_layer_t* base, int flags)
{
	ASSERT(base);

	nn_urrdbLayer_t* self;
	self = (nn_urrdbLayer_t*) base;

	nn_layer_post(&self->coder0->base, flags);

	cc_listIter_t* iter = cc_list_head(self->blocks);
	while(iter)
	{
		nn_layer_t* block;
		block = (nn_layer_t*) cc_list_peekIter(iter);

		nn_layer_post(block, flags);

		iter = cc_list_next(iter);
	}

	nn_layer_post(&self->coder1->base, flags);
	nn_layer_post(&self->coder2->base, flags);
}

static nn_dim_t*
nn_urrdbLayer_dimXFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_urrdbLayer_t* self = (nn_urrdbLayer_t*) base;

	return nn_layer_dimX(&self->coder0->base);
}

static nn_dim_t*
nn_urrdbLayer_dimYFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_urrdbLayer_t* self = (nn_urrdbLayer_t*) base;

	return nn_layer_dimY(&self->coder2->base);
}

static void
nn_urrdbLayer_deleteBlocks(nn_urrdbLayer_t* self)
{
	ASSERT(self);

	cc_listIter_t* iter = cc_list_head(self->blocks);
	while(iter)
	{
		nn_urrdbBlockLayer_t* block;
		block = (nn_urrdbBlockLayer_t*)
		        cc_list_remove(self->blocks, &iter);
		nn_urrdbBlockLayer_delete(&block);
	}
	cc_list_delete(&self->blocks);
}

static int
nn_urrdbLayer_addBlock(nn_urrdbLayer_t* self,
                       nn_urrdbLayerInfo_t* info,
                       nn_dim_t* dimX)
{
	ASSERT(self);
	ASSERT(info);
	ASSERT(dimX);

	nn_urrdbBlockLayer_t* block;
	block = nn_urrdbBlockLayer_new(info, dimX);
	if(block == NULL)
	{
		return 0;
	}

	if(cc_list_append(self->blocks, NULL, block) == NULL)
	{
		goto fail_append;
	}

	// success
	return 1;

	// failure
	fail_append:
		nn_urrdbBlockLayer_delete(&block);
	return 0;
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_urrdbLayer_t*
nn_urrdbLayer_new(nn_urrdbLayerInfo_t* info)
{
	ASSERT(info);

	nn_dim_t* dim = info->dimX;

	nn_layerInfo_t layer_info =
	{
		.arch            = info->arch,
		.forward_pass_fn = nn_urrdbLayer_forwardPassFn,
		.backprop_fn     = nn_urrdbLayer_backpropFn,
		.post_fn         = nn_urrdbLayer_postFn,
		.dimX_fn         = nn_urrdbLayer_dimXFn,
		.dimY_fn         = nn_urrdbLayer_dimYFn,
	};

	nn_urrdbLayer_t* self;
	self = (nn_urrdbLayer_t*)
	       nn_layer_new(sizeof(nn_urrdbLayer_t),
	                    &layer_info);
	if(self == NULL)
	{
		return NULL;
	}

	nn_coderLayerInfo_t info_coder0 =
	{
		.arch        = info->arch,
		.dimX        = dim,
		.fc          = info->fc,
		.conv_flags  = info->norm_flags0,
		.conv_size   = info->conv_size0,
		.conv_stride = 1,
		.skip_mode   = NN_CODER_SKIP_MODE_FORK_ADD,
		// NO BN/RELU
	};

	self->coder0 = nn_coderLayer_new(&info_coder0);
	if(self->coder0 == NULL)
	{
		goto fail_coder0;
	}
	dim = nn_layer_dimY(&self->coder0->base);

	self->blocks = cc_list_new();
	if(self->blocks == NULL)
	{
		goto fail_blocks;
	}

	int i;
	for(i = 0; i < info->blocks; ++i)
	{
		if(nn_urrdbLayer_addBlock(self, info, dim) == 0)
		{
			goto fail_block;
		}
	}

	nn_coderLayerInfo_t info_coder1 =
	{
		.arch    = info->arch,
		.dimX    = dim,
		.bn_mode = info->bn_mode0,
		.fact_fn = info->fact_fn0,
	};

	self->coder1 = nn_coderLayer_new(&info_coder1);
	if(self->coder1 == NULL)
	{
		goto fail_coder1;
	}

	nn_coderLayerInfo_t info_coder2 =
	{
		.arch        = info->arch,
		.dimX        = dim,
		.fc          = info->fc,
		.conv_flags  = info->norm_flags0,
		.conv_size   = info->conv_size0,
		.conv_stride = 1,
		.skip_mode   = NN_CODER_SKIP_MODE_ADD,
		.skip_coder  = self->coder0,
		.skip_beta   = info->skip_beta0,
		.bn_mode     = info->bn_mode0,
		.fact_fn     = info->fact_fn0,
	};

	self->coder2 = nn_coderLayer_new(&info_coder2);
	if(self->coder2 == NULL)
	{
		goto fail_coder2;
	}

	// success
	return self;

	// failure
	fail_coder2:
		nn_coderLayer_delete(&self->coder1);
	fail_coder1:
	fail_block:
		nn_urrdbLayer_deleteBlocks(self);
	fail_blocks:
		nn_coderLayer_delete(&self->coder0);
	fail_coder0:
		nn_layer_delete((nn_layer_t**) &self);
	return NULL;
}

void nn_urrdbLayer_delete(nn_urrdbLayer_t** _self)
{
	ASSERT(_self);

	nn_urrdbLayer_t* self = *_self;
	if(self)
	{
		nn_coderLayer_delete(&self->coder2);
		nn_coderLayer_delete(&self->coder1);
		nn_urrdbLayer_deleteBlocks(self);
		nn_coderLayer_delete(&self->coder0);
		nn_layer_delete((nn_layer_t**) _self);
	}
}

nn_urrdbLayer_t*
nn_urrdbLayer_import(nn_arch_t* arch, jsmn_val_t* val)
{
	ASSERT(arch);
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	// val_blocks references
	cc_list_t* val_blocks = cc_list_new();
	if(val_blocks == NULL)
	{
		return NULL;
	}

	jsmn_val_t* val_coder0 = NULL;
	jsmn_val_t* val_coder1 = NULL;
	jsmn_val_t* val_coder2 = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		jsmn_keyval_t* kv;
		kv = (jsmn_keyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == JSMN_TYPE_OBJECT)
		{
			if(strcmp(kv->key, "coder0") == 0)
			{
				val_coder0 = kv->val;
			}
			else if(strcmp(kv->key, "coder1") == 0)
			{
				val_coder1 = kv->val;
			}
			else if(strcmp(kv->key, "coder2") == 0)
			{
				val_coder2 = kv->val;
			}
			else if(strcmp(kv->key, "block") == 0)
			{
				if(cc_list_append(val_blocks, NULL,
				                  kv->val) == NULL)
				{
					goto fail_append;
				}
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_coder0 == NULL) ||
	   (val_coder1 == NULL) ||
	   (val_coder2 == NULL) ||
	   (cc_list_size(val_blocks) < 1))
	{
		LOGE("invalid");
		goto fail_param;
	}

	nn_layerInfo_t layer_info =
	{
		.arch            = arch,
		.forward_pass_fn = nn_urrdbLayer_forwardPassFn,
		.backprop_fn     = nn_urrdbLayer_backpropFn,
		.post_fn         = nn_urrdbLayer_postFn,
		.dimX_fn         = nn_urrdbLayer_dimXFn,
		.dimY_fn         = nn_urrdbLayer_dimYFn,
	};

	nn_urrdbLayer_t* self;
	self = (nn_urrdbLayer_t*)
	       nn_layer_new(sizeof(nn_urrdbLayer_t),
	                    &layer_info);
	if(self == NULL)
	{
		goto fail_layer;
	}

	self->coder0 = nn_coderLayer_import(arch, val_coder0,
	                                    NULL);
	if(self->coder0 == NULL)
	{
		goto fail_coder0;
	}

	self->blocks = cc_list_new();
	if(self->blocks == NULL)
	{
		goto fail_blocks;
	}

	// process val_blocks references
	iter = cc_list_head(val_blocks);
	while(iter)
	{
		jsmn_val_t* val;
		val = (jsmn_val_t*)
		      cc_list_remove(val_blocks, &iter);

		nn_urrdbBlockLayer_t* block;
		block = nn_urrdbBlockLayer_import(arch, val);
		if(block == NULL)
		{
			goto fail_block;
		}

		if(cc_list_append(self->blocks, NULL, block) == NULL)
		{
			nn_urrdbBlockLayer_delete(&block);
			goto fail_block;
		}
	}

	self->coder1 = nn_coderLayer_import(arch, val_coder1,
	                                    NULL);
	if(self->coder1 == NULL)
	{
		goto fail_coder1;
	}

	self->coder2 = nn_coderLayer_import(arch, val_coder2,
	                                    self->coder0);
	if(self->coder2 == NULL)
	{
		goto fail_coder2;
	}

	// delete empty val_blocks
	cc_list_delete(&val_blocks);

	// success
	return self;

	// failure
	fail_coder2:
		nn_coderLayer_delete(&self->coder1);
	fail_coder1:
	fail_block:
		nn_urrdbLayer_deleteBlocks(self);
	fail_blocks:
		nn_coderLayer_delete(&self->coder0);
	fail_coder0:
		nn_layer_delete((nn_layer_t**) &self);
	fail_layer:
	fail_param:
	fail_append:
	{
		cc_list_discard(val_blocks);
		cc_list_delete(&val_blocks);
	}
	return NULL;
}

int nn_urrdbLayer_export(nn_urrdbLayer_t* self,
                         jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	ret &= jsmn_stream_key(stream, "%s", "coder0");
	ret &= nn_coderLayer_export(self->coder0, stream);

	cc_listIter_t* iter = cc_list_head(self->blocks);
	while(iter)
	{
		nn_urrdbBlockLayer_t* block;
		block = (nn_urrdbBlockLayer_t*)
		        cc_list_peekIter(iter);

		ret &= jsmn_stream_key(stream, "%s", "block");
		ret &= nn_urrdbBlockLayer_export(block, stream);

		iter = cc_list_next(iter);
	}

	ret &= jsmn_stream_key(stream, "%s", "coder1");
	ret &= nn_coderLayer_export(self->coder1, stream);
	ret &= jsmn_stream_key(stream, "%s", "coder2");
	ret &= nn_coderLayer_export(self->coder2, stream);
	ret &= jsmn_stream_end(stream);

	return ret;
}
