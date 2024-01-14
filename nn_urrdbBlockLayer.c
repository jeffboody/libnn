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
#include "nn_coderLayer.h"
#include "nn_urrdbBlockLayer.h"
#include "nn_urrdbLayer.h"
#include "nn_urrdbNodeLayer.h"

/***********************************************************
* private                                                  *
***********************************************************/

static nn_tensor_t*
nn_urrdbBlockLayer_forwardPassFn(nn_layer_t* base, int flags,
                                 uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_urrdbBlockLayer_t* self;
	self = (nn_urrdbBlockLayer_t*) base;

	X = nn_layer_forwardPass(&self->coder0->base,
	                         flags, bs, X);
	if(X == NULL)
	{
		return NULL;
	}

	cc_listIter_t* iter = cc_list_head(self->nodes);
	while(iter)
	{
		nn_layer_t* node;
		node = (nn_layer_t*) cc_list_peekIter(iter);

		X = nn_layer_forwardPass(node, flags, bs, X);
		if(X == NULL)
		{
			return NULL;
		}

		iter = cc_list_next(iter);
	}

	return nn_layer_forwardPass(&self->coder1->base,
	                            flags, bs, X);
}

static nn_tensor_t*
nn_urrdbBlockLayer_backpropFn(nn_layer_t* base, int flags,
                              uint32_t bs,
                              nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,xh,xw,xd)

	nn_urrdbBlockLayer_t* self;
	self = (nn_urrdbBlockLayer_t*) base;

	dL_dY = nn_layer_backprop(&self->coder1->base, flags,
	                          bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	cc_listIter_t* iter = cc_list_tail(self->nodes);
	while(iter)
	{
		nn_layer_t* node;
		node = (nn_layer_t*) cc_list_peekIter(iter);

		dL_dY = nn_layer_backprop(node, flags, bs, dL_dY);
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
nn_urrdbBlockLayer_postFn(nn_layer_t* base, int flags)
{
	ASSERT(base);

	nn_urrdbBlockLayer_t* self;
	self = (nn_urrdbBlockLayer_t*) base;

	nn_layer_post(&self->coder0->base, flags);

	cc_listIter_t* iter = cc_list_head(self->nodes);
	while(iter)
	{
		nn_layer_t* node;
		node = (nn_layer_t*) cc_list_peekIter(iter);

		nn_layer_post(node, flags);

		iter = cc_list_next(iter);
	}

	nn_layer_post(&self->coder1->base, flags);
}

static nn_dim_t*
nn_urrdbBlockLayer_dimXFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_urrdbBlockLayer_t* self;
	self = (nn_urrdbBlockLayer_t*) base;

	return nn_layer_dimX(&self->coder0->base);
}

static nn_dim_t*
nn_urrdbBlockLayer_dimYFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_urrdbBlockLayer_t* self;
	self = (nn_urrdbBlockLayer_t*) base;

	return nn_layer_dimY(&self->coder1->base);
}

static void
nn_urrdbBlockLayer_deleteNodes(nn_urrdbBlockLayer_t* self)
{
	ASSERT(self);

	cc_listIter_t* iter = cc_list_head(self->nodes);
	while(iter)
	{
		nn_urrdbNodeLayer_t* node;
		node = (nn_urrdbNodeLayer_t*)
		        cc_list_remove(self->nodes, &iter);
		nn_urrdbNodeLayer_delete(&node);
	}
	cc_list_delete(&self->nodes);
}

static nn_dim_t*
nn_urrdbBlockLayer_addNode(nn_urrdbBlockLayer_t* self,
                           nn_urrdbLayerInfo_t* info,
                           nn_dim_t* dimX)
{
	ASSERT(self);
	ASSERT(info);
	ASSERT(dimX);

	nn_urrdbNodeLayer_t* node;
	node = nn_urrdbNodeLayer_new(info, dimX);
	if(node == NULL)
	{
		return NULL;
	}

	if(cc_list_append(self->nodes, NULL, node) == NULL)
	{
		goto fail_append;
	}

	// success
	return nn_layer_dimY(&node->base);

	// failure
	fail_append:
		nn_urrdbNodeLayer_delete(&node);
	return NULL;
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_urrdbBlockLayer_t*
nn_urrdbBlockLayer_new(nn_urrdbLayerInfo_t* info,
                       nn_dim_t* dimX)
{
	ASSERT(info);
	ASSERT(dimX);

	nn_dim_t* dim = dimX;

	nn_layerInfo_t layer_info =
	{
		.arch            = info->arch,
		.forward_pass_fn = nn_urrdbBlockLayer_forwardPassFn,
		.backprop_fn     = nn_urrdbBlockLayer_backpropFn,
		.post_fn         = nn_urrdbBlockLayer_postFn,
		.dimX_fn         = nn_urrdbBlockLayer_dimXFn,
		.dimY_fn         = nn_urrdbBlockLayer_dimYFn,
	};

	nn_urrdbBlockLayer_t* self;
	self = (nn_urrdbBlockLayer_t*)
	       nn_layer_new(sizeof(nn_urrdbBlockLayer_t),
	                    &layer_info);
	if(self == NULL)
	{
		return NULL;
	}

	nn_coderLayerInfo_t info_coder0 =
	{
		.arch       = info->arch,
		.dimX       = dim,
		.skip_mode  = NN_CODER_SKIP_MODE_FORK_ADD,
		.bn_mode    = info->bn_mode1,
		.fact_fn    = info->fact_fn1,
	};

	self->coder0 = nn_coderLayer_new(&info_coder0);
	if(self->coder0 == NULL)
	{
		goto fail_coder0;
	}

	self->nodes = cc_list_new();
	if(self->nodes == NULL)
	{
		goto fail_nodes;
	}

	// final node is coder1
	int i;
	for(i = 0; i < (info->nodes - 1); ++i)
	{
		dim = nn_urrdbBlockLayer_addNode(self, info, dim);
		if(dim == NULL)
		{
			goto fail_node;
		}
	}

	nn_coderLayerInfo_t info_coder1 =
	{
		.arch       = info->arch,
		.dimX       = dim,
		.fc         = info->fc,
		.norm_flags = info->norm_flags1,
		.conv_size  = info->conv_size1,
		.skip_mode  = NN_CODER_SKIP_MODE_ADD,
		.skip_coder = self->coder0,
		.skip_beta  = info->skip_beta1,
		// NO BN/RELU
	};

	self->coder1 = nn_coderLayer_new(&info_coder1);
	if(self->coder1 == NULL)
	{
		goto fail_coder1;
	}

	// success
	return self;

	// failure
	fail_coder1:
	fail_node:
		nn_urrdbBlockLayer_deleteNodes(self);
	fail_nodes:
		nn_coderLayer_delete(&self->coder0);
	fail_coder0:
		nn_layer_delete((nn_layer_t**) &self);
	return NULL;
}

void nn_urrdbBlockLayer_delete(nn_urrdbBlockLayer_t** _self)
{
	ASSERT(_self);

	nn_urrdbBlockLayer_t* self = *_self;
	if(self)
	{
		nn_coderLayer_delete(&self->coder1);
		nn_urrdbBlockLayer_deleteNodes(self);
		nn_coderLayer_delete(&self->coder0);
		nn_layer_delete((nn_layer_t**) _self);
	}
}

nn_urrdbBlockLayer_t*
nn_urrdbBlockLayer_import(nn_arch_t* arch, jsmn_val_t* val)
{
	ASSERT(arch);
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	// val_nodes references
	cc_list_t* val_nodes = cc_list_new();
	if(val_nodes == NULL)
	{
		return NULL;
	}

	jsmn_val_t* val_coder0 = NULL;
	jsmn_val_t* val_coder1 = NULL;

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
			else if(strcmp(kv->key, "node") == 0)
			{
				if(cc_list_append(val_nodes, NULL,
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
	   (cc_list_size(val_nodes) < 1))
	{
		LOGE("invalid");
		goto fail_param;
	}

	nn_layerInfo_t layer_info =
	{
		.arch            = arch,
		.forward_pass_fn = nn_urrdbBlockLayer_forwardPassFn,
		.backprop_fn     = nn_urrdbBlockLayer_backpropFn,
		.post_fn         = nn_urrdbBlockLayer_postFn,
		.dimX_fn         = nn_urrdbBlockLayer_dimXFn,
		.dimY_fn         = nn_urrdbBlockLayer_dimYFn,
	};

	nn_urrdbBlockLayer_t* self;
	self = (nn_urrdbBlockLayer_t*)
	       nn_layer_new(sizeof(nn_urrdbBlockLayer_t),
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

	self->nodes = cc_list_new();
	if(self->nodes == NULL)
	{
		goto fail_nodes;
	}

	// process val_nodes references
	iter = cc_list_head(val_nodes);
	while(iter)
	{
		jsmn_val_t* val_node;
		val_node = (jsmn_val_t*)
		           cc_list_remove(val_nodes, &iter);

		nn_urrdbNodeLayer_t* node;
		node = nn_urrdbNodeLayer_import(arch, val_node);
		if(node == NULL)
		{
			goto fail_node;
		}

		if(cc_list_append(self->nodes, NULL, node) == NULL)
		{
			nn_urrdbNodeLayer_delete(&node);
			goto fail_node;
		}
	}

	self->coder1 = nn_coderLayer_import(arch, val_coder1,
	                                    self->coder0);
	if(self->coder1 == NULL)
	{
		goto fail_coder1;
	}

	// delete empty val_nodes
	cc_list_delete(&val_nodes);

	// success
	return self;

	// failure
	fail_coder1:
	fail_node:
		nn_urrdbBlockLayer_deleteNodes(self);
	fail_nodes:
		nn_coderLayer_delete(&self->coder0);
	fail_coder0:
		nn_layer_delete((nn_layer_t**) &self);
	fail_layer:
	fail_param:
	fail_append:
	{
		cc_list_discard(val_nodes);
		cc_list_delete(&val_nodes);
	}
	return NULL;
}

int nn_urrdbBlockLayer_export(nn_urrdbBlockLayer_t* self,
                              jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	ret &= jsmn_stream_key(stream, "%s", "coder0");
	ret &= nn_coderLayer_export(self->coder0, stream);

	cc_listIter_t* iter = cc_list_head(self->nodes);
	while(iter)
	{
		nn_urrdbNodeLayer_t* node;
		node = (nn_urrdbNodeLayer_t*)
		       cc_list_peekIter(iter);

		ret &= jsmn_stream_key(stream, "%s", "node");
		ret &= nn_urrdbNodeLayer_export(node, stream);

		iter = cc_list_next(iter);
	}

	ret &= jsmn_stream_key(stream, "%s", "coder1");
	ret &= nn_coderLayer_export(self->coder1, stream);
	ret &= jsmn_stream_end(stream);

	return ret;
}
