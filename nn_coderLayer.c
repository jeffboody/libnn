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
#include "nn_poolingLayer.h"
#include "nn_skipLayer.h"
#include "nn_tensor.h"

/***********************************************************
* private - nn_coderOpLayer_t                              *
***********************************************************/

static nn_tensor_t*
nn_coderOpLayer_forwardPassFn(nn_layer_t* base, int flags,
                              uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_coderOpLayer_t* self;
	self = (nn_coderOpLayer_t*) base;

	if((self->op_mode == NN_CODER_OP_MODE_CONVT_2X2_S2) ||
	   (self->op_mode == NN_CODER_OP_MODE_CONVT_6X6_S2) ||
	   (self->op_mode == NN_CODER_OP_MODE_CONV_3X3_S2))
	{
		return nn_layer_forwardPass(&self->conv->base,
		                            flags, bs, X);
	}
	else
	{
		return nn_layer_forwardPass(&self->pool->base,
		                            flags, bs, X);
	}
}

static nn_tensor_t*
nn_coderOpLayer_backpropFn(nn_layer_t* base,
                           int flags, uint32_t bs,
                           nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,xh,xw,xd)

	nn_coderOpLayer_t* self;
	self = (nn_coderOpLayer_t*) base;

	if((self->op_mode == NN_CODER_OP_MODE_CONVT_2X2_S2) ||
	   (self->op_mode == NN_CODER_OP_MODE_CONVT_6X6_S2) ||
	   (self->op_mode == NN_CODER_OP_MODE_CONV_3X3_S2))
	{
		return nn_layer_backprop(&self->conv->base, flags,
		                         bs, dL_dY);
	}
	else
	{
		return nn_layer_backprop(&self->pool->base, flags,
		                         bs, dL_dY);
	}
}

static void
nn_coderOpLayer_postFn(nn_layer_t* base, int flags)
{
	ASSERT(base);

	nn_coderOpLayer_t* self = (nn_coderOpLayer_t*) base;

	if((self->op_mode == NN_CODER_OP_MODE_CONVT_2X2_S2) ||
	   (self->op_mode == NN_CODER_OP_MODE_CONVT_6X6_S2) ||
	   (self->op_mode == NN_CODER_OP_MODE_CONV_3X3_S2))
	{
		return nn_layer_post(&self->conv->base, flags);
	}
	else
	{
		return nn_layer_post(&self->pool->base, flags);
	}
}

static nn_dim_t*
nn_coderOpLayer_dimXFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_coderOpLayer_t* self;
	self = (nn_coderOpLayer_t*) base;

	if((self->op_mode == NN_CODER_OP_MODE_CONVT_2X2_S2) ||
	   (self->op_mode == NN_CODER_OP_MODE_CONVT_6X6_S2) ||
	   (self->op_mode == NN_CODER_OP_MODE_CONV_3X3_S2))
	{
		return nn_layer_dimX(&self->conv->base);
	}
	else
	{
		return nn_layer_dimX(&self->pool->base);
	}
}

static nn_dim_t*
nn_coderOpLayer_dimYFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_coderOpLayer_t* self;
	self = (nn_coderOpLayer_t*) base;

	if((self->op_mode == NN_CODER_OP_MODE_CONVT_2X2_S2) ||
	   (self->op_mode == NN_CODER_OP_MODE_CONVT_6X6_S2) ||
	   (self->op_mode == NN_CODER_OP_MODE_CONV_3X3_S2))
	{
		return nn_layer_dimY(&self->conv->base);
	}
	else
	{
		return nn_layer_dimY(&self->pool->base);
	}
}

static nn_coderOpLayer_t*
nn_coderOpLayer_new(nn_arch_t* arch, nn_dim_t* dimX,
                    nn_coderOpMode_e op_mode)
{
	ASSERT(arch);
	ASSERT(dimX);

	nn_layerInfo_t info =
	{
		.arch            = arch,
		.forward_pass_fn = nn_coderOpLayer_forwardPassFn,
		.backprop_fn     = nn_coderOpLayer_backpropFn,
		.post_fn         = nn_coderOpLayer_postFn,
		.dimX_fn         = nn_coderOpLayer_dimXFn,
		.dimY_fn         = nn_coderOpLayer_dimYFn,
	};

	nn_coderOpLayer_t*  self;
	self = (nn_coderOpLayer_t*)
	       nn_layer_new(sizeof(nn_coderOpLayer_t),
	                    &info);
	if(self == NULL)
	{
		return NULL;
	}

	self->op_mode = op_mode;

	uint32_t xd = dimX->depth;
	if(op_mode == NN_CODER_OP_MODE_CONVT_2X2_S2)
	{
		nn_dim_t dimW =
		{
			.count  = xd,
			.width  = 2,
			.height = 2,
			.depth  = xd,
		};

		self->conv = nn_convLayer_new(arch, dimX, &dimW, 2,
		                              NN_CONV_LAYER_FLAG_TRANSPOSE |
		                              NN_CONV_LAYER_FLAG_XAVIER);
		if(self->conv == NULL)
		{
			goto fail_op;
		}
	}
	else if(op_mode == NN_CODER_OP_MODE_CONVT_6X6_S2)
	{
		nn_dim_t dimW =
		{
			.count  = xd,
			.width  = 6,
			.height = 6,
			.depth  = xd,
		};

		self->conv = nn_convLayer_new(arch, dimX, &dimW, 2,
		                              NN_CONV_LAYER_FLAG_TRANSPOSE |
		                              NN_CONV_LAYER_FLAG_XAVIER);
		if(self->conv == NULL)
		{
			goto fail_op;
		}
	}
	else if(op_mode == NN_CODER_OP_MODE_CONV_3X3_S2)
	{
		nn_dim_t dimW =
		{
			.count  = xd,
			.width  = 3,
			.height = 3,
			.depth  = xd,
		};

		self->conv = nn_convLayer_new(arch, dimX, &dimW, 2,
		                              NN_CONV_LAYER_FLAG_XAVIER);
		if(self->conv == NULL)
		{
			goto fail_op;
		}
	}
	else if(op_mode == NN_CODER_OP_MODE_POOL_AVG_S2)
	{
		self->pool = nn_poolingLayer_new(arch, dimX, 2,
		                                 NN_POOLING_MODE_AVERAGE);
		if(self->pool == NULL)
		{
			goto fail_op;
		}
	}
	else if(op_mode == NN_CODER_OP_MODE_POOL_MAX_S2)
	{
		self->pool = nn_poolingLayer_new(arch, dimX, 2,
		                                 NN_POOLING_MODE_MAX);
		if(self->pool == NULL)
		{
			goto fail_op;
		}
	}
	else
	{
		LOGE("invalid op_mode=%i", (int) op_mode);
		goto fail_op;
	}

	// success
	return self;

	// failure
	fail_op:
		nn_layer_delete((nn_layer_t**) &self);
	return NULL;
}

static void
nn_coderOpLayer_delete(nn_coderOpLayer_t** _self)
{
	ASSERT(_self);

	nn_coderOpLayer_t* self = *_self;
	if(self)
	{
		if((self->op_mode == NN_CODER_OP_MODE_CONVT_2X2_S2) ||
		   (self->op_mode == NN_CODER_OP_MODE_CONVT_6X6_S2) ||
		   (self->op_mode == NN_CODER_OP_MODE_CONV_3X3_S2))
		{
			nn_convLayer_delete(&self->conv);
		}
		else
		{
			nn_poolingLayer_delete(&self->pool);
		}
		nn_layer_delete((nn_layer_t**) _self);
	}
}

static nn_coderOpLayer_t*
nn_coderOpLayer_import(nn_arch_t* arch, jsmn_val_t* val)
{
	ASSERT(arch);
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	jsmn_val_t* val_op_mode = NULL;
	jsmn_val_t* val_conv    = NULL;
	jsmn_val_t* val_pool    = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		jsmn_keyval_t* kv;
		kv = (jsmn_keyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == JSMN_TYPE_OBJECT)
		{
			if(strcmp(kv->key, "conv") == 0)
			{
				val_conv = kv->val;
			}
			else if(strcmp(kv->key, "pool") == 0)
			{
				val_pool = kv->val;
			}
		}
		else if(kv->val->type == JSMN_TYPE_STRING)
		{
			if(strcmp(kv->key, "op_mode") == 0)
			{
				val_op_mode = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	// conv or pool also required depending on op_mode
	if(val_op_mode == NULL)
	{
		LOGE("invalid");
		return NULL;
	}

	nn_layerInfo_t info =
	{
		.arch            = arch,
		.forward_pass_fn = nn_coderOpLayer_forwardPassFn,
		.backprop_fn     = nn_coderOpLayer_backpropFn,
		.post_fn         = nn_coderOpLayer_postFn,
		.dimX_fn         = nn_coderOpLayer_dimXFn,
		.dimY_fn         = nn_coderOpLayer_dimYFn,
	};

	nn_coderOpLayer_t*  self;
	self = (nn_coderOpLayer_t*)
	       nn_layer_new(sizeof(nn_coderOpLayer_t),
	                    &info);
	if(self == NULL)
	{
		return NULL;
	}

	if(strcmp(val_op_mode->data, "CONVT_2X2_S2") == 0)
	{
		self->op_mode = NN_CODER_OP_MODE_CONVT_2X2_S2;
	}
	else if(strcmp(val_op_mode->data, "CONVT_6X6_S2") == 0)
	{
		self->op_mode = NN_CODER_OP_MODE_CONVT_6X6_S2;
	}
	else if(strcmp(val_op_mode->data, "CONV_3X3_S2") == 0)
	{
		self->op_mode = NN_CODER_OP_MODE_CONV_3X3_S2;
	}
	else if(strcmp(val_op_mode->data, "POOL_AVG_S2") == 0)
	{
		self->op_mode = NN_CODER_OP_MODE_POOL_AVG_S2;
	}
	else if(strcmp(val_op_mode->data, "POOL_MAX_S2") == 0)
	{
		self->op_mode = NN_CODER_OP_MODE_POOL_MAX_S2;
	}

	if(val_conv &&
	   ((self->op_mode == NN_CODER_OP_MODE_CONVT_2X2_S2) ||
	    (self->op_mode == NN_CODER_OP_MODE_CONVT_6X6_S2) ||
	    (self->op_mode == NN_CODER_OP_MODE_CONV_3X3_S2)))
	{
		self->conv = nn_convLayer_import(arch, val_conv);
		if(self->conv == NULL)
		{
			goto fail_op;
		}
	}
	else if(val_pool &&
	        ((self->op_mode == NN_CODER_OP_MODE_POOL_AVG_S2) ||
	         (self->op_mode == NN_CODER_OP_MODE_POOL_MAX_S2)))
	{
		self->pool = nn_poolingLayer_import(arch, val_pool);
		if(self->pool == NULL)
		{
			goto fail_op;
		}
	}
	else
	{
		LOGE("invalid op_mode=%i", self->op_mode);
		goto fail_op;
	}

	// success
	return self;

	// failure
	fail_op:
		nn_layer_delete((nn_layer_t**) &self);
	return NULL;
}

static int
nn_coderOpLayer_export(nn_coderOpLayer_t* self,
                       jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	if(self->op_mode == NN_CODER_OP_MODE_CONVT_2X2_S2)
	{
		ret &= jsmn_stream_key(stream, "%s", "op_mode");
		ret &= jsmn_stream_string(stream, "%s", "CONVT_2X2_S2");
		ret &= jsmn_stream_key(stream, "%s", "conv");
		ret &= nn_convLayer_export(self->conv, stream);
	}
	else if(self->op_mode == NN_CODER_OP_MODE_CONVT_6X6_S2)
	{
		ret &= jsmn_stream_key(stream, "%s", "op_mode");
		ret &= jsmn_stream_string(stream, "%s", "CONVT_6X6_S2");
		ret &= jsmn_stream_key(stream, "%s", "conv");
		ret &= nn_convLayer_export(self->conv, stream);
	}
	else if(self->op_mode == NN_CODER_OP_MODE_CONV_3X3_S2)
	{
		ret &= jsmn_stream_key(stream, "%s", "op_mode");
		ret &= jsmn_stream_string(stream, "%s", "CONV_3X3_S2");
		ret &= jsmn_stream_key(stream, "%s", "conv");
		ret &= nn_convLayer_export(self->conv, stream);
	}
	else if(self->op_mode == NN_CODER_OP_MODE_POOL_AVG_S2)
	{
		ret &= jsmn_stream_key(stream, "%s", "op_mode");
		ret &= jsmn_stream_string(stream, "%s", "POOL_AVG_S2");
		ret &= jsmn_stream_key(stream, "%s", "pool");
		ret &= nn_poolingLayer_export(self->pool, stream);
	}
	else if(self->op_mode == NN_CODER_OP_MODE_POOL_MAX_S2)
	{
		ret &= jsmn_stream_key(stream, "%s", "op_mode");
		ret &= jsmn_stream_string(stream, "%s", "POOL_MAX_S2");
		ret &= jsmn_stream_key(stream, "%s", "pool");
		ret &= nn_poolingLayer_export(self->pool, stream);
	}
	ret &= jsmn_stream_end(stream);

	return ret;
}

/***********************************************************
* private - nn_coderRepeaterLayer_t                        *
***********************************************************/

typedef struct nn_coderRepeaterLayer_s
{
	nn_layer_t base;

	// repeater layer
	// he, relu
	// xd : fc
	// W  : dim(xd,3,3,xd)
	// Y  : dim(bs,xh,xw,xd)
	nn_convLayer_t* conv;
	nn_factLayer_t* fact;
} nn_coderRepeaterLayer_t;

static nn_tensor_t*
nn_coderRepeaterLayer_forwardPassFn(nn_layer_t* base,
                                    int flags, uint32_t bs,
                                    nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_coderRepeaterLayer_t* self;
	self = (nn_coderRepeaterLayer_t*) base;

	X = nn_layer_forwardPass(&self->conv->base, flags,
	                         bs, X);
	if(X == NULL)
	{
		return NULL;
	}

	return nn_layer_forwardPass(&self->fact->base,
	                            flags, bs, X);
}

static nn_tensor_t*
nn_coderRepeaterLayer_backpropFn(nn_layer_t* base,
                                 int flags, uint32_t bs,
                                 nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,xh,xw,xd)

	nn_coderRepeaterLayer_t* self;
	self = (nn_coderRepeaterLayer_t*) base;

	dL_dY = nn_layer_backprop(&self->fact->base, flags,
	                          bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	return nn_layer_backprop(&self->conv->base, flags,
	                         bs, dL_dY);
}

static void
nn_coderRepeaterLayer_postFn(nn_layer_t* base, int flags)
{
	ASSERT(base);

	nn_coderRepeaterLayer_t* self;
	self = (nn_coderRepeaterLayer_t*) base;

	nn_layer_post(&self->conv->base, flags);
	nn_layer_post(&self->fact->base, flags);
}

static nn_dim_t*
nn_coderRepeaterLayer_dimXFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_coderRepeaterLayer_t* self;
	self = (nn_coderRepeaterLayer_t*) base;

	return nn_layer_dimX(&self->conv->base);
}

static nn_dim_t*
nn_coderRepeaterLayer_dimYFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_coderRepeaterLayer_t* self;
	self = (nn_coderRepeaterLayer_t*) base;

	return nn_layer_dimY(&self->fact->base);
}

static nn_coderRepeaterLayer_t*
nn_coderRepeaterLayer_new(nn_arch_t* arch, nn_dim_t* dimX)
{
	ASSERT(arch);
	ASSERT(dimX);

	nn_layerInfo_t info =
	{
		.arch            = arch,
		.forward_pass_fn = nn_coderRepeaterLayer_forwardPassFn,
		.backprop_fn     = nn_coderRepeaterLayer_backpropFn,
		.post_fn         = nn_coderRepeaterLayer_postFn,
		.dimX_fn         = nn_coderRepeaterLayer_dimXFn,
		.dimY_fn         = nn_coderRepeaterLayer_dimYFn,
	};

	nn_dim_t* dim = dimX;
	uint32_t  xd  = dim->depth;

	nn_coderRepeaterLayer_t*  self;
	self = (nn_coderRepeaterLayer_t*)
	       nn_layer_new(sizeof(nn_coderRepeaterLayer_t),
	                    &info);
	if(self == NULL)
	{
		return NULL;
	}

	nn_dim_t dimW =
	{
		.count  = xd,
		.width  = 3,
		.height = 3,
		.depth  = xd,
	};

	self->conv = nn_convLayer_new(arch, dim, &dimW, 1,
	                              NN_CONV_LAYER_FLAG_HE);
	if(self->conv == NULL)
	{
		goto fail_conv;
	}
	dim = nn_layer_dimY(&self->conv->base);

	self->fact = nn_factLayer_new(arch, dim,
	                              NN_FACT_LAYER_FN_RELU);
	if(self->fact == NULL)
	{
		goto fail_fact;
	}

	// success
	return self;

	// failure
	fail_fact:
		nn_convLayer_delete(&self->conv);
	fail_conv:
		nn_layer_delete((nn_layer_t**) &self);
	return NULL;
}

static void
nn_coderRepeaterLayer_delete(nn_coderRepeaterLayer_t** _self)
{
	ASSERT(_self);

	nn_coderRepeaterLayer_t* self = *_self;
	if(self)
	{
		nn_factLayer_delete(&self->fact);
		nn_convLayer_delete(&self->conv);
		nn_layer_delete((nn_layer_t**) _self);
	}
}

static nn_coderRepeaterLayer_t*
nn_coderRepeaterLayer_import(nn_arch_t* arch,
                             jsmn_val_t* val)
{
	ASSERT(arch);
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	jsmn_val_t* val_conv = NULL;
	jsmn_val_t* val_fact = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		jsmn_keyval_t* kv;
		kv = (jsmn_keyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == JSMN_TYPE_OBJECT)
		{
			if(strcmp(kv->key, "conv") == 0)
			{
				val_conv = kv->val;
			}
			else if(strcmp(kv->key, "fact") == 0)
			{
				val_fact = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_conv == NULL) ||
	   (val_fact == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_layerInfo_t info =
	{
		.arch            = arch,
		.forward_pass_fn = nn_coderRepeaterLayer_forwardPassFn,
		.backprop_fn     = nn_coderRepeaterLayer_backpropFn,
		.post_fn         = nn_coderRepeaterLayer_postFn,
		.dimX_fn         = nn_coderRepeaterLayer_dimXFn,
		.dimY_fn         = nn_coderRepeaterLayer_dimYFn,
	};

	nn_coderRepeaterLayer_t*  self;
	self = (nn_coderRepeaterLayer_t*)
	       nn_layer_new(sizeof(nn_coderRepeaterLayer_t),
	                    &info);
	if(self == NULL)
	{
		return NULL;
	}

	self->conv = nn_convLayer_import(arch, val_conv);
	if(self->conv == NULL)
	{
		goto fail_conv;
	}

	self->fact = nn_factLayer_import(arch, val_fact);
	if(self->fact == NULL)
	{
		goto fail_fact;
	}

	// success
	return self;

	// failure
	fail_fact:
		nn_convLayer_delete(&self->conv);
	fail_conv:
		nn_layer_delete((nn_layer_t**) &self);
	return NULL;
}

static int
nn_coderRepeaterLayer_export(nn_coderRepeaterLayer_t* self,
                             jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	ret &= jsmn_stream_key(stream, "%s", "conv");
	ret &= nn_convLayer_export(self->conv, stream);
	ret &= jsmn_stream_key(stream, "%s", "fact");
	ret &= nn_factLayer_export(self->fact, stream);
	ret &= jsmn_stream_end(stream);

	return ret;
}

/***********************************************************
* private - nn_coderLayer_t                                *
***********************************************************/

static nn_tensor_t*
nn_coderLayer_forwardPassFn(nn_layer_t* base, int flags,
                            uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_coderLayer_t* self = (nn_coderLayer_t*) base;

	if(self->pre_op)
	{
		X = nn_layer_forwardPass(&self->pre_op->base, flags,
		                         bs, X);
		if(X == NULL)
		{
			return NULL;
		}
	}

	if(self->conv)
	{
		X = nn_layer_forwardPass(&self->conv->base, flags,
		                         bs, X);
		if(X == NULL)
		{
			return NULL;
		}
	}

	if(self->skip)
	{
		X = nn_layer_forwardPass(&self->skip->base, flags,
		                         bs, X);
		if(X == NULL)
		{
			return NULL;
		}
	}

	if(self->bn)
	{
		X = nn_layer_forwardPass(&self->bn->base, flags,
		                         bs, X);
		if(X == NULL)
		{
			return NULL;
		}
	}

	if(self->fact)
	{
		X = nn_layer_forwardPass(&self->fact->base, flags,
		                         bs, X);
		if(X == NULL)
		{
			return NULL;
		}
	}

	if(self->repeater)
	{
		cc_listIter_t* iter;
		iter = cc_list_head(self->repeater);
		while(iter)
		{
			nn_coderRepeaterLayer_t* r;
			r = (nn_coderRepeaterLayer_t*)
			    cc_list_peekIter(iter);

			X = nn_layer_forwardPass(&r->base, flags, bs, X);
			if(X == NULL)
			{
				return NULL;
			}

			iter = cc_list_next(iter);
		}
	}

	if(self->post_op)
	{
		X = nn_layer_forwardPass(&self->post_op->base, flags,
		                         bs, X);
		if(X == NULL)
		{
			return NULL;
		}
	}

	return X;
}

static nn_tensor_t*
nn_coderLayer_backpropFn(nn_layer_t* base, int flags,
                         uint32_t bs, nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,xh,xw,xd)

	nn_coderLayer_t* self = (nn_coderLayer_t*) base;

	if(self->post_op)
	{
		dL_dY = nn_layer_backprop(&self->post_op->base, flags,
		                          bs, dL_dY);
		if(dL_dY == NULL)
		{
			return NULL;
		}
	}

	if(self->repeater)
	{
		cc_listIter_t* iter;
		iter = cc_list_tail(self->repeater);
		while(iter)
		{
			nn_coderRepeaterLayer_t* r;
			r = (nn_coderRepeaterLayer_t*)
			    cc_list_peekIter(iter);

			dL_dY = nn_layer_backprop(&r->base, flags, bs, dL_dY);
			if(dL_dY == NULL)
			{
				return NULL;
			}

			iter = cc_list_prev(iter);
		}
	}

	if(self->fact)
	{
		dL_dY = nn_layer_backprop(&self->fact->base, flags,
		                          bs, dL_dY);
		if(dL_dY == NULL)
		{
			return NULL;
		}
	}

	if(self->bn)
	{
		dL_dY = nn_layer_backprop(&self->bn->base, flags,
		                          bs, dL_dY);
		if(dL_dY == NULL)
		{
			return NULL;
		}
	}

	if(self->skip)
	{
		dL_dY = nn_layer_backprop(&self->skip->base, flags,
		                          bs, dL_dY);
		if(dL_dY == NULL)
		{
			return NULL;
		}
	}

	if(self->conv)
	{
		dL_dY = nn_layer_backprop(&self->conv->base, flags,
		                          bs, dL_dY);
		if(dL_dY == NULL)
		{
			return NULL;
		}
	}

	if(self->pre_op)
	{
		dL_dY = nn_layer_backprop(&self->pre_op->base, flags,
		                          bs, dL_dY);
		if(dL_dY == NULL)
		{
			return NULL;
		}
	}

	return dL_dY;
}

static void
nn_coderLayer_postFn(nn_layer_t* base, int flags)
{
	ASSERT(base);

	nn_coderLayer_t* self = (nn_coderLayer_t*) base;

	if(self->pre_op)
	{
		nn_layer_post(&self->pre_op->base, flags);
	}

	if(self->conv)
	{
		nn_layer_post(&self->conv->base, flags);
	}

	if(self->skip)
	{
		nn_layer_post(&self->skip->base, flags);
	}

	if(self->bn)
	{
		nn_layer_post(&self->bn->base, flags);
	}

	if(self->fact)
	{
		nn_layer_post(&self->fact->base, flags);
	}

	cc_listIter_t* iter;
	if(self->repeater)
	{
		iter = cc_list_head(self->repeater);
		while(iter)
		{
			nn_layer_t* layer;
			layer = (nn_layer_t*) cc_list_peekIter(iter);

			nn_layer_post(layer, flags);

			iter = cc_list_next(iter);
		}
	}

	if(self->post_op)
	{
		nn_layer_post(&self->post_op->base, flags);
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

static void
nn_coderLayer_deleteRepeater(nn_coderLayer_t* self)
{
	ASSERT(self);

	if(self->repeater == NULL)
	{
		return;
	}

	cc_listIter_t* iter;
	iter = cc_list_head(self->repeater);
	while(iter)
	{
		nn_coderRepeaterLayer_t* r;
		r = (nn_coderRepeaterLayer_t*)
		    cc_list_remove(self->repeater, &iter);
		nn_coderRepeaterLayer_delete(&r);
	}
	cc_list_delete(&self->repeater);
}

static int
nn_coderLayer_newRepeater(nn_coderLayer_t* self,
                          uint32_t repeat, nn_dim_t* dimX)
{
	ASSERT(self);
	ASSERT(dimX);

	nn_layer_t* base = &self->base;

	// check if repeater is used
	if(repeat == 0)
	{
		return 1;
	}

	self->repeater = cc_list_new();
	if(self->repeater == NULL)
	{
		return 0;
	}

	int i;
	nn_coderRepeaterLayer_t* r = NULL;
	for(i = 0; i < repeat; ++i)
	{
		r = nn_coderRepeaterLayer_new(base->arch, dimX);
		if(r == NULL)
		{
			goto fail_repeater;
		}

		if(cc_list_append(self->repeater, NULL, r) == NULL)
		{
			goto fail_append;
		}
	}

	// success
	return 1;

	// failure
	fail_append:
		nn_coderRepeaterLayer_delete(&r);
	fail_repeater:
		nn_coderLayer_deleteRepeater(self);
	return 0;
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_coderLayer_t*
nn_coderLayer_new(nn_coderLayerInfo_t* info)
{
	ASSERT(info);

	nn_layerInfo_t base_info =
	{
		.arch            = info->arch,
		.forward_pass_fn = nn_coderLayer_forwardPassFn,
		.backprop_fn     = nn_coderLayer_backpropFn,
		.post_fn         = nn_coderLayer_postFn,
		.dimX_fn         = nn_coderLayer_dimXFn,
		.dimY_fn         = nn_coderLayer_dimYFn,
	};

	nn_coderLayer_t* self;
	self = (nn_coderLayer_t*)
	       nn_layer_new(sizeof(nn_coderLayer_t),
	                    &base_info);
	if(self == NULL)
	{
		return NULL;
	}

	nn_dim_t* dim = info->dimX;
	nn_dim_copy(dim, &self->dimX);

	if(info->pre_op_mode != NN_CODER_OP_MODE_NONE)
	{
		self->pre_op = nn_coderOpLayer_new(info->arch, dim,
		                                   info->pre_op_mode);
		if(self->pre_op == NULL)
		{
			goto fail_pre_op;
		}
		dim = nn_layer_dimY(&self->pre_op->base);
	}

	if(info->conv_mode == NN_CODER_CONV_MODE_3X3_RELU)
	{
		uint32_t xd = dim->depth;

		nn_dim_t dimW =
		{
			.count  = info->fc,
			.width  = 3,
			.height = 3,
			.depth  = xd,
		};

		int flags = NN_CONV_LAYER_FLAG_HE;
		if(info->bn_mode > NN_CODER_BATCH_NORM_MODE_NONE)
		{
			flags |= NN_CONV_LAYER_FLAG_DISABLE_BIAS;
		}

		self->conv = nn_convLayer_new(info->arch, dim, &dimW, 1,
		                              flags);
		if(self->conv == NULL)
		{
			goto fail_conv;
		}
		dim = nn_layer_dimY(&self->conv->base);
	}

	if(info->skip_mode != NN_CODER_SKIP_MODE_NONE)
	{
		if(info->skip_mode == NN_CODER_SKIP_MODE_FORK)
		{
			self->skip = nn_skipLayer_newFork(info->arch, dim);
		}
		else if(info->skip_mode == NN_CODER_SKIP_MODE_ADD)
		{
			ASSERT(info->skip_coder);
			self->skip = nn_skipLayer_newAdd(info->arch, dim,
			                                 info->skip_coder->skip);
		}
		else if(info->skip_mode == NN_CODER_SKIP_MODE_CAT)
		{
			ASSERT(info->skip_coder);
			self->skip = nn_skipLayer_newCat(info->arch, dim,
			                                 info->skip_coder->skip);
		}

		if(self->skip == NULL)
		{
			LOGE("invalid");
			goto fail_skip;
		}
		dim = nn_layer_dimY(&self->skip->base);
	}

	if(info->bn_mode > NN_CODER_BATCH_NORM_MODE_NONE)
	{
		self->bn = nn_batchNormLayer_new(info->arch,
		                                 info->bn_mode, dim);
		if(self->bn == NULL)
		{
			goto fail_bn;
		}
	}

	if(info->conv_mode == NN_CODER_CONV_MODE_3X3_RELU)
	{
		self->fact = nn_factLayer_new(info->arch, dim,
		                              NN_FACT_LAYER_FN_RELU);
		if(self->fact == NULL)
		{
			goto fail_fact;
		}
	}

	if(info->repeat_mode == NN_CODER_CONV_MODE_3X3_RELU)
	{
		if(nn_coderLayer_newRepeater(self, info->repeat,
		                             dim) == 0)
		{
			goto fail_repeater;
		}
	}

	if(info->post_op_mode != NN_CODER_OP_MODE_NONE)
	{
		self->post_op = nn_coderOpLayer_new(info->arch, dim,
		                                    info->post_op_mode);
		if(self->post_op == NULL)
		{
			goto fail_post_op;
		}
		dim = nn_layer_dimY(&self->post_op->base);
	}

	nn_dim_copy(dim, &self->dimY);

	// success
	return self;

	// failure
	fail_post_op:
		nn_coderLayer_deleteRepeater(self);
	fail_repeater:
		nn_factLayer_delete(&self->fact);
	fail_fact:
		nn_batchNormLayer_delete(&self->bn);
	fail_bn:
		nn_skipLayer_delete(&self->skip);
	fail_skip:
		nn_convLayer_delete(&self->conv);
	fail_conv:
		nn_coderOpLayer_delete(&self->pre_op);
	fail_pre_op:
		nn_layer_delete((nn_layer_t**) &self);
	return NULL;
}

void nn_coderLayer_delete(nn_coderLayer_t** _self)
{
	ASSERT(_self);

	nn_coderLayer_t* self = *_self;
	if(self)
	{
		nn_coderOpLayer_delete(&self->post_op);
		nn_coderLayer_deleteRepeater(self);
		nn_factLayer_delete(&self->fact);
		nn_batchNormLayer_delete(&self->bn);
		nn_skipLayer_delete(&self->skip);
		nn_convLayer_delete(&self->conv);
		nn_coderOpLayer_delete(&self->pre_op);
		nn_layer_delete((nn_layer_t**) _self);
	}
}

nn_coderLayer_t*
nn_coderLayer_import(nn_arch_t* arch, jsmn_val_t* val,
                     nn_coderLayer_t* skip_coder)
{
	// skip_coder is optional
	ASSERT(arch);
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	cc_list_t* repeater = cc_list_new();
	if(repeater == NULL)
	{
		return NULL;
	}

	jsmn_val_t* val_dimX    = NULL;
	jsmn_val_t* val_dimY    = NULL;
	jsmn_val_t* val_pre_op  = NULL;
	jsmn_val_t* val_conv    = NULL;
	jsmn_val_t* val_skip    = NULL;
	jsmn_val_t* val_bn      = NULL;
	jsmn_val_t* val_fact    = NULL;
	jsmn_val_t* val_post_op = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		jsmn_keyval_t* kv;
		kv = (jsmn_keyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == JSMN_TYPE_OBJECT)
		{
			if(strcmp(kv->key, "dimX") == 0)
			{
				val_dimX = kv->val;
			}
			else if(strcmp(kv->key, "dimY") == 0)
			{
				val_dimY = kv->val;
			}
			else if(strcmp(kv->key, "pre_op") == 0)
			{
				val_pre_op = kv->val;
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
			else if(strcmp(kv->key, "repeater") == 0)
			{
				if(cc_list_append(repeater, NULL,
				                  kv->val) == NULL)
				{
					goto fail_append_val;
				}
			}
			else if(strcmp(kv->key, "post_op") == 0)
			{
				val_post_op = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	// layers are optional
	if((val_dimX == NULL) || (val_dimY == NULL))
	{
		LOGE("invalid");
		goto fail_check;
	}

	nn_layerInfo_t info =
	{
		.arch            = arch,
		.forward_pass_fn = nn_coderLayer_forwardPassFn,
		.backprop_fn     = nn_coderLayer_backpropFn,
		.post_fn         = nn_coderLayer_postFn,
		.dimX_fn         = nn_coderLayer_dimXFn,
		.dimY_fn         = nn_coderLayer_dimYFn,
	};

	nn_coderLayer_t*  self;
	self = (nn_coderLayer_t*)
	       nn_layer_new(sizeof(nn_coderLayer_t),
	                    &info);
	if(self == NULL)
	{
		goto fail_layer;
	}

	if(nn_dim_load(&self->dimX, val_dimX) == 0)
	{
		goto fail_dimX;
	}

	if(nn_dim_load(&self->dimY, val_dimY) == 0)
	{
		goto fail_dimY;
	}

	if(val_pre_op)
	{
		self->pre_op = nn_coderOpLayer_import(arch, val_pre_op);
		if(self->pre_op == NULL)
		{
			goto fail_pre_op;
		}
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

	if(cc_list_size(repeater) > 0)
	{
		self->repeater = cc_list_new();
		if(self->repeater == NULL)
		{
			goto fail_new_repeater;
		}

		iter = cc_list_head(repeater);
		while(iter)
		{
			jsmn_val_t* val_rpt;
			val_rpt = (jsmn_val_t*)
			          cc_list_peekIter(iter);

			nn_coderRepeaterLayer_t* r;
			r = nn_coderRepeaterLayer_import(arch, val_rpt);
			if(r == NULL)
			{
				goto fail_add_repeater;
			}

			if(cc_list_append(self->repeater, NULL, r) == NULL)
			{
				nn_coderRepeaterLayer_delete(&r);
				goto fail_add_repeater;
			}

			iter = cc_list_next(iter);
		}
	}

	if(val_post_op)
	{
		self->post_op = nn_coderOpLayer_import(arch, val_post_op);
		if(self->post_op == NULL)
		{
			goto fail_post_op;
		}
	}

	cc_list_discard(repeater);
	cc_list_delete(&repeater);

	// success
	return self;

	// failure
	fail_post_op:
	fail_add_repeater:
	{
		iter = cc_list_head(self->repeater);
		while(iter)
		{
			nn_coderRepeaterLayer_t* r;
			r = (nn_coderRepeaterLayer_t*)
			    cc_list_remove(self->repeater, &iter);
			nn_coderRepeaterLayer_delete(&r);
		}
		cc_list_delete(&self->repeater);
	}
	fail_new_repeater:
		nn_factLayer_delete(&self->fact);
	fail_fact:
		nn_batchNormLayer_delete(&self->bn);
	fail_bn:
		nn_skipLayer_delete(&self->skip);
	fail_skip:
		nn_convLayer_delete(&self->conv);
	fail_conv:
		nn_coderOpLayer_delete(&self->pre_op);
	fail_pre_op:
	fail_dimY:
	fail_dimX:
		nn_layer_delete((nn_layer_t**) &self);
	fail_layer:
	fail_check:
	fail_append_val:
	{
		cc_list_discard(repeater);
		cc_list_delete(&repeater);
	}
	return NULL;
}

int nn_coderLayer_export(nn_coderLayer_t* self,
                         jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	ret &= jsmn_stream_key(stream, "%s", "dimX");
	ret &= nn_dim_store(&self->dimX, stream);
	ret &= jsmn_stream_key(stream, "%s", "dimY");
	ret &= nn_dim_store(&self->dimY, stream);

	if(self->pre_op)
	{
		ret &= jsmn_stream_key(stream, "%s", "pre_op");
		ret &= nn_coderOpLayer_export(self->pre_op, stream);
	}

	if(self->conv)
	{
		ret &= jsmn_stream_key(stream, "%s", "conv");
		ret &= nn_convLayer_export(self->conv, stream);
	}

	if(self->skip)
	{
		ret &= jsmn_stream_key(stream, "%s", "skip");
		ret &= nn_skipLayer_export(self->skip, stream);
	}

	if(self->bn)
	{
		ret &= jsmn_stream_key(stream, "%s", "bn");
		ret &= nn_batchNormLayer_export(self->bn, stream);
	}

	if(self->fact)
	{
		ret &= jsmn_stream_key(stream, "%s", "fact");
		ret &= nn_factLayer_export(self->fact, stream);
	}

	if(self->repeater)
	{
		cc_listIter_t* iter;
		iter = cc_list_head(self->repeater);
		while(iter)
		{
			nn_coderRepeaterLayer_t* r;
			r = (nn_coderRepeaterLayer_t*)
			    cc_list_peekIter(iter);
			ret &= jsmn_stream_key(stream, "%s", "repeater");
			ret &= nn_coderRepeaterLayer_export(r, stream);
			iter = cc_list_next(iter);
		}
	}

	if(self->post_op)
	{
		ret &= jsmn_stream_key(stream, "%s", "post_op");
		ret &= nn_coderOpLayer_export(self->post_op, stream);
	}
	ret &= jsmn_stream_end(stream);

	return ret;
}

int nn_coderLayer_lerp(nn_coderLayer_t* self,
                       nn_coderLayer_t* lerp,
                       float s1, float s2)
{
	ASSERT(self);
	ASSERT(lerp);

	int ret = 1;
	ret &= nn_factLayer_lerp(self->fact, lerp->fact, s1, s2);
	ret &= nn_factLayer_lerp(lerp->fact, self->fact, s2, s1);
	return ret;
}
