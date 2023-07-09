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

typedef struct nn_coderOpLayer_s
{
	nn_layer_t base;

	int op_mode;
	union
	{
		// upscale layer
		// transpose, same, xavier, stride=2
		// W : dim(xd,2,2,xd)
		// Y : dim(bs,2*xh,2*xw,xd)
		//
		// downscale layer
		// same, xavier, stride=2
		// W : dim(xd,3,3,xd)
		// Y : dim(bs,xh/2,xw/2,xd)
		nn_convLayer_t* conv;

		// pooling layer
		// 2x2, max or avg
		// Y : dim(bs,xh/2,xw/2,xd)
		nn_poolingLayer_t* pool;
	};
} nn_coderOpLayer_t;

static nn_tensor_t*
nn_coderOpLayer_forwardPassFn(nn_layer_t* base,
                              int mode,
                              uint32_t bs,
                              nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_coderOpLayer_t* self;
	self = (nn_coderOpLayer_t*) base;

	if((self->op_mode == NN_CODER_OP_MODE_UPSCALE) ||
	   (self->op_mode == NN_CODER_OP_MODE_DOWNSCALE))
	{
		return nn_layer_forwardPass(&self->conv->base,
		                            mode, bs, X);
	}
	else
	{
		return nn_layer_forwardPass(&self->pool->base,
		                            mode, bs, X);
	}
}

static nn_tensor_t*
nn_coderOpLayer_backpropFn(nn_layer_t* base,
                           uint32_t bs,
                           nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,xh,xw,xd)

	nn_coderOpLayer_t* self;
	self = (nn_coderOpLayer_t*) base;

	if((self->op_mode == NN_CODER_OP_MODE_UPSCALE) ||
	   (self->op_mode == NN_CODER_OP_MODE_DOWNSCALE))
	{
		return nn_layer_backprop(&self->conv->base, bs, dL_dY);
	}
	else
	{
		return nn_layer_backprop(&self->pool->base, bs, dL_dY);
	}
}

static nn_dim_t*
nn_coderOpLayer_dimXFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_coderOpLayer_t* self;
	self = (nn_coderOpLayer_t*) base;

	if((self->op_mode == NN_CODER_OP_MODE_UPSCALE) ||
	   (self->op_mode == NN_CODER_OP_MODE_DOWNSCALE))
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

	if((self->op_mode == NN_CODER_OP_MODE_UPSCALE) ||
	   (self->op_mode == NN_CODER_OP_MODE_DOWNSCALE))
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
                    int op_mode)
{
	ASSERT(arch);
	ASSERT(dimX);

	nn_layerInfo_t info =
	{
		.arch            = arch,
		.forward_pass_fn = nn_coderOpLayer_forwardPassFn,
		.backprop_fn     = nn_coderOpLayer_backpropFn,
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
	if(op_mode == NN_CODER_OP_MODE_UPSCALE)
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
		                              NN_CONV_LAYER_FLAG_PAD_SAME  |
		                              NN_CONV_LAYER_FLAG_XAVIER);
		if(self->conv == NULL)
		{
			goto fail_op;
		}
	}
	else if(op_mode == NN_CODER_OP_MODE_DOWNSCALE)
	{
		nn_dim_t dimW =
		{
			.count  = xd,
			.width  = 3,
			.height = 3,
			.depth  = xd,
		};

		self->conv = nn_convLayer_new(arch, dimX, &dimW, 2,
		                              NN_CONV_LAYER_FLAG_PAD_SAME  |
		                              NN_CONV_LAYER_FLAG_XAVIER);
		if(self->conv == NULL)
		{
			goto fail_op;
		}
	}
	else if(op_mode == NN_CODER_OP_MODE_POOLAVG)
	{
		self->pool = nn_poolingLayer_new(arch, dimX, 2, 2,
		                                 NN_POOLING_LAYER_MODE_AVERAGE);
		if(self->pool == NULL)
		{
			goto fail_op;
		}
	}
	else if(op_mode == NN_CODER_OP_MODE_POOLMAX)
	{
		self->pool = nn_poolingLayer_new(arch, dimX, 2, 2,
		                                 NN_POOLING_LAYER_MODE_MAX);
		if(self->pool == NULL)
		{
			goto fail_op;
		}
	}
	else
	{
		LOGE("invalid op_mode=%i", op_mode);
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
		if((self->op_mode == NN_CODER_OP_MODE_UPSCALE) ||
		   (self->op_mode == NN_CODER_OP_MODE_DOWNSCALE))
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

	if(strcmp(val_op_mode->data, "upscale") == 0)
	{
		self->op_mode = NN_CODER_OP_MODE_UPSCALE;
	}
	else if(strcmp(val_op_mode->data, "downscale") == 0)
	{
		self->op_mode = NN_CODER_OP_MODE_DOWNSCALE;
	}
	else if(strcmp(val_op_mode->data, "poolavg") == 0)
	{
		self->op_mode = NN_CODER_OP_MODE_POOLAVG;
	}
	else if(strcmp(val_op_mode->data, "poolmax") == 0)
	{
		self->op_mode = NN_CODER_OP_MODE_POOLMAX;
	}

	if(val_conv &&
	   ((self->op_mode == NN_CODER_OP_MODE_UPSCALE) ||
	    (self->op_mode == NN_CODER_OP_MODE_DOWNSCALE)))
	{
		self->conv = nn_convLayer_import(arch, val_conv);
		if(self->conv == NULL)
		{
			goto fail_op;
		}
	}
	else if(val_pool &&
	        ((self->op_mode == NN_CODER_OP_MODE_POOLAVG) ||
	         (self->op_mode == NN_CODER_OP_MODE_POOLMAX)))
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
	if(self->op_mode == NN_CODER_OP_MODE_UPSCALE)
	{
		ret &= jsmn_stream_key(stream, "%s", "op_mode");
		ret &= jsmn_stream_string(stream, "%s", "upscale");
		ret &= jsmn_stream_key(stream, "%s", "conv");
		ret &= nn_convLayer_export(self->conv, stream);
	}
	else if(self->op_mode == NN_CODER_OP_MODE_DOWNSCALE)
	{
		ret &= jsmn_stream_key(stream, "%s", "op_mode");
		ret &= jsmn_stream_string(stream, "%s", "downscale");
		ret &= jsmn_stream_key(stream, "%s", "conv");
		ret &= nn_convLayer_export(self->conv, stream);
	}
	else if(self->op_mode == NN_CODER_OP_MODE_POOLAVG)
	{
		ret &= jsmn_stream_key(stream, "%s", "op_mode");
		ret &= jsmn_stream_string(stream, "%s", "poolavg");
		ret &= jsmn_stream_key(stream, "%s", "pool");
		ret &= nn_poolingLayer_export(self->pool, stream);
	}
	else if(self->op_mode == NN_CODER_OP_MODE_POOLMAX)
	{
		ret &= jsmn_stream_key(stream, "%s", "op_mode");
		ret &= jsmn_stream_string(stream, "%s", "poolmax");
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
	// same, he, relu
	// xd : fc
	// W  : dim(xd,3,3,xd)
	// Y  : dim(bs,xh,xw,xd)
	nn_convLayer_t* conv;
	nn_factLayer_t* fact;
} nn_coderRepeaterLayer_t;

static nn_tensor_t*
nn_coderRepeaterLayer_forwardPassFn(nn_layer_t* base,
                                    int mode,
                                    uint32_t bs,
                                    nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_coderRepeaterLayer_t* self;
	self = (nn_coderRepeaterLayer_t*) base;

	X = nn_layer_forwardPass(&self->conv->base, mode, bs, X);
	if(X == NULL)
	{
		return NULL;
	}

	return nn_layer_forwardPass(&self->fact->base,
	                            mode, bs, X);
}

static nn_tensor_t*
nn_coderRepeaterLayer_backpropFn(nn_layer_t* base,
                                 uint32_t bs,
                                 nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,xh,xw,xd)

	nn_coderRepeaterLayer_t* self;
	self = (nn_coderRepeaterLayer_t*) base;

	dL_dY = nn_layer_backprop(&self->fact->base, bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	return nn_layer_backprop(&self->conv->base, bs, dL_dY);
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
	                              NN_CONV_LAYER_FLAG_PAD_SAME |
	                              NN_CONV_LAYER_FLAG_HE);
	if(self->conv == NULL)
	{
		goto fail_conv;
	}
	dim = nn_layer_dimY(&self->conv->base);

	self->fact = nn_factLayer_new(arch, dim,
	                              nn_factLayer_ReLU,
	                              nn_factLayer_dReLU);
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
nn_coderLayer_forwardPassFn(nn_layer_t* base,
                            int mode,
                            uint32_t bs,
                            nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_coderLayer_t* self;
	self = (nn_coderLayer_t*) base;

	X = nn_layer_forwardPass(&self->conv->base, mode, bs, X);
	if(X == NULL)
	{
		return NULL;
	}

	if(self->skip)
	{
		X = nn_layer_forwardPass(&self->skip->base, mode, bs, X);
		if(X == NULL)
		{
			return NULL;
		}
	}

	X = nn_layer_forwardPass(&self->bn->base, mode, bs, X);
	if(X == NULL)
	{
		return NULL;
	}

	X = nn_layer_forwardPass(&self->fact->base, mode, bs, X);
	if(X == NULL)
	{
		return NULL;
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

			X = nn_layer_forwardPass(&r->base, mode, bs, X);
			if(X == NULL)
			{
				return NULL;
			}

			iter = cc_list_next(iter);
		}
	}

	if(self->op)
	{
		X = nn_layer_forwardPass(&self->op->base, mode, bs, X);
		if(X == NULL)
		{
			return NULL;
		}
	}

	return X;
}

static nn_tensor_t*
nn_coderLayer_backpropFn(nn_layer_t* base,
                         uint32_t bs,
                         nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,xh,xw,xd)

	nn_coderLayer_t* self;
	self = (nn_coderLayer_t*) base;

	if(self->op)
	{
		dL_dY = nn_layer_backprop(&self->op->base, bs, dL_dY);
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

			dL_dY = nn_layer_backprop(&r->base, bs, dL_dY);
			if(dL_dY == NULL)
			{
				return NULL;
			}

			iter = cc_list_prev(iter);
		}
	}

	dL_dY = nn_layer_backprop(&self->fact->base, bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	dL_dY = nn_layer_backprop(&self->bn->base, bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	if(self->skip)
	{
		dL_dY = nn_layer_backprop(&self->skip->base, bs, dL_dY);
		if(dL_dY == NULL)
		{
			return NULL;
		}
	}

	dL_dY = nn_layer_backprop(&self->conv->base, bs, dL_dY);
	if(dL_dY == NULL)
	{
		return NULL;
	}

	return dL_dY;
}

static nn_dim_t*
nn_coderLayer_dimXFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_coderLayer_t* self;
	self = (nn_coderLayer_t*) base;

	return nn_layer_dimX(&self->conv->base);
}

static nn_dim_t*
nn_coderLayer_dimYFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_coderLayer_t* self;
	self = (nn_coderLayer_t*) base;

	if(self->op)
	{
		return nn_layer_dimY(&self->op->base);
	}

	// repeater layers do not affect dimY
	return nn_layer_dimY(&self->fact->base);
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

static int
nn_coderLayer_newOp(nn_coderLayer_t* self, int op_mode,
                    nn_dim_t* dimX)
{
	ASSERT(self);
	ASSERT(dimX);

	nn_layer_t* base = &self->base;

	// check if op is used
	if(op_mode == NN_CODER_OP_MODE_NONE)
	{
		return 1;
	}

	self->op = nn_coderOpLayer_new(base->arch, dimX, op_mode);
	if(self->op == NULL)
	{
		return 0;
	}

	return 1;
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

	uint32_t xd = info->dimX->depth;

	nn_dim_t dimW =
	{
		.count  = info->fc,
		.width  = 3,
		.height = 3,
		.depth  = xd,
	};

	nn_dim_t* dim = info->dimX;

	self->conv = nn_convLayer_new(info->arch, dim, &dimW, 1,
	                              NN_CONV_LAYER_FLAG_DISABLE_BIAS |
	                              NN_CONV_LAYER_FLAG_PAD_SAME     |
	                              NN_CONV_LAYER_FLAG_HE);
	if(self->conv == NULL)
	{
		goto fail_conv;
	}
	dim = nn_layer_dimY(&self->conv->base);

	if(info->skip_enable)
	{
		if(info->skip_mode == NN_SKIP_LAYER_MODE_FORK)
		{
			self->skip = nn_skipLayer_newFork(info->arch, dim);
		}
		else if(info->skip_mode == NN_SKIP_LAYER_MODE_ADD)
		{
			self->skip = nn_skipLayer_newAdd(info->arch, dim,
			                                 info->skip_coder->skip);
		}
		else if(info->skip_mode == NN_SKIP_LAYER_MODE_CAT)
		{
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

	self->bn = nn_batchNormLayer_new(info->arch, dim);
	if(self->bn == NULL)
	{
		goto fail_bn;
	}

	self->fact = nn_factLayer_new(info->arch, dim,
	                              nn_factLayer_ReLU,
	                              nn_factLayer_dReLU);
	if(self->fact == NULL)
	{
		goto fail_fact;
	}

	if(nn_coderLayer_newRepeater(self, info->repeat,
	                             dim) == 0)
	{
		goto fail_repeater;
	}

	if(nn_coderLayer_newOp(self, info->op_mode,
	                       dim) == 0)
	{
		goto fail_op;
	}

	// success
	return self;

	// failure
	fail_op:
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
		nn_layer_delete((nn_layer_t**) &self);
	return NULL;
}

void nn_coderLayer_delete(nn_coderLayer_t** _self)
{
	ASSERT(_self);

	nn_coderLayer_t* self = *_self;
	if(self)
	{
		nn_coderOpLayer_delete(&self->op);
		nn_coderLayer_deleteRepeater(self);
		nn_factLayer_delete(&self->fact);
		nn_batchNormLayer_delete(&self->bn);
		nn_skipLayer_delete(&self->skip);
		nn_convLayer_delete(&self->conv);
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

	jsmn_val_t* val_conv = NULL;
	jsmn_val_t* val_skip = NULL;
	jsmn_val_t* val_bn   = NULL;
	jsmn_val_t* val_fact = NULL;
	jsmn_val_t* val_op   = NULL;

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
			else if(strcmp(kv->key, "op") == 0)
			{
				val_op = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	// skip, repeater and op are optional
	if((val_conv == NULL) ||
	   (val_bn   == NULL) ||
	   (val_fact == NULL))
	{
		LOGE("invalid");
		goto fail_check;
	}

	nn_layerInfo_t info =
	{
		.arch            = arch,
		.forward_pass_fn = nn_coderLayer_forwardPassFn,
		.backprop_fn     = nn_coderLayer_backpropFn,
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

	self->conv = nn_convLayer_import(arch, val_conv);
	if(self->conv == NULL)
	{
		goto fail_conv;
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

	self->bn = nn_batchNormLayer_import(arch, val_bn);
	if(self->bn == NULL)
	{
		goto fail_bn;
	}

	self->fact = nn_factLayer_import(arch, val_fact);
	if(self->fact == NULL)
	{
		goto fail_fact;
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

	if(val_op)
	{
		self->op = nn_coderOpLayer_import(arch, val_op);
		if(self->op == NULL)
		{
			goto fail_op;
		}
	}

	cc_list_discard(repeater);
	cc_list_delete(&repeater);

	// success
	return self;

	// failure
	fail_op:
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
	ret &= jsmn_stream_key(stream, "%s", "conv");
	ret &= nn_convLayer_export(self->conv, stream);
	if(self->skip)
	{
		ret &= jsmn_stream_key(stream, "%s", "skip");
		ret &= nn_skipLayer_export(self->skip, stream);
	}
	ret &= jsmn_stream_key(stream, "%s", "bn");
	ret &= nn_batchNormLayer_export(self->bn, stream);
	ret &= jsmn_stream_key(stream, "%s", "fact");
	ret &= nn_factLayer_export(self->fact, stream);

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

	if(self->op)
	{
		ret &= jsmn_stream_key(stream, "%s", "op");
		ret &= nn_coderOpLayer_export(self->op, stream);
	}
	ret &= jsmn_stream_end(stream);

	return ret;
}
