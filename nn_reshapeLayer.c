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
#include "nn_arch.h"
#include "nn_reshapeLayer.h"
#include "nn_tensor.h"

/***********************************************************
* private                                                  *
***********************************************************/

static nn_tensor_t*
nn_reshapeLayer_forwardPassFn(nn_layer_t* base,
                              nn_layerMode_e layer_mode,
                              uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_reshapeLayer_t* self = (nn_reshapeLayer_t*) base;
	nn_tensor_t*       Y    = &self->Y;

	if(nn_dim_equals(nn_tensor_dim(X), &self->dimX) == 0)
	{
		LOGE("invalid");
		return NULL;
	}

	Y->data    = X->data;
	Y->sb_data = X->sb_data;

	return Y;
}

static nn_tensor_t*
nn_reshapeLayer_backpropFn(nn_layer_t* base,
                           nn_layerMode_e layer_mode,
                           uint32_t bs,
                           nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY);

	return dL_dY;
}

static nn_dim_t*
nn_reshapeLayer_dimXFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_reshapeLayer_t* self = (nn_reshapeLayer_t*) base;

	return &self->dimX;
}

static nn_dim_t*
nn_reshapeLayer_dimYFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_reshapeLayer_t* self = (nn_reshapeLayer_t*) base;

	return nn_tensor_dim(&self->Y);
}

static int
nn_reshapeLayer_newCompute(nn_reshapeLayer_t* self,
                           nn_dim_t* dimY)
{
	ASSERT(self);

	nn_arch_t*   arch = self->base.arch;
	nn_tensor_t* Y    = &self->Y;

	vkk_updateMode_e um;
	um = vkk_compute_updateMode(arch->compute);

	Y->sb_dim = vkk_buffer_new(arch->engine, um,
	                           VKK_BUFFER_USAGE_STORAGE,
	                           sizeof(nn_dim_t),
	                           dimY);
	if(Y->sb_dim == NULL)
	{
		return 0;
	}

	return 1;
}

static void
nn_reshapeLayer_deleteCompute(nn_reshapeLayer_t* self)
{
	ASSERT(self);

	nn_tensor_t* Y = &self->Y;
	vkk_buffer_delete(&Y->sb_dim);
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_reshapeLayer_t*
nn_reshapeLayer_new(nn_arch_t* arch, nn_dim_t* dimX,
                    nn_dim_t* dimY)
{
	ASSERT(arch);
	ASSERT(dimX);
	ASSERT(dimY);

	size_t sizeX = nn_dim_sizeof(dimX);
	size_t sizeY = nn_dim_sizeof(dimY);
	if(sizeY > sizeX)
	{
		LOGE("invalid sizeX=%u, sizeY=%u",
		     (uint32_t) sizeX, (uint32_t) sizeY);
		return NULL;
	}

	nn_layerInfo_t info =
	{
		.arch            = arch,
		.forward_pass_fn = nn_reshapeLayer_forwardPassFn,
		.backprop_fn     = nn_reshapeLayer_backpropFn,
		.dimX_fn         = nn_reshapeLayer_dimXFn,
		.dimY_fn         = nn_reshapeLayer_dimYFn,
	};

	nn_reshapeLayer_t* self;
	self = (nn_reshapeLayer_t*)
	       nn_layer_new(sizeof(nn_reshapeLayer_t), &info);
	if(self == NULL)
	{
		return NULL;
	}

	nn_tensor_t* Y = &self->Y;
	nn_dim_copy(dimX, &self->dimX);
	nn_dim_copy(dimY, nn_tensor_dim(Y));

	if(nn_reshapeLayer_newCompute(self, dimY) == 0)
	{
		goto fail_compute;
	}

	// success
	return self;

	// failure
	fail_compute:
		nn_layer_delete((nn_layer_t**) &self);
	return NULL;
}

nn_reshapeLayer_t*
nn_reshapeLayer_import(nn_arch_t* arch, jsmn_val_t* val)
{
	ASSERT(arch);
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	jsmn_val_t* val_dimX = NULL;
	jsmn_val_t* val_dimY = NULL;

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
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_dimX == NULL) || (val_dimY == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_dim_t dimX;
	if(nn_dim_load(&dimX, val_dimX) == 0)
	{
		return NULL;
	}

	nn_dim_t dimY;
	if(nn_dim_load(&dimY, val_dimY) == 0)
	{
		return NULL;
	}

	return nn_reshapeLayer_new(arch, &dimX, &dimY);
}

int nn_reshapeLayer_export(nn_reshapeLayer_t* self,
                           jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	nn_dim_t* dimX = &self->dimX;
	nn_dim_t* dimY = nn_tensor_dim(&self->Y);

	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	ret &= jsmn_stream_key(stream, "%s", "dimX");
	ret &= nn_dim_store(dimX, stream);
	ret &= jsmn_stream_key(stream, "%s", "dimY");
	ret &= nn_dim_store(dimY, stream);
	ret &= jsmn_stream_end(stream);

	return ret;
}

void nn_reshapeLayer_delete(nn_reshapeLayer_t** _self)
{
	ASSERT(_self);

	nn_reshapeLayer_t* self = *_self;
	if(self)
	{
		nn_reshapeLayer_deleteCompute(self);
		nn_layer_delete((nn_layer_t**) _self);
	}
}
