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
#include "nn_flattenLayer.h"
#include "nn_tensor.h"

/***********************************************************
* private                                                  *
***********************************************************/

static nn_tensor_t*
nn_flattenLayer_forwardPassFn(nn_layer_t* base, int mode,
                              uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_flattenLayer_t* self = (nn_flattenLayer_t*) base;
	nn_tensor_t*       Y    = &self->Y;

	if(nn_dim_equals(nn_tensor_dim(X), &self->dimX) == 0)
	{
		LOGE("invalid");
		return NULL;
	}

	Y->data = X->data;

	#ifdef NN_USE_COMPUTE
	Y->sb_data = X->sb_data;
	#endif

	return Y;
}

static nn_tensor_t*
nn_flattenLayer_backpropFn(nn_layer_t* base, uint32_t bs,
                           nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY);

	return dL_dY;
}

static nn_dim_t*
nn_flattenLayer_dimXFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_flattenLayer_t* self = (nn_flattenLayer_t*) base;

	return &self->dimX;
}

static nn_dim_t*
nn_flattenLayer_dimYFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_flattenLayer_t* self = (nn_flattenLayer_t*) base;

	return nn_tensor_dim(&self->Y);
}

#ifdef NN_USE_COMPUTE

static int
nn_flattenLayer_newCompute(nn_flattenLayer_t* self,
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
nn_flattenLayer_deleteCompute(nn_flattenLayer_t* self)
{
	ASSERT(self);

	nn_tensor_t* Y = &self->Y;
	vkk_buffer_delete(&Y->sb_dim);
}

#else // NN_USE_COMPUTE not defined

static int
nn_flattenLayer_newCompute(nn_flattenLayer_t* self,
                           nn_dim_t* dimY)
{
	return 1;
}

static void
nn_flattenLayer_deleteCompute(nn_flattenLayer_t* self)
{
}

#endif

/***********************************************************
* public                                                   *
***********************************************************/

nn_flattenLayer_t*
nn_flattenLayer_new(nn_arch_t* arch, nn_dim_t* dimX)
{
	ASSERT(arch);
	ASSERT(dimX);

	nn_layerInfo_t info =
	{
		.arch            = arch,
		.forward_pass_fn = nn_flattenLayer_forwardPassFn,
		.backprop_fn     = nn_flattenLayer_backpropFn,
		.dimX_fn         = nn_flattenLayer_dimXFn,
		.dimY_fn         = nn_flattenLayer_dimYFn,
	};

	nn_flattenLayer_t* self;
	self = (nn_flattenLayer_t*)
	       nn_layer_new(sizeof(nn_flattenLayer_t), &info);
	if(self == NULL)
	{
		return NULL;
	}

	nn_dim_copy(dimX, &self->dimX);

	nn_tensor_t* Y;
	nn_dim_t*    dimY;
	Y            = &self->Y;
	dimY         = nn_tensor_dim(Y);
	dimY->count  = dimX->count;
	dimY->height = 1;
	dimY->width  = 1;
	dimY->depth  = dimX->height*dimX->width*dimX->depth;

	if(nn_flattenLayer_newCompute(self, dimY) == 0)
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

nn_flattenLayer_t*
nn_flattenLayer_import(nn_arch_t* arch, jsmn_val_t* val)
{
	ASSERT(arch);
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	jsmn_val_t* val_dimX = NULL;

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
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if(val_dimX == NULL)
	{
		LOGE("invalid");
		return NULL;
	}

	nn_dim_t dimX;
	if(nn_dim_load(&dimX, val_dimX) == 0)
	{
		return NULL;
	}

	return nn_flattenLayer_new(arch, &dimX);
}

int nn_flattenLayer_export(nn_flattenLayer_t* self,
                           jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	nn_dim_t* dimX = &self->dimX;

	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	ret &= jsmn_stream_key(stream, "%s", "dimX");
	ret &= nn_dim_store(dimX, stream);
	ret &= jsmn_stream_end(stream);

	return ret;
}

void nn_flattenLayer_delete(nn_flattenLayer_t** _self)
{
	ASSERT(_self);

	nn_flattenLayer_t* self = *_self;
	if(self)
	{
		nn_flattenLayer_deleteCompute(self);
		nn_layer_delete((nn_layer_t**) _self);
	}
}
