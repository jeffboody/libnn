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

#define LOG_TAG "nn"
#include "../libcc/cc_log.h"
#include "../libcc/cc_memory.h"
#include "nn_layer.h"

/***********************************************************
* public                                                   *
***********************************************************/

nn_layer_t* nn_layer_new(size_t base_size,
                         nn_layerInfo_t* info)
{
	ASSERT(info);

	if(base_size == 0)
	{
		base_size = sizeof(nn_layer_t);
	}

	nn_layer_t* self;
	self = (nn_layer_t*) CALLOC(1, base_size);
	if(self == NULL)
	{
		LOGE("CALLOC failed");
		return NULL;
	}

	self->arch            = info->arch;
	self->forward_pass_fn = info->forward_pass_fn;
	self->backprop_fn     = info->backprop_fn;
	self->post_fn         = info->post_fn;
	self->dimX_fn         = info->dimX_fn;
	self->dimY_fn         = info->dimY_fn;

	// success
	return self;
}

void nn_layer_delete(nn_layer_t** _self)
{
	ASSERT(_self);

	nn_layer_t* self = *_self;
	if(self)
	{
		FREE(self);
		*_self = self;
	}
}

nn_tensor_t*
nn_layer_forwardPass(nn_layer_t* self, int flags,
                     uint32_t bs, nn_tensor_t* X)
{
	ASSERT(self);
	ASSERT(X);

	nn_layer_forwardPassFn forward_pass_fn;
	forward_pass_fn = self->forward_pass_fn;
	return (*forward_pass_fn)(self, flags, bs, X);
}

nn_tensor_t*
nn_layer_backprop(nn_layer_t* self, int flags,
                  uint32_t bs, nn_tensor_t* dL_dY)
{
	ASSERT(self);
	ASSERT(dL_dY);

	nn_layer_backpropFn backprop_fn;
	backprop_fn = self->backprop_fn;
	return (*backprop_fn)(self, flags, bs, dL_dY);
}

void nn_layer_post(nn_layer_t* self, int flags)
{
	ASSERT(self);

	// optional post training/prediction operation
	nn_layer_postFn post_fn = self->post_fn;
	if(post_fn)
	{
		return (*post_fn)(self, flags);
	}
}

nn_dim_t* nn_layer_dimX(nn_layer_t* self)
{
	ASSERT(self);

	nn_layer_dimFn dimX_fn = self->dimX_fn;
	return (*dimX_fn)(self);
}

nn_dim_t* nn_layer_dimY(nn_layer_t* self)
{
	ASSERT(self);

	nn_layer_dimFn dimY_fn = self->dimY_fn;
	return (*dimY_fn)(self);
}
