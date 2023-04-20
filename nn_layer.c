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
	self->dim_fn          = info->dim_fn;

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

nn_dim_t* nn_layer_dim(nn_layer_t* self)
{
	ASSERT(self);

	nn_layer_dimFn dim_fn = self->dim_fn;
	return (*dim_fn)(self);
}
