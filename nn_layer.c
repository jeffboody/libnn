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
#include "nn_tensor.h"

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

	self->arch          = info->arch;
	self->compute_fp_fn = info->compute_fp_fn;
	self->compute_bp_fn = info->compute_bp_fn;
	self->post_fn       = info->post_fn;
	self->dimX_fn       = info->dimX_fn;
	self->dimY_fn       = info->dimY_fn;

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

nn_dim_t* nn_layer_dimX(nn_layer_t* self)
{
	ASSERT(self);

	nn_layerDim_fn dimX_fn = self->dimX_fn;
	return (*dimX_fn)(self);
}

nn_dim_t* nn_layer_dimY(nn_layer_t* self)
{
	ASSERT(self);

	nn_layerDim_fn dimY_fn = self->dimY_fn;
	return (*dimY_fn)(self);
}

nn_tensor_t*
nn_layer_computeFp(nn_layer_t* self,
                   int flags, uint32_t bs,
                   nn_tensor_t* X)
{
	ASSERT(self);
	ASSERT(X);

	nn_dim_t* dimX1 = nn_layer_dimX(self);
	nn_dim_t* dimX2 = nn_tensor_dim(X);
	if(nn_dim_sizeEquals(dimX1, dimX2) == 0)
	{
		LOGE("invalid count=%u:%u, height=%u:%u, width=%u:%u, depth=%u:%u",
		     dimX1->count,  dimX2->count,
		     dimX1->height, dimX2->height,
		     dimX1->width,  dimX2->width,
		     dimX1->depth,  dimX2->depth);
		return NULL;
	}

	nn_layerComputeFp_fn compute_fp_fn;
	compute_fp_fn = self->compute_fp_fn;
	return (*compute_fp_fn)(self, flags, bs, X);
}

nn_tensor_t*
nn_layer_computeBp(nn_layer_t* self,
                   int flags, uint32_t bs,
                   nn_tensor_t* dL_dY)
{
	ASSERT(self);
	ASSERT(dL_dY);

	nn_dim_t* dimY1 = nn_layer_dimY(self);
	nn_dim_t* dimY2 = nn_tensor_dim(dL_dY);
	if(nn_dim_sizeEquals(dimY1, dimY2) == 0)
	{
		LOGE("invalid count=%u:%u, height=%u:%u, width=%u:%u, depth=%u:%u",
		     dimY1->count,  dimY2->count,
		     dimY1->height, dimY2->height,
		     dimY1->width,  dimY2->width,
		     dimY1->depth,  dimY2->depth);
		return NULL;
	}

	nn_layerComputeBp_fn compute_bp_fn;
	compute_bp_fn = self->compute_bp_fn;
	return (*compute_bp_fn)(self, flags, bs, dL_dY);
}

void nn_layer_post(nn_layer_t* self, int flags, uint32_t bs)
{
	ASSERT(self);

	// optional post training/prediction operation
	nn_layerPost_fn post_fn = self->post_fn;
	if(post_fn)
	{
		return (*post_fn)(self, flags, bs);
	}
}
