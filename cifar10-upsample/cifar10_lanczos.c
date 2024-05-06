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

#define LOG_TAG "cifar10"
#include "libcc/math/cc_pow2n.h"
#include "libcc/cc_log.h"
#include "libcc/cc_memory.h"
#include "libnn/nn_lanczosLayer.h"
#include "libnn/nn_tensor.h"
#include "cifar10_lanczos.h"

/***********************************************************
* public                                                   *
***********************************************************/

cifar10_lanczos_t*
cifar10_lanczos_new(nn_engine_t* engine,
                    nn_dim_t* dimX, nn_dim_t* dimY)
{
	ASSERT(engine);
	ASSERT(dimX);

	// arch state is unused
	nn_archState_t arch_state =
	{
		.adam_alpha  = 0.0001f,
		.adam_beta1  = 0.9f,
		.adam_beta2  = 0.999f,
		.adam_beta1t = 1.0f,
		.adam_beta2t = 1.0f,
		.adam_lambda = 0.25f*0.001f,
		.adam_nu     = 1.0f,
		.bn_momentum = 0.99f,
	};

	cifar10_lanczos_t* self;
	self = (cifar10_lanczos_t*)
	       nn_arch_new(engine, sizeof(cifar10_lanczos_t),
	                   &arch_state);
	if(self == NULL)
	{
		return NULL;
	}

	self->Xio = nn_tensor_new(engine, dimX,
	                          NN_TENSOR_INIT_ZERO,
	                          NN_TENSOR_MODE_IO);
	if(self->Xio == NULL)
	{
		goto failure;
	}

	nn_dim_t dimT =
	{
		.count  = dimX->count,
		.height = dimX->height,
		.width  = dimY->width,
		.depth  = dimX->depth,
	};

	self->Tio = nn_tensor_new(engine, &dimT,
	                          NN_TENSOR_INIT_ZERO,
	                          NN_TENSOR_MODE_IO);
	if(self->Tio == NULL)
	{
		goto failure;
	}

	self->Yio = nn_tensor_new(engine, dimY,
	                          NN_TENSOR_INIT_ZERO,
	                          NN_TENSOR_MODE_IO);
	if(self->Yio == NULL)
	{
		goto failure;
	}

	self->lanczos = nn_lanczosLayer_new(&self->base,
	                                    dimX, dimY, 3);
	if(self->lanczos == NULL)
	{
		goto failure;
	}

	if(nn_arch_attachLayer(&self->base,
	                       &self->lanczos->base) == 0)
	{
		goto failure;
	}

	// success
	return self;

	// failure
	failure:
		cifar10_lanczos_delete(&self);
	return NULL;
}

void cifar10_lanczos_delete(cifar10_lanczos_t** _self)
{
	ASSERT(_self);

	cifar10_lanczos_t* self = *_self;
	if(self)
	{
		nn_tensor_delete(&self->Xio);
		nn_tensor_delete(&self->Tio);
		nn_tensor_delete(&self->Yio);
		nn_lanczosLayer_delete(&self->lanczos);
		nn_arch_delete((nn_arch_t**) _self);
	}
}

nn_tensor_t*
cifar10_lanczos_computeFp(cifar10_lanczos_t* self,
                          int flags, uint32_t bs,
                          nn_tensor_t* X)
{
	ASSERT(self);
	ASSERT(X);

	// update references
	self->X = X;
	self->T = self->lanczos->T;
	self->Y = nn_arch_forwardPass(&self->base, flags, bs, X);

	// mark dirty
	self->X_dirty = 1;
	self->T_dirty = 1;
	self->Y_dirty = 1;

	return self->Y;
}

int cifar10_lanczos_exportX(cifar10_lanczos_t* self,
                            const char* fname,
                            uint32_t n)
{
	ASSERT(self);
	ASSERT(fname);

	nn_dim_t* dim = nn_tensor_dim(self->Xio);

	if(self->X == NULL)
	{
		LOGE("invalid");
		return 0;
	}

	if(self->X_dirty)
	{
		if(nn_tensor_copy(self->X, self->Xio,
		                  0, 0, dim->count) == 0)
		{
			return 0;
		}

		self->X_dirty = 0;
	}

	return nn_tensor_ioExportPng(self->Xio, fname,
	                             n, 0, dim->depth,
	                             0.0f, 1.0f);
}

int cifar10_lanczos_exportT(cifar10_lanczos_t* self,
                            const char* fname,
                            uint32_t n)
{
	ASSERT(self);
	ASSERT(fname);

	nn_dim_t* dim = nn_tensor_dim(self->Tio);

	if(self->T == NULL)
	{
		LOGE("invalid");
		return 0;
	}

	if(self->T_dirty)
	{
		if(nn_tensor_copy(self->T, self->Tio,
		                  0, 0, dim->count) == 0)
		{
			return 0;
		}

		self->T_dirty = 0;
	}

	return nn_tensor_ioExportPng(self->Tio, fname,
	                             n, 0, dim->depth,
	                             0.0f, 1.0f);
}

int cifar10_lanczos_exportY(cifar10_lanczos_t* self,
                            const char* fname,
                            uint32_t n)
{
	ASSERT(self);
	ASSERT(fname);

	nn_dim_t* dim = nn_tensor_dim(self->Yio);

	if(self->Y == NULL)
	{
		LOGE("invalid");
		return 0;
	}

	if(self->Y_dirty)
	{
		if(nn_tensor_copy(self->Y, self->Yio,
		                  0, 0, dim->count) == 0)
		{
			return 0;
		}

		self->Y_dirty = 0;
	}

	return nn_tensor_ioExportPng(self->Yio, fname,
	                             n, 0, dim->depth,
	                             0.0f, 1.0f);
}
