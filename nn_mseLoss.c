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
#include "nn_arch.h"
#include "nn_mseLoss.h"
#include "nn_tensor.h"

/***********************************************************
* private                                                  *
***********************************************************/

static nn_tensor_t*
nn_mseLoss_backpropFn(nn_loss_t* base, nn_tensor_t* Y,
                      nn_tensor_t* Yt)
{
	ASSERT(base);
	ASSERT(Y);
	ASSERT(Yt);

	nn_mseLoss_t* self  = (nn_mseLoss_t*) base;
	nn_tensor_t*  dL_dY = self->dL_dY;
	nn_dim_t*     dim   = nn_tensor_dim(dL_dY);

	uint32_t bs = base->arch->batch_size;

	uint32_t m;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	uint32_t size = bs*dim->height*dim->width*dim->depth;
	float    y;
	float    yt;
	float    dl_dy;
	float    s = 2.0f/((float) size);
	for(i = 0; i < dim->height; ++i)
	{
		for(j = 0; j < dim->width; ++j)
		{
			for(k = 0; k < dim->depth; ++k)
			{
				dl_dy = 0.0f;
				for(m = 0; m < bs; ++m)
				{
					y  = nn_tensor_get(Y, m, i, j, k);
					yt = nn_tensor_get(Yt, m, i, j, k);
					dl_dy += y - yt;
				}
				nn_tensor_set(dL_dY, 0, i, j, k, s*dl_dy);
			}
		}
	}

	return dL_dY;
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_mseLoss_t*
nn_mseLoss_new(nn_arch_t* arch, nn_dim_t* dim)
{
	ASSERT(arch);

	nn_lossInfo_t info =
	{
		.arch        = arch,
		.backprop_fn = nn_mseLoss_backpropFn,
	};

	nn_mseLoss_t* self;
	self = (nn_mseLoss_t*)
	       nn_loss_new(sizeof(nn_mseLoss_t), &info);
	if(self == NULL)
	{
		return NULL;
	}

	nn_dim_t dim_1hwd =
	{
		.count  = 1,
		.height = dim->height,
		.width  = dim->width,
		.depth  = dim->depth,
	};

	self->dL_dY = nn_tensor_new(&dim_1hwd);
	if(self->dL_dY == NULL)
	{
		goto fail_dL_dY;
	}

	// success
	return self;

	// failure
	fail_dL_dY:
		nn_loss_delete((nn_loss_t**) &self);
	return NULL;
}

void nn_mseLoss_delete(nn_mseLoss_t** _self)
{
	ASSERT(_self);

	nn_mseLoss_t* self = *_self;
	if(self)
	{
		nn_tensor_delete(&self->dL_dY);
		nn_loss_delete((nn_loss_t**) _self);
	}
}
