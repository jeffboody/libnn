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

#define LOG_TAG "mnist"
#include "../../libcc/cc_log.h"
#include "../../libcc/cc_memory.h"
#include "../nn_coderLayer.h"
#include "../nn_convLayer.h"
#include "../nn_reshapeLayer.h"
#include "../nn_weightLayer.h"
#include "mnist_ganGen.h"

/***********************************************************
* public                                                   *
***********************************************************/

mnist_ganGen_t*
mnist_ganGen_new(nn_engine_t* engine, uint32_t bs)
{
	ASSERT(engine);

	nn_archState_t state =
	{
		.adam_alpha  = 0.0002f,
		.adam_beta1  = 0.5f,
		.adam_beta2  = 0.999f,
		.adam_beta1t = 1.0f,
		.adam_beta2t = 1.0f,
		.bn_momentum = 0.99f,
	};

	mnist_ganGen_t* self;
	self = (mnist_ganGen_t*)
	       nn_arch_new(engine, sizeof(mnist_ganGen_t),
	                   &state);
	if(self == NULL)
	{
		return NULL;
	}

	nn_dim_t dimX =
	{
		.count  = bs,
		.height = 1,
		.width  = 1,
		.depth  = 100,
	};

	nn_dim_t dimW =
	{
		.count  = 7*7*128,
		.height = 1,
		.width  = 1,
		.depth  = dimX.depth,
	};

	nn_dim_t* dim = &dimX;

	self->w0 = nn_weightLayer_new(&self->base, dim, &dimW,
	                              NN_WEIGHT_LAYER_FLAG_HE);
	if(self->w0 == NULL)
	{
		goto failure;
	}
	dim = nn_layer_dimY(&self->w0->base);

	self->f0 = nn_factLayer_new(&self->base, dim,
	                            NN_FACT_LAYER_FN_LRELU);
	if(self->f0 == NULL)
	{
		goto failure;
	}
	dim = nn_layer_dimY(&self->f0->base);

	nn_dim_t dim_r1 =
	{
		.count  = bs,
		.height = 7,
		.width  = 7,
		.depth  = 128,
	};

	self->r1 = nn_reshapeLayer_new(&self->base,
	                               dim, &dim_r1);
	if(self->r1 == NULL)
	{
		goto failure;
	}
	dim = nn_layer_dimY(&self->r1->base);

	nn_coderLayerInfo_t c2_info =
	{
		.arch = &self->base,

		.dimX = dim,
		.fc   = 128,

		// conv layer
		.conv_flags  = NN_CONV_LAYER_FLAG_TRANSPOSE,
		.conv_size   = 4,
		.conv_stride = 2,

		// fact layer
		.fact_fn = NN_FACT_LAYER_FN_LRELU,
	};

	self->c2 = nn_coderLayer_new(&c2_info);
	if(self->c2 == NULL)
	{
		goto failure;
	}
	dim = nn_layer_dimY(&self->c2->base);

	nn_coderLayerInfo_t c3_info =
	{
		.arch = &self->base,

		.dimX = dim,
		.fc   = 128,

		// conv layer
		.conv_flags  = NN_CONV_LAYER_FLAG_TRANSPOSE,
		.conv_size   = 4,
		.conv_stride = 2,

		// fact layer
		.fact_fn = NN_FACT_LAYER_FN_LRELU,
	};

	self->c3 = nn_coderLayer_new(&c3_info);
	if(self->c3 == NULL)
	{
		goto failure;
	}
	dim = nn_layer_dimY(&self->c3->base);

	nn_coderLayerInfo_t c4_info =
	{
		.arch = &self->base,

		.dimX = dim,
		.fc   = 1,

		// conv layer
		.conv_flags  = 0,
		.conv_size   = 7,
		.conv_stride = 1,

		// fact layer
		.fact_fn = NN_FACT_LAYER_FN_LOGISTIC,
	};

	self->c4 = nn_coderLayer_new(&c4_info);
	if(self->c4 == NULL)
	{
		goto failure;
	}
	dim = nn_layer_dimY(&self->c4->base);

	if((nn_arch_attachLayer(&self->base, &self->w0->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->f0->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->r1->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->c2->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->c3->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->c4->base) == 0))
	{
		goto failure;
	}

	// success
	return self;

	// failure
	failure:
		mnist_ganGen_delete(&self);
	return NULL;
}

void mnist_ganGen_delete(mnist_ganGen_t** _self)
{
	ASSERT(_self);

	mnist_ganGen_t* self = *_self;
	if(self)
	{
		nn_coderLayer_delete(&self->c4);
		nn_coderLayer_delete(&self->c3);
		nn_coderLayer_delete(&self->c2);
		nn_reshapeLayer_delete(&self->r1);
		nn_factLayer_delete(&self->f0);
		nn_weightLayer_delete(&self->w0);
		nn_arch_delete((nn_arch_t**) _self);
	}
}
