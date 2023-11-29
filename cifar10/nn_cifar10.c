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

#include <stdio.h>
#include <stdlib.h>

#define LOG_TAG "nn"
#include "libcc/cc_log.h"
#include "libcc/cc_memory.h"
#include "libnn/nn_tensor.h"
#include "nn_cifar10.h"

/***********************************************************
* public                                                   *
***********************************************************/

nn_cifar10_t* nn_cifar10_load(nn_engine_t* engine, int idx)
{
	ASSERT(engine);

	char fname[256];
	if(idx == 0)
	{
		snprintf(fname, 256, "%s",
		         "libnn/cifar10/cifar-10-batches-bin/test_batch.bin");
	}
	else if((idx >= 1) && (idx <= 5))
	{
		snprintf(fname, 256,
		         "libnn/cifar10/cifar-10-batches-bin/data_batch_%i.bin",
		         idx);
	}
	else
	{
		LOGE("invalid idx=%i", idx);
		return NULL;
	}

	FILE* f = fopen(fname, "r");
	if(f == NULL)
	{
		LOGE("invalid fname=%s", fname);
		return NULL;
	}

	// allocate buffer
	size_t   size = 30730000;
	uint8_t* buf  = (uint8_t*) CALLOC(1, size);
	if(buf == NULL)
	{
		LOGE("CALLOC failed");
		goto fail_buf;
	}

	// read buffer
	if(fread((void*) buf, size, 1, f) != 1)
	{
		LOGE("fread failed");
		goto fail_read;
	}

	nn_cifar10_t* self;
	self = (nn_cifar10_t*)
	       CALLOC(1, sizeof(nn_cifar10_t));
	if(self == NULL)
	{
		LOGE("CALLOC failed");
		goto fail_alloc;
	}

	self->labels = (uint8_t*) CALLOC(10000, sizeof(uint8_t));
	if(self->labels == NULL)
	{
		goto fail_labels;
	}

	nn_dim_t dim =
	{
		.count  = 10000,
		.height = 32,
		.width  = 32,
		.depth  = 3,
	};

	self->images = nn_tensor_new(engine, &dim,
	                             NN_TENSOR_INIT_ZERO,
	                             NN_TENSOR_MODE_IO);
	if(self->images == NULL)
	{
		goto fail_images;
	}

	// parse data
	float    c;
	uint32_t offset = 0;
	uint32_t n;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(n = 0; n < 10000; ++n)
	{
		// parse label
		self->labels[n] = buf[offset++];

		// parse image
		for(k = 0; k < 3; ++k)
		{
			for(i = 0; i < 32; ++i)
			{
				for(j = 0; j < 32; ++j)
				{
					c = ((float) buf[offset++])/255.0f;
					nn_tensor_set(self->images, n, i, j, k, c);
				}
			}
		}
	}

	FREE(buf);
	fclose(f);

	// success
	return self;

	// failure
	fail_images:
		FREE(self->labels);
	fail_labels:
		FREE(self);
	fail_alloc:
	fail_read:
		FREE(buf);
	fail_buf:
		fclose(f);
	return NULL;
}

void nn_cifar10_delete(nn_cifar10_t** _self)
{
	ASSERT(_self);

	nn_cifar10_t* self = *_self;
	if(self)
	{
		nn_tensor_delete(&self->images);
		FREE(self->labels);
		FREE(self);
		*_self = NULL;
	}
}
