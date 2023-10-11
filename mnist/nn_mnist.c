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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define LOG_TAG "nn"
#include "libcc/cc_log.h"
#include "libcc/cc_memory.h"
#include "libnn/nn_tensor.h"
#include "nn_mnist.h"

/***********************************************************
* private                                                  *
***********************************************************/

static int nn_mnist_readU32(FILE* f, uint32_t* _data)
{
	ASSERT(f);
	ASSERT(_data);

	uint32_t data;
	if(fread((void*) &data, sizeof(uint32_t), 1, f) != 1)
	{
		LOGE("fread failed");
		return 0;
	}

	// swap endian
	*_data = ((data << 24) & 0xFF000000) |
	         ((data << 8)  & 0x00FF0000) |
	         ((data >> 8)  & 0x0000FF00) |
	         ((data >> 24) & 0x000000FF);

	return 1;
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_tensor_t* nn_mnist_load(nn_arch_t* arch)
{
	ASSERT(arch);

	FILE* f = fopen("libnn/mnist/train-images-idx3-ubyte", "r");
	if(f == NULL)
	{
		LOGE("invalid");
		return NULL;
	}

	// read header
	uint32_t magic = 0;
	nn_dim_t dim =
	{
		.depth = 1,
	};
	if((nn_mnist_readU32(f, &magic)      == 0) ||
	   (nn_mnist_readU32(f, &dim.count)  == 0) ||
	   (nn_mnist_readU32(f, &dim.width)  == 0) ||
	   (nn_mnist_readU32(f, &dim.height) == 0))
	{
		goto fail_header;
	}

	// check header
	size_t size = dim.count*dim.height*dim.width;
	if((magic != 0x00000803) || (size == 0))
	{
		LOGE("invalid magic=0x%X, size=%u",
		     magic, (uint32_t) size);
		goto fail_check;
	}

	// allocate ubyte data
	uint8_t* data = (uint8_t*) CALLOC(1, size);
	if(data == NULL)
	{
		LOGE("CALLOC failed");
		goto fail_allocate;
	}

	// read ubyte data
	if(fread((void*) data, size, 1, f) != 1)
	{
		LOGE("fread failed");
		goto fail_read;
	}

	nn_tensor_t* T;
	T = nn_tensor_new(arch, &dim,
	                  NN_TENSOR_INIT_ZERO,
	                  NN_TENSOR_MODE_IO);
	if(T == NULL)
	{
		goto fail_T;
	}

	// convert data
	float    t;
	uint32_t m;
	uint32_t i;
	uint32_t j;
	uint32_t idx = 0;
	for(m = 0; m < dim.count; ++m)
	{
		for(i = 0; i < dim.height; ++i)
		{
			for(j = 0; j < dim.width; ++j)
			{
				t = ((float) data[idx++])/255.0f;
				nn_tensor_set(T, m, i, j, 1, t);
			}
		}
	}

	FREE(data);
	fclose(f);

	// success
	return T;

	// failure
	fail_T:
	fail_read:
		FREE(data);
	fail_allocate:
	fail_check:
	fail_header:
		fclose(f);
	return NULL;
}
