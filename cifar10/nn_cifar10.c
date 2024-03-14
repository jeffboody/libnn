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
#include <stdio.h>
#include <stdlib.h>

#define LOG_TAG "nn"
#include "libcc/math/cc_float.h"
#include "libcc/cc_log.h"
#include "libcc/cc_memory.h"
#include "libnn/nn_tensor.h"
#include "nn_cifar10.h"

/***********************************************************
* public                                                   *
***********************************************************/

nn_cifar10_t*
nn_cifar10_load(nn_engine_t* engine, nn_cifar10Mode_e mode,
                int idx)
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

	nn_dim_t dim_color =
	{
		.count  = 10000,
		.height = 32,
		.width  = 32,
		.depth  = (uint32_t) NN_CIFAR10_MODE_COLOR,
	};

	nn_tensor_t* images;
	images = nn_tensor_new(engine, &dim_color,
	                       NN_TENSOR_INIT_ZERO,
	                       NN_TENSOR_MODE_IO);
	if(images == NULL)
	{
		goto fail_color;
	}
	self->images = images;

	if(mode == NN_CIFAR10_MODE_LUMINANCE)
	{
		nn_dim_t dim_lum =
		{
			.count  = 10000,
			.height = 32,
			.width  = 32,
			.depth  = (uint32_t) NN_CIFAR10_MODE_LUMINANCE,
		};

		self->images = nn_tensor_new(engine, &dim_lum,
		                             NN_TENSOR_INIT_ZERO,
		                             NN_TENSOR_MODE_IO);
		if(self->images == NULL)
		{
			goto fail_lum;
		}
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
					nn_tensor_ioSet(images, n, i, j, k, c);
				}
			}
		}
	}

	// optionally convert color to luminance
	// https://github.com/antimatter15/rgb-lab/blob/master/color.js
	if(mode == NN_CIFAR10_MODE_LUMINANCE)
	{
		float r;
		float g;
		float b;
		float yy;
		float labl;
		for(n = 0; n < 10000; ++n)
		{
			for(i = 0; i < 32; ++i)
			{
				for(j = 0; j < 32; ++j)
				{
					r = nn_tensor_ioGet(images, n, i, j, 0);
					g = nn_tensor_ioGet(images, n, i, j, 1);
					b = nn_tensor_ioGet(images, n, i, j, 2);

					r = (r > 0.04045f) ?
					    powf((r + 0.055f)/1.055f, 2.4f) : r/12.92f;
					g = (g > 0.04045f) ?
					    powf((g + 0.055f)/1.055f, 2.4f) : g/12.92f;
					b = (b > 0.04045f) ?
					    powf((b + 0.055f)/1.055f, 2.4f) : b/12.92f;

					yy = (r*0.2126f + g*0.7152f + b*0.0722f)/1.00000f;
					yy = (yy > 0.008856f) ?
					     powf(yy, 0.333333f) :
					     (7.787f*yy) + 16.0f/116.0f;

					labl = cc_clamp((1.0f/100.0f)*(116.0f*yy - 16.0f),
					                0.0f, 1.0f);

					nn_tensor_ioSet(self->images, n, i, j, 0, labl);
				}
			}
		}

		nn_tensor_delete(&images);
	}

	FREE(buf);
	fclose(f);

	// success
	return self;

	// failure
	fail_lum:
		nn_tensor_delete(&images);
	fail_color:
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
