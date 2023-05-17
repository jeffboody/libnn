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
#include "libnn/nn_arch.h"
#include "libnn/nn_dim.h"
#include "libnn/nn_mseLoss.h"
#include "libnn/nn_tensor.h"
#include "libnn/nn_convLayer.h"

/***********************************************************
* private                                                  *
***********************************************************/

static void
fillXY(uint32_t m, cc_rngNormal_t* rng,
       nn_tensor_t* X, nn_tensor_t* Y)
{
	ASSERT(rng);
	ASSERT(X);
	ASSERT(Y);

	nn_dim_t* dimX = nn_tensor_dim(X);
	nn_dim_t* dimY = nn_tensor_dim(Y);

	// fill X
	float    x;
	uint32_t xh = dimX->height;
	uint32_t xw = dimX->width;
	uint32_t i;
	uint32_t j;
	uint32_t k = 0;
	for(i = 0; i < xh; ++i)
	{
		for(j = 0; j < xw; ++j)
		{
			x = cc_rngNormal_rand1F(rng);
			nn_tensor_set(X, m, i, j, k, x);
		}
	}

	// fill Y
	float sobel[] =
	{
		 0.25f,  0.5f,  0.25f,
		  0.0f,  0.0f,   0.0f,
		-0.25f, -0.5f, -0.25f,
	};
	float s;
	float y;
	uint32_t ii;
	uint32_t jj;
	uint32_t fh = 3;
	uint32_t fw = 3;
	uint32_t yh = dimY->height;
	uint32_t yw = dimY->width;;
	for(i = 0; i < yh; ++i)
	{
		for(j = 0; j < yw; ++j)
		{
			y = 0.0f;
			for(ii = 0; ii < fh; ++ii)
			{
				for(jj = 0; jj < fw; ++jj)
				{
					s  = sobel[ii*fw + jj];
					x  = nn_tensor_get(X, m, i + ii, j + jj, k);
					y += s*x;
				}
			}
			nn_tensor_set(Y, m, i, j, k, y);
		}
	}
}

/***********************************************************
* public                                                   *
***********************************************************/

int main(int argc, char** argv)
{
	uint32_t bs = 16;

	nn_archInfo_t arch_info =
	{
		.learning_rate = 0.00001f,
	};

	nn_arch_t* arch = nn_arch_new(0, &arch_info);
	if(arch == NULL)
	{
		return EXIT_FAILURE;
	}

	nn_dim_t dimX =
	{
		.count  = bs,
		.width  = 64,
		.height = 64,
		.depth  = 1,
	};

	nn_tensor_t* X = nn_tensor_new(&dimX);
	if(X == NULL)
	{
		goto fail_X;
	}

	nn_dim_t* dim = nn_tensor_dim(X);

	nn_dim_t dimW =
	{
		.count  = 1,
		.width  = 3,
		.height = 3,
		.depth  = dim->depth,
	};

	nn_convLayer_t* conv;
	conv = nn_convLayer_new(arch, &dimX, &dimW, 1,
	                        NN_CONV_LAYER_FLAG_XAVIER);
	if(conv == NULL)
	{
		goto fail_conv;
	}
	dim = nn_layer_dimY(&conv->base);

	nn_tensor_t* Y = nn_tensor_new(dim);
	if(Y == NULL)
	{
		goto fail_Y;
	}

	nn_mseLoss_t* mse_loss;
	mse_loss = nn_mseLoss_new(arch, dim);
	if(mse_loss == NULL)
	{
		goto fail_mse_loss;
	}

	if((nn_arch_attachLayer(arch, (nn_layer_t*) conv) == 0) ||
	   (nn_arch_attachLoss(arch, (nn_loss_t*) mse_loss) == 0))
	{
		goto fail_attach;
	}

	float mu    = 0.0f;
	float sigma = 1.0f;
	cc_rngNormal_t rng;
	cc_rngNormal_init(&rng, mu, sigma);

	// training
	uint32_t idx;
	uint32_t m;
	uint32_t count = 1000;
	for(idx = 0; idx < count; ++idx)
	{
		for(m = 0; m < bs; ++m)
		{
			fillXY(m, &rng, X, Y);
		}

		nn_arch_train(arch, bs, X, Y);

		// if(idx%10 == 0)
		{
			LOGI("train-%u, loss=%f",
			     idx, nn_arch_loss(arch));
			// nn_tensor_print(X, "X");
			// nn_tensor_print(Y, "Y");
			nn_tensor_print(conv->W, "W");
			nn_tensor_print(conv->B, "B");
		}
	}

	nn_mseLoss_delete(&mse_loss);
	nn_tensor_delete(&Y);
	nn_convLayer_delete(&conv);
	nn_tensor_delete(&X);
	nn_arch_delete(&arch);

	// success
	return EXIT_SUCCESS;

	// failure
	fail_attach:
		nn_mseLoss_delete(&mse_loss);
	fail_mse_loss:
		nn_tensor_delete(&Y);
	fail_Y:
		nn_convLayer_delete(&conv);
	fail_conv:
		nn_tensor_delete(&X);
	fail_X:
		nn_arch_delete(&arch);
	return EXIT_FAILURE;
}
