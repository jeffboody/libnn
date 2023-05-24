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
#include "libnn/nn_batchNormLayer.h"
#include "libnn/nn_convLayer.h"
#include "libnn/nn_dim.h"
#include "libnn/nn_loss.h"
#include "libnn/nn_tensor.h"

/***********************************************************
* private                                                  *
***********************************************************/

static void
fillXY(uint32_t m,
       cc_rngNormal_t* rng1, cc_rngNormal_t* rng2,
       nn_tensor_t* X, nn_tensor_t* Y)
{
	ASSERT(rng1);
	ASSERT(rng2);
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
			x = cc_rngNormal_rand1F(rng1);
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

					// add noise
					x += cc_rngNormal_rand1F(rng2);
					y += s*x;
				}
			}
			nn_tensor_set(Y, m, i, j, k, y);
		}
	}

	// shift/scale X
	for(i = 0; i < xh; ++i)
	{
		for(j = 0; j < xw; ++j)
		{
			nn_tensor_mul(X, m, i, j, k, 100.0f);
			nn_tensor_add(X, m, i, j, k, 10.0f);
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
		.learning_rate  = 0.000001f,
		.momentum_decay = 0.5f,
		.batch_momentum = 0.99f,
		.l2_lambda      = 0.0001f,
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

	nn_batchNormLayer_t* bn;
	bn = nn_batchNormLayer_new(arch, dim);
	if(bn == NULL)
	{
		goto fail_bn;
	}

	nn_dim_t dimW =
	{
		.count  = 1,
		.width  = 3,
		.height = 3,
		.depth  = dim->depth,
	};

	nn_convLayer_t* conv;
	conv = nn_convLayer_new(arch, dim, &dimW, 1,
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

	nn_loss_t* loss;
	loss = nn_loss_new(arch, dim, nn_loss_mse);
	if(loss == NULL)
	{
		goto fail_loss;
	}

	if((nn_arch_attachLayer(arch, (nn_layer_t*) bn)   == 0) ||
	   (nn_arch_attachLayer(arch, (nn_layer_t*) conv) == 0) ||
	   (nn_arch_attachLoss(arch,  (nn_loss_t*) loss)  == 0))
	{
		goto fail_attach;
	}

	float mu     = 0.0f;
	float sigma1 = 1.0f;
	float sigma2 = 0.1f;
	cc_rngNormal_t rng1;
	cc_rngNormal_t rng2;
	cc_rngNormal_init(&rng1, mu, sigma1);
	cc_rngNormal_init(&rng2, mu, sigma2);

	// training
	uint32_t idx;
	uint32_t m;
	uint32_t count = 1000;
	for(idx = 0; idx < count; ++idx)
	{
		for(m = 0; m < bs; ++m)
		{
			fillXY(m, &rng1, &rng2, X, Y);
		}

		nn_arch_train(arch, bs, X, Y);

		// if(idx%10 == 0)
		{
			LOGI("train-%u, loss=%f",
			     idx, nn_arch_loss(arch));
			// nn_tensor_print(X, "X");
			// nn_tensor_print(Y, "Y");
			nn_tensor_print(bn->G, "G");
			nn_tensor_print(bn->B, "B");
			nn_tensor_print(bn->Xmean_mb, "Xmean_mb");
			nn_tensor_print(bn->Xmean_ra, "Xmean_ra");
			nn_tensor_print(bn->Xvar_mb, "Xvar_mb");
			nn_tensor_print(bn->Xvar_ra, "Xvar_ra");
			nn_tensor_print(conv->W, "W");
			nn_tensor_print(conv->B, "B");
		}
	}

	nn_loss_delete(&loss);
	nn_tensor_delete(&Y);
	nn_convLayer_delete(&conv);
	nn_batchNormLayer_delete(&bn);
	nn_tensor_delete(&X);
	nn_arch_delete(&arch);

	// success
	return EXIT_SUCCESS;

	// failure
	fail_attach:
		nn_loss_delete(&loss);
	fail_loss:
		nn_tensor_delete(&Y);
	fail_Y:
		nn_convLayer_delete(&conv);
	fail_conv:
		nn_batchNormLayer_delete(&bn);
	fail_bn:
		nn_tensor_delete(&X);
	fail_X:
		nn_arch_delete(&arch);
	return EXIT_FAILURE;
}
