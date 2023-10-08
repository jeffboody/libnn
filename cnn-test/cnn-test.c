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
#include "libvkk/vkk_platform.h"

/***********************************************************
* private                                                  *
***********************************************************/

static void
fillXYt(uint32_t m,
        cc_rngNormal_t* rng1, cc_rngNormal_t* rng2,
        nn_tensor_t* X, nn_tensor_t* Yt)
{
	ASSERT(rng1);
	ASSERT(rng2);
	ASSERT(X);
	ASSERT(Yt);

	nn_dim_t* dimX  = nn_tensor_dim(X);
	nn_dim_t* dimYt = nn_tensor_dim(Yt);

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

	// fill Yt
	float sobel[] =
	{
		 0.25f,  0.5f,  0.25f,
		  0.0f,  0.0f,   0.0f,
		-0.25f, -0.5f, -0.25f,
	};
	float s;
	float y;
	uint32_t fi;
	uint32_t fj;
	uint32_t fh = 3;
	uint32_t fw = 3;
	uint32_t yh = dimYt->height;
	uint32_t yw = dimYt->width;;
	int      ii;
	int      jj;
	for(i = 0; i < yh; ++i)
	{
		for(j = 0; j < yw; ++j)
		{
			y = 0.0f;
			for(fi = 0; fi < fh; ++fi)
			{
				ii = i + fi - fh/2;
				if((ii < 0) || (ii >= xh))
				{
					continue;
				}

				for(fj = 0; fj < fw; ++fj)
				{
					jj = j + fj - fw/2;
					if((jj < 0) || (jj >= xw))
					{
						continue;
					}

					s = sobel[fi*fw + fj];
					x = nn_tensor_get(X, m, ii, jj, k);

					// add noise
					x += cc_rngNormal_rand1F(rng2);
					y += s*x;
				}
			}
			nn_tensor_set(Yt, m, i, j, k, y);
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
* callbacks                                                *
***********************************************************/

static int
cnn_test_onMain(vkk_engine_t* engine, int argc, char** argv)
{
	ASSERT(engine);

	uint32_t bs = 16;

	nn_archState_t arch_state =
	{
		.learning_rate  = 0.000001f,
		.momentum_decay = 0.5f,
		.batch_momentum = 0.99f,
		.l2_lambda      = 0.0001f,
	};

	nn_arch_t* arch = nn_arch_new(engine, 0, &arch_state);
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

	nn_tensor_t* X;
	X = nn_tensor_new(arch, &dimX,
	                  NN_TENSOR_INIT_ZERO,
	                  NN_TENSOR_MODE_IO);
	if(X == NULL)
	{
		goto fail_X;
	}

	nn_batchNormMode_e bn_mode = NN_BATCH_NORM_MODE_RUNNING;

	nn_dim_t* dim = nn_tensor_dim(X);

	nn_batchNormLayer_t* bn;
	bn = nn_batchNormLayer_new(arch, bn_mode, dim);
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

	nn_tensor_t* Yt;
	Yt = nn_tensor_new(arch, dim,
	                   NN_TENSOR_INIT_ZERO,
	                   NN_TENSOR_MODE_IO);
	if(Yt == NULL)
	{
		goto fail_Yt;
	}

	nn_loss_t* loss;
	loss = nn_loss_new(arch, dim, NN_LOSS_FN_MSE);
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
			fillXYt(m, &rng1, &rng2, X, Yt);
		}

		nn_arch_train(arch, NN_LAYER_MODE_TRAIN, bs,
		              X, Yt, NULL);

		if(idx%10 == 0)
		{
			LOGI("train-%u, loss=%f",
			     idx, nn_arch_loss(arch));
			#if 0
			nn_tensor_print(X,"X");
			nn_tensor_print(Yt, "Yt");
			nn_tensor_print(bn->G, "bn->G");
			nn_tensor_print(bn->B, "bn->B");
			nn_tensor_print(bn->Xhat, "bn->Xhat");
			nn_tensor_print(bn->Y, "bn->Y");
			nn_tensor_print(bn->Xmean_mb, "bn->Xmean_mb");
			nn_tensor_print(bn->Xmean_ra, "bn->Xmean_ra");
			nn_tensor_print(bn->Xvar_mb, "bn->Xvar_mb");
			nn_tensor_print(bn->Xvar_ra, "bn->Xvar_ra");
			nn_tensor_print(bn->dL_dXhat, "bn->dL_dXhat");
			nn_tensor_print(bn->Bsum, "bn->Bsum");
			nn_tensor_print(bn->Csum, "bn->Csum");
			nn_tensor_print(conv->W, "conv->W");
			nn_tensor_print(conv->B, "conv->B");
			nn_tensor_print(conv->Y, "conv->Y");
			nn_tensor_print(conv->VW, "conv->VW");
			nn_tensor_print(conv->VB, "conv->VB");
			nn_tensor_print(conv->dL_dW, "conv->dL_dW");
			nn_tensor_print(conv->dL_dB, "conv->dL_dB");
			nn_tensor_print(conv->dL_dX, "conv->dL_dX");
			nn_tensor_print(loss->dL_dY, "loss->dL_dY");
			#else
			nn_tensor_print(bn->G, "bn->G");
			nn_tensor_print(bn->B, "bn->B");
			nn_tensor_print(conv->W, "conv->W");
			nn_tensor_print(conv->B, "conv->B");
			#endif
		}
	}

	nn_loss_delete(&loss);
	nn_tensor_delete(&Yt);
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
		nn_tensor_delete(&Yt);
	fail_Yt:
		nn_convLayer_delete(&conv);
	fail_conv:
		nn_batchNormLayer_delete(&bn);
	fail_bn:
		nn_tensor_delete(&X);
	fail_X:
		nn_arch_delete(&arch);
	return EXIT_FAILURE;
}

vkk_platformInfo_t VKK_PLATFORM_INFO =
{
	.app_name    = "CNN-Test",
	.app_version =
	{
		.major = 1,
		.minor = 0,
		.patch = 0,
	},
	.app_dir = "cnn-test",
	.onMain  = cnn_test_onMain,
};
