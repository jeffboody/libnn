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

#define LOG_TAG "cnn-test"
#include "libcc/rng/cc_rngNormal.h"
#include "libcc/cc_log.h"
#include "libcc/cc_memory.h"
#include "libnn/nn_arch.h"
#include "libnn/nn_batchNormLayer.h"
#include "libnn/nn_convLayer.h"
#include "libnn/nn_dim.h"
#include "libnn/nn_engine.h"
#include "libnn/nn_loss.h"
#include "libnn/nn_tensor.h"
#include "libvkk/vkk_platform.h"

/***********************************************************
* private                                                  *
***********************************************************/

static void cnn_print(nn_tensor_t* self, const char* name)
{
	ASSERT(self);
	ASSERT(name);

	jsmn_stream_t* stream = jsmn_stream_new();
	if(stream == NULL)
	{
		return;
	}

	nn_tensor_store(self, stream);

	size_t size = 0;
	const char* buffer = jsmn_stream_buffer(stream, &size);
	if(buffer)
	{
		printf("%s: %s\n", name, buffer);
	}

	jsmn_stream_delete(&stream);
}

static void
cnn_fillXYt(uint32_t m,
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
			x = nn_tensor_get(X, m, i, j, k);
			nn_tensor_set(X, m, i, j, k, 100.0f*x + 10.0f);
		}
	}
}

/***********************************************************
* callbacks                                                *
***********************************************************/

static int
cnn_test_onMain(vkk_engine_t* ve, int argc, char** argv)
{
	ASSERT(ve);

	uint32_t bs = 16;

	nn_engine_t* engine = nn_engine_new(ve);
	if(engine == NULL)
	{
		return EXIT_FAILURE;
	}

	nn_archState_t arch_state =
	{
		.adam_alpha  = 0.01f,
		.adam_beta1  = 0.9f,
		.adam_beta2  = 0.999f,
		.adam_beta1t = 1.0f,
		.adam_beta2t = 1.0f,
		.adam_lambda = 0.25f*0.001f,
		.adam_nu     = 1.0f,
		.bn_momentum = 0.99f,
	};

	nn_arch_t* arch = nn_arch_new(engine, 0, &arch_state);
	if(arch == NULL)
	{
		goto fail_arch;
	}

	nn_dim_t dimX =
	{
		.count  = bs,
		.width  = 64,
		.height = 64,
		.depth  = 1,
	};

	nn_tensor_t* X;
	X = nn_tensor_new(engine, &dimX,
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
	Yt = nn_tensor_new(engine, dim,
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
			cnn_fillXYt(m, &rng1, &rng2, X, Yt);
		}

		nn_arch_train(arch, NN_LAYER_FLAG_TRAIN, bs,
		              X, Yt, NULL);

		if(idx%10 == 0)
		{
			LOGI("train-%u, loss=%f",
			     idx, nn_arch_loss(arch));
			#if 0
			cnn_print(X,"X");
			cnn_print(Yt, "Yt");
			cnn_print(bn->G, "bn->G");
			cnn_print(bn->B, "bn->B");
			cnn_print(bn->Xhat, "bn->Xhat");
			cnn_print(bn->Y, "bn->Y");
			cnn_print(bn->Xmean_mb, "bn->Xmean_mb");
			cnn_print(bn->Xmean_ra, "bn->Xmean_ra");
			cnn_print(bn->Xvar_mb, "bn->Xvar_mb");
			cnn_print(bn->Xvar_ra, "bn->Xvar_ra");
			cnn_print(bn->dL_dXhat, "bn->dL_dXhat");
			cnn_print(bn->Bsum, "bn->Bsum");
			cnn_print(bn->Csum, "bn->Csum");
			cnn_print(conv->W, "conv->W");
			cnn_print(conv->B, "conv->B");
			cnn_print(conv->Y, "conv->Y");
			cnn_print(conv->VW, "conv->VW");
			cnn_print(conv->VB, "conv->VB");
			cnn_print(conv->dL_dW, "conv->dL_dW");
			cnn_print(conv->dL_dB, "conv->dL_dB");
			cnn_print(conv->dL_dX, "conv->dL_dX");
			cnn_print(loss->dL_dY, "loss->dL_dY");
			#else
			cnn_print(bn->G, "bn->G");
			cnn_print(bn->B, "bn->B");
			cnn_print(conv->W, "conv->W");
			cnn_print(conv->B, "conv->B");
			#endif
		}
	}

	nn_loss_delete(&loss);
	nn_tensor_delete(&Yt);
	nn_convLayer_delete(&conv);
	nn_batchNormLayer_delete(&bn);
	nn_tensor_delete(&X);
	nn_arch_delete(&arch);
	nn_engine_delete(&engine);

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
	fail_arch:
		nn_engine_delete(&engine);
	return EXIT_FAILURE;
}

vkk_platformInfo_t VKK_PLATFORM_INFO =
{
	.app_name    = "cnn-test",
	.app_version =
	{
		.major = 1,
		.minor = 0,
		.patch = 0,
	},
	.app_dir = "cnn-test",
	.onMain  = cnn_test_onMain,
};
