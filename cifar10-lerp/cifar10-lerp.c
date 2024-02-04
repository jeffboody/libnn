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

#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define LOG_TAG "cifar10"
#include "libcc/cc_log.h"
#include "libcc/cc_memory.h"
#include "libcc/cc_timestamp.h"
#include "libnn/cifar10/nn_cifar10.h"
#include "libnn/nn_coderLayer.h"
#include "libnn/nn_engine.h"
#include "libnn/nn_tensor.h"
#include "libvkk/vkk_platform.h"
#include "cifar10_lerp.h"

/***********************************************************
* private                                                  *
***********************************************************/

static void
cifar10_sample(nn_cifar10_t* cifar10, cc_rngUniform_t* rng,
               uint32_t bs, nn_tensor_t* Xt)
{
	ASSERT(cifar10);
	ASSERT(Xt);

	nn_dim_t* dim = nn_tensor_dim(cifar10->images);

	uint32_t m;
	uint32_t n;
	uint32_t max = dim->count - 1;
	for(m = 0; m < bs; ++m)
	{
		n = cc_rngUniform_rand2U(rng, 0, max);
		nn_tensor_blit(cifar10->images, Xt, 1, n, m);
	}
}

/***********************************************************
* callbacks                                                *
***********************************************************/

static int
cifar10_lerp_onMain(vkk_engine_t* ve, int argc,
                    char** argv)
{
	ASSERT(ve);

	cc_rngUniform_t rng;
	cc_rngUniform_init(&rng);

	nn_engine_t* engine = nn_engine_new(ve);
	if(engine == NULL)
	{
		return EXIT_FAILURE;
	}

	nn_cifar10_t* cifar10;
	cifar10 = nn_cifar10_load(engine,
	                          NN_CIFAR10_MODE_COLOR, 1);
	if(cifar10 == NULL)
	{
		goto fail_cifar10;
	}

	nn_dim_t* dim = nn_tensor_dim(cifar10->images);

	float bs = 32;
	float fc = 32;
	nn_dim_t dimXt =
	{
		.count  = bs,
		.height = dim->height,
		.width  = dim->width,
		.depth  = dim->depth,
	};

	nn_tensor_t* Xt;
	Xt = nn_tensor_new(engine, &dimXt,
	                   NN_TENSOR_INIT_ZERO,
	                   NN_TENSOR_MODE_IO);
	if(Xt == NULL)
	{
		goto fail_Xt;
	}

	nn_tensor_t* X;
	X = nn_tensor_new(engine, &dimXt,
	                  NN_TENSOR_INIT_ZERO,
	                  NN_TENSOR_MODE_IO);
	if(X == NULL)
	{
		goto fail_X;
	}

	nn_tensor_t* Y;
	Y = nn_tensor_new(engine, &dimXt,
	                  NN_TENSOR_INIT_ZERO,
	                  NN_TENSOR_MODE_IO);
	if(Y == NULL)
	{
		goto fail_Y;
	}

	nn_tensor_t* Y3;
	Y3 = nn_tensor_new(engine, &dimXt,
	                   NN_TENSOR_INIT_ZERO,
	                   NN_TENSOR_MODE_IO);
	if(Y3 == NULL)
	{
		goto fail_Y3;
	}

	cifar10_lerp_t* nn1;
	nn1 = cifar10_lerp_new(engine, bs, fc, dimXt.height,
	                       dimXt.width, dimXt.depth);
	if(nn1 == NULL)
	{
		goto fail_nn1;
	}

	cifar10_lerp_t* nn2;
	nn2 = cifar10_lerp_new(engine, bs, fc, dimXt.height,
	                       dimXt.width, dimXt.depth);
	if(nn2 == NULL)
	{
		goto fail_nn2;
	}

	cifar10_lerp_t* nn3;
	nn3 = cifar10_lerp_new(engine, bs, fc, dimXt.height,
	                       dimXt.width, dimXt.depth);
	if(nn3 == NULL)
	{
		goto fail_nn3;
	}

	int lerp = 1;
	lerp &= nn_coderLayer_lerp(nn1->coder1, nn2->coder3,
	                           0.25f, 0.75f);
	lerp &= nn_coderLayer_lerp(nn1->coder2, nn2->coder2,
	                           0.5f, 0.5f);
	lerp &= nn_coderLayer_lerp(nn1->coder3, nn2->coder1,
	                           0.75f, 0.25f);
	if(lerp == 0)
	{
		goto fail_lerp;
	}

	FILE* fplot = fopen("data/plot.dat", "w");
	if(fplot == NULL)
	{
		goto fail_fplot;
	}

	// training
	uint32_t epoch = 0;
	uint32_t step  = 0;
	uint32_t steps;
	char     fname[256];
	float    loss1;
	float    sum_loss1 = 0.0f;
	float    min_loss1 = FLT_MAX;
	float    max_loss1 = 0.0f;
	float    loss2;
	float    sum_loss2 = 0.0f;
	float    min_loss2 = FLT_MAX;
	float    max_loss2 = 0.0f;
	float    loss3;
	float    sum_loss3 = 0.0f;
	float    min_loss3 = FLT_MAX;
	float    max_loss3 = 0.0f;
	double   t0        = cc_timestamp();
	while(epoch < 20)
	{
		steps = (epoch + 1)*dim->count/bs;
		while(step < steps)
		{
			cifar10_sample(cifar10, &rng, bs, Xt);

			if(nn_arch_trainLERP(&nn1->base, &nn2->base, bs,
			                     Xt, Xt, X, Y) == NULL)
			{
				goto fail_train;
			}

			if(nn_arch_train(&nn3->base, NN_LAYER_FLAG_TRAIN, bs,
			                 Xt, Xt, Y3) == NULL)
			{
				goto fail_train;
			}

			// update loss
			loss1 = nn_arch_loss(&nn1->base);
			loss2 = nn_arch_loss(&nn2->base);
			loss3 = nn_arch_loss(&nn3->base);
			sum_loss1 += loss1;
			sum_loss2 += loss2;
			sum_loss3 += loss3;
			if(loss1 < min_loss1)
			{
				min_loss1 = loss1;
			}
			if(loss1 > max_loss1)
			{
				max_loss1 = loss1;
			}
			if(loss2 < min_loss2)
			{
				min_loss2 = loss2;
			}
			if(loss2 > max_loss2)
			{
				max_loss2 = loss2;
			}
			if(loss3 < min_loss3)
			{
				min_loss3 = loss3;
			}
			if(loss3 > max_loss3)
			{
				max_loss3 = loss3;
			}

			// export images
			uint32_t export_interval = 100;
			if((step%export_interval) == (export_interval - 1))
			{
				snprintf(fname, 256, "data/Xt-%u-%u.png",
				         epoch, step);
				nn_tensor_exportPng(Xt, fname, 0, 0, dim->depth - 1,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/Y-%u-%u.png",
				         epoch, step);
				nn_tensor_exportPng(Y, fname, 0, 0, dim->depth - 1,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/X-%u-%u.png",
				         epoch, step);
				nn_tensor_exportPng(X, fname, 0, 0, dim->depth - 1,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/Y3-%u-%u.png",
				         epoch, step);
				nn_tensor_exportPng(Y3, fname, 0, 0, dim->depth - 1,
				                    0.0f, 1.0f);
			}

			// plot loss
			uint32_t plot_interval = 100;
			if((step%plot_interval) == (plot_interval - 1))
			{
				float avg_loss1 = sum_loss1/((float) plot_interval);
				float avg_loss2 = sum_loss2/((float) plot_interval);
				float avg_loss3 = sum_loss3/((float) plot_interval);
				fprintf(fplot, "%u %u %f %f %f %f %f %f %f %f %f\n",
				        epoch, step,
				        avg_loss1, min_loss1, max_loss1,
				        avg_loss2, min_loss2, max_loss2,
				        avg_loss3, min_loss3, max_loss3);
				fflush(fplot);

				// reset loss
				sum_loss1 = 0.0f;
				min_loss1 = FLT_MAX;
				max_loss1 = 0.0f;
				sum_loss2 = 0.0f;
				min_loss2 = FLT_MAX;
				max_loss2 = 0.0f;
				sum_loss3 = 0.0f;
				min_loss3 = FLT_MAX;
				max_loss3 = 0.0f;
			}

			// export arch
			uint32_t arch_interval = 1000;
			if((step%arch_interval) == (arch_interval - 1))
			{
				snprintf(fname, 256, "data/nn1-%i-%i.json",
				         epoch, step);
				cifar10_lerp_export(nn1, fname);
				snprintf(fname, 256, "data/nn2-%i-%i.json",
				         epoch, step);
				cifar10_lerp_export(nn2, fname);
				snprintf(fname, 256, "data/nn3-%i-%i.json",
				         epoch, step);
				cifar10_lerp_export(nn3, fname);
			}

			LOGI("epoch=%u, step=%u, elapsed=%lf, loss1=%f, loss2=%f, loss3=%f",
			     epoch, step, cc_timestamp() - t0,
			     loss1, loss2, loss3);
			++step;
		}

		++epoch;
	}

	// cleanup
	fclose(fplot);
	cifar10_lerp_delete(&nn3);
	cifar10_lerp_delete(&nn2);
	cifar10_lerp_delete(&nn1);
	nn_tensor_delete(&Y3);
	nn_tensor_delete(&Y);
	nn_tensor_delete(&X);
	nn_tensor_delete(&Xt);
	nn_cifar10_delete(&cifar10);
	nn_engine_delete(&engine);

	// success
	return EXIT_SUCCESS;

	// failure
	fail_train:
		fclose(fplot);
	fail_fplot:
	fail_lerp:
		cifar10_lerp_delete(&nn3);
	fail_nn3:
		cifar10_lerp_delete(&nn2);
	fail_nn2:
		cifar10_lerp_delete(&nn1);
	fail_nn1:
		nn_tensor_delete(&Y3);
	fail_Y3:
		nn_tensor_delete(&Y);
	fail_Y:
		nn_tensor_delete(&X);
	fail_X:
		nn_tensor_delete(&Xt);
	fail_Xt:
		nn_cifar10_delete(&cifar10);
	fail_cifar10:
		nn_engine_delete(&engine);
	return EXIT_FAILURE;
}

vkk_platformInfo_t VKK_PLATFORM_INFO =
{
	.app_name    = "cifar10-lerp",
	.app_version =
	{
		.major = 1,
		.minor = 0,
		.patch = 0,
	},
	.app_dir = "cifar10-lerp",
	.onMain  = cifar10_lerp_onMain,
};
