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
#include "texgz/texgz_tex.h"
#include "cifar10_regen1.h"
#include "cifar10_regen2.h"

/***********************************************************
* private                                                  *
***********************************************************/

static int
cifar10_samplem(nn_cifar10_t* cifar10, cc_rngUniform_t* rng,
                uint32_t m, nn_tensor_t* X1, nn_tensor_t* X0,
                nn_tensor_t* Xt, nn_tensor_t* Yt)
{
	ASSERT(cifar10);
	ASSERT(rng);
	ASSERT(X1);
	ASSERT(X0);
	ASSERT(Xt);
	ASSERT(Yt);

	nn_dim_t* dim = nn_tensor_dim(cifar10->images);

	// input interpolation
	#if 1
	// linear interpolation
	float s = cc_rngUniform_rand2F(rng, 0.0f, 1.0f);
	if(m == 0)
	{
		s = 0.0f;
	}
	else if(m == 1)
	{
		s = 1.0f;
	}
	#else
	// no interpolation (GAN default)
	float s = 0.0f;
	if(m%2)
	{
		s = 1.0f;
	}
	#endif

	// realness coefficient
	float realness = s*s;

	// create texX1
	texgz_tex_t* texX1;
	texX1 = texgz_tex_new(dim->width, dim->height,
	                      dim->width, dim->height,
	                      TEXGZ_UNSIGNED_BYTE,
	                      TEXGZ_RGBA, NULL);
	if(texX1 == NULL)
	{
		return 0;
	}

	// initialize texX1
	unsigned char x[4] = { 0, 0, 0, 0 };;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(i = 0; i < dim->height; ++i)
	{
		for(j = 0; j < dim->width; ++j)
		{
			for(k = 0; k < dim->depth; ++k)
			{
				x[k] = (unsigned char)
				       (255.0f*nn_tensor_get(X1, m, i, j, k));
			}
			texgz_tex_setPixel(texX1, j, i, x);
		}
	}

	// X02 = lanczos3(X1)
	texgz_tex_t* texX02 = texgz_tex_lanczos3(texX1, 1);
	if(texX02 == NULL)
	{
		goto fail_texX02;
	}

	// X0 = resize(lanczos3(X1))
	texgz_tex_t* texX0;
	texX0 = texgz_tex_resize(texX02, dim->width, dim->height);
	if(texX0 == NULL)
	{
		goto fail_texX0;
	}

	// blit X0
	for(i = 0; i < dim->height; ++i)
	{
		for(j = 0; j < dim->width; ++j)
		{
			texgz_tex_getPixel(texX0, j, i, x);
			for(k = 0; k < dim->depth; ++k)
			{
				nn_tensor_set(X0, m, i, j, k,
				              ((float) x[k])/255.0f);
			}
		}
	}

	// Yt = lanczos3(X1)
	texgz_tex_t* texYt = texgz_tex_lanczos3(texX1, 1);
	if(texYt == NULL)
	{
		goto fail_texYt;
	}

	// blit Yt|R
	unsigned char yt[4] = { 0, 0, 0, 0 };
	for(i = 0; i < texYt->height; ++i)
	{
		for(j = 0; j < texYt->width; ++j)
		{
			texgz_tex_getPixel(texYt, j, i, yt);
			for(k = 0; k < dim->depth; ++k)
			{
				nn_tensor_set(Yt, m, i, j, k,
				              ((float) yt[k])/255.0f);
			}

			nn_tensor_set(Yt, m, i, j, dim->depth, realness);
		}
	}

	// blit Xt = s*X1 + (1 - s)*X0
	float x0;
	float x1;
	for(i = 0; i < dim->height; ++i)
	{
		for(j = 0; j < dim->width; ++j)
		{
			for(k = 0; k < dim->depth; ++k)
			{
				x0 = nn_tensor_get(X0, m, i, j, k);
				x1 = nn_tensor_get(X1, m, i, j, k);
				nn_tensor_set(Xt, m, i, j, k, s*x1 + (1.0f - s)*x0);
			}
		}
	}

	texgz_tex_delete(&texYt);
	texgz_tex_delete(&texX0);
	texgz_tex_delete(&texX02);
	texgz_tex_delete(&texX1);

	// success
	return 1;

	// failure
	fail_texYt:
		texgz_tex_delete(&texX0);
	fail_texX0:
		texgz_tex_delete(&texX02);
	fail_texX02:
		texgz_tex_delete(&texX1);
	return 0;
}

static int
cifar10_sample(nn_cifar10_t* cifar10, cc_rngUniform_t* rng,
               uint32_t bs, nn_tensor_t* X1, nn_tensor_t* X0,
               nn_tensor_t* Xt, nn_tensor_t* Yt)
{
	ASSERT(cifar10);
	ASSERT(rng);
	ASSERT(X1);
	ASSERT(X0);
	ASSERT(Xt);
	ASSERT(Yt);

	nn_dim_t* dim = nn_tensor_dim(cifar10->images);

	if(dim->depth > 4)
	{
		LOGE("invalid depth=%u", dim->depth);
		return 0;
	}

	uint32_t m;
	uint32_t n;
	uint32_t max = dim->count - 1;
	for(m = 0; m < bs; ++m)
	{
		// blit X1
		n = cc_rngUniform_rand2U(rng, 0, max);
		nn_tensor_blit(cifar10->images, X1, 1, n, m);

		// sample X0, Xt, Yt
		if(cifar10_samplem(cifar10, rng, m, X1, X0, Xt, Yt) == 0)
		{
			return 0;
		}
	}

	return 1;
}

/***********************************************************
* callbacks                                                *
***********************************************************/

static int
cifar10_regen_onMain(vkk_engine_t* ve, int argc,
                     char** argv)
{
	ASSERT(ve);

	/*
	 * Regenerator
	 * X1: Real Input
	 * X0: Generated Input = Resize(Lanczos3(X1))
	 * Yt: Lanczos3(X0)
	 * s:  Interpolation = RNG(0.0, 1.0)
	 * R:  Realness = s*s
	 * Xt: s*X1 + (1 - s)*X2
	 */

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
	nn_dim_t dimX =
	{
		.count  = bs,
		.height = dim->height,
		.width  = dim->width,
		.depth  = dim->depth,
	};

	// Y|R = lanczos3(X)|R
	nn_dim_t dimY =
	{
		.count  = bs,
		.height = dim->height/2,
		.width  = dim->width/2,
		.depth  = dim->depth + 1,
	};

	nn_tensor_t* X1;
	X1 = nn_tensor_new(engine, &dimX,
	                   NN_TENSOR_INIT_ZERO,
	                   NN_TENSOR_MODE_IO);
	if(X1 == NULL)
	{
		goto fail_X1;
	}

	nn_tensor_t* X0;
	X0 = nn_tensor_new(engine, &dimX,
	                   NN_TENSOR_INIT_ZERO,
	                   NN_TENSOR_MODE_IO);
	if(X0 == NULL)
	{
		goto fail_X0;
	}

	nn_tensor_t* X;
	X = nn_tensor_new(engine, &dimX,
	                  NN_TENSOR_INIT_ZERO,
	                  NN_TENSOR_MODE_IO);
	if(X == NULL)
	{
		goto fail_X;
	}

	nn_tensor_t* X4;
	X4 = nn_tensor_new(engine, &dimX,
	                   NN_TENSOR_INIT_ZERO,
	                   NN_TENSOR_MODE_IO);
	if(X4 == NULL)
	{
		goto fail_X4;
	}

	nn_tensor_t* Xt;
	Xt = nn_tensor_new(engine, &dimX,
	                   NN_TENSOR_INIT_ZERO,
	                   NN_TENSOR_MODE_IO);
	if(Xt == NULL)
	{
		goto fail_Xt;
	}

	nn_tensor_t* Y;
	Y = nn_tensor_new(engine, &dimY,
	                  NN_TENSOR_INIT_ZERO,
	                  NN_TENSOR_MODE_IO);
	if(Y == NULL)
	{
		goto fail_Y;
	}

	nn_tensor_t* Y3;
	Y3 = nn_tensor_new(engine, &dimY,
	                   NN_TENSOR_INIT_ZERO,
	                   NN_TENSOR_MODE_IO);
	if(Y3 == NULL)
	{
		goto fail_Y3;
	}

	nn_tensor_t* Yt;
	Yt = nn_tensor_new(engine, &dimY,
	                   NN_TENSOR_INIT_ZERO,
	                   NN_TENSOR_MODE_IO);
	if(Yt == NULL)
	{
		goto fail_Yt;
	}

	cifar10_regen1_t* regen1;
	regen1 = cifar10_regen1_new(engine, bs, fc, dim->height,
	                            dim->width, dim->depth);
	if(regen1 == NULL)
	{
		goto fail_regen1;
	}

	cifar10_regen2_t* regen2;
	regen2 = cifar10_regen2_new(engine, bs, fc, dim->height,
	                            dim->width, dim->depth);
	if(regen2 == NULL)
	{
		goto fail_regen2;
	}

	cifar10_regen1_t* regen3;
	regen3 = cifar10_regen1_new(engine, bs, fc, dim->height,
	                            dim->width, dim->depth);
	if(regen3 == NULL)
	{
		goto fail_regen3;
	}

	cifar10_regen2_t* regen4;
	regen4 = cifar10_regen2_new(engine, bs, fc, dim->height,
	                            dim->width, dim->depth);
	if(regen4 == NULL)
	{
		goto fail_regen4;
	}

	int lerp = 1;
	lerp &= nn_coderLayer_lerp(regen1->coder1,
	                           regen2->coder5,
	                           0.16f, 0.84f);
	lerp &= nn_coderLayer_lerp(regen1->coder2,
	                           regen2->coder4,
	                           0.33f, 0.67f);
	lerp &= nn_coderLayer_lerp(regen1->coder3,
	                           regen2->coder3,
	                           0.5f, 0.5f);
	lerp &= nn_coderLayer_lerp(regen1->coder4,
	                           regen2->coder2,
	                           0.67f, 0.33f);
	lerp &= nn_coderLayer_lerp(regen1->coder5,
	                           regen2->coder1,
	                           0.84f, 0.16f);
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
	float    loss4;
	float    sum_loss4 = 0.0f;
	float    min_loss4 = FLT_MAX;
	float    max_loss4 = 0.0f;
	double   t0        = cc_timestamp();
	while(epoch < 20)
	{
		steps = (epoch + 1)*dim->count/bs;
		while(step < steps)
		{
			if(cifar10_sample(cifar10, &rng, bs,
			                  X1, X0, Xt, Yt) == 0)
			{
				goto fail_train;
			}

			if(nn_arch_trainLERP(&regen1->base,
			                     &regen2->base, bs,
			                     Xt, Yt, X, Y) == NULL)
			{
				goto fail_train;
			}

			if(nn_arch_train(&regen3->base,
			                 NN_LAYER_FLAG_TRAIN, bs,
			                 Xt, Yt, Y3) == NULL)
			{
				goto fail_train;
			}

			if(nn_arch_train(&regen4->base,
			                 NN_LAYER_FLAG_TRAIN, bs,
			                 Yt, Xt, X4) == NULL)
			{
				goto fail_train;
			}

			// update loss
			loss1 = nn_arch_loss(&regen1->base);
			loss2 = nn_arch_loss(&regen2->base);
			loss3 = nn_arch_loss(&regen3->base);
			loss4 = nn_arch_loss(&regen4->base);
			sum_loss1 += loss1;
			sum_loss2 += loss2;
			sum_loss3 += loss3;
			sum_loss4 += loss4;
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
			if(loss4 < min_loss4)
			{
				min_loss4 = loss4;
			}
			if(loss4 > max_loss4)
			{
				max_loss4 = loss4;
			}

			// export images
			uint32_t export_interval = 100;
			if((step%export_interval) == (export_interval - 1))
			{
				snprintf(fname, 256, "data/X1-%u-%u-0.png",
				         epoch, step);
				nn_tensor_exportPng(X1, fname, 0, 0, dim->depth - 1,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/X1-%u-%u-1.png",
				         epoch, step);
				nn_tensor_exportPng(X1, fname, 1, 0, dim->depth - 1,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/X0-%u-%u-0.png",
				         epoch, step);
				nn_tensor_exportPng(X0, fname, 0, 0, dim->depth - 1,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/X0-%u-%u-1.png",
				         epoch, step);
				nn_tensor_exportPng(X0, fname, 1, 0, dim->depth - 1,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/X-%u-%u-0.png",
				         epoch, step);
				nn_tensor_exportPng(X, fname, 0, 0, dim->depth - 1,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/X-%u-%u-1.png",
				         epoch, step);
				nn_tensor_exportPng(X, fname, 1, 0, dim->depth - 1,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/X4-%u-%u-0.png",
				         epoch, step);
				nn_tensor_exportPng(X4, fname, 0, 0, dim->depth - 1,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/X4-%u-%u-1.png",
				         epoch, step);
				nn_tensor_exportPng(X4, fname, 1, 0, dim->depth - 1,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/Xt-%u-%u-0.png",
				         epoch, step);
				nn_tensor_exportPng(Xt, fname, 0, 0, dim->depth - 1,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/Xt-%u-%u-1.png",
				         epoch, step);
				nn_tensor_exportPng(Xt, fname, 1, 0, dim->depth - 1,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/Y-%u-%u-0.png",
				         epoch, step);
				nn_tensor_exportPng(Y, fname, 0, 0, dim->depth - 1,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/Y-%u-%u-1.png",
				         epoch, step);
				nn_tensor_exportPng(Y, fname, 1, 0, dim->depth - 1,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/R-%u-%u-0.png",
				         epoch, step);
				nn_tensor_exportPng(Y, fname, 0, dim->depth, dim->depth,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/R-%u-%u-1.png",
				         epoch, step);
				nn_tensor_exportPng(Y, fname, 1, dim->depth, dim->depth,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/Y3-%u-%u-0.png",
				         epoch, step);
				nn_tensor_exportPng(Y3, fname, 0, 0, dim->depth - 1,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/Y3-%u-%u-1.png",
				         epoch, step);
				nn_tensor_exportPng(Y3, fname, 1, 0, dim->depth - 1,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/R3-%u-%u-0.png",
				         epoch, step);
				nn_tensor_exportPng(Y3, fname, 0, dim->depth, dim->depth,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/R3-%u-%u-1.png",
				         epoch, step);
				nn_tensor_exportPng(Y3, fname, 1, dim->depth, dim->depth,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/Yt-%u-%u-0.png",
				         epoch, step);
				nn_tensor_exportPng(Yt, fname, 0, 0, dim->depth - 1,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/Yt-%u-%u-1.png",
				         epoch, step);
				nn_tensor_exportPng(Yt, fname, 1, 0, dim->depth - 1,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/Rt-%u-%u-0.png",
				         epoch, step);
				nn_tensor_exportPng(Yt, fname, 0, dim->depth, dim->depth,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/Rt-%u-%u-1.png",
				         epoch, step);
				nn_tensor_exportPng(Yt, fname, 1, dim->depth, dim->depth,
				                    0.0f, 1.0f);
			}

			// plot loss
			uint32_t plot_interval = 100;
			if((step%plot_interval) == (plot_interval - 1))
			{
				float avg_loss1 = sum_loss1/((float) plot_interval);
				float avg_loss2 = sum_loss2/((float) plot_interval);
				float avg_loss3 = sum_loss3/((float) plot_interval);
				float avg_loss4 = sum_loss4/((float) plot_interval);
				fprintf(fplot, "%u %u %f %f %f %f %f %f %f %f\n",
				        epoch, step,
				        avg_loss1, min_loss1, max_loss1,
				        avg_loss2, min_loss2, max_loss2,
				        avg_loss3, avg_loss4);
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
				sum_loss4 = 0.0f;
				min_loss4 = FLT_MAX;
				max_loss4 = 0.0f;
			}

			// export arch
			uint32_t arch_interval = 1000;
			if((step%arch_interval) == (arch_interval - 1))
			{
				snprintf(fname, 256, "data/regen1-%i-%i.json",
				         epoch, step);
				cifar10_regen1_export(regen1, fname);
			}

			LOGI("epoch=%u, step=%u, elapsed=%lf, loss1=%f, loss2=%f, loss3=%f, loss4=%f",
			     epoch, step, cc_timestamp() - t0,
			     loss1, loss2, loss3, loss4);
			++step;
		}

		++epoch;
	}

	// cleanup
	fclose(fplot);
	cifar10_regen2_delete(&regen4);
	cifar10_regen1_delete(&regen3);
	cifar10_regen2_delete(&regen2);
	cifar10_regen1_delete(&regen1);
	nn_tensor_delete(&Yt);
	nn_tensor_delete(&Y3);
	nn_tensor_delete(&Y);
	nn_tensor_delete(&Xt);
	nn_tensor_delete(&X4);
	nn_tensor_delete(&X);
	nn_tensor_delete(&X0);
	nn_tensor_delete(&X1);
	nn_cifar10_delete(&cifar10);
	nn_engine_delete(&engine);

	// success
	return EXIT_SUCCESS;

	// failure
	fail_train:
		fclose(fplot);
	fail_fplot:
	fail_lerp:
		cifar10_regen2_delete(&regen4);
	fail_regen4:
		cifar10_regen1_delete(&regen3);
	fail_regen3:
		cifar10_regen2_delete(&regen2);
	fail_regen2:
		cifar10_regen1_delete(&regen1);
	fail_regen1:
		nn_tensor_delete(&Yt);
	fail_Yt:
		nn_tensor_delete(&Y3);
	fail_Y3:
		nn_tensor_delete(&Y);
	fail_Y:
		nn_tensor_delete(&Xt);
	fail_Xt:
		nn_tensor_delete(&X4);
	fail_X4:
		nn_tensor_delete(&X);
	fail_X:
		nn_tensor_delete(&X0);
	fail_X0:
		nn_tensor_delete(&X1);
	fail_X1:
		nn_cifar10_delete(&cifar10);
	fail_cifar10:
		nn_engine_delete(&engine);
	return EXIT_FAILURE;
}

vkk_platformInfo_t VKK_PLATFORM_INFO =
{
	.app_name    = "cifar10-regen",
	.app_version =
	{
		.major = 1,
		.minor = 0,
		.patch = 0,
	},
	.app_dir = "cifar10-regen",
	.onMain  = cifar10_regen_onMain,
};
