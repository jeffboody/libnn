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

#define LOG_TAG "cifar10-gan"
#include "libcc/math/cc_float.h"
#include "libcc/cc_log.h"
#include "libcc/cc_memory.h"
#include "libnn/cifar10-denoise/cifar10_denoise.h"
#include "libnn/cifar10-disc/cifar10_disc.h"
#include "libnn/cifar10/nn_cifar10.h"
#include "libnn/nn_arch.h"
#include "libnn/nn_engine.h"
#include "libnn/nn_layer.h"
#include "libnn/nn_tensor.h"
#include "libvkk/vkk_platform.h"

/***********************************************************
* callbacks                                                *
***********************************************************/

#ifdef CIFAR10_USE_INTERPOLATE
static void
cifar10_gan_interpolateYt(cc_rngUniform_t* rng,
                          nn_tensor_t* Yt11,
                          nn_tensor_t* Yt10,
                          nn_tensor_t* Ytr,
                          nn_tensor_t* Yr)
{
	ASSERT(rng);
	ASSERT(Yt11);
	ASSERT(Yt10);
	ASSERT(Ytr);
	ASSERT(Yr);

	nn_dim_t* dimYtXX = nn_tensor_dim(Yt10);
	nn_dim_t* dimYr   = nn_tensor_dim(Yr);
	uint32_t  bs      = dimYr->count;
	uint32_t  bs2     = bs/2;

	// interpolate real samples
	uint32_t n;
	uint32_t i;
	uint32_t j;
	float    s;
	float    ss;
	float    ytr;
	float    yr;
	for(n = 0; n < bs2; ++n)
	{
		s  = cc_rngUniform_rand2F(rng, 0.0f, 1.0f);
		ss = s*s;

		for(i = 0; i < dimYtXX->height; ++i)
		{
			for(j = 0; j < dimYtXX->width; ++j)
			{
				nn_tensor_set(Yt11, n, i, j, 0, ss);
				nn_tensor_set(Yt10, n, i, j, 0, ss);
			}
		}

		for(i = 0; i < dimYr->height; ++i)
		{
			for(j = 0; j < dimYr->width; ++j)
			{
				ytr = nn_tensor_get(Ytr, n, i, j, 0);
				yr  = nn_tensor_get(Yr,  n, i, j, 0);
				nn_tensor_set(Ytr, n, i, j, 0, s*ytr + (1.0f - s)*yr);
			}
		}
	}
}
#endif

static void
cifar10_gan_initYt(nn_tensor_t* Yt, uint32_t n0,
                   uint32_t count, float yt)
{
	ASSERT(Yt);

	nn_dim_t* dim = nn_tensor_dim(Yt);

	uint32_t n;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(n = n0; n < n0 + count; ++n)
	{
		for(i = 0; i < dim->height; ++i)
		{
			for(j = 0; j < dim->width; ++j)
			{
				for(k = 0; k < dim->depth; ++k)
				{
					nn_tensor_set(Yt, n, i, j, k, yt);
				}
			}
		}
	}
}

static int
cifar10_gan_onMain(vkk_engine_t* ve, int argc, char** argv)
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

	nn_dim_t* dimXt = nn_tensor_dim(cifar10->images);
	uint32_t  bs    = 32;
	uint32_t  bs2   = bs/2;

	cifar10_denoise_t* dn;
	dn = cifar10_denoise_new(engine, bs2, 32, dimXt->height,
	                         dimXt->width, dimXt->depth,
	                         0.0, 0.0);
	if(dn == NULL)
	{
		goto fail_dn;
	}

	nn_archState_t* dn_state = &dn->base.state;

	cifar10_disc_t* disc;
	disc = cifar10_disc_new(engine, bs, 32, dimXt->height,
	                        dimXt->width, dimXt->depth);
	if(disc == NULL)
	{
		goto fail_disc;
	}

	nn_dim_t dimX =
	{
		.count  = bs2,
		.height = dimXt->height,
		.width  = dimXt->width,
		.depth  = dimXt->depth,
	};

	nn_dim_t dimXd =
	{
		.count  = bs,
		.height = dimXt->height,
		.width  = dimXt->width,
		.depth  = 2*dimXt->depth,
	};

	nn_dim_t dimY =
	{
		.count  = bs,
		.height = dimXt->height/4,
		.width  = dimXt->height/4,
		.depth  = 1,
	};

	nn_tensor_t* Cg;
	Cg = nn_tensor_new(engine, &dimX,
	                   NN_TENSOR_INIT_ZERO,
	                   NN_TENSOR_MODE_IO);
	if(Cg == NULL)
	{
		goto fail_Cg;
	}

	nn_tensor_t* Cr;
	Cr = nn_tensor_new(engine, &dimX,
	                   NN_TENSOR_INIT_ZERO,
	                   NN_TENSOR_MODE_IO);
	if(Cr == NULL)
	{
		goto fail_Cr;
	}

	nn_tensor_t* Ytg;
	Ytg = nn_tensor_new(engine, &dimX,
	                    NN_TENSOR_INIT_ZERO,
	                    NN_TENSOR_MODE_IO);
	if(Ytg == NULL)
	{
		goto fail_Ytg;
	}

	nn_tensor_t* Ytr;
	Ytr = nn_tensor_new(engine, &dimX,
	                    NN_TENSOR_INIT_ZERO,
	                    NN_TENSOR_MODE_IO);
	if(Ytr == NULL)
	{
		goto fail_Ytr;
	}

	nn_tensor_t* Yr;
	Yr = nn_tensor_new(engine, &dimX,
	                   NN_TENSOR_INIT_ZERO,
	                   NN_TENSOR_MODE_IO);
	if(Yr == NULL)
	{
		goto fail_Yr;
	}

	nn_tensor_t* Yt11;
	Yt11 = nn_tensor_new(engine, &dimY,
	                     NN_TENSOR_INIT_ZERO,
	                     NN_TENSOR_MODE_IO);
	if(Yt11 == NULL)
	{
		goto fail_Yt11;
	}
	cifar10_gan_initYt(Yt11, 0, bs, 1.0f);

	nn_tensor_t* Yt10;
	Yt10 = nn_tensor_new(engine, &dimY,
	                     NN_TENSOR_INIT_ZERO,
	                     NN_TENSOR_MODE_IO);
	if(Yt10 == NULL)
	{
		goto fail_Yt10;
	}
	cifar10_gan_initYt(Yt10, 0,   bs2, 1.0f);
	cifar10_gan_initYt(Yt10, bs2, bs2, 0.0f);

	nn_tensor_t* dL_dYb;
	dL_dYb = nn_tensor_new(engine, &dimX,
	                       NN_TENSOR_INIT_ZERO,
	                       NN_TENSOR_MODE_IO);
	if(dL_dYb == NULL)
	{
		goto fail_dL_dYb;
	}

	nn_tensor_t* dL_dYg;
	dL_dYg = nn_tensor_new(engine, &dimX,
	                       NN_TENSOR_INIT_ZERO,
	                       NN_TENSOR_MODE_IO);
	if(dL_dYg == NULL)
	{
		goto fail_dL_dYg;
	}

	nn_tensor_t* dL_dYd;
	dL_dYd = nn_tensor_new(engine, &dimXd,
	                       NN_TENSOR_INIT_ZERO,
	                       NN_TENSOR_MODE_IO);
	if(dL_dYd == NULL)
	{
		goto fail_dL_dYd;
	}

	nn_tensor_t* Yg;
	Yg = nn_tensor_new(engine, &dimX,
	                   NN_TENSOR_INIT_ZERO,
	                   NN_TENSOR_MODE_IO);
	if(Yg == NULL)
	{
		goto fail_Yg;
	}

	nn_tensor_t* Yd;
	Yd = nn_tensor_new(engine, &dimY,
	                   NN_TENSOR_INIT_ZERO,
	                   NN_TENSOR_MODE_IO);
	if(Yd == NULL)
	{
		goto fail_Yd;
	}

	FILE* fplot = fopen("data/plot.dat", "w");
	if(fplot == NULL)
	{
		goto fail_fplot;
	}

	// training
	uint32_t epoch     = 0;
	uint32_t min_epoch = 1;
	uint32_t step      = 0;
	uint32_t steps;
	char     fname[256];
	float    loss       = 0.0f;
	float    sum_loss   = 0.0f;
	float    min_loss   = FLT_MAX;
	float    max_loss   = 0.0f;
	float    d_loss     = 0.0f;
	float    d_sum_loss = 0.0f;
	float    d_min_loss = FLT_MAX;
	float    d_max_loss = 0.0f;
	float    g_loss     = 0.0f;
	float    g_sum_loss = 0.0f;
	float    g_min_loss = FLT_MAX;
	float    g_max_loss = 0.0f;
	while(epoch < 20)
	{
		steps = (epoch + 1)*dimXt->count/bs;
		while(step < steps)
		{
			// randomly sample data
			cifar10_denoise_sampleXt2(dn, cifar10->images,
			                          Cg, Ytg);

			// training
			if(epoch < min_epoch)
			{
				if(nn_arch_train(&dn->base, NN_LAYER_FLAG_TRAIN,
				                 bs2, Cg, Ytg,
				                 Yg) == NULL)
				{
					goto fail_train;
				}

				loss = nn_arch_loss(&dn->base);
			}
			else
			{
				cifar10_denoise_sampleXt2(dn, cifar10->images,
				                          Cr, Ytr);

				#ifdef CIFAR10_USE_INTERPOLATE
				if(nn_arch_predict(&dn->base, bs2, Cr, Yr) == 0)
				{
					goto fail_train;
				}

				cifar10_gan_interpolateYt(&rng, Yt11, Yt10, Ytr, Yr);
				#endif

				if(nn_arch_trainFairCGAN(&dn->base, &disc->base, bs,
				                         Cg, NULL, Cr, NULL, Ytg, Ytr,
				                         Yt11, Yt10, dL_dYb, dL_dYg,
				                         dL_dYd, Yg, Yd, &loss, &g_loss,
				                         &d_loss) == NULL)
				{
					goto fail_train;
				}

				// update generator loss
				g_sum_loss += g_loss;
				if(g_loss < g_min_loss)
				{
					g_min_loss = g_loss;
				}
				if(g_loss > g_max_loss)
				{
					g_max_loss = g_loss;
				}

				// update discriminator loss
				d_sum_loss += d_loss;
				if(d_loss < d_min_loss)
				{
					d_min_loss = d_loss;
				}
				if(d_loss > d_max_loss)
				{
					d_max_loss = d_loss;
				}
			}

			// update loss
			sum_loss += loss;
			if(loss < min_loss)
			{
				min_loss = loss;
			}
			if(loss > max_loss)
			{
				max_loss = loss;
			}

			// export images
			uint32_t image_interval = 100;
			uint32_t xd             = dimXt->depth;
			if((step%image_interval) == (image_interval - 1))
			{
				snprintf(fname, 256, "data/Cg-%u-%u.png",
				         epoch, step);
				nn_tensor_exportPng(Cg, fname, 0, 0, xd - 1,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/Ytg-%u-%u.png",
				         epoch, step);
				nn_tensor_exportPng(Ytg, fname, 0, 0, xd - 1,
				                    0.0f, 1.0f);
				snprintf(fname, 256, "data/Yg-%u-%u.png",
				         epoch, step);
				nn_tensor_exportPng(Yg, fname, 0, 0, xd - 1,
				                    0.0f, 1.0f);

				if(epoch >= min_epoch)
				{
					snprintf(fname, 256, "data/Cr-%u-%u.png",
					         epoch, step);
					nn_tensor_exportPng(Cr, fname, 0, 0, xd - 1,
					                    0.0f, 1.0f);
					snprintf(fname, 256, "data/Ytr-%u-%u.png",
					         epoch, step);
					nn_tensor_exportPng(Ytr, fname, 0, 0, xd - 1,
					                    0.0f, 1.0f);
					snprintf(fname, 256, "data/Yr-%u-%u.png",
					         epoch, step);
					nn_tensor_exportPng(Yr, fname, 0, 0, xd - 1,
					                    0.0f, 1.0f);
					snprintf(fname, 256, "data/dL_dYb-%u-%u.png",
					         epoch, step);
					nn_tensor_exportPng(dL_dYb, fname, 0, 0, xd - 1,
					                    -1.0f, 1.0f);
					snprintf(fname, 256, "data/dL_dYg-%u-%u.png",
					         epoch, step);
					nn_tensor_exportPng(dL_dYg, fname, 0, 0, xd - 1,
					                    -1.0f, 1.0f);
					snprintf(fname, 256, "data/dL_dYdr-%u-%u.png",
					         epoch, step);
					nn_tensor_exportPng(dL_dYd, fname, 0, 0, xd - 1,
					                    -1.0f, 1.0f);
					snprintf(fname, 256, "data/dL_dYdCr-%u-%u.png",
					         epoch, step);
					nn_tensor_exportPng(dL_dYd, fname, 0, xd, 2*xd - 1,
					                    -1.0f, 1.0f);
					snprintf(fname, 256, "data/dL_dYdg-%u-%u.png",
					         epoch, step);
					nn_tensor_exportPng(dL_dYd, fname, bs2, 0, xd - 1,
					                    -1.0f, 1.0f);
					snprintf(fname, 256, "data/dL_dYdCg-%u-%u.png",
					         epoch, step);
					nn_tensor_exportPng(dL_dYd, fname, bs2, xd, 2*xd - 1,
					                    -1.0f, 1.0f);
					snprintf(fname, 256, "data/Yd-%u-%u-%u.png",
					         epoch, step, 0);
					nn_tensor_exportPng(Yd, fname, 0, 0, 0, 0.0f, 1.0f);
					snprintf(fname, 256, "data/Yd-%u-%u-%u.png",
					         epoch, step, bs2);
					nn_tensor_exportPng(Yd, fname, bs2, 0, 0, 0.0f, 1.0f);
				}
			}

			// plot loss
			uint32_t plot_interval = 100;
			if((step%plot_interval) == (plot_interval - 1))
			{
				if(epoch < min_epoch)
				{
					g_min_loss = 0.0f;
					d_min_loss = 0.0f;
				}

				// scale blend_factor range for visualization
				float avg_loss   = sum_loss/((float) plot_interval);
				float d_avg_loss = d_sum_loss/((float) plot_interval);
				float g_avg_loss = g_sum_loss/((float) plot_interval);
				fprintf(fplot, "%u %u %f %f %f %f %f %f %f %f %f %f\n",
				        epoch, step,
				        avg_loss, min_loss, max_loss,
				        g_avg_loss,  g_min_loss,  g_max_loss,
				        d_avg_loss,  d_min_loss,  d_max_loss,
				        dn_state->gan_blend_factor/10.0f);
				fflush(fplot);

				// reset loss
				sum_loss   = 0.0f;
				min_loss   = FLT_MAX;
				max_loss   = 0.0f;
				g_sum_loss = 0.0f;
				g_min_loss = FLT_MAX;
				g_max_loss = 0.0f;
				d_sum_loss = 0.0f;
				d_min_loss = FLT_MAX;
				d_max_loss = 0.0f;
			}

			// export arch
			uint32_t arch_interval = 1000;
			if((step%arch_interval) == (arch_interval - 1))
			{
				snprintf(fname, 256, "data/disc-%i-%i.json",
				         epoch, step);
				cifar10_disc_export(disc, fname);

				snprintf(fname, 256, "data/dn-%i-%i.json",
				         epoch, step);
				cifar10_denoise_export(dn, fname);
			}

			LOGI("epoch=%u, step=%u, loss=%f, g_loss=%f, d_loss=%f, blend_factor=%f",
			     epoch, step, loss, g_loss, d_loss,
			     dn_state->gan_blend_factor);
			++step;
		}

		++epoch;
	}

	// cleanup
	fclose(fplot);
	nn_tensor_delete(&Yd);
	nn_tensor_delete(&Yg);
	nn_tensor_delete(&dL_dYd);
	nn_tensor_delete(&dL_dYg);
	nn_tensor_delete(&dL_dYb);
	nn_tensor_delete(&Yt10);
	nn_tensor_delete(&Yt11);
	nn_tensor_delete(&Yr);
	nn_tensor_delete(&Ytr);
	nn_tensor_delete(&Ytg);
	nn_tensor_delete(&Cr);
	nn_tensor_delete(&Cg);
	cifar10_disc_delete(&disc);
	cifar10_denoise_delete(&dn);
	nn_cifar10_delete(&cifar10);
	nn_engine_delete(&engine);

	// success
	return EXIT_SUCCESS;

	// failure
	fail_train:
		fclose(fplot);
	fail_fplot:
		nn_tensor_delete(&Yd);
	fail_Yd:
		nn_tensor_delete(&Yg);
	fail_Yg:
		nn_tensor_delete(&dL_dYd);
	fail_dL_dYd:
		nn_tensor_delete(&dL_dYg);
	fail_dL_dYg:
		nn_tensor_delete(&dL_dYb);
	fail_dL_dYb:
		nn_tensor_delete(&Yt10);
	fail_Yt10:
		nn_tensor_delete(&Yt11);
	fail_Yt11:
		nn_tensor_delete(&Yr);
	fail_Yr:
		nn_tensor_delete(&Ytr);
	fail_Ytr:
		nn_tensor_delete(&Ytg);
	fail_Ytg:
		nn_tensor_delete(&Cr);
	fail_Cr:
		nn_tensor_delete(&Cg);
	fail_Cg:
		cifar10_disc_delete(&disc);
	fail_disc:
		cifar10_denoise_delete(&dn);
	fail_dn:
		nn_cifar10_delete(&cifar10);
	fail_cifar10:
		nn_engine_delete(&engine);
	return EXIT_FAILURE;
}

vkk_platformInfo_t VKK_PLATFORM_INFO =
{
	.app_name    = "cifar10-gan",
	.app_version =
	{
		.major = 1,
		.minor = 0,
		.patch = 0,
	},
	.app_dir = "cifar10-gan",
	.onMain  = cifar10_gan_onMain,
};
