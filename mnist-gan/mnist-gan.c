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

#define LOG_TAG "mnist-gan"
#include "libcc/math/cc_float.h"
#include "libcc/cc_log.h"
#include "libcc/cc_memory.h"
#include "libnn/mnist-denoise/mnist_denoise.h"
#include "libnn/mnist-disc/mnist_disc.h"
#include "libnn/mnist/nn_mnist.h"
#include "libnn/nn_arch.h"
#include "libnn/nn_engine.h"
#include "libnn/nn_tensor.h"
#include "libvkk/vkk_platform.h"

/***********************************************************
* callbacks                                                *
***********************************************************/

static void mnist_gan_initYt11(nn_tensor_t* Yt11)
{
	ASSERT(Yt11);

	nn_dim_t* dim = nn_tensor_dim(Yt11);
	uint32_t  bs  = dim->count;

	// all ones
	uint32_t n;
	uint32_t i;
	uint32_t j;
	for(n = 0; n < bs; ++n)
	{
		for(i = 0; i < dim->height; ++i)
		{
			for(j = 0; j < dim->width; ++j)
			{
				nn_tensor_set(Yt11, n, i, j, 0, 1.0f);
			}
		}
	}
}

static void mnist_gan_initYt10(nn_tensor_t* Yt10)
{
	ASSERT(Yt10);

	nn_dim_t* dim = nn_tensor_dim(Yt10);
	uint32_t  bs  = dim->count;
	uint32_t  bs2 = bs/2;

	// half ones
	uint32_t n;
	uint32_t i;
	uint32_t j;
	for(n = 0; n < bs2; ++n)
	{
		for(i = 0; i < dim->height; ++i)
		{
			for(j = 0; j < dim->width; ++j)
			{
				nn_tensor_set(Yt10, n, i, j, 0, 1.0f);
			}
		}
	}

	// half zeros
	for(n = bs2; n < bs; ++n)
	{
		for(i = 0; i < dim->height; ++i)
		{
			for(j = 0; j < dim->width; ++j)
			{
				nn_tensor_set(Yt10, n, i, j, 0, 0.0f);
			}
		}
	}
}

static int
mnist_gan_onMain(vkk_engine_t* ve, int argc, char** argv)
{
	ASSERT(ve);

	nn_engine_t* engine = nn_engine_new(ve);
	if(engine == NULL)
	{
		return EXIT_FAILURE;
	}

	nn_tensor_t* Xt = nn_mnist_load(engine);
	if(Xt == NULL)
	{
		goto fail_Xt;
	}

	nn_dim_t* dimXt = nn_tensor_dim(Xt);
	uint32_t  xh    = dimXt->height;
	uint32_t  xw    = dimXt->width;
	uint32_t  count = dimXt->count;
	uint32_t  bs    = 32;
	uint32_t  bs2   = bs/2;

	mnist_denoise_t* dn;
	dn = mnist_denoise_new(engine, bs2, 32, xh, xw, 0.1, 0.1);
	if(dn == NULL)
	{
		goto fail_dn;
	}

	nn_archState_t* dn_state = &dn->base.state;

	mnist_disc_t* disc;
	disc = mnist_disc_new(engine, bs, 32, xh, xw);
	if(disc == NULL)
	{
		goto fail_disc;
	}

	nn_dim_t dimX =
	{
		.count  = bs2,
		.height = xh,
		.width  = xw,
		.depth  = 1,
	};

	nn_dim_t dimY =
	{
		.count  = bs,
		.height = xh/4,
		.width  = xh/4,
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

	nn_tensor_t* Yt11;
	Yt11 = nn_tensor_new(engine, &dimY,
	                     NN_TENSOR_INIT_ZERO,
	                     NN_TENSOR_MODE_IO);
	if(Yt11 == NULL)
	{
		goto fail_Yt11;
	}
	mnist_gan_initYt11(Yt11);

	nn_tensor_t* Yt10;
	Yt10 = nn_tensor_new(engine, &dimY,
	                     NN_TENSOR_INIT_ZERO,
	                     NN_TENSOR_MODE_IO);
	if(Yt10 == NULL)
	{
		goto fail_Yt10;
	}
	mnist_gan_initYt10(Yt10);

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

	nn_tensor_t* dL_dYdg;
	dL_dYdg = nn_tensor_new(engine, &dimX,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_IO);
	if(dL_dYdg == NULL)
	{
		goto fail_dL_dYdg;
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
	uint32_t epoch = 0;
	uint32_t step  = 0;
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
		steps = (epoch + 1)*count/bs;
		while(step < steps)
		{
			mnist_denoise_sampleXt2(dn, Xt, Cg, Ytg);
			mnist_denoise_sampleXt2(dn, Xt, Cr, Ytr);

			if(nn_arch_trainFairCGAN(&dn->base, &disc->base, bs,
			                         Cg, NULL, Cr, NULL, Ytg, Ytr,
			                         Yt11, Yt10, dL_dYb, dL_dYg,
			                         dL_dYdg, Yg, Yd, &loss, &g_loss,
			                         &d_loss) == NULL)
			{
				goto fail_train;
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

			// export images
			uint32_t image_interval = 100;
			if((step%image_interval) == (image_interval - 1))
			{
				snprintf(fname, 256, "data/Cg-%u-%u.png",
				         epoch, step);
				nn_tensor_exportPng(Cg, fname, 0, 0, 0, 0.0f, 1.0f);
				snprintf(fname, 256, "data/Ytg-%u-%u.png",
				         epoch, step);
				nn_tensor_exportPng(Ytg, fname, 0, 0, 0, 0.0f, 1.0f);
				snprintf(fname, 256, "data/Yg-%u-%u.png",
				         epoch, step);
				nn_tensor_exportPng(Yg, fname, 0, 0, 0, 0.0f, 1.0f);
				snprintf(fname, 256, "data/dL_dYb-%u-%u.png",
				         epoch, step);
				nn_tensor_exportPng(dL_dYb, fname, 0, 0, 0, -1.0f, 1.0f);
				snprintf(fname, 256, "data/dL_dYg-%u-%u.png",
				         epoch, step);
				nn_tensor_exportPng(dL_dYg, fname, 0, 0, 0, -1.0f, 1.0f);
				snprintf(fname, 256, "data/dL_dYdg-%u-%u.png",
				         epoch, step);
				nn_tensor_exportPng(dL_dYdg, fname, 0, 0, 0, -1.0f, 1.0f);
				snprintf(fname, 256, "data/Yd-%u-%u-%u.png",
				         epoch, step, 0);
				nn_tensor_exportPng(Yd, fname, 0, 0, 0, 0.0f, 1.0f);
				snprintf(fname, 256, "data/Yd-%u-%u-%u.png",
				         epoch, step, bs2);
				nn_tensor_exportPng(Yd, fname, bs2, 0, 0, 0.0f, 1.0f);
			}

			// plot loss
			uint32_t plot_interval = 100;
			if((step%plot_interval) == (plot_interval - 1))
			{
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
				mnist_disc_export(disc, fname);

				snprintf(fname, 256, "data/dn-%i-%i.json",
				         epoch, step);
				mnist_denoise_export(dn, fname);
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
	nn_tensor_delete(&dL_dYdg);
	nn_tensor_delete(&dL_dYg);
	nn_tensor_delete(&dL_dYb);
	nn_tensor_delete(&Yt10);
	nn_tensor_delete(&Yt11);
	nn_tensor_delete(&Ytr);
	nn_tensor_delete(&Ytg);
	nn_tensor_delete(&Cr);
	nn_tensor_delete(&Cg);
	mnist_disc_delete(&disc);
	mnist_denoise_delete(&dn);
	nn_tensor_delete(&Xt);
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
		nn_tensor_delete(&dL_dYdg);
	fail_dL_dYdg:
		nn_tensor_delete(&dL_dYg);
	fail_dL_dYg:
		nn_tensor_delete(&dL_dYb);
	fail_dL_dYb:
		nn_tensor_delete(&Yt10);
	fail_Yt10:
		nn_tensor_delete(&Yt11);
	fail_Yt11:
		nn_tensor_delete(&Ytr);
	fail_Ytr:
		nn_tensor_delete(&Ytg);
	fail_Ytg:
		nn_tensor_delete(&Cr);
	fail_Cr:
		nn_tensor_delete(&Cg);
	fail_Cg:
		mnist_disc_delete(&disc);
	fail_disc:
		mnist_denoise_delete(&dn);
	fail_dn:
		nn_tensor_delete(&Xt);
	fail_Xt:
		nn_engine_delete(&engine);
	return EXIT_FAILURE;
}

vkk_platformInfo_t VKK_PLATFORM_INFO =
{
	.app_name    = "mnist-gan",
	.app_version =
	{
		.major = 1,
		.minor = 0,
		.patch = 0,
	},
	.app_dir = "mnist-gan",
	.onMain  = mnist_gan_onMain,
};
