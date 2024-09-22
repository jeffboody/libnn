/* * Copyright (c) 2024 Jeff Boody
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

/*
 * Implementation Based On:
 * How to Develop a GAN for Generating MNIST Handwritten Digits
 * https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/
 *
 * Development Status:
 * WARNING: This implementation using libnn is under
 * development and currently does not work correctly.
 */

#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define LOG_TAG "mnist"
#include "libcc/cc_log.h"
#include "libcc/cc_memory.h"
#include "libcc/cc_timestamp.h"
#include "libnn/mnist/nn_mnist.h"
#include "libnn/nn_engine.h"
#include "libnn/nn_loss.h"
#include "libnn/nn_tensor.h"
#include "libvkk/vkk_platform.h"
#include "mnist_ganDisc.h"
#include "mnist_ganGen.h"

/***********************************************************
* private                                                  *
***********************************************************/

static int
mnist_gan_loadGX(cc_rngUniform_t* rng, nn_tensor_t* GXio,
                 nn_tensor_t* GX)
{
	ASSERT(rng);
	ASSERT(GXio);
	ASSERT(GX);

	nn_dim_t* dim1 = nn_tensor_dim(GXio);
	nn_dim_t* dim2 = nn_tensor_dim(GX);

	if(nn_dim_sizeEquals(dim1, dim2) == 0)
	{
		LOGE("invalid count=%u:%u, height=%u:%u, width=%u:%u, depth=%u:%u",
		     dim1->count,  dim2->count,
		     dim1->height, dim2->height,
		     dim1->width,  dim2->width,
		     dim1->depth,  dim2->depth);
		return 0;
	}

	// z / uniform distribution
	uint32_t n;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	float    f;
	for(n = 0; n < dim1->count; ++n)
	{
		for(i = 0; i < dim1->height; ++i)
		{
			for(j = 0; j < dim1->width; ++j)
			{
				for(k = 0; k < dim1->depth; ++k)
				{
					f = cc_rngUniform_rand2F(rng, 0.0f, 1.0f);
					nn_tensor_ioSet(GXio, n, i, j, k, f);
				}
			}
		}
	}

	return nn_tensor_copy(GXio, GX, 0, 0, dim1->count);
}

static int
mnist_gan_loadDX(cc_rngUniform_t* rng, nn_tensor_t* Xt,
                 nn_tensor_t* DXio, nn_tensor_t* DX)
{
	ASSERT(rng);
	ASSERT(Xt);
	ASSERT(DXio);
	ASSERT(DX);

	nn_dim_t* dimXt   = nn_tensor_dim(Xt);
	nn_dim_t* dimDXio = nn_tensor_dim(DXio);
	nn_dim_t* dimDX   = nn_tensor_dim(DX);

	if((dimDXio->count != dimDX->count)    ||
	   (dimXt->height  != 28)              ||
	   (dimXt->height  != dimDXio->height) ||
	   (dimXt->height  != dimDX->height)   ||
	   (dimXt->width   != 28)              ||
	   (dimXt->width   != dimDXio->width)  ||
	   (dimXt->width   != dimDX->width)    ||
	   (dimXt->depth   != 1)               ||
	   (dimDXio->depth != 1)               ||
	   (dimDX->depth   != 1))
	{
		LOGE("invalid count=%u:%u, height=%u:%u:%u, width=%u:%u:%u, depth=%u:%u:%u",
		     dimDXio->count, dimDX->count,
		     dimXt->height, dimDXio->height, dimDX->height,
		     dimXt->width,  dimDXio->width,  dimDX->width,
		     dimXt->depth,  dimDXio->depth,  dimDX->depth);
		return 0;
	}

	uint32_t m;
	uint32_t n;
	uint32_t max = dimXt->count - 1;
	for(m = 0; m < dimDX->count; ++m)
	{
		n = cc_rngUniform_rand2U(rng, 0, max);
		if(nn_tensor_copy(Xt, DXio, n, m, 1) == 0)
		{
			return 0;
		}
	}

	return nn_tensor_copy(DXio, DX, 0, 0, dimDX->count);
}

static int
mnist_gan_initDY(nn_engine_t* engine, nn_tensor_t* DY,
                 uint32_t bs, float a, float b)
{
	ASSERT(engine);
	ASSERT(DY);

	uint32_t bs2 = bs/2;

	// DY = a|b
	if((nn_engine_computeBegin(engine) == 0) ||
	   (nn_tensor_computeFill(DY, VKK_HAZARD_NONE,
	                          0, bs2, a) == 0)  ||
	   (nn_tensor_computeFill(DY, VKK_HAZARD_RAW,
	                          bs2, bs2, b) == 0))
	{
		nn_engine_computeEnd(engine);
		return 0;
	}
	nn_engine_computeEnd(engine);

	return 1;
}

/***********************************************************
* callbacks                                                *
***********************************************************/

static int
mnist_gan_onMain(vkk_engine_t* ve, int argc,
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

	nn_tensor_t* Xt = nn_mnist_load(engine, -1.0f, 1.0f);
	if(Xt == NULL)
	{
		goto fail_Xt;
	}

	nn_dim_t* dimXt = nn_tensor_dim(Xt);
	uint32_t  count = dimXt->count;
	uint32_t  xh    = dimXt->height;
	uint32_t  xw    = dimXt->width;
	uint32_t  xd    = dimXt->depth;

	if((xh != 28) || (xw != 28) || (xd != 1))
	{
		LOGE("invalid xh=%u, xw=%u, xd=%u", xh, xw, xd);
		goto fail_dim;
	}

	uint32_t bs  = 32;
	uint32_t bs2 = bs/2;

	nn_dim_t dimGX =
	{
		.count  = bs,
		.height = 1,
		.width  = 1,
		.depth  = 100,
	};

	nn_dim_t dimGY =
	{
		.count  = bs,
		.height = xh,
		.width  = xw,
		.depth  = xd,
	};

	nn_dim_t dimDX =
	{
		.count  = bs,
		.height = xh,
		.width  = xw,
		.depth  = xd,
	};

	nn_dim_t dimDY =
	{
		.count  = bs,
		.height = 1,
		.width  = 1,
		.depth  = 1,
	};

	nn_tensor_t* GXio;
	GXio = nn_tensor_new(engine, &dimGX,
	                     NN_TENSOR_INIT_ZERO,
	                     NN_TENSOR_MODE_IO);
	if(GXio == NULL)
	{
		goto fail_GXio;
	}

	nn_tensor_t* GX;
	GX = nn_tensor_new(engine, &dimGX,
	                   NN_TENSOR_INIT_ZERO,
	                   NN_TENSOR_MODE_COMPUTE);
	if(GX == NULL)
	{
		goto fail_GX;
	}

	nn_tensor_t* GYio;
	GYio = nn_tensor_new(engine, &dimGY,
	                     NN_TENSOR_INIT_ZERO,
	                     NN_TENSOR_MODE_IO);
	if(GYio == NULL)
	{
		goto fail_GYio;
	}

	nn_tensor_t* DXio;
	DXio = nn_tensor_new(engine, &dimDX,
	                     NN_TENSOR_INIT_ZERO,
	                     NN_TENSOR_MODE_IO);
	if(DXio == NULL)
	{
		goto fail_DXio;
	}

	nn_tensor_t* DX;
	DX = nn_tensor_new(engine, &dimDX,
	                   NN_TENSOR_INIT_ZERO,
	                   NN_TENSOR_MODE_COMPUTE);
	if(DX == NULL)
	{
		goto fail_DX;
	}

	nn_tensor_t* DYio;
	DYio = nn_tensor_new(engine, &dimDY,
	                     NN_TENSOR_INIT_ZERO,
	                     NN_TENSOR_MODE_IO);
	if(DYio == NULL)
	{
		goto fail_DYio;
	}

	nn_tensor_t* DY01;
	DY01 = nn_tensor_new(engine, &dimDY,
	                     NN_TENSOR_INIT_ZERO,
	                     NN_TENSOR_MODE_COMPUTE);
	if(DY01 == NULL)
	{
		goto fail_DY01;
	}

	if(mnist_gan_initDY(engine, DY01, bs, 0.0f, 1.0f) == 0)
	{
		goto fail_initDY01;
	}

	nn_tensor_t* DY11;
	DY11 = nn_tensor_new(engine, &dimDY,
	                     NN_TENSOR_INIT_ZERO,
	                     NN_TENSOR_MODE_COMPUTE);
	if(DY11 == NULL)
	{
		goto fail_DY11;
	}

	if(mnist_gan_initDY(engine, DY11, bs, 1.0f, 1.0f) == 0)
	{
		goto fail_initDY11;
	}

	mnist_ganGen_t* G;
	G = mnist_ganGen_new(engine, bs);
	if(G == NULL)
	{
		goto fail_G;
	}

	mnist_ganDisc_t* D;
	D = mnist_ganDisc_new(engine, bs);
	if(D == NULL)
	{
		goto fail_D;
	}

	nn_loss_t* DL;
	DL = nn_loss_new(engine, &dimDY, NN_LOSS_FN_BCE);
	if(DL == NULL)
	{
		goto fail_DL;
	}

	FILE* fplot = fopen("data/plot.dat", "w");
	if(fplot == NULL)
	{
		goto fail_fplot;
	}

	// training
	double   t0    = cc_timestamp();
	uint32_t epoch = 0;
	uint32_t step  = 0;
	uint32_t steps;
	float    D_loss     = 0.0f;
	float    D_sum_loss = 0.0f;
	float    D_min_loss = FLT_MAX;
	float    D_max_loss = 0.0f;
	float    G_loss     = 0.0f;
	float    G_sum_loss = 0.0f;
	float    G_min_loss = FLT_MAX;
	float    G_max_loss = 0.0f;
	while(epoch < 20)
	{
		steps = (epoch + 1)*count/bs;
		while(step < steps)
		{
			/*
			 * train D
			 */

			// load GX
			if(mnist_gan_loadGX(&rng, GXio, GX) == 0)
			{
				goto fail_train;
			}

			// load DX
			if(mnist_gan_loadDX(&rng, Xt, DXio, DX) == 0)
			{
				goto fail_train;
			}

			// GX > G > GY
			nn_tensor_t* GY;
			GY = nn_arch_forwardPass(&G->base,
			                         NN_ARCH_FLAG_FP_BN_COMPUTE,
			                         bs, GX);
			if(GY == NULL)
			{
				goto fail_train;
			}

			// DX = GY|DX
			if((nn_engine_computeBegin(engine) == 0) ||
			   (nn_tensor_computeCopy(GY, DX, VKK_HAZARD_NONE,
			                          0, 0, bs2) == 0))
			{
				nn_engine_computeEnd(engine);
				goto fail_train;
			}
			nn_engine_computeEnd(engine);

			// DX > D > DY
			nn_tensor_t* DY;
			DY = nn_arch_forwardPass(&D->base,
			                         0, bs, DX);
			if(DY == NULL)
			{
				goto fail_train;
			}

			// DY + DY01 > DL > DL_dL_dY
			nn_tensor_t* DL_dL_dY;
			DL_dL_dY = nn_loss_pass(DL, 0, bs, DY, DY01);
			if(DL_dL_dY == NULL)
			{
				goto fail_train;
			}
			D_loss = nn_loss_loss(DL);

			// DL_dL_dY > D > D_dL_dY
			LOGD("D: DL_dL_dY > D > D_dL_dY");
			nn_tensor_t* D_dL_dY;
			D_dL_dY = nn_arch_backprop(&D->base,
			                           0, bs, DL_dL_dY);
			if(D_dL_dY == NULL)
			{
				goto fail_train;
			}

			// export images
			char     fname[256];
			uint32_t export_interval = 100;
			if((step%export_interval) == (export_interval - 1))
			{
				if(nn_tensor_copy(DX, DXio, 0, 0, bs) == 0)
				{
					goto fail_train;
				}

				snprintf(fname, 256, "data/D-DX0-%u-%u.png",
				         epoch, step);
				nn_tensor_ioExportPng(DXio, fname,
				                      0, 0, 1,
				                      0.0f, 1.0f);

				snprintf(fname, 256, "data/D-DX1-%u-%u.png",
				         epoch, step);
				nn_tensor_ioExportPng(DXio, fname,
				                      bs2, 0, 1,
				                      0.0f, 1.0f);

				if(nn_tensor_copy(DY, DYio, 0, 0, bs) == 0)
				{
					goto fail_train;
				}

				snprintf(fname, 256, "data/D-DY0-%u-%u.png",
				         epoch, step);
				nn_tensor_ioExportPng(DYio, fname,
				                      0, 0, 1,
				                      -2.0f, 2.0f);

				snprintf(fname, 256, "data/D-DY1-%u-%u.png",
				         epoch, step);
				nn_tensor_ioExportPng(DYio, fname,
				                      bs2, 0, 1,
				                      -2.0f, 2.0f);

				if(nn_tensor_copy(D_dL_dY, DXio, 0, 0, bs) == 0)
				{
					goto fail_train;
				}

				snprintf(fname, 256, "data/D-dL_dY0-%u-%u.png",
				         epoch, step);
				nn_tensor_ioExportPng(DXio, fname,
				                      0, 0, 1,
				                      -2.0f, 2.0f);

				snprintf(fname, 256, "data/D-dL_dY1-%u-%u.png",
				         epoch, step);
				nn_tensor_ioExportPng(DXio, fname,
				                      bs2, 0, 1,
				                      -2.0f, 2.0f);
			}

			/*
			 * train G
			 */

			// optionally start training G after N epochs training D
			if(epoch >= 0)
			{
				// load GX
				if(mnist_gan_loadGX(&rng, GXio, GX) == 0)
				{
					goto fail_train;
				}

				// GX > G > GY
				GY = nn_arch_forwardPass(&G->base,
			                             0, bs, GX);
				if(GY == NULL)
				{
					goto fail_train;
				}

				// GY > D > DY
				DY = nn_arch_forwardPass(&D->base,
			                             0, bs, GY);
				if(DY == NULL)
				{
					goto fail_train;
				}

				// DY + DY11 > DL > DL_dL_dY
				DL_dL_dY = nn_loss_pass(DL, 0, bs, DY, DY11);
				if(DL_dL_dY == NULL)
				{
					goto fail_train;
				}
				G_loss = nn_loss_loss(DL);

				// DL_dL_dY > D > D_dL_dY
				LOGD("G: DL_dL_dY > D > D_dL_dY");
				D_dL_dY = nn_arch_backprop(&D->base,
				                           NN_ARCH_FLAG_BP_NOP,
				                           bs, DL_dL_dY);
				if(D_dL_dY == NULL)
				{
					goto fail_train;
				}

				// D_dL_dY > G > G_dL_dY
				LOGD("G: D_dL_dY > G > G_dL_dY");
				nn_tensor_t* G_dL_dY;
				G_dL_dY = nn_arch_backprop(&G->base,
				                           0, bs, D_dL_dY);
				if(G_dL_dY == NULL)
				{
					goto fail_train;
				}

				if((step%export_interval) == (export_interval - 1))
				{
					if(nn_tensor_copy(GY, GYio, 0, 0, bs) == 0)
					{
						goto fail_train;
					}

					snprintf(fname, 256, "data/G-GY-%u-%u.png",
					         epoch, step);
					nn_tensor_ioExportPng(GYio, fname,
					                      0, 0, 1,
					                      0.0f, 1.0f);

					if(nn_tensor_copy(DY, DYio, 0, 0, bs) == 0)
					{
						goto fail_train;
					}

					snprintf(fname, 256, "data/G-DY-%u-%u.png",
					         epoch, step);
					nn_tensor_ioExportPng(DYio, fname,
					                      0, 0, 1,
					                      -2.0f, 2.0f);

					if(nn_tensor_copy(D_dL_dY, GYio, 0, 0, bs) == 0)
					{
						goto fail_train;
					}

					snprintf(fname, 256, "data/G-D_dL_dY-%u-%u.png",
					         epoch, step);
					nn_tensor_ioExportPng(GYio, fname,
					                      0, 0, 1,
					                      -2.0f, 2.0f);
				}
			}

			// update loss
			D_sum_loss += D_loss;
			G_sum_loss += G_loss;
			if(D_loss < D_min_loss)
			{
				D_min_loss = D_loss;
			}
			if(D_loss > D_max_loss)
			{
				D_max_loss = D_loss;
			}
			if(G_loss < G_min_loss)
			{
				G_min_loss = G_loss;
			}
			if(G_loss > G_max_loss)
			{
				G_max_loss = G_loss;
			}

			// plot loss
			uint32_t plot_interval = 10;
			if((step%plot_interval) == (plot_interval - 1))
			{
				float D_avg_loss = D_sum_loss/((float) plot_interval);
				float G_avg_loss = G_sum_loss/((float) plot_interval);
				fprintf(fplot, "%u %u %f %f %f %f %f %f\n",
				        epoch, step,
				        D_avg_loss, D_min_loss, D_max_loss,
				        G_avg_loss, G_min_loss, G_max_loss);
				fflush(fplot);

				// reset loss
				D_sum_loss = 0.0f;
				G_sum_loss = 0.0f;
				D_min_loss = FLT_MAX;
				G_min_loss = FLT_MAX;
				D_max_loss = 0.0f;
				G_max_loss = 0.0f;
			}

			LOGI("epoch=%u, step=%u, elapsed=%lf, D_loss=%f, G_loss=%f",
			     epoch, step, cc_timestamp() - t0, D_loss, G_loss);
			++step;
		}

		++epoch;
	}

	// cleanup
	fclose(fplot);
	nn_loss_delete(&DL);
	mnist_ganDisc_delete(&D);
	mnist_ganGen_delete(&G);
	nn_tensor_delete(&DY11);
	nn_tensor_delete(&DY01);
	nn_tensor_delete(&DYio);
	nn_tensor_delete(&DX);
	nn_tensor_delete(&DXio);
	nn_tensor_delete(&GYio);
	nn_tensor_delete(&GX);
	nn_tensor_delete(&GXio);
	nn_tensor_delete(&Xt);
	nn_engine_delete(&engine);

	// success
	return EXIT_SUCCESS;

	// failure
	fail_train:
		fclose(fplot);
	fail_fplot:
		nn_loss_delete(&DL);
	fail_DL:
		mnist_ganDisc_delete(&D);
	fail_D:
		mnist_ganGen_delete(&G);
	fail_G:
	fail_initDY11:
		nn_tensor_delete(&DY11);
	fail_DY11:
	fail_initDY01:
		nn_tensor_delete(&DY01);
	fail_DY01:
		nn_tensor_delete(&DYio);
	fail_DYio:
		nn_tensor_delete(&DX);
	fail_DX:
		nn_tensor_delete(&DXio);
	fail_DXio:
		nn_tensor_delete(&GYio);
	fail_GYio:
		nn_tensor_delete(&GX);
	fail_GX:
		nn_tensor_delete(&GXio);
	fail_GXio:
	fail_dim:
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
