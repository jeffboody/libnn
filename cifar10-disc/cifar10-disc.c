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
#include "libcc/math/cc_float.h"
#include "libcc/cc_log.h"
#include "libcc/cc_memory.h"
#include "libcc/cc_timestamp.h"
#include "libnn/cifar10/nn_cifar10.h"
#include "libnn/nn_arch.h"
#include "libnn/nn_batchNormLayer.h"
#include "libnn/nn_coderLayer.h"
#include "libnn/nn_convLayer.h"
#include "libnn/nn_factLayer.h"
#include "libnn/nn_engine.h"
#include "libnn/nn_loss.h"
#include "libnn/nn_skipLayer.h"
#include "libnn/nn_tensor.h"
#include "libvkk/vkk_platform.h"
#include "texgz/texgz_png.h"
#include "cifar10_disc.h"

/***********************************************************
* callbacks                                                *
***********************************************************/

static int
cifar10_disc_onMain(vkk_engine_t* ve, int argc,
                    char** argv)
{
	ASSERT(ve);

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
	dn = cifar10_denoise_import(engine, dimXt->height,
	                            dimXt->width, dimXt->depth,
	                            "data/dn.json");
	if(dn == NULL)
	{
		goto fail_dn;
	}

	cifar10_disc_t* disc;
	disc = cifar10_disc_new(engine, bs, 32, dimXt->height,
	                        dimXt->width, dimXt->depth);
	if(disc == NULL)
	{
		goto fail_disc;
	}

	if(cifar10_disc_bs(disc) != cifar10_denoise_bs(dn))
	{
		LOGE("invalid bs=%u:%u",
		     cifar10_disc_bs(disc), cifar10_denoise_bs(dn));
		goto fail_bs;
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
	float    loss;
	float    sum_loss = 0.0f;
	float    min_loss = FLT_MAX;
	float    max_loss = 0.0f;
	double   t0       = cc_timestamp();
	while(epoch < 20)
	{
		steps = (epoch + 1)*dimXt->count/bs;
		while(step < steps)
		{
			cifar10_disc_sampleXt(disc, dn, cifar10->images);
			if(cifar10_disc_train(disc, &loss) == 0)
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

			// export images
			uint32_t export_interval = 100;
			if((step%export_interval) == (export_interval - 1))
			{
				snprintf(fname, 256, "data/Ytr-%u-%u-%u.png",
				         epoch, step, 0);
				cifar10_disc_exportXd0(disc, fname, 0);
				snprintf(fname, 256, "data/Cr-%u-%u-%u.png",
				         epoch, step, 0);
				cifar10_disc_exportXd1(disc, fname, 0);

				snprintf(fname, 256, "data/Yg-%u-%u-%u.png",
				         epoch, step, bs2);
				cifar10_disc_exportXd0(disc, fname, bs2);
				snprintf(fname, 256, "data/Cg-%u-%u-%u.png",
				         epoch, step, bs2);
				cifar10_disc_exportXd1(disc, fname, bs2);

				snprintf(fname, 256, "data/Y-%u-%u-%u.png",
				         epoch, step, 0);
				cifar10_disc_exportY(disc, fname, 0);
				snprintf(fname, 256, "data/Y-%u-%u-%u.png",
				         epoch, step, bs2);
				cifar10_disc_exportY(disc, fname, bs2);
			}

			// plot loss
			uint32_t plot_interval = 100;
			if((step%plot_interval) == (plot_interval - 1))
			{
				float avg_loss = sum_loss/((float) plot_interval);
				fprintf(fplot, "%u %u %f %f %f\n",
				        epoch, step, avg_loss, min_loss, max_loss);
				fflush(fplot);

				// reset loss
				sum_loss = 0.0f;
				min_loss = FLT_MAX;
				max_loss = 0.0f;
			}

			// export arch
			uint32_t arch_interval = 1000;
			if((step%arch_interval) == (arch_interval - 1))
			{
				snprintf(fname, 256, "data/arch-%i-%i.json",
				         epoch, step);
				cifar10_disc_export(disc, fname);
			}

			LOGI("epoch=%u, step=%u, elapsed=%lf, loss=%f",
			     epoch, step, cc_timestamp() - t0, loss);
			++step;
		}

		++epoch;
	}

	// cleanup
	fclose(fplot);
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
	fail_bs:
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
	.app_name    = "cifar10-disc",
	.app_version =
	{
		.major = 1,
		.minor = 0,
		.patch = 0,
	},
	.app_dir = "cifar10-disc",
	.onMain  = cifar10_disc_onMain,
};
