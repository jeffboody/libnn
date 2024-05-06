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
#include "libcc/rng/cc_rngUniform.h"
#include "libcc/cc_log.h"
#include "libcc/cc_memory.h"
#include "libcc/cc_timestamp.h"
#include "libnn/cifar10/nn_cifar10.h"
#include "libnn/nn_engine.h"
#include "libnn/nn_tensor.h"
#include "libvkk/vkk_platform.h"
#include "cifar10_upsample.h"

/***********************************************************
* callbacks                                                *
***********************************************************/

static int
cifar10_upsample_onMain(vkk_engine_t* ve, int argc,
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

	nn_cifar10_t* cifar10[5] = { 0 };

	int i;
	for(i = 0; i < 5; ++i)
	{
		cifar10[i] = nn_cifar10_load(engine,
		                             NN_CIFAR10_MODE_COLOR,
		                             i + 1);
		if(cifar10[i] == NULL)
		{
			goto fail_cifar10;
		}
	}

	nn_dim_t* dimXt = nn_tensor_dim(cifar10[0]->images);

	cifar10_upsample_t* self;
	self = cifar10_upsample_new(engine, 32, 32,
	                            dimXt->height, dimXt->width,
	                            dimXt->depth);
	if(self == NULL)
	{
		goto fail_dn;
	}

	FILE* fplot = fopen("data/plot.dat", "w");
	if(fplot == NULL)
	{
		goto fail_fplot;
	}

	// training
	uint32_t bs    = cifar10_upsample_bs(self);
	uint32_t epoch = 0;
	uint32_t step  = 0;
	uint32_t steps;
	uint32_t idx;
	char     fname[256];
	float    loss;
	float    sum_loss = 0.0f;
	float    min_loss = FLT_MAX;
	float    max_loss = 0.0f;
	double   t0       = cc_timestamp();
	while(epoch < 20)
	{
		// reduce learning rate for final epochs
		if(epoch == 15)
		{
			nn_archState_t* state = &self->base.state;
			state->adam_alpha = 0.0001f;
		}

		steps = (epoch + 1)*(5*dimXt->count)/bs;
		while(step < steps)
		{
			idx = cc_rngUniform_rand2U(&rng, 0, 4);
			cifar10_upsample_sampleXt(self, cifar10[idx]->images);
			if(cifar10_upsample_train(self, &loss) == 0)
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
				snprintf(fname, 256, "data/X-%u-%u.png",
				         epoch, step);
				cifar10_upsample_exportX(self, fname, 0);
				snprintf(fname, 256, "data/Y-%u-%u.png",
				         epoch, step);
				cifar10_upsample_exportY(self, fname, 0);
				snprintf(fname, 256, "data/U-%u-%u.png",
				         epoch, step);
				cifar10_upsample_exportU(self, fname, 0);
				snprintf(fname, 256, "data/L-%u-%u.png",
				         epoch, step);
				cifar10_upsample_exportL(self, fname, 0);
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
				cifar10_upsample_export(self, fname);
			}

			LOGI("epoch=%u, step=%u, elapsed=%lf, loss=%f",
			     epoch, step, cc_timestamp() - t0, loss);
			++step;
		}

		++epoch;
	}

	// cleanup
	fclose(fplot);
	cifar10_upsample_delete(&self);
	for(i = 0; i < 5; ++i)
	{
		nn_cifar10_delete(&cifar10[i]);
	}
	nn_engine_delete(&engine);

	// success
	return EXIT_SUCCESS;

	// failure
	fail_train:
		fclose(fplot);
	fail_fplot:
		cifar10_upsample_delete(&self);
	fail_dn:
	{
		int j;
		for(j = 0; j < i; ++j)
		{
			nn_cifar10_delete(&cifar10[j]);
		}
	}
	fail_cifar10:
		nn_engine_delete(&engine);
	return EXIT_FAILURE;
}

vkk_platformInfo_t VKK_PLATFORM_INFO =
{
	.app_name    = "cifar10-upsample",
	.app_version =
	{
		.major = 1,
		.minor = 0,
		.patch = 0,
	},
	.app_dir = "cifar10-upsample",
	.onMain  = cifar10_upsample_onMain,
};
