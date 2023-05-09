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
#include "libnn/nn_dim.h"
#include "libnn/nn_factLayer.h"
#include "libnn/nn_mseLoss.h"
#include "libnn/nn_tensor.h"
#include "libnn/nn_weightLayer.h"

/***********************************************************
* public                                                   *
***********************************************************/

int main(int argc, char** argv)
{
	uint32_t max_batch_size = 64;

	nn_archInfo_t arch_info =
	{
		.learning_rate  = 0.01f,
		.momentum_decay = 0.0f,
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
		.count  = max_batch_size,
		.width  = 1,
		.height = 1,
		.depth  = 1,
	};

	nn_tensor_t* X = nn_tensor_new(&dimX);
	if(X == NULL)
	{
		goto fail_X;
	}

	nn_dim_t* dim = nn_tensor_dim(X);

	nn_dim_t dimW1 =
	{
		.count  = 4,
		.width  = 1,
		.height = 1,
		.depth  = dim->depth,
	};

	nn_weightLayer_t* l1;
	l1 = nn_weightLayer_new(arch, dim, &dimW1,
	                        NN_WEIGHT_LAYER_FLAG_XAVIER);
	if(l1 == NULL)
	{
		goto fail_l1;
	}
	dim = nn_layer_dimY(&l1->base);

	nn_factLayer_t* l2;
	l2 = nn_factLayer_new(arch, dim,
	                      nn_factLayer_tanh,
	                      nn_factLayer_dtanh);
	if(l2 == NULL)
	{
		goto fail_l2;
	}
	dim = nn_layer_dimY(&l2->base);

	nn_dim_t dimW3 =
	{
		.count  = 1,
		.width  = 1,
		.height = 1,
		.depth  = dim->depth,
	};

	nn_weightLayer_t* l3;
	l3 = nn_weightLayer_new(arch, dim,
	                        &dimW3,
	                        NN_WEIGHT_LAYER_FLAG_XAVIER);
	if(l3 == NULL)
	{
		goto fail_l3;
	}
	dim = nn_layer_dimY(&l3->base);

	nn_tensor_t* Y = nn_tensor_new(dim);
	if(Y == NULL)
	{
		goto fail_Y;
	}

	nn_mseLoss_t* mse_loss;
	mse_loss = nn_mseLoss_new(arch, dim);
	if(mse_loss == NULL)
	{
		goto fail_mse_loss;
	}

	if((nn_arch_attachLayer(arch, (nn_layer_t*) l1) == 0) ||
	   (nn_arch_attachLayer(arch, (nn_layer_t*) l2) == 0) ||
	   (nn_arch_attachLayer(arch, (nn_layer_t*) l3) == 0) ||
	   (nn_arch_attachLoss(arch, (nn_loss_t*) mse_loss) == 0))
	{
		goto fail_attach;
	}

	// training
	float    x;
	float    y;
	float    yt;
	uint32_t i;
	uint32_t m;
	uint32_t epoch;
	uint32_t bs    = 1;
	uint32_t count = 100000;
	for(epoch = 0; epoch < 10; ++epoch)
	{
		bs *= 2;
		if(bs > max_batch_size)
		{
			bs = max_batch_size;
		}

		for(i = 0; i < count; ++i)
		{
			if(i%1000 == 0)
			{
				LOGI("train %i:%i", epoch, i);
			}

			for(m = 0; m < bs; ++m)
			{
				x  = 1.0f*((float) (rand()%(count + 1)))/
				     ((float) count);
				yt = 2.0f*x*x + 1.0f;

				nn_tensor_set(X, m, 0, 0, 0, x);
				nn_tensor_set(Y, m, 0, 0, 0, yt);
			}

			nn_arch_train(arch, bs, X, Y);
		}

		char fname[256];
		snprintf(fname, 256, "output-%u.dat", epoch);

		FILE* fdat;
		fdat = fopen(fname, "w");
		if(fdat)
		{
			// prediction
			uint32_t predictions = 20;
			for(i = 0; i < predictions; ++i)
			{
				LOGI("predict %i", i);

				x  = 1.0f*((float) i)/((float) predictions);
				yt = 2.0f*x*x + 1.0f;

				nn_tensor_set(X, 0, 0, 0, 0, x);
				if(nn_arch_predict(arch, X, Y))
				{
					y = nn_tensor_get(Y, 0, 0, 0, 0);

					fprintf(fdat, "%f %f %f\n", x, yt, y);
				}
			}
			fclose(fdat);
		}
	}

	nn_mseLoss_delete(&mse_loss);
	nn_tensor_delete(&Y);
	nn_weightLayer_delete(&l3);
	nn_factLayer_delete(&l2);
	nn_weightLayer_delete(&l1);
	nn_tensor_delete(&X);
	nn_arch_delete(&arch);

	// success
	return EXIT_SUCCESS;

	// failure
	fail_attach:
		nn_mseLoss_delete(&mse_loss);
	fail_mse_loss:
		nn_tensor_delete(&Y);
	fail_Y:
		nn_weightLayer_delete(&l3);
	fail_l3:
		nn_factLayer_delete(&l2);
	fail_l2:
		nn_weightLayer_delete(&l1);
	fail_l1:
		nn_tensor_delete(&X);
	fail_X:
		nn_arch_delete(&arch);
	return EXIT_FAILURE;
}
