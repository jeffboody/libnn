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
	uint32_t bs = 32;

	nn_archInfo_t arch_info =
	{
		.max_batch_size = bs,
		.learning_rate  = 0.01f,
		.momentum_decay = 0.0f,
		.batch_momentum = 0.99f,
		.l2_lambda      = 0.01f,
	};

	nn_arch_t* arch = nn_arch_new(0, &arch_info);
	if(arch == NULL)
	{
		return EXIT_FAILURE;
	}

	nn_dim_t dimX =
	{
		.n = bs,
		.w = 1,
		.h = 1,
		.d = 1,
	};

	nn_tensor_t* X = nn_tensor_new(&dimX);
	if(X == NULL)
	{
		goto fail_X;
	}

	nn_dim_t dimY1 =
	{
		.n = bs,
		.w = 1,
		.h = 1,
		.d = 4,
	};

	nn_weightLayer_t* l1;
	l1 = nn_weightLayer_new(arch, &dimX, &dimY1,
	                        NN_WEIGHT_LAYER_INITMODE_XAVIER);
	if(l1 == NULL)
	{
		goto fail_l1;
	}

	nn_factLayer_t* l2;
	l2 = nn_factLayer_new(arch, nn_layer_dim(&l1->base),
	                      nn_factLayer_tanh,
	                      nn_factLayer_dtanh);
	if(l2 == NULL)
	{
		goto fail_l2;
	}

	nn_dim_t dimY3 =
	{
		.n = bs,
		.w = 1,
		.h = 1,
		.d = 1,
	};

	nn_weightLayer_t* l3;
	l3 = nn_weightLayer_new(arch, nn_layer_dim(&l2->base),
	                        &dimY3,
	                        NN_WEIGHT_LAYER_INITMODE_XAVIER);
	if(l3 == NULL)
	{
		goto fail_l3;
	}

	nn_tensor_t* Y = nn_tensor_new(&dimY3);
	if(Y == NULL)
	{
		goto fail_Y;
	}

	nn_mseLoss_t* mse_loss;
	mse_loss = nn_mseLoss_new(arch, nn_layer_dim(&l3->base));
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

	FILE* fdat;
	fdat = fopen("output.dat", "w");
	if(fdat == NULL)
	{
		goto fail_fdat;
	}

	// training
	float    x;
	float    y;
	float    yt;
	uint32_t i;
	uint32_t b;
	uint32_t count = 1000000;
	for(i = 0; i < count; ++i)
	{
		if(i%1000 == 0)
		{
			LOGI("train %i", i);
		}

		for(b = 0; b < bs; ++b)
		{
			x  = 1.0f*((float) (rand()%(count + 1)))/
			     ((float) count);
			yt = 2.0f*x*x + 1.0f;

			nn_tensor_set(X, b, 0, 0, 0, x);
			nn_tensor_set(Y, b, 0, 0, 0, yt);
		}

		nn_arch_train(arch, bs, X, Y);
	}

	// prediction
	count = 20;
	for(i = 0; i < count; ++i)
	{
		LOGI("predict %i", i);

		x  = 1.0f*((float) i)/((float) count);
		yt = 2.0f*x*x + 1.0f;

		nn_tensor_set(X, 0, 0, 0, 0, x);
		if(nn_arch_predict(arch, X, Y))
		{
			y = nn_tensor_get(Y, 0, 0, 0, 0);

			fprintf(fdat, "%f %f %f\n", x, yt, y);
		}
	}

	fclose(fdat);
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
	fail_fdat:
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
