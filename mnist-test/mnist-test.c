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

#define LOG_TAG "nn"
#include "libcc/math/cc_float.h"
#include "libcc/cc_log.h"
#include "libcc/cc_memory.h"
#include "libnn/nn_arch.h"
#include "libnn/nn_batchNormLayer.h"
#include "libnn/nn_coderLayer.h"
#include "libnn/nn_convLayer.h"
#include "libnn/nn_factLayer.h"
#include "libnn/nn_loss.h"
#include "libnn/nn_poolingLayer.h"
#include "libnn/nn_skipLayer.h"
#include "libnn/nn_tensor.h"
#include "libvkk/vkk_platform.h"
#include "texgz/texgz_png.h"

/***********************************************************
* private                                                  *
***********************************************************/

static int mnist_readU32(FILE* f, uint32_t* _data)
{
	ASSERT(f);
	ASSERT(_data);

	uint32_t data;
	if(fread((void*) &data, sizeof(uint32_t), 1, f) != 1)
	{
		LOGE("fread failed");
		return 0;
	}

	// swap endian
	*_data = ((data << 24) & 0xFF000000) |
	         ((data << 8)  & 0x00FF0000) |
	         ((data >> 8)  & 0x0000FF00) |
	         ((data >> 24) & 0x000000FF);

	return 1;
}

static nn_tensor_t* mnist_load(nn_arch_t* arch)
{
	ASSERT(arch);

	FILE* f = fopen("data/train-images-idx3-ubyte", "r");
	if(f == NULL)
	{
		LOGE("invalid");
		return NULL;
	}

	// read header
	uint32_t magic = 0;
	nn_dim_t dim =
	{
		.depth = 1,
	};
	if((mnist_readU32(f, &magic)      == 0) ||
	   (mnist_readU32(f, &dim.count)  == 0) ||
	   (mnist_readU32(f, &dim.width)  == 0) ||
	   (mnist_readU32(f, &dim.height) == 0))
	{
		goto fail_header;
	}

	// check header
	size_t size = dim.count*dim.height*dim.width;
	if((magic != 0x00000803) || (size == 0))
	{
		LOGE("invalid magic=0x%X, size=%u",
		     magic, (uint32_t) size);
		goto fail_check;
	}

	// allocate ubyte data
	uint8_t* data = (uint8_t*) CALLOC(1, size);
	if(data == NULL)
	{
		LOGE("CALLOC failed");
		goto fail_allocate;
	}

	// read ubyte data
	if(fread((void*) data, size, 1, f) != 1)
	{
		LOGE("fread failed");
		goto fail_read;
	}

	nn_tensor_t* T;
	T = nn_tensor_new(arch, &dim,
	                  NN_TENSOR_INIT_ZERO,
	                  NN_TENSOR_MODE_IO);
	if(T == NULL)
	{
		goto fail_T;
	}

	// convert data
	float    t;
	uint32_t m;
	uint32_t i;
	uint32_t j;
	uint32_t idx = 0;
	for(m = 0; m < dim.count; ++m)
	{
		for(i = 0; i < dim.height; ++i)
		{
			for(j = 0; j < dim.width; ++j)
			{
				t = ((float) data[idx++])/255.0f;
				nn_tensor_set(T, m, i, j, 1, t);
			}
		}
	}

	FREE(data);
	fclose(f);

	// success
	return T;

	// failure
	fail_T:
	fail_read:
		FREE(data);
	fail_allocate:
	fail_check:
	fail_header:
		fclose(f);
	return NULL;
}

static void
mnist_noise(cc_rngNormal_t* rng, uint32_t bs,
            nn_tensor_t* X, nn_tensor_t* Y)
{
	ASSERT(rng);
	ASSERT(X);
	ASSERT(Y);

	nn_dim_t* dimX = nn_tensor_dim(X);
	uint32_t  xh   = dimX->height;
	uint32_t  xw   = dimX->width;

	float    x;
	float    y;
	float    n;
	uint32_t m;
	uint32_t i;
	uint32_t j;
	for(m = 0; m < bs; ++m)
	{
		for(i = 0; i < xh; ++i)
		{
			for(j = 0; j < xw; ++j)
			{
				y = nn_tensor_get(Y, m, i, j, 0);
				n = cc_rngNormal_rand1F(rng);
				x = cc_clamp(y + n, 0.0f, 1.0f);
				nn_tensor_set(X, m, i, j, 0, x);
			}
		}
	}
}

static void
mnist_savepng(const char* fname, texgz_tex_t* tex,
              nn_tensor_t* X, uint32_t m)
{
	ASSERT(fname);
	ASSERT(tex);
	ASSERT(X);

	nn_dim_t* dimX = nn_tensor_dim(X);
	uint32_t  xh   = dimX->height;
	uint32_t  xw   = dimX->width;

	float    x;
	uint32_t i;
	uint32_t j;
	unsigned char pixel[4] =
	{
		0x00, 0x00, 0x00, 0xFF,
	};
	for(i = 0; i < xh; ++i)
	{
		for(j = 0; j < xw; ++j)
		{
			x        = nn_tensor_get(X, m, i, j, 0);
			pixel[0] = (unsigned char)
			           cc_clamp(255.0f*x, 0.0f, 255.0f);
			pixel[1] = pixel[0];
			pixel[2] = pixel[0];
			texgz_tex_setPixel(tex, j, i, pixel);
		}
	}

	texgz_png_export(tex, fname);
}

/***********************************************************
* callbacks                                                *
***********************************************************/

static int
mnist_test_onMain(vkk_engine_t* engine, int argc, char** argv)
{
	ASSERT(engine);

	nn_archState_t arch_state =
	{
		.learning_rate   = 0.01f,
		.momentum_decay  = 0.5f,
		.batch_momentum  = 0.99f,
		.l2_lambda       = 0.01f,
		.clip_max_weight = 10.0f,
		.clip_max_bias   = 10.0f,
		.clip_mu_inc     = 0.99f,
		.clip_mu_dec     = 0.90f,
		.clip_scale      = 0.1f,
	};

	nn_arch_t* arch = nn_arch_new(engine, 0, &arch_state);
	if(arch == NULL)
	{
		return EXIT_FAILURE;
	}

	nn_tensor_t* Xt = mnist_load(arch);
	if(Xt == NULL)
	{
		goto fail_Xt;
	}

	uint32_t  fc     = 32;
	uint32_t  max_bs = 32;
	nn_dim_t* dimXt  = nn_tensor_dim(Xt);
	nn_dim_t  dimX   =
	{
		.count  = max_bs,
		.height = dimXt->height,
		.width  = dimXt->width,
		.depth  = 1,
	};

	nn_tensor_t* X;
	X = nn_tensor_new(arch, &dimX,
	                  NN_TENSOR_INIT_ZERO,
	                  NN_TENSOR_MODE_IO);
	if(X == NULL)
	{
		goto fail_X;
	}

	nn_dim_t* dim = nn_tensor_dim(X);

	nn_batchNormLayer_t* bn0;
	bn0 = nn_batchNormLayer_new(arch, dim);
	if(bn0 == NULL)
	{
		goto fail_bn0;
	}

	nn_coderLayerInfo_t info_enc1 =
	{
		.arch        = arch,
		.dimX        = dim,
		.fc          = fc,
		.skip_enable = 0,
		.skip_mode   = NN_SKIP_LAYER_MODE_FORK,
		.skip_coder  = NULL,
		.repeat      = 0,
		.op_mode     = NN_CODER_OP_MODE_POOLMAX,
	};

	nn_coderLayer_t* enc1;
	enc1 = nn_coderLayer_new(&info_enc1);
	if(enc1 == NULL)
	{
		goto fail_enc1;
	}
	dim = nn_layer_dimY(&enc1->base);

	nn_coderLayerInfo_t info_enc2 =
	{
		.arch        = arch,
		.dimX        = dim,
		.fc          = fc,
		.skip_enable = 0,
		.skip_mode   = NN_SKIP_LAYER_MODE_FORK,
		.skip_coder  = NULL,
		.repeat      = 0,
		.op_mode     = NN_CODER_OP_MODE_POOLMAX,
	};

	nn_coderLayer_t* enc2;
	enc2 = nn_coderLayer_new(&info_enc2);
	if(enc2 == NULL)
	{
		goto fail_enc2;
	}
	dim = nn_layer_dimY(&enc2->base);

	nn_coderLayerInfo_t info_dec3 =
	{
		.arch        = arch,
		.dimX        = dim,
		.fc          = fc,
		.skip_enable = 0,
		.skip_mode   = NN_SKIP_LAYER_MODE_ADD,
		.skip_coder  = NULL,
		.repeat      = 0,
		.op_mode     = NN_CODER_OP_MODE_UPSCALE,
	};

	nn_coderLayer_t* dec3;
	dec3 = nn_coderLayer_new(&info_dec3);
	if(dec3 == NULL)
	{
		goto fail_dec3;
	}
	dim = nn_layer_dimY(&dec3->base);

	nn_coderLayerInfo_t info_dec4 =
	{
		.arch        = arch,
		.dimX        = dim,
		.fc          = fc,
		.skip_enable = 0,
		.skip_mode   = NN_SKIP_LAYER_MODE_ADD,
		.skip_coder  = NULL,
		.repeat      = 0,
		.op_mode     = NN_CODER_OP_MODE_UPSCALE,
	};

	nn_coderLayer_t* dec4;
	dec4 = nn_coderLayer_new(&info_dec4);
	if(dec4 == NULL)
	{
		goto fail_dec4;
	}
	dim = nn_layer_dimY(&dec4->base);

	nn_dim_t dimWO =
	{
		.count  = 1,
		.width  = 3,
		.height = 3,
		.depth  = dim->depth,
	};

	nn_convLayer_t* convO;
	convO = nn_convLayer_new(arch, dim, &dimWO, 1,
	                         NN_CONV_LAYER_FLAG_XAVIER);
	if(convO == NULL)
	{
		goto fail_convO;
	}
	dim = nn_layer_dimY(&convO->base);

	nn_factLayer_t* factO;
	factO = nn_factLayer_new(arch, dim,
	                         NN_FACT_LAYER_FN_LOGISTIC);
	if(factO == NULL)
	{
		goto fail_factO;
	}

	nn_loss_t* loss;
	loss = nn_loss_new(arch, dim, NN_LOSS_FN_MSE);
	if(loss == NULL)
	{
		goto fail_loss;
	}

	nn_tensor_t* Y;
	Y = nn_tensor_new(arch, dim,
	                  NN_TENSOR_INIT_ZERO,
	                  NN_TENSOR_MODE_IO);
	if(Y == NULL)
	{
		goto fail_Y;
	}

	if((nn_arch_attachLayer(arch, (nn_layer_t*) bn0)   == 0) ||
	   (nn_arch_attachLayer(arch, (nn_layer_t*) enc1)  == 0) ||
	   (nn_arch_attachLayer(arch, (nn_layer_t*) enc2)  == 0) ||
	   (nn_arch_attachLayer(arch, (nn_layer_t*) dec3)  == 0) ||
	   (nn_arch_attachLayer(arch, (nn_layer_t*) dec4)  == 0) ||
	   (nn_arch_attachLayer(arch, (nn_layer_t*) convO) == 0) ||
	   (nn_arch_attachLayer(arch, (nn_layer_t*) factO) == 0) ||
	   (nn_arch_attachLoss(arch,  (nn_loss_t*)  loss)  == 0))
	{
		goto fail_attach;
	}

	texgz_tex_t* tex;
	tex = texgz_tex_new(dimX.width, dimX.height,
	                    dimX.width, dimX.height,
	                    TEXGZ_UNSIGNED_BYTE, TEXGZ_RGBA,
	                    NULL);
	if(tex == NULL)
	{
		goto fail_tex;
	}

	FILE* fplot = fopen("data/plot.dat", "w");
	if(fplot == NULL)
	{
		goto fail_fplot;
	}

	// training
	uint32_t epoch;
	uint32_t step = 0;
	uint32_t m;
	uint32_t n;
	uint32_t bs;
	char     fname[256];
	float    sum_loss = 0.0f;
	float    min_loss = FLT_MAX;
	float    max_loss = 0.0f;
	cc_rngNormal_t rng;
	cc_rngNormal_init(&rng, 0.5f, 0.5f);
	for(epoch = 0; epoch < 20; ++epoch)
	{
		for(n = 0; n < dimXt->count; n += max_bs)
		{
			// initialize Y
			bs = 0;
			for(m = 0; m < max_bs; ++m)
			{
				if(m + n >= dimXt->count)
				{
					break;
				}

				nn_tensor_blit(Xt, Y, 1, n + m, m);
				++bs;
			}

			// add noise to X
			// mnist_noise causes skip layers to perform poorly
			mnist_noise(&rng, bs, X, Y);

			// export training images
			if((n%1024 == 0) && (epoch == 0))
			{
				snprintf(fname, 256, "data/x%u.png", n);
				mnist_savepng(fname, tex, X, 0);
				snprintf(fname, 256, "data/yt%u.png", n);
				mnist_savepng(fname, tex, Y, 0);
			}

			nn_arch_train(arch, bs, X, Y);

			// export prediction images
			if((n%1024 == 0) && nn_arch_predict(arch, X, Y))
			{
				snprintf(fname, 256, "data/y%u-%u-%u.png", n, epoch, step);
				mnist_savepng(fname, tex, Y, 0);
			}

			// update loss
			float l = nn_arch_loss(arch);
			sum_loss += l;
			if(l < min_loss)
			{
				min_loss = l;
			}
			if(l > max_loss)
			{
				max_loss = l;
			}

			// plot loss
			uint32_t plot_interval = 10;
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

			LOGI("epoch=%u, step=%u, n=%u, loss=%f",
			     epoch, step, n, l);
			#if 0
			// if too many results are output then nn_tensor_store
			// may be modified to output the first instance
			nn_tensor_print(bn0->Y, "bn0->Y");
			nn_tensor_print(enc1->conv->Y, "enc1->conv->Y");
			nn_tensor_print(enc1->op->pool->Y, "enc1->op->pool->Y");
			nn_tensor_print(enc2->conv->Y, "enc2->conv->Y");
			nn_tensor_print(enc2->op->pool->Y, "enc2->op->pool->Y");
			nn_tensor_print(dec3->conv->Y, "dec3->conv->Y");
			nn_tensor_print(dec3->op->conv->Y, "dec3->op->conv->Y");
			nn_tensor_print(dec4->conv->Y, "dec4->conv->Y");
			nn_tensor_print(dec4->op->conv->Y, "dec4->op->conv->Y");
			nn_tensor_print(convO->Y, "convO->Y");
			nn_tensor_print(factO->Y, "factO->Y");
			nn_tensor_print(loss->dL_dY, "loss->dL_dY");
			nn_tensor_print(convO->dL_dX, "convO->dL_dX");
			nn_tensor_print(dec4->op->conv->dL_dX, "dec4->op->conv->dL_dX");
			nn_tensor_print(dec4->bn->dL_dXhat, "dec4->bn->dL_dXhat");
			nn_tensor_print(dec4->conv->dL_dX, "dec4->conv->dL_dX");
			nn_tensor_print(dec3->op->conv->dL_dX, "dec3->op->conv->dL_dX");
			nn_tensor_print(dec3->bn->dL_dXhat, "dec3->bn->dL_dXhat");
			nn_tensor_print(dec3->conv->dL_dX, "dec3->conv->dL_dX");
			nn_tensor_print(enc2->op->pool->dL_dX, "enc2->op->pool->dL_dX");
			nn_tensor_print(enc2->bn->dL_dXhat, "enc2->bn->dL_dXhat");
			nn_tensor_print(enc2->conv->dL_dX, "enc2->conv->dL_dX");
			nn_tensor_print(enc1->op->pool->dL_dX, "enc1->op->pool->dL_dX");
			nn_tensor_print(enc1->bn->dL_dXhat, "enc1->bn->dL_dXhat");
			nn_tensor_print(enc1->conv->dL_dX, "enc1->conv->dL_dX");
			nn_tensor_print(bn0->dL_dXhat, "bn0->dL_dXhat");
			#endif
			++step;
		}

		// save arch
		jsmn_stream_t* stream = jsmn_stream_new();
		if(stream)
		{
			snprintf(fname, 256, "data/arch-%i-%i.json",
			         epoch, step - 1);

			FILE* farch = fopen(fname, "w");
			if(farch)
			{
				int ret = 1;
				ret &= jsmn_stream_beginObject(stream);
				ret &= jsmn_stream_key(stream, "%s", "arch");
				ret &= nn_arch_export(arch, stream);
				ret &= jsmn_stream_key(stream, "%s", "bn0");
				ret &= nn_batchNormLayer_export(bn0, stream);
				ret &= jsmn_stream_key(stream, "%s", "enc1");
				ret &= nn_coderLayer_export(enc1, stream);
				ret &= jsmn_stream_key(stream, "%s", "enc2");
				ret &= nn_coderLayer_export(enc2, stream);
				ret &= jsmn_stream_key(stream, "%s", "dec3");
				ret &= nn_coderLayer_export(dec3, stream);
				ret &= jsmn_stream_key(stream, "%s", "dec4");
				ret &= nn_coderLayer_export(dec4, stream);
				ret &= jsmn_stream_key(stream, "%s", "convO");
				ret &= nn_convLayer_export(convO, stream);
				ret &= jsmn_stream_key(stream, "%s", "factO");
				ret &= nn_factLayer_export(factO, stream);
				ret &= jsmn_stream_key(stream, "%s", "loss");
				ret &= nn_loss_export(loss, stream);
				ret &= jsmn_stream_end(stream);

				size_t size = 0;
				const char* buf = jsmn_stream_buffer(stream, &size);
				if(buf)
				{
					fprintf(farch, "%s", buf);
				}
				fclose(farch);
			}
			jsmn_stream_delete(&stream);
		}
	}

	// cleanup
	fclose(fplot);
	texgz_tex_delete(&tex);
	nn_loss_delete(&loss);
	nn_tensor_delete(&Y);
	nn_factLayer_delete(&factO);
	nn_convLayer_delete(&convO);
	nn_coderLayer_delete(&dec4);
	nn_coderLayer_delete(&dec3);
	nn_coderLayer_delete(&enc2);
	nn_coderLayer_delete(&enc1);
	nn_batchNormLayer_delete(&bn0);
	nn_tensor_delete(&X);
	nn_arch_delete(&arch);
	nn_tensor_delete(&Xt);

	// success
	return EXIT_SUCCESS;

	// failure
	fail_fplot:
		texgz_tex_delete(&tex);
	fail_tex:
	fail_attach:
		nn_tensor_delete(&Y);
	fail_Y:
		nn_loss_delete(&loss);
	fail_loss:
		nn_factLayer_delete(&factO);
	fail_factO:
		nn_convLayer_delete(&convO);
	fail_convO:
		nn_coderLayer_delete(&dec4);
	fail_dec4:
		nn_coderLayer_delete(&dec3);
	fail_dec3:
		nn_coderLayer_delete(&enc2);
	fail_enc2:
		nn_coderLayer_delete(&enc1);
	fail_enc1:
		nn_batchNormLayer_delete(&bn0);
	fail_bn0:
		nn_tensor_delete(&X);
	fail_X:
		nn_tensor_delete(&Xt);
	fail_Xt:
		nn_arch_delete(&arch);
	return EXIT_FAILURE;
}

vkk_platformInfo_t VKK_PLATFORM_INFO =
{
	.app_name    = "MNIST-Test",
	.app_version =
	{
		.major = 1,
		.minor = 0,
		.patch = 0,
	},
	.app_dir = "MNISTTest",
	.onMain  = mnist_test_onMain,
};
