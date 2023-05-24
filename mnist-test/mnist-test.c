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
#include "libcc/math/cc_float.h"
#include "libcc/cc_log.h"
#include "libcc/cc_memory.h"
#include "libnn/nn_arch.h"
#include "libnn/nn_batchNormLayer.h"
#include "libnn/nn_convLayer.h"
#include "libnn/nn_factLayer.h"
#include "libnn/nn_loss.h"
#include "libnn/nn_poolingLayer.h"
#include "libnn/nn_tensor.h"
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

static nn_tensor_t* mnist_load(void)
{
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

	nn_tensor_t* T = nn_tensor_new(&dim);
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
* public                                                   *
***********************************************************/

int main(int argc, char** argv)
{
	nn_tensor_t* Xt = mnist_load();
	if(Xt == NULL)
	{
		return EXIT_FAILURE;
	}

	nn_archInfo_t arch_info =
	{
		.learning_rate  = 0.001f,
		.momentum_decay = 0.5f,
		.batch_momentum = 0.99f,
		.l2_lambda      = 0.01f,
	};

	nn_arch_t* arch = nn_arch_new(0, &arch_info);
	if(arch == NULL)
	{
		goto fail_arch;
	}

	uint32_t  max_bs = 32;
	nn_dim_t* dimXt  = nn_tensor_dim(Xt);
	nn_dim_t  dimX   =
	{
		.count  = max_bs,
		.height = dimXt->height,
		.width  = dimXt->width,
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
		.count  = 32,
		.width  = 3,
		.height = 3,
		.depth  = dim->depth,
	};

	nn_convLayer_t* conv1;
	conv1 = nn_convLayer_new(arch, dim, &dimW1, 1,
	                        NN_CONV_LAYER_FLAG_PAD_SAME |
	                        NN_CONV_LAYER_FLAG_HE);
	if(conv1 == NULL)
	{
		goto fail_conv1;
	}
	dim = nn_layer_dimY(&conv1->base);

	nn_factLayer_t* fact1;
	fact1 = nn_factLayer_new(arch, dim,
	                         nn_factLayer_ReLU,
	                         nn_factLayer_dReLU);
	if(fact1 == NULL)
	{
		goto fail_fact1;
	}

	nn_poolingLayer_t* pool1;
	pool1 = nn_poolingLayer_new(arch, dim, 2, 2,
	                            NN_POOLING_LAYER_MODE_MAX);
	if(pool1 == NULL)
	{
		goto fail_pool1;
	}
	dim = nn_layer_dimY(&pool1->base);

	nn_dim_t dimW2 =
	{
		.count  = 32,
		.width  = 3,
		.height = 3,
		.depth  = dim->depth,
	};

	nn_convLayer_t* conv2;
	conv2 = nn_convLayer_new(arch, dim, &dimW2, 1,
	                        NN_CONV_LAYER_FLAG_PAD_SAME |
	                        NN_CONV_LAYER_FLAG_HE);
	if(conv2 == NULL)
	{
		goto fail_conv2;
	}
	dim = nn_layer_dimY(&conv2->base);

	nn_factLayer_t* fact2;
	fact2 = nn_factLayer_new(arch, dim,
	                         nn_factLayer_ReLU,
	                         nn_factLayer_dReLU);
	if(fact2 == NULL)
	{
		goto fail_fact2;
	}

	nn_poolingLayer_t* pool2;
	pool2 = nn_poolingLayer_new(arch, dim, 2, 2,
	                            NN_POOLING_LAYER_MODE_MAX);
	if(pool2 == NULL)
	{
		goto fail_pool2;
	}
	dim = nn_layer_dimY(&pool2->base);

	nn_dim_t dimW3 =
	{
		.count  = 32,
		.width  = 3,
		.height = 3,
		.depth  = dim->depth,
	};

	nn_convLayer_t* conv3;
	conv3 = nn_convLayer_new(arch, dim, &dimW3, 1,
	                        NN_CONV_LAYER_FLAG_PAD_SAME |
	                        NN_CONV_LAYER_FLAG_HE);
	if(conv3 == NULL)
	{
		goto fail_conv3;
	}
	dim = nn_layer_dimY(&conv3->base);

	nn_factLayer_t* fact3;
	fact3 = nn_factLayer_new(arch, dim,
	                         nn_factLayer_ReLU,
	                         nn_factLayer_dReLU);
	if(fact3 == NULL)
	{
		goto fail_fact3;
	}

	nn_dim_t dimWT3 =
	{
		.count  = dim->depth,
		.width  = 2,
		.height = 2,
		.depth  = dim->depth,
	};

	nn_convLayer_t* convT3;
	convT3 = nn_convLayer_new(arch, dim, &dimWT3, 2,
	                          NN_CONV_LAYER_FLAG_TRANSPOSE |
	                          NN_CONV_LAYER_FLAG_PAD_SAME  |
	                          NN_CONV_LAYER_FLAG_XAVIER);
	if(convT3 == NULL)
	{
		goto fail_convT3;
	}
	dim = nn_layer_dimY(&convT3->base);

	nn_dim_t dimW4 =
	{
		.count  = 32,
		.width  = 3,
		.height = 3,
		.depth  = dim->depth,
	};

	nn_convLayer_t* conv4;
	conv4 = nn_convLayer_new(arch, dim, &dimW4, 1,
	                        NN_CONV_LAYER_FLAG_PAD_SAME |
	                        NN_CONV_LAYER_FLAG_HE);
	if(conv4 == NULL)
	{
		goto fail_conv4;
	}
	dim = nn_layer_dimY(&conv4->base);

	nn_factLayer_t* fact4;
	fact4 = nn_factLayer_new(arch, dim,
	                         nn_factLayer_ReLU,
	                         nn_factLayer_dReLU);
	if(fact4 == NULL)
	{
		goto fail_fact4;
	}

	nn_dim_t dimWT4 =
	{
		.count  = dim->depth,
		.width  = 2,
		.height = 2,
		.depth  = dim->depth,
	};

	nn_convLayer_t* convT4;
	convT4 = nn_convLayer_new(arch, dim, &dimWT4, 2,
	                          NN_CONV_LAYER_FLAG_TRANSPOSE |
	                          NN_CONV_LAYER_FLAG_PAD_SAME  |
	                          NN_CONV_LAYER_FLAG_XAVIER);
	if(convT4 == NULL)
	{
		goto fail_convT4;
	}
	dim = nn_layer_dimY(&convT4->base);

	nn_dim_t dimW5 =
	{
		.count  = 1,
		.width  = 3,
		.height = 3,
		.depth  = dim->depth,
	};

	nn_convLayer_t* conv5;
	conv5 = nn_convLayer_new(arch, dim, &dimW5, 1,
	                         NN_CONV_LAYER_FLAG_PAD_SAME |
	                         NN_CONV_LAYER_FLAG_XAVIER);
	if(conv5 == NULL)
	{
		goto fail_conv5;
	}
	dim = nn_layer_dimY(&conv5->base);

	nn_factLayer_t* fact5;
	fact5 = nn_factLayer_new(arch, dim,
	                         nn_factLayer_logistic,
	                         nn_factLayer_dlogistic);
	if(fact5 == NULL)
	{
		goto fail_fact5;
	}

	nn_loss_t* loss;
	loss = nn_loss_new(arch, dim, nn_loss_mse);
	if(loss == NULL)
	{
		goto fail_loss;
	}

	nn_tensor_t* Y = nn_tensor_new(dim);
	if(Y == NULL)
	{
		goto fail_Y;
	}

	if((nn_arch_attachLayer(arch, (nn_layer_t*) conv1)    == 0) ||
	   (nn_arch_attachLayer(arch, (nn_layer_t*) fact1)    == 0) ||
	   (nn_arch_attachLayer(arch, (nn_layer_t*) pool1)    == 0) ||
	   (nn_arch_attachLayer(arch, (nn_layer_t*) conv2)    == 0) ||
	   (nn_arch_attachLayer(arch, (nn_layer_t*) fact2)    == 0) ||
	   (nn_arch_attachLayer(arch, (nn_layer_t*) pool2)    == 0) ||
	   (nn_arch_attachLayer(arch, (nn_layer_t*) conv3)    == 0) ||
	   (nn_arch_attachLayer(arch, (nn_layer_t*) fact3)    == 0) ||
	   (nn_arch_attachLayer(arch, (nn_layer_t*) convT3)   == 0) ||
	   (nn_arch_attachLayer(arch, (nn_layer_t*) conv4)    == 0) ||
	   (nn_arch_attachLayer(arch, (nn_layer_t*) fact4)    == 0) ||
	   (nn_arch_attachLayer(arch, (nn_layer_t*) convT4)   == 0) ||
	   (nn_arch_attachLayer(arch, (nn_layer_t*) conv5)    == 0) ||
	   (nn_arch_attachLayer(arch, (nn_layer_t*) fact5)    == 0) ||
	   (nn_arch_attachLoss(arch,  (nn_loss_t*)  loss) == 0))
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

				nn_tensor_blit(Xt, Y, n + m, m);
				++bs;
			}

			// add noise to X
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

			// plot loss
			float l = nn_arch_loss(arch);
			if((step % 50) == 0)
			{
				fprintf(fplot, "%u %f\n", step, l);
				fflush(fplot);
			}

			LOGI("epoch=%u, step=%u, n=%u, loss=%f",
			     epoch, step, n, l);
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
				ret &= jsmn_stream_key(stream, "%s", "conv1");
				ret &= nn_convLayer_export(conv1, stream);
				ret &= jsmn_stream_key(stream, "%s", "fact1");
				ret &= nn_factLayer_export(fact1, stream);
				ret &= jsmn_stream_key(stream, "%s", "pool1");
				ret &= nn_poolingLayer_export(pool1, stream);
				ret &= jsmn_stream_key(stream, "%s", "conv2");
				ret &= nn_convLayer_export(conv2, stream);
				ret &= jsmn_stream_key(stream, "%s", "fact2");
				ret &= nn_factLayer_export(fact2, stream);
				ret &= jsmn_stream_key(stream, "%s", "pool2");
				ret &= nn_poolingLayer_export(pool2, stream);
				ret &= jsmn_stream_key(stream, "%s", "conv3");
				ret &= nn_convLayer_export(conv3, stream);
				ret &= jsmn_stream_key(stream, "%s", "fact3");
				ret &= nn_factLayer_export(fact3, stream);
				ret &= jsmn_stream_key(stream, "%s", "convT3");
				ret &= nn_convLayer_export(convT3, stream);
				ret &= jsmn_stream_key(stream, "%s", "conv4");
				ret &= nn_convLayer_export(conv4, stream);
				ret &= jsmn_stream_key(stream, "%s", "fact4");
				ret &= nn_factLayer_export(fact4, stream);
				ret &= jsmn_stream_key(stream, "%s", "convT4");
				ret &= nn_convLayer_export(convT4, stream);
				ret &= jsmn_stream_key(stream, "%s", "conv5");
				ret &= nn_convLayer_export(conv5, stream);
				ret &= jsmn_stream_key(stream, "%s", "fact5");
				ret &= nn_factLayer_export(fact5, stream);
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
	nn_factLayer_delete(&fact5);
	nn_convLayer_delete(&conv5);
	nn_convLayer_delete(&convT4);
	nn_factLayer_delete(&fact4);
	nn_convLayer_delete(&conv4);
	nn_convLayer_delete(&convT3);
	nn_factLayer_delete(&fact3);
	nn_convLayer_delete(&conv3);
	nn_poolingLayer_delete(&pool2);
	nn_factLayer_delete(&fact2);
	nn_convLayer_delete(&conv2);
	nn_poolingLayer_delete(&pool1);
	nn_factLayer_delete(&fact1);
	nn_convLayer_delete(&conv1);
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
		nn_factLayer_delete(&fact5);
	fail_fact5:
		nn_convLayer_delete(&conv5);
	fail_conv5:
		nn_convLayer_delete(&convT4);
	fail_convT4:
		nn_factLayer_delete(&fact4);
	fail_fact4:
		nn_convLayer_delete(&conv4);
	fail_conv4:
		nn_convLayer_delete(&convT3);
	fail_convT3:
		nn_factLayer_delete(&fact3);
	fail_fact3:
		nn_convLayer_delete(&conv3);
	fail_conv3:
		nn_poolingLayer_delete(&pool2);
	fail_pool2:
		nn_factLayer_delete(&fact2);
	fail_fact2:
		nn_convLayer_delete(&conv2);
	fail_conv2:
		nn_poolingLayer_delete(&pool1);
	fail_pool1:
		nn_factLayer_delete(&fact1);
	fail_fact1:
		nn_convLayer_delete(&conv1);
	fail_conv1:
		nn_tensor_delete(&X);
	fail_X:
		nn_arch_delete(&arch);
	fail_arch:
		nn_tensor_delete(&Xt);
	return EXIT_FAILURE;
}
