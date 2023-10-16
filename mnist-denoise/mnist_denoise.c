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
#include <string.h>

#define LOG_TAG "mnist-denoise"
#include "jsmn/wrapper/jsmn_stream.h"
#include "libcc/math/cc_float.h"
#include "libcc/rng/cc_rngNormal.h"
#include "libcc/rng/cc_rngUniform.h"
#include "libcc/cc_log.h"
#include "libcc/cc_memory.h"
#include "libnn/mnist/nn_mnist.h"
#include "libnn/nn_arch.h"
#include "libnn/nn_batchNormLayer.h"
#include "libnn/nn_coderLayer.h"
#include "libnn/nn_convLayer.h"
#include "libnn/nn_factLayer.h"
#include "libnn/nn_loss.h"
#include "libnn/nn_poolingLayer.h"
#include "libnn/nn_skipLayer.h"
#include "libnn/nn_tensor.h"
#include "texgz/texgz_png.h"
#include "mnist_denoise.h"

/***********************************************************
* private                                                  *
***********************************************************/

static void
mnist_denoise_addNoise(mnist_denoise_t* self, uint32_t bs)
{
	ASSERT(self);

	nn_dim_t* dimX = nn_tensor_dim(self->X);
	uint32_t  xh   = dimX->height;
	uint32_t  xw   = dimX->width;

	float    x;
	float    yt;
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
				yt = nn_tensor_get(self->Yt, m, i, j, 0);
				n  = cc_rngNormal_rand1F(&self->rngN);
				x  = cc_clamp(yt + n, 0.0f, 1.0f);
				nn_tensor_set(self->X, m, i, j, 0, x);
			}
		}
	}
}

static int
mnist_denoise_exportPng(mnist_denoise_t* self,
                        const char* fname,
                        uint32_t n,
                        nn_tensor_t* T)
{
	ASSERT(self);
	ASSERT(fname);
	ASSERT(T);

	if(n >= mnist_denoise_bs(self))
	{
		LOGE("invalid n=%u", n);
		return 0;
	}

	nn_dim_t* dim = nn_tensor_dim(T);
	uint32_t  h   = dim->height;
	uint32_t  w   = dim->width;

	texgz_tex_t* tex;
	tex = texgz_tex_new(w, h, w, h,
	                    TEXGZ_UNSIGNED_BYTE,
	                    TEXGZ_RGBA, NULL);
	if(tex == NULL)
	{
		return 0;
	}

	float    t;
	uint32_t i;
	uint32_t j;
	unsigned char pixel[4] =
	{
		0x00, 0x00, 0x00, 0xFF,
	};
	for(i = 0; i < h; ++i)
	{
		for(j = 0; j < w; ++j)
		{
			t        = nn_tensor_get(T, n, i, j, 0);
			pixel[0] = (unsigned char)
			           cc_clamp(255.0f*t, 0.0f, 255.0f);
			pixel[1] = pixel[0];
			pixel[2] = pixel[0];
			texgz_tex_setPixel(tex, j, i, pixel);
		}
	}

	texgz_png_export(tex, fname);
	texgz_tex_delete(&tex);

	return 1;
}

static mnist_denoise_t*
mnist_denoise_parse(vkk_engine_t* engine, jsmn_val_t* val)
{
	ASSERT(engine);
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	jsmn_val_t* val_base  = NULL;
	jsmn_val_t* val_bs    = NULL;
	jsmn_val_t* val_fc    = NULL;
	jsmn_val_t* val_bn0   = NULL;
	jsmn_val_t* val_enc1  = NULL;
	jsmn_val_t* val_enc2  = NULL;
	jsmn_val_t* val_dec3  = NULL;
	jsmn_val_t* val_dec4  = NULL;
	jsmn_val_t* val_convO = NULL;
	jsmn_val_t* val_factO = NULL;
	jsmn_val_t* val_loss  = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		jsmn_keyval_t* kv;
		kv = (jsmn_keyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == JSMN_TYPE_OBJECT)
		{
			if(strcmp(kv->key, "base") == 0)
			{
				val_base = kv->val;
			}
			else if(strcmp(kv->key, "bn0") == 0)
			{
				val_bn0 = kv->val;
			}
			else if(strcmp(kv->key, "enc1") == 0)
			{
				val_enc1 = kv->val;
			}
			else if(strcmp(kv->key, "enc2") == 0)
			{
				val_enc2 = kv->val;
			}
			else if(strcmp(kv->key, "dec3") == 0)
			{
				val_dec3 = kv->val;
			}
			else if(strcmp(kv->key, "dec4") == 0)
			{
				val_dec4 = kv->val;
			}
			else if(strcmp(kv->key, "convO") == 0)
			{
				val_convO = kv->val;
			}
			else if(strcmp(kv->key, "factO") == 0)
			{
				val_factO = kv->val;
			}
			else if(strcmp(kv->key, "loss") == 0)
			{
				val_loss = kv->val;
			}
		}
		else if(kv->val->type == JSMN_TYPE_PRIMITIVE)
		{
			if(strcmp(kv->key, "bs") == 0)
			{
				val_bs = kv->val;
			}
			else if(strcmp(kv->key, "fc") == 0)
			{
				val_fc = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_base  == NULL) ||
	   (val_bs    == NULL) ||
	   (val_fc    == NULL) ||
	   (val_bn0   == NULL) ||
	   (val_enc1  == NULL) ||
	   (val_enc2  == NULL) ||
	   (val_dec3  == NULL) ||
	   (val_dec4  == NULL) ||
	   (val_convO == NULL) ||
	   (val_factO == NULL) ||
	   (val_loss  == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	mnist_denoise_t* self;
	self = (mnist_denoise_t*)
	       nn_arch_import(engine, sizeof(mnist_denoise_t),
	                      val_base);
	if(self == NULL)
	{
		return NULL;
	}

	self->bs = strtol(val_bs->data, NULL, 0);
	self->fc = strtol(val_fc->data, NULL, 0);

	self->Xt = nn_mnist_load(&self->base);
	if(self->Xt == NULL)
	{
		goto fail_Xt;
	}

	nn_dim_t* dimXt = nn_tensor_dim(self->Xt);

	nn_dim_t dim =
	{
		.count  = self->bs,
		.height = dimXt->height,
		.width  = dimXt->width,
		.depth  = 1,
	};

	self->X = nn_tensor_new(&self->base, &dim,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_IO);
	if(self->X == NULL)
	{
		goto fail_X;
	}

	self->bn0 = nn_batchNormLayer_import(&self->base,
	                                     val_bn0);
	if(self->bn0 == NULL)
	{
		goto fail_bn0;
	}

	self->enc1 = nn_coderLayer_import(&self->base,
	                                  val_enc1, NULL);
	if(self->enc1 == NULL)
	{
		goto fail_enc1;
	}

	self->enc2 = nn_coderLayer_import(&self->base,
	                                  val_enc2, NULL);
	if(self->enc2 == NULL)
	{
		goto fail_enc2;
	}

	self->dec3 = nn_coderLayer_import(&self->base,
	                                  val_dec3, self->enc1);
	if(self->dec3 == NULL)
	{
		goto fail_dec3;
	}

	self->dec4 = nn_coderLayer_import(&self->base,
	                                  val_dec4, self->enc1);
	if(self->dec4 == NULL)
	{
		goto fail_dec4;
	}

	self->convO = nn_convLayer_import(&self->base,
	                                  val_convO);
	if(self->convO == NULL)
	{
		goto fail_convO;
	}

	self->factO = nn_factLayer_import(&self->base,
	                                  val_factO);
	if(self->factO == NULL)
	{
		goto fail_factO;
	}

	self->loss = nn_loss_import(&self->base,
	                            val_loss);
	if(self->loss == NULL)
	{
		goto fail_loss;
	}

	self->Yt = nn_tensor_new(&self->base, &dim,
	                         NN_TENSOR_INIT_ZERO,
	                         NN_TENSOR_MODE_IO);
	if(self->Yt == NULL)
	{
		goto fail_Yt;
	}

	self->Y = nn_tensor_new(&self->base, &dim,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_IO);
	if(self->Y == NULL)
	{
		goto fail_Y;
	}

	if((nn_arch_attachLayer(&self->base, &self->bn0->base)   == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->enc1->base)  == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->enc2->base)  == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->dec3->base)  == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->dec4->base)  == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->convO->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->factO->base) == 0) ||
	   (nn_arch_attachLoss(&self->base,  self->loss)         == 0))
	{
		goto fail_attach;
	}

	cc_rngNormal_init(&self->rngN, 0.5f, 0.5f);
	cc_rngUniform_init(&self->rngU);

	// success
	return self;

	// failure
	fail_attach:
		nn_tensor_delete(&self->Y);
	fail_Y:
		nn_tensor_delete(&self->Yt);
	fail_Yt:
		nn_loss_delete(&self->loss);
	fail_loss:
		nn_factLayer_delete(&self->factO);
	fail_factO:
		nn_convLayer_delete(&self->convO);
	fail_convO:
		nn_coderLayer_delete(&self->dec4);
	fail_dec4:
		nn_coderLayer_delete(&self->dec3);
	fail_dec3:
		nn_coderLayer_delete(&self->enc2);
	fail_enc2:
		nn_coderLayer_delete(&self->enc1);
	fail_enc1:
		nn_batchNormLayer_delete(&self->bn0);
	fail_bn0:
		nn_tensor_delete(&self->X);
	fail_X:
		nn_tensor_delete(&self->Xt);
	fail_Xt:
		nn_arch_delete((nn_arch_t**) &self);
	return 0;
}

/***********************************************************
* public                                                   *
***********************************************************/

mnist_denoise_t*
mnist_denoise_new(vkk_engine_t* engine,
                  uint32_t bs, uint32_t fc)
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

	mnist_denoise_t* self;
	self = (mnist_denoise_t*)
	       nn_arch_new(engine, sizeof(mnist_denoise_t),
	                   &arch_state);
	if(self == NULL)
	{
		return NULL;
	}

	self->bs = bs;
	self->fc = fc;

	self->Xt = nn_mnist_load(&self->base);
	if(self->Xt == NULL)
	{
		goto fail_Xt;
	}

	nn_dim_t* dimXt = nn_tensor_dim(self->Xt);

	nn_dim_t dimX  =
	{
		.count  = bs,
		.height = dimXt->height,
		.width  = dimXt->width,
		.depth  = 1,
	};

	self->X = nn_tensor_new(&self->base, &dimX,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_IO);
	if(self->X == NULL)
	{
		goto fail_X;
	}

	nn_batchNormMode_e bn_mode = NN_BATCH_NORM_MODE_RUNNING;

	nn_dim_t* dim = nn_tensor_dim(self->X);

	self->bn0 = nn_batchNormLayer_new(&self->base,
	                                  bn_mode, dim);
	if(self->bn0 == NULL)
	{
		goto fail_bn0;
	}

	nn_coderBatchNormMode_e cbn_mode;
	cbn_mode = NN_CODER_BATCH_NORM_MODE_RUNNING;

	nn_coderLayerInfo_t info_enc1 =
	{
		.arch         = &self->base,
		.dimX         = dim,
		.fc           = fc,
		.conv_mode    = NN_CODER_CONV_MODE_3X3_RELU,
		.skip_mode    = NN_CODER_SKIP_MODE_NONE,
		.bn_mode      = cbn_mode,
		.repeat_mode  = NN_CODER_CONV_MODE_NONE,
		.post_op_mode = NN_CODER_OP_MODE_POOL_MAX_S2,
	};

	self->enc1 = nn_coderLayer_new(&info_enc1);
	if(self->enc1 == NULL)
	{
		goto fail_enc1;
	}
	dim = nn_layer_dimY(&self->enc1->base);

	nn_coderLayerInfo_t info_enc2 =
	{
		.arch         = &self->base,
		.dimX         = dim,
		.fc           = fc,
		.conv_mode    = NN_CODER_CONV_MODE_3X3_RELU,
		.skip_mode    = NN_CODER_SKIP_MODE_NONE,
		.bn_mode      = cbn_mode,
		.repeat_mode  = NN_CODER_CONV_MODE_NONE,
		.post_op_mode = NN_CODER_OP_MODE_POOL_MAX_S2,
	};

	self->enc2 = nn_coderLayer_new(&info_enc2);
	if(self->enc2 == NULL)
	{
		goto fail_enc2;
	}
	dim = nn_layer_dimY(&self->enc2->base);

	nn_coderLayerInfo_t info_dec3 =
	{
		.arch         = &self->base,
		.dimX         = dim,
		.fc           = fc,
		.conv_mode    = NN_CODER_CONV_MODE_3X3_RELU,
		.skip_mode    = NN_CODER_SKIP_MODE_NONE,
		.bn_mode      = cbn_mode,
		.repeat_mode  = NN_CODER_CONV_MODE_NONE,
		.post_op_mode = NN_CODER_OP_MODE_CONVT_2X2_S2,
	};

	self->dec3 = nn_coderLayer_new(&info_dec3);
	if(self->dec3 == NULL)
	{
		goto fail_dec3;
	}
	dim = nn_layer_dimY(&self->dec3->base);

	nn_coderLayerInfo_t info_dec4 =
	{
		.arch         = &self->base,
		.dimX         = dim,
		.fc           = fc,
		.conv_mode    = NN_CODER_CONV_MODE_3X3_RELU,
		.skip_mode    = NN_CODER_SKIP_MODE_NONE,
		.bn_mode      = cbn_mode,
		.repeat_mode  = NN_CODER_CONV_MODE_NONE,
		.post_op_mode = NN_CODER_OP_MODE_CONVT_2X2_S2,
	};

	self->dec4 = nn_coderLayer_new(&info_dec4);
	if(self->dec4 == NULL)
	{
		goto fail_dec4;
	}
	dim = nn_layer_dimY(&self->dec4->base);

	nn_dim_t dimWO =
	{
		.count  = 1,
		.width  = 3,
		.height = 3,
		.depth  = dim->depth,
	};

	self->convO = nn_convLayer_new(&self->base, dim, &dimWO, 1,
	                               NN_CONV_LAYER_FLAG_XAVIER);
	if(self->convO == NULL)
	{
		goto fail_convO;
	}
	dim = nn_layer_dimY(&self->convO->base);

	self->factO = nn_factLayer_new(&self->base, dim,
	                               NN_FACT_LAYER_FN_LOGISTIC);
	if(self->factO == NULL)
	{
		goto fail_factO;
	}

	self->loss = nn_loss_new(&self->base, dim, NN_LOSS_FN_MSE);
	if(self->loss == NULL)
	{
		goto fail_loss;
	}

	self->Yt = nn_tensor_new(&self->base, dim,
	                         NN_TENSOR_INIT_ZERO,
	                         NN_TENSOR_MODE_IO);
	if(self->Yt == NULL)
	{
		goto fail_Yt;
	}

	self->Y = nn_tensor_new(&self->base, dim,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_IO);
	if(self->Y == NULL)
	{
		goto fail_Y;
	}

	if((nn_arch_attachLayer(&self->base, &self->bn0->base)   == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->enc1->base)  == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->enc2->base)  == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->dec3->base)  == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->dec4->base)  == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->convO->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->factO->base) == 0) ||
	   (nn_arch_attachLoss(&self->base,  self->loss)         == 0))
	{
		goto fail_attach;
	}

	cc_rngNormal_init(&self->rngN, 0.5f, 0.5f);
	cc_rngUniform_init(&self->rngU);

	// success
	return self;

	// failure
	fail_attach:
		nn_tensor_delete(&self->Y);
	fail_Y:
		nn_tensor_delete(&self->Yt);
	fail_Yt:
		nn_loss_delete(&self->loss);
	fail_loss:
		nn_factLayer_delete(&self->factO);
	fail_factO:
		nn_convLayer_delete(&self->convO);
	fail_convO:
		nn_coderLayer_delete(&self->dec4);
	fail_dec4:
		nn_coderLayer_delete(&self->dec3);
	fail_dec3:
		nn_coderLayer_delete(&self->enc2);
	fail_enc2:
		nn_coderLayer_delete(&self->enc1);
	fail_enc1:
		nn_batchNormLayer_delete(&self->bn0);
	fail_bn0:
		nn_tensor_delete(&self->X);
	fail_X:
		nn_tensor_delete(&self->Xt);
	fail_Xt:
		nn_arch_delete((nn_arch_t**) &self);
	return NULL;
}

void mnist_denoise_delete(mnist_denoise_t** _self)
{
	ASSERT(_self);

	mnist_denoise_t* self = *_self;
	if(self)
	{
		nn_tensor_delete(&self->Y);
		nn_tensor_delete(&self->Yt);
		nn_loss_delete(&self->loss);
		nn_factLayer_delete(&self->factO);
		nn_convLayer_delete(&self->convO);
		nn_coderLayer_delete(&self->dec4);
		nn_coderLayer_delete(&self->dec3);
		nn_coderLayer_delete(&self->enc2);
		nn_coderLayer_delete(&self->enc1);
		nn_batchNormLayer_delete(&self->bn0);
		nn_tensor_delete(&self->X);
		nn_tensor_delete(&self->Xt);
		nn_arch_delete((nn_arch_t**) &self);
	}
}

mnist_denoise_t*
mnist_denoise_import(vkk_engine_t* engine,
                     const char* fname)
{
	ASSERT(engine);
	ASSERT(fname);

	FILE* f = fopen(fname, "r");
	if(f == NULL)
	{
		LOGE("invalid %s", fname);
		return NULL;
	}

	// get file size
	if(fseek(f, (long) 0, SEEK_END) == -1)
	{
		LOGE("fseek failed");
		goto fail_size;
	}
	size_t size = ftell(f);

	// rewind to start
	if(fseek(f, 0, SEEK_SET) == -1)
	{
		LOGE("fseek failed");
		goto fail_rewind;
	}

	// allocate buffer
	char* str = (char*) CALLOC(1, size);
	if(str == NULL)
	{
		LOGE("CALLOC failed");
		goto fail_str;
	}

	// read file
	if(fread((void*) str, size, 1, f) != 1)
	{
		LOGE("fread failed");
		goto fail_read;
	}

	jsmn_val_t* val = jsmn_val_new(str, size);
	if(val == NULL)
	{
		goto fail_val;
	}

	mnist_denoise_t* self;
	self = mnist_denoise_parse(engine, val);
	if(self == NULL)
	{
		goto fail_parse;
	}

	jsmn_val_delete(&val);
	FREE(str);
	fclose(f);

	// success
	return self;

	// failure
	fail_parse:
		jsmn_val_delete(&val);
	fail_val:
	fail_read:
		FREE(str);
	fail_str:
	fail_rewind:
	fail_size:
		fclose(f);
	return NULL;
}

int mnist_denoise_export(mnist_denoise_t* self,
                         const char* fname)
{
	ASSERT(self);
	ASSERT(fname);

	jsmn_stream_t* stream = jsmn_stream_new();
	if(stream == NULL)
	{
		return 0;
	}

	jsmn_stream_beginObject(stream);
	jsmn_stream_key(stream, "%s", "base");
	nn_arch_export(&self->base, stream);
	jsmn_stream_key(stream, "%s", "bs");
	jsmn_stream_int(stream, (int) self->bs);
	jsmn_stream_key(stream, "%s", "fc");
	jsmn_stream_int(stream, (int) self->fc);
	jsmn_stream_key(stream, "%s", "bn0");
	nn_batchNormLayer_export(self->bn0, stream);
	jsmn_stream_key(stream, "%s", "enc1");
	nn_coderLayer_export(self->enc1, stream);
	jsmn_stream_key(stream, "%s", "enc2");
	nn_coderLayer_export(self->enc2, stream);
	jsmn_stream_key(stream, "%s", "dec3");
	nn_coderLayer_export(self->dec3, stream);
	jsmn_stream_key(stream, "%s", "dec4");
	nn_coderLayer_export(self->dec4, stream);
	jsmn_stream_key(stream, "%s", "convO");
	nn_convLayer_export(self->convO, stream);
	jsmn_stream_key(stream, "%s", "factO");
	nn_factLayer_export(self->factO, stream);
	jsmn_stream_key(stream, "%s", "loss");
	nn_loss_export(self->loss, stream);
	jsmn_stream_end(stream);

	size_t size = 0;
	const char* buf = jsmn_stream_buffer(stream, &size);
	if(buf == NULL)
	{
		goto fail_buf;
	}

	FILE* f = fopen(fname, "w");
	if(f == NULL)
	{
		LOGE("invalid %s", fname);
		goto fail_fopen;
	}

	fprintf(f, "%s", buf);
	fclose(f);
	jsmn_stream_delete(&stream);

	// success
	return 1;

	// failure
	fail_fopen:
	fail_buf:
		jsmn_stream_delete(&stream);
	return 0;
}

int mnist_denoise_exportX(mnist_denoise_t* self,
                          const char* fname,
                          uint32_t n)
{
	ASSERT(self);
	ASSERT(fname);

	return mnist_denoise_exportPng(self, fname,
	                               n, self->X);
}

int mnist_denoise_exportYt(mnist_denoise_t* self,
                           const char* fname,
                           uint32_t n)
{
	ASSERT(self);
	ASSERT(fname);

	return mnist_denoise_exportPng(self, fname,
	                               n, self->Yt);
}

int mnist_denoise_exportY(mnist_denoise_t* self,
                          const char* fname,
                          uint32_t n)
{
	ASSERT(self);
	ASSERT(fname);

	return mnist_denoise_exportPng(self, fname,
	                               n, self->Y);
}

void mnist_denoise_sampleXt(mnist_denoise_t* self,
                            uint32_t bs)
{
	ASSERT(self);

	if(bs > mnist_denoise_bs(self))
	{
		LOGE("invalid bs=%u", bs);
		return;
	}

	nn_dim_t* dimXt = nn_tensor_dim(self->Xt);

	uint32_t m;
	uint32_t n;
	uint32_t max = dimXt->count - 1;
	for(m = 0; m < bs; ++m)
	{
		n = cc_rngUniform_rand2U(&self->rngU, 0, max);
		nn_tensor_blit(self->Xt, self->Yt, 1, n, m);
	}

	// skip layers to perform poorly when noise is added
	mnist_denoise_addNoise(self, bs);
}

int mnist_denoise_train(mnist_denoise_t* self,
                        uint32_t bs,
                        float* _loss)
{
	// _loss may be NULL
	ASSERT(self);

	if(bs > mnist_denoise_bs(self))
	{
		LOGE("invalid bs=%u", bs);
		return 0;
	}

	if(nn_arch_train(&self->base, NN_LAYER_MODE_TRAIN, bs,
	                 self->X, self->Yt, self->Y) == NULL)
	{
		return 0;
	}

	if(_loss)
	{
		*_loss = nn_arch_loss(&self->base);
	}

	return 1;
}

int mnist_denoise_predict(mnist_denoise_t* self,
                          uint32_t bs)
{
	ASSERT(self);

	if(bs > mnist_denoise_bs(self))
	{
		LOGE("invalid bs=%u", bs);
		return 0;
	}

	return nn_arch_predict(&self->base, bs,
	                       self->X, self->Y);
}

uint32_t mnist_denoise_countXt(mnist_denoise_t* self)
{
	ASSERT(self);

	nn_dim_t* dimXt = nn_tensor_dim(self->Xt);

	return dimXt->count;
}

uint32_t mnist_denoise_bs(mnist_denoise_t* self)
{
	ASSERT(self);

	nn_dim_t* dimX = nn_layer_dimX(&self->bn0->base);

	return dimX->count;
}
