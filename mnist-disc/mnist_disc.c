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
#include "mnist_disc.h"

/***********************************************************
* private                                                  *
***********************************************************/

static void mnist_disc_initYt(mnist_disc_t* self)
{
	ASSERT(self);

	nn_tensor_t* Yt  = self->Yt;
	nn_dim_t*    dim = nn_tensor_dim(Yt);
	uint32_t     n2  = dim->count/2;

	// real samples
	uint32_t n;
	uint32_t i;
	uint32_t j;
	for(n = 0; n < n2; ++n)
	{
		for(i = 0; i < dim->height; ++i)
		{
			for(j = 0; j < dim->width; ++j)
			{
				nn_tensor_set(Yt, n, i, j, 0, 1.0f);
			}
		}
	}

	// generated samples
	for(n = n2; n < dim->count; ++n)
	{
		for(i = 0; i < dim->height; ++i)
		{
			for(j = 0; j < dim->width; ++j)
			{
				nn_tensor_set(Yt, n, i, j, 0, 0.0f);
			}
		}
	}
}

static mnist_disc_t*
mnist_disc_parse(nn_engine_t* engine,
                 uint32_t xh, uint32_t xw,
                 jsmn_val_t* val)
{
	ASSERT(engine);
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	jsmn_val_t* val_base   = NULL;
	jsmn_val_t* val_bs     = NULL;
	jsmn_val_t* val_fc     = NULL;
	jsmn_val_t* val_bn0    = NULL;
	jsmn_val_t* val_coder1 = NULL;
	jsmn_val_t* val_coder2 = NULL;
	jsmn_val_t* val_coder3 = NULL;
	jsmn_val_t* val_convO  = NULL;
	jsmn_val_t* val_factO  = NULL;
	jsmn_val_t* val_loss   = NULL;

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
			else if(strcmp(kv->key, "coder1") == 0)
			{
				val_coder1 = kv->val;
			}
			else if(strcmp(kv->key, "coder2") == 0)
			{
				val_coder2 = kv->val;
			}
			else if(strcmp(kv->key, "coder3") == 0)
			{
				val_coder3 = kv->val;
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
	if((val_base   == NULL) ||
	   (val_bs     == NULL) ||
	   (val_fc     == NULL) ||
	   (val_bn0    == NULL) ||
	   (val_coder1 == NULL) ||
	   (val_coder2 == NULL) ||
	   (val_coder3 == NULL) ||
	   (val_convO  == NULL) ||
	   (val_factO  == NULL) ||
	   (val_loss   == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	mnist_disc_t* self;
	self = (mnist_disc_t*)
	       nn_arch_import(engine, sizeof(mnist_disc_t),
	                      val_base);
	if(self == NULL)
	{
		return NULL;
	}

	self->bs = strtol(val_bs->data, NULL, 0);
	self->fc = strtol(val_fc->data, NULL, 0);

	// depth is 2 for real/generated and noisy inputs
	nn_dim_t dim =
	{
		.count  = self->bs,
		.height = xh,
		.width  = xw,
		.depth  = 2,
	};

	self->X = nn_tensor_new(engine, &dim,
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

	self->coder1 = nn_coderLayer_import(&self->base,
	                                    val_coder1, NULL);
	if(self->coder1 == NULL)
	{
		goto fail_coder1;
	}

	self->coder2 = nn_coderLayer_import(&self->base,
	                                    val_coder2, NULL);
	if(self->coder2 == NULL)
	{
		goto fail_coder2;
	}

	self->coder3 = nn_coderLayer_import(&self->base,
	                                    val_coder3, NULL);
	if(self->coder3 == NULL)
	{
		goto fail_coder3;
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

	self->Yt = nn_tensor_new(engine, &dim,
	                         NN_TENSOR_INIT_ZERO,
	                         NN_TENSOR_MODE_IO);
	if(self->Yt == NULL)
	{
		goto fail_Yt;
	}

	mnist_disc_initYt(self);

	self->Y = nn_tensor_new(engine, &dim,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_IO);
	if(self->Y == NULL)
	{
		goto fail_Y;
	}

	if((nn_arch_attachLayer(&self->base, &self->bn0->base)    == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->coder1->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->coder2->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->coder3->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->convO->base)  == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->factO->base)  == 0) ||
	   (nn_arch_attachLoss(&self->base,  self->loss)          == 0))
	{
		goto fail_attach;
	}

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
		nn_coderLayer_delete(&self->coder3);
	fail_coder3:
		nn_coderLayer_delete(&self->coder2);
	fail_coder2:
		nn_coderLayer_delete(&self->coder1);
	fail_coder1:
		nn_batchNormLayer_delete(&self->bn0);
	fail_bn0:
		nn_tensor_delete(&self->X);
	fail_X:
		nn_arch_delete((nn_arch_t**) &self);
	return 0;
}

/***********************************************************
* public                                                   *
***********************************************************/

mnist_disc_t*
mnist_disc_new(nn_engine_t* engine, uint32_t bs,
               uint32_t fc, uint32_t xh, uint32_t xw)
{
	ASSERT(engine);

	nn_archState_t arch_state =
	{
		.learning_rate  = 0.00005f,
		.momentum_decay = 0.5f,
		.batch_momentum = 0.99f,
		.l2_lambda      = 0.01f,
	};

	mnist_disc_t* self;
	self = (mnist_disc_t*)
	       nn_arch_new(engine, sizeof(mnist_disc_t),
	                   &arch_state);
	if(self == NULL)
	{
		return NULL;
	}

	self->bs = bs;
	self->fc = fc;

	// depth is 2 for real/generated and noisy inputs
	nn_dim_t dimX  =
	{
		.count  = bs,
		.height = xh,
		.width  = xw,
		.depth  = 2,
	};

	self->X = nn_tensor_new(engine, &dimX,
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

	nn_coderLayerInfo_t info_coder1 =
	{
		.arch         = &self->base,
		.dimX         = dim,
		.fc           = fc,
		.conv_mode    = NN_CODER_CONV_MODE_3X3_RELU,
		.skip_mode    = NN_CODER_SKIP_MODE_NONE,
		.bn_mode      = cbn_mode,
		.repeat_mode  = NN_CODER_CONV_MODE_3X3_RELU,
		.repeat       = 2,
		.post_op_mode = NN_CODER_OP_MODE_CONV_3X3_S2,
	};

	self->coder1 = nn_coderLayer_new(&info_coder1);
	if(self->coder1 == NULL)
	{
		goto fail_coder1;
	}
	dim = nn_layer_dimY(&self->coder1->base);

	nn_coderLayerInfo_t info_coder2 =
	{
		.arch         = &self->base,
		.dimX         = dim,
		.fc           = fc,
		.conv_mode    = NN_CODER_CONV_MODE_3X3_RELU,
		.skip_mode    = NN_CODER_SKIP_MODE_NONE,
		.bn_mode      = cbn_mode,
		.repeat_mode  = NN_CODER_CONV_MODE_3X3_RELU,
		.repeat       = 2,
		.post_op_mode = NN_CODER_OP_MODE_CONV_3X3_S2,
	};

	self->coder2 = nn_coderLayer_new(&info_coder2);
	if(self->coder2 == NULL)
	{
		goto fail_coder2;
	}
	dim = nn_layer_dimY(&self->coder2->base);

	nn_coderLayerInfo_t info_coder3 =
	{
		.arch         = &self->base,
		.dimX         = dim,
		.fc           = fc,
		.conv_mode    = NN_CODER_CONV_MODE_3X3_RELU,
		.skip_mode    = NN_CODER_SKIP_MODE_NONE,
		.bn_mode      = cbn_mode,
		.repeat_mode  = NN_CODER_CONV_MODE_NONE,
		.post_op_mode = NN_CODER_OP_MODE_NONE,
	};

	self->coder3 = nn_coderLayer_new(&info_coder3);
	if(self->coder3 == NULL)
	{
		goto fail_coder3;
	}
	dim = nn_layer_dimY(&self->coder3->base);

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

	self->Yt = nn_tensor_new(engine, dim,
	                         NN_TENSOR_INIT_ZERO,
	                         NN_TENSOR_MODE_IO);
	if(self->Yt == NULL)
	{
		goto fail_Yt;
	}

	mnist_disc_initYt(self);

	self->Y = nn_tensor_new(engine, dim,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_IO);
	if(self->Y == NULL)
	{
		goto fail_Y;
	}

	if((nn_arch_attachLayer(&self->base, &self->bn0->base)    == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->coder1->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->coder2->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->coder3->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->convO->base)  == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->factO->base)  == 0) ||
	   (nn_arch_attachLoss(&self->base,  self->loss)          == 0))
	{
		goto fail_attach;
	}

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
		nn_coderLayer_delete(&self->coder3);
	fail_coder3:
		nn_coderLayer_delete(&self->coder2);
	fail_coder2:
		nn_coderLayer_delete(&self->coder1);
	fail_coder1:
		nn_batchNormLayer_delete(&self->bn0);
	fail_bn0:
		nn_tensor_delete(&self->X);
	fail_X:
		nn_arch_delete((nn_arch_t**) &self);
	return NULL;
}

void mnist_disc_delete(mnist_disc_t** _self)
{
	ASSERT(_self);

	mnist_disc_t* self = *_self;
	if(self)
	{
		nn_tensor_delete(&self->Y);
		nn_tensor_delete(&self->Yt);
		nn_loss_delete(&self->loss);
		nn_factLayer_delete(&self->factO);
		nn_convLayer_delete(&self->convO);
		nn_coderLayer_delete(&self->coder3);
		nn_coderLayer_delete(&self->coder2);
		nn_coderLayer_delete(&self->coder1);
		nn_batchNormLayer_delete(&self->bn0);
		nn_tensor_delete(&self->X);
		nn_arch_delete((nn_arch_t**) &self);
	}
}

mnist_disc_t*
mnist_disc_import(nn_engine_t* engine,
                  uint32_t xh,
                  uint32_t xw,
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

	mnist_disc_t* self;
	self = mnist_disc_parse(engine, xh, xw, val);
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

int mnist_disc_export(mnist_disc_t* self,
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
	jsmn_stream_key(stream, "%s", "coder1");
	nn_coderLayer_export(self->coder1, stream);
	jsmn_stream_key(stream, "%s", "coder2");
	nn_coderLayer_export(self->coder2, stream);
	jsmn_stream_key(stream, "%s", "coder3");
	nn_coderLayer_export(self->coder3, stream);
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

int mnist_disc_exportX(mnist_disc_t* self,
                       const char* fname,
                       uint32_t n)
{
	ASSERT(self);
	ASSERT(fname);

	return nn_tensor_exportPng(self->X, fname,
	                           n, 0.0f, 1.0f);
}

int mnist_disc_exportYt(mnist_disc_t* self,
                        const char* fname,
                        uint32_t n)
{
	ASSERT(self);
	ASSERT(fname);

	return nn_tensor_exportPng(self->Yt, fname,
	                           n, 0.0f, 1.0f);
}

int mnist_disc_exportY(mnist_disc_t* self,
                       const char* fname,
                       uint32_t n)
{
	ASSERT(self);
	ASSERT(fname);

	return nn_tensor_exportPng(self->Y, fname,
	                           n, 0.0f, 1.0f);
}

void
mnist_disc_sampleXt(mnist_disc_t* self,
                    mnist_denoise_t* dn,
                    nn_tensor_t* Xt)
{
	ASSERT(self);
	ASSERT(dn);
	ASSERT(Xt);

	mnist_denoise_sampleXt(dn, Xt);
	if(mnist_denoise_predict(dn, self->bs) == 0)
	{
		return;
	}

	nn_tensor_t* dnX  = dn->X;
	nn_tensor_t* dnYt = dn->Yt;
	nn_tensor_t* dnY  = dn->Y;

	// depth is 2 for real/generated and noisy inputs
	nn_tensor_t* X    = self->X;
	nn_dim_t*    dimX = nn_tensor_dim(X);
	uint32_t     n2   = dimX->count/2;

	// real samples
	float    x;
	float    y;
	uint32_t n;
	uint32_t i;
	uint32_t j;
	for(n = 0; n < n2; ++n)
	{
		for(i = 0; i < dimX->height; ++i)
		{
			for(j = 0; j < dimX->width; ++j)
			{
				x = nn_tensor_get(dnX, n, i, j, 0);
				y = nn_tensor_get(dnYt, n, i, j, 0);
				nn_tensor_set(X, n, i, j, 0, y);
				nn_tensor_set(X, n, i, j, 1, x);
			}
		}
	}

	// generated samples
	for(n = n2; n < dimX->count; ++n)
	{
		for(i = 0; i < dimX->height; ++i)
		{
			for(j = 0; j < dimX->width; ++j)
			{
				x = nn_tensor_get(dnX, n, i, j, 0);
				y = nn_tensor_get(dnY, n, i, j, 0);
				nn_tensor_set(X, n, i, j, 0, y);
				nn_tensor_set(X, n, i, j, 1, x);
			}
		}
	}
}

int mnist_disc_train(mnist_disc_t* self,
                     float* _loss)
{
	// _loss may be NULL
	ASSERT(self);

	if(nn_arch_train(&self->base, NN_LAYER_FLAG_TRAIN,
	                 self->bs, self->X, self->Yt,
	                 NULL) == NULL)
	{
		return 0;
	}

	if(_loss)
	{
		*_loss = nn_arch_loss(&self->base);
	}

	return 1;
}

int mnist_disc_predict(mnist_disc_t* self,
                       uint32_t bs)
{
	ASSERT(self);

	if(bs > mnist_disc_bs(self))
	{
		LOGE("invalid bs=%u", bs);
		return 0;
	}

	return nn_arch_predict(&self->base, bs,
	                       self->X, self->Y);
}

uint32_t mnist_disc_bs(mnist_disc_t* self)
{
	ASSERT(self);

	nn_dim_t* dimX = nn_layer_dimX(&self->bn0->base);

	return dimX->count;
}
