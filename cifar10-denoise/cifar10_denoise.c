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

#define LOG_TAG "cifar10"
#include "jsmn/wrapper/jsmn_stream.h"
#include "libcc/math/cc_float.h"
#include "libcc/rng/cc_rngNormal.h"
#include "libcc/rng/cc_rngUniform.h"
#include "libcc/cc_log.h"
#include "libcc/cc_memory.h"
#include "libnn/nn_arch.h"
#include "libnn/nn_batchNormLayer.h"
#include "libnn/nn_coderLayer.h"
#include "libnn/nn_convLayer.h"
#include "libnn/nn_factLayer.h"
#include "libnn/nn_loss.h"
#include "libnn/nn_skipLayer.h"
#include "libnn/nn_tensor.h"
#include "cifar10_denoise.h"

/***********************************************************
* private                                                  *
***********************************************************/

static void
cifar10_denoise_addNoise(cifar10_denoise_t* self,
                         nn_tensor_t* X, nn_tensor_t* Yt)
{
	ASSERT(self);
	ASSERT(X);
	ASSERT(Yt);

	nn_dim_t* dimX = nn_tensor_dim(X);

	float    x;
	float    yt;
	float    n = 0.0f;
	uint32_t m;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(m = 0; m < self->bs; ++m)
	{
		for(i = 0; i < dimX->height; ++i)
		{
			for(j = 0; j < dimX->width; ++j)
			{
				for(k = 0; k < dimX->depth; ++k)
				{
					if((self->mu != 0.0) && (self->sigma != 0.0))
					{
						n = cc_rngNormal_rand1F(&self->rngN);
					}
					yt = nn_tensor_ioGet(Yt, m, i, j, k);
					x  = cc_clamp(yt + n, 0.0f, 1.0f);
					nn_tensor_ioSet(X, m, i, j, k, x);
				}
			}
		}
	}
}

static cifar10_denoise_t*
cifar10_denoise_parse(nn_engine_t* engine, uint32_t xh,
                      uint32_t xw, uint32_t xd,
                      jsmn_val_t* val)
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
	jsmn_val_t* val_mu    = NULL;
	jsmn_val_t* val_sigma = NULL;
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
			else if(strcmp(kv->key, "mu") == 0)
			{
				val_mu = kv->val;
			}
			else if(strcmp(kv->key, "sigma") == 0)
			{
				val_sigma = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_base  == NULL) ||
	   (val_bs    == NULL) ||
	   (val_fc    == NULL) ||
	   (val_mu    == NULL) ||
	   (val_sigma == NULL) ||
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

	cifar10_denoise_t* self;
	self = (cifar10_denoise_t*)
	       nn_arch_import(engine, sizeof(cifar10_denoise_t),
	                      val_base);
	if(self == NULL)
	{
		return NULL;
	}

	self->bs    = strtol(val_bs->data, NULL, 0);
	self->fc    = strtol(val_fc->data, NULL, 0);
	self->mu    = strtod(val_mu->data, NULL);
	self->sigma = strtod(val_sigma->data, NULL);

	nn_dim_t dim =
	{
		.count  = self->bs,
		.height = xh,
		.width  = xw,
		.depth  = xd,
	};

	self->Xio = nn_tensor_new(engine, &dim,
	                          NN_TENSOR_INIT_ZERO,
	                          NN_TENSOR_MODE_IO);
	if(self->Xio == NULL)
	{
		goto fail_Xio;
	}

	self->X = nn_tensor_new(engine, &dim,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_COMPUTE);
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

	self->loss = nn_loss_import(engine, val_loss);
	if(self->loss == NULL)
	{
		goto fail_loss;
	}

	self->Ytio = nn_tensor_new(engine, &dim,
	                           NN_TENSOR_INIT_ZERO,
	                           NN_TENSOR_MODE_IO);
	if(self->Ytio == NULL)
	{
		goto fail_Ytio;
	}

	self->Yt = nn_tensor_new(engine, &dim,
	                         NN_TENSOR_INIT_ZERO,
	                         NN_TENSOR_MODE_COMPUTE);
	if(self->Yt == NULL)
	{
		goto fail_Yt;
	}

	self->Yio = nn_tensor_new(engine, &dim,
	                          NN_TENSOR_INIT_ZERO,
	                          NN_TENSOR_MODE_IO);
	if(self->Yio == NULL)
	{
		goto fail_Yio;
	}

	if((nn_arch_attachLayer(&self->base, &self->bn0->base)   == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->enc1->base)  == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->enc2->base)  == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->dec3->base)  == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->dec4->base)  == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->convO->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->factO->base) == 0))
	{
		goto fail_attach;
	}

	cc_rngNormal_init(&self->rngN, self->mu, self->sigma);
	cc_rngUniform_init(&self->rngU);

	// success
	return self;

	// failure
	fail_attach:
		nn_tensor_delete(&self->Yio);
	fail_Yio:
		nn_tensor_delete(&self->Yt);
	fail_Yt:
		nn_tensor_delete(&self->Ytio);
	fail_Ytio:
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
		nn_tensor_delete(&self->Xio);
	fail_Xio:
		nn_arch_delete((nn_arch_t**) &self);
	return 0;
}

/***********************************************************
* public                                                   *
***********************************************************/

cifar10_denoise_t*
cifar10_denoise_new(nn_engine_t* engine, uint32_t bs,
                    uint32_t fc, uint32_t xh,
                    uint32_t xw, uint32_t xd,
                    double mu, double sigma)
{
	ASSERT(engine);

	nn_archState_t arch_state =
	{
		.adam_alpha  = 0.0001f,
		.adam_beta1  = 0.9f,
		.adam_beta2  = 0.999f,
		.adam_beta1t = 1.0f,
		.adam_beta2t = 1.0f,
		.adam_lambda = 0.25f*0.001f,
		.adam_nu     = 1.0f,
		.bn_momentum = 0.99f,
	};

	cifar10_denoise_t* self;
	self = (cifar10_denoise_t*)
	       nn_arch_new(engine, sizeof(cifar10_denoise_t),
	                   &arch_state);
	if(self == NULL)
	{
		return NULL;
	}

	self->bs    = bs;
	self->fc    = fc;
	self->mu    = mu;
	self->sigma = sigma;

	nn_dim_t dimX  =
	{
		.count  = bs,
		.height = xh,
		.width  = xw,
		.depth  = xd,
	};

	self->Xio = nn_tensor_new(engine, &dimX,
	                          NN_TENSOR_INIT_ZERO,
	                          NN_TENSOR_MODE_IO);
	if(self->Xio == NULL)
	{
		goto fail_Xio;
	}

	self->X = nn_tensor_new(engine, &dimX,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_COMPUTE);
	if(self->X == NULL)
	{
		goto fail_X;
	}

	nn_dim_t* dim = nn_tensor_dim(self->X);

	self->bn0 = nn_batchNormLayer_new(&self->base,
	                                  dim);
	if(self->bn0 == NULL)
	{
		goto fail_bn0;
	}

	nn_coderLayerInfo_t info_enc1 =
	{
		.arch        = &self->base,
		.dimX        = dim,
		.fc          = fc,
		.conv_flags  = NN_CONV_LAYER_FLAG_NORM_BSSN,
		.conv_size   = 3,
		.conv_stride = 2,
		.bn_mode     = NN_CODER_BATCH_NORM_MODE_ENABLE,
		.fact_fn     = NN_FACT_LAYER_FN_RELU,
	};

	self->enc1 = nn_coderLayer_new(&info_enc1);
	if(self->enc1 == NULL)
	{
		goto fail_enc1;
	}
	dim = nn_layer_dimY(&self->enc1->base);

	nn_coderLayerInfo_t info_enc2 =
	{
		.arch        = &self->base,
		.dimX        = dim,
		.fc          = fc,
		.conv_flags  = NN_CONV_LAYER_FLAG_NORM_BSSN,
		.conv_size   = 3,
		.conv_stride = 2,
		.bn_mode     = NN_CODER_BATCH_NORM_MODE_ENABLE,
		.fact_fn     = NN_FACT_LAYER_FN_RELU,
	};

	self->enc2 = nn_coderLayer_new(&info_enc2);
	if(self->enc2 == NULL)
	{
		goto fail_enc2;
	}
	dim = nn_layer_dimY(&self->enc2->base);

	nn_coderLayerInfo_t info_dec3 =
	{
		.arch        = &self->base,
		.dimX        = dim,
		.fc          = fc,
		.conv_flags  = NN_CONV_LAYER_FLAG_NORM_BSSN |
		               NN_CONV_LAYER_FLAG_TRANSPOSE,
		.conv_size   = 2,
		.conv_stride = 2,
		.bn_mode     = NN_CODER_BATCH_NORM_MODE_ENABLE,
		.fact_fn     = NN_FACT_LAYER_FN_RELU,
	};

	self->dec3 = nn_coderLayer_new(&info_dec3);
	if(self->dec3 == NULL)
	{
		goto fail_dec3;
	}
	dim = nn_layer_dimY(&self->dec3->base);

	nn_coderLayerInfo_t info_dec4 =
	{
		.arch        = &self->base,
		.dimX        = dim,
		.fc          = fc,
		.conv_flags  = NN_CONV_LAYER_FLAG_NORM_BSSN |
		               NN_CONV_LAYER_FLAG_TRANSPOSE,
		.conv_size   = 2,
		.conv_stride = 2,
		.bn_mode     = NN_CODER_BATCH_NORM_MODE_ENABLE,
		.fact_fn     = NN_FACT_LAYER_FN_RELU,
	};

	self->dec4 = nn_coderLayer_new(&info_dec4);
	if(self->dec4 == NULL)
	{
		goto fail_dec4;
	}
	dim = nn_layer_dimY(&self->dec4->base);

	nn_dim_t dimWO =
	{
		.count  = xd,
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

	self->loss = nn_loss_new(engine, dim, NN_LOSS_FN_MSE);
	if(self->loss == NULL)
	{
		goto fail_loss;
	}

	self->Ytio = nn_tensor_new(engine, dim,
	                           NN_TENSOR_INIT_ZERO,
	                           NN_TENSOR_MODE_IO);
	if(self->Ytio == NULL)
	{
		goto fail_Ytio;
	}

	self->Yt = nn_tensor_new(engine, dim,
	                         NN_TENSOR_INIT_ZERO,
	                         NN_TENSOR_MODE_COMPUTE);
	if(self->Yt == NULL)
	{
		goto fail_Yt;
	}

	self->Yio = nn_tensor_new(engine, dim,
	                          NN_TENSOR_INIT_ZERO,
	                          NN_TENSOR_MODE_IO);
	if(self->Yio == NULL)
	{
		goto fail_Yio;
	}

	if((nn_arch_attachLayer(&self->base, &self->bn0->base)   == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->enc1->base)  == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->enc2->base)  == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->dec3->base)  == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->dec4->base)  == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->convO->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->factO->base) == 0))
	{
		goto fail_attach;
	}

	cc_rngNormal_init(&self->rngN, mu, sigma);
	cc_rngUniform_init(&self->rngU);

	// success
	return self;

	// failure
	fail_attach:
		nn_tensor_delete(&self->Yio);
	fail_Yio:
		nn_tensor_delete(&self->Yt);
	fail_Yt:
		nn_tensor_delete(&self->Ytio);
	fail_Ytio:
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
		nn_tensor_delete(&self->Xio);
	fail_Xio:
		nn_arch_delete((nn_arch_t**) &self);
	return NULL;
}

void cifar10_denoise_delete(cifar10_denoise_t** _self)
{
	ASSERT(_self);

	cifar10_denoise_t* self = *_self;
	if(self)
	{
		nn_tensor_delete(&self->Yio);
		nn_tensor_delete(&self->Yt);
		nn_tensor_delete(&self->Ytio);
		nn_loss_delete(&self->loss);
		nn_factLayer_delete(&self->factO);
		nn_convLayer_delete(&self->convO);
		nn_coderLayer_delete(&self->dec4);
		nn_coderLayer_delete(&self->dec3);
		nn_coderLayer_delete(&self->enc2);
		nn_coderLayer_delete(&self->enc1);
		nn_batchNormLayer_delete(&self->bn0);
		nn_tensor_delete(&self->X);
		nn_tensor_delete(&self->Xio);
		nn_arch_delete((nn_arch_t**) &self);
	}
}

cifar10_denoise_t*
cifar10_denoise_import(nn_engine_t* engine,
                       uint32_t xh, uint32_t xw,
                       uint32_t xd,
                       const char* fname)
{
	ASSERT(engine);
	ASSERT(fname);

	jsmn_val_t* val = jsmn_val_import(fname);
	if(val == NULL)
	{
		return NULL;
	}

	cifar10_denoise_t* self;
	self = cifar10_denoise_parse(engine, xh, xw, xd, val);
	if(self == NULL)
	{
		goto fail_parse;
	}

	jsmn_val_delete(&val);

	// success
	return self;

	// failure
	fail_parse:
		jsmn_val_delete(&val);
	return NULL;
}

int cifar10_denoise_export(cifar10_denoise_t* self,
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
	jsmn_stream_key(stream, "%s", "mu");
	jsmn_stream_double(stream, self->mu);
	jsmn_stream_key(stream, "%s", "sigma");
	jsmn_stream_double(stream, self->sigma);
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
	if(jsmn_stream_export(stream, fname) == 0)
	{
		goto fail_export;
	}
	jsmn_stream_delete(&stream);

	// success
	return 1;

	// failure
	fail_export:
		jsmn_stream_delete(&stream);
	return 0;
}

int cifar10_denoise_exportX(cifar10_denoise_t* self,
                            const char* fname,
                            uint32_t n)
{
	ASSERT(self);
	ASSERT(fname);

	nn_dim_t* dim = nn_tensor_dim(self->Xio);

	return nn_tensor_ioExportPng(self->Xio, fname,
	                             n, 0, dim->depth,
	                             0.0f, 1.0f);
}

int cifar10_denoise_exportYt(cifar10_denoise_t* self,
                             const char* fname,
                             uint32_t n)
{
	ASSERT(self);
	ASSERT(fname);

	nn_dim_t* dim = nn_tensor_dim(self->Ytio);

	return nn_tensor_ioExportPng(self->Ytio, fname,
	                             n, 0, dim->depth,
	                             0.0f, 1.0f);
}

int cifar10_denoise_exportY(cifar10_denoise_t* self,
                            const char* fname,
                            uint32_t n)
{
	ASSERT(self);
	ASSERT(fname);

	nn_dim_t* dim = nn_tensor_dim(self->Yio);

	return nn_tensor_ioExportPng(self->Yio, fname,
	                             n, 0, dim->depth,
	                             0.0f, 1.0f);
}

void
cifar10_denoise_sampleXt(cifar10_denoise_t* self,
                         nn_tensor_t* Xt)
{
	ASSERT(self);
	ASSERT(Xt);

	cifar10_denoise_sampleXt2(self, Xt, self->Xio, self->Ytio);
}

void cifar10_denoise_sampleXt2(cifar10_denoise_t* self,
                               nn_tensor_t* Xt,
                               nn_tensor_t* X,
                               nn_tensor_t* Yt)
{
	ASSERT(self);
	ASSERT(Xt);
	ASSERT(X);
	ASSERT(Yt);

	nn_dim_t* dimXt = nn_tensor_dim(Xt);
	nn_dim_t* dimX  = nn_tensor_dim(X);
	nn_dim_t* dimYt = nn_tensor_dim(Yt);

	if((dimX->count   != dimYt->count)  ||
	   (dimXt->height != 32)            ||
	   (dimXt->height != dimX->height)  ||
	   (dimXt->height != dimYt->height) ||
	   (dimXt->width  != 32)            ||
	   (dimXt->width  != dimX->width)   ||
	   (dimXt->width  != dimYt->width)  ||
	   (dimXt->depth  != dimX->depth)   ||
	   (dimXt->depth  != dimYt->depth))
	{
		LOGE("invalid count=%u:%u, height=%u:%u:%u, width=%u:%u:%u, depth=%u:%u:%u",
		     dimX->count, dimYt->count,
		     dimXt->height, dimX->height, dimYt->height,
		     dimXt->width, dimX->width, dimYt->width,
		     dimXt->depth, dimX->depth, dimYt->depth);
		return;
	}

	uint32_t m;
	uint32_t n;
	uint32_t max = dimXt->count - 1;
	for(m = 0; m < self->bs; ++m)
	{
		n = cc_rngUniform_rand2U(&self->rngU, 0, max);
		nn_tensor_copy(Xt, Yt, n, m, 1);
	}

	// skip layers to perform poorly when noise is added
	cifar10_denoise_addNoise(self, X, Yt);
}

int cifar10_denoise_train(cifar10_denoise_t* self,
                          float* _loss)
{
	// _loss may be NULL
	ASSERT(self);

	uint32_t bs = self->bs;

	if((nn_tensor_copy(self->Xio, self->X, 0, 0, bs) == 0) ||
	   (nn_tensor_copy(self->Ytio, self->Yt, 0, 0, bs) == 0))
	{
		return 0;
	}

	nn_tensor_t* Y;
	Y = nn_arch_forwardPass(&self->base, 0, bs, self->X);
	if(Y == NULL)
	{
		return 0;
	}

	if(nn_tensor_copy(Y, self->Yio, 0, 0, bs) == 0)
	{
		return 0;
	}

	nn_tensor_t* dL_dY;
	dL_dY = nn_loss_pass(self->loss, 0, bs, Y, self->Yt);
	if(dL_dY == NULL)
	{
		return 0;
	}

	dL_dY = nn_arch_backprop(&self->base, 0, bs, dL_dY);
	if(dL_dY == NULL)
	{
		return 0;
	}

	if(_loss)
	{
		*_loss = nn_loss_loss(self->loss);
	}

	return 1;
}

int cifar10_denoise_predict(cifar10_denoise_t* self,
                            uint32_t bs)
{
	ASSERT(self);

	if(bs > cifar10_denoise_bs(self))
	{
		LOGE("invalid bs=%u", bs);
		return 0;
	}

	if(nn_tensor_copy(self->Xio, self->X, 0, 0, bs) == 0)
	{
		return 0;
	}

	nn_tensor_t* Y;
	Y = nn_arch_forwardPass(&self->base,
	                        NN_ARCH_FLAG_FP_BN_RUNNING,
	                        bs, self->X);
	if(Y == NULL)
	{
		return 0;
	}

	if(nn_tensor_copy(Y, self->Yio, 0, 0, bs) == 0)
	{
		return 0;
	}

	return 1;
}

uint32_t cifar10_denoise_bs(cifar10_denoise_t* self)
{
	ASSERT(self);

	nn_dim_t* dimX = nn_layer_dimX(&self->bn0->base);

	return dimX->count;
}
