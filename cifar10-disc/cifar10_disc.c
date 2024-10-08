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
#include "libcc/jsmn/cc_jsmnStream.h"
#include "libcc/math/cc_float.h"
#include "libcc/cc_log.h"
#include "libcc/cc_memory.h"
#include "libnn/cifar10/nn_cifar10.h"
#include "libnn/nn_arch.h"
#include "libnn/nn_batchNormLayer.h"
#include "libnn/nn_coderLayer.h"
#include "libnn/nn_convLayer.h"
#include "libnn/nn_factLayer.h"
#include "libnn/nn_loss.h"
#include "libnn/nn_skipLayer.h"
#include "libnn/nn_tensor.h"
#include "cifar10_disc.h"

/***********************************************************
* private                                                  *
***********************************************************/

static void cifar10_disc_initYt(cifar10_disc_t* self)
{
	ASSERT(self);

	nn_tensor_t* Yt  = self->Ytio;
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
				nn_tensor_ioSet(Yt, n, i, j, 0, 1.0f);
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
				nn_tensor_ioSet(Yt, n, i, j, 0, 0.0f);
			}
		}
	}
}

static cifar10_disc_t*
cifar10_disc_parse(nn_engine_t* engine,
                   uint32_t xh, uint32_t xw, uint32_t xd,
                   cc_jsmnVal_t* val)
{
	ASSERT(engine);
	ASSERT(val);

	if(val->type != CC_JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	cc_jsmnVal_t* val_base   = NULL;
	cc_jsmnVal_t* val_bs     = NULL;
	cc_jsmnVal_t* val_fc     = NULL;
	cc_jsmnVal_t* val_bn0    = NULL;
	cc_jsmnVal_t* val_coder1 = NULL;
	cc_jsmnVal_t* val_coder2 = NULL;
	cc_jsmnVal_t* val_coder3 = NULL;
	cc_jsmnVal_t* val_convO  = NULL;
	cc_jsmnVal_t* val_factO  = NULL;
	cc_jsmnVal_t* val_loss   = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		cc_jsmnKeyval_t* kv;
		kv = (cc_jsmnKeyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == CC_JSMN_TYPE_OBJECT)
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
		else if(kv->val->type == CC_JSMN_TYPE_PRIMITIVE)
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

	cifar10_disc_t* self;
	self = (cifar10_disc_t*)
	       nn_arch_import(engine, sizeof(cifar10_disc_t),
	                      val_base);
	if(self == NULL)
	{
		return NULL;
	}

	self->bs = strtol(val_bs->data, NULL, 0);
	self->fc = strtol(val_fc->data, NULL, 0);

	// depth is doubled for real/generated and noisy inputs
	nn_dim_t dim =
	{
		.count  = self->bs,
		.height = xh,
		.width  = xw,
		.depth  = 2*xd,
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

	self->loss = nn_loss_import(engine, val_loss);
	if(self->loss == NULL)
	{
		goto fail_loss;
	}

	nn_dim_t dimY =
	{
		.count  = self->bs,
		.height = xh/4,
		.width  = xw/4,
		.depth  = 1,
	};

	self->Ytio = nn_tensor_new(engine, &dimY,
	                           NN_TENSOR_INIT_ZERO,
	                           NN_TENSOR_MODE_IO);
	if(self->Ytio == NULL)
	{
		goto fail_Ytio;
	}

	self->Yt = nn_tensor_new(engine, &dimY,
	                         NN_TENSOR_INIT_ZERO,
	                         NN_TENSOR_MODE_COMPUTE);
	if(self->Yt == NULL)
	{
		goto fail_Yt;
	}

	cifar10_disc_initYt(self);

	self->Yio = nn_tensor_new(engine, &dimY,
	                          NN_TENSOR_INIT_ZERO,
	                          NN_TENSOR_MODE_IO);
	if(self->Yio == NULL)
	{
		goto fail_Yio;
	}

	if((nn_arch_attachLayer(&self->base, &self->bn0->base)    == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->coder1->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->coder2->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->coder3->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->convO->base)  == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->factO->base)  == 0))
	{
		goto fail_attach;
	}

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
		nn_tensor_delete(&self->Xio);
	fail_Xio:
		nn_arch_delete((nn_arch_t**) &self);
	return 0;
}

/***********************************************************
* public                                                   *
***********************************************************/

cifar10_disc_t*
cifar10_disc_new(nn_engine_t* engine, uint32_t bs,
                 uint32_t fc, uint32_t xh, uint32_t xw,
                 uint32_t xd)
{
	ASSERT(engine);

	nn_archState_t arch_state =
	{
		.adam_alpha  = 0.0001f,
		.adam_beta1  = 0.9f,
		.adam_beta2  = 0.999f,
		.adam_beta1t = 1.0f,
		.adam_beta2t = 1.0f,
		.bn_momentum = 0.99f,
	};

	cifar10_disc_t* self;
	self = (cifar10_disc_t*)
	       nn_arch_new(engine, sizeof(cifar10_disc_t),
	                   &arch_state);
	if(self == NULL)
	{
		return NULL;
	}

	self->bs = bs;
	self->fc = fc;

	// depth is doubled for real/generated and noisy inputs
	nn_dim_t dimX  =
	{
		.count  = bs,
		.height = xh,
		.width  = xw,
		.depth  = 2*xd,
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

	self->bn0 = nn_batchNormLayer_new(&self->base, dim);
	if(self->bn0 == NULL)
	{
		goto fail_bn0;
	}

	nn_coderLayerInfo_t info_coder1 =
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

	self->coder1 = nn_coderLayer_new(&info_coder1);
	if(self->coder1 == NULL)
	{
		goto fail_coder1;
	}
	dim = nn_layer_dimY(&self->coder1->base);

	nn_coderLayerInfo_t info_coder2 =
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

	self->coder2 = nn_coderLayer_new(&info_coder2);
	if(self->coder2 == NULL)
	{
		goto fail_coder2;
	}
	dim = nn_layer_dimY(&self->coder2->base);

	nn_coderLayerInfo_t info_coder3 =
	{
		.arch        = &self->base,
		.dimX        = dim,
		.fc          = fc,
		.conv_flags  = NN_CONV_LAYER_FLAG_NORM_BSSN,
		.conv_size   = 3,
		.conv_stride = 1,
		.bn_mode     = NN_CODER_BATCH_NORM_MODE_ENABLE,
		.fact_fn     = NN_FACT_LAYER_FN_RELU,
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

	cifar10_disc_initYt(self);

	self->Yio = nn_tensor_new(engine, dim,
	                          NN_TENSOR_INIT_ZERO,
	                          NN_TENSOR_MODE_IO);
	if(self->Yio == NULL)
	{
		goto fail_Yio;
	}

	if((nn_arch_attachLayer(&self->base, &self->bn0->base)    == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->coder1->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->coder2->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->coder3->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->convO->base)  == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->factO->base)  == 0))
	{
		goto fail_attach;
	}

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
		nn_tensor_delete(&self->Xio);
	fail_Xio:
		nn_arch_delete((nn_arch_t**) &self);
	return NULL;
}

void cifar10_disc_delete(cifar10_disc_t** _self)
{
	ASSERT(_self);

	cifar10_disc_t* self = *_self;
	if(self)
	{
		nn_tensor_delete(&self->Yio);
		nn_tensor_delete(&self->Yt);
		nn_tensor_delete(&self->Ytio);
		nn_loss_delete(&self->loss);
		nn_factLayer_delete(&self->factO);
		nn_convLayer_delete(&self->convO);
		nn_coderLayer_delete(&self->coder3);
		nn_coderLayer_delete(&self->coder2);
		nn_coderLayer_delete(&self->coder1);
		nn_batchNormLayer_delete(&self->bn0);
		nn_tensor_delete(&self->X);
		nn_tensor_delete(&self->Xio);
		nn_arch_delete((nn_arch_t**) &self);
	}
}

cifar10_disc_t*
cifar10_disc_import(nn_engine_t* engine,
                    uint32_t xh,
                    uint32_t xw,
                    uint32_t xd,
                    const char* fname)
{
	ASSERT(engine);
	ASSERT(fname);

	cc_jsmnVal_t* val = cc_jsmnVal_import(fname);
	if(val == NULL)
	{
		return NULL;
	}

	cifar10_disc_t* self;
	self = cifar10_disc_parse(engine, xh, xw, xd, val);
	if(self == NULL)
	{
		goto fail_parse;
	}

	cc_jsmnVal_delete(&val);

	// success
	return self;

	// failure
	fail_parse:
		cc_jsmnVal_delete(&val);
	return NULL;
}

int cifar10_disc_export(cifar10_disc_t* self,
                        const char* fname)
{
	ASSERT(self);
	ASSERT(fname);

	cc_jsmnStream_t* stream = cc_jsmnStream_new();
	if(stream == NULL)
	{
		return 0;
	}
	cc_jsmnStream_beginObject(stream);
	cc_jsmnStream_key(stream, "%s", "base");
	nn_arch_export(&self->base, stream);
	cc_jsmnStream_key(stream, "%s", "bs");
	cc_jsmnStream_int(stream, (int) self->bs);
	cc_jsmnStream_key(stream, "%s", "fc");
	cc_jsmnStream_int(stream, (int) self->fc);
	cc_jsmnStream_key(stream, "%s", "bn0");
	nn_batchNormLayer_export(self->bn0, stream);
	cc_jsmnStream_key(stream, "%s", "coder1");
	nn_coderLayer_export(self->coder1, stream);
	cc_jsmnStream_key(stream, "%s", "coder2");
	nn_coderLayer_export(self->coder2, stream);
	cc_jsmnStream_key(stream, "%s", "coder3");
	nn_coderLayer_export(self->coder3, stream);
	cc_jsmnStream_key(stream, "%s", "convO");
	nn_convLayer_export(self->convO, stream);
	cc_jsmnStream_key(stream, "%s", "factO");
	nn_factLayer_export(self->factO, stream);
	cc_jsmnStream_key(stream, "%s", "loss");
	nn_loss_export(self->loss, stream);
	cc_jsmnStream_end(stream);
	if(cc_jsmnStream_export(stream, fname) == 0)
	{
		goto fail_export;
	}
	cc_jsmnStream_delete(&stream);

	// success
	return 1;

	// failure
	fail_export:
		cc_jsmnStream_delete(&stream);
	return 0;
}

int cifar10_disc_exportXd0(cifar10_disc_t* self,
                           const char* fname,
                           uint32_t n)
{
	ASSERT(self);
	ASSERT(fname);

	// Ytr and Yg

	// depth is doubled for real/generated and noisy inputs
	nn_dim_t* dim = nn_tensor_dim(self->Xio);
	uint32_t  xd2 = dim->depth/2;

	return nn_tensor_ioExportPng(self->Xio, fname,
	                             n, 0, xd2,
	                             0.0f, 1.0f);
}

int cifar10_disc_exportXd1(cifar10_disc_t* self,
                           const char* fname,
                           uint32_t n)
{
	ASSERT(self);
	ASSERT(fname);

	// Cr and Cg

	// depth is doubled for real/generated and noisy inputs
	nn_dim_t* dim = nn_tensor_dim(self->Xio);
	uint32_t  xd2 = dim->depth/2;

	return nn_tensor_ioExportPng(self->Xio, fname,
	                             n, xd2, dim->depth - xd2,
	                             0.0f, 1.0f);
}

int cifar10_disc_exportY(cifar10_disc_t* self,
                         const char* fname,
                         uint32_t n)
{
	ASSERT(self);
	ASSERT(fname);

	return nn_tensor_ioExportPng(self->Yio, fname,
	                             n, 0, 1,
	                             0.0f, 1.0f);
}

void
cifar10_disc_sampleXt(cifar10_disc_t* self,
                      cifar10_denoise_t* dn,
                      nn_tensor_t* Xt)
{
	ASSERT(self);
	ASSERT(dn);
	ASSERT(Xt);

	cifar10_denoise_sampleXt(dn, Xt);
	if(cifar10_denoise_predict(dn, self->bs) == 0)
	{
		return;
	}

	nn_tensor_t* dnX  = dn->Xio;
	nn_tensor_t* dnYt = dn->Ytio;
	nn_tensor_t* dnY  = dn->Yio;

	// depth is doubled for real/generated and noisy inputs
	nn_tensor_t* X    = self->Xio;
	nn_dim_t*    dimX = nn_tensor_dim(X);
	uint32_t     n2   = dimX->count/2;
	uint32_t     xd2  = dimX->depth/2;

	// real samples
	float    x;
	float    y;
	uint32_t n;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(n = 0; n < n2; ++n)
	{
		for(i = 0; i < dimX->height; ++i)
		{
			for(j = 0; j < dimX->width; ++j)
			{
				for(k = 0; k < xd2; ++k)
				{
					// Ytr and Cr
					x = nn_tensor_ioGet(dnX, n, i, j, k);
					y = nn_tensor_ioGet(dnYt, n, i, j, k);
					nn_tensor_ioSet(X, n, i, j, k, y);
					nn_tensor_ioSet(X, n, i, j, k + xd2, x);
				}
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
				for(k = 0; k < xd2; ++k)
				{
					// Yg and Cg
					x = nn_tensor_ioGet(dnX, n, i, j, k);
					y = nn_tensor_ioGet(dnY, n, i, j, k);
					nn_tensor_ioSet(X, n, i, j, k, y);
					nn_tensor_ioSet(X, n, i, j, k + xd2, x);
				}
			}
		}
	}
}

int cifar10_disc_train(cifar10_disc_t* self,
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

int cifar10_disc_predict(cifar10_disc_t* self,
                         uint32_t bs)
{
	ASSERT(self);

	if(bs > cifar10_disc_bs(self))
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

uint32_t cifar10_disc_bs(cifar10_disc_t* self)
{
	ASSERT(self);

	nn_dim_t* dimX = nn_layer_dimX(&self->bn0->base);

	return dimX->count;
}
