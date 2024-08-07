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
#include "libcc/rng/cc_rngNormal.h"
#include "libcc/rng/cc_rngUniform.h"
#include "libcc/cc_log.h"
#include "libcc/cc_memory.h"
#include "libnn/nn_arch.h"
#include "libnn/nn_coderLayer.h"
#include "libnn/nn_encdecLayer.h"
#include "libnn/nn_loss.h"
#include "libnn/nn_tensor.h"
#include "libnn/nn_urrdbLayer.h"
#include "cifar10_denoise.h"

#define CIFAR10_DENOISE_URRDB

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
                      cc_jsmnVal_t* val)
{
	ASSERT(engine);
	ASSERT(val);

	if(val->type != CC_JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	cc_jsmnVal_t* val_base    = NULL;
	cc_jsmnVal_t* val_bs      = NULL;
	cc_jsmnVal_t* val_fc      = NULL;
	cc_jsmnVal_t* val_mu      = NULL;
	cc_jsmnVal_t* val_sigma   = NULL;
	cc_jsmnVal_t* val_encdec0 = NULL;
	cc_jsmnVal_t* val_urrdb0  = NULL;
	cc_jsmnVal_t* val_coder1  = NULL;
	cc_jsmnVal_t* val_coder2  = NULL;
	cc_jsmnVal_t* val_loss    = NULL;

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
			else if(strcmp(kv->key, "encdec0") == 0)
			{
				val_encdec0 = kv->val;
			}
			else if(strcmp(kv->key, "urrdb0") == 0)
			{
				val_urrdb0 = kv->val;
			}
			else if(strcmp(kv->key, "coder1") == 0)
			{
				val_coder1 = kv->val;
			}
			else if(strcmp(kv->key, "coder2") == 0)
			{
				val_coder2 = kv->val;
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
	if((val_base   == NULL) ||
	   (val_bs     == NULL) ||
	   (val_fc     == NULL) ||
	   (val_mu     == NULL) ||
	   (val_sigma  == NULL) ||
	   ((val_encdec0 == NULL) &&
	    (val_urrdb0 == NULL)) ||
	   (val_coder1 == NULL) ||
	   (val_coder2 == NULL) ||
	   (val_loss   == NULL))
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
		goto failure;
	}

	self->X = nn_tensor_new(engine, &dim,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_COMPUTE);
	if(self->X == NULL)
	{
		goto failure;
	}

	if(val_encdec0)
	{
		self->encdec0 = nn_encdecLayer_import(&self->base,
		                                      val_encdec0);
		if(self->encdec0 == NULL)
		{
			goto failure;
		}
	}
	else
	{
		self->urrdb0 = nn_urrdbLayer_import(&self->base,
		                                    val_urrdb0);
		if(self->urrdb0 == NULL)
		{
			goto failure;
		}
	}

	self->coder1 = nn_coderLayer_import(&self->base,
	                                    val_coder1, NULL);
	if(self->coder1 == NULL)
	{
		goto failure;
	}

	self->coder2 = nn_coderLayer_import(&self->base,
	                                    val_coder2, NULL);
	if(self->coder2 == NULL)
	{
		goto failure;
	}

	self->loss = nn_loss_import(engine, val_loss);
	if(self->loss == NULL)
	{
		goto failure;
	}

	self->Ytio = nn_tensor_new(engine, &dim,
	                           NN_TENSOR_INIT_ZERO,
	                           NN_TENSOR_MODE_IO);
	if(self->Ytio == NULL)
	{
		goto failure;
	}

	self->Yt = nn_tensor_new(engine, &dim,
	                         NN_TENSOR_INIT_ZERO,
	                         NN_TENSOR_MODE_COMPUTE);
	if(self->Yt == NULL)
	{
		goto failure;
	}

	self->Yio = nn_tensor_new(engine, &dim,
	                          NN_TENSOR_INIT_ZERO,
	                          NN_TENSOR_MODE_IO);
	if(self->Yio == NULL)
	{
		goto failure;
	}

	if(self->encdec0)
	{
		if((nn_arch_attachLayer(&self->base, &self->encdec0->base) == 0) ||
		   (nn_arch_attachLayer(&self->base, &self->coder1->base) == 0)  ||
		   (nn_arch_attachLayer(&self->base, &self->coder2->base) == 0))
		{
			goto failure;
		}
	}
	else
	{
		if((nn_arch_attachLayer(&self->base, &self->urrdb0->base) == 0) ||
		   (nn_arch_attachLayer(&self->base, &self->coder1->base) == 0) ||
		   (nn_arch_attachLayer(&self->base, &self->coder2->base) == 0))
		{
			goto failure;
		}
	}

	cc_rngNormal_init(&self->rngN, self->mu, self->sigma);
	cc_rngUniform_init(&self->rngU);

	// success
	return self;

	// failure
	failure:
		cifar10_denoise_delete(&self);
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
		goto failure;
	}

	self->X = nn_tensor_new(engine, &dimX,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_COMPUTE);
	if(self->X == NULL)
	{
		goto failure;
	}

	nn_dim_t* dim = nn_tensor_dim(self->X);

	#ifdef CIFAR10_DENOISE_URRDB
	uint32_t blocks = 2;
	uint32_t nodes  = 2;
	nn_urrdbLayerInfo_t urrdb0_info =
	{
		.arch = &self->base,

		// blocks: number of dense blocks
		// nodes:  number of nodes per block (nodes >= 2)
		.dimX   = dim,
		.fc     = fc,
		.blocks = blocks,
		.nodes  = nodes,

		// begin/end
		.norm_flags0 = 0,
		.conv_size0  = 3,
		.skip_beta0  = 0.2f,
		.bn_mode0    = NN_CODER_BATCH_NORM_MODE_DISABLE,
		.fact_fn0    = NN_FACT_LAYER_FN_RELU,

		// dense blocks/nodes
		.norm_flags1 = 0,
		.conv_size1  = 3,
		.skip_beta1  = 0.2f, // add only
		.bn_mode1    = NN_CODER_BATCH_NORM_MODE_ENABLE,
		.fact_fn1    = NN_FACT_LAYER_FN_RELU,
	};

	self->urrdb0 = nn_urrdbLayer_new(&urrdb0_info);
	if(self->urrdb0 == NULL)
	{
		goto failure;
	}
	dim = nn_layer_dimY(&self->urrdb0->base);
	#else
	nn_encdecLayerInfo_t encdec0_info =
	{
		.arch         = &self->base,
		.sampler      = NN_ENCDEC_SAMPLER_LANCZOS,
		.dimX         = dim,
		.fc           = fc,
		.norm_flags0  = 0,
		.norm_flags12 = 0,
		.skip_mode    = NN_CODER_SKIP_MODE_CAT,
		.skip_beta    = 0.2f,
		.bn_mode0     = NN_CODER_BATCH_NORM_MODE_DISABLE,
		.bn_mode12    = NN_CODER_BATCH_NORM_MODE_ENABLE,
		.fact_fn      = NN_FACT_LAYER_FN_RELU,
		.a            = 3,
	};

	self->encdec0 = nn_encdecLayer_new(&encdec0_info);
	if(self->encdec0 == NULL)
	{
		goto failure;
	}
	dim = nn_layer_dimY(&self->encdec0->base);
	#endif

	nn_coderLayerInfo_t coder1_info =
	{
		.arch = &self->base,

		.dimX = dim,
		.fc   = fc,

		// conv layer
		.conv_flags  = 0,
		.conv_size   = 3,
		.conv_stride = 1,

		// skip layer
		// skip_coder must be set for add/cat modes
		.skip_mode  = NN_CODER_SKIP_MODE_NONE,
		.skip_coder = NULL,
		.skip_beta  = 0.0f,

		// bn layer
		.bn_mode = NN_CODER_BATCH_NORM_MODE_DISABLE,

		// fact layer
		.fact_fn = NN_FACT_LAYER_FN_RELU,
	};

	self->coder1 = nn_coderLayer_new(&coder1_info);
	if(self->coder1 == NULL)
	{
		goto failure;
	}
	dim = nn_layer_dimY(&self->coder1->base);

	nn_coderLayerInfo_t coder2_info =
	{
		.arch = &self->base,

		.dimX = dim,
		.fc   = xd,

		// conv layer
		.conv_flags  = 0,
		.conv_size   = 3,
		.conv_stride = 1,

		// skip layer
		// skip_coder must be set for add/cat modes
		.skip_mode  = NN_CODER_SKIP_MODE_NONE,
		.skip_coder = NULL,
		.skip_beta  = 0.0f,

		// bn layer
		.bn_mode = NN_CODER_BATCH_NORM_MODE_DISABLE,

		// fact layer
		.fact_fn = NN_FACT_LAYER_FN_SINK,
	};

	self->coder2 = nn_coderLayer_new(&coder2_info);
	if(self->coder2 == NULL)
	{
		goto failure;
	}
	dim = nn_layer_dimY(&self->coder2->base);

	self->loss = nn_loss_new(engine, dim, NN_LOSS_FN_MSE);
	if(self->loss == NULL)
	{
		goto failure;
	}

	self->Ytio = nn_tensor_new(engine, dim,
	                           NN_TENSOR_INIT_ZERO,
	                           NN_TENSOR_MODE_IO);
	if(self->Ytio == NULL)
	{
		goto failure;
	}

	self->Yt = nn_tensor_new(engine, dim,
	                         NN_TENSOR_INIT_ZERO,
	                         NN_TENSOR_MODE_COMPUTE);
	if(self->Yt == NULL)
	{
		goto failure;
	}

	self->Yio = nn_tensor_new(engine, dim,
	                          NN_TENSOR_INIT_ZERO,
	                          NN_TENSOR_MODE_IO);
	if(self->Yio == NULL)
	{
		goto failure;
	}

	if(self->encdec0)
	{
		if((nn_arch_attachLayer(&self->base, &self->encdec0->base) == 0) ||
		   (nn_arch_attachLayer(&self->base, &self->coder1->base) == 0)  ||
		   (nn_arch_attachLayer(&self->base, &self->coder2->base) == 0))
		{
			goto failure;
		}
	}
	else
	{
		if((nn_arch_attachLayer(&self->base, &self->urrdb0->base) == 0) ||
		   (nn_arch_attachLayer(&self->base, &self->coder1->base) == 0) ||
		   (nn_arch_attachLayer(&self->base, &self->coder2->base) == 0))
		{
			goto failure;
		}
	}

	cc_rngNormal_init(&self->rngN, mu, sigma);
	cc_rngUniform_init(&self->rngU);

	// success
	return self;

	// failure
	failure:
		cifar10_denoise_delete(&self);
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
		nn_coderLayer_delete(&self->coder2);
		nn_coderLayer_delete(&self->coder1);
		nn_urrdbLayer_delete(&self->urrdb0);
		nn_encdecLayer_delete(&self->encdec0);
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

	cc_jsmnVal_t* val = cc_jsmnVal_import(fname);
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

	cc_jsmnVal_delete(&val);

	// success
	return self;

	// failure
	fail_parse:
		cc_jsmnVal_delete(&val);
	return NULL;
}

int cifar10_denoise_export(cifar10_denoise_t* self,
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
	cc_jsmnStream_key(stream, "%s", "mu");
	cc_jsmnStream_double(stream, self->mu);
	cc_jsmnStream_key(stream, "%s", "sigma");
	cc_jsmnStream_double(stream, self->sigma);
	if(self->encdec0)
	{
		cc_jsmnStream_key(stream, "%s", "encdec0");
		nn_encdecLayer_export(self->encdec0, stream);
	}
	else
	{
		cc_jsmnStream_key(stream, "%s", "urrdb0");
		nn_urrdbLayer_export(self->urrdb0, stream);
	}
	cc_jsmnStream_key(stream, "%s", "coder1");
	nn_coderLayer_export(self->coder1, stream);
	cc_jsmnStream_key(stream, "%s", "coder2");
	nn_coderLayer_export(self->coder2, stream);
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

	return self->bs;
}
