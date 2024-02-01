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
#include "libcc/cc_log.h"
#include "libcc/cc_memory.h"
#include "libnn/nn_arch.h"
#include "libnn/nn_coderLayer.h"
#include "libnn/nn_convLayer.h"
#include "libnn/nn_factLayer.h"
#include "libnn/nn_loss.h"
#include "libnn/nn_tensor.h"
#include "cifar10_regen2.h"

/***********************************************************
* private                                                  *
***********************************************************/

static cifar10_regen2_t*
cifar10_regen2_parse(nn_engine_t* engine, jsmn_val_t* val)
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
	jsmn_val_t* val_coder1 = NULL;
	jsmn_val_t* val_coder2 = NULL;
	jsmn_val_t* val_coder3 = NULL;
	jsmn_val_t* val_coder4 = NULL;
	jsmn_val_t* val_coder5 = NULL;
	jsmn_val_t* val_convO  = NULL;
	jsmn_val_t* val_sinkO  = NULL;
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
			else if(strcmp(kv->key, "coder4") == 0)
			{
				val_coder4 = kv->val;
			}
			else if(strcmp(kv->key, "coder5") == 0)
			{
				val_coder5 = kv->val;
			}
			else if(strcmp(kv->key, "convO") == 0)
			{
				val_convO = kv->val;
			}
			else if(strcmp(kv->key, "sinkO") == 0)
			{
				val_sinkO = kv->val;
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
	   (val_coder1 == NULL) ||
	   (val_coder2 == NULL) ||
	   (val_coder3 == NULL) ||
	   (val_coder4 == NULL) ||
	   (val_coder5 == NULL) ||
	   (val_convO  == NULL) ||
	   (val_sinkO  == NULL) ||
	   (val_loss   == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	cifar10_regen2_t* self;
	self = (cifar10_regen2_t*)
	       nn_arch_import(engine, sizeof(cifar10_regen2_t),
	                      val_base);
	if(self == NULL)
	{
		return NULL;
	}

	self->bs = strtol(val_bs->data, NULL, 0);
	self->fc = strtol(val_fc->data, NULL, 0);

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

	self->coder4 = nn_coderLayer_import(&self->base,
	                                    val_coder4, NULL);
	if(self->coder4 == NULL)
	{
		goto fail_coder4;
	}

	self->coder5 = nn_coderLayer_import(&self->base,
	                                    val_coder5, NULL);
	if(self->coder5 == NULL)
	{
		goto fail_coder5;
	}

	self->convO = nn_convLayer_import(&self->base,
	                                  val_convO);
	if(self->convO == NULL)
	{
		goto fail_convO;
	}

	self->sinkO = nn_factLayer_import(&self->base,
	                                 val_sinkO);
	if(self->sinkO == NULL)
	{
		goto fail_sinkO;
	}

	self->loss = nn_loss_import(&self->base,
	                            val_loss);
	if(self->loss == NULL)
	{
		goto fail_loss;
	}

	if((nn_arch_attachLayer(&self->base, &self->coder1->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->coder2->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->coder3->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->coder4->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->coder5->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->convO->base)  == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->sinkO->base)  == 0) ||
	   (nn_arch_attachLoss(&self->base,  self->loss)          == 0))
	{
		goto fail_attach;
	}

	// success
	return self;

	// failure
	fail_attach:
		nn_loss_delete(&self->loss);
	fail_loss:
		nn_factLayer_delete(&self->sinkO);
	fail_sinkO:
		nn_convLayer_delete(&self->convO);
	fail_convO:
		nn_coderLayer_delete(&self->coder5);
	fail_coder5:
		nn_coderLayer_delete(&self->coder4);
	fail_coder4:
		nn_coderLayer_delete(&self->coder3);
	fail_coder3:
		nn_coderLayer_delete(&self->coder2);
	fail_coder2:
		nn_coderLayer_delete(&self->coder1);
	fail_coder1:
		nn_arch_delete((nn_arch_t**) &self);
	return 0;
}

/***********************************************************
* public                                                   *
***********************************************************/

cifar10_regen2_t*
cifar10_regen2_new(nn_engine_t* engine, uint32_t bs,
                   uint32_t fc, uint32_t xh,
                   uint32_t xw, uint32_t xd)
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
		.lerp_s      = 0.25,
		.lerp_min    = 0.5,
		.lerp_max    = 2.0,
	};

	cifar10_regen2_t* self;
	self = (cifar10_regen2_t*)
	       nn_arch_new(engine, sizeof(cifar10_regen2_t),
	                   &arch_state);
	if(self == NULL)
	{
		return NULL;
	}

	self->bs = bs;
	self->fc = fc;

	nn_dim_t dimX =
	{
		.count  = bs,
		.height = xh/2,
		.width  = xw/2,
		.depth  = xd + 1, // Y | R
	};

	nn_dim_t* dim = &dimX;

	// 1:        Xt > CF > Down > CF > CF > C > F > Y
	// 2: X < F < C < CF < Up   < CF < CF < Yt
	nn_coderLayerInfo_t info1 =
	{
		.arch       = &self->base,
		.dimX       = dim,
		.fc         = fc,
		.conv_flags = NN_CONV_LAYER_FLAG_NORM_BSSN,
		.conv_size  = 3,
		.bn_mode    = NN_CODER_BATCH_NORM_MODE_INSTANCE,
		.fact_fn    = NN_FACT_LAYER_FN_RELU,
	};

	self->coder1 = nn_coderLayer_new(&info1);
	if(self->coder1 == NULL)
	{
		goto fail_coder1;
	}
	dim = nn_layer_dimY(&self->coder1->base);

	nn_coderLayerInfo_t info2 =
	{
		.arch       = &self->base,
		.dimX       = dim,
		.fc         = fc,
		.conv_flags = NN_CONV_LAYER_FLAG_NORM_BSSN,
		.conv_size  = 3,
		.bn_mode    = NN_CODER_BATCH_NORM_MODE_INSTANCE,
		.fact_fn    = NN_FACT_LAYER_FN_RELU,
	};

	self->coder2 = nn_coderLayer_new(&info2);
	if(self->coder2 == NULL)
	{
		goto fail_coder2;
	}
	dim = nn_layer_dimY(&self->coder2->base);

	nn_coderLayerInfo_t info3 =
	{
		.arch       = &self->base,
		.dimX       = dim,
		.fc         = fc,
		.conv_flags = NN_CONV_LAYER_FLAG_NORM_BSSN,
		.conv_size  = 3,
		.bn_mode    = NN_CODER_BATCH_NORM_MODE_INSTANCE,
		.fact_fn    = NN_FACT_LAYER_FN_RELU,
	};

	self->coder3 = nn_coderLayer_new(&info3);
	if(self->coder3 == NULL)
	{
		goto fail_coder3;
	}
	dim = nn_layer_dimY(&self->coder3->base);

	nn_coderLayerInfo_t info4 =
	{
		.arch       = &self->base,
		.dimX       = dim,
		.fc         = fc,
		.conv_flags = NN_CONV_LAYER_FLAG_NORM_BSSN,
		.conv_size  = 3,
		.bn_mode    = NN_CODER_BATCH_NORM_MODE_INSTANCE,
		.fact_fn    = NN_FACT_LAYER_FN_RELU,
		.op_mode    = NN_CODER_OP_MODE_CONVT_2X2_S2,
	};

	self->coder4 = nn_coderLayer_new(&info4);
	if(self->coder4 == NULL)
	{
		goto fail_coder4;
	}
	dim = nn_layer_dimY(&self->coder4->base);

	nn_coderLayerInfo_t info5 =
	{
		.arch       = &self->base,
		.dimX       = dim,
		.fc         = fc,
		.conv_flags = NN_CONV_LAYER_FLAG_NORM_BSSN,
		.conv_size  = 3,
		.bn_mode    = NN_CODER_BATCH_NORM_MODE_INSTANCE,
		.fact_fn    = NN_FACT_LAYER_FN_RELU,
	};

	self->coder5 = nn_coderLayer_new(&info5);
	if(self->coder5 == NULL)
	{
		goto fail_coder5;
	}
	dim = nn_layer_dimY(&self->coder5->base);

	nn_dim_t dimW =
	{
		.count  = xd,
		.height = 3,
		.width  = 3,
		.depth  = dim->depth,
	};

	self->convO = nn_convLayer_new(&self->base, dim, &dimW, 1,
	                               NN_CONV_LAYER_FLAG_XAVIER);
	if(self->convO == NULL)
	{
		goto fail_convO;
	}
	dim = nn_layer_dimY(&self->convO->base);

	self->sinkO = nn_factLayer_new(&self->base, dim,
	                               NN_FACT_LAYER_FN_SINK);
	if(self->sinkO == NULL)
	{
		goto fail_sinkO;
	}

	self->loss = nn_loss_new(&self->base, dim,
	                         NN_LOSS_FN_MSE);
	if(self->loss == NULL)
	{
		goto fail_loss;
	}

	if((nn_arch_attachLayer(&self->base, &self->coder1->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->coder2->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->coder3->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->coder4->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->coder5->base) == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->convO->base)  == 0) ||
	   (nn_arch_attachLayer(&self->base, &self->sinkO->base)  == 0) ||
	   (nn_arch_attachLoss(&self->base,  self->loss)          == 0))
	{
		goto fail_attach;
	}

	// success
	return self;

	// failure
	fail_attach:
		nn_loss_delete(&self->loss);
	fail_loss:
		nn_factLayer_delete(&self->sinkO);
	fail_sinkO:
		nn_convLayer_delete(&self->convO);
	fail_convO:
		nn_coderLayer_delete(&self->coder5);
	fail_coder5:
		nn_coderLayer_delete(&self->coder4);
	fail_coder4:
		nn_coderLayer_delete(&self->coder3);
	fail_coder3:
		nn_coderLayer_delete(&self->coder2);
	fail_coder2:
		nn_coderLayer_delete(&self->coder1);
	fail_coder1:
		nn_arch_delete((nn_arch_t**) &self);
	return NULL;
}

void cifar10_regen2_delete(cifar10_regen2_t** _self)
{
	ASSERT(_self);

	cifar10_regen2_t* self = *_self;
	if(self)
	{
		nn_loss_delete(&self->loss);
		nn_factLayer_delete(&self->sinkO);
		nn_convLayer_delete(&self->convO);
		nn_coderLayer_delete(&self->coder5);
		nn_coderLayer_delete(&self->coder4);
		nn_coderLayer_delete(&self->coder3);
		nn_coderLayer_delete(&self->coder2);
		nn_coderLayer_delete(&self->coder1);
		nn_arch_delete((nn_arch_t**) &self);
	}
}

cifar10_regen2_t*
cifar10_regen2_import(nn_engine_t* engine,
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

	cifar10_regen2_t* self;
	self = cifar10_regen2_parse(engine, val);
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

int cifar10_regen2_export(cifar10_regen2_t* self,
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
	jsmn_stream_key(stream, "%s", "coder1");
	nn_coderLayer_export(self->coder1, stream);
	jsmn_stream_key(stream, "%s", "coder2");
	nn_coderLayer_export(self->coder2, stream);
	jsmn_stream_key(stream, "%s", "coder3");
	nn_coderLayer_export(self->coder3, stream);
	jsmn_stream_key(stream, "%s", "coder4");
	nn_coderLayer_export(self->coder4, stream);
	jsmn_stream_key(stream, "%s", "coder5");
	nn_coderLayer_export(self->coder5, stream);
	jsmn_stream_key(stream, "%s", "convO");
	nn_convLayer_export(self->convO, stream);
	jsmn_stream_key(stream, "%s", "sinkO");
	nn_factLayer_export(self->sinkO, stream);
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
