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

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define LOG_TAG "nn"
#include "../libcc/cc_log.h"
#include "../libcc/cc_memory.h"
#include "nn_arch.h"
#include "nn_engine.h"
#include "nn_factLayer.h"
#include "nn_layer.h"
#include "nn_tensor.h"

const char* NN_FACT_LAYER_STRING_LINEAR   = "linear";
const char* NN_FACT_LAYER_STRING_LOGISTIC = "logistic";
const char* NN_FACT_LAYER_STRING_RELU     = "ReLU";
const char* NN_FACT_LAYER_STRING_PRELU    = "PReLU";
const char* NN_FACT_LAYER_STRING_TANH     = "tanh";
const char* NN_FACT_LAYER_STRING_SINK     = "sink";

/***********************************************************
* private                                                  *
***********************************************************/

static nn_tensor_t*
nn_factLayer_computeFpFn(nn_layer_t* base, int flags,
                         uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_factLayer_t* self   = (nn_factLayer_t*) base;
	nn_arch_t*      arch   = base->arch;
	nn_engine_t*    engine = arch->engine;

	nn_dim_t* dimX = nn_tensor_dim(X);

	vkk_computePipeline_t* cp[NN_FACT_LAYER_FN_COUNT] =
	{
		engine->cp_fact_forwardPassLinear,
		engine->cp_fact_forwardPassLogistic,
		engine->cp_fact_forwardPassReLU,
		engine->cp_fact_forwardPassPReLU,
		engine->cp_fact_forwardPassTanh,
		engine->cp_fact_forwardPassSink,
	};

	// sb100: bs
	// sb101: state
	// sb102: X
	vkk_uniformAttachment_t ua1_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = arch->sb100_bs,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = arch->sb101_state,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X->sb_data,
		},
	};
	vkk_compute_updateUniformSetRefs(engine->compute,
	                                 self->us1_fp,
	                                 3, ua1_array);

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1_fp,
	};

	// nn_factLayer_forwardPass
	// dispatch(RAW, bs, xh, xw, 1, 8, 8)
	if(nn_engine_computeBind(engine, cp[self->fn]) == 0)
	{
		return NULL;
	}
	vkk_compute_bindUniformSets(engine->compute, 2, us_array);
	nn_engine_computeDispatch(engine, VKK_HAZARD_RAW,
	                          bs, dimX->height, dimX->width,
	                          1, 8, 8);

	// reference for backprop
	self->X = X;

	return self->Y;
}

static nn_tensor_t*
nn_factLayer_computeBpFn(nn_layer_t* base, int flags,
                         uint32_t bs, nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY);

	nn_factLayer_t* self   = (nn_factLayer_t*) base;
	nn_arch_t*      arch   = base->arch;
	nn_engine_t*    engine = arch->engine;

	nn_dim_t* dimX = nn_tensor_dim(self->X);

	vkk_computePipeline_t* cp[NN_FACT_LAYER_FN_COUNT] =
	{
		engine->cp_fact_backpropLinear,
		engine->cp_fact_backpropLogistic,
		engine->cp_fact_backpropReLU,
		engine->cp_fact_backpropPReLU,
		engine->cp_fact_backpropTanh,
		engine->cp_fact_backpropSink,
	};

	// sb100: bs
	// sb101: state
	// sb102: X
	// sb103: dL_dY
	vkk_uniformAttachment_t ua1_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = arch->sb100_bs,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = arch->sb101_state,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->X->sb_data,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY->sb_data
		},
	};
	vkk_compute_updateUniformSetRefs(engine->compute,
	                                 self->us1_bp,
	                                 4, ua1_array);

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1_bp,
	};

	// nn_factLayer_backprop
	// dispatch(RAW, bs, xh, xw, 1, 8, 8)
	if(nn_engine_computeBind(engine, cp[self->fn]) == 0)
	{
		return NULL;
	}
	vkk_compute_bindUniformSets(engine->compute, 2, us_array);
	nn_engine_computeDispatch(engine, VKK_HAZARD_RAW,
	                          bs, dimX->height, dimX->width,
	                          1, 8, 8);

	// dL_dY replaced by dL_dX
	return dL_dY;
}

static nn_dim_t*
nn_factLayer_dimXFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_factLayer_t* self = (nn_factLayer_t*) base;

	// Y and X are the same dimensions
	// but X is a reference
	return nn_tensor_dim(self->Y);
}

static nn_dim_t*
nn_factLayer_dimYFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_factLayer_t* self = (nn_factLayer_t*) base;

	return nn_tensor_dim(self->Y);
}

static const char* nn_factLayer_string(nn_factLayerFn_e fn)
{
	ASSERT(fn >= 0);
	ASSERT(fn < NN_FACT_LAYER_FN_COUNT);

	const char* str_array[NN_FACT_LAYER_FN_COUNT] =
	{
		NN_FACT_LAYER_STRING_LINEAR,
		NN_FACT_LAYER_STRING_LOGISTIC,
		NN_FACT_LAYER_STRING_RELU,
		NN_FACT_LAYER_STRING_PRELU,
		NN_FACT_LAYER_STRING_TANH,
		NN_FACT_LAYER_STRING_SINK,
	};

	return str_array[fn];
}

static nn_factLayerFn_e nn_factLayer_function(const char* str)
{
	ASSERT(str);

	const char* str_fn[NN_FACT_LAYER_FN_COUNT] =
	{
		NN_FACT_LAYER_STRING_LINEAR,
		NN_FACT_LAYER_STRING_LOGISTIC,
		NN_FACT_LAYER_STRING_RELU,
		NN_FACT_LAYER_STRING_PRELU,
		NN_FACT_LAYER_STRING_TANH,
		NN_FACT_LAYER_STRING_SINK,
	};

	int i;
	for(i = 0; i < NN_FACT_LAYER_FN_COUNT; ++i)
	{
		if(strcmp(str, str_fn[i]) == 0)
		{
			return (nn_factLayerFn_e) i;
		}
	}

	LOGE("invalid %s", str);
	return NN_FACT_LAYER_FN_ERROR;
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_factLayer_t*
nn_factLayer_new(nn_arch_t* arch, nn_dim_t* dimX,
                 nn_factLayerFn_e fn)
{
	ASSERT(arch);
	ASSERT(dimX);

	nn_engine_t* engine = arch->engine;

	if(((int) fn < 0) || ((int) fn >= NN_FACT_LAYER_FN_COUNT))
	{
		LOGE("invalid fn=%i", (int) fn);
		return NULL;
	}

	nn_layerInfo_t info =
	{
		.arch          = arch,
		.compute_fp_fn = nn_factLayer_computeFpFn,
		.compute_bp_fn = nn_factLayer_computeBpFn,
		.dimX_fn       = nn_factLayer_dimXFn,
		.dimY_fn       = nn_factLayer_dimYFn,
	};

	nn_factLayer_t* self;
	self = (nn_factLayer_t*)
	       nn_layer_new(sizeof(nn_factLayer_t), &info);
	if(self == NULL)
	{
		return NULL;
	}

	self->fn = fn;

	self->Y = nn_tensor_new(engine, dimX,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_COMPUTE);
	if(self->Y == NULL)
	{
		goto fail_Y;
	}

	self->us0 = vkk_uniformSet_new(engine->engine, 0, 0, NULL,
	                               engine->usf0_fact);
	if(self->us0 == NULL)
	{
		goto fail_us0;
	}

	self->us1_fp = vkk_uniformSet_new(engine->engine, 1, 0,
	                                  NULL,
	                                  engine->usf1_fact_fp);
	if(self->us1_fp == NULL)
	{
		goto fail_us1_fp;
	}

	self->us1_bp = vkk_uniformSet_new(engine->engine, 1, 0,
	                                  NULL,
	                                  engine->usf1_fact_bp);
	if(self->us1_bp == NULL)
	{
		goto fail_us1_bp;
	}

	// sb000: dimX
	// sb001: Y
	vkk_uniformAttachment_t ua0_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->Y->sb_dim,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->Y->sb_data,
		},
	};

	vkk_compute_updateUniformSetRefs(engine->compute,
	                                 self->us0,
	                                 2, ua0_array);

	// success
	return self;

	// failure
	fail_us1_bp:
		vkk_uniformSet_delete(&self->us1_fp);
	fail_us1_fp:
		vkk_uniformSet_delete(&self->us0);
	fail_us0:
		nn_tensor_delete(&self->Y);
	fail_Y:
		nn_layer_delete((nn_layer_t**) &self);
	return NULL;
}

void nn_factLayer_delete(nn_factLayer_t** _self)
{
	ASSERT(_self);

	nn_factLayer_t* self = *_self;
	if(self)
	{
		vkk_uniformSet_delete(&self->us1_bp);
		vkk_uniformSet_delete(&self->us1_fp);
		vkk_uniformSet_delete(&self->us0);
		nn_tensor_delete(&self->Y);
		nn_layer_delete((nn_layer_t**) _self);
	}
}

nn_factLayer_t*
nn_factLayer_import(nn_arch_t* arch, jsmn_val_t* val)
{
	ASSERT(arch);
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	jsmn_val_t* val_dimX = NULL;
	jsmn_val_t* val_fn   = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		jsmn_keyval_t* kv;
		kv = (jsmn_keyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == JSMN_TYPE_STRING)
		{
			if(strcmp(kv->key, "fn") == 0)
			{
				val_fn = kv->val;
			}
		}
		else if(kv->val->type == JSMN_TYPE_OBJECT)
		{
			if(strcmp(kv->key, "dimX") == 0)
			{
				val_dimX = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_dimX == NULL) || (val_fn == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_dim_t dimX;
	if(nn_dim_import(&dimX, val_dimX) == 0)
	{
		return NULL;
	}

	nn_factLayerFn_e fn = nn_factLayer_function(val_fn->data);
	return nn_factLayer_new(arch, &dimX, fn);
}

int nn_factLayer_export(nn_factLayer_t* self,
                        jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	nn_dim_t* dimX = nn_factLayer_dimXFn(&self->base);

	const char* str_fn = nn_factLayer_string(self->fn);
	if(str_fn == NULL)
	{
		LOGE("invalid");
		return 0;
	}

	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	ret &= jsmn_stream_key(stream, "%s", "dimX");
	ret &= nn_dim_export(dimX, stream);
	ret &= jsmn_stream_key(stream, "%s", "fn");
	ret &= jsmn_stream_string(stream, "%s", str_fn);
	ret &= jsmn_stream_end(stream);

	return ret;
}
