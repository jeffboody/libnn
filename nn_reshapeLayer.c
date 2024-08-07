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
#include "nn_reshapeLayer.h"
#include "nn_tensor.h"

/***********************************************************
* private                                                  *
***********************************************************/

static nn_tensor_t*
nn_reshapeLayer_computeFpFn(nn_layer_t* base,
                            int flags, uint32_t bs,
                            nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_reshapeLayer_t* self = (nn_reshapeLayer_t*) base;
	nn_tensor_t*       Y    = &self->Y;

	if(nn_dim_sizeEquals(nn_tensor_dim(X), &self->dimX) == 0)
	{
		LOGE("invalid");
		return NULL;
	}

	Y->data    = X->data;
	Y->sb_data = X->sb_data;

	return Y;
}

static nn_tensor_t*
nn_reshapeLayer_computeBpFn(nn_layer_t* base,
                            int flags, uint32_t bs,
                            nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY);

	return dL_dY;
}

static nn_dim_t*
nn_reshapeLayer_dimXFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_reshapeLayer_t* self = (nn_reshapeLayer_t*) base;

	return &self->dimX;
}

static nn_dim_t*
nn_reshapeLayer_dimYFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_reshapeLayer_t* self = (nn_reshapeLayer_t*) base;

	return nn_tensor_dim(&self->Y);
}

static int
nn_reshapeLayer_newCompute(nn_reshapeLayer_t* self,
                           nn_dim_t* dimY)
{
	ASSERT(self);

	nn_arch_t*   arch   = self->base.arch;
	nn_engine_t* engine = arch->engine;
	nn_tensor_t* Y      = &self->Y;

	vkk_updateMode_e um;
	um = vkk_compute_updateMode(engine->compute);

	Y->sb_dim = vkk_buffer_new(engine->engine, um,
	                           VKK_BUFFER_USAGE_STORAGE,
	                           sizeof(nn_dim_t),
	                           dimY);
	if(Y->sb_dim == NULL)
	{
		return 0;
	}

	return 1;
}

static void
nn_reshapeLayer_deleteCompute(nn_reshapeLayer_t* self)
{
	ASSERT(self);

	nn_tensor_t* Y = &self->Y;
	vkk_buffer_delete(&Y->sb_dim);
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_reshapeLayer_t*
nn_reshapeLayer_new(nn_arch_t* arch, nn_dim_t* dimX,
                    nn_dim_t* dimY)
{
	ASSERT(arch);
	ASSERT(dimX);
	ASSERT(dimY);

	size_t sizeX = nn_dim_sizeBytes(dimX);
	size_t sizeY = nn_dim_sizeBytes(dimY);
	if(sizeY > sizeX)
	{
		LOGE("invalid sizeX=%u, sizeY=%u",
		     (uint32_t) sizeX, (uint32_t) sizeY);
		return NULL;
	}

	nn_layerInfo_t info =
	{
		.arch          = arch,
		.compute_fp_fn = nn_reshapeLayer_computeFpFn,
		.compute_bp_fn = nn_reshapeLayer_computeBpFn,
		.dimX_fn       = nn_reshapeLayer_dimXFn,
		.dimY_fn       = nn_reshapeLayer_dimYFn,
	};

	nn_reshapeLayer_t* self;
	self = (nn_reshapeLayer_t*)
	       nn_layer_new(sizeof(nn_reshapeLayer_t), &info);
	if(self == NULL)
	{
		return NULL;
	}

	nn_tensor_t* Y = &self->Y;
	nn_dim_copy(dimX, &self->dimX);
	nn_dim_copy(dimY, nn_tensor_dim(Y));

	if(nn_reshapeLayer_newCompute(self, dimY) == 0)
	{
		goto fail_compute;
	}

	// success
	return self;

	// failure
	fail_compute:
		nn_layer_delete((nn_layer_t**) &self);
	return NULL;
}

void nn_reshapeLayer_delete(nn_reshapeLayer_t** _self)
{
	ASSERT(_self);

	nn_reshapeLayer_t* self = *_self;
	if(self)
	{
		nn_reshapeLayer_deleteCompute(self);
		nn_layer_delete((nn_layer_t**) _self);
	}
}

nn_reshapeLayer_t*
nn_reshapeLayer_import(nn_arch_t* arch, cc_jsmnVal_t* val)
{
	ASSERT(arch);
	ASSERT(val);

	if(val->type != CC_JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	cc_jsmnVal_t* val_dimX = NULL;
	cc_jsmnVal_t* val_dimY = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		cc_jsmnKeyval_t* kv;
		kv = (cc_jsmnKeyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == CC_JSMN_TYPE_OBJECT)
		{
			if(strcmp(kv->key, "dimX") == 0)
			{
				val_dimX = kv->val;
			}
			else if(strcmp(kv->key, "dimY") == 0)
			{
				val_dimY = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_dimX == NULL) || (val_dimY == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_dim_t dimX;
	if(nn_dim_import(&dimX, val_dimX) == 0)
	{
		return NULL;
	}

	nn_dim_t dimY;
	if(nn_dim_import(&dimY, val_dimY) == 0)
	{
		return NULL;
	}

	return nn_reshapeLayer_new(arch, &dimX, &dimY);
}

int nn_reshapeLayer_export(nn_reshapeLayer_t* self,
                           cc_jsmnStream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	nn_dim_t* dimX = &self->dimX;
	nn_dim_t* dimY = nn_tensor_dim(&self->Y);

	int ret = 1;
	ret &= cc_jsmnStream_beginObject(stream);
	ret &= cc_jsmnStream_key(stream, "%s", "dimX");
	ret &= nn_dim_export(dimX, stream);
	ret &= cc_jsmnStream_key(stream, "%s", "dimY");
	ret &= nn_dim_export(dimY, stream);
	ret &= cc_jsmnStream_end(stream);

	return ret;
}
