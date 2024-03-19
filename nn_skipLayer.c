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
#include "nn_skipLayer.h"
#include "nn_layer.h"
#include "nn_tensor.h"

/***********************************************************
* private                                                  *
***********************************************************/

typedef struct
{
	float beta;
} nn_skipLayerParam_t;

static nn_tensor_t*
nn_skipLayer_computeFpForkFn(nn_layer_t* base,
                             int flags, uint32_t bs,
                             nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_skipLayer_t* self = (nn_skipLayer_t*) base;

	// reference
	self->Y = X;

	return X;
}

static nn_tensor_t*
nn_skipLayer_computeBpForkFn(nn_layer_t* base,
                             int flags, uint32_t bs,
                             nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,xh,xw,xd)

	nn_skipLayer_t* self   = (nn_skipLayer_t*) base;
	nn_arch_t*      arch   = base->arch;
	nn_engine_t*    engine = arch->engine;

	if((self->skip == NULL) || (self->skip->dL_dX2 == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	// reference
	self->dL_dX1 = dL_dY;

	nn_tensor_t* Null   = engine->Null;
	nn_tensor_t* dL_dX1 = self->dL_dX1;
	nn_tensor_t* dL_dY1 = dL_dY;
	nn_tensor_t* dL_dY2 = self->skip->dL_dX2;
	nn_dim_t*    dimX   = &self->dimX;

	// fork:
	//   dL_dY1 = dL_dY
	//   dL_dY2 = skip->dL_dX2
	//   dL_dX1 = dL_dY (reference)
	//   dL_dX2 = NULL
	// sb100: bs
	// sb101: state
	// sb102: dim_dL_dX1
	// sb103: dL_dX1
	// sb104: dim_dL_dX2
	// sb105: dL_dX2
	// sb106: dim_dL_dY1
	// sb107: dL_dY1
	// sb108: dim_dL_dY2
	// sb109: dL_dY2
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
			.buffer  = dL_dX1->sb_dim,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dX1->sb_data,
		},
		{
			.binding = 4,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Null->sb_dim,
		},
		{
			.binding = 5,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Null->sb_data,
		},
		{
			.binding = 6,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY1->sb_dim,
		},
		{
			.binding = 7,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY1->sb_data,
		},
		{
			.binding = 8,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY2->sb_dim,
		},
		{
			.binding = 9,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY2->sb_data,
		},
	};

	vkk_compute_updateUniformSetRefs(engine->compute,
	                                 self->us1_bp, 10,
	                                 ua1_array);

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1_bp,
	};

	// nn_skipLayer_backpropFork
	// dispatch(RAW, bs, xh, xw, 1, 8, 8)
	vkk_computePipeline_t* cp;
	cp = engine->cp_skip_backpropFork;
	if(nn_engine_computeBind(engine, cp) == 0)
	{
		return NULL;
	}
	vkk_compute_bindUniformSets(engine->compute, 2, us_array);
	nn_engine_computeDispatch(engine, VKK_HAZARD_RAW,
	                          bs, dimX->height, dimX->width,
	                          1, 8, 8);

	return self->dL_dX1;
}

static nn_tensor_t*
nn_skipLayer_computeFpAddFn(nn_layer_t* base,
                            int flags, uint32_t bs,
                            nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_skipLayer_t* self   = (nn_skipLayer_t*) base;
	nn_arch_t*      arch   = base->arch;
	nn_engine_t*    engine = arch->engine;

	if((self->skip == NULL) || (self->skip->Y == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_tensor_t* X1   = X;
	nn_tensor_t* X2   = self->skip->Y;
	nn_tensor_t* Y    = self->Y;
	nn_dim_t*    dimX = &self->dimX;

	// add:
	//   X1 = X
	//   X2 = skip->Y
	//   Y: dim(bs,xh,xw,xd)
	//      x1h==x2h, x1w==x2w, x1d==x2d
	//   Y = beta*X1 + X2
	// sb100: bs
	// sb101: state
	// sb102: dimX1
	// sb103: X1
	// sb104: dimX2
	// sb105: X2
	// sb106: dimY
	// sb107: Y
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
			.buffer  = X1->sb_dim,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X1->sb_data,
		},
		{
			.binding = 4,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X2->sb_dim,
		},
		{
			.binding = 5,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X2->sb_data,
		},
		{
			.binding = 6,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Y->sb_dim,
		},
		{
			.binding = 7,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Y->sb_data,
		},
	};

	vkk_compute_updateUniformSetRefs(engine->compute,
	                                 self->us1_fp, 8,
	                                 ua1_array);

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1_fp,
	};

	// nn_skipLayer_forwardPassAdd
	// dispatch(RAW, bs, xh, xw, 1, 8, 8)
	vkk_computePipeline_t* cp;
	cp = engine->cp_skip_forwardPassAdd;
	if(nn_engine_computeBind(engine, cp) == 0)
	{
		return NULL;
	}
	vkk_compute_bindUniformSets(engine->compute, 2, us_array);
	nn_engine_computeDispatch(engine, VKK_HAZARD_RAW,
	                          bs, dimX->height, dimX->width,
	                          1, 8, 8);

	return Y;
}

static nn_tensor_t*
nn_skipLayer_computeBpAddFn(nn_layer_t* base,
                            int flags, uint32_t bs,
                            nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,xh,xw,xd)

	nn_skipLayer_t* self = (nn_skipLayer_t*) base;

	nn_arch_t*   arch   = base->arch;
	nn_engine_t* engine = arch->engine;

	nn_tensor_t* Null   = engine->Null;
	nn_tensor_t* dL_dY1 = dL_dY;
	nn_tensor_t* dL_dY2 = Null;
	nn_tensor_t* dL_dX1 = self->dL_dX1;
	nn_tensor_t* dL_dX2 = self->dL_dX2;
	nn_dim_t*    dimX   = &self->dimX;
	uint32_t     xh     = dimX->height;
	uint32_t     xw     = dimX->width;
	uint32_t     xd     = dimX->depth;

	// reference
	self->dL_dX2 = dL_dY;

	// add (skip_beta == 1.0): fast path
	if(self->skip_beta == 1.0f)
	{
		size_t size = bs*xh*xw*xd*sizeof(float);
		vkk_compute_copyStorage(engine->compute,
		                        VKK_HAZARD_RAW,
		                        dL_dY->sb_data,
		                        dL_dX1->sb_data,
		                        0, 0, size);
		return dL_dX1;
	}

	// add (skip_beta != 1.0):
	//   dL_dY1 = dL_dY
	//   dL_dY2 = NULL
	//   dL_dX1: dim(bs,xh,xw,xd)
	//   dL_dX1 = beta*dL_dY
	//   dL_dX2 = dL_dY (reference)
	// sb100: bs
	// sb101: state
	// sb102: dim_dL_dX1
	// sb103: dL_dX1
	// sb104: dim_dL_dX2
	// sb105: dL_dX2
	// sb106: dim_dL_dY1
	// sb107: dL_dY1
	// sb108: dim_dL_dY2
	// sb109: dL_dY2
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
			.buffer  = dL_dX1->sb_dim,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dX1->sb_data,
		},
		{
			.binding = 4,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dX2->sb_dim,
		},
		{
			.binding = 5,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dX2->sb_data,
		},
		{
			.binding = 6,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY1->sb_dim,
		},
		{
			.binding = 7,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY1->sb_data,
		},
		{
			.binding = 8,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY2->sb_dim,
		},
		{
			.binding = 9,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY2->sb_data,
		},
	};

	vkk_compute_updateUniformSetRefs(engine->compute,
	                                 self->us1_bp, 10,
	                                 ua1_array);

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1_bp,
	};

	// nn_skipLayer_backpropAdd
	// dispatch(RAW, bs, xh, xw, 1, 8, 8)
	vkk_computePipeline_t* cp;
	cp = engine->cp_skip_backpropAdd;
	if(nn_engine_computeBind(engine, cp) == 0)
	{
		return NULL;
	}
	vkk_compute_bindUniformSets(engine->compute, 2, us_array);
	nn_engine_computeDispatch(engine, VKK_HAZARD_RAW,
	                          bs, dimX->height, dimX->width,
	                          1, 8, 8);

	return dL_dX1;
}

static nn_tensor_t*
nn_skipLayer_computeFpCatFn(nn_layer_t* base,
                            int flags, uint32_t bs,
                            nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_skipLayer_t* self   = (nn_skipLayer_t*) base;
	nn_arch_t*      arch   = base->arch;
	nn_engine_t*    engine = arch->engine;

	if((self->skip == NULL) || (self->skip->Y == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_tensor_t* X1   = X;
	nn_tensor_t* X2   = self->skip->Y;
	nn_tensor_t* Y    = self->Y;
	nn_dim_t*    dimX = nn_tensor_dim(X);

	// cat:
	//   X1 = X
	//   X2 = skip->Y
	//   Y: dim(bs,xh,xw,x1d + x2d)
	//      x1h==x2h, x1w==x2w
	//   Y = X1 | X2
	// sb100: bs
	// sb101: state
	// sb102: dimX1
	// sb103: X1
	// sb104: dimX2
	// sb105: X2
	// sb106: dimY
	// sb107: Y
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
			.buffer  = X1->sb_dim,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X1->sb_data,
		},
		{
			.binding = 4,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X2->sb_dim,
		},
		{
			.binding = 5,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X2->sb_data,
		},
		{
			.binding = 6,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Y->sb_dim,
		},
		{
			.binding = 7,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Y->sb_data,
		},
	};

	vkk_compute_updateUniformSetRefs(engine->compute,
	                                 self->us1_fp, 8,
	                                 ua1_array);

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1_fp,
	};

	// nn_skipLayer_forwardPassCat
	// dispatch(RAW, bs, xh, xw, 1, 8, 8)
	vkk_computePipeline_t* cp;
	cp = engine->cp_skip_forwardPassCat;
	if(nn_engine_computeBind(engine, cp) == 0)
	{
		return NULL;
	}
	vkk_compute_bindUniformSets(engine->compute, 2, us_array);
	nn_engine_computeDispatch(engine, VKK_HAZARD_RAW,
	                          bs, dimX->height, dimX->width,
	                          1, 8, 8);

	return Y;
}

static nn_tensor_t*
nn_skipLayer_computeBpCatFn(nn_layer_t* base,
                            int flags, uint32_t bs,
                            nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,xh,xw,x1d + x2d)

	nn_skipLayer_t* self   = (nn_skipLayer_t*) base;
	nn_arch_t*      arch   = base->arch;
	nn_engine_t*    engine = arch->engine;

	nn_tensor_t* Null   = engine->Null;
	nn_tensor_t* dL_dY1 = dL_dY;
	nn_tensor_t* dL_dY2 = Null;
	nn_tensor_t* dL_dX1 = self->dL_dX1;
	nn_tensor_t* dL_dX2 = self->dL_dX2;
	nn_dim_t*    dimX   = &self->dimX;

	// cat:
	//   dL_dY1 = dL_dY
	//   dL_dY2 = NULL
	//   dL_dX1 : dim(bs,xh,xw,x1d)
	//   dL_dX2 : dim(bs,xh,xw,x2d)
	//   dL_dX1 = select(dL_dY1, 0, x1d)
	//   dL_dX2 = select(dL_dY1, x1d, x1d + x2d)
	// sb100: bs
	// sb101: state
	// sb102: dim_dL_dX1
	// sb103: dL_dX1
	// sb104: dim_dL_dX2
	// sb105: dL_dX2
	// sb106: dim_dL_dY1
	// sb107: dL_dY1
	// sb108: dim_dL_dY2
	// sb109: dL_dY2
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
			.buffer  = dL_dX1->sb_dim,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dX1->sb_data,
		},
		{
			.binding = 4,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dX2->sb_dim,
		},
		{
			.binding = 5,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dX2->sb_data,
		},
		{
			.binding = 6,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY1->sb_dim,
		},
		{
			.binding = 7,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY1->sb_data,
		},
		{
			.binding = 8,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY2->sb_dim,
		},
		{
			.binding = 9,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY2->sb_data,
		},
	};

	vkk_compute_updateUniformSetRefs(engine->compute,
	                                 self->us1_bp, 10,
	                                 ua1_array);

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
		self->us1_bp,
	};

	// nn_skipLayer_backpropCat
	// dispatch(RAW, bs, xh, xw, 1, 8, 8)
	vkk_computePipeline_t* cp;
	cp = engine->cp_skip_backpropCat;
	if(nn_engine_computeBind(engine, cp) == 0)
	{
		return NULL;
	}
	vkk_compute_bindUniformSets(engine->compute, 2, us_array);
	nn_engine_computeDispatch(engine, VKK_HAZARD_RAW,
	                          bs, dimX->height, dimX->width,
	                          1, 8, 8);

	return dL_dX1;
}

static int nn_skipLayer_newCompute(nn_skipLayer_t* self)
{
	ASSERT(self);

	nn_arch_t*   arch   = self->base.arch;
	nn_engine_t* engine = arch->engine;

	nn_skipLayerParam_t param =
	{
		.beta = self->skip_beta,
	};
	self->sb000_param = vkk_buffer_new(engine->engine,
	                                   VKK_UPDATE_MODE_STATIC,
	                                   VKK_BUFFER_USAGE_STORAGE,
	                                   sizeof(nn_skipLayerParam_t),
	                                   &param);
	if(self->sb000_param == NULL)
	{
		return 0;
	}

	self->us0 = vkk_uniformSet_new(engine->engine, 0, 0, NULL,
	                               engine->usf0_skip);
	if(self->us0 == NULL)
	{
		goto fail_us0;
	}

	self->us1_fp = vkk_uniformSet_new(engine->engine, 1, 0, NULL,
	                                  engine->usf1_skip_fp);
	if(self->us1_fp == NULL)
	{
		goto fail_us1_fp;
	}

	self->us1_bp = vkk_uniformSet_new(engine->engine, 1, 0, NULL,
	                                  engine->usf1_skip_bp);
	if(self->us1_bp == NULL)
	{
		goto fail_us1_bp;
	}

	// sb000: param (beta)
	vkk_uniformAttachment_t ua0_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = self->sb000_param,
		},
	};

	vkk_compute_updateUniformSetRefs(engine->compute,
	                                 self->us0, 1,
	                                 ua0_array);

	// success
	return 1;

	// failure
	fail_us1_bp:
		vkk_uniformSet_delete(&self->us1_fp);
	fail_us1_fp:
		vkk_uniformSet_delete(&self->us0);
	fail_us0:
		vkk_buffer_delete(&self->sb000_param);
	return 0;
}

static nn_dim_t*
nn_skipLayer_dimXFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_skipLayer_t* self = (nn_skipLayer_t*) base;

	return &self->dimX;
}

static nn_dim_t*
nn_skipLayer_dimYFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_skipLayer_t* self = (nn_skipLayer_t*) base;

	if((self->skip_mode == NN_SKIP_MODE_FORK_ADD) ||
	   (self->skip_mode == NN_SKIP_MODE_FORK_CAT))
	{
		return &self->dimX;
	}

	return nn_tensor_dim(self->Y);
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_skipLayer_t*
nn_skipLayer_newFork(nn_arch_t* arch, nn_dim_t* dimX,
                     nn_skipMode_e skip_mode)
{
	ASSERT(arch);
	ASSERT(dimX);
	ASSERT((skip_mode == NN_SKIP_MODE_FORK_ADD) ||
	       (skip_mode == NN_SKIP_MODE_FORK_CAT));

	nn_layerInfo_t info =
	{
		.arch          = arch,
		.compute_fp_fn = nn_skipLayer_computeFpForkFn,
		.compute_bp_fn = nn_skipLayer_computeBpForkFn,
		.dimX_fn       = nn_skipLayer_dimXFn,
		.dimY_fn       = nn_skipLayer_dimYFn,
	};

	nn_skipLayer_t* self;
	self = (nn_skipLayer_t*)
	       nn_layer_new(sizeof(nn_skipLayer_t), &info);
	if(self == NULL)
	{
		return NULL;
	}

	self->skip_mode = skip_mode;

	// skip reference is set by add/cat

	nn_dim_copy(dimX, &self->dimX);

	// Y reference is set by forwardPassForkFn

	if(nn_skipLayer_newCompute(self) == 0)
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

nn_skipLayer_t*
nn_skipLayer_newAdd(nn_arch_t* arch,
                    nn_dim_t* dimX1,
                    nn_skipLayer_t* skip_fork,
                    float skip_beta)
{
	ASSERT(arch);
	ASSERT(dimX1);
	ASSERT(skip_fork);

	nn_engine_t* engine = arch->engine;

	if(skip_beta == 0.0f)
	{
		// default
		skip_beta = 1.0f;
	}
	else if((skip_beta < 0.0f) || (skip_beta > 1.0f))
	{
		LOGE("invalid skip_beta=%f", skip_beta);
		return NULL;
	}

	// check required dimensions
	// x1h==x2h, x1w==x2w, x1d==x2d
	nn_dim_t* dimX2 = nn_layer_dimY(&skip_fork->base);
	if((dimX1->count  != dimX2->count)  ||
	   (dimX1->height != dimX2->height) ||
	   (dimX1->width  != dimX2->width)  ||
	   (dimX1->depth  != dimX2->depth))
	{
		LOGE("invalid");
		return NULL;
	}

	// only one skip reference is allowed
	if(skip_fork->skip)
	{
		LOGE("invalid");
		return NULL;
	}

	nn_layerInfo_t info =
	{
		.arch          = arch,
		.compute_fp_fn = nn_skipLayer_computeFpAddFn,
		.compute_bp_fn = nn_skipLayer_computeBpAddFn,
		.dimX_fn       = nn_skipLayer_dimXFn,
		.dimY_fn       = nn_skipLayer_dimYFn,
	};

	nn_skipLayer_t* self;
	self = (nn_skipLayer_t*)
	       nn_layer_new(sizeof(nn_skipLayer_t), &info);
	if(self == NULL)
	{
		return NULL;
	}

	self->skip_mode = NN_SKIP_MODE_ADD;
	self->skip_beta = skip_beta,

	// skip reference
	self->skip = skip_fork;

	nn_dim_copy(dimX1, &self->dimX);

	self->Y = nn_tensor_new(engine, dimX1,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_COMPUTE);
	if(self->Y == NULL)
	{
		goto fail_Y;
	}

	self->dL_dX1 = nn_tensor_new(engine, dimX1,
	                             NN_TENSOR_INIT_ZERO,
	                             NN_TENSOR_MODE_COMPUTE);
	if(self->dL_dX1 == NULL)
	{
		goto fail_dL_dX1;
	}

	// dL_dX2 reference is set by backpropAddFn

	// skip reference
	skip_fork->skip = self;

	if(nn_skipLayer_newCompute(self) == 0)
	{
		goto fail_compute;
	}

	// success
	return self;

	// failure
	fail_compute:
		nn_tensor_delete(&self->dL_dX1);
	fail_dL_dX1:
		nn_tensor_delete(&self->Y);
	fail_Y:
		nn_layer_delete((nn_layer_t**) &self);
	return NULL;
}

nn_skipLayer_t*
nn_skipLayer_newCat(nn_arch_t* arch,
                    nn_dim_t* dimX1,
                    nn_skipLayer_t* skip_fork)
{
	ASSERT(arch);
	ASSERT(skip_fork);

	nn_engine_t* engine = arch->engine;

	// check required dimensions
	// x1h==x2h, x1w==x2w
	nn_dim_t* dimX2 = nn_layer_dimY(&skip_fork->base);
	if((dimX1->count  != dimX2->count)  ||
	   (dimX1->height != dimX2->height) ||
	   (dimX1->width  != dimX2->width))
	{
		LOGE("invalid count=%u:%u, height=%u:%u, width=%u:%u",
		     dimX1->count,  dimX2->count,
		     dimX1->height, dimX2->height,
		     dimX1->width,  dimX2->width);
		return NULL;
	}

	// only one skip reference is allowed
	if(skip_fork->skip)
	{
		LOGE("invalid");
		return NULL;
	}

	nn_layerInfo_t info =
	{
		.arch          = arch,
		.compute_fp_fn = nn_skipLayer_computeFpCatFn,
		.compute_bp_fn = nn_skipLayer_computeBpCatFn,
		.dimX_fn       = nn_skipLayer_dimXFn,
		.dimY_fn       = nn_skipLayer_dimYFn,
	};

	nn_skipLayer_t* self;
	self = (nn_skipLayer_t*)
	       nn_layer_new(sizeof(nn_skipLayer_t), &info);
	if(self == NULL)
	{
		return NULL;
	}

	self->skip_mode = NN_SKIP_MODE_CAT;

	// skip reference
	self->skip = skip_fork;

	nn_dim_copy(dimX1, &self->dimX);

	nn_dim_t dimY =
	{
		.count  = dimX1->count,
		.height = dimX1->height,
		.width  = dimX1->width,
		.depth  = dimX1->depth + dimX2->depth,
	};

	self->Y = nn_tensor_new(engine, &dimY,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_COMPUTE);
	if(self->Y == NULL)
	{
		goto fail_Y;
	}

	self->dL_dX1 = nn_tensor_new(engine, dimX1,
	                             NN_TENSOR_INIT_ZERO,
	                             NN_TENSOR_MODE_COMPUTE);
	if(self->dL_dX1 == NULL)
	{
		goto fail_dL_dX1;
	}

	self->dL_dX2 = nn_tensor_new(engine, dimX2,
	                             NN_TENSOR_INIT_ZERO,
	                             NN_TENSOR_MODE_COMPUTE);
	if(self->dL_dX2 == NULL)
	{
		goto fail_dL_dX2;
	}

	// skip reference
	skip_fork->skip = self;

	if(nn_skipLayer_newCompute(self) == 0)
	{
		goto fail_compute;
	}

	// success
	return self;

	// failure
	fail_compute:
		nn_tensor_delete(&self->dL_dX2);
	fail_dL_dX2:
		nn_tensor_delete(&self->dL_dX1);
	fail_dL_dX1:
		nn_tensor_delete(&self->Y);
	fail_Y:
		nn_layer_delete((nn_layer_t**) &self);
	return NULL;
}

void nn_skipLayer_delete(nn_skipLayer_t** _self)
{
	ASSERT(_self);

	nn_skipLayer_t* self = *_self;
	if(self)
	{
		vkk_uniformSet_delete(&self->us1_bp);
		vkk_uniformSet_delete(&self->us1_fp);
		vkk_uniformSet_delete(&self->us0);
		vkk_buffer_delete(&self->sb000_param);

		// Y, dL_dX1, dL_dX2 may be references
		if(self->skip_mode == NN_SKIP_MODE_CAT)
		{
			nn_tensor_delete(&self->dL_dX2);
			nn_tensor_delete(&self->dL_dX1);
			nn_tensor_delete(&self->Y);
		}
		else if(self->skip_mode == NN_SKIP_MODE_ADD)
		{
			nn_tensor_delete(&self->dL_dX1);
			nn_tensor_delete(&self->Y);
		}
		nn_layer_delete((nn_layer_t**) _self);
	}
}

nn_skipLayer_t*
nn_skipLayer_import(nn_arch_t* arch, jsmn_val_t* val,
                    nn_skipLayer_t* skip_fork)
{
	// skip_fork is optional for add/cat
	ASSERT(arch);
	ASSERT(val);

	if(val->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid");
		return NULL;
	}

	jsmn_val_t* val_dimX      = NULL;
	jsmn_val_t* val_skip_mode = NULL;
	jsmn_val_t* val_skip_beta = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		jsmn_keyval_t* kv;
		kv = (jsmn_keyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == JSMN_TYPE_STRING)
		{
			if(strcmp(kv->key, "skip_mode") == 0)
			{
				val_skip_mode = kv->val;
			}
		}
		else if(kv->val->type == JSMN_TYPE_OBJECT)
		{
			if(strcmp(kv->key, "dimX") == 0)
			{
				val_dimX = kv->val;
			}
		}
		if(kv->val->type == JSMN_TYPE_PRIMITIVE)
		{
			if(strcmp(kv->key, "skip_beta") == 0)
			{
				val_skip_beta = kv->val;
			}
		}

		iter = cc_list_next(iter);
	}

	// check for required parameters
	if((val_dimX      == NULL) ||
	   (val_skip_mode == NULL) ||
	   (val_skip_beta == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	float skip_beta = strtof(val_skip_beta->data, NULL);

	nn_dim_t dimX;
	if(nn_dim_import(&dimX, val_dimX) == 0)
	{
		return NULL;
	}

	if(strcmp(val_skip_mode->data, "FORK_ADD") == 0)
	{
		return nn_skipLayer_newFork(arch, &dimX,
		                            NN_SKIP_MODE_FORK_ADD);
	}
	else if(strcmp(val_skip_mode->data, "FORK_CAT") == 0)
	{
		return nn_skipLayer_newFork(arch, &dimX,
		                            NN_SKIP_MODE_FORK_CAT);
	}
	else if(strcmp(val_skip_mode->data, "ADD") == 0)
	{
		return nn_skipLayer_newAdd(arch, &dimX, skip_fork,
		                           skip_beta);
	}
	else if(strcmp(val_skip_mode->data, "CAT") == 0)
	{
		return nn_skipLayer_newCat(arch, &dimX, skip_fork);
	}
	else
	{
		LOGE("invalid skip_mode=%s", val_skip_mode->data);
		return NULL;
	}
}

int nn_skipLayer_export(nn_skipLayer_t* self,
                        jsmn_stream_t* stream)
{
	ASSERT(self);
	ASSERT(stream);

	nn_dim_t* dimX = nn_skipLayer_dimXFn(&self->base);

	int ret = 1;
	ret &= jsmn_stream_beginObject(stream);
	ret &= jsmn_stream_key(stream, "%s", "dimX");
	ret &= nn_dim_export(dimX, stream);
	ret &= jsmn_stream_key(stream, "%s", "skip_mode");
	if(self->skip_mode == NN_SKIP_MODE_FORK_ADD)
	{
		ret &= jsmn_stream_string(stream, "%s", "FORK_ADD");
	}
	if(self->skip_mode == NN_SKIP_MODE_FORK_CAT)
	{
		ret &= jsmn_stream_string(stream, "%s", "FORK_CAT");
	}
	else if(self->skip_mode == NN_SKIP_MODE_ADD)
	{
		ret &= jsmn_stream_string(stream, "%s", "ADD");
	}
	else if(self->skip_mode == NN_SKIP_MODE_CAT)
	{
		ret &= jsmn_stream_string(stream, "%s", "CAT");
	}
	else
	{
		return 0;
	}
	ret &= jsmn_stream_key(stream, "%s", "skip_beta");
	ret &= jsmn_stream_float(stream, self->skip_beta);
	ret &= jsmn_stream_end(stream);

	return ret;
}
