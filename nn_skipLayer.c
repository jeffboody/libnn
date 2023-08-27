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
#include "nn_skipLayer.h"
#include "nn_layer.h"
#include "nn_tensor.h"

/***********************************************************
* private                                                  *
***********************************************************/

static nn_tensor_t*
nn_skipLayer_forwardPassForkFn(nn_layer_t* base,
                               nn_layerMode_e mode,
                               uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_skipLayer_t* self = (nn_skipLayer_t*) base;

	self->Y = X;

	return X;
}

static nn_tensor_t*
nn_skipLayer_backpropAddFn(nn_layer_t* base, uint32_t bs,
                           nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,xh,xw,xd)

	nn_skipLayer_t* self = (nn_skipLayer_t*) base;

	self->dL_dX2 = dL_dY; // reference

	return dL_dY;
}

#ifdef NN_USE_COMPUTE

static nn_tensor_t*
nn_skipLayer_forwardPassAddFn(nn_layer_t* base,
                              nn_layerMode_e mode,
                              uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_skipLayer_t* self = (nn_skipLayer_t*) base;
	nn_arch_t*      arch = base->arch;

	if((self->skip == NULL) || (self->skip->Y == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_tensor_t* X1   = X;
	nn_tensor_t* X2   = self->skip->Y;
	nn_tensor_t* Y    = self->Y;
	nn_dim_t*    dimX = &self->dimX;

	// sb00: dimX/dimX1
	// sb01: X/X1
	// sb02: dimY
	// sb03: Y
	// sb04: dimX2
	// sb05: X2
	vkk_uniformAttachment_t ua0_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X1->sb_dim,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X1->sb_data,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Y->sb_dim,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Y->sb_data,
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
	};

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
	};

	// nn_skipLayer_forwardPassAdd
	// dispatch(RAW, bs, xh, xw, 1, 8, 8)
	vkk_compute_bindComputePipeline(arch->compute,
	                                arch->cp_skip_forwardPassAdd);
	vkk_compute_updateUniformSetRefs(arch->compute, self->us0,
	                                 6, ua0_array);
	vkk_compute_bindUniformSets(arch->compute, 1, us_array);
	vkk_compute_dispatch(arch->compute, VKK_HAZZARD_RAW,
	                     bs, dimX->height, dimX->width,
	                     1, 8, 8);

	return Y;
}

static nn_tensor_t*
nn_skipLayer_forwardPassCatFn(nn_layer_t* base,
                              nn_layerMode_e mode,
                              uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_skipLayer_t* self = (nn_skipLayer_t*) base;
	nn_arch_t*      arch = base->arch;

	if((self->skip == NULL) || (self->skip->Y == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_tensor_t* X1   = X;
	nn_tensor_t* X2   = self->skip->Y;
	nn_tensor_t* Y    = self->Y;
	nn_dim_t*    dimX = nn_tensor_dim(X);

	// sb00: dimX/dimX1
	// sb01: X/X1
	// sb02: dimY
	// sb03: Y
	// sb04: dimX2
	// sb05: X2
	vkk_uniformAttachment_t ua0_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X1->sb_dim,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = X1->sb_data,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Y->sb_dim,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Y->sb_data,
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
	};

	vkk_uniformSet_t* us_array[] =
	{
		self->us0,
	};

	// nn_skipLayer_forwardPassCat
	// dispatch(RAW, bs, xh, xw, 1, 8, 8)
	vkk_compute_bindComputePipeline(arch->compute,
	                                arch->cp_skip_forwardPassCat);
	vkk_compute_updateUniformSetRefs(arch->compute, self->us0,
	                                 6, ua0_array);
	vkk_compute_bindUniformSets(arch->compute, 1, us_array);
	vkk_compute_dispatch(arch->compute, VKK_HAZZARD_RAW,
	                     bs, dimX->height, dimX->width,
	                     1, 8, 8);

	return Y;
}

static nn_tensor_t*
nn_skipLayer_backpropForkFn(nn_layer_t* base, uint32_t bs,
                            nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,xh,xw,xd)

	nn_skipLayer_t* self = (nn_skipLayer_t*) base;
	nn_arch_t*      arch = base->arch;

	if((self->skip == NULL) || (self->skip->dL_dX2 == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_tensor_t* Null   = arch->Null;
	nn_tensor_t* dL_dY2 = self->skip->dL_dX2;
	nn_dim_t*    dimX   = &self->dimX;

	// sb10: dim_dL_dY
	// sb11: dL_dY
	// sb12: dim_dL_dX/dim_dL_dX1
	// sb13: dL_dX/dL_dX1
	// sb14: dim_dL_dX2/dim_dL_dY2
	// sb15: dL_dX2/dL_dY2
	// sb16: dim_dL_dY2
	// sb17: dL_dY2
	vkk_uniformAttachment_t ua1_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY->sb_dim,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY->sb_data,
		},
		{
			.binding = 2,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Null->sb_dim,
		},
		{
			.binding = 3,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Null->sb_data,
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
			.buffer  = dL_dY2->sb_dim,
		},
		{
			.binding = 7,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY2->sb_data,
		},
	};

	vkk_uniformSet_t* us_array[] =
	{
		self->us1,
	};

	// nn_skipLayer_backpropFork
	// dispatch(RAW, bs, xh, xw, 1, 8, 8)
	vkk_compute_bindComputePipeline(arch->compute,
	                                arch->cp_skip_backpropFork);
	vkk_compute_updateUniformSetRefs(arch->compute, self->us1,
	                                 8, ua1_array);
	vkk_compute_bindUniformSets(arch->compute, 1, us_array);
	vkk_compute_dispatch(arch->compute, VKK_HAZZARD_RAW,
	                     bs, dimX->height, dimX->width,
	                     1, 8, 8);

	// dL_dY replaced by dL_dY1 + dL_dY2
	return dL_dY;
}

static nn_tensor_t*
nn_skipLayer_backpropCatFn(nn_layer_t* base, uint32_t bs,
                           nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,xh,xw,x1d + x2d)

	nn_skipLayer_t* self = (nn_skipLayer_t*) base;
	nn_arch_t*      arch = base->arch;

	nn_tensor_t* Null   = arch->Null;
	nn_tensor_t* dL_dX1 = self->dL_dX1;
	nn_tensor_t* dL_dX2 = self->dL_dX2;
	nn_dim_t*    dimX   = &self->dimX;

	// sb10: dim_dL_dY
	// sb11: dL_dY
	// sb12: dim_dL_dX/dim_dL_dX1
	// sb13: dL_dX/dL_dX1
	// sb14: dim_dL_dX2/dim_dL_dY2
	// sb15: dL_dX2/dL_dY2
	// sb16: dim_dL_dY2
	// sb17: dL_dY2
	vkk_uniformAttachment_t ua1_array[] =
	{
		{
			.binding = 0,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY->sb_dim,
		},
		{
			.binding = 1,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = dL_dY->sb_data,
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
			.buffer  = Null->sb_dim,
		},
		{
			.binding = 7,
			.type    = VKK_UNIFORM_TYPE_STORAGE_REF,
			.buffer  = Null->sb_data,
		},
	};

	vkk_uniformSet_t* us_array[] =
	{
		self->us1,
	};

	// nn_skipLayer_backpropCat
	// dispatch(RAW, bs, xh, xw, 1, 8, 8)
	vkk_compute_bindComputePipeline(arch->compute,
	                                arch->cp_skip_backpropCat);
	vkk_compute_updateUniformSetRefs(arch->compute, self->us1,
	                                 8, ua1_array);
	vkk_compute_bindUniformSets(arch->compute, 1, us_array);
	vkk_compute_dispatch(arch->compute, VKK_HAZZARD_RAW,
	                     bs, dimX->height, dimX->width,
	                     1, 8, 8);

	return dL_dX1;
}

static int nn_skipLayer_newCompute(nn_skipLayer_t* self)
{
	ASSERT(self);

	nn_arch_t* arch = self->base.arch;

	self->us0 = vkk_uniformSet_new(arch->engine, 0, 0, NULL,
	                               arch->usf0_skip);
	if(self->us0 == NULL)
	{
		return 0;
	}

	self->us1 = vkk_uniformSet_new(arch->engine, 1, 0, NULL,
	                               arch->usf1_skip);
	if(self->us1 == NULL)
	{
		goto fail_us1;
	}

	// success
	return 1;

	// failure
	fail_us1:
		vkk_uniformSet_delete(&self->us0);
	return 0;
}

static void
nn_skipLayer_deleteCompute(nn_skipLayer_t* self)
{
	ASSERT(self);

	vkk_uniformSet_delete(&self->us1);
	vkk_uniformSet_delete(&self->us0);
}

#else // NN_USE_COMPUTE not defined

static nn_tensor_t*
nn_skipLayer_forwardPassAddFn(nn_layer_t* base,
                              nn_layerMode_e mode,
                              uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_skipLayer_t* self = (nn_skipLayer_t*) base;

	if((self->skip == NULL) || (self->skip->Y == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_tensor_t* X1   = X;
	nn_tensor_t* X2   = self->skip->Y;
	nn_tensor_t* Y    = self->Y;
	nn_dim_t*    dimX = &self->dimX;
	uint32_t     xh   = dimX->height;
	uint32_t     xw   = dimX->width;
	uint32_t     xd   = dimX->depth;

	// output
	float    x1;
	float    x2;
	uint32_t m;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(m = 0; m < bs; ++m)
	{
		for(i = 0; i < xh; ++i)
		{
			for(j = 0; j < xw; ++j)
			{
				for(k = 0; k < xd; ++k)
				{
					x1 = nn_tensor_get(X1, m, i, j, k);
					x2 = nn_tensor_get(X2, m, i, j, k);
					nn_tensor_set(Y, m, i, j, k, x1 + x2);
				}
			}
		}
	}

	return Y;
}

static nn_tensor_t*
nn_skipLayer_forwardPassCatFn(nn_layer_t* base,
                              nn_layerMode_e mode,
                              uint32_t bs, nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_skipLayer_t* self = (nn_skipLayer_t*) base;

	if((self->skip == NULL) || (self->skip->Y == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_tensor_t* X1    = X;
	nn_tensor_t* X2    = self->skip->Y;
	nn_tensor_t* Y     = self->Y;
	nn_dim_t*    dimX1 = nn_tensor_dim(X1);
	nn_dim_t*    dimX2 = nn_tensor_dim(X2);
	uint32_t     xh    = dimX1->height;
	uint32_t     xw    = dimX1->width;
	uint32_t     x1d   = dimX1->depth;
	uint32_t     x2d   = dimX2->depth;

	// output
	float    x1;
	float    x2;
	uint32_t m;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(m = 0; m < bs; ++m)
	{
		for(i = 0; i < xh; ++i)
		{
			for(j = 0; j < xw; ++j)
			{
				for(k = 0; k < x1d; ++k)
				{
					x1 = nn_tensor_get(X1, m, i, j, k);
					nn_tensor_set(Y, m, i, j, k, x1);
				}

				for(k = 0; k < x2d; ++k)
				{
					x2 = nn_tensor_get(X2, m, i, j, k);
					nn_tensor_set(Y, m, i, j, x1d + k, x2);
				}
			}
		}
	}

	return Y;
}

static nn_tensor_t*
nn_skipLayer_backpropForkFn(nn_layer_t* base, uint32_t bs,
                            nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,xh,xw,xd)

	nn_skipLayer_t* self = (nn_skipLayer_t*) base;

	if((self->skip == NULL) || (self->skip->dL_dX2 == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_tensor_t* dL_dY2 = self->skip->dL_dX2;
	nn_dim_t*    dimX   = &self->dimX;
	uint32_t     xh     = dimX->height;
	uint32_t     xw     = dimX->width;
	uint32_t     xd     = dimX->depth;

	// backpropagate loss
	float    dl_dy2;
	uint32_t m;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(m = 0; m < bs; ++m)
	{
		for(i = 0; i < xh; ++i)
		{
			for(j = 0; j < xw; ++j)
			{
				for(k = 0; k < xd; ++k)
				{
					// dL_dY replaced by dL_dY1 + dL_dY2
					dl_dy2 = nn_tensor_get(dL_dY2, m, i, j, k);
					nn_tensor_add(dL_dY, m, i, j, k, dl_dy2);
				}
			}
		}
	}

	// dL_dY replaced by dL_dY1 + dL_dY2
	return dL_dY;
}

static nn_tensor_t*
nn_skipLayer_backpropCatFn(nn_layer_t* base, uint32_t bs,
                           nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(bs,xh,xw,x1d + x2d)

	nn_skipLayer_t* self = (nn_skipLayer_t*) base;

	nn_tensor_t* dL_dX1 = self->dL_dX1;
	nn_tensor_t* dL_dX2 = self->dL_dX2;
	nn_dim_t*    dimX1  = nn_tensor_dim(dL_dX1);
	nn_dim_t*    dimX2  = nn_tensor_dim(dL_dX2);
	uint32_t     xh     = dimX1->height;
	uint32_t     xw     = dimX1->width;
	uint32_t     x1d    = dimX1->depth;
	uint32_t     x2d    = dimX2->depth;

	// backpropagate loss
	float    dl_dy;
	uint32_t m;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(m = 0; m < bs; ++m)
	{
		for(i = 0; i < xh; ++i)
		{
			for(j = 0; j < xw; ++j)
			{
				for(k = 0; k < x1d; ++k)
				{
					dl_dy = nn_tensor_get(dL_dY, m, i, j, k);
					nn_tensor_set(dL_dX1, m, i, j, k, dl_dy);
				}

				for(k = 0; k < x2d; ++k)
				{
					dl_dy = nn_tensor_get(dL_dY, m, i, j, x1d + k);
					nn_tensor_set(dL_dX2, m, i, j, k, dl_dy);
				}
			}
		}
	}

	return dL_dX1;
}

static int nn_skipLayer_newCompute(nn_skipLayer_t* self)
{
	return 1;
}

static void
nn_skipLayer_deleteCompute(nn_skipLayer_t* self)
{
}

#endif

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

	if(self->mode == NN_SKIP_LAYER_MODE_FORK)
	{
		return &self->dimX;
	}

	return nn_tensor_dim(self->Y);
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_skipLayer_t*
nn_skipLayer_newFork(nn_arch_t* arch, nn_dim_t* dimX)
{
	ASSERT(arch);
	ASSERT(dimX);

	nn_layerInfo_t info =
	{
		.arch            = arch,
		.forward_pass_fn = nn_skipLayer_forwardPassForkFn,
		.backprop_fn     = nn_skipLayer_backpropForkFn,
		.dimX_fn         = nn_skipLayer_dimXFn,
		.dimY_fn         = nn_skipLayer_dimYFn,
	};

	nn_skipLayer_t* self;
	self = (nn_skipLayer_t*)
	       nn_layer_new(sizeof(nn_skipLayer_t), &info);
	if(self == NULL)
	{
		return NULL;
	}

	self->mode = NN_SKIP_LAYER_MODE_FORK;

	// skip is set by add/cat

	nn_dim_copy(dimX, &self->dimX);

	// Y is set by forwardPassForkFn

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
                    nn_skipLayer_t* skip_fork)
{
	ASSERT(arch);
	ASSERT(dimX1);
	ASSERT(skip_fork);

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

	// only one skip connection is allowed
	if(skip_fork->skip)
	{
		LOGE("invalid");
		return NULL;
	}

	nn_layerInfo_t info =
	{
		.arch            = arch,
		.forward_pass_fn = nn_skipLayer_forwardPassAddFn,
		.backprop_fn     = nn_skipLayer_backpropAddFn,
		.dimX_fn         = nn_skipLayer_dimXFn,
		.dimY_fn         = nn_skipLayer_dimYFn,
	};

	nn_skipLayer_t* self;
	self = (nn_skipLayer_t*)
	       nn_layer_new(sizeof(nn_skipLayer_t), &info);
	if(self == NULL)
	{
		return NULL;
	}

	self->mode = NN_SKIP_LAYER_MODE_ADD;
	self->skip = skip_fork;

	nn_dim_copy(dimX1, &self->dimX);

	self->Y = nn_tensor_new(arch, dimX1,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_COMPUTE);
	if(self->Y == NULL)
	{
		goto fail_Y;
	}

	// dL_X1 and dL_X2 are set by backpropAddFn

	// connect skip
	skip_fork->skip = self;

	if(nn_skipLayer_newCompute(self) == 0)
	{
		goto fail_compute;
	}

	// success
	return self;

	// failure
	fail_compute:
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

	// check required dimensions
	// x1h==x2h, x1w==x2w
	nn_dim_t* dimX2 = nn_layer_dimY(&skip_fork->base);
	if((dimX1->count  != dimX2->count)  ||
	   (dimX1->height != dimX2->height) ||
	   (dimX1->width  != dimX2->width))
	{
		LOGE("invalid");
		return NULL;
	}

	// only one skip connection is allowed
	if(skip_fork->skip)
	{
		LOGE("invalid");
		return NULL;
	}

	nn_layerInfo_t info =
	{
		.arch            = arch,
		.forward_pass_fn = nn_skipLayer_forwardPassCatFn,
		.backprop_fn     = nn_skipLayer_backpropCatFn,
		.dimX_fn         = nn_skipLayer_dimXFn,
		.dimY_fn         = nn_skipLayer_dimYFn,
	};

	nn_skipLayer_t* self;
	self = (nn_skipLayer_t*)
	       nn_layer_new(sizeof(nn_skipLayer_t), &info);
	if(self == NULL)
	{
		return NULL;
	}

	self->mode = NN_SKIP_LAYER_MODE_CAT;
	self->skip = skip_fork;

	nn_dim_copy(dimX1, &self->dimX);

	nn_dim_t dimY =
	{
		.count  = dimX1->count,
		.height = dimX1->height,
		.width  = dimX1->width,
		.depth  = dimX1->depth + dimX2->depth,
	};

	self->Y = nn_tensor_new(arch, &dimY,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_COMPUTE);
	if(self->Y == NULL)
	{
		goto fail_Y;
	}

	self->dL_dX1 = nn_tensor_new(arch, dimX1,
	                             NN_TENSOR_INIT_ZERO,
	                             NN_TENSOR_MODE_COMPUTE);
	if(self->dL_dX1 == NULL)
	{
		goto fail_dL_dX1;
	}

	self->dL_dX2 = nn_tensor_new(arch, dimX2,
	                             NN_TENSOR_INIT_ZERO,
	                             NN_TENSOR_MODE_COMPUTE);
	if(self->dL_dX2 == NULL)
	{
		goto fail_dL_dX2;
	}

	// connect skip
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

	jsmn_val_t* val_dimX = NULL;
	jsmn_val_t* val_mode = NULL;

	cc_listIter_t* iter = cc_list_head(val->obj->list);
	while(iter)
	{
		jsmn_keyval_t* kv;
		kv = (jsmn_keyval_t*) cc_list_peekIter(iter);

		if(kv->val->type == JSMN_TYPE_STRING)
		{
			if(strcmp(kv->key, "mode") == 0)
			{
				val_mode = kv->val;
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
	if((val_dimX == NULL) ||
	   (val_mode == NULL))
	{
		LOGE("invalid");
		return NULL;
	}

	nn_dim_t dimX;
	if(nn_dim_load(&dimX, val_dimX) == 0)
	{
		return NULL;
	}

	if(strcmp(val_mode->data, "fork") == 0)
	{
		return nn_skipLayer_newFork(arch, &dimX);
	}
	else if(strcmp(val_mode->data, "add") == 0)
	{
		return nn_skipLayer_newAdd(arch, &dimX, skip_fork);
	}
	else if(strcmp(val_mode->data, "cat") == 0)
	{
		return nn_skipLayer_newCat(arch, &dimX, skip_fork);
	}
	else
	{
		LOGE("invalid mode=%s", val_mode->data);
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
	ret &= nn_dim_store(dimX, stream);
	ret &= jsmn_stream_key(stream, "%s", "mode");
	if(self->mode == NN_SKIP_LAYER_MODE_ADD)
	{
		ret &= jsmn_stream_string(stream, "%s", "add");
	}
	else if(self->mode == NN_SKIP_LAYER_MODE_CAT)
	{
		ret &= jsmn_stream_string(stream, "%s", "cat");
	}
	else
	{
		ret &= jsmn_stream_string(stream, "%s", "fork");
	}
	ret &= jsmn_stream_end(stream);

	return ret;
}

void nn_skipLayer_delete(nn_skipLayer_t** _self)
{
	ASSERT(_self);

	nn_skipLayer_t* self = *_self;
	if(self)
	{
		nn_skipLayer_deleteCompute(self);

		if(self->mode == NN_SKIP_LAYER_MODE_CAT)
		{
			nn_tensor_delete(&self->dL_dX2);
			nn_tensor_delete(&self->dL_dX1);
			nn_tensor_delete(&self->Y);
		}
		else if(self->mode == NN_SKIP_LAYER_MODE_ADD)
		{
			nn_tensor_delete(&self->Y);
		}
		nn_layer_delete((nn_layer_t**) _self);
	}
}
