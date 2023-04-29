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

#define LOG_TAG "nn"
#include "../libcc/cc_log.h"
#include "../libcc/cc_memory.h"
#include "nn_arch.h"
#include "nn_batchNormLayer.h"
#include "nn_layer.h"
#include "nn_tensor.h"

/***********************************************************
* private                                                  *
***********************************************************/

static nn_tensor_t*
nn_batchNormLayer_forwardPassFn(nn_layer_t* base,
                                nn_tensor_t* X)
{
	ASSERT(base);
	ASSERT(X);

	nn_batchNormLayer_t* self = (nn_batchNormLayer_t*) base;

	float epsilon = 0.0001f;

	nn_tensor_t* G           = self->G;
	nn_tensor_t* B           = self->B;
	nn_tensor_t* Y           = self->Y;
	nn_tensor_t* Xmean_mb    = self->Xmean_mb;
	nn_tensor_t* Xvar_mb     = self->Xvar_mb;
	nn_tensor_t* Xmean_ra    = self->Xmean_ra;
	nn_tensor_t* Xvar_ra     = self->Xvar_ra;
	nn_tensor_t* dXvar_dX    = self->dXvar_dX;
	nn_tensor_t* dXhat_dX    = self->dXhat_dX;
	nn_tensor_t* dXhat_dXvar = self->dXhat_dXvar;
	nn_tensor_t* dY_dG       = self->dY_dG;

	// prediction or training
	nn_tensor_t* Xmean = self->Xmean_ra;
	nn_tensor_t* Xvar  = self->Xvar_ra;
	uint32_t     bs    = base->arch->batch_size;
	if(bs > 1)
	{
		Xmean = self->Xmean_mb;
		Xvar  = self->Xvar_mb;
	}

	// compute mini-batch mean
	// update running mean
	float     xmean_ra;
	float     xmean_mb;
	float     momentum = base->arch->batch_momentum;
	uint32_t  m;
	uint32_t  i;
	uint32_t  j;
	uint32_t  k;
	nn_dim_t* dim = nn_tensor_dim(X);
	if(bs > 1)
	{
		for(k = 0; k < dim->depth; ++k)
		{
			// compute mini-batch mean
			xmean_mb = 0.0f;
			for(m = 0; m < bs; ++m)
			{
				for(i = 0; i < dim->height; ++i)
				{
					for(j = 0; j < dim->width; ++j)
					{
						xmean_mb += nn_tensor_get(X, m, i, j, k);
					}
				}
			}
			xmean_mb /= (float) (bs*dim->width*dim->height);
			nn_tensor_set(Xmean_mb, 0, 0, 0, k, xmean_mb);

			// update running mean
			xmean_ra = nn_tensor_get(Xmean_ra, 0, 0, 0, k);
			xmean_ra = momentum*xmean_ra + (1 - momentum)*xmean_mb;
			nn_tensor_set(Xmean_ra, 0, 0, 0, k, xmean_ra);
		}
	}

	// compute mini-batch variance
	// update running variance
	float xvar_ra;
	float xvar_mb;
	float dx;
	if(bs > 1)
	{
		for(k = 0; k < dim->depth; ++k)
		{
			// compute mini-batch variance
			xvar_mb  = 0.0f;
			xmean_mb = nn_tensor_get(Xmean_mb, 0, 0, 0, k);
			for(m = 0; m < bs; ++m)
			{
				for(i = 0; i < dim->height; ++i)
				{
					for(j = 0; j < dim->width; ++j)
					{
						dx       = nn_tensor_get(X, m, i, j, k) - xmean_mb;
						xvar_mb += dx*dx;
					}
				}
			}
			xvar_mb /= (float) (bs*dim->width*dim->height);
			nn_tensor_set(Xvar_mb, 0, 0, 0, k, xvar_mb);

			// update running variance
			xvar_ra = nn_tensor_get(Xvar_ra, 0, 0, 0, k);
			xvar_ra = momentum*xvar_ra + (1 - momentum)*xvar_mb;
			nn_tensor_set(Xvar_ra, 0, 0, 0, k, xvar_ra);
		}
	}

	// compute dXvar_dX
	float dxvar_dx;
	float xmean;
	float xvar;
	float x;
	for(k = 0; k < dim->depth; ++k)
	{
		dxvar_dx = 0.0f;
		xmean    = nn_tensor_get(Xmean, 0, 0, 0, k);
		for(m = 0; m < bs; ++m)
		{
			for(i = 0; i < dim->height; ++i)
			{
				for(j = 0; j < dim->width; ++j)
				{
					x         = nn_tensor_get(X, m, i, j, k);
					dxvar_dx += x - xmean;
				}
			}
		}
		dxvar_dx *= 2.0f/((float) (bs*dim->height*dim->width));
		nn_tensor_set(dXvar_dX, 0, 0, 0, k, dxvar_dx);
	}

	// compute dXhat_dX
	for(k = 0; k < dim->depth; ++k)
	{
		xvar = nn_tensor_get(Xvar, 0, 0, 0, k);
		nn_tensor_set(dXhat_dX, 0, 0, 0, k,
		              1.0f/(sqrtf(xvar) + epsilon));
	}

	// compute dXhat_dXvar
	float dxhat_dxvar;
	float s;
	for(k = 0; k < dim->depth; ++k)
	{
		xmean = nn_tensor_get(Xmean, 0, 0, 0, k);
		xvar  = nn_tensor_get(Xvar,  0, 0, 0, k);
		s     = (1.0f/((float) bs))*(-0.5f)*powf(xvar, -3.0f/2.0f);
		for(i = 0; i < dim->height; ++i)
		{
			for(j = 0; j < dim->width; ++j)
			{
				dxhat_dxvar = 0.0f;
				for(m = 0; m < bs; ++m)
				{
					x = nn_tensor_get(X, m, i, j, k);
					dxhat_dxvar += x - xmean;
				}
				nn_tensor_set(dXhat_dXvar, 0, i, j, k, s*dxhat_dxvar);
			}
		}
	}

	// compute Xhat (store in Y)
	nn_tensor_t* Xhat = Y;
	float xhat;
	for(k = 0; k < dim->depth; ++k)
	{
		xmean = nn_tensor_get(Xmean, 0, 0, 0, k);
		xvar  = nn_tensor_get(Xvar,  0, 0, 0, k);
		for(m = 0; m < bs; ++m)
		{
			for(i = 0; i < dim->height; ++i)
			{
				for(j = 0; j < dim->width; ++j)
				{
					x    = nn_tensor_get(X, m, i, j, k);
					xhat = (x - xmean)/(sqrtf(xvar) + epsilon);
					nn_tensor_set(Xhat, m, i, j, k, xhat);
				}
			}
		}
	}

	// compute dY_dG (use Xhat)
	float dy_dg;
	for(i = 0; i < dim->height; ++i)
	{
		for(j = 0; j < dim->width; ++j)
		{
			for(k = 0; k < dim->depth; ++k)
			{
				dy_dg = 0.0f;
				for(m = 0; m < bs; ++m)
				{
					xhat = nn_tensor_get(Xhat, m, i, j, k);
					dy_dg += xhat;
				}
				nn_tensor_set(dY_dG, 0, i, j, k, dy_dg/((float) bs));
			}
		}
	}

	// compute Y (replace Xhat)
	float y;
	float gamma;
	float beta;
	for(k = 0; k < dim->depth; ++k)
	{
		gamma = nn_tensor_get(G, 0, 0, 0, k);
		beta  = nn_tensor_get(B,  0, 0, 0, k);
		for(m = 0; m < bs; ++m)
		{
			for(i = 0; i < dim->height; ++i)
			{
				for(j = 0; j < dim->width; ++j)
				{
					xhat = nn_tensor_get(Xhat, m, i, j, k);
					y    = gamma*xhat + beta;
					nn_tensor_set(Y, m, i, j, k, y);
				}
			}
		}
	}

	return Y;
}

static nn_tensor_t*
nn_batchNormLayer_backpropFn(nn_layer_t* base,
                             nn_tensor_t* dL_dY)
{
	ASSERT(base);
	ASSERT(dL_dY); // dim(1,X.w,X.h,X.d)

	nn_batchNormLayer_t* self = (nn_batchNormLayer_t*) base;
	nn_arch_t*           arch = base->arch;

	nn_tensor_t* G           = self->G;
	nn_tensor_t* B           = self->B;
	nn_tensor_t* dXvar_dX    = self->dXvar_dX;
	nn_tensor_t* dXhat_dX    = self->dXhat_dX;
	nn_tensor_t* dXhat_dXvar = self->dXhat_dXvar;
	nn_tensor_t* dY_dG       = self->dY_dG;
	nn_tensor_t* dL_dX       = self->dL_dX;
	nn_tensor_t* dL_dXvar    = self->dL_dXvar;
	nn_tensor_t* dL_dXmean   = self->dL_dXmean;
	nn_dim_t*    dim         = nn_tensor_dim(dL_dY);
	float        lr          = arch->learning_rate;

	// compute dL_dXhat (store in dL_dX)
	nn_tensor_t* dL_dXhat = dL_dX;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	float    gamma;
	float    dl_dy;
	for(k = 0; k < dim->depth; ++k)
	{
		gamma = nn_tensor_get(G, 0, 0, 0, k);
		for(i = 0; i < dim->height; ++i)
		{
			for(j = 0; j < dim->width; ++j)
			{
				dl_dy = nn_tensor_get(dL_dY, 0, i, j, k);
				nn_tensor_set(dL_dXhat, 0, i, j, k, dl_dy*gamma);
			}
		}
	}

	// compute dL_dG and update G
	float dy_dg;
	float dl_dg;
	for(k = 0; k < dim->depth; ++k)
	{
		dl_dg = 0.0f;
		for(i = 0; i < dim->height; ++i)
		{
			for(j = 0; j < dim->width; ++j)
			{
				dl_dy  = nn_tensor_get(dL_dY, 0, i, j, k);
				dy_dg  = nn_tensor_get(dY_dG, 0, i, j, k);
				dl_dg += dl_dy*dy_dg;
			}
		}
		nn_tensor_add(G, 0, 0, 0, k, -lr*dl_dg);
	}

	// compute dL_dB and update B
	float dl_db;
	for(k = 0; k < dim->depth; ++k)
	{
		dl_db = 0.0f;
		for(i = 0; i < dim->height; ++i)
		{
			for(j = 0; j < dim->width; ++j)
			{
				dl_dy = nn_tensor_get(dL_dY, 0, i, j, k);
				dl_db += dl_dy;
			}
		}
		nn_tensor_add(B, 0, 0, 0, k, -lr*dl_db);
	}

	// compute dL_dXvar (use dL_dXhat)
	float dl_dxvar;
	float dl_dxhat;
	float dxhat_dxvar;
	for(k = 0; k < dim->depth; ++k)
	{
		dl_dxvar = 0.0f;
		for(i = 0; i < dim->height; ++i)
		{
			for(j = 0; j < dim->width; ++j)
			{
				dl_dxhat     = nn_tensor_get(dL_dXhat, 0, i, j, k);
				dxhat_dxvar  = nn_tensor_get(dXhat_dXvar, 0, i, j, k);
				dl_dxvar    += dl_dxhat*dxhat_dxvar;
			}
		}
		nn_tensor_set(dL_dXvar, 0, 0, 0, k, dl_dxvar);
	}

	// compute dL2_dXmean (use dL_dXhat, store in dL_dXmean)
	float dl_dxmean;
	float dxhat_dx;
	for(k = 0; k < dim->depth; ++k)
	{
		dl_dxmean = 0.0f;
		dxhat_dx  = nn_tensor_get(dXhat_dX, 0, 0, 0, k);
		for(i = 0; i < dim->height; ++i)
		{
			for(j = 0; j < dim->width; ++j)
			{
				dl_dxhat   = nn_tensor_get(dL_dXhat, 0, i, j, k);
				dl_dxmean += dl_dxhat;
			}
		}
		dl_dxmean *= -dxhat_dx;

		// dL_dXmean = dL2_dXmean
		nn_tensor_set(dL_dXmean, 0, 0, 0, k, dl_dxmean);
	}

	// compute dL1_dX (use/replace dL_dXhat, store in dL_dX)
	nn_tensor_t* dL1_dX = dL_dX;
	for(k = 0; k < dim->depth; ++k)
	{
		dxhat_dx = nn_tensor_get(dXhat_dX, 0, 0, 0, k);
		for(i = 0; i < dim->height; ++i)
		{
			for(j = 0; j < dim->width; ++j)
			{
				// dL_dX = dL1_dX = dL_dXhat*dXhat_dX
				nn_tensor_mul(dL1_dX, 0, i, j, k, dxhat_dx);
			}
		}
	}

	// compute dL1_dXmean and combine with dL_dXmean
	float dxvar_dx;
	for(k = 0; k < dim->depth; ++k)
	{
		dl_dxvar = nn_tensor_get(dL_dXvar, 0, 0, 0, k);
		dxvar_dx = nn_tensor_get(dXvar_dX, 0, 0, 0, k);

		// dL_dXmean = dL2_dXmean + dL1_dXmean
		nn_tensor_add(dL_dXmean, 0, 0, 0, k, -dl_dxvar*dxvar_dx);
	}

	// compute dL2_dX and combine with dL_dX
	float dl2_dx;
	for(k = 0; k < dim->depth; ++k)
	{
		dl_dxvar = nn_tensor_get(dL_dXvar, 0, 0, 0, k);
		dxvar_dx = nn_tensor_get(dXvar_dX, 0, 0, 0, k);
		dl2_dx   = dl_dxvar*dxvar_dx;

		for(i = 0; i < dim->height; ++i)
		{
			for(j = 0; j < dim->width; ++j)
			{
				// dL_dX = dL1_dX + dL2_dX
				nn_tensor_add(dL_dX, 0, i, j, k, dl2_dx);
			}
		}
	}

	// compute dL3_dX and combine with dL_dX
	float dl3_dx;
	float dxmean_dx = 1.0f/((float) (dim->height*dim->width));
	for(k = 0; k < dim->depth; ++k)
	{
		dl_dxmean = nn_tensor_get(dL_dXmean, 0, 0, 0, k);
		dl3_dx    = dl_dxmean*dxmean_dx;

		for(i = 0; i < dim->height; ++i)
		{
			for(j = 0; j < dim->width; ++j)
			{
				// dL_dX = dL1_dX + dL2_dX + dL3_dX
				nn_tensor_add(dL_dX, 0, i, j, k, dl3_dx);
			}
		}
	}

	return dL_dX;
}

static nn_dim_t*
nn_batchNormLayer_dimFn(nn_layer_t* base)
{
	ASSERT(base);

	nn_batchNormLayer_t* self = (nn_batchNormLayer_t*) base;

	return nn_tensor_dim(self->Y);
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_batchNormLayer_t*
nn_batchNormLayer_new(nn_arch_t* arch, nn_dim_t* dim)
{
	ASSERT(arch);
	ASSERT(dim);

	nn_dim_t dim_111d =
	{
		.count  = 1,
		.height = 1,
		.width  = 1,
		.depth  = dim->depth,
	};

	nn_dim_t dim_1hwd =
	{
		.count  = 1,
		.height = dim->height,
		.width  = dim->width,
		.depth  = dim->depth,
	};

	nn_layerInfo_t info =
	{
		.arch            = arch,
		.forward_pass_fn = nn_batchNormLayer_forwardPassFn,
		.backprop_fn     = nn_batchNormLayer_backpropFn,
		.dim_fn          = nn_batchNormLayer_dimFn,
	};

	nn_batchNormLayer_t* self;
	self = (nn_batchNormLayer_t*)
	       nn_layer_new(sizeof(nn_batchNormLayer_t), &info);
	if(self == NULL)
	{
		return NULL;
	}

	self->G = nn_tensor_new(&dim_111d);
	if(self->G == NULL)
	{
		goto fail_G;
	}

	// initialize G to 1.0f
	uint32_t k;
	for(k = 0; k < dim->depth; ++k)
	{
		nn_tensor_set(self->G, 0, 0, 0, k, 1.0f);
	}

	self->B = nn_tensor_new(&dim_111d);
	if(self->B == NULL)
	{
		goto fail_B;
	}

	self->Y = nn_tensor_new(dim);
	if(self->Y == NULL)
	{
		goto fail_Y;
	}

	self->Xmean_mb = nn_tensor_new(&dim_111d);
	if(self->Xmean_mb == NULL)
	{
		goto fail_Xmean_mb;
	}

	self->Xvar_mb = nn_tensor_new(&dim_111d);
	if(self->Xvar_mb == NULL)
	{
		goto fail_Xvar_mb;
	}

	self->Xmean_ra = nn_tensor_new(&dim_111d);
	if(self->Xmean_ra == NULL)
	{
		goto fail_Xmean_ra;
	}

	self->Xvar_ra = nn_tensor_new(&dim_111d);
	if(self->Xvar_ra == NULL)
	{
		goto fail_Xvar_ra;
	}

	self->dXvar_dX = nn_tensor_new(&dim_111d);
	if(self->dXvar_dX == NULL)
	{
		goto fail_dXvar_dX;
	}

	self->dXhat_dX = nn_tensor_new(&dim_111d);
	if(self->dXhat_dX == NULL)
	{
		goto fail_dXhat_dX;
	}

	self->dXhat_dXvar = nn_tensor_new(&dim_1hwd);
	if(self->dXhat_dXvar == NULL)
	{
		goto fail_dXhat_dXvar;
	}

	self->dY_dG = nn_tensor_new(&dim_1hwd);
	if(self->dY_dG == NULL)
	{
		goto fail_dY_dG;
	}

	self->dL_dX = nn_tensor_new(&dim_1hwd);
	if(self->dL_dX == NULL)
	{
		goto fail_dL_dX;
	}

	self->dL_dXvar = nn_tensor_new(&dim_111d);
	if(self->dL_dXvar == NULL)
	{
		goto fail_dL_dXvar;
	}

	self->dL_dXmean = nn_tensor_new(&dim_111d);
	if(self->dL_dXmean == NULL)
	{
		goto fail_dL_dXmean;
	}

	// success
	return self;

	// failure
	fail_dL_dXmean:
		nn_tensor_delete(&self->dL_dXvar);
	fail_dL_dXvar:
		nn_tensor_delete(&self->dL_dX);
	fail_dL_dX:
		nn_tensor_delete(&self->dY_dG);
	fail_dY_dG:
		nn_tensor_delete(&self->dXhat_dXvar);
	fail_dXhat_dXvar:
		nn_tensor_delete(&self->dXhat_dX);
	fail_dXhat_dX:
		nn_tensor_delete(&self->dXvar_dX);
	fail_dXvar_dX:
		nn_tensor_delete(&self->Xvar_ra);
	fail_Xvar_ra:
		nn_tensor_delete(&self->Xmean_ra);
	fail_Xmean_ra:
		nn_tensor_delete(&self->Xvar_mb);
	fail_Xvar_mb:
		nn_tensor_delete(&self->Xmean_mb);
	fail_Xmean_mb:
		nn_tensor_delete(&self->Y);
	fail_Y:
		nn_tensor_delete(&self->B);
	fail_B:
		nn_tensor_delete(&self->G);
	fail_G:
		nn_layer_delete((nn_layer_t**) &self);
	return NULL;
}

void nn_batchNormLayer_delete(nn_batchNormLayer_t** _self)
{
	ASSERT(_self);

	nn_batchNormLayer_t* self = *_self;
	if(self)
	{
		nn_tensor_delete(&self->dL_dXmean);
		nn_tensor_delete(&self->dL_dXvar);
		nn_tensor_delete(&self->dL_dX);
		nn_tensor_delete(&self->dY_dG);
		nn_tensor_delete(&self->dXhat_dXvar);
		nn_tensor_delete(&self->dXhat_dX);
		nn_tensor_delete(&self->dXvar_dX);
		nn_tensor_delete(&self->Xvar_ra);
		nn_tensor_delete(&self->Xmean_ra);
		nn_tensor_delete(&self->Xvar_mb);
		nn_tensor_delete(&self->Xmean_mb);
		nn_tensor_delete(&self->Y);
		nn_tensor_delete(&self->B);
		nn_tensor_delete(&self->G);
		nn_layer_delete((nn_layer_t**) &self);
	}
}
