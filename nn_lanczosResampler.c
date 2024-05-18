/*
 * Copyright (c) 2024 Jeff Boody
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
#include "../libcc/math/cc_pow2n.h"
#include "../libcc/cc_log.h"
#include "../libcc/cc_memory.h"
#include "nn_dim.h"
#include "nn_lanczosResampler.h"
#include "nn_tensor.h"

/***********************************************************
* private                                                  *
***********************************************************/

static float nn_lanczosResampler_sinc(float x)
{
	if(x == 0.0f)
	{
		return 1.0f;
	}

	return sin(M_PI*x)/(M_PI*x);
}

static float nn_lanczosResampler_L(float x, float a)
{
	if((-a <= x) && (x < a))
	{
		return nn_lanczosResampler_sinc(x)*
		       nn_lanczosResampler_sinc(x/a);
	}

	return 0.0;
}

static nn_tensor_t*
nn_lanczosResampler_newL(nn_lanczosResampler_t* self,
                         nn_engine_t* engine,
                         int fs, int fc, int sz,
                         uint32_t n1, uint32_t n2)
{
	ASSERT(self);
	ASSERT(engine);

	nn_dim_t dimL =
	{
		.count  = fc,
		.height = 1,
		.width  = 1,
		.depth  = sz,
	};

	nn_tensor_t* L;
	L = nn_tensor_new(engine, &dimL, NN_TENSOR_INIT_ZERO,
	                  NN_TENSOR_MODE_IO);
	if(L == NULL)
	{
		return NULL;
	}

	// compute L premultiplied by 1/w
	int      i;
	int      j;
	int      a = self->param.a;
	float    l;
	float    w;
	float    x;
	float    step = ((float) n1)/((float) n2);
	float    fsf  = (float) fs;
	uint32_t n;
	for(j = 0; j < fc; ++j)
	{
		n = 0;
		w = 0.0f;
		x = (((float) j) + 0.5f)*step - 0.5f;
		for(i = -(fs*a) + 1; i <= (fs*a); ++i)
		{
			w += nn_lanczosResampler_L((i - x + floorf(x))/fsf, a);
		}
		for(i = -(fs*a) + 1; i <= (fs*a); ++i)
		{
			l = nn_lanczosResampler_L((i - x + floorf(x))/fsf, a);
			nn_tensor_ioSet(L, j, 0, 0, n, (1.0f/w)*l);
			++n;
		}
	}

	return L;
}

static int
nn_lanczosResampler_validate(uint32_t x, uint32_t y)
{
	// swap order if x > y
	if(x > y)
	{
		return nn_lanczosResampler_validate(y, x);
	}

	// y must be x*2^n
	uint32_t x2 = x;
	while(x2 <= y)
	{
		if(x2 == y)
		{
			return 1;
		}

		x2 *= 2;
	}

	return 0;
}

static float
nn_lanczosResampler_getLw(nn_lanczosResampler_t* self,
                          uint32_t j, uint32_t n)
{
	ASSERT(self);

	uint32_t fcw  = self->param.fcw;
	uint32_t szw  = self->param.szw;
	float*   data = self->Lw->data;

	// dim(fcw,1,1,szw)
	return data[(j%fcw)*szw + n];
}

static float
nn_lanczosResampler_getLh(nn_lanczosResampler_t* self,
                          uint32_t i, uint32_t n)
{
	ASSERT(self);

	uint32_t fch  = self->param.fch;
	uint32_t szh  = self->param.szh;
	float*   data = self->Lh->data;

	// dim(fch,1,1,szh)
	return data[(i%fch)*szh + n];
}

static void
nn_lanczosResampler_computeT(nn_lanczosResampler_t* self,
                             nn_tensor_t* X,
                             uint32_t m, uint32_t i,
                             uint32_t j, uint32_t k)
{
	ASSERT(self);
	ASSERT(X);

	nn_dim_t* dimX = &self->dimX;
	nn_dim_t* dimY = &self->dimY;

	int   jj;
	int   lj;
	int   a  = self->param.a;
	int   fs = self->param.fsw;
	int   n  = 0;
	int   xw = (int) dimX->width;
	int   yw = (int) dimY->width;
	float lw;
	float s1;
	float s2   = 0.0f;
	float step = ((float) xw)/((float) yw);
	float x    = (((float) j) + 0.5f)*step - 0.5f;
	for(lj = -(fs*a) + 1; lj <= (fs*a); ++lj)
	{
		jj = ((int) (floorf(x))) + lj;
		if(jj < 0)
		{
			jj = 0;
		}
		else if(jj >= xw)
		{
			jj = xw - 1;
		}

		s1  = nn_tensor_ioGet(X, m, i, jj, k);
		lw  = nn_lanczosResampler_getLw(self, j, n);
		s2 += s1*lw;

		++n;
	}
	nn_tensor_ioSet(self->T, m, i, j, k, s2);
}

static void
nn_lanczosResampler_computeY(nn_lanczosResampler_t* self,
                             nn_tensor_t* Y,
                             uint32_t m, uint32_t i,
                             uint32_t j, uint32_t k)
{
	ASSERT(self);
	ASSERT(Y);

	nn_dim_t* dimX = &self->dimX;
	nn_dim_t* dimY = &self->dimY;

	int   ii;
	int   li;
	int   a  = self->param.a;
	int   fs = self->param.fsh;
	int   n  = 0;
	int   xh = (int) dimX->height;
	int   yh = (int) dimY->height;
	float lh;
	float s1;
	float s2   = 0.0;
	float step = ((float) xh)/((float) yh);
	float y    = (((float) i) + 0.5f)*step - 0.5f;
	for(li = -(fs*a) + 1; li <= (fs*a); ++li)
	{
		ii = ((int) (floorf(y))) + li;
		if(ii < 0)
		{
			ii = 0;
		}
		else if(ii >= xh)
		{
			ii = xh - 1;
		}

		s1  = nn_tensor_ioGet(self->T, m, ii, j, k);
		lh  = nn_lanczosResampler_getLh(self, i, n);
		s2 += s1*lh;

		++n;
	}
	nn_tensor_ioSet(Y, m, i, j, k, s2);
}

/***********************************************************
* public                                                   *
***********************************************************/

void nn_lanczosParam_copy(nn_lanczosParam_t* src,
                          nn_lanczosParam_t* dst)
{
	ASSERT(src);
	ASSERT(dst);

	memcpy(dst, src, sizeof(nn_lanczosParam_t));
}

nn_lanczosResampler_t*
nn_lanczosResampler_new(nn_engine_t* engine,
                        nn_dim_t* dimX, nn_dim_t* dimY,
                        int a)
{
	ASSERT(engine);
	ASSERT(dimX);
	ASSERT(dimY);

	uint32_t xn = dimX->count;
	uint32_t xh = dimX->height;
	uint32_t xw = dimX->width;
	uint32_t xd = dimX->depth;
	uint32_t yn = dimY->count;
	uint32_t yh = dimY->height;
	uint32_t yw = dimY->width;
	uint32_t yd = dimY->depth;

	// validate a, dimX and dimY
	if((a < 1) || (xn != yn) || (xd != yd) ||
	   (nn_lanczosResampler_validate(xh, yh) == 0) ||
	   (nn_lanczosResampler_validate(xw, yw) == 0))
	{
		LOGE("invalid a=%i, dimX=%u,%u,%u,%u, dimY=%u,%u,%u,%u",
		     a, xn, xh, xw, xd, yn, yh, yw, yd);
		return NULL;
	}

	nn_dim_t dimT =
	{
		.count  = xn,
		.height = xh,
		.width  = yw,
		.depth  = xd,
	};

	nn_lanczosResampler_t* self;
	self = (nn_lanczosResampler_t*)
	       CALLOC(1, sizeof(nn_lanczosResampler_t));
	if(self == NULL)
	{
		LOGE("CALLOC failed");
		return NULL;
	}

	nn_lanczosParam_t* param = &self->param;

	// support size
	param->a = a;

	// filter scale
	param->fsw = xw/yw;
	param->fsh = xh/yh;
	if(param->fsw < 1)
	{
		param->fsw = 1;
	}
	if(param->fsh < 1)
	{
		param->fsh = 1;
	}

	// filter count
	param->fcw = yw/xw;
	param->fch = yh/xh;
	if(param->fcw < 1)
	{
		param->fcw = 1;
	}
	if(param->fch < 1)
	{
		param->fch = 1;
	}

	// filter size
	param->szw = 2*param->fsw*param->a;
	param->szh = 2*param->fsh*param->a;

	nn_dim_copy(dimX, &self->dimX);
	nn_dim_copy(dimY, &self->dimY);

	self->T = nn_tensor_new(engine, &dimT,
	                        NN_TENSOR_INIT_ZERO,
	                        NN_TENSOR_MODE_IO);
	if(self->T == NULL)
	{
		goto failure;
	}

	self->Lw = nn_lanczosResampler_newL(self, engine,
	                                    param->fsw, param->fcw,
	                                    param->szw, xw, yw);
	if(self->Lw == NULL)
	{
		goto failure;
	}

	self->Lh = nn_lanczosResampler_newL(self, engine,
	                                    param->fsh, param->fch,
	                                    param->szh, xh, yh);
	if(self->Lh == NULL)
	{
		goto failure;
	}

	// success
	return self;

	// failure
	failure:
		nn_lanczosResampler_delete(&self);
	return NULL;
}

void
nn_lanczosResampler_delete(nn_lanczosResampler_t** _self)
{
	ASSERT(_self);

	nn_lanczosResampler_t* self = *_self;
	if(self)
	{
		nn_tensor_delete(&self->Lh);
		nn_tensor_delete(&self->Lw);
		nn_tensor_delete(&self->T);
		FREE(self);
		*_self = NULL;
	}
}

int
nn_lanczosResampler_resample(nn_lanczosResampler_t* self,
                             nn_tensor_t* X, nn_tensor_t* Y,
                             uint32_t bs)
{
	ASSERT(self);
	ASSERT(X);
	ASSERT(Y);

	nn_dim_t* dimX = nn_tensor_dim(X);
	if(nn_dim_sizeEquals(&self->dimX, dimX) == 0)
	{
		LOGE("invalid dimX: count=%u:%u, height=%u:%u, width=%u:%u, depth=%u:%u",
		     self->dimX.count,  dimX->count,
		     self->dimX.height, dimX->height,
		     self->dimX.width,  dimX->width,
		     self->dimX.depth,  dimX->depth);
		return 0;
	}

	nn_dim_t* dimY = nn_tensor_dim(Y);
	if(nn_dim_sizeEquals(&self->dimY, dimY) == 0)
	{
		LOGE("invalid dimY: count=%u:%u, height=%u:%u, width=%u:%u, depth=%u:%u",
		     self->dimY.count,  dimY->count,
		     self->dimY.height, dimY->height,
		     self->dimY.width,  dimY->width,
		     self->dimY.depth,  dimY->depth);
		return 0;
	}

	if(self->dimX.count < bs)
	{
		LOGE("invalid count=%u, bs=%u",
		     self->dimX.count, bs);
		return 0;
	}

	if((X->mode != NN_TENSOR_MODE_IO) ||
	   (Y->mode != NN_TENSOR_MODE_IO))
	{
		LOGE("invalid mode=%u:%u", X->mode, Y->mode);
		return 0;
	}

	nn_dim_t* dimT = nn_tensor_dim(self->T);

	// CPU implementation of computeT is
	// equivalent to nn_lanczosLayer_forwardPassT.comp
	uint32_t m;
	uint32_t i;
	uint32_t j;
	uint32_t k;
	for(m = 0; m < bs; ++m)
	{
		for(i = 0; i < dimT->height; ++i)
		{
			for(j = 0; j < dimT->width; ++j)
			{
				for(k = 0; k < dimT->depth; ++k)
				{
					nn_lanczosResampler_computeT(self, X,
					                             m, i, j, k);
				}
			}
		}
	}

	// CPU implementation of computeY is
	// equivalent to nn_lanczosLayer_forwardPassY.comp
	for(m = 0; m < bs; ++m)
	{
		for(i = 0; i < dimY->height; ++i)
		{
			for(j = 0; j < dimY->width; ++j)
			{
				for(k = 0; k < dimY->depth; ++k)
				{
					nn_lanczosResampler_computeY(self, Y,
					                             m, i, j, k);
				}
			}
		}
	}

	return 1;
}
