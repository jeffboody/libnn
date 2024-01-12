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

#ifndef nn_skipLayer_H
#define nn_skipLayer_H

#include "../libvkk/vkk.h"
#include "nn_dim.h"
#include "nn_layer.h"

// see nn_coderSkipMode_e
typedef enum
{
	NN_SKIP_MODE_FORK_ADD = 1,
	NN_SKIP_MODE_FORK_CAT = 2,
	NN_SKIP_MODE_ADD      = 3,
	NN_SKIP_MODE_CAT      = 4,
} nn_skipMode_e;

typedef struct nn_skipLayer_s
{
	nn_layer_t base;

	// skip_beta is residule scaling for add/cat modes
	// set to 1.0f to disable
	nn_skipMode_e skip_mode;
	float         skip_beta;

	// fork/add/cat layer
	nn_skipLayer_t* skip; // reference

	// fork    : dim(bs,xh,xw,xd)
	// add/cat : dim(bs,x1h,x1w,x1d)
	nn_dim_t dimX;

	// output
	// Yfork : X (reference)
	// Yadd  : dim(bs,xh,xw,xd)
	//         x1h==x2h, x1w==x2w, x1d==x2d
	// Ycat  : dim(bs,xh,xw,x1d + x2d)
	//         x1h==x2h, x1w==x2w
	nn_tensor_t* Y;

	// forward gradients are always 1

	// backprop gradients
	// fork:
	//   dL_dY replaced by dL_dY1 + dL_dY2
	//   dL_dY  : dim(bs,xh,xw,xd)
	//   dL_dY1 : dim(bs,xh,xw,xd)
	//   dL_dY2 : dim(bs,xh,xw,xd)
	//   dL_dX1 : NULL
	//   dL_dX2 : NULL
	// add (skip_beta == 1.0):
	//   dL_dY passed through
	//   dL_dY  : dim(bs,xh,xw,xd)
	//            x1h==x2h, x1w==x2w, x1d==x2d
	//   dL_dX1 : NULL
	//   dL_dX2 : dL_dY (reference)
	// add (skip_beta != 1.0):
	//   dL_dY passed through
	//   dL_dY  : dim(bs,xh,xw,xd)
	//            x1h==x2h, x1w==x2w, x1d==x2d
	//   dL_dX1 : dim(bs,xh,xw,xd)
	//   dL_dX2 : dL_dY (reference)
	// cat:
	//   dL_dY  : dim(bs,xh,xw,x1d + x2d)
	//            x1h==x2h, x1w==x2w
	//   dL_dX1 : dim(bs,xh,xw,x1d)
	//   dL_dX2 : dim(bs,xh,xw,x2d)
	nn_tensor_t* dL_dX1;
	nn_tensor_t* dL_dX2;

	vkk_uniformSet_t* us0;
	vkk_uniformSet_t* us1;
	vkk_uniformSet_t* us2;
	vkk_buffer_t*     sb01_param;
} nn_skipLayer_t;

nn_skipLayer_t* nn_skipLayer_newFork(nn_arch_t* arch,
                                     nn_dim_t* dimX,
                                     nn_skipMode_e skip_mode);
nn_skipLayer_t* nn_skipLayer_newAdd(nn_arch_t* arch,
                                    nn_dim_t* dimX,
                                    nn_skipLayer_t* skip_fork,
                                    float skip_beta);
nn_skipLayer_t* nn_skipLayer_newCat(nn_arch_t* arch,
                                    nn_dim_t* dimX,
                                    nn_skipLayer_t* skip_fork,
                                    float skip_beta);
nn_skipLayer_t* nn_skipLayer_import(nn_arch_t* arch,
                                    jsmn_val_t* val,
                                    nn_skipLayer_t* skip_fork);
int             nn_skipLayer_export(nn_skipLayer_t* self,
                                    jsmn_stream_t* stream);
void            nn_skipLayer_delete(nn_skipLayer_t** _self);

#endif
