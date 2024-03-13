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

#ifndef nn_batchNormLayer_H
#define nn_batchNormLayer_H

#include "../jsmn/wrapper/jsmn_stream.h"
#include "../jsmn/wrapper/jsmn_wrapper.h"
#include "../libvkk/vkk.h"
#include "nn_layer.h"

// see nn_coderBatchNormMode_e
typedef enum
{
	NN_BATCH_NORM_MODE_RUNNING  = 1,
	NN_BATCH_NORM_MODE_INSTANCE = 2,
} nn_batchNormMode_e;

typedef struct nn_batchNormUs2Key_s
{
	uint32_t k;
} nn_batchNormUs2Key_t;

typedef struct nn_batchNormUs2Data_s
{
	vkk_buffer_t*     sb200;
	vkk_uniformSet_t* us2;
} nn_batchNormUs2Data_t;

nn_batchNormUs2Data_t* nn_batchNormUs2Data_new(nn_engine_t* engine,
                                               nn_batchNormUs2Key_t* key);
void                   nn_batchNormUs2Data_delete(nn_batchNormUs2Data_t** _self);

typedef struct nn_batchNormLayer_s
{
	nn_layer_t base;

	nn_batchNormMode_e bn_mode;

	// gamma, beta, xhat, output
	nn_tensor_t* G;    // dim(1,1,1,xd)
	nn_tensor_t* B;    // dim(1,1,1,xd)
	nn_tensor_t* Xhat; // dim(bs,xh,xw,xd)
	nn_tensor_t* Y;    // dim(bs,xh,xw,xd)

	// Adam moment estimates
	nn_tensor_t* MG; // dim(1,1,1,xd)
	nn_tensor_t* VG; // dim(1,1,1,xd)
	nn_tensor_t* MB; // dim(1,1,1,xd)
	nn_tensor_t* VB; // dim(1,1,1,xd)

	// mini-batch mean/variance
	nn_tensor_t* Xmean_mb; // dim(1,1,1,xd)
	nn_tensor_t* Xvar_mb;  // dim(1,1,1,xd)

	// running averages
	nn_tensor_t* Xmean_ra; // dim(1,1,1,xd)
	nn_tensor_t* Xvar_ra;  // dim(1,1,1,xd)

	// backprop gradients (dL_dY replaced by dL_dX)
	//           dL_dY;    // dim(bs,xh,xw,xd)
	//           dL_dX;    // dim(bs,xh,xw,xd)
	nn_tensor_t* dL_dXhat; // dim(bs,xh,xw,xd)

	// working sums
	nn_tensor_t* Bsum; // dim(1,1,1,xd)
	nn_tensor_t* Csum; // dim(1,1,1,xd)

	vkk_uniformSet_t* us0;
	vkk_uniformSet_t* us1_fp;
	vkk_uniformSet_t* us1_bp;
} nn_batchNormLayer_t;

nn_batchNormLayer_t* nn_batchNormLayer_new(nn_arch_t* arch,
                                           nn_batchNormMode_e bn_mode,
                                           nn_dim_t* dimX);
void                 nn_batchNormLayer_delete(nn_batchNormLayer_t** _self);
nn_batchNormLayer_t* nn_batchNormLayer_import(nn_arch_t* arch,
                                              jsmn_val_t* val);
int                  nn_batchNormLayer_export(nn_batchNormLayer_t* self,
                                              jsmn_stream_t* stream);

#endif
