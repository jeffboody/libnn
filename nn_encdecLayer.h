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

#ifndef nn_encdecLayer_H
#define nn_encdecLayer_H

#include "../libcc/jsmn/cc_jsmnStream.h"
#include "../libcc/jsmn/cc_jsmnWrapper.h"
#include "nn_coderLayer.h"
#include "nn_dim.h"
#include "nn_factLayer.h"
#include "nn_layer.h"

typedef enum
{
	NN_ENCDEC_SAMPLER_CODER   = 0,
	NN_ENCDEC_SAMPLER_LANCZOS = 1,
} nn_encdecSampler_e;

// downsampling and upsampling
// coder: strided and transpose convolutions
// lanczos: 1/2x or 2x resampling
typedef union
{
	nn_layer_t*        base;
	nn_coderLayer_t*   coder;
	nn_lanczosLayer_t* lanczos;
} nn_encdecSampler_t;

typedef struct nn_encdecLayerInfo_s
{
	nn_arch_t* arch;

	nn_encdecSampler_e sampler;

	nn_dim_t* dimX;
	uint32_t  fc;

	// conv layer
	// flags for SN and BSSN
	int norm_flags0;
	int norm_flags12;

	// skip layer
	// ADD: Residual Network
	// CAT: U-Net
	nn_coderSkipMode_e skip_mode;
	float              skip_beta;

	// bn layer
	nn_coderBatchNormMode_e bn_mode0;
	nn_coderBatchNormMode_e bn_mode12;

	// fact layer
	nn_factLayerFn_e fact_fn;

	// lanczos layer (optional)
	int a;
} nn_encdecLayerInfo_t;

typedef struct nn_encdecLayer_s
{
	nn_layer_t base;

	nn_encdecSampler_e sampler;

	// * first digit represents the level or scale
	// * skip connections from encoder to decoder
	//   enc0 -> dec0
	//   enc1 -> dec1
	nn_coderLayer_t*   enc0;
	nn_encdecSampler_t down1;
	nn_coderLayer_t*   enc1;
	nn_encdecSampler_t down2;
	nn_coderLayer_t*   node20;
	nn_coderLayer_t*   node21;
	nn_coderLayer_t*   node22;
	nn_coderLayer_t*   node23;
	nn_encdecSampler_t up1;
	nn_coderLayer_t*   dec1;
	nn_encdecSampler_t up0;
	nn_coderLayer_t*   dec0;
} nn_encdecLayer_t;

nn_encdecLayer_t* nn_encdecLayer_new(nn_encdecLayerInfo_t* info);
void              nn_encdecLayer_delete(nn_encdecLayer_t** _self);
nn_encdecLayer_t* nn_encdecLayer_import(nn_arch_t* arch,
                                        cc_jsmnVal_t* val);
int               nn_encdecLayer_export(nn_encdecLayer_t* self,
                                        cc_jsmnStream_t* stream);

#endif
