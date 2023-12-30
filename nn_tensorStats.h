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

#ifndef nn_tensorStats_H
#define nn_tensorStats_H

#include "../libvkk/vkk.h"
#include "nn.h"

typedef struct
{
	uint32_t count;
	float    min;
	float    max;
	float    mean;
	float    stddev;
	float    norm;
} nn_tensorStatsData_t;

typedef struct nn_tensorStats_s
{
	nn_engine_t* engine;

	int dirty;

	nn_tensorStatsData_t data;

	vkk_uniformSet_t* us1;
	vkk_buffer_t*     sb10_stats;
} nn_tensorStats_t;

nn_tensorStats_t* nn_tensorStats_new(nn_engine_t* engine);
void              nn_tensorStats_delete(nn_tensorStats_t** _self);
float             nn_tensorStats_min(nn_tensorStats_t* self);
float             nn_tensorStats_max(nn_tensorStats_t* self);
float             nn_tensorStats_mean(nn_tensorStats_t* self);
float             nn_tensorStats_stddev(nn_tensorStats_t* self);
float             nn_tensorStats_norm(nn_tensorStats_t* self);

#endif
