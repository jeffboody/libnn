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

#include <stdlib.h>

#define LOG_TAG "nn"
#include "../libcc/cc_log.h"
#include "../libcc/cc_memory.h"
#include "nn_engine.h"
#include "nn_tensorStats.h"

/***********************************************************
* private                                                  *
***********************************************************/

static void nn_tensorStats_update(nn_tensorStats_t* self)
{
	ASSERT(self);

	if(self->dirty)
	{
		vkk_buffer_readStorage(self->sb10_stats, 0,
		                       sizeof(nn_tensorStatsData_t),
		                       &self->data);
		self->dirty = 0;
	}
}

/***********************************************************
* public                                                   *
***********************************************************/

nn_tensorStats_t* nn_tensorStats_new(nn_engine_t* engine)
{
	ASSERT(engine);

	nn_tensorStats_t* self;
	self = (nn_tensorStats_t*)
	       CALLOC(1, sizeof(nn_tensorStats_t));
	if(self == NULL)
	{
		LOGE("CALLOC failed");
		return NULL;
	}

	self->engine = engine;

	vkk_updateMode_e um;
	um = vkk_compute_updateMode(engine->compute);

	self->us1 = vkk_uniformSet_new(engine->engine,
	                               1, 0, NULL,
	                               engine->usf1_tensor_stats);
	if(self->us1 == NULL)
	{
		goto fail_us1;
	}

	self->sb10_stats = vkk_buffer_new(engine->engine, um,
	                                  VKK_BUFFER_USAGE_STORAGE,
	                                  sizeof(nn_tensorStatsData_t),
	                                  NULL);
	if(self->sb10_stats == NULL)
	{
		goto fail_sb10_stats;
	}

	// success
	return self;

	// failure
	fail_sb10_stats:
		vkk_uniformSet_delete(&self->us1);
	fail_us1:
		FREE(self);
	return NULL;
}

void nn_tensorStats_delete(nn_tensorStats_t** _self)
{
	ASSERT(_self);

	nn_tensorStats_t* self = *_self;
	if(self)
	{
		vkk_buffer_delete(&self->sb10_stats);
		vkk_uniformSet_delete(&self->us1);
		FREE(self);
		*_self = NULL;
	}
}

float nn_tensorStats_min(nn_tensorStats_t* self)
{
	ASSERT(self);

	nn_tensorStats_update(self);

	return self->data.min;
}

float nn_tensorStats_max(nn_tensorStats_t* self)
{
	ASSERT(self);

	nn_tensorStats_update(self);

	return self->data.max;
}

float nn_tensorStats_mean(nn_tensorStats_t* self)
{
	ASSERT(self);

	nn_tensorStats_update(self);

	return self->data.mean;
}

float nn_tensorStats_stddev(nn_tensorStats_t* self)
{
	ASSERT(self);

	nn_tensorStats_update(self);

	return self->data.stddev;
}

float nn_tensorStats_norm(nn_tensorStats_t* self)
{
	ASSERT(self);

	nn_tensorStats_update(self);

	return self->data.norm;
}
