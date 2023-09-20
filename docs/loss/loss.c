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

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, const char** argv)
{
	float epsilon = FLT_EPSILON;

	float y;
	float yt;
	float dy;
	float mse;
	float mae;
	float bce;

	FILE* fmse = fopen("mse.dat", "w");
	if(fmse == NULL)
	{
		printf("fopen mse.dat failed\n");
		return EXIT_FAILURE;
	}

	FILE* fmae = fopen("mae.dat", "w");
	if(fmae == NULL)
	{
		printf("fopen mae.dat failed\n");
		goto fail_mae;
	}

	FILE* fbce = fopen("bce.dat", "w");
	if(fbce == NULL)
	{
		printf("fopen bce.dat failed\n");
		goto fail_bce;
	}

	int i;
	int j;
	for(i = 1; i < 100; ++i)
	{
		yt  = ((float) i)/100.0f;

		for(j = 1; j < 100; ++j)
		{
			y   = ((float) j)/100.0f;
			dy  = y - yt;
			mse = dy*dy;
			mae = fabs(dy);
			bce = -(y*log10f(yt + epsilon) +
			        (1.0f - y)*log10f(1.0f - yt + epsilon));
			if(j == 0)
			{
				fprintf(fmse, "%f", mse);
				fprintf(fmae, "%f", mae);
				fprintf(fbce, "%f", bce);
			}
			else
			{
				fprintf(fmse, " %f", mse);
				fprintf(fmae, " %f", mae);
				fprintf(fbce, " %f", bce);
			}
		}
		fprintf(fmse, "%s", "\n");
		fprintf(fmae, "%s", "\n");
		fprintf(fbce, "%s", "\n");
	}

	fclose(fbce);
	fclose(fmae);
	fclose(fmse);

	// success
	return EXIT_SUCCESS;

	// failure
	fail_bce:
		fclose(fmae);
	fail_mae:
		fclose(fmse);
	return EXIT_FAILURE;
}
