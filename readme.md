NN: High Performance Neural Network Library in C with Vulkan
============================================================

Overview
--------

NN is a C library for building and training neural networks,
optimized for high performance on modern GPUs.

Features
--------

* Comprehensive layer support including convolutional,
  dense, activation, normalization, and resampling layers;
  efficient implementation of optimization algorithms like
  Adam; and advanced techniques like batch normalization and
  spectral normalization.
* Composite layer support including encoder-decoder and
  unified residual-in-residual dense blocks (URDDB) to
  simplify implementation.
* Tensor operations optimized for filling, copying, adding,
  mixing, scaling, normalizing and computing statistics.
* Experimental Lanczos resampling layer for both upsampling
  and downsampling, offering improved resampling compared
  to pooling or nearest neighbor methods. In some cases the
  Lanczos resampling layer may also replace strided
  convolutions or convolution transpose.
* Leveraging Vulkan compute for efficient utilization of
  computing cores, SIMD vectorization, and reduced memory
  bandwidth.

Development Status
------------------

NN is currently in beta and under active development. Future
releases will include additional layers, optimization
algorithms, improved performance, documentation and
examples.

License
-------

The NN library was implemented by
[Jeff Boody](mailto:jeffboody@gmail.com)
under The MIT License.

	Copyright (c) 2023 Jeff Boody

	Permission is hereby granted, free of charge, to any person obtaining a
	copy of this software and associated documentation files (the "Software"),
	to deal in the Software without restriction, including without limitation
	the rights to use, copy, modify, merge, publish, distribute, sublicense,
	and/or sell copies of the Software, and to permit persons to whom the
	Software is furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included
	in all copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
	THE SOFTWARE.
