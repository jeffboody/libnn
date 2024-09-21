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

// Pseudocode developed using https://www.perplexity.ai

#include <stdint.h>
#include <math.h>

typedef struct
{
    uint32_t count;
    uint32_t height;
    uint32_t width;
    uint32_t depth;
    float*   data;
} tensor_t;

float tensor_get(tensor_t* T, uint32_t n, uint32_t i,
                 uint32_t j, uint32_t k)
{
    uint32_t sn = T->height * T->width * T->depth;
    uint32_t si = T->width * T->depth;
    uint32_t sj = T->depth;
    return T->data[n*sn + i*si + j*sj + k];
}

void tensor_set(tensor_t* T, uint32_t n, uint32_t i,
                uint32_t j, uint32_t k, float val)
{
    uint32_t sn = T->height * T->width * T->depth;
    uint32_t si = T->width * T->depth;
    uint32_t sj = T->depth;
    T->data[n*sn + i*si + j*sj + k] = val;
}

// Helper function to clamp a value between a minimum and maximum
int32_t clamp(int32_t value, int32_t min, int32_t max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

void convForwardPass(tensor_t* X, tensor_t* W, tensor_t* B, tensor_t* Y,
                     uint32_t m, uint32_t yi, uint32_t yj, uint32_t f,
                     uint32_t stride)
{
    float y = 0.0f;

    // Iterate over the filter dimensions
    for (uint32_t fi = 0; fi < W->height; fi++) {
        // Calculate input indices
        int32_t xi = (int32_t)yi * stride + fi - (W->height / 2);

        // Clamp the input indices to the edge of X
        uint32_t clamped_xi = (uint32_t) clamp(xi, 0, (int32_t)X->height - 1);

        for (uint32_t fj = 0; fj < W->width; fj++) {
            // Calculate input indices
            int32_t xj = (int32_t)yj * stride + fj - (W->width / 2);

            // Clamp the input indices to the edge of X
            uint32_t clamped_xj = (uint32_t) clamp(xj, 0, (int32_t)X->width - 1);

            for (uint32_t xk = 0; xk < X->depth; xk++) {
                float x = tensor_get(X, m, clamped_xi, clamped_xj, xk);
                float w = tensor_get(W, f, fi, fj, xk);
                y += x * w;
            }
        }
    }

    // Add bias
    y += tensor_get(B, 0, 0, 0, f);

    // Set the output value (without applying ReLU)
    tensor_set(Y, m, yi, yj, f, y);
}

void convBackprop_dL_dW(tensor_t* dL_dY, tensor_t* X, tensor_t* dL_dW,
                        uint32_t f, uint32_t fi, uint32_t fj, uint32_t xk,
                        uint32_t stride)
{
    float dl_dw = 0.0f;

    // Iterate over all positions in the output tensor
    for (uint32_t m = 0; m < dL_dY->count; m++) {
        for (uint32_t yi = 0; yi < dL_dY->height; yi++) {
            // Calculate xi for each yi
            int32_t xi = (int32_t)(yi * stride) + fi - (dL_dW->height / 2);

            // Skip this row if xi is out of bounds
            if (xi < 0 || xi >= X->height) continue;

            for (uint32_t yj = 0; yj < dL_dY->width; yj++) {
                // Calculate xj for each yj
                int32_t xj = (int32_t)(yj * stride) + fj - (dL_dW->width / 2);

                // Skip this column if xj is out of bounds
                if (xj < 0 || xj >= X->width) continue;

                float dl_dy = tensor_get(dL_dY, m, yi, yj, f);
                float x = tensor_get(X, m, xi, xj, xk);
                dl_dw += dl_dy * x;
            }
        }
    }

    // Set the computed gradient for this weight
    tensor_set(dL_dW, f, fi, fj, xk, dl_dw);
}

// Used for both standard and transposed convolution computation of dL_dB
void convBackprop_dL_dB(tensor_t* dL_dY, tensor_t* dL_dB, uint32_t f)
{
    float dl_db = 0.0f;

    // Iterate over all positions in the output tensor
    for (uint32_t m = 0; m < dL_dY->count; m++) {
        for (uint32_t yi = 0; yi < dL_dY->height; yi++) {
            for (uint32_t yj = 0; yj < dL_dY->width; yj++) {
                dl_db += tensor_get(dL_dY, m, yi, yj, f);
            }
        }
    }

    // Set the computed gradient for this bias
    tensor_add(dL_dB, 0, 0, 0, f, dl_db);
}

void convBackprop_dL_dX(tensor_t* dL_dY, tensor_t* W, tensor_t* dL_dX,
                        uint32_t m, uint32_t xi, uint32_t xj, uint32_t xk,
                        uint32_t stride)
{
    float dl_dx = 0.0f;

    // Iterate over filter height
    for (uint32_t fi = 0; fi < W->height; fi++) {
        // Calculate the corresponding output y-position
        int32_t yi = (int32_t)(xi - fi + (W->height / 2)) / stride;

        // Skip this row if yi is out of bounds
        if (yi < 0 || yi >= dL_dY->height) continue;

        // Iterate over filter width
        for (uint32_t fj = 0; fj < W->width; fj++) {
            // Calculate the corresponding output x-position
            int32_t yj = (int32_t)(xj - fj + (W->width / 2)) / stride;

            // Skip this column if yj is out of bounds
            if (yj < 0 || yj >= dL_dY->width) continue;

            // Iterate over all filters
            for (uint32_t f = 0; f < W->count; f++) {
                float dl_dy = tensor_get(dL_dY, m, yi, yj, f);
                float w = tensor_get(W, f, fi, fj, xk);
                dl_dx += dl_dy * w;
            }
        }
    }

    // Set the computed gradient for this input element
    tensor_set(dL_dX, m, xi, xj, xk, dl_dx);
}

// Used for both standard and transposed convolution weight updates
void convBackpropUpdateW(tensor_t* dL_dW, tensor_t* MW, tensor_t* VW, tensor_t* W,
                         uint32_t f, uint32_t fi, uint32_t fj, uint32_t xk,
                         float adam_alpha, float adam_beta1, float adam_beta2,
                         float adam_epsilon, uint32_t t)
{
    // Compute bias-correction terms
    float bc1 = 1.0f - powf(adam_beta1, t);
    float bc2 = 1.0f - powf(adam_beta2, t);

    // Get current values
    float g = tensor_get(dL_dW, f, fi, fj, xk);
    float m = tensor_get(MW, f, fi, fj, xk);
    float v = tensor_get(VW, f, fi, fj, xk);
    float w = tensor_get(W, f, fi, fj, xk);

    // Update biased first moment estimate
    m = adam_beta1 * m + (1.0f - adam_beta1) * g;

    // Update biased second raw moment estimate
    v = adam_beta2 * v + (1.0f - adam_beta2) * g * g;

    // Compute bias-corrected first moment estimate
    float m_hat = m / bc1;

    // Compute bias-corrected second raw moment estimate
    float v_hat = v / bc2;

    // Update weight
    w -= adam_alpha * m_hat / (sqrtf(v_hat) + adam_epsilon);

    // Store updated values
    tensor_set(MW, f, fi, fj, xk, m);
    tensor_set(VW, f, fi, fj, xk, v);
    tensor_set(W, f, fi, fj, xk, w);
}

// Used for both standard and transposed convolution weight updates
void convBackpropUpdateB(tensor_t* dL_dB, tensor_t* MB, tensor_t* VB, tensor_t* B,
                         uint32_t f, float adam_alpha, float adam_beta1, float adam_beta2,
                         float adam_epsilon, uint32_t t)
{
    // Compute bias-correction terms
    float bc1 = 1.0f - powf(adam_beta1, t);
    float bc2 = 1.0f - powf(adam_beta2, t);

    // Get current values
    float g = tensor_get(dL_dB, 0, 0, 0, f);
    float m = tensor_get(MB, 0, 0, 0, f);
    float v = tensor_get(VB, 0, 0, 0, f);
    float b = tensor_get(B, 0, 0, 0, f);

    // Update biased first moment estimate
    m = adam_beta1 * m + (1.0f - adam_beta1) * g;

    // Update biased second raw moment estimate
    v = adam_beta2 * v + (1.0f - adam_beta2) * g * g;

    // Compute bias-corrected first moment estimate
    float m_hat = m / bc1;

    // Compute bias-corrected second raw moment estimate
    float v_hat = v / bc2;

    // Update bias
    b -= adam_alpha * m_hat / (sqrtf(v_hat) + adam_epsilon);

    // Store updated values
    tensor_set(MB, 0, 0, 0, f, m);
    tensor_set(VB, 0, 0, 0, f, v);
    tensor_set(B, 0, 0, 0, f, b);
}

void convTForwardPass(tensor_t* X, tensor_t* W, tensor_t* B, tensor_t* Y,
                      uint32_t m, uint32_t yi, uint32_t yj, uint32_t f,
                      uint32_t stride)
{
    float y = 0.0f;

    // Calculate the starting position in the input
    int32_t start_xi = (int32_t)yi / stride;
    int32_t start_xj = (int32_t)yj / stride;

    // Iterate over the input region that contributes to this output pixel
    for (int32_t xi = start_xi; xi < start_xi + (int32_t)W->height; xi++) {
        // Calculate the corresponding filter x-position (centered filter approach)
        int32_t fi = (int32_t)yi - xi * stride + ((int32_t)W->height / 2);

        // Check if the filter x-position is valid
        if (fi < 0 || fi >= (int32_t)W->height) continue;

        // Apply clamp-to-edge for input positions
        uint32_t clamped_xi = (uint32_t) clamp(xi, 0, (int32_t)X->height - 1);

        for (int32_t xj = start_xj; xj < start_xj + (int32_t)W->width; xj++) {
            // Calculate the corresponding filter y-position (centered filter approach)
            int32_t fj = (int32_t)yj - xj * stride + ((int32_t)W->width / 2);

            // Check if the filter y-position is valid
            if (fj < 0 || fj >= (int32_t)W->width) continue;

            // Apply clamp-to-edge for input positions
            uint32_t clamped_xj = (uint32_t) clamp(xj, 0, (int32_t)X->width - 1);

            for (uint32_t xk = 0; xk < X->depth; xk++) {
                float x = tensor_get(X, m, clamped_xi, clamped_xj, xk);
                float w = tensor_get(W, f, (uint32_t)fi, (uint32_t)fj, xk);
                y += x * w;
            }
        }
    }

    // Add bias
    y += tensor_get(B, 0, 0, 0, f);

    // Set the output value
    tensor_set(Y, m, yi, yj, f, y);
}

void convTBackprop_dL_dW(tensor_t* dL_dY, tensor_t* X, tensor_t* dL_dW,
                         uint32_t f, uint32_t fi, uint32_t fj, uint32_t xk,
                         uint32_t stride)
{
    float dl_dw = 0.0f;

    // Iterate over all positions in the input tensor
    for (uint32_t m = 0; m < X->count; m++) {
        for (uint32_t xi = 0; xi < X->height; xi++) {
            // Calculate the corresponding output position
            int32_t yi = xi * stride + fi - (dL_dW->height / 2);

            // Check if the output position is valid
            if (yi < 0 || yi >= dL_dY->height) continue;

            for (uint32_t xj = 0; xj < X->width; xj++) {
                // Calculate the corresponding output position
                int32_t yj = xj * stride + fj - (dL_dW->width / 2);

                // Check if the output position is valid
                if (yj < 0 || yj >= dL_dY->width) continue;

                float x = tensor_get(X, m, xi, xj, xk);
                float dl_dy = tensor_get(dL_dY, m, yi, yj, f);
                dl_dw += x * dl_dy;
            }
        }
    }

    // Update the gradient for this weight
    tensor_set(dL_dW, f, fi, fj, xk, dl_dw);
}

void convTBackprop_dL_dX(tensor_t* dL_dY, tensor_t* W, tensor_t* dL_dX,
                         uint32_t m, uint32_t xi, uint32_t xj, uint32_t xk,
                         uint32_t stride)
{
    float dl_dx = 0.0f;

    // Iterate over all filters
    for (uint32_t f = 0; f < W->count; f++) {
        // Iterate over filter dimensions
        for (uint32_t fi = 0; fi < W->height; fi++) {
            // Calculate the corresponding output y-position
            int32_t yi = xi * stride + fi - (W->height / 2);

            // Skip if the output y-position is out of bounds
            if (yi < 0 || yi >= dL_dY->height) continue;

            for (uint32_t fj = 0; fj < W->width; fj++) {
                // Calculate the corresponding output x-position
                int32_t yj = xj * stride + fj - (W->width / 2);

                // Skip if the output x-position is out of bounds
                if (yj < 0 || yj >= dL_dY->width) continue;

                float dl_dy = tensor_get(dL_dY, m, yi, yj, f);
                float w = tensor_get(W, f, fi, fj, xk);
                dl_dx += dl_dy * w;
            }
        }
    }

    // Set the computed gradient for this input element
    tensor_set(dL_dX, m, xi, xj, xk, dl_dx);
}

void denseForwardPass(tensor_t* X, tensor_t* W, tensor_t* B, tensor_t* Y,
                      uint32_t m, uint32_t n)
{
    float y = 0.0f;

    // Iterate over all input nodes
    for (uint32_t xk = 0; xk < X->depth; xk++) {
        float x = tensor_get(X, m, 0, 0, xk);
        float w = tensor_get(W, n, 0, 0, xk);
        y += x * w;
    }

    // Add bias
    y += tensor_get(B, 0, 0, 0, n);

    // Set the output value (without applying activation function)
    tensor_set(Y, m, 0, 0, n, y);
}

void denseBackprop_dL_dX(tensor_t* dL_dY, tensor_t* W, tensor_t* dL_dX,
                         uint32_t m, uint32_t xk)
{
    float dl_dx = 0.0f;

    // Iterate over all output nodes
    for (uint32_t n = 0; n < W->count; n++) {
        float dl_dy = tensor_get(dL_dY, m, 0, 0, n);
        float w = tensor_get(W, n, 0, 0, xk);
        dl_dx += dl_dy * w;
    }

    // Set the computed gradient for this input element
    tensor_set(dL_dX, m, 0, 0, xk, dl_dx);
}

void denseBackprop_dL_dW(tensor_t* dL_dY, tensor_t* X, tensor_t* dL_dW,
                         uint32_t n, uint32_t xk)
{
    float dl_dw = 0.0f;

    // Iterate over all batches
    for (uint32_t m = 0; m < X->count; m++) {
        float dl_dy = tensor_get(dL_dY, m, 0, 0, n);
        float x = tensor_get(X, m, 0, 0, xk);
        dl_dw += dl_dy * x;
    }

    // Set the computed gradient for this weight
    tensor_set(dL_dW, n, 0, 0, xk, dl_dw);
}

void denseBackprop_dL_dB(tensor_t* dL_dY, tensor_t* dL_dB, uint32_t n)
{
    float dl_db = 0.0f;

    // Iterate over all batches
    for (uint32_t m = 0; m < dL_dY->count; m++) {
        dl_db += tensor_get(dL_dY, m, 0, 0, n);
    }

    // Set the computed gradient for this bias
    tensor_set(dL_dB, 0, 0, 0, n, dl_db);
}

void denseBackpropUpdateW(tensor_t* dL_dW, tensor_t* MW, tensor_t* VW, tensor_t* W,
                          uint32_t n, uint32_t xk,
                          float adam_alpha, float adam_beta1, float adam_beta2,
                          float adam_epsilon, uint32_t t)
{
    // Compute bias-correction terms
    float bc1 = 1.0f - powf(adam_beta1, t);
    float bc2 = 1.0f - powf(adam_beta2, t);

    // Get current values
    float g = tensor_get(dL_dW, n, 0, 0, xk);
    float m = tensor_get(MW, n, 0, 0, xk);
    float v = tensor_get(VW, n, 0, 0, xk);
    float w = tensor_get(W, n, 0, 0, xk);

    // Update biased first moment estimate
    m = adam_beta1 * m + (1.0f - adam_beta1) * g;

    // Update biased second raw moment estimate
    v = adam_beta2 * v + (1.0f - adam_beta2) * g * g;

    // Compute bias-corrected first moment estimate
    float m_hat = m / bc1;

    // Compute bias-corrected second raw moment estimate
    float v_hat = v / bc2;

    // Update weight
    w -= adam_alpha * m_hat / (sqrtf(v_hat) + adam_epsilon);

    // Store updated values
    tensor_set(MW, n, 0, 0, xk, m);
    tensor_set(VW, n, 0, 0, xk, v);
    tensor_set(W, n, 0, 0, xk, w);
}

void denseBackpropUpdateB(tensor_t* dL_dB, tensor_t* MB, tensor_t* VB, tensor_t* B,
                          uint32_t n,
                          float adam_alpha, float adam_beta1, float adam_beta2,
                          float adam_epsilon, uint32_t t)
{
    // Compute bias-correction terms
    float bc1 = 1.0f - powf(adam_beta1, t);
    float bc2 = 1.0f - powf(adam_beta2, t);

    // Get current values
    float g = tensor_get(dL_dB, 0, 0, 0, n);
    float m = tensor_get(MB, 0, 0, 0, n);
    float v = tensor_get(VB, 0, 0, 0, n);
    float b = tensor_get(B, 0, 0, 0, n);

    // Update biased first moment estimate
    m = adam_beta1 * m + (1.0f - adam_beta1) * g;

    // Update biased second raw moment estimate
    v = adam_beta2 * v + (1.0f - adam_beta2) * g * g;

    // Compute bias-corrected first moment estimate
    float m_hat = m / bc1;

    // Compute bias-corrected second raw moment estimate
    float v_hat = v / bc2;

    // Update bias
    b -= adam_alpha * m_hat / (sqrtf(v_hat) + adam_epsilon);

    // Store updated values
    tensor_set(MB, 0, 0, 0, n, m);
    tensor_set(VB, 0, 0, 0, n, v);
    tensor_set(B, 0, 0, 0, n, b);
}

void logisticForwardPass(tensor_t* X, tensor_t* Y,
                         uint32_t m, uint32_t xi, uint32_t xj, uint32_t xk)
{
    // Get the input value
    float x = tensor_get(X, m, xi, xj, xk);

    // Compute the logistic (sigmoid) function
    float y = 1.0f / (1.0f + expf(-x));

    // Set the output value
    tensor_set(Y, m, xi, xj, xk, y);
}

void logisticBackprop(tensor_t* X, tensor_t* dL_dY, tensor_t* dL_dX,
                      uint32_t m, uint32_t xi, uint32_t xj, uint32_t xk)
{
    // Get the input value
    float x = tensor_get(X, m, xi, xj, xk);

    // Compute the sigmoid function
    float sigmoid_x = 1.0f / (1.0f + expf(-x));

    // Get the gradient of the loss with respect to the output
    float dl_dy = tensor_get(dL_dY, m, xi, xj, xk);

    // Compute the gradient of the loss with respect to the input
    // dL/dX = dL/dY * dY/dX
    // where dY/dX = sigmoid(x) * (1 - sigmoid(x))
    float dl_dx = dl_dy * sigmoid_x * (1.0f - sigmoid_x);

    // Set the computed gradient for this input element
    tensor_set(dL_dX, m, xi, xj, xk, dl_dx);
}

void LReLUForwardPass(tensor_t* X, tensor_t* Y,
                      uint32_t m, uint32_t xi, uint32_t xj, uint32_t xk)
{
    // Get the input value
    float x = tensor_get(X, m, xi, xj, xk);

    // Apply Leaky ReLU activation
    // Leaky ReLU: f(x) = max(alpha * x, x), where alpha is a small positive value
    float alpha = 0.01f;  // This is a common value for alpha, but it can be adjusted
    float y = (x > 0) ? x : alpha * x;

    // Set the output value
    tensor_set(Y, m, xi, xj, xk, y);
}

void LReLUBackprop(tensor_t* X, tensor_t* dL_dY, tensor_t* dL_dX,
                   uint32_t m, uint32_t xi, uint32_t xj, uint32_t xk)
{
    // Get the input value
    float x = tensor_get(X, m, xi, xj, xk);

    // Get the gradient of the loss with respect to the output
    float dl_dy = tensor_get(dL_dY, m, xi, xj, xk);

    // Define the alpha value for Leaky ReLU
    float alpha = 0.01f;  // This is a common value, but it can be adjusted

    // Compute the gradient of the loss with respect to the input
    float dl_dx;
    if (x > 0) {
        dl_dx = dl_dy;  // If x > 0, the gradient passes through unchanged
    } else {
        dl_dx = alpha * dl_dy;  // If x <= 0, the gradient is scaled by alpha
    }

    // Set the computed gradient for this input element
    tensor_set(dL_dX, m, xi, xj, xk, dl_dx);
}

void lossMSE(tensor_t* Y, tensor_t* Yt, tensor_t* dL_dY,
             uint32_t m, uint32_t yi, uint32_t yj, uint32_t yk)
{
    // Get the predicted and target values
    float y = tensor_get(Y, m, yi, yj, yk);
    float yt = tensor_get(Yt, m, yi, yj, yk);

    // Compute the error
    float error = y - yt;

    // Compute the gradient of the loss with respect to Y
    // For MSE, dL/dY = (Y - Yt)
    // The factor of 2 and division by N are typically absorbed into the learning rate
    float dl_dy = error;

    // Set the computed gradient
    tensor_set(dL_dY, m, yi, yj, yk, dl_dy);

    // Note: The actual loss value (squared_error) is not stored or returned
    // in this function as per the given prototype. If needed, it should be
    // accumulated elsewhere.
}

void normalized_xavier_init(tensor_t* W) {
    uint32_t fan_in = W->height * W->width * W->depth;
    uint32_t fan_out = W->count;

    float std_dev = sqrtf(2.0f / (fan_in + fan_out));

    for (uint32_t f = 0; f < W->count; f++) {
        for (uint32_t i = 0; i < W->height; i++) {
            for (uint32_t j = 0; j < W->width; j++) {
                for (uint32_t k = 0; k < W->depth; k++) {
                    float random_value = rand_normal(0.0f, std_dev);
                    tensor_set(W, f, i, j, k, random_value);
                }
            }
        }
    }
}

void he_init(tensor_t* W) {
    uint32_t fan_in = W->height * W->width * W->depth;

    float std_dev = sqrtf(2.0f / fan_in);

    for (uint32_t f = 0; f < W->count; f++) {
        for (uint32_t i = 0; i < W->height; i++) {
            for (uint32_t j = 0; j < W->width; j++) {
                for (uint32_t k = 0; k < W->depth; k++) {
                    float random_value = rand_normal(0.0f, std_dev);
                    tensor_set(W, f, i, j, k, random_value);
                }
            }
        }
    }
}
