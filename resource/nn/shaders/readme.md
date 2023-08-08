Neural Network Compute Shader Notes
===================================

Batch Normalization Layer
-------------------------

Shared Uniforms

* sb00: state
* sb01: dimXhat
* sb02: Xhat
* sb03: dimG
* sb04: G
* sb05: dimB
* sb06: B
* sb07: dimXvar_mb
* sb08: Xvar_mb

Forward Pass Uniforms

* sb10:  dimX
* sb11:  X
* sb12:  dimY
* sb13:  Y
* sb14:  dimXmean
* sb15:  Xmean
* sb16:  dimXvar
* sb17:  Xvar
* sb18:  dimXmean_mb
* sb19:  Xmean_mb
* sb110: dimXmean_ra
* sb111: Xmean_ra
* sb112: dimXvar_ra
* sb113: Xvar_ra

Backprop Uniforms

* sb20: dim_dL_dXhat
* sb21: dL_dXhat
* sb22: dim_dL_dY
* sb23: dL_dY
* sb24: dimBsum
* sb25: Bsum
* sb26: dimCsum
* sb27: Csum

Dispatch Uniforms

* sb30: idx (k)

Forward Pass Dispatch Order

* nn_batchNormLayer_forwardPassXmean (training, for each k)
* nn_batchNormLayer_forwardPassXvar  (training, for each k)
* nn_batchNormLayer_forwardPassXhat
* nn_batchNormLayer_forwardPassY

Backprop Dispatch Order

* nn_batchNormLayer_backprop_dL_dXhat
* nn_batchNormLayer_backpropSum (for each k)
* nn_batchNormLayer_backprop_dL_dX

Convolution Layer
-----------------

Shared Uniforms

* sb00: state
* sb01: param (disable_bias and stride)
* sb02: dimX
* sb03: X
* sb04: dimW
* sb05: W
* sb06: dimB
* sb07: B

Forward Pass Uniforms

* sb10: dimY
* sb11: Y

Backprop Uniforms

* sb20:  gc
* sb21:  dim_dL_dY
* sb22:  dL_dY
* sb23:  dim_dL_dW
* sb24:  dL_dW
* sb25:  dim_dL_dB
* sb26:  dL_dB
* sb27:  dim_dL_dX
* sb28:  dL_dX
* sb29:  dimVW
* sb210: VW
* sb211: dimVB
* sb212: VB

Dispatch Uniforms

* sb30: idx (f,fi,fj,k)

Backprop Dispatch Order

* nn_tensor_clear(hazzard=NONE, dL_dW)
* nn_tensor_clear(hazzard=NONE, dL_dB) (bias enabled)
* nn_tensor_clear(hazzard=NONE, dL_dX)
* nn_convLayer_backprop_dL_dX (for each fi,fj)
* nn_convLayer_backprop_dL_dW (for each f,fi,fj,k)
* nn_convLayer_backprop_dL_dB (for each f)
* nn_convLayer_backpropGradientClipping
* nn_convLayer_backpropUpdateW
* nn_convLayer_backpropUpdateB

Backprop Dispatch Order (Transpose)

* nn_tensor_clear(hazzard=NONE, dL_dW)
* nn_tensor_clear(hazzard=NONE, dL_dB) (bias enabled)
* nn_tensor_clear(hazzard=NONE, dL_dX)
* nn_convLayer_backpropT_dL_dX (for each fi,fj)
* nn_convLayer_backpropT_dL_dW (for each f,fi,fj,k)
* nn_convLayer_backprop_dL_dB  (for each f)
* nn_convLayer_backpropGradientClipping
* nn_convLayer_backpropUpdateW
* nn_convLayer_backpropUpdateB

Fact Layer
----------

Shared Uniforms

* sb00: dimX
* sb01: X

Forward Pass Uniforms

* sb10: dimY
* sb11: Y

Backprop Uniforms

* sb20: dim_dL_dY
* sb21: dL_dY

Pooling Layer
-------------

Shared Uniforms

* sb00: state
* sb01: param (stride)
* sb02: dim_dY_dX
* sb03: dY_dX

Forward Pass Uniforms

* sb10: dimX
* sb11: X
* sb12: dimY
* sb13: Y

Backprop Uniforms

* sb20: dim_dL_dY
* sb21: dL_dY
* sb22: dim_dL_dX
* sb23: dL_dX

Skip Layer
----------

Forward Pass Uniforms

* sb00: dimX/dimX1
* sb01: X/X1
* sb02: dimY
* sb03: Y
* sb04: dimX2
* sb05: X2

Backprop Uniforms

* sb10: dim_dL_dY
* sb11: dL_dY
* sb12: dim_dL_dX/dim_dL_dX1
* sb13: dL_dX/dL_dX1
* sb14: dim_dL_dX2
* sb15: dL_dX2
* sb16: dim_dL_dY2
* sb17: dL_dY2

Weight Layer
------------

Shared Uniforms

* sb00: state
* sb01: param (disable_bias)
* sb02: dimX
* sb03: X
* sb04: dimW
* sb05: W
* sb06: dimB
* sb07: B

Forward Pass Uniforms

* sb10: dimY
* sb11: Y

Backprop Uniforms

* sb20:  gc
* sb21:  dim_dL_dY
* sb22:  dL_dY
* sb23:  dim_dL_dW
* sb24:  dL_dW
* sb25:  dim_dL_dB
* sb26:  dL_dB
* sb27:  dim_dL_dX
* sb28:  dL_dX
* sb29:  dimVW
* sb210: VW
* sb211: dimVB
* sb212: VB

Backprop Dispatch Order

* nn_tensor_clear(hazzard=NONE, dL_dW)
* nn_tensor_clear(hazzard=NONE, dL_dB) (bias enabled)
* nn_tensor_clear(hazzard=NONE, dL_dX)
* nn_weightLayer_backprop_dL_dX
* nn_weightLayer_backprop_dL_dW
* nn_weightLayer_backprop_dL_dB
* nn_weightLayer_backpropGradientClipping
* nn_weightLayer_backpropUpdateW
* nn_weightLayer_backpropUpdateB

Loss
----

Uniforms

* sb00: state
* sb01: dimY
* sb02: Y
* sb03: dimYt
* sb04: Yt
* sb05: dim_dL_dY
* sb06: dL_dY
* sb07: loss

Backprop Dispatch Order

* nn_loss_TYPE
* nn_loss_dL_dY_TYPE

Tensor
------

Uniforms

* sb00: dimX
* sb01: X
