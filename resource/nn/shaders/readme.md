Neural Network Compute Shader Notes
===================================

Batch Normalization Layer
-------------------------

Global Uniforms

* sb00: arch
* sb01: param

Shared Uniforms

* sb10: idx
* sb11: dimXhat
* sb12: Xhat
* sb13: dimG
* sb14: G
* sb15: dimB
* sb16: B
* sb17: dimXvar_mb
* sb18: Xvar_mb

Forward Pass Uniforms

* sb20:  dimX
* sb21:  X
* sb22:  dimY
* sb23:  Y
* sb24:  dimXmean
* sb25:  Xmean
* sb26:  dimXvar
* sb27:  Xvar
* sb28:  dimXmean_mb
* sb29:  Xmean_mb
* sb210: dimXmean_ra
* sb211: Xmean_ra
* sb212: dimXvar_ra
* sb213: Xvar_ra

Backprop Uniforms

* sb30: dim_dL_dXhat
* sb31: dL_dXhat
* sb32: dim_dL_dY
* sb33: dL_dY
* sb34: dimBsum
* sb35: Bsum
* sb36: dimCsum
* sb37: Csum

Backprop Dispatch Order

* nn_batchNormLayer_backprop_dL_dXhat
* nn_batchNormLayer_backpropSum
* nn_batchNormLayer_backprop_dL_dX

Convolution Layer
-----------------

Global Uniforms

* sb00: arch
* sb01: param

Shared Uniforms

* sb10: dimX
* sb11: X
* sb12: dimW
* sb13: W
* sb14: dimB
* sb15: B

Forward Pass Uniforms

* sb20: dimY
* sb21: Y

Backprop Uniforms

* sb30:  idx
* sb31:  gc
* sb32:  dim_dL_dY
* sb33:  dL_dY
* sb34:  dim_dL_dW
* sb35:  dL_dW
* sb36:  dim_dL_dB
* sb37:  dL_dB
* sb38:  dim_dL_dX
* sb39:  dL_dX
* sb310: dimVW
* sb311: VW
* sb312: dimVB
* sb313: VB

Backprop Dispatch Order

* nn_tensor_clear(hazzard=NONE, dL_dX)
* nn_convLayer_backprop_dL_dX
* nn_convLayer_backprop_dL_dW
* nn_convLayer_backprop_dL_dB
* nn_convLayer_backpropGc
* nn_convLayer_backpropUpW
* nn_convLayer_backpropUpB

Backprop Dispatch Order (Transpose)

* nn_tensor_clear(hazzard=NONE, dL_dX)
* nn_convLayer_backpropT_dL_dX
* nn_convLayer_backpropT_dL_dW
* nn_convLayer_backprop_dL_dB
* nn_convLayer_backpropGc
* nn_convLayer_backpropUpW
* nn_convLayer_backpropUpB

Fact Layer
----------

Shared Uniforms

* sb10: dimX
* sb11: X

Forward Pass Uniforms

* sb20: dimY
* sb21: Y

Backprop Uniforms

* sb30: dim_dL_dY
* sb31: dL_dY

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

Global Uniforms

* sb00: arch
* sb01: param

Shared Uniforms

* sb10: dimX
* sb11: X
* sb12: dimW
* sb13: W
* sb14: dimB
* sb15: B

Forward Pass Uniforms

* sb20: dimY
* sb21: Y

Backprop Uniforms

* sb30:  idx
* sb31:  gc
* sb32:  dim_dL_dY
* sb33:  dL_dY
* sb34:  dim_dL_dW
* sb35:  dL_dW
* sb36:  dim_dL_dB
* sb37:  dL_dB
* sb38:  dim_dL_dX
* sb39:  dL_dX
* sb310: dimVW
* sb311: VW
* sb312: dimVB
* sb313: VB

Loss
----

Global Uniforms

* sb00: arch
* sb01: param

Uniforms

* sb10: dimY
* sb11: Y
* sb12: dimYt
* sb13: Yt
* sb14: dim_dL_dY
* sb15: dL_dY
* sb16: loss_loss

Backprop Dispatch Order

* nn_loss_dL_dY_TYPE
* nn_loss_TYPE

Tensor
------

Uniforms

* sb00: dimX
* sb01: X
