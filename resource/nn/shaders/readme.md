Neural Network Compute Shader Notes
===================================

Batch Normalization Layer
-------------------------

Layer Uniforms

* sb000: dimX (xbs,xh,xw,xd)
* sb001: G
* sb002: B
* sb003: Xhat
* sb004: Y
* sb005: MG
* sb006: VG
* sb007: MB
* sb008: VB
* sb009: Xmean_mb
* sb010: Xvar_mb
* sb011: Xmean_ra
* sb012: Xvar_ra
* sb013: dL_dXhat
* sb014: Bsum
* sb015: Csum

Forward Pass Uniforms

* sb100: bs
* sb101: state
* sb102: X
* sb103: Xmean
* sb104: Xvar

* sb200: idx (k)

Backprop Uniforms

* sb100: bs
* sb101: state
* sb102: dL_dY

* sb200: idx (k)

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

Layer Uniforms

* sb000: dimX (xbs,xh,xw,xd)
* sb001: dimW (fc,fh,fw,xd)
* sb002: W
* sb003: B
* sb004: dimY
* sb005: Y
* sb006: MW
* sb007: VW
* sb008: MB
* sb009: VB
* sb010: dL_dW
* sb011: dL_dB
* sb012: dL_dX
* sb013: param (disable_bias,stride)

Forward Pass Uniforms

* sb100: bs
* sb101: state
* sb102: X

Backprop Uniforms

* sb100: bs
* sb101: state
* sb102: X
* sb103: dL_dY

* sb200: idx (f,fi,fj,k)

Backprop Dispatch Order

* nn_tensor_clear(hazard=NONE, dL_dW)
* nn_tensor_clear(hazard=NONE, dL_dB) (bias enabled)
* nn_tensor_clear(hazard=NONE, dL_dX)
* nn_convLayer_backprop_dL_dX (for each fi,fj)
* nn_convLayer_backprop_dL_dW (for each f,fi,fj,k)
* nn_convLayer_backprop_dL_dB (for each f)
* nn_convLayer_backpropUpdateW
* nn_convLayer_backpropUpdateB

Backprop Dispatch Order (Transpose)

* nn_tensor_clear(hazard=NONE, dL_dW)
* nn_tensor_clear(hazard=NONE, dL_dB) (bias enabled)
* nn_tensor_clear(hazard=NONE, dL_dX)
* nn_convLayer_backpropT_dL_dX (for each fi,fj)
* nn_convLayer_backpropT_dL_dW (for each f,fi,fj,k)
* nn_convLayer_backprop_dL_dB  (for each f)
* nn_convLayer_backpropUpdateW
* nn_convLayer_backpropUpdateB

Fact Layer
----------

Layer Uniforms

* sb000: dimX
* sb001: Y

Forward Pass Uniforms

* sb100: bs
* sb101: state
* sb102: X

Backprop Uniforms

* sb100: bs
* sb101: state
* sb102: X
* sb103: dL_dY

Skip Layer
----------

Shared Uniforms

* sb00: state
* sb01: param (beta)

Forward Pass Uniforms

* sb10: dimX/dimX1
* sb11: X/X1
* sb12: dimY
* sb13: Y
* sb14: dimX2
* sb15: X2

Backprop Uniforms

* sb20: dim_dL_dY
* sb21: dL_dY
* sb22: dim_dL_dX/dim_dL_dX1
* sb23: dL_dX/dL_dX1
* sb24: dim_dL_dX2
* sb25: dL_dX2
* sb26: dim_dL_dY2
* sb27: dL_dY2

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

* sb20:  dim_dL_dY
* sb21:  dL_dY
* sb22:  dim_dL_dW
* sb23:  dL_dW
* sb24:  dim_dL_dB
* sb25:  dL_dB
* sb26:  dim_dL_dX
* sb27:  dL_dX
* sb28:  MW // dimW
* sb29:  VW // dimW
* sb210: MB // dimB
* sb211: VB // dimB

Backprop Dispatch Order

* nn_tensor_clear(hazard=NONE, dL_dW)
* nn_tensor_clear(hazard=NONE, dL_dB) (bias enabled)
* nn_tensor_clear(hazard=NONE, dL_dX)
* nn_weightLayer_backprop_dL_dX
* nn_weightLayer_backprop_dL_dW
* nn_weightLayer_backprop_dL_dB
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

Stats

* sb10: stats

Spectral Normalization

* sb20: u1
* sb21: v1
* sb22: u2
* sb23: v2
* sb24: c
