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

* nn_batchNormLayer_forwardPassXmean(Train|Instance)
  (for each k)
* nn_batchNormLayer_forwardPassXvar(Train|Instance)
  (for each k)
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

Layer Uniforms

* sb000: param (beta)

Forward Pass Uniforms

* sb100: bs
* sb101: state
* sb102: dimX1
* sb103: X1
* sb104: dimX2
* sb105: X2
* sb106: dimY
* sb107: Y

Backprop Uniforms

* sb100: bs
* sb101: state
* sb102: dim_dL_dX1
* sb103: dL_dX1
* sb104: dim_dL_dX2
* sb105: dL_dX2
* sb106: dim_dL_dY1
* sb107: dL_dY1
* sb108: dim_dL_dY2
* sb109: dL_dY2

Weight Layer
------------

Layer Uniforms

* sb000: dimX
* sb001: dimW
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
* sb013: param (disable_bias)

Forward Pass Uniforms

* sb100: bs
* sb101: state
* sb102: X

Backprop Uniforms

* sb100: bs
* sb101: state
* sb102: X
* sb103: dL_dY

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

* sb000: bs
* sb001: loss
* sb002: dimY
* sb003: dL_dY

* sb100: Y
* sb101: Yt

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

* sb10: u1
* sb11: v1
* sb12: u2
* sb13: v2
* sb14: c

OpK (FillK, CopyK, AddK, MixK)

* sb000: dimX1
* sb001: X1
* sb002: dimX2
* sb003: X2
* sb004: dimX3
* sb005: X3
* sb006: idx (n1,n2,n3,count,k1,k2,k3,depth,value)
