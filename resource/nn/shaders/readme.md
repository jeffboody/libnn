Neural Network Compute Shader Notes
===================================

Batch Normalization Layer
-------------------------

Layer Uniforms

* sb000: dimX (bs,xh,xw,xd)
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

* nn_batchNormLayer_forwardPassXmean(Train|Compute)
  (for each k)
* nn_batchNormLayer_forwardPassXvar(Train|Compute)
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

* sb000: dimX (bs,xh,xw,xd)
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

Lanczos Layer
-------------

Layer Uniforms

* sb000: dimX (bs,xh,xw,xd)
* sb001: T    (bs,xh,yw,xd)
* sb002: dimY (bs,yh,yw,xd)
* sb003: Y
* sb004: Lw
* sb005: Lh
* sb006: dL_dT
* sb007: dL_dX
* sb008: param (a, fsw, fsh, fcw, fch, szw, szh)

Forward Pass Uniforms

* sb100: bs
* sb101: state
* sb102: X

Backprop Uniforms

* sb100: bs
* sb101: state
* sb102: dL_dY

* sb200: idx (n)

Backprop Dispatch Order

* nn_tensor_clear(hazard=NONE, dL_dT)
* nn_tensor_clear(hazard=NONE, dL_dX)
* nn_lanczosLayer_backprop_dL_dT (for each n)
* nn_lanczosLayer_backprop_dL_dX (for each n)

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

* sb000: dimX
* sb001: X

Stats

* sb100: stats

Spectral Normalization

* sb100: u1
* sb101: v1
* sb102: u2
* sb103: v2
* sb104: c

Op

* sb000: dimX1
* sb001: X1
* sb002: dimX2
* sb003: X2
* sb004: dimY
* sb005: Y
* sb006: idx (x1n,...,value)

Op Functions

* FillOp
* CopyOp
* AddOp
* MixOp
* MulOp
* ScaleOp
* ScaleAddOp
