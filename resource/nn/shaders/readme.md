Neural Network Compute Shader Notes
===================================

Input, Weights and Output
-------------------------

* sb00: X/X1
* sb01: W
* sb02: B
* sb03: Y
* sb04: dY_dX
* sb05: X2

	                      +----+----+----+----+----+----+
	                      |sb00|sb01|sb02|sb03|sb04|sb05|
	+---------------------+----+----+----+----+----+----+
	| clear_dY_dX         |    |    |    |    |  * |    |
	| clear_dL_dX         |    |    |    |    |    |    |
	| convLayerFp         |  * |  * |  * |  * |    |    |
	| convLayerFpT        |  * |  * |  * |  * |    |    |
	| convLayerBp_dL_dX   |    |  * |    |    |    |    |
	| convLayerBp_dL_dW   |    |    |    |    |    |    |
	| convLayerBp_dL_dB   |    |    |    |    |    |    |
	| convLayerBpT_dL_dX  |    |  * |    |    |    |    |
	| convLayerBpT_dL_dW  |    |    |    |    |    |    |
	| convLayerBpGc       |    |  * |  * |    |    |    |
	| convLayerBpUpW      |    |  * |    |    |    |    |
	| convLayerBpUpB      |    |    |  * |    |    |    |
	| factLayerFp*        |  * |  * |    |  * |    |    |
	| factLayerBp*        |  * |  * |    |    |    |    |
	| skipLayerFpAdd      |  * |    |    |  * |    |  * |
	| skipLayerFpCat      |  * |    |    |  * |    |  * |
	| skipLayerBpCat      |    |    |    |    |    |    |
	| skipLayerBpFork     |    |    |    |    |    |    |
	| weightLayerFp       |  * |  * |  * |  * |    |    |
	| weightLayerBp_dL_dX |    |  * |    |    |    |    |
	| weightLayerBp_dL_dW |  * |    |    |    |    |    |
	| weightLayerBp_dL_dB |    |    |    |    |    |    |
	| weightLayerBpGc     |    |  * |  * |    |    |    |
	| weightLayerBpUpW    |    |  * |    |    |    |    |
	| weightLayerBpUpB    |    |    |  * |    |    |    |
	+---------------------+----+----+----+----+----+----+

Gradients and Velocity
----------------------

* sb10: dL_dY
* sb11: dL_dW
* sb12: dL_dB
* sb13: dL_dX/dL_dX1
* sb14: VW
* sb15: VB
* sb16: dL_dX2
* sb17: dL_dY2

	                      +----+----+----+----+----+----+----+----+
	                      |sb10|sb11|sb12|sb13|sb14|sb15|sb16|sb17|
	+---------------------+----+----+----+----+----+----+----+----+
	| clear_dY_dX         |    |    |    |    |    |    |    |    |
	| clear_dL_dX         |    |    |    |  * |    |    |    |    |
	| convLayerFp         |    |    |    |    |    |    |    |    |
	| convLayerFpT        |    |    |    |    |    |    |    |    |
	| convLayerBp_dL_dX   |  * |    |    |  * |    |    |    |    |
	| convLayerBp_dL_dW   |  * |  * |    |  * |    |    |    |    |
	| convLayerBp_dL_dB   |  * |    |  * |    |    |    |    |    |
	| convLayerBpT_dL_dX  |  * |    |    |  * |    |    |    |    |
	| convLayerBpT_dL_dW  |  * |  * |    |  * |    |    |    |    |
	| convLayerBpGc       |    |  * |  * |    |    |    |    |    |
	| convLayerBpUpW      |    |  * |    |    |  * |    |    |    |
	| convLayerBpUpB      |    |    |  * |    |    |  * |    |    |
	| factLayerFp*        |    |    |    |    |    |    |    |    |
	| factLayerBp*        |  * |    |    |    |    |    |    |    |
	| skipLayerFpAdd      |    |    |    |    |    |    |    |    |
	| skipLayerFpCat      |    |    |    |    |    |    |    |    |
	| skipLayerBpCat      |  * |    |    |  * |    |    |  * |    |
	| skipLayerBpFork     |  * |    |    |    |    |    |    |  * |
	| weightLayerFp       |    |    |    |    |    |    |    |    |
	| weightLayerBp_dL_dX |  * |    |    |  * |    |    |    |    |
	| weightLayerBp_dL_dW |  * |  * |    |  * |    |    |    |    |
	| weightLayerBp_dL_dB |  * |    |  * |    |    |    |    |    |
	| weightLayerBpGc     |    |  * |  * |    |    |    |    |    |
	| weightLayerBpUpW    |    |  * |    |    |  * |    |    |    |
	| weightLayerBpUpB    |    |    |  * |    |    |  * |    |    |
	+---------------------+----+----+----+----+----+----+----+----+

Architecture, Parameters, Indices and Gradient Clipping
-------------------------------------------------------

* sb20: arch
* sb21: param
* sb22: idx
* sb23: gc

	                      +----+----+----+----+
	                      |sb20|sb21|sb22|sb23|
	+---------------------+----+----+----+----+
	| clear_dY_dX         |    |  * |    |    |
	| clear_dL_dX         |    |  * |    |    |
	| convLayerFp         |    |  * |    |    |
	| convLayerFpT        |    |  * |    |    |
	| convLayerBp_dL_dX   |    |  * |  * |    |
	| convLayerBp_dL_dW   |    |  * |  * |    |
	| convLayerBp_dL_dB   |    |  * |  * |    |
	| convLayerBpT_dL_dX  |    |  * |  * |    |
	| convLayerBpT_dL_dW  |    |  * |  * |    |
	| convLayerBpGc       |  * |  * |    |  * |
	| convLayerBpUpW      |  * |  * |    |  * |
	| convLayerBpUpB      |  * |  * |    |  * |
	| factLayerFp*        |    |    |    |    |
	| factLayerBp*        |    |    |    |    |
	| skipLayerFpAdd      |    |    |    |    |
	| skipLayerFpCat      |    |    |    |    |
	| skipLayerBpCat      |    |    |    |    |
	| skipLayerBpFork     |    |    |    |    |
	| weightLayerFp       |    |  * |    |    |
	| weightLayerBp_dL_dX |    |  * |  * |    |
	| weightLayerBp_dL_dW |    |  * |  * |    |
	| weightLayerBp_dL_dB |    |  * |  * |    |
	| weightLayerBpGc     |  * |  * |    |  * |
	| weightLayerBpUpW    |  * |  * |    |  * |
	| weightLayerBpUpB    |  * |  * |    |  * |
	+---------------------+----+----+----+----+

Backpropagation Dispatch Order
------------------------------

Convolution Layer

* clear_dL_dX
* convLayerBp_dL_dX
* convLayerBp_dL_dW
* convLayerBp_dL_dB
* convLayerBpGc
* convLayerBpUpW
* convLayerBpUpB

Convolution Layer Transpose

* clear_dL_dX
* convLayerBpT_dL_dX
* convLayerBpT_dL_dW
* convLayerBp_dL_dB
* convLayerBpGc
* convLayerBpUpW
* convLayerBpUpB
