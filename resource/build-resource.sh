cd nn/shaders
glslangValidator -V nn_batchNormLayer_forwardPassXmeanTrain.comp -o nn_batchNormLayer_forwardPassXmeanTrain_comp.spv
glslangValidator -V nn_batchNormLayer_forwardPassXvarTrain.comp -o nn_batchNormLayer_forwardPassXvarTrain_comp.spv
glslangValidator -V nn_batchNormLayer_forwardPassXmeanCompute.comp -o nn_batchNormLayer_forwardPassXmeanCompute_comp.spv
glslangValidator -V nn_batchNormLayer_forwardPassXvarCompute.comp -o nn_batchNormLayer_forwardPassXvarCompute_comp.spv
glslangValidator -V nn_batchNormLayer_forwardPassXhat.comp -o nn_batchNormLayer_forwardPassXhat_comp.spv
glslangValidator -V nn_batchNormLayer_forwardPassY.comp -o nn_batchNormLayer_forwardPassY_comp.spv
glslangValidator -V nn_batchNormLayer_backprop_dL_dX.comp -o nn_batchNormLayer_backprop_dL_dX_comp.spv
glslangValidator -V nn_batchNormLayer_backprop_dL_dXhat.comp -o nn_batchNormLayer_backprop_dL_dXhat_comp.spv
glslangValidator -V nn_batchNormLayer_backpropSum.comp -o nn_batchNormLayer_backpropSum_comp.spv
glslangValidator -V nn_batchNormLayer_backpropSumNOP.comp -o nn_batchNormLayer_backpropSumNOP_comp.spv
glslangValidator -V nn_convLayer_forwardPass.comp -o nn_convLayer_forwardPass_comp.spv
glslangValidator -V nn_convLayer_forwardPassT.comp -o nn_convLayer_forwardPassT_comp.spv
glslangValidator -V nn_convLayer_backprop_dL_dX.comp -o nn_convLayer_backprop_dL_dX_comp.spv
glslangValidator -V nn_convLayer_backprop_dL_dW.comp -o nn_convLayer_backprop_dL_dW_comp.spv
glslangValidator -V nn_convLayer_backprop_dL_dW_dB.comp -o nn_convLayer_backprop_dL_dW_dB_comp.spv
glslangValidator -V nn_convLayer_backprop_dL_dB.comp -o nn_convLayer_backprop_dL_dB_comp.spv
glslangValidator -V nn_convLayer_backpropT_dL_dX.comp -o nn_convLayer_backpropT_dL_dX_comp.spv
glslangValidator -V nn_convLayer_backpropT_dL_dW.comp -o nn_convLayer_backpropT_dL_dW_comp.spv
glslangValidator -V nn_convLayer_backpropUpdateW.comp -o nn_convLayer_backpropUpdateW_comp.spv
glslangValidator -V nn_convLayer_backpropUpdateB.comp -o nn_convLayer_backpropUpdateB_comp.spv
glslangValidator -V nn_factLayer_forwardPassLinear.comp -o nn_factLayer_forwardPassLinear_comp.spv
glslangValidator -V nn_factLayer_forwardPassLogistic.comp -o nn_factLayer_forwardPassLogistic_comp.spv
glslangValidator -V nn_factLayer_forwardPassReLU.comp -o nn_factLayer_forwardPassReLU_comp.spv
glslangValidator -V nn_factLayer_forwardPassPReLU.comp -o nn_factLayer_forwardPassPReLU_comp.spv
glslangValidator -V nn_factLayer_forwardPassLReLU.comp -o nn_factLayer_forwardPassLReLU_comp.spv
glslangValidator -V nn_factLayer_forwardPassTanh.comp -o nn_factLayer_forwardPassTanh_comp.spv
glslangValidator -V nn_factLayer_forwardPassSink.comp -o nn_factLayer_forwardPassSink_comp.spv
glslangValidator -V nn_factLayer_backpropLinear.comp -o nn_factLayer_backpropLinear_comp.spv
glslangValidator -V nn_factLayer_backpropLogistic.comp -o nn_factLayer_backpropLogistic_comp.spv
glslangValidator -V nn_factLayer_backpropReLU.comp -o nn_factLayer_backpropReLU_comp.spv
glslangValidator -V nn_factLayer_backpropPReLU.comp -o nn_factLayer_backpropPReLU_comp.spv
glslangValidator -V nn_factLayer_backpropLReLU.comp -o nn_factLayer_backpropLReLU_comp.spv
glslangValidator -V nn_factLayer_backpropTanh.comp -o nn_factLayer_backpropTanh_comp.spv
glslangValidator -V nn_factLayer_backpropSink.comp -o nn_factLayer_backpropSink_comp.spv
glslangValidator -V nn_lanczosLayer_forwardPassT.comp -o nn_lanczosLayer_forwardPassT_comp.spv
glslangValidator -V nn_lanczosLayer_forwardPassY.comp -o nn_lanczosLayer_forwardPassY_comp.spv
glslangValidator -V nn_lanczosLayer_backprop_dL_dT.comp -o nn_lanczosLayer_backprop_dL_dT_comp.spv
glslangValidator -V nn_lanczosLayer_backprop_dL_dX.comp -o nn_lanczosLayer_backprop_dL_dX_comp.spv
glslangValidator -V nn_skipLayer_forwardPassAdd.comp -o nn_skipLayer_forwardPassAdd_comp.spv
glslangValidator -V nn_skipLayer_forwardPassCat.comp -o nn_skipLayer_forwardPassCat_comp.spv
glslangValidator -V nn_skipLayer_backpropAdd.comp -o nn_skipLayer_backpropAdd_comp.spv
glslangValidator -V nn_skipLayer_backpropCat.comp -o nn_skipLayer_backpropCat_comp.spv
glslangValidator -V nn_skipLayer_backpropFork.comp -o nn_skipLayer_backpropFork_comp.spv
glslangValidator -V nn_tensor_stats.comp -o nn_tensor_stats_comp.spv
glslangValidator -V nn_tensor_sn.comp -o nn_tensor_sn_comp.spv
glslangValidator -V nn_tensor_bssn.comp -o nn_tensor_bssn_comp.spv
glslangValidator -V nn_tensor_computeAddOp.comp -o nn_tensor_computeAddOp_comp.spv
glslangValidator -V nn_tensor_computeCopyOp.comp -o nn_tensor_computeCopyOp_comp.spv
glslangValidator -V nn_tensor_computeFillOp.comp -o nn_tensor_computeFillOp_comp.spv
glslangValidator -V nn_tensor_computeMixOp.comp -o nn_tensor_computeMixOp_comp.spv
glslangValidator -V nn_tensor_computeMulOp.comp -o nn_tensor_computeMulOp_comp.spv
glslangValidator -V nn_tensor_computeScaleOp.comp -o nn_tensor_computeScaleOp_comp.spv
glslangValidator -V nn_tensor_computeScaleAddOp.comp -o nn_tensor_computeScaleAddOp_comp.spv
glslangValidator -V nn_weightLayer_forwardPass.comp -o nn_weightLayer_forwardPass_comp.spv
glslangValidator -V nn_weightLayer_backpropUpdateW.comp -o nn_weightLayer_backpropUpdateW_comp.spv
glslangValidator -V nn_weightLayer_backpropUpdateB.comp -o nn_weightLayer_backpropUpdateB_comp.spv
glslangValidator -V nn_weightLayer_backprop_dL_dX.comp -o nn_weightLayer_backprop_dL_dX_comp.spv
glslangValidator -V nn_weightLayer_backprop_dL_dW.comp -o nn_weightLayer_backprop_dL_dW_comp.spv
glslangValidator -V nn_weightLayer_backprop_dL_dB.comp -o nn_weightLayer_backprop_dL_dB_comp.spv
glslangValidator -V nn_loss_dL_dY_mse.comp -o nn_loss_dL_dY_mse_comp.spv
glslangValidator -V nn_loss_dL_dY_mae.comp -o nn_loss_dL_dY_mae_comp.spv
glslangValidator -V nn_loss_dL_dY_bce.comp -o nn_loss_dL_dY_bce_comp.spv
glslangValidator -V nn_loss_mse.comp -o nn_loss_mse_comp.spv
glslangValidator -V nn_loss_mae.comp -o nn_loss_mae_comp.spv
glslangValidator -V nn_loss_bce.comp -o nn_loss_bce_comp.spv
cd ../..

# shaders
bfs $1 blobSet nn/shaders/nn_batchNormLayer_forwardPassXmeanTrain_comp.spv
bfs $1 blobSet nn/shaders/nn_batchNormLayer_forwardPassXvarTrain_comp.spv
bfs $1 blobSet nn/shaders/nn_batchNormLayer_forwardPassXmeanCompute_comp.spv
bfs $1 blobSet nn/shaders/nn_batchNormLayer_forwardPassXvarCompute_comp.spv
bfs $1 blobSet nn/shaders/nn_batchNormLayer_forwardPassXhat_comp.spv
bfs $1 blobSet nn/shaders/nn_batchNormLayer_forwardPassY_comp.spv
bfs $1 blobSet nn/shaders/nn_batchNormLayer_backprop_dL_dX_comp.spv
bfs $1 blobSet nn/shaders/nn_batchNormLayer_backprop_dL_dXhat_comp.spv
bfs $1 blobSet nn/shaders/nn_batchNormLayer_backpropSum_comp.spv
bfs $1 blobSet nn/shaders/nn_batchNormLayer_backpropSumNOP_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayer_forwardPass_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayer_forwardPassT_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayer_backprop_dL_dX_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayer_backprop_dL_dW_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayer_backprop_dL_dW_dB_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayer_backprop_dL_dB_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayer_backpropT_dL_dX_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayer_backpropT_dL_dW_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayer_backpropUpdateW_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayer_backpropUpdateB_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayer_forwardPassLinear_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayer_forwardPassLogistic_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayer_forwardPassReLU_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayer_forwardPassPReLU_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayer_forwardPassLReLU_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayer_forwardPassTanh_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayer_forwardPassSink_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayer_backpropLinear_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayer_backpropLogistic_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayer_backpropReLU_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayer_backpropPReLU_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayer_backpropLReLU_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayer_backpropTanh_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayer_backpropSink_comp.spv
bfs $1 blobSet nn/shaders/nn_lanczosLayer_forwardPassT_comp.spv
bfs $1 blobSet nn/shaders/nn_lanczosLayer_forwardPassY_comp.spv
bfs $1 blobSet nn/shaders/nn_lanczosLayer_backprop_dL_dT_comp.spv
bfs $1 blobSet nn/shaders/nn_lanczosLayer_backprop_dL_dX_comp.spv
bfs $1 blobSet nn/shaders/nn_skipLayer_forwardPassAdd_comp.spv
bfs $1 blobSet nn/shaders/nn_skipLayer_forwardPassCat_comp.spv
bfs $1 blobSet nn/shaders/nn_skipLayer_backpropAdd_comp.spv
bfs $1 blobSet nn/shaders/nn_skipLayer_backpropCat_comp.spv
bfs $1 blobSet nn/shaders/nn_skipLayer_backpropFork_comp.spv
bfs $1 blobSet nn/shaders/nn_tensor_stats_comp.spv
bfs $1 blobSet nn/shaders/nn_tensor_sn_comp.spv
bfs $1 blobSet nn/shaders/nn_tensor_bssn_comp.spv
bfs $1 blobSet nn/shaders/nn_tensor_computeAddOp_comp.spv
bfs $1 blobSet nn/shaders/nn_tensor_computeCopyOp_comp.spv
bfs $1 blobSet nn/shaders/nn_tensor_computeFillOp_comp.spv
bfs $1 blobSet nn/shaders/nn_tensor_computeMixOp_comp.spv
bfs $1 blobSet nn/shaders/nn_tensor_computeMulOp_comp.spv
bfs $1 blobSet nn/shaders/nn_tensor_computeScaleOp_comp.spv
bfs $1 blobSet nn/shaders/nn_tensor_computeScaleAddOp_comp.spv
bfs $1 blobSet nn/shaders/nn_weightLayer_forwardPass_comp.spv
bfs $1 blobSet nn/shaders/nn_weightLayer_backpropUpdateW_comp.spv
bfs $1 blobSet nn/shaders/nn_weightLayer_backpropUpdateB_comp.spv
bfs $1 blobSet nn/shaders/nn_weightLayer_backprop_dL_dX_comp.spv
bfs $1 blobSet nn/shaders/nn_weightLayer_backprop_dL_dW_comp.spv
bfs $1 blobSet nn/shaders/nn_weightLayer_backprop_dL_dB_comp.spv
bfs $1 blobSet nn/shaders/nn_loss_dL_dY_mse_comp.spv
bfs $1 blobSet nn/shaders/nn_loss_dL_dY_mae_comp.spv
bfs $1 blobSet nn/shaders/nn_loss_dL_dY_bce_comp.spv
bfs $1 blobSet nn/shaders/nn_loss_mse_comp.spv
bfs $1 blobSet nn/shaders/nn_loss_mae_comp.spv
bfs $1 blobSet nn/shaders/nn_loss_bce_comp.spv
rm nn/shaders/*.spv
