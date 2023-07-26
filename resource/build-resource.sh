cd nn/shaders
glslangValidator -V nn_batchNormLayer_forwardPassXmean.comp -o nn_batchNormLayer_forwardPassXmean_comp.spv
glslangValidator -V nn_batchNormLayer_forwardPassXvar.comp -o nn_batchNormLayer_forwardPassXvar_comp.spv
glslangValidator -V nn_batchNormLayer_forwardPassXhat.comp -o nn_batchNormLayer_forwardPassXhat_comp.spv
glslangValidator -V nn_batchNormLayer_forwardPassY.comp -o nn_batchNormLayer_forwardPassY_comp.spv
glslangValidator -V nn_batchNormLayer_backprop_dL_dX.comp -o nn_batchNormLayer_backprop_dL_dX_comp.spv
glslangValidator -V nn_batchNormLayer_backprop_dL_dXhat.comp -o nn_batchNormLayer_backprop_dL_dXhat_comp.spv
glslangValidator -V nn_batchNormLayer_backpropSum.comp -o nn_batchNormLayer_backpropSum_comp.spv
glslangValidator -V nn_convLayer_forwardPass.comp -o nn_convLayer_forwardPass_comp.spv
glslangValidator -V nn_convLayer_forwardPassT.comp -o nn_convLayer_forwardPassT_comp.spv
glslangValidator -V nn_convLayer_backprop_dL_dX.comp -o nn_convLayer_backprop_dL_dX_comp.spv
glslangValidator -V nn_convLayer_backprop_dL_dW.comp -o nn_convLayer_backprop_dL_dW_comp.spv
glslangValidator -V nn_convLayer_backprop_dL_dB.comp -o nn_convLayer_backprop_dL_dB_comp.spv
glslangValidator -V nn_convLayer_backpropT_dL_dX.comp -o nn_convLayer_backpropT_dL_dX_comp.spv
glslangValidator -V nn_convLayer_backpropT_dL_dW.comp -o nn_convLayer_backpropT_dL_dW_comp.spv
glslangValidator -V nn_convLayer_backpropGradientClipping.comp -o nn_convLayer_backpropGradientClipping_comp.spv
glslangValidator -V nn_convLayer_backpropUpdateW.comp -o nn_convLayer_backpropUpdateW_comp.spv
glslangValidator -V nn_convLayer_backpropUpdateB.comp -o nn_convLayer_backpropUpdateB_comp.spv
glslangValidator -V nn_factLayer_forwardPassLinear.comp -o nn_factLayer_forwardPassLinear_comp.spv
glslangValidator -V nn_factLayer_forwardPassLogistic.comp -o nn_factLayer_forwardPassLogistic_comp.spv
glslangValidator -V nn_factLayer_forwardPassReLU.comp -o nn_factLayer_forwardPassReLU_comp.spv
glslangValidator -V nn_factLayer_forwardPassPReLU.comp -o nn_factLayer_forwardPassPReLU_comp.spv
glslangValidator -V nn_factLayer_forwardPassTanh.comp -o nn_factLayer_forwardPassTanh_comp.spv
glslangValidator -V nn_factLayer_backpropLinear.comp -o nn_factLayer_backpropLinear_comp.spv
glslangValidator -V nn_factLayer_backpropLogistic.comp -o nn_factLayer_backpropLogistic_comp.spv
glslangValidator -V nn_factLayer_backpropReLU.comp -o nn_factLayer_backpropReLU_comp.spv
glslangValidator -V nn_factLayer_backpropPReLU.comp -o nn_factLayer_backpropPReLU_comp.spv
glslangValidator -V nn_factLayer_backpropTanh.comp -o nn_factLayer_backpropTanh_comp.spv
glslangValidator -V nn_poolingLayer_forwardPassAvg.comp -o nn_poolingLayer_forwardPassAvg_comp.spv
glslangValidator -V nn_poolingLayer_forwardPassMax.comp -o nn_poolingLayer_forwardPassMax_comp.spv
glslangValidator -V nn_poolingLayer_backprop.comp -o nn_poolingLayer_backprop_comp.spv
glslangValidator -V nn_skipLayer_forwardPassAdd.comp -o nn_skipLayer_forwardPassAdd_comp.spv
glslangValidator -V nn_skipLayer_forwardPassCat.comp -o nn_skipLayer_forwardPassCat_comp.spv
glslangValidator -V nn_skipLayer_backpropCat.comp -o nn_skipLayer_backpropCat_comp.spv
glslangValidator -V nn_skipLayer_backpropFork.comp -o nn_skipLayer_backpropFork_comp.spv
glslangValidator -V nn_tensor_clear.comp -o nn_tensor_clear_comp.spv
glslangValidator -V nn_tensor_clearAligned.comp -o nn_tensor_clearAligned_comp.spv
glslangValidator -V nn_weightLayer_forwardPass.comp -o nn_weightLayer_forwardPass_comp.spv
glslangValidator -V nn_weightLayer_backpropGradientClipping.comp -o nn_weightLayer_backpropGradientClipping_comp.spv
glslangValidator -V nn_weightLayer_backpropUpdateW.comp -o nn_weightLayer_backpropUpdateW_comp.spv
glslangValidator -V nn_weightLayer_backpropUpdateB.comp -o nn_weightLayer_backpropUpdateB_comp.spv
glslangValidator -V nn_weightLayer_backprop_dL_dX.comp -o nn_weightLayer_backprop_dL_dX_comp.spv
glslangValidator -V nn_weightLayer_backprop_dL_dW.comp -o nn_weightLayer_backprop_dL_dW_comp.spv
glslangValidator -V nn_weightLayer_backprop_dL_dB.comp -o nn_weightLayer_backprop_dL_dB_comp.spv
cd ../..

# shaders
bfs $1 blobSet nn/shaders/nn_batchNormLayer_forwardPassXmean_comp.spv
bfs $1 blobSet nn/shaders/nn_batchNormLayer_forwardPassXvar_comp.spv
bfs $1 blobSet nn/shaders/nn_batchNormLayer_forwardPassXhat_comp.spv
bfs $1 blobSet nn/shaders/nn_batchNormLayer_forwardPassY_comp.spv
bfs $1 blobSet nn/shaders/nn_batchNormLayer_backprop_dL_dX_comp.spv
bfs $1 blobSet nn/shaders/nn_batchNormLayer_backprop_dL_dXhat_comp.spv
bfs $1 blobSet nn/shaders/nn_batchNormLayer_backpropSum_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayer_forwardPass_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayer_forwardPassT_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayer_backprop_dL_dX_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayer_backprop_dL_dW_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayer_backprop_dL_dB_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayer_backpropT_dL_dX_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayer_backpropT_dL_dW_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayer_backpropGradientClipping_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayer_backpropUpdateW_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayer_backpropUpdateB_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayer_forwardPassLinear_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayer_forwardPassLogistic_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayer_forwardPassReLU_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayer_forwardPassPReLU_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayer_forwardPassTanh_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayer_backpropLinear_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayer_backpropLogistic_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayer_backpropReLU_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayer_backpropPReLU_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayer_backpropTanh_comp.spv
bfs $1 blobSet nn/shaders/nn_poolingLayer_forwardPassAvg_comp.spv
bfs $1 blobSet nn/shaders/nn_poolingLayer_forwardPassMax_comp.spv
bfs $1 blobSet nn/shaders/nn_poolingLayer_backprop_comp.spv
bfs $1 blobSet nn/shaders/nn_skipLayer_forwardPassAdd_comp.spv
bfs $1 blobSet nn/shaders/nn_skipLayer_forwardPassCat_comp.spv
bfs $1 blobSet nn/shaders/nn_skipLayer_backpropCat_comp.spv
bfs $1 blobSet nn/shaders/nn_skipLayer_backpropFork_comp.spv
bfs $1 blobSet nn/shaders/nn_tensor_clear_comp.spv
bfs $1 blobSet nn/shaders/nn_tensor_clearAligned_comp.spv
bfs $1 blobSet nn/shaders/nn_weightLayer_forwardPass_comp.spv
bfs $1 blobSet nn/shaders/nn_weightLayer_backpropGradientClipping_comp.spv
bfs $1 blobSet nn/shaders/nn_weightLayer_backpropUpdateW_comp.spv
bfs $1 blobSet nn/shaders/nn_weightLayer_backpropUpdateB_comp.spv
bfs $1 blobSet nn/shaders/nn_weightLayer_backprop_dL_dX_comp.spv
bfs $1 blobSet nn/shaders/nn_weightLayer_backprop_dL_dW_comp.spv
bfs $1 blobSet nn/shaders/nn_weightLayer_backprop_dL_dB_comp.spv
rm nn/shaders/*.spv
