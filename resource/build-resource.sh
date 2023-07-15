cd nn/shaders
glslangValidator -V nn_clear_dY_dX.comp  -o nn_clear_dY_dX_comp.spv
glslangValidator -V nn_clear_dL_dX.comp  -o nn_clear_dL_dX_comp.spv
glslangValidator -V nn_convLayerForwardPass.comp  -o nn_convLayerForwardPass_comp.spv
glslangValidator -V nn_convLayerForwardPassT.comp  -o nn_convLayerForwardPassT_comp.spv
glslangValidator -V nn_convLayerBackprop_dL_dX.comp  -o nn_convLayerBackprop_dL_dX_comp.spv
glslangValidator -V nn_convLayerBackprop_dL_dW.comp  -o nn_convLayerBackprop_dL_dW_comp.spv
glslangValidator -V nn_convLayerBackprop_dL_dB.comp  -o nn_convLayerBackprop_dL_dB_comp.spv
glslangValidator -V nn_convLayerBackpropT_dL_dX.comp  -o nn_convLayerBackpropT_dL_dX_comp.spv
glslangValidator -V nn_convLayerBackpropT_dL_dW.comp  -o nn_convLayerBackpropT_dL_dW_comp.spv
glslangValidator -V nn_convLayerBackpropGradientClipping.comp  -o nn_convLayerBackpropGradientClipping_comp.spv
glslangValidator -V nn_convLayerBackpropUpdateW.comp  -o nn_convLayerBackpropUpdateW_comp.spv
glslangValidator -V nn_convLayerBackpropUpdateB.comp  -o nn_convLayerBackpropUpdateB_comp.spv
glslangValidator -V nn_factLayerForwardPassLinear.comp  -o nn_factLayerForwardPassLinear_comp.spv
glslangValidator -V nn_factLayerForwardPassLogistic.comp  -o nn_factLayerForwardPassLogistic_comp.spv
glslangValidator -V nn_factLayerForwardPassReLU.comp  -o nn_factLayerForwardPassReLU_comp.spv
glslangValidator -V nn_factLayerForwardPassPReLU.comp  -o nn_factLayerForwardPassPReLU_comp.spv
glslangValidator -V nn_factLayerForwardPassTanh.comp  -o nn_factLayerForwardPassTanh_comp.spv
glslangValidator -V nn_factLayerBackpropLinear.comp  -o nn_factLayerBackpropLinear_comp.spv
glslangValidator -V nn_factLayerBackpropLogistic.comp  -o nn_factLayerBackpropLogistic_comp.spv
glslangValidator -V nn_factLayerBackpropReLU.comp  -o nn_factLayerBackpropReLU_comp.spv
glslangValidator -V nn_factLayerBackpropPReLU.comp  -o nn_factLayerBackpropPReLU_comp.spv
glslangValidator -V nn_factLayerBackpropTanh.comp  -o nn_factLayerBackpropTanh_comp.spv
glslangValidator -V nn_poolingLayerForwardPassAvg.comp  -o nn_poolingLayerForwardPassAvg_comp.spv
glslangValidator -V nn_poolingLayerForwardPassMax.comp  -o nn_poolingLayerForwardPassMax_comp.spv
glslangValidator -V nn_poolingLayerBackprop.comp  -o nn_poolingLayerBackprop_comp.spv
glslangValidator -V nn_skipLayerForwardPassAdd.comp  -o nn_skipLayerForwardPassAdd_comp.spv
glslangValidator -V nn_skipLayerForwardPassCat.comp  -o nn_skipLayerForwardPassCat_comp.spv
glslangValidator -V nn_skipLayerBackpropCat.comp  -o nn_skipLayerBackpropCat_comp.spv
glslangValidator -V nn_skipLayerBackpropFork.comp  -o nn_skipLayerBackpropFork_comp.spv
glslangValidator -V nn_weightLayerForwardPass.comp  -o nn_weightLayerForwardPass_comp.spv
glslangValidator -V nn_weightLayerBackpropGradientClipping.comp  -o nn_weightLayerBackpropGradientClipping_comp.spv
glslangValidator -V nn_weightLayerBackpropUpdateW.comp  -o nn_weightLayerBackpropUpdateW_comp.spv
glslangValidator -V nn_weightLayerBackpropUpdateB.comp  -o nn_weightLayerBackpropUpdateB_comp.spv
glslangValidator -V nn_weightLayerBackprop_dL_dX.comp  -o nn_weightLayerBackprop_dL_dX.spv
glslangValidator -V nn_weightLayerBackprop_dL_dW.comp  -o nn_weightLayerBackprop_dL_dW.spv
glslangValidator -V nn_weightLayerBackprop_dL_dB.comp  -o nn_weightLayerBackprop_dL_dB.spv
cd ../..

# shaders
bfs $1 blobSet nn/shaders/nn_clear_dY_dX_comp.spv
bfs $1 blobSet nn/shaders/nn_clear_dL_dX_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayerForwardPass_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayerForwardPassT_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayerBackprop_dL_dX_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayerBackprop_dL_dW_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayerBackprop_dL_dB_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayerBackpropT_dL_dX_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayerBackpropT_dL_dW_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayerBackpropGradientClipping_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayerBackpropUpdateW_comp.spv
bfs $1 blobSet nn/shaders/nn_convLayerBackpropUpdateB_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayerForwardPassLinear_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayerForwardPassLogistic_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayerForwardPassReLU_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayerForwardPassPReLU_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayerForwardPassTanh_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayerBackpropLinear_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayerBackpropLogistic_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayerBackpropReLU_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayerBackpropPReLU_comp.spv
bfs $1 blobSet nn/shaders/nn_factLayerBackpropTanh_comp.spv
bfs $1 blobSet nn/shaders/nn_poolingLayerForwardPassAvg_comp.spv
bfs $1 blobSet nn/shaders/nn_poolingLayerForwardPassMax_comp.spv
bfs $1 blobSet nn/shaders/nn_poolingLayerBackprop_comp.spv
bfs $1 blobSet nn/shaders/nn_skipLayerForwardPassAdd_comp.spv
bfs $1 blobSet nn/shaders/nn_skipLayerForwardPassCat_comp.spv
bfs $1 blobSet nn/shaders/nn_skipLayerBackpropCat_comp.spv
bfs $1 blobSet nn/shaders/nn_skipLayerBackpropFork_comp.spv
bfs $1 blobSet nn/shaders/nn_weightLayerForwardPass_comp.spv
bfs $1 blobSet nn/shaders/nn_weightLayerBackpropGradientClipping_comp.spv
bfs $1 blobSet nn/shaders/nn_weightLayerBackpropUpdateW_comp.spv
bfs $1 blobSet nn/shaders/nn_weightLayerBackpropUpdateB_comp.spv
bfs $1 blobSet nn/shaders/nn_weightLayerBackprop_dL_dX.spv
bfs $1 blobSet nn/shaders/nn_weightLayerBackprop_dL_dW.spv
bfs $1 blobSet nn/shaders/nn_weightLayerBackprop_dL_dB.spv
rm nn/shaders/*.spv
