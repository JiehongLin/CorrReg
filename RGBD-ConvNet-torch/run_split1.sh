export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

th main.lua -CorrReg true  -drop true  -trials 1
th main.lua -CorrReg true  -drop false -trials 1
th main.lua -CorrReg false -drop true  -trials 1
th main.lua -CorrReg false -drop false -trials 1