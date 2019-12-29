export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

th main.lua -CorrReg true  -drop true  -trials 1
th main.lua -CorrReg true  -drop false -trials 1
th main.lua -CorrReg false -drop true  -trials 1
th main.lua -CorrReg false -drop false -trials 1

th main.lua -CorrReg true  -drop true  -trials 2
th main.lua -CorrReg true  -drop false -trials 2
th main.lua -CorrReg false -drop true  -trials 2
th main.lua -CorrReg false -drop false -trials 2

th main.lua -CorrReg true  -drop true  -trials 3
th main.lua -CorrReg true  -drop false -trials 3
th main.lua -CorrReg false -drop true  -trials 3
th main.lua -CorrReg false -drop false -trials 3

th main.lua -CorrReg true  -drop true  -trials 4
th main.lua -CorrReg true  -drop false -trials 4
th main.lua -CorrReg false -drop true  -trials 4
th main.lua -CorrReg false -drop false -trials 4

th main.lua -CorrReg true  -drop true  -trials 5
th main.lua -CorrReg true  -drop false -trials 5
th main.lua -CorrReg false -drop true  -trials 5
th main.lua -CorrReg false -drop false -trials 5

th main.lua -CorrReg true  -drop true  -trials 6
th main.lua -CorrReg true  -drop false -trials 6
th main.lua -CorrReg false -drop true  -trials 6
th main.lua -CorrReg false -drop false -trials 6

th main.lua -CorrReg true  -drop true  -trials 7
th main.lua -CorrReg true  -drop false -trials 7
th main.lua -CorrReg false -drop true  -trials 7
th main.lua -CorrReg false -drop false -trials 7

th main.lua -CorrReg true  -drop true  -trials 8
th main.lua -CorrReg true  -drop false -trials 8
th main.lua -CorrReg false -drop true  -trials 8
th main.lua -CorrReg false -drop false -trials 8

th main.lua -CorrReg true  -drop true  -trials 9
th main.lua -CorrReg true  -drop false -trials 9
th main.lua -CorrReg false -drop true  -trials 9
th main.lua -CorrReg false -drop false -trials 9

th main.lua -CorrReg true  -drop true  -trials 10
th main.lua -CorrReg true  -drop false -trials 10
th main.lua -CorrReg false -drop true  -trials 10
th main.lua -CorrReg false -drop false -trials 10
