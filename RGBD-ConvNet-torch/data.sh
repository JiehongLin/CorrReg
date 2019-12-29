
cd ./Data/utils
matlab -nosplash -nodesktop -nodisplay -r "get_data();quit"

th CreateDir.lua
th mat2t7.lua
th MeanStd.lua



