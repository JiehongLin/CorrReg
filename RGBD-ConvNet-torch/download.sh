
wget -c https://d2j0dndfm35trm.cloudfront.net/resnet-50.t7
mv resnet-50.t7 ./Model

wget -c https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset/rgbd-dataset.tar
tar -xvf rgbd-dataset.tar -C ./Data
rm  rgbd-dataset.tar



