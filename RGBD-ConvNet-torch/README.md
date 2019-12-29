## Code for RGB-D object recognition 

### Installation
The code depends on the [Matlab](https://www.mathworks.com/products/matlab.html) and [Torch library](http://torch.ch/). Please install matlab and torch first.

### Usage
To download the [RGB-D object dataset](https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset/rgbd-dataset.tar) and the pre-trained [resnet-50](https://d2j0dndfm35trm.cloudfront.net/resnet-50.t7) model, please run:

```sh download.sh```

To pre-process the dataset, please run:

`sh data.sh`

If you want to conduct the experiments with all data splits and four settings in the paper, please run:

`sh run_all.sh`

If you just want to test the experiments with only one data split and four settings, please run: 

`sh run_split1.sh `
