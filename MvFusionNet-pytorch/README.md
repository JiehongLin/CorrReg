## Code for Multi-view Recognition of 3D Object Shapes

### Installation
Code is tested on Python 2.7 and [pytorch 0.3.1](https://pytorch.org/). Please install pytorch first.

### Usage
* Download dataset:

We use the data processed in paper "[A Deeper Look at 3D Shape Classifiers](https://people.cs.umass.edu/~jcsu/papers/shape_recog/)". Please download [ModelNet40 dataset](http://supermoe.cs.umass.edu/shape_recog/shaded_images.tar.gz) here and put it under the folder  `modelnet40_images_new_12x`. 



* Run the training script:

`python train.py --lamda 0.0005`
