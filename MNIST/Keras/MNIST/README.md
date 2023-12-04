# MNIST Exmaples

Some Keras examples over the MNIST dataset

## Example 1

Basic MNIST example with a NN with fully conected layers. This first example shows a very basic NN with three hidden layers (1024). No regularization or normalization is performed. 

Python Notebook: [here](1_mlp_basic.ipynb)

Python code: [here](1_mlp_basic.py)


## Example 2

The same configuration than exercise 1 adding Batch Normalization.

Python code: [here](2_mlp_batchnorm.py)

## Example 3

The same configuration than exercise 2 adding a gaussian noise regularizer.

Python code: [here](3_mlp_BN_GN.py)

## Example 4

The same configuration than exercise 3 adding learning rate annealing (learning rate scheduler).

Python code: [here](4_mlp_BN_GN_LRA.py)

## Example 5

Finally, Data Augmentation is performed to increase virtually the training set using **known** transformations

Python code: [here](5_mlp_BN_GN_LRA_DA.py)


