# Attribution Priors

A repository for training explainable models using attribution priors.

This repository contains tools for connecting the machine learning topics of *model priors* and *model explanations* with a new method called *attribution priors*, discussed in our paper "Learning Explainable Models Using Attribution Priors. This package contains: 
* A differentiable axiomatic feature attribution method called *expected gradients*.
* Tensorflow operations to directly regularize expected gradients attributions during training. 
* Examples of how arbitrary differentiable functions of expected gradient attributions can be regularized during training to encode prior knowledge about a modeling task. 

The main directory contains the file `ops.py`, with Tensorflow operations that allow calculating expected gradients feature attributions, as well as the gradients of those attributions. There are also three folders containing experiments that demonstrate the use of the library.

The `mnist` folder shows how penalizing differences between pixel attributions can lead to smoother explanations of image classifications, and improve performance on images where random salt-and-pepper noise is added.

The `graph` folder shows how penalizing differences between the attributions of neighbors in an arbitrary graph connecting the features can be used to incorporate prior biological knowledgea bout the relationships between genes, yield more biologically plausible explanations of drug response predictions, and improve test error.

The `sparsity` folder shows how encouraging inequality in the distribution of feature attributions can build sparser models that can perform more accurately when training data is limited. 
