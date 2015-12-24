#git-cnn-Plankton
This project is created by Ziheng Huang, Pan Li and Yifang Song.

In this project, we have explored deep learning method with
parallel training using Spark in plankton classification data 
set, and gain fairly good performance. Our best model has 
achieved 70% test set loss and 0.7 test set accuracy. The 
parallelized stochastic gradient descent training process in 
Spark greatly accelerated the training. For example, the 
training of our final model needed 6 hours for each epoch 
but only needed 2 hours when we use spark to parallelize it 
on 4 kernels.
