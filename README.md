# Computing-on-WassersteinSpace

This repository contains the codes to run the experiments of the work "Approximation Theory, Computing, and Deep Learning on the Wasserstein Space" by Massimo Fornasier, Pascal Heid, and Giacomo E. Sodini (https://arxiv.org/abs/2310.19548). 

Since we didn't set a seed, the result will vary throughout different runs.

For all our experiments, we need the Wasserstein distance (as well as the corresponding Kantorovich potentials) to the reference image/measure. For that purpose, first run the Python scripts "CIFAR_Data.py" and "MNIST_Data.py" for the CIFAR-10 and MNIST datasets, respectively. We note that we didn't set any seed for our non-determinstic algorithms, thus the figures might (slightly) differ throughout several runs.

Sec6p1_MNIST.ipynb is a Jupyter Notebook for the experiments in Section 6.1 based on the MNIST dataset. Similarly, Sec6p1_CIFAR.ipynb is the corresponding Jupyter Notebbok for the CIFAR-10 dataset.

Accordingly, Sec6p2_MNIST.ipynb and Sec6p2_CIFAR.ipynb are the the Jupyter Notebooks for the experiments in Section 6.2.

