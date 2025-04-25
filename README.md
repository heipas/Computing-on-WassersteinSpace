# Computing-on-WassersteinSpace

This repository contains the codes to run (most of) the experiments in the manuscript [M. Fornasier, P. Heid, and G.E. Sodini, *Approximation theory, computing, and deep learning on the Wasserstein space*, Math. Mod. Meth. Appl. S. 35 (04) (2025)]. 

We note that we didn't set any seed for our non-determinstic algorithms, thus the figures might (slightly) differ throughout several runs.

For all our experiments, we need the Wasserstein distance (as well as the corresponding Kantorovich potentials) to the reference image/measure. For that purpose, first run the Python scripts "CIFAR_Data.py" and "MNIST_Data.py" for the CIFAR-10 and MNIST datasets, respectively. 

Sec6p1_MNIST.ipynb is a Jupyter Notebook for the experiments in Section 6.1 based on the MNIST dataset. Similarly, Sec6p1_CIFAR.ipynb is the corresponding Jupyter Notebbok for the CIFAR-10 dataset.

Accordingly, Sec6p2_MNIST.ipynb and Sec6p2_CIFAR.ipynb are the the Jupyter Notebooks for the experiments in Section 6.2.

In Sec6p3_CIFAR.ipynb we perform the experiments for a fully trainable neural network as considered in Section 6.3.

Finally, Sec6p4_Cheeger.ipynb and Sec6p4_L2.ipynb are the Jupyter Notebooks for the experiments leveraging the Euler-Lagrange approach.

**Remark (25.04.2025):** The repository will be updated in the coming days! 
