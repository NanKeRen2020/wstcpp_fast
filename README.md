## About

This project implement a faster version of repo[10] in C++, with employing a faster fft, 
gemm and openmp parallel programming to reduce the running time to ~1/3 of original.

Rencently, much of the AI excitement is around LLM's and the transformer[1][2][3]. 
There are many other promising AI architectures that are worth exploring. 
I'm particularly interested in and highly recommend the wavelet scattering transform
[4][5]and think it can be used to build a simple and interpretable AI model.

Wavelet scattering transforms are a type of convolutional neural network that are invariant 
to translation, stable to deformations, and provide a multiscale representation of the input. 
They are a great alternative to the convolutional neural network and have been shown 
to be effective in many applications[6][7], and provide state-of-the-art classification results 
among predefined or unsupervised representations. They are nearly as efficient as learned 
deep networks on relatively simple image datasets, such as digits in MNIST, textures or 
small CIFAR images. For complex datasets such as ImageNet, [7] shows that learning a single dictionary 
is sufficient to improve the performance of a predefined scattering representation beyond 
the accuracy of AlexNet on ImageNet. A sparse scattering network reduces the 
convolutional network learning to a single dictionary learning problem. 
It opens the possibility to study the network properties by analyzing the resulting dictionary. 
It also offers a simpler mathematical framework to analyze optimization issues.
 


## Environments

Ubuntu1604  OpenCV3.4.x 

sudo apt-get install liblapack-dev

sudo apt-get install libblas-dev

sudo apt-get install libboost-dev

sudo apt-get install libarmadillo-dev


## Build & Usage

cd wstcpp_fast

mkdir build && cd build

cmake ..

make -j4 

```bash

./test_wst ../data/t10k-images-idx3-ubyte  ../data/model.txt -1
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ * * _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ * _ _ _ * * _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ * _ _ _ _ _ * * _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ * _ _ _ _ _ _ _ * _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ * _ _ _ _ _ _ _ _ * * _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ * _ _ _ _ _ _ _ _ * _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ * _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ * _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ * _ _ _ _ _ _ _ _ * _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ * * _ _ _ _ _ _ * _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ * _ _ _ _ _ * * _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ * * * _ * * _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ * _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ * _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ * _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ * _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ * _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ * _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

0: 0.00
1: 0.00
2: 0.00
3: 0.00
4: 0.01
5: 0.00
6: 0.00
7: 0.02
8: 0.00
9: 0.96 << 🙂

./test_wst ../data/t10k-images-idx3-ubyte  ../data/model.txt -1

_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ * _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ * * * * * * * * _ * * _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ * * * * * * * * * * * * _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ * _ * * * * * * _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ * * * _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ * * * _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ * * * _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ * * * _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ * * * _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ * * * _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ * * * _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ * * * * _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ * * * _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ * * _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ * * _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ * * * _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ * * * _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ * * _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ * * _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ * _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

0: 0.00
1: 0.00
2: 0.00
3: 0.00
4: 0.00
5: 0.00
6: 0.00
7: 1.00 << 🙂
8: 0.00
9: 0.00
```

## Todo

I strongly want to reimplemnt this repo https://github.com/j-zarka/SparseScatNet in C++ to test and verify 

the accuracy in complex datasets such as ImageNet. But i am not sure can finish this work.


## Train 

```bash
python3 scripts/train.py
```

## Discusses

As we kown, deep convolutional networks have spectacular applications to classification and regression, 
but they are black boxes that are hard to analyze mathematically because of their architecture complexity. 
Scattering transforms are simplified convolutional neural networks with wavelet filters which are not learned.


Since scattering transforms are instantiations of CNNs, they have been studied as mathematical models 
for understanding the impressive success of CNNs in image classification. As discussed in [5], 
first-order scattering coefficients are similar to SIFT descriptors, and higher-order scattering 
can provide insight into the information added with depth. Moreover, theoretical and 
empirical study of information encoded in scattering networks indicates that they often promote 
linear separability, which leads to effective representations for downstream classification tasks.

Deep convolutional image classifiers progressively transform the spatial variability into 
a smaller number of channels, which linearly separates all classes. A fundamental challenge 
is to understand the role of rectifiers together with convolutional filters in this transformation. 
Rectifiers with biases are often interpreted as thresholding operators which improve sparsity and discrimination. 

[8] shows that the improvement of linear separability for image classification in deep convolutional 
networks mostly relies on a phase collapse phenomenon. Eliminating the phase of zero-mean
filters improves the separation of class means.This phase collapse network reaches the classification accuracy 
of ResNets of similar depths, whereas its performance is considerably degraded when replacing
the phase collapse with thresholding operators. This is justified by explaining
how iterated phase collapses progressively improve separation of class means, as
opposed to thresholding non-linearities.


## References

[1] Vaswani, Ashish, et al. Attention is all you need. Advances in Neural Information Processing Systems. 2017.

[2] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, 
    Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 
    An image is worth 16x16 words:Transformers for image recognition at scale. In ICLR, 2021.

[3] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick. Masked Autoencoders Are
    Scalable Vision Learners. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 16000-16009 

[4] S. Mallat. Group invariant scattering. Communications on Pure and Applied Mathematics, 
    65(10):1331–1398, 2012.(https://www.di.ens.fr/~mallat/papiers/ScatCPAM.pdf)

[5] Bruna, J., and S. Mallat. Invariant Scattering Convolution Networks. IEEE Transactions on Pattern Analysis
    and Machine Intelligence. Vol. 35, Number 8, 2013, pp. 1872–1886.

[6] E. Oyallon, S. Zagoruyko, G. Huang, N. Komodakis, S. Lacoste-Julien, M. Blaschko, and
    E. Belilovsky. Scattering networks for hybrid representation learning. IEEE Transactions on
    Pattern Analysis and Machine Intelligence, 41(9):2208–2221, Sep. 2019.

[7] Zarka J , Thiry L ,Angles, Tomás,et al. Deep Network Classification by Scattering and 
    Homotopy Dictionary Learning[J]. 2019.DOI:10.48550/arXiv.1910.03561.

[8] Guth, Florentin & Zarka, John & Mallat, Stéphane. (2021). Phase Collapse in Neural Networks. 

[9] S. Mallat. Understanding deep convolutional networks. Phil. Trans. R.Soc. A, 374(2065):20150203, 2016.
 
[10] https://github.com/drbh/wst.cpp

[11] [Stéphane Mallat: "Scattering Invariant Deep Networks for Classification, Pt. 1 (video)"](https://www.youtube.com/watch?v=4eyUReyIPXg)

[12] [A ConvNet that works well with 20 samples: Wavelet Scattering (article)](https://towardsdatascience.com/-a-convnet-that-works-on-like-20-samples-scatter-wavelets-b2e858f8a385)

[13] [Wavelet Scattering (matlab docs)](https://www.mathworks.com/help/wavelet/ug/wavelet-scattering.html)

[14] [A better way to define and describe Morlet wavelets for time-frequency analysis (paper)](https://www.biorxiv.org/content/biorxiv/early/2018/08/21/397182.full.pdf)

[15] [Kymatio: Wavelet scattering in Python (site)](https://www.kymat.io/)

[16] [Kymatio: Scattering Transforms in Python (paper)](https://jmlr.org/papers/volume21/19-047/19-047.pdf)
