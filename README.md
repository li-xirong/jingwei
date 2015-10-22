# Jingwei #

The Jingwei is an open-source testbed for evaluating methods for image tag assignment, tag refinement and tag-based image retrieval. It is developed as part of [our survey effort](http://arxiv.org/abs/1503.08248), aiming to provide a timely reflection of the state-of-the-art in the field.


## Methods implemented ##

Method        | Media         | Learning       | Code          | Platform
------------- | ------------- | -------------  | ------------- | -------------
SemanticField | tag           | instance-based | python        | linux, windows
TagCooccur    |	tag           |	instance based | Python        | linux, windows
TagRanking    |	tag + image   |	instance based | Python        | linux, windows
KNN           | tag + image   | instance based | C + Python    | linux, windows
TagVote       | tag + image   | instance based | C + Python    | linux, windows
TagCooccur+   |	tag + image   | instance based | C + Python    | linux, windows
TagProp	      | tag + image   | model based    | C + Matlab + Python | linux
TagFeature    |	tag + image   | model based    | C + Python    | linux, windows    
RelExample    | tag + image   | model based   |	C + Python     | linux, windows
RobustPCA     | tag + image   | transduction based | C + Matlab + Python | linux


![Code architecture: A high-level view](code-framework.png = 800x)

## Python Dependencies ##

* [NLTK](http://www.nltk.org) and [nltk\_data](https://drive.google.com/file/d/0B89Vll9z5OVEQkN1cmlGVlB5RTA/view?usp=sharing) for [SemanticField](instance_based/dosemtagrel.py) and [RobustPCA](transduction_based/robustpca/robustpca.py).
* [h5py](http://www.h5py.org) and [Numpy / Scipy](http://www.scipy.org) for MATLAB<->Python data exchange of [TagProp](model_based/tagprop/tagprop.py) and [RobustPCA](transduction_based/robustpca/robustpca.py).
* We recommend [Anaconda](https://www.continuum.io/downloads) (Python 2.7), a free Python distribution which include NLTK, h5py, numpy, scipy, etc.
* [web.py](http://webpy.org/) for [web demo](visualize/webdemo)

## Training and Test Data ##

* [http://www.micc.unifi.it/tagsurvey](http://www.micc.unifi.it/tagsurvey)

## Setup ##

* Modify Paths in ``start.sh`` (for linux/mac) and ``start.bat`` for windows.

This file includes several environment variables that the methods depend on, to select proper input and output folders.
From a shell, you can prepare the environment for using the framework with:
```
$ source start.sh 
```

* Configuration and Dependencies.

Depending on the method to be run, several different dependencies must be met and some external packages must be downloaded.
The script `setup.sh` will report ready to run methods, depending on the available system packages.
For some methods, it will also try to download and compile the provided libraries.
```
$ bash setup.sh 
```

## Use a specific method ##

* 


Scripts in [doit](doit) provide step-by-step usages of each method.

## Contact ##

* Xirong Li, Renmin University of China (xirong@ruc.edu.cn)
* Tiberio Uricchio, University of Florence (tiberio.uricchio@unifi.it)
