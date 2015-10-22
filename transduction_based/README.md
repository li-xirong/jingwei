

# Transduction_based methods

## RobustPCA 

This method factorizes the image-tag matrix **D** by a low rank decomposition with error sparsity.
It is written in MATLAB but also uses KNN and NLTK to build the required image and tag graph Laplacians. The script `do_robustpca.sh` handles the process, it goes from the merge of training and test set, the required precomputation and the final method execution. 

./[do_robustpca.sh](../doit/do_robustpca.sh) train10k mirflickr08 vgg-verydeep-16-fc7relu 2

# References

**RobustPCA**: G. Zhu, S. Yan, and Y. Ma. [Image tag refinement towards low-rank, content-tag prior and error sparsity](http://dx.doi.org/10.1145/1873951.1874028). In Proc. of ACM Multimedia, 2010.
