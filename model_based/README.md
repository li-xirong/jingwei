
# Model_based methods


## TagProp

TagProp is obtained from the authors [website](http://lear.inrialpes.fr/people/guillaumin/code.php). The main `setup.sh` automatically calls the `setup-tagprop.sh` script that downloads and prepare the code for the execution.
The code is composed by a MATLAB wrapper and a MEX file that performs the actual optimization. The setup script patch the code to optimize TagProp efficiency and then compiles the MEX code.

After setup, TagProp can be launched with:

./[do_tagprop.sh](../doit/do_tagprop.sh) train10k mirflickr08 vgg-verydeep-16-fc7relu


## TagFeature


The basic idea of TagFeature is to enrich image features by adding an extra tag feature. 
As the tag features have to be extracted in advance for all training and test data (see [do\_extract_tagfeat.sh](../doit/do_extract_tagfeat.sh)), the method is computationally expensive.

./[do_tagfeat.sh](../doit/do_tagfeat.sh) train10k mirflickr08 vgg-verydeep-16-fc7relul2

## RelExamples

RelExample exploits positive and negative training examples which are deemed to be more relevant with respect to a test tag. 
Relative positive examples are implemented by [do\_create\_refined_annotation.sh](../doit/do_create_refined_annotation.sh), 
while relevant negatives are iteratively selected by [Negative Bootstrap](negbp.py).


./[do_relexample.sh](../doit/do_relexample.sh) train10k mirflickr08 vgg-verydeep-16-fc7relu

# References

**TagProp**: M. Guillaumin, T. Mensink, J. Verbeek, and C. Schmid. [TagProp: Discriminative metric learning in nearest neighbor models for image auto-annotation](http://dx.doi.org/10.1109/ICCV.2009.5459266). In Proc. of ICCV, 2009.

**TagFeature**: L. Chen, D. Xu, I. Tsang, J. Luo. [Tag-Based Image Retrieval Improved by Augmented Features and Group-Based Refinement](http://dx.doi.org/10.1109/TMM.2012.2187435), IEEE Transactions on Multimedia, 2012

**RelExamples**: X. Li, C. Snoek. [Classifying tag relevance by relevant positive and negative examples](http://dx.doi.org/10.1145/2502081.2502129). In Proc. of ACM Multimedia, 2013



