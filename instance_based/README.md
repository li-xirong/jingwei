

# Instance_based methods

## SemanticField 

This method measures tag relevance in terms of an averaged semantic similarity between a given tag and the other social tags associated with a specific image.
It is implemented in [semantictagrel.py](semantictagrel.py).

+ linux: ./[do_semfield.sh](../doit/do_semfield.sh) train10k flickr51 avgcos
+ windows: ./[do_semfield.bat](../doit/do_semfield.bat)

## TagCooccur

While both SemanticField and TagCooccur are tag-based, the main difference lies in how they compute the contribution of a specific tag to the test tag's relevance score. Different from SemanticField which uses tag similarities, TagCooccur uses the test tag's rank in the tag ranking list created by sorting all tags in terms of their co-occurrence frequency with the tag in training data. In addition, TagCooccur takes into account the stability of the tag, measured by its frequency. 

+ linux: ./[do_tagcooccur.sh](../doit/do_tagcooccur.sh) train10k mirflickr08 tagcooccur vgg-verydeep-16-fc7relul2
+ windows: ./[do_tagcooccur.bat](../doit/do_tagcooccur.bat)

## TagCooccur+

TagCooccur+ aims to improve TagCooccur by adding the visual content. This is achieved by multiplying the tag relevance function of TagCooccur with a content-based term (see ```TagCooccurPlusTagger._compute_relevance_score``` in [tagcooccur.py](tagcooccur.py).


+ linux: ./[do_tagcooccur.sh](../doit/do_tagcooccur.sh) train10k mirflickr08 tagcooccurplus vgg-verydeep-16-fc7relul2
+ windows: ./[do_tagcooccur.bat](../doit/do_tagcooccur.bat)

## TagRanking


The tag ranking algorithm consists of two steps. Given an image and its tags, the first step produces an initial tag relevance score for each of the tags, obtained by (Gaussian) kernel density estimation on a set of images labeled with each tag, separately. Secondly, a random walk is performed on a tag graph where the edges are weighted by a tag-wise similarity. 

+ linux: ./[do_tagranking.sh](../doit/do_tagranking.sh) train10k mirflickr08 vgg-verydeep-16-fc7relul2


## KNN

The KNN algorithm estimates the relevance of a tag with respect to an image by counting the occurrence frequency of the tag in social annotations of the visual neighbors of the image.
It is implemented as the ```PreKnnTagger``` class in [tagvote.py](tagvote.py).

+ linux: ./[do_knntagrel.sh](../doit/do_knntagrel.sh) train10k mirflickr08 vgg-verydeep-16-fc7relu



## TagVote

The TagVote algorithm estimates the relevance of a tag with respect to an image by counting the occurrence frequency of the tag in social annotations of the visual neighbors of the image.
Different from KNN, TagVote introduces a unique-user constraint on the neighbor set to make the voting results more objective. Each user has at most one image in the neighbor set.
Moreover, TagVote also takes into account tag prior frequency to suppress over frequent tags.
The algorithm is implemented as the ```TagVotetagger``` class (or alternatively ```PreTagVoteTagger``` if the _k_ visual neighbors of each test image has been pre-computed) in  [tagvote.py](tagvote.py).

+ linux: ./[do_tagvote.sh](../doit/do_tagvote.sh) train10k mirflickr08 vgg-verydeep-16-fc7relu
+ windows: ./[do_tagvote.bat](../doit/do_tagvote.bat)


# References

**SemanticField**: S. Zhu, C.-W. Ngo, Y.-G. Jiang, [Sampling and Ontologically Pooling Web Images for Visual Concept Learning](http://dx.doi.org/10.1109/TMM.2012.2190387), IEEE Transactions on Multimedia, 2012

**TagCooccur**: B. Sigurbjornsson, R. van Zwol, [Flickr tag recommendation based on collective knowledge](http://dx.doi.org/10.1145/1367497.1367542), Proceedings of WWW, 2008

**TagRanking***: D. Liu, X.-S. Hua, L. Yang, M. Wang, H.-J. Zhang, [Tag Ranking](http://dx.doi.org/10.1145/1526709.1526757), Proceedings of WWW, 2009

**TagVote**, **TagCooccur+**: X. Li, C. Snoek, M. Worring, [Learning social tag relevance by neighbor voting](http://dx.doi.org/10.1109/TMM.2009.2030598), IEEE Transactions on Multimedia, 2009
