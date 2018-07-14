## Downloads ##

|          | Train10k | Train100k | Train1M | MIRFlickr | Flickr51 | NUS-WIDE |
| :---    | ---: |  ---: | ---: | ---: | ---: | ---: | 
| # images | 10,000	| 100,000	| 1,198,818	| 25,000	| 81,541	| 259,233 |
| # tags | - | - | - | 14	| 51	| 81 |
| vggnet16-fc7relu | [link](http://lixirong.net/data/csur2016/train10k-vggnet16-fc7relu.tar.gz) |  [link](http://lixirong.net/data/csur2016/train100k-vggnet16-fc7relu.tar.gz) |  [link](http://lixirong.net/data/csur2016/train1m-vggnet16-fc7relu.tar.gz) | [link](http://lixirong.net/data/csur2016/mirflickr08-vggnet16-fc7relu.tar.gz) | [link](http://lixirong.net/data/csur2016/flickr51-vggnet16-fc7relu.tar.gz) | [link](http://lixirong.net/data/csur2016/flickr81-vggnet16-fc7relu.tar.gz) | 
| social tags | [link](http://lixirong.net/data/csur2016/train10k-tag.tar.gz) | [link](http://lixirong.net/data/csur2016/train100k-tag.tar.gz) | [link](http://lixirong.net/data/csur2016/train1m-tag.tar.gz) | [link](http://lixirong.net/data/csur2016/mirflickr08-tag.tar.gz) | [link](http://lixirong.net/data/csur2016/flickr51-tag.tar.gz) | [link](http://lixirong.net/data/csur2016/flickr81-tag.tar.gz) |
| tag frequency | [link](http://lixirong.net/data/csur2016/train10k-tagfreq.tar.gz) | [link](http://lixirong.net/data/csur2016/train100k-tagfreq.tar.gz) | [link](http://lixirong.net/data/csur2016/train1m-tagfreq.tar.gz) | - | - | - |
| ground truth | - | - | - | [link](http://lixirong.net/data/csur2016/mirflickr08-anno.tar.gz) | [link](http://lixirong.net/data/csur2016/flickr51-anno.tar.gz) | [link](http://lixirong.net/data/csur2016/flickr81-anno.tar.gz) | 
| Flickr image urls | - | - | [link](http://lixirong.net/data/csur2016/train1m-urls.txt.gz) | - | - | [link](http://dl.nextcenter.org/public/nuswide/NUS-WIDE-urls.rar) |


## Vocabulary ##

As what tags a person may use is meant to be open, the need for specifying a tag vocabulary is merely an engineering convenience. For a tag to be meaningfully modeled, there has to be a reasonable amount of training images with respect to that tag. For methods where tags are processed independently from the others, the size of the vocabulary has no impact on the performance. In the other cases, in particular, for transductive methods that rely on the image-tag association matrix, the tag dimension has to be constrained to make the methods runnable. To construct a fixed-size vocabulary per training dataset, a three-step automatic cleaning procedure was performed. First, all the tags are lemmatized to their base forms by the [NLTK](http://www.nltk.org/) software. Second, tags not defined in [WordNet](https://wordnet.princeton.edu/) are removed. Finally, in order to avoid insufficient sampling, we remove tags that cannot meet a threshold on tag occurrence. The thresholds were empirically set as 50, 250, and 750 for Train10k, Train100k, and Train1m, respectively, in order to have a linear increase in vocabulary size versus a logarithmic increase in the number of labeled images. This results in the following three vocabularies, 
+ [train10k_vocab.txt](train10k_vocab.txt): 237 tags
+ [train100k_vocab.txt](train100k_vocab.txt): 419 tags
+ [train1m_vocab.txt](train1m_vocab.txt): 1,549 tags
