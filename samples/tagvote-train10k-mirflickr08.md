# TagVote-train10k-mirflickr08

This example shows how to perform and evaluate the [TagVote](http://dx.doi.org/10.1109/TMM.2009.2030598) method using two provided scripts: [doit/do_tagvote.sh](doit/do_tagvote.sh) and [eval/eval_pickle.sh](eval/eval_pickle.sh). 
For efficiency, we prepare the tutorial using the smallest training and test datasets available in Jingwei, namely train10k and mirflickr08.

## Setup

By default we assume the Jingwei code is placed at `$HOME/github/jingwei` and the training and test data are at `$HOME/VisualSearch`. 
Of course, this setting can be cutomized by modifying two environment variables `SURVEY_CODE` and `SURVEY_DATA` in [start.sh](start.sh) accordingly.

#### Get the Jingwei package

```script
mkdir $HOME/github 
cd $HOME/github
git clone https://github.com/li-xirong/jingwei
source $HOME/github/jingwei/start.sh
```

#### Get train10k and mirflickr08
```script
cd $SURVEY_DATA
wget http://lixirong.net/data/csur2016/train10k-tag.tar.gz
wget http://lixirong.net/data/csur2016/train10k-pygooglenet_bu4k-pool5_7x7_s1.tar.gz
wget http://lixirong.net/data/csur2016/train10k-vggnet16-fc7relu.tar.gz

wget http://lixirong.net/data/csur2016/mirflickr08-anno.tar.gz
wget http://lixirong.net/data/csur2016/mirflickr08-pygooglenet_bu4k-pool5_7x7_s1.tar.gz
wget http://lixirong.net/data/csur2016/mirflickr08-vggnet16-fc7relu.tar.gz

tar xzf train10k-tag.tar.gz
tar xzf train10k-vggnet16-fc7relu.tar.gz
tar xzf train10k-pygooglenet_bu4k-pool5_7x7_s1.tar.gz
tar xzf mirflickr08-anno.tar.gz
tar xzf mirflickr08-vggnet16-fc7relu.tar.gz
tar xzf mirflickr08-pygooglenet_bu4k-pool5_7x7_s1.tar.gz
```

## Perform tag relevance learning

#### Using the 4,096-dim vggnet feature
```script
$HOME/github/jingwei/doit/do_tagvote.sh train10k mirflickr08 vgg-verydeep-16-fc7relu
```

#### Using the 1,024-dim googlenet-bu4k feature
```script
$HOME/github/jingwei/doit/do_tagvote.sh train10k mirflickr08 pygooglenet_bu4k-pool5_7x7_s1
```

In a few minutes, you shall see two result files saved in a pickle format at `$SURVEY_DATA/surveyruns`:

```script
train10k_mirflickr08_vgg-verydeep-16-fc7relu,tagvote.pkl
train10k_mirflickr08_pygooglenet_bu4k-pool5_7x7_s1,tagvote.pkl
```

## Evaluation

Given a list of pkl files, the [eval/do_eval.sh](eval/eval_pickle.sh) script can compute the image-centric Mean image Average Precision (MiAP) to measure the quality of tag
ranking, and the tag-centric Mean Average Precision (MAP) to measure the quality of image ranking. 

Before running the evalution script, a run file that contains the list of pkl files needs to be placed at `$SURVEY_DATA/eval_output`. Also, to simplify the input parameters, the run filename is formated as `runs_$method_$testCollection.txt`.
```script
mkdir $SURVEY_DATA/eval_output
cd $SURVEY_DATA/eval_output
find $SURVEY_DATA/surveyruns | grep -E "tagvote|mirflickr08" > runs_tagvote_mirflickr08.txt
```

```script
$HOME/github/jingwei/eval/eval_pickle.sh mirflickr08 tagvote
```
The performance scores are saved in `$SURVEY_DATA/eval_output/runs_tagvote_mirflickr08.res`.

Following are main numbers taken from the result file, organized with tables for the ease of read. 

#### Method: TagVote, Training: train10k, Test: mirflickr08, Metric: MiAP
| Feature  | MiAP |
| ------------- | -------------: |
| vgg-verydeep-16-fc7relu | 0.355  |
| pygooglenet_bu4k-pool5_7x7_s1 | 0.379 |


#### Method: TagVote, Training: train10k, Test: mirflickr08, Metric: AP
| Concept  | vgg-verydeep-16-fc7relu | pygooglenet_bu4k-pool5_7x7_s1 |
| ------------- | -------------: | ---:  |
| baby   | 0.332 | 0.441
| bird   | 0.761 | 0.811
| car    | 0.813 | 0.861
| cloud  | 0.507 | 0.467
| dog    | 0.849 | 0.867
| flower | 0.782 | 0.856
| girl   | 0.651 | 0.724
| man    | 0.450 | 0.557
| night  | 0.228 | 0.304
| people | 0.827 | 0.880
| portrait | 0.735 | 0.825
| river  | 0.115 | 0.139
| sea    | 0.291 | 0.300
| tree   | 0.512 | 0.686
| meanAP | 0.561 | 0.623
