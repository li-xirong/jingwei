The matlab file [extract_vggnet.m](extract_vggnet.m) uses the MatConvNet toolbox to extract VGGNet features. 

Given a collection, say `train10k`, the code reads image paths from a text file `train10k.txt` in a precise folder hierarchy as follows:

```
$SURVEY_DATA
---- train10k
-------- ImageData # this folder contains all image files of train10k
------------ train10k.txt # this contains absolute paths of all image files in ImageData
```

The file `train10k.txt` can be generated using 
```
find $SURVEY_DATA/train10k/ImageData | grep .jpg > $SURVEY_DATA/train10k/ImageData/train10k.txt
```

The code has been tested on a GPU with 12 GB memory. It process images in chunks. For GPUs with less memory, the parameter `chunk_size` should be decreased accordingly.

### Dependencies ###

Download the MatConvNet toolbox and a pretrained 16-layer VGGNet model, and unzip them in the same folder as the script.

```
wget http://lixirong.net/data/csur2016/matconvnet-1.0-beta8.tar.gz
wget http://lixirong.net/data/csur2016/matconvnet-models.tar.gz
```


#### Reference ####

[1] A. Vedaldi and K. Lenc, [MatConvNet - Convolutional Neural Networks for MATLAB](http://www.vlfeat.org/matconvnet/), ACMMM 2015
