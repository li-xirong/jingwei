#!/bin/bash
codepath=/home/urix/shared/tagrelcodebase
datapath=/home/urix/shared/tagrelfeatures

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 trainCollection testCollection feature"
    exit
fi

coll1=$1
coll2=$2
feature=$3

mkdir -p $datapath/$coll1+$coll2/FeatureData/$feature
mkdir -p $datapath/$coll1+$coll2/ImageSets
mkdir -p $datapath/$coll1+$coll2/Annotations
mkdir -p $datapath/$coll1+$coll2/TextData
cat $datapath/$coll1/FeatureData/$feature/feature.bin $datapath/$coll2/FeatureData/$feature/feature.bin > $datapath/$coll1+$coll2/FeatureData/$feature/feature.bin
cat $datapath/$coll1/FeatureData/$feature/id.txt $datapath/$coll2/FeatureData/$feature/id.txt > $datapath/$coll1+$coll2/FeatureData/$feature/id.txt 
cat $datapath/$coll1/ImageSets/$coll1.txt $datapath/$coll2/ImageSets/$coll2.txt > $datapath/$coll1+$coll2/ImageSets/$coll1+$coll2.txt 
cat $datapath/$coll1/TextData/id.userid.lemmtags.txt $datapath/$coll2/TextData/id.userid.lemmtags.txt > $datapath/$coll1+$coll2/TextData/id.userid.lemmtags.txt
cp -r $datapath/$coll1/Annotations $datapath/$coll1+$coll2/

