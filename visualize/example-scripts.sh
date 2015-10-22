
rootpath=/home/urix/shared/tagrelfeatures
codepath=/home/urix/shared/tagrelcodebase
overwrite=0

collection=flickr81
annotationName=concepts66imagenet.txt
runfile=runs.$collection.txt

conceptfile=$rootpath/$collection/Annotations/$annotationName
tagfile=$rootpath/$collection/TextData/id.userid.lemmtags.txt


for datafile in $conceptfile $tagfile
do

    if [ ! -f "$datafile" ]; then
        echo "$datafile does not exist!"
        exit
    fi
done

python $codepath/imagesearch/obtain_labeled_examples.py $collection $conceptfile --rootpath $rootpath --overwrite $overwrite
python pickle_to_imagerank.py $collection $annotationName $runfile --rootpath $rootpath --overwrite $overwrite > $runfile.log

