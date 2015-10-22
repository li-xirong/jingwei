
rootpath=$SURVEY_DATA
codepath=$SURVEY_CODE

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 trainCollection feature"
    exit
fi

overwrite=0
trainCollection=$1
feature=$2 #vgg-verydeep-16-fc7relu
conceptset=concepts130
annotationName=$conceptset.txt
socialAnnotationName="$conceptset"social.txt

if [ "$feature" = "color64+dsift" ]; then
    distance=l1
    posName=fcswnsiftbc
elif [ "$feature" = "vgg-verydeep-16-fc7relu" ]; then
    distance=cosine
    posName=fcswncnnbc
else
    echo "unknown feature $feature"
    exit
fi

tagsimMethod=fcs
semantic="$tagsimMethod"-wn
visual=$feature,"$distance"knn,1000,lemm


$codepath/doit/do_semfield.sh $trainCollection $trainCollection $tagsimMethod


conceptfile=$rootpath/$trainCollection/Annotations/$annotationName
tagfile=$rootpath/$trainCollection/TextData/id.userid.lemmtags.txt

semfile=$rootpath/$trainCollection/tagrel/$trainCollection/$trainCollection/$semantic/id.tagvotes.txt
visfile=$rootpath/$trainCollection/tagrel/$trainCollection/$trainCollection/$visual/id.tagvotes.txt

runfile=$codepath/data/"$semantic"_"$feature"_"$trainCollection".txt
newRunName=tagged,lemm/$trainCollection/"$semantic"_"$feature"_borda


for datafile in $conceptfile $tagfile $semfile $visfile $runfile
do

    if [ ! -f "$datafile" ]; then
        echo "$datafile does not exist!"
        exit
    fi
done

python $codepath/util/imagesearch/obtain_labeled_examples.py $trainCollection $rootpath/$trainCollection/Annotations/$annotationName --overwrite $overwrite
python $codepath/util/tagsim/expand_tags.py $trainCollection $annotationName
python $codepath/model_based/dataengine/createSocialAnnotations.py $trainCollection $annotationName --overwrite $overwrite

for tagrelMethod in $semantic $visual
do
    python $codepath/util/imagesearch/sortImages.py $trainCollection $annotationName tagrel $trainCollection/$tagrelMethod  --overwrite $overwrite
done


python $codepath/util/imagesearch/combineImageRanking.py $trainCollection $socialAnnotationName $runfile $newRunName --torank 1 --overwrite $overwrite
python $codepath/model_based/dataengine/createRefinedAnnotations.py $trainCollection $socialAnnotationName $newRunName $posName --overwrite $overwrite

