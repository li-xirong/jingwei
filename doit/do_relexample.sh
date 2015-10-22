rootpath=$SURVEY_DATA
codepath=$SURVEY_CODE

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 trainCollection testCollection feature" # testCollection"
    exit
fi

overwrite=0
trainCollection=$1 #train10k train100k train1m
testCollection=$2 #mirflickr08 flickr51 flickr81
feature=$3
prob_output=1


if [ "$feature" = "color64+dsift" ]; then
    posName=fcswnsiftbc
elif [ "$feature" = "vgg-verydeep-16-fc7relu" -o "$feature" = "vgg-verydeep-16-fc7relul2" ]; then
    posName=fcswncnnbc
else
    echo "unknown feature $feature"
    exit
fi


conceptset=concepts130
baseAnnotationName="$conceptset"social.txt
posNum=500
startAnnotationName="$conceptset"social."$posName""$posNum".random"$posNum".0.txt
modelName=fik
fullModelName=fik50
$codepath/doit/do_create_refined_annotation.sh $trainCollection $feature



minmaxfile=$rootpath/$trainCollection/FeatureData/$feature/minmax.txt

if [ ! -f "$minmaxfile" ]; then 
    feat_dir=$rootpath/train1m/FeatureData/$feature
    python $codepath/model_based/svms/fiksvm/find_min_max.py $feat_dir

    if [ "$trainCollection" != "train1m" ]; then 
        cp $minmaxfile $rootpath/$trainCollection/FeatureData/$feature/
    fi
fi

python $codepath/model_based/negbp.py $trainCollection $baseAnnotationName $startAnnotationName $feature $modelName


modelAnnotationName="$conceptset"social."$posName""$posNum".random"$posNum".0."$fullModelName".top.npr10.T10.txt
trainAnnotationName="$conceptset"social.random"$posNum".0.npr5.0.txt
conceptfile=$rootpath/$trainCollection/Annotations/$trainAnnotationName

python $codepath/model_based/generate_train_bags.py $trainCollection $baseAnnotationName $posNum --neg_pos_ratio 5 --neg_bag_num 1

if [ ! -f "$conceptfile" ]; then
    echo "$conceptfile does not exist"
    exit
fi


python $codepath/model_based/svms/find_ab.py $trainCollection $modelAnnotationName $trainAnnotationName $feature --overwrite $overwrite



if [ "$testCollection" = "mirflickr08" ]; then
    testAnnotationName=conceptsmir14.txt
elif [ "$testCollection" = "flickr51" ]; then
    testAnnotationName=concepts51ms.txt
elif [ "$testCollection" = "flickr81" ]; then
    testAnnotationName=concepts81.txt
else
    echo "unknown testCollection $testCollection"
    exit
fi

python  $codepath/model_based/svms/applyConcepts.py $testCollection $trainCollection $modelAnnotationName $feature $fullModelName --prob_output $prob_output
    
tagvotesfile=$rootpath/$testCollection/autotagging/$testCollection/$trainCollection/$modelAnnotationName/$feature,$fullModelName,prob/id.tagvotes.txt
conceptfile=$rootpath/$testCollection/Annotations/$testAnnotationName
resfile=$SURVEY_DB/"$trainCollection"_"$testCollection"_$feature,relexample.pkl

python $codepath/postprocess/pickle_tagvotes.py $conceptfile $tagvotesfile $resfile

