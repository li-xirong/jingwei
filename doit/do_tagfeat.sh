rootpath=$SURVEY_DATA
codepath=$SURVEY_CODE


if [ "$#" -ne 3 ]; then
    echo "Usage: $0 trainCollection testCollection vis_feature"
    exit
fi

overwrite=0
do_training=1
modelName=fastlinear

trainCollection=$1
testCollection=$2
vis_feature=$3


if [ "$vis_feature" != "color64+dsift" -a "$vis_feature" != "vgg-verydeep-16-fc7relul2" ]; then
    echo "unknown visual feature $vis_feature"
    exit
fi

feature=tag400-$trainCollection+$vis_feature


if [ $do_training == 1 ]; then
    feat_dir=$rootpath/$trainCollection/FeatureData/$feature
    if [ ! -d "$feat_dir" ]; then
        echo "$feat_dir does not exist"
        #$codepath/doit/do_extract_tagfeat.sh $trainCollection $vis_feature
        exit
    fi
fi

conceptset=concepts130social
baseAnnotationName=$conceptset.txt

conceptfile=$rootpath/$trainCollection/Annotations/$baseAnnotationName
if [ ! -f "$conceptfile" ]; then
    echo "$conceptfile does not exist"
    exit
fi

nr_pos=500
neg_pos_ratio=1
nr_neg=$(($nr_pos * $neg_pos_ratio))
nr_pos_bags=1
nr_neg_bags=5
pos_end=$(($nr_pos_bags - 1))
neg_end=$(($nr_neg_bags - 1))

modelAnnotationName=$conceptset.random$nr_pos.0-$pos_end.npr"$neg_pos_ratio".0-$neg_end.txt
trainAnnotationName=$conceptset.random$nr_pos.0.npr5.0.txt

if [ $do_training == 1 ]; then
    python $codepath/model_based/generate_train_bags.py $trainCollection $baseAnnotationName $nr_pos --neg_pos_ratio $neg_pos_ratio --neg_bag_num $nr_neg_bags

    bagfile=$rootpath/$trainCollection/annotationfiles/$conceptset.random$nr_pos.0-$pos_end.npr"$neg_pos_ratio".0-$neg_end.txt
    if [ ! -f "$bagfile" ]; then
        echo "$bagfile does not exist"
        exit
    fi

    python $codepath/model_based/negative_bagging.py $trainCollection $bagfile $feature $modelName

    python $codepath/model_based/generate_train_bags.py $trainCollection $baseAnnotationName $nr_pos --neg_pos_ratio 5 --neg_bag_num 1

    conceptfile=$rootpath/$trainCollection/Annotations/$trainAnnotationName

    if [ ! -f "$conceptfile" ]; then
        echo "$conceptfile does not exist"
        exit
    fi

    python $codepath/model_based/svms/find_ab.py $trainCollection $modelAnnotationName $trainAnnotationName $feature --model $modelName 
fi


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

python $codepath/model_based/svms/applyConcepts_s.py $testCollection $trainCollection $modelAnnotationName $feature $modelName --prob_output 1

tagvotesfile=$rootpath/$testCollection/autotagging/$testCollection/$trainCollection/$modelAnnotationName/$feature,$modelName,prob/id.tagvotes.txt
conceptfile=$rootpath/$testCollection/Annotations/$testAnnotationName
resfile=$SURVEY_DB/"$trainCollection"_"$testCollection"_$vis_feature,tagfeat.pkl

python $codepath/postprocess/pickle_tagvotes.py $conceptfile $tagvotesfile $resfile
