rootpath=$SURVEY_DATA
codepath=$SURVEY_CODE

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 trainCollection feature"
    exit
fi

overwrite=0
trainCollection=$1
vobsize=400
modelName=fastlinear
feature=$2

if [ "$feature" != "color64+dsift" -a "$feature" != "vgg-verydeep-16-fc7relul2" ]; then
    echo "unknown feature $feature"
    exit
fi

annotationName=concepts"$trainCollection"top"$vobsize".txt
newAnnotationName=concepts"$trainCollection"top"$vobsize"social.txt

nr_pos=500
neg_pos_ratio=1
nr_neg=$(($nr_pos * $neg_pos_ratio))
nr_pos_bags=1
nr_neg_bags=5
pos_end=$(($nr_pos_bags - 1))
neg_end=$(($nr_neg_bags - 1))

python $codepath/preprocess/selectToptags.py $trainCollection $vobsize
python $codepath/util/imagesearch/obtain_labeled_examples.py $trainCollection $rootpath/$trainCollection/Annotations/$annotationName 
python $codepath/util/tagsim/expand_tags.py $trainCollection $annotationName
python $codepath/model_based/dataengine/createSocialAnnotations.py $trainCollection $annotationName 
python $codepath/model_based/generate_train_bags.py $trainCollection $newAnnotationName $nr_pos --neg_pos_ratio $neg_pos_ratio --neg_bag_num $nr_neg_bags

bagfile=$rootpath/$trainCollection/annotationfiles/concepts"$trainCollection"top"$vobsize"social.random$nr_pos.0-$pos_end.npr"$neg_pos_ratio".0-$neg_end.txt
if [ ! -f "$bagfile" ]; then
    echo "$bagfile does not exist"
    exit
fi

python $codepath/model_based/negative_bagging.py $trainCollection $bagfile $feature $modelName



modelAnnotationName=concepts"$trainCollection"top"$vobsize"social.random$nr_pos.0-$pos_end.npr"$neg_pos_ratio".0-$neg_end.txt
trainAnnotationName=concepts"$trainCollection"top"$vobsize"social.random"$nr_pos".0.npr5.0.txt
conceptfile=$rootpath/$trainCollection/Annotations/$trainAnnotationName

python $codepath/model_based/generate_train_bags.py $trainCollection $newAnnotationName $nr_pos --neg_pos_ratio 5 --neg_bag_num 1

if [ ! -f "$conceptfile" ]; then
    echo "$conceptfile does not exist"
    exit
fi


python $codepath/model_based/svms/find_ab.py $trainCollection $modelAnnotationName $trainAnnotationName $feature --model $modelName 


if [ ! -f "$rootpath/$trainCollection/Annotations/$modelAnnotationName" ]; then
    echo "Tag models *** $modelAnnotationName *** not ready. Stop test."
    exit
fi

for testCollection in mirflickr08 flickr51 flickr81
do
    python $codepath/model_based/svms/applyConcepts_s.py $testCollection $trainCollection $modelAnnotationName $feature $modelName --prob_output 1
    
    tagrelMethod=$modelAnnotationName/$feature,$modelName,prob
    tagvotesfile=$rootpath/$testCollection/autotagging/$testCollection/$trainCollection/$tagrelMethod/id.tagvotes.txt
    if [ ! -f "$tagvotesfile" ]; then
        echo "$tagvotesfile does not exist"
        exit
    fi
    tagfeature=tag"$vobsize"-"$trainCollection"
    python $codepath/model_based/tagfeat/tagvotes2feature.py $testCollection $trainCollection $newAnnotationName $tagrelMethod $tagfeature 
    python $codepath/model_based/tagfeat/concat_tag_visual.py $testCollection $tagfeature $feature
done

