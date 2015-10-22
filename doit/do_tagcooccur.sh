rootpath=$SURVEY_DATA
codepath=$SURVEY_CODE

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 trainCollection testCollection tagger feature"
    exit
fi

trainCollection=$1
testCollection=$2
tagger=$3
feature=$4 #vgg-verydeep-16-fc7relu


if [ "$feature" != "color64+dsift" -a "$feature" != "vgg-verydeep-16-fc7relu" ]; then
    echo "unknown feature $feature"
    exit
fi


if [ "$testCollection" == "flickr81" ]; then
    conceptset=concepts81
elif [ "$testCollection" == "flickr51" ]; then
    conceptset=concepts51ms
elif [ "$testCollection" == "mirflickr08" ]; then
    conceptset=conceptsmir14
else
    echo "unknown collection $testCollection"
    exit
fi

annotationName="$conceptset".txt
conceptfile=$rootpath/$trainCollection/Annotations/$annotationName


if [ "$tagger" == "tagcooccur" ]; then
    kc=0
    #tagvotesfile=$rootpath/$testCollection/autotagging/$testCollection/$trainCollection/$annotationName/cotag,m25,kr4,kd11,ks9,kc"$kc",bonus0/id.tagvotes.txt
    tagvotesfile=$rootpath/$testCollection/autotagging/$testCollection/$trainCollection/$annotationName/cotag/id.tagvotes.txt
    pklfile=$SURVEY_DB/"$trainCollection"_"$testCollection"_"$tagger".pkl
elif [ "$tagger" == "tagcooccurplus" ]; then
    kc=1
    #tagvotesfile=$rootpath/$testCollection/autotagging/$testCollection/$trainCollection/$annotationName/cotag,m25,kr4,kd11,ks9,kc"$kc",bonus0/$feature/id.tagvotes.txt
    tagvotesfile=$rootpath/$testCollection/autotagging/$testCollection/$trainCollection/$annotationName/cotag/$feature/id.tagvotes.txt
    pklfile=$SURVEY_DB/"$trainCollection"_"$testCollection"_"$feature","$tagger".pkl
else
    echo "unknown tagger $tagger"
    exit
fi


python $codepath/instance_based/compute_concept_rank_based_on_tagcooccur.py $trainCollection $annotationName

tagrelfile=$SURVEY_DB/"$trainCollection"_"$testCollection"_"$feature",tagvote.pkl
if [ ! -f "$tagrelfile" -a "$tagger" == "tagcooccurplus" ]; then
    echo "$tagrelfile does not exist"
    exit
fi

if [ -f "$tagrelfile" -a "$tagger" == "tagcooccurplus" ]; then
    rankfile=$rootpath/$testCollection/autotagging/"$trainCollection"_"$testCollection"_tagvote,"$feature"_"$conceptset"_rank.pkl
    python $codepath/instance_based/tagrel_to_concept_rank.py $tagrelfile $rankfile
fi

python $codepath/instance_based/apply_tagcooccur.py $trainCollection $annotationName $testCollection --kc $kc --bonus 0 --feature $feature


if [ ! -f "$tagvotesfile" ]; then
    echo "tagvotesfile $tagvotesfile not exists!"
    exit
fi

python $codepath/postprocess/pickle_tagvotes.py $conceptfile $tagvotesfile $pklfile

