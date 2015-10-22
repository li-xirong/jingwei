rootpath=$SURVEY_DATA
codepath=$SURVEY_CODE

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 testCollection"
    exit
fi

testCollection=$1

if [ "$testCollection" == "flickr81" ]; then
    conceptset=concepts81
elif [ "$testCollection" == "flickr55" ]; then
    conceptset=concepts51ms
elif [ "$testCollection" == "mirflickr08" ]; then
    conceptset=conceptsmir14
else
    echo "unknown collection $testCollection"
    exit
fi

annotationName="$conceptset".txt
conceptfile=$rootpath/$testCollection/Annotations/$annotationName

python $codepath/instance_based/tagpos.py $testCollection 
    
tagvotesfile=$rootpath/$testCollection/tagrel/$testCollection/tagpos,lemm/id.tagvotes.txt
resultfile=$SURVEY_DB/"$testCollection"_tagpos.pkl
python $codepath/postprocess/pickle_tagvotes.py $conceptfile $tagvotesfile $resultfile
