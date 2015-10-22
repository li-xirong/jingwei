rootpath=$SURVEY_DATA
codepath=$SURVEY_CODE

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 trainCollection testCollection feature"
    exit
fi

trainCollection=$1
testCollection=$2
feature=$3
numjobs=1 #$4
job=1 #$5

#python $codepath/preprocess/index_features_by_tag.py $trainCollection $feature
python $codepath/instance_based/tagranking.py $testCollection $trainCollection $feature --numjobs $numjobs --job $job

$codepath/doit/make_tagrank_runs.sh $trainCollection $testCollection $feature
