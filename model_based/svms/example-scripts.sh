
source ~/common.ini
#codepath=/Users/xirong/bitbucket/mlengine
#rootpath=/Users/xirong/VisualSearch


trainCollection=voc2008train
trainAnnotationName='conceptsvoc2008train.txt'

valCollection='voc2008val'
valAnnotationName='conceptsvoc2008val.txt'

testCollection='voc2008val'
testAnnotationName=$valAnnotationName

feature=dsift

minmaxfile=$rootpath/$trainCollection/FeatureData/$feature/minmax.txt
if [ ! -f "$minmaxfile" ]; then
    feat_dir=$rootpath/$trainCollection/FeatureData/$feature
    python $codepath/mlengine/fiksvm/find_min_max.py $feat_dir
fi


for modelName in fastlinear fik
do
    echo "optimize hyper parameters for $modelName"
    python $codepath/mlengine/optimize_hyper_params.py $trainCollection $trainAnnotationName $valCollection $valAnnotationName $feature $modelName
done


python $codepath/mlengine/fastlinear/trainLinearConcepts.py $trainCollection $trainAnnotationName $feature 
best_param_dir=$rootpath/$trainCollection/Models/$trainAnnotationName/fastlinear,best_params/$valCollection,$valAnnotationName,$feature
python $codepath/mlengine/fastlinear/trainLinearConcepts.py $trainCollection $trainAnnotationName $feature --best_param_dir $best_param_dir


python $codepath/mlengine/fiksvm/trainFikConcepts.py $trainCollection $trainAnnotationName $feature 
best_param_dir=$rootpath/$trainCollection/Models/$trainAnnotationName/fik50,best_params/$valCollection,$valAnnotationName,$feature
python $codepath/mlengine/fiksvm/trainFikConcepts.py $trainCollection $trainAnnotationName $feature --best_param_dir $best_param_dir


for modelName in fastlinear fastlinear-tuned fik50 fik50-tuned
do
    modelfile=$rootpath/$trainCollection/Models/$trainAnnotationName/$feature/$modelName/dog.model
    if [ ! -f "$modelfile" ]; then
        echo "$modelfile does not exist!"
        continue
    fi
    python $codepath/mlengine/applyConcepts.py $testCollection $trainCollection $trainAnnotationName $feature $modelName
done


#python ~/myCode/cross-platform/autotagging/evalTagvotes.py $testCollection $testAnnotationName $codepath/mlengine/runs-$testCollection.txt

