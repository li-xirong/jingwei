codepath=/home/urix/shared/tagrelcodebase

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 testCollection method"
    exit
fi

testCollection=$1
method=$2
runfile=$codepath/eval/runs_"$method"_"$testCollection".txt
resfile=$codepath/eval/runs_"$method"_"$testCollection".res
outDir=$codepath/eval/hdf5/

if [ -f "$resfile" ]; then
    echo "result file exists at $resfile"
    exit
fi

if [ ! -f "$runfile" ]; then
    echo "runfile $runfile not exists!"
    exit
fi

if [ "$testCollection" == "flickr81" ]; then
    annotationName=concepts81.txt
elif [ "$testCollection" == "mirflickr08" ]; then
    annotationName=conceptsmir14.txt
elif [ "$testCollection" == "flickr51" ]; then
    annotationName=concepts51.txt
else
    echo "unknown testCollection $testCollection"
    exit
fi

python $codepath/tools/eval_pickle_save.py $testCollection $annotationName $runfile $outDir > $resfile
