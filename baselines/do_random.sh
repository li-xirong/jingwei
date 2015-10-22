if [ "$#" -ne 2 ]; then
    echo "Usage: $0 testCollection conceptfile"
    exit
fi

testCollection=$1
concepts=$2
resultpath=/home/urix/surveydbdas

mkdir -p /home/urix/surveydbdas/random/$testCollection
for ((i=0;i<100;i++))
do 
	python baselines/randomtags.py $testCollection $concepts $resultpath/random/$testCollection/$i.pkl
	echo $resultpath/random/$testCollection/$i.pkl >> eval/runs_random_$testCollection.txt
done
eval/eval_pickle.sh $testCollection random
