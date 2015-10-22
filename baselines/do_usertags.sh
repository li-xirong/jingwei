if [ "$#" -ne 2 ]; then
    echo "Usage: $0 testCollection conceptfile"
    exit
fi

testCollection=$1
concepts=$2
resultpath=/home/urix/surveydbdas

mkdir -p /home/urix/surveydbdas/usertags/$testCollection
for ((i=0;i<100;i++))
do 
	python baselines/usertags.py $testCollection $concepts $resultpath/usertags/$testCollection/$i.pkl --random 1
	echo $resultpath/usertags/$testCollection/$i.pkl >> eval/runs_usertags_$testCollection.txt
done
eval/eval_pickle.sh $testCollection usertags
