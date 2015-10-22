if [ "$#" -ne 1 ]; then
    echo "Usage: $0 surveyrunsdir"
    exit
fi

# most methods
for train in train10k train100k train1m imagenet166
do
  for test in mirflickr08 flickr81 flickr51
  do
    for feature in color64+dsift vgg-verydeep-16-fc7relu
    do
      for method in knn tagvote relexample afsvm tagprop robustpca tagcooccurplus tensoranalysis
      do
        ls $1/$train*$test*$feature*$method* >> $SURVEY_CODE/eval/runs_${method}_${test}.txt 2> /dev/null
      done
    done
  done
done

# tagcooccur
for train in train10k train100k train1m
do
  for test in mirflickr08 flickr81 flickr51
  do
    ls $1/$train*$test*tagcooccur.pkl >> $SURVEY_CODE/eval/runs_tagcooccur_${test}.txt 2> /dev/null
  done
done

# tagpos
for test in mirflickr08 flickr81 flickr51
do
  ls $1/$test*tagpos.pkl > $SURVEY_CODE/eval/runs_tagpos_${test}.txt 2> /dev/null
done
