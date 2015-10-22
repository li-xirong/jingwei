if [ "$#" -ne 5 ]; then
    echo "Usage: $0 trainCollection testCollection conceptfile feature njobs"
    exit
fi

trainCollection=$1
testCollection=$2
concepts=$3
feature=$4
njobs=$5
resultpath=/home/urix/shared/surveydbdas

if [ "$feature" = "color64+dsift" ]; then
    distance=l1
elif [ "$feature" = "vgg-verydeep-16-fc7relu" ]; then
    distance=cosine
else
    echo "unknown feature $feature"
    exit
fi

# merge collections
echo "Merging collections..."
echo python tools/merge_datasets.py $trainCollection $testCollection vgg-verydeep-16-fc7relu

# setup tag files based on training set vocabulary
echo "Setup tag vocabulary..."
echo cd $SURVEY_DATA/${trainCollection}+${testCollection}
echo python $SURVEY_CODE/tools/wordnet_frequency_tags.py $SURVEY_DATA/${trainCollection}/TextData/lemm_wordnet_freq_tags.h5
echo cd -

# compute laplacian tags
echo "Computing laplacian of tags..."
echo python transduction_based/laplacian_tags.py ${trainCollection}+${testCollection}

# setup tag files based on training set vocabulary
echo "Compute NNs for the merged collection..."
echo parallel instance_based/do_getknn.sh  ${trainCollection}+${testCollection} ${trainCollection}+${testCollection} $feature 0 $njobs {1} ::: `seq $njobs`

# compute laplacian images
echo python transduction_based/laplacian_images.py --distance $distance ${trainCollection}+${testCollection} $feature

# convert robustpca data to matlab format
echo python robustpca/robustpca_preprocessing.py --distance $distance ${trainCollection}+${testCollection} $feature

# start the robustpca process in parallel
echo parallel --noswap --ungroup --delay 10 --linebuffer -j $njobs python robustpca/robustpca.py --distance $distance --lambda1 {1} --lambda2 {2} --outputonlytest 1 $trainCollection+$testCollection $concepts $feature /home/urix/shared/surveydbdas/robustpca/${trainCollection}/${testCollection}/${trainCollection}_${testCollection}_robustpca_{1}_{2}.pkl ::: 0.015625 0.0625 0.25 1 4 16 ::: 0.0009765625 0.00390625 0.015625 0.0625 0.25 1
