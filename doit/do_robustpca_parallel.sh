rootpath=$SURVEY_DATA
codepath=$SURVEY_CODE

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 trainCollection testCollection feature njobs"
    exit
fi

trainCollection=$1
testCollection=$2
feature=$3
njobs=$4
resultpath=$SURVEY_DB

if [ "$feature" = "color64+dsift" ]; then
    distance=l1
elif [ "$feature" = "vgg-verydeep-16-fc7relu" ]; then
    distance=cosine
else
    echo "unknown feature $feature"
    exit
fi

if [ "$testCollection" == "flickr81" ]; then
    testAnnotationName=concepts81.txt
elif [ "$testCollection" == "flickr51" ]; then
    testAnnotationName=concepts51ms.txt
elif [ "$testCollection" == "mirflickr08" ]; then
    testAnnotationName=conceptsmir14.txt
else
    echo "unknown testCollection $testCollection"
    exit
fi

mergedir=$rootpath/${trainCollection}+${testCollection}/
if [ ! -e "$mergedir" ]; then
    # merge collections
    echo "Merging collections..."
    python tools/merge_datasets.py $trainCollection $testCollection vgg-verydeep-16-fc7relu
fi

# setup tag files based on training set vocabulary
tagsh5file=$rootpath/${trainCollection}+${testCollection}/TextData/lemm_wordnet_freq_tags.h5
if [ ! -e "$tagsh5file" ]; then
    echo "Setup tag vocabulary..."
    cd $SURVEY_DATA/${trainCollection}+${testCollection}
    python $SURVEY_CODE/tools/wordnet_frequency_tags.py $SURVEY_DATA/${trainCollection}/TextData/lemm_wordnet_freq_tags.h5
    cd -
fi

# compute laplacian tags
echo "Computing laplacian of tags..."
python transduction_based/laplacian_tags.py ${trainCollection}+${testCollection}

# setup tag files based on training set vocabulary
echo "Compute NNs for the merged collection..."
parallel doit/do_getknn.sh  ${trainCollection}+${testCollection} ${trainCollection}+${testCollection} $feature 0 $njobs {1} ::: `seq $njobs`

# compute laplacian images
python transduction_based/laplacian_images.py --distance $distance ${trainCollection}+${testCollection} $feature

# convert robustpca data to matlab format
python transduction_based/robustpca/robustpca_preprocessing.py --distance $distance ${trainCollection}+${testCollection} $feature

# start the robustpca process in parallel
#parallel --noswap --ungroup --delay 10 --linebuffer -j $njobs python transduction_based/robustpca/robustpca.py --distance $distance --lambda1 {1} --lambda2 {2} --outputonlytest 1 $trainCollection+$testCollection $testAnnotationName $feature ${resultpath}/robustpca/${trainCollection}/${testCollection}/${trainCollection}_${testCollection}_robustpca_{1}_{2}.pkl ::: 0.015625 0.0625 0.25 1 4 16 ::: 0.0009765625 0.00390625 0.015625 0.0625 0.25 1

# start the robustpca process
python transduction_based/robustpca/robustpca.py --distance $distance --lambda1 1 --lambda2 0.0009765625 --outputonlytest 1 $trainCollection+$testCollection $testAnnotationName $feature ${resultpath}/${trainCollection}_${testCollection}_robustpca_1_0.0009765625.pkl
