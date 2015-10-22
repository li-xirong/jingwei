if [ "$#" -ne 4 ]; then
    echo "Usage: $0 trainCollection testCollection conceptfile njobs"
    exit
fi

trainCollection=$1
testCollection=$2
concepts=$3
njobs=$4
resultpath=/home/urix/surveydbdas

parallel --noswap --ungroup --linebuffer --delay 60 -j $njobs python robustpca/robustpca.py --lambda1 {1} --lambda2 {2} --outputonlytest 1 --rawtagmatrix 1 $trainCollection+$testCollection $concepts color64+dsift /home/urix/surveydbdas/robustpca_raw/${trainCollection}/${testCollection}/${trainCollection}_${testCollection}_robustpca_{1}_{2}.pkl ::: 0.015625 0.0625 0.25 2 4 16 ::: 0.0009765625 0.00390625 0.015625 0.0625 0.25 1