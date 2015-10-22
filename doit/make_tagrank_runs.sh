rootpath=$SURVEY_DATA
codepath=$SURVEY_CODE

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 trainCollection testCollection feature"
    exit
fi

overwrite=0
trainCollection=$1
testCollection=$2
feature=$3

k=1000
engine=tagrank
doRandomwalk=1
uniqueUser=0
#engineparams=$trainCollection/$feature,tagrank$doRandomwalk$uniqueUser,$k,lemm
engineparams=$trainCollection/$feature,tagrank,lemm

if [ "$testCollection" == "flickr81" ]; then
    annotationName=concepts81.txt
elif [ "$testCollection" == "flickr51" ]; then
    annotationName=concepts51ms.txt
else
    echo "unknown test collection $testCollection"
    exit
fi    


tagvotesfile=$rootpath/$testCollection/tagrel/$testCollection/$engineparams/id.tagvotes.txt
if [ ! -f "$tagvotesfile" ]; then
    echo "$tagvotesfile does not exist!"
    exit
fi
    
python $codepath/util/imagesearch/countRawTagNum.py $testCollection
python $codepath/util/imagesearch/sortImages.py $testCollection $annotationName $engine $engineparams --overwrite $overwrite

simdir=$rootpath/$testCollection/SimilarityIndex/$testCollection/tagged,lemm/tagrank/$engineparams
resultfile=$SURVEY_DB/"$trainCollection"_"$testCollection"_$feature,tagranking.pkl

python $codepath/postprocess/pickle_imagerank.py $testCollection $annotationName $simdir $resultfile  --overwrite $overwrite

