
codepath=/home/xirong/github/tagrel

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 collection feautre"
    exit
fi

collection=$1
feature=$2


if [ "$feature" = "color64+dsift" ]; then
    distance=l1
elif [ "$feature" = "vgg-verydeep-16-fc7relu" ]; then 
    distance=cosine
else
    echo "unknown feature $feature"
    exit
fi 

python $codepath/dotagrel.py $collection $feature $collection --distance $distance
