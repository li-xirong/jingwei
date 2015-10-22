#rootpath=/local/xirong/VisualSearch
#codepath=/home/xirong/tagrelcodebase

rootpath=/home/urix/tagrelfeatures
codepath=/home/urix/tagrelcodebase

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 collection"
    exit
fi

collection=$1
features=color64l1,dsift
newfeature=color64+dsift

python $codepath/tools/concat_features.py $collection $features --newfeature $newfeature

