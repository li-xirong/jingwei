
source ~/common.ini

if [ ! -d "$rootpath" ]; then
    echo "rootpath $rootpath does not exit"
    exit
fi


DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

wget www.mmc.ruc.edu.cn/research/negbp/voc2008train-voc2008val.zip

echo "Unzipping..."

unzip voc2008train-voc2008val.zip -d $rootpath && rm -rf voc2008train-voc2008val.zip

echo "Done."

