import sys, os, random

from basic.common import ROOT_PATH
from basic.util import readImageSet
from simpleknn.bigfile import BigFile

if __name__ == '__main__':
    rootpath = ROOT_PATH
    collection = sys.argv[1]
    feature = sys.argv[2]
    
    imset = readImageSet(collection, collection)
    feat_dir = os.path.join(rootpath, collection, 'FeatureData', feature)
    feat_file = BigFile(feat_dir)
   
    imset = random.sample(imset, 50) 
    #imset = imset[:5]
     
    renamed,vectors = feat_file.read(imset)
    for name,vec in zip(renamed,vectors):
        print name, sum(vec), sum(vec[:64]), sum(vec[64:])

        
    

