
import sys, os, random, math, numpy as np
from basic.common import ROOT_PATH
from basic.util import readImageSet
from simpleknn.bigfile import BigFile

if __name__ == '__main__':
    rootpath = ROOT_PATH
    #collection = 'train100k'
    #feature = 'color64+dsift'
    
    collection = sys.argv[1]
    feature = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv)>=4 else 1000
    
    imset = readImageSet(collection, collection, rootpath)
    feat_dir = os.path.join(rootpath, collection, 'FeatureData', feature)
    feat_file = BigFile(feat_dir)
    feat_dim = feat_file.ndims
    
    imset = random.sample(imset, n)
    renamed, vectors = feat_file.read(imset)
    n = len(renamed)

    result = []

    for i in range(n-1):
        x = vectors[i]
        for j in range(i+1, n):
            y = vectors[j]
            d = math.sqrt(sum([ (x[k]-y[k])**2 for k in range(feat_dim) ]))
            result.append(d)

    print 'mean', np.mean(result)
    print 'median', np.median(result)
    print 'max', np.max(result)
    print 'min', np.min(result)
    
                





