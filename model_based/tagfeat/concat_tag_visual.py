import sys
import os
import random
import numpy as np

from basic.constant import ROOT_PATH
from basic.common import makedirsforfile,niceNumber,checkToSkip,printStatus
from basic.util import readImageSet
from util.simpleknn.bigfile import BigFile

INFO = __file__

if __name__ == '__main__':
    rootpath = ROOT_PATH
    collection = sys.argv[1]
    tagfeature = sys.argv[2]
    visualfeature = sys.argv[3]
    overwrite = 0 #int(sys.argv[4])

    srcfeatures = [tagfeature, visualfeature]
    newfeature = '+'.join(srcfeatures)
    feat_dir = os.path.join(rootpath, collection, 'FeatureData', newfeature)
    binary_file = os.path.join(feat_dir, 'feature.bin')
    id_file = os.path.join(feat_dir, 'id.txt')
    shape_file = os.path.join(feat_dir,'shape.txt')

    if checkToSkip(binary_file, overwrite):
        sys.exit(0)

    nr_of_images_list = []
    feat_dim_list = []
    feat_files = []

    for feature in srcfeatures:
        shapefile = os.path.join(rootpath, collection, 'FeatureData', feature, 'shape.txt')
        nr_of_images, feat_dim = map(int, open(shapefile).readline().strip().split())
        nr_of_images_list.append(nr_of_images)
        feat_dim_list.append(feat_dim)
        feat_files.append(BigFile( os.path.join(rootpath, collection, 'FeatureData', feature) ) )

    #assert(nr_of_images_list[0] == nr_of_images_list[1])
    new_feat_dim = sum(feat_dim_list)

    imset = readImageSet(collection, collection, rootpath)
    nr_of_images = len(imset)
    blocksize = 1000

    makedirsforfile(binary_file)
    fw = open(binary_file, 'wb')
    new_imset = []
    start = 0

    while start < nr_of_images:
        end = min(nr_of_images, start + blocksize)
        printStatus(INFO, 'processing images from %d to %d' % (start, end-1))

        renamed_0, vecs_0 = feat_files[0].read(imset[start:end])
        renamed_1, vecs_1 = feat_files[1].read(imset[start:end])

        sorted_idx_0 = np.argsort(renamed_0)
        sorted_idx_1 = np.argsort(renamed_1)
  
        for x,y in zip(sorted_idx_0, sorted_idx_1):
            assert(renamed_0[x] == renamed_1[y])
            vec = np.array(vecs_0[x] + vecs_1[y], dtype=np.float32)
            vec.tofile(fw)
            new_imset.append(renamed_0[x])

        start = end

    fw.close()

    fw = open(id_file, 'w')
    fw.write(' '.join(new_imset))
    fw.close()

    fw = open(shape_file, 'w')
    fw.write('%d %d' % (len(new_imset), new_feat_dim))
    fw.close()

