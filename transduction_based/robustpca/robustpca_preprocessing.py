#!/usr/bin/env python
# encoding: utf-8

import sys, os
import numpy as np
import bisect
import scipy.io
import math
import h5py

from basic.constant import ROOT_PATH
from basic.common import makedirsforfile, checkToSkip, printStatus
from basic.util import readImageSet, bisect_index
from basic.annotationtable import readConcepts
from util.simpleknn.bigfile import BigFile
from instance_based.tagvote import *

INFO = 'robustpca.preprocessing'
DEFAULT_LAPLACIAN_K_RATIO = 0.001
DEFAULT_K_RATIO = 0.1
DEFAULT_DISTANCE = 'l1'

def _get_neighbors(context, rootpath, k, feature, distance):
    testCollection,testid = context.split(',')
    knndir = os.path.join(testCollection, '%s,%sknn,1500' % (feature, distance))
    knnfile = os.path.join(rootpath, testCollection, 'SimilarityIndex', testCollection, knndir, testid[-2:], '%s.txt' % testid)
    knn = readRankingResults(knnfile)
    knn = knn[:k]
    return knn

def process(options, workingCollection, feature):
    rootpath = options.rootpath
    k_ratio = options.kratio
    distance = options.distance
    overwrite = options.overwrite
    laplaciankratio = options.laplaciankratio

    nnName = distance + "knn"
    resultfile = os.path.join(rootpath, workingCollection, 'RobustPCA', '%s,%s,%f'%(feature,nnName,k_ratio), 'tagmatrix.h5')

    if checkToSkip(resultfile, overwrite):
        return 0

    tagmatrix_file = os.path.join(rootpath, workingCollection, 'TextData', "lemm_wordnet_freq_tags.h5")
    if not os.path.exists(tagmatrix_file):
        printStatus(INFO, 'Tagmatrix file not found in %s Did you run wordnet_frequency_tags.py?' % (tagmatrix_file))
        sys.exit(1)

    laplacianI_file = os.path.join(rootpath, workingCollection, 'LaplacianI', workingCollection, '%s,%s,%f'%(feature,nnName,laplaciankratio), 'laplacianI.mat')
    if not os.path.exists(laplacianI_file):
        printStatus(INFO, 'LaplacianI file not found in %s Did you run laplacian_images.py?' % (laplacianI_file))
        sys.exit(1)

    tagmatrix_data = h5py.File(tagmatrix_file, 'r')
    tagmatrix = tagmatrix_data['tagmatrix'][:]
    printStatus(INFO, 'tagmatrix.shape = %s' % (str(tagmatrix.shape)))

    laplacian_data = scipy.io.loadmat(laplacianI_file)
    sigma = laplacian_data['sigma']
    printStatus(INFO, 'Sigma^2 = %f' % (sigma))

    workingSet = readImageSet(workingCollection, workingCollection, rootpath)
    workingSet.sort()
    #print map(int, workingSet)[0:10], map(int, list(tagmatrix_data['id_images'][:])[0:10])
    #assert(np.all(map(int, workingSet) == list(tagmatrix_data['id_images'][:])))
    assert(np.all(workingSet == list(tagmatrix_data['id_images'][:])))

    tot_images = len(workingSet)
    printStatus(INFO, '%d images in %s' % (tot_images, workingCollection))

    printStatus(INFO, 'Mean images per tag = %f' % (np.mean(tagmatrix.sum(axis=0))))
    K_neighs = int(math.floor(np.mean(tagmatrix.sum(axis=0)) * k_ratio))
    printStatus(INFO, '%d nearest neighbor per image (ratio = %f)' % (K_neighs, k_ratio))

    printStatus(INFO, 'Starting the propagation pre-processing')
    tagmatrix_new = np.zeros(tagmatrix.shape)
    for i in xrange(tot_images):
        neighbors = _get_neighbors('%s,%s' % (workingCollection, workingSet[i]), rootpath, K_neighs * 2, feature, distance)

        #NNrow = np.array([bisect_index(workingSet, x[0]) for x in neighbors])
        #NNDrow = np.array([x[1] for x in neighbors])

        # remove images with features but not in the working set
        NNrow = []
        NNDrow = []
        new_neighs = []
        for x in neighbors:
            try:
                NNrow.append(bisect_index(workingSet, x[0]))
                NNDrow.append(x[1])
                new_neighs.append(x)
            except ValueError:
                pass
        #NNrow = np.array([bisect_index(workingSet, x[0]) for x in neighbors])
        #NNDrow = np.array([x[1] for x in neighbors])
        NNrow = np.array(NNrow)
        NNDrow = np.array(NNDrow)
        neighbors = new_neighs[0:K_neighs]
        
        C = np.sum(np.exp(-(NNDrow)/sigma))
        tagmatrix_new[i,:] = np.sum((np.exp(-(NNDrow)/sigma).T * tagmatrix[NNrow]) / C, axis=0);

        if (i+1) % 1000 == 0:
            printStatus(INFO, '%d / %d done' % (i+1, tot_images))

    # save output
    printStatus(INFO, 'Saving propagated tagmatrix to %s' % resultfile)
    makedirsforfile(resultfile)
    fout = h5py.File(resultfile, 'w')
    fout['tagmatrix'] = tagmatrix_new
    fout['vocab'] = tagmatrix_data['vocab'][:]
    fout['id_images'] = workingSet
    fout.close()


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] workingCollection feature""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--kratio", default=DEFAULT_K_RATIO, type="float", help="ratio of nearest neighbor images with respect to the images of the set (%f)" % DEFAULT_K_RATIO)
    parser.add_option("--laplaciankratio", default=DEFAULT_LAPLACIAN_K_RATIO, type="float", help="laplacianI kratio to be loaded (%f)" % DEFAULT_LAPLACIAN_K_RATIO)
    parser.add_option("--distance", default=DEFAULT_DISTANCE, type="string", help="visual distance, can be l1 or l2 (default: %s)" % DEFAULT_DISTANCE)
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="(default: %s)" % ROOT_PATH)

    (options, args) = parser.parse_args(argv)
    if len(args) < 2:
        parser.print_help()
        return 1

    return process(options, args[0], args[1])

if __name__ == "__main__":
    sys.exit(main())
