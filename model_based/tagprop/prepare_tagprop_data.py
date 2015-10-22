#!/usr/bin/env python
# encoding: utf-8

import sys, os
import numpy as np
import bisect
import h5py

from basic.common import makedirsforfile, checkToSkip, printStatus
from basic.constant import ROOT_PATH
from basic.util import readImageSet, bisect_index
from basic.annotationtable import readConcepts
from util.simpleknn.bigfile import BigFile
from instance_based.tagvote import *

INFO = 'tagprop.prepare_tagprop_data'
DEFAULT_K=1000
DEFAULT_DISTANCE = 'cosine'

def process(options, testCollection, trainCollection, annotationName, feature):
    rootpath = options.rootpath
    k = options.k
    distance = options.distance
    overwrite = options.overwrite
    testset = testCollection
    onlytest = options.onlytest
    
    nnName = distance + "knn"
    resultfile_train = os.path.join(rootpath, trainCollection, 'TagProp-data', trainCollection, '%s,%s,%d'%(feature,nnName,k), 'nn_train.h5')
    resultfile_test = os.path.join(rootpath, testCollection, 'TagProp-data', testset, trainCollection, annotationName, '%s,%s,%d'%(feature,nnName,k), 'nn_test.h5')
    
    if (not onlytest and checkToSkip(resultfile_train, overwrite)) or checkToSkip(resultfile_test, overwrite):
        return 0

    testSet = readImageSet(testCollection, testset, rootpath)
    trainSet = readImageSet(trainCollection, trainCollection, rootpath)
    testSet.sort()
    trainSet.sort()

    #train_feat_dir = os.path.join(rootpath, trainCollection, 'FeatureData', feature)
    #train_feat_file = BigFile(train_feat_dir)

    tagger = NAME_TO_TAGGER["preknn"](trainCollection, annotationName, feature, distance, rootpath=rootpath, k=1001)

    printStatus(INFO, '%d test images, %d train images' % (len(testSet), len(trainSet)))

    # allocate train -> train nearest neighbors
    if not onlytest:
        printStatus(INFO, 'Allocating NN, NND matrices')    
        NN = np.zeros((len(trainSet), k+1), dtype=np.int32)
        NND = np.zeros((len(trainSet), k+1))

        printStatus(INFO, 'Filling NN, NND matrices')    
        for i,id_img in enumerate(trainSet):
            neighbors = tagger._get_neighbors(content=None, context='%s,%s' % (trainCollection, id_img))
            if len(neighbors) < k+1:
                printStatus(INFO, 'ERROR: id_img %s has %d < %d neighbors!' % (id_img, len(neighbors), k+1))    
                sys.exit(1)

            NNrow = np.array([bisect_index(trainSet, x[0]) for x in neighbors])
            NNDrow = np.array([x[1] for x in neighbors])

            NN[i,:] = NNrow[0:k+1]
            NND[i,:] = NNDrow[0:k+1]

            if i % 1000 == 0:
                printStatus(INFO, '%d / %d images' % (i, len(trainSet)))    

        printStatus(INFO, 'Saving train matrices to file %s' % (resultfile_train))
        makedirsforfile(resultfile_train)
        fout = h5py.File(resultfile_train, 'w')
        fout['NN'] = NN
        fout['NND'] = NND
        fout['trainSet'] = trainSet
        fout['concepts'] = tagger.concepts
        fout.close()

        del NN
        del NND
   
    # allocate test -> train nearest neighbors
    printStatus(INFO, 'Allocating NNT, NNDT matrices')        
    NNT = np.zeros((len(testSet), k), dtype=np.int32)
    NNDT = np.zeros((len(testSet), k))

    printStatus(INFO, 'Filling NNT, NNDT matrices')    
    for i,id_img in enumerate(testSet):
        neighbors = tagger._get_neighbors(content=None, context='%s,%s' % (testCollection, id_img))
        if len(neighbors) < k:
            printStatus(INFO, 'ERROR: id_img %s has %d < %d neighbors!' % (id_img, len(neighbors), k))    
            sys.exit(1)

        NNrow = np.array([bisect_index(trainSet, x[0]) for x in neighbors])
        NNDrow = np.array([x[1] for x in neighbors])

        NNT[i,:] = NNrow[0:k]
        NNDT[i,:] = NNDrow[0:k]

        if i % 1000 == 0:
            printStatus(INFO, '%d / %d images' % (i, len(testSet)))    
   
    printStatus(INFO, 'Saving test matrices to file %s' % (resultfile_test))
    makedirsforfile(resultfile_test)
    fout = h5py.File(resultfile_test, 'w')
    fout['NNT'] = NNT
    fout['NNDT'] = NNDT
    fout['trainSet'] = trainSet
    fout['testSet'] = testSet
    fout['concepts'] = tagger.concepts   
    fout.close()
    
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] testCollection trainCollection annotationName feature""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--onlytest", default=0, type="int", help="skip preparing train collection (default=0)")
    parser.add_option("--k", default=DEFAULT_K, type="int", help="number of neighbors (%d)" % DEFAULT_K)
    parser.add_option("--distance", default=DEFAULT_DISTANCE, type="string", help="visual distance, can be l1, l2 or cosine (default: %s)" % DEFAULT_DISTANCE)
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="(default: %s)" % ROOT_PATH)    
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 4:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1], args[2], args[3])

if __name__ == "__main__":
    sys.exit(main())

