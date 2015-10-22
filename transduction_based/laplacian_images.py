#!/usr/bin/env python
# encoding: utf-8

import sys, os
import numpy as np
import bisect
import scipy.io
import bisect
import math
import scipy.sparse
from scipy.sparse import coo_matrix

from basic.constant import ROOT_PATH
from basic.common import makedirsforfile, checkToSkip, printStatus
from basic.util import readImageSet, bisect_index
from basic.annotationtable import readConcepts
from util.simpleknn.bigfile import BigFile
from instance_based.tagvote import *

INFO = 'transduction_based.laplacian_images'
DEFAULT_K_RATIO=0.001
DEFAULT_DISTANCE = 'l1'

def _unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

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

    nnName = distance + "knn"
    resultfile = os.path.join(rootpath, workingCollection, 'LaplacianI', workingCollection, '%s,%s,%f'%(feature,nnName,k_ratio), 'laplacianI.mat')

    if checkToSkip(resultfile, overwrite):
        return 0

    workingSet = readImageSet(workingCollection, workingCollection, rootpath)
    workingSet.sort()

    tot_images = len(workingSet)
    printStatus(INFO, '%d images' % (tot_images))

    K_neighs = int(math.floor(len(workingSet) * k_ratio))
    printStatus(INFO, '%d nearest neighbor per image (ratio = %f)' % (K_neighs, k_ratio))

    printStatus(INFO, 'Allocating I,J,V arrays')
    I = np.zeros((K_neighs * tot_images * 2))
    J = np.zeros((K_neighs * tot_images * 2))
    V = np.zeros((K_neighs * tot_images * 2))
    n_entries = 0

    # distances
    printStatus(INFO, 'Starting to fill I,J,V arrays')
    for i in xrange(tot_images):
        try:
            neighbors = _get_neighbors('%s,%s' % (workingCollection, workingSet[i]), rootpath, K_neighs*2, feature, distance)
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
        except ValueError:
            printStatus(INFO, 'ERROR: id_img %s has non-standard format!' % (workingSet[i]))
            sys.exit(1)

        if len(neighbors) < K_neighs:
            printStatus(INFO, 'ERROR: id_img %s has %d < %d neighbors!' % (workingSet[i], len(neighbors), K_neighs))
            sys.exit(1)

        if (i+1) % 1000 == 0:
            printStatus(INFO, '%d / %d done' % (i+1, tot_images))
        for k in xrange(K_neighs):
            if i != int(NNrow[k]): # -1 zero on the diagonal for a later step
                I[n_entries] = i
                J[n_entries] = int(NNrow[k]) # -1
                V[n_entries] = NNDrow[k]
                n_entries += 1
                I[n_entries] = int(NNrow[k]) # -1
                J[n_entries] = i
                V[n_entries] = NNDrow[k]
                n_entries += 1

    I = I[0:n_entries]
    J = J[0:n_entries]
    V = V[0:n_entries]

    printStatus(INFO, 'Removing duplicates')
    ind = np.lexsort((V,J,I))
    I = I[ind]
    J = J[ind]
    V = V[ind]
    a = np.concatenate([I.reshape(1, len(I)), J.reshape(1, len(J))], axis=0).T
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    del a
    _, idx = np.unique(b, return_index=True)
    del b

    I = I[idx]
    J = J[idx]
    V = V[idx]

    printStatus(INFO, 'Computing the final laplacian matrix')
    sigma = np.median(V) ** 2.;
    printStatus(INFO, 'Estimated sigma^2 = %f' % sigma)
    V = np.exp(-V / sigma)

    matrix = coo_matrix((V, (I, J)), shape=(tot_images, tot_images)).tocsr()
    new_diag = matrix.sum(axis=0).T
    V = -V

    I_add = np.zeros((tot_images))
    J_add = np.zeros((tot_images))
    V_add = np.zeros((tot_images))
    for i,v in enumerate(new_diag):
        I_add[i] = i
        J_add[i] = i
        V_add[i] = v

    I = np.append(I, I_add)
    J = np.append(J, J_add)
    V = np.append(V, V_add)

    matrix = coo_matrix((V, (I, J)), shape=(tot_images, tot_images)).tolil()

    printStatus(INFO, 'Saving laplacian matrix to %s' % resultfile)
    makedirsforfile(resultfile)
    scipy.io.savemat(resultfile, {'im_similarity' : matrix, 'sigma' : sigma})


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] workingCollection feature""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--kratio", default=DEFAULT_K_RATIO, type="float", help="ratio of nearest neighbor images with respect to the images of the set (%f)" % DEFAULT_K_RATIO)
    parser.add_option("--distance", default=DEFAULT_DISTANCE, type="string", help="visual distance, can be l1 or l2 (default: %s)" % DEFAULT_DISTANCE)
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="(default: %s)" % ROOT_PATH)

    (options, args) = parser.parse_args(argv)
    if len(args) < 2:
        parser.print_help()
        return 1

    return process(options, args[0], args[1])

if __name__ == "__main__":
    sys.exit(main())
