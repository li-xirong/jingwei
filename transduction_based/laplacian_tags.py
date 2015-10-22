#!/usr/bin/env python
# encoding: utf-8

import sys, os
import numpy as np
import bisect
import scipy.io
import bisect
import math
import h5py
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

from basic.constant import ROOT_PATH
from basic.common import makedirsforfile, checkToSkip, printStatus
from basic.util import readImageSet
from basic.annotationtable import readConcepts
from util.simpleknn.bigfile import BigFile
from instance_based.tagvote import *

INFO = 'transduction_based.laplacian_tags'

DEFAULT_RATIOCS = 0.9

def tag_semantic_similarity(x, y, ic):
    mx = wn.morphy(x)
    my = wn.morphy(y)

    if mx is None or my is None:
        return 0

    synX = wn.synsets(mx, pos=wn.NOUN)
    synY = wn.synsets(my, pos=wn.NOUN)

    if len(synX) > 0 and len(synY) > 0:
        maxSim = synX[0].lin_similarity(synY[0], ic)
    else:
        maxSim = 0

    return maxSim

def process(options, workingCollection):
    rootpath = options.rootpath
    overwrite = options.overwrite
    chunk = options.chunk - 1
    n_chunks = options.nchunks
    ratio_cs = options.ratiocs
    assert chunk < n_chunks and chunk >= 0 and n_chunks > 0

    printStatus(INFO, 'RatioCS = %f' % ratio_cs)

    printStatus(INFO, 'Using Brown Corpus for the ic')
    brown_ic = wordnet_ic.ic('ic-brown.dat')

    tags_file = os.path.join(rootpath, workingCollection, 'TextData', 'lemm_wordnet_freq_tags.h5')
    if not os.path.exists(tags_file):
        printStatus(INFO, 'Tags file not found at %s Did you run wordnet_frequency_tags.py ?' % tags_file)
        sys.exit(1)

    if n_chunks > 1:
        resultfile = os.path.join(rootpath, workingCollection, 'LaplacianT', '%f'%(ratio_cs), 'laplacianT_%d.mat' % chunk)
    else:
        resultfile = os.path.join(rootpath, workingCollection, 'LaplacianT', '%f'%(ratio_cs), 'laplacianT.mat')
    if checkToSkip(resultfile, overwrite):
        return 0

    tags_data = h5py.File(tags_file, 'r')

    vocab = list(tags_data['vocab'][:])
    tagmatrix = tags_data['tagmatrix'][:]
    N_tags = len(vocab)

    # single tag frequency
    frequency = tagmatrix.sum(axis=0)
    assert len(frequency) == len(vocab), "%s " % len(frequency) == len(vocab)

    final_matrix = np.zeros((N_tags, N_tags))

    # similarity matrix
    printStatus(INFO, 'Building the similarity matrix')
    start_chunk = chunk * int(math.floor(N_tags / n_chunks))
    if chunk == (n_chunks - 1):
        end_chunk = N_tags
    else:
        end_chunk = (chunk + 1) * int(math.floor(N_tags / n_chunks))

    for i in xrange(start_chunk, end_chunk):
        if i % 100 == 0:
            printStatus(INFO, '%d / %d done' % (i+1, end_chunk))
        for k in xrange(i+1, N_tags):
            context = ratio_cs * np.sum(tagmatrix[:, [i, k]].sum(axis=1) > 1.5) / (frequency[i] + frequency[k])
            semantic = max(0, (1. - ratio_cs) * tag_semantic_similarity(vocab[i], vocab[k], brown_ic))
            final_matrix[i, k] = context + semantic
            final_matrix[k, i] = final_matrix[i, k]

    # laplacian
    if n_chunks < 2:
        printStatus(INFO, 'Computing the laplacian matrix')
        new_diag = final_matrix.sum(axis=0).T
        final_matrix = - final_matrix
        for i in xrange(N_tags):
            final_matrix[i, i] = new_diag[i]

    if n_chunks < 2:
        printStatus(INFO, 'Saving laplacian matrix to %s' % resultfile)
    else:
        printStatus(INFO, 'Saving partial matrix to %s' % resultfile)
    makedirsforfile(resultfile)
    scipy.io.savemat(resultfile, {'tag_similarity' : final_matrix})

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] workingCollection""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--ratiocs", default=DEFAULT_RATIOCS, type="float", help="ratio of context vs wordnet similarity in tag similarity compitation (default %f)" % DEFAULT_RATIOCS)
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="(default: %s)" % ROOT_PATH)
    parser.add_option("--chunk", default=1, type="int", help="job number (default: 1)")
    parser.add_option("--nchunks", default=1, type="int", help="total number of jobs (default: 1)")

    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1

    return process(options, args[0])

if __name__ == "__main__":
    sys.exit(main())
