#!/usr/bin/env python
# encoding: utf-8

import sys, os, time
import subprocess
import cPickle as pickle
import h5py
import numpy as np
from basic.common import ROOT_PATH, checkToSkip, niceNumber, printStatus, makedirsforfile
from basic.util import readImageSet, getVocabMap, bisect_index
from basic.annotationtable import readConcepts

INFO = 'baselines.usertags'

def process(options, workingCollection, annotationName, outputpkl):
    rootpath = options.rootpath
    overwrite = options.overwrite
    random = options.random

    resultfile = os.path.join(outputpkl)
    if checkToSkip(resultfile, overwrite):
        return 0

    concepts = readConcepts(workingCollection, annotationName, rootpath)
    id_images = readImageSet(workingCollection, workingCollection, rootpath)
    tagmatrix = np.zeros((len(id_images), len(concepts)))

    id_images = []
    tag2idx = dict(zip(concepts, xrange(len(concepts))))
    with open(os.path.join(rootpath, workingCollection, 'TextData', 'id.userid.lemmtags.txt')) as f:
        cnt = 0
        for line in f:
            id_img, _, tags = line.split('\t')
            tags = tags.split()
            if len(tags) > 0:
                tags = [(tag2idx.get(x,-1), y) for x,y in zip(tags, xrange(len(tags)))]
                idx = np.array([x[0] for x in tags])
                vals = 1. / (1. + np.array([x[1] for x in tags]))
                tagmatrix[cnt, idx] = vals

            id_images.append(id_img)
            cnt += 1

    # random rank for untagged images
    if random:
        tagmatrix += np.min(tagmatrix[tagmatrix > 0]) * np.random.rand(tagmatrix.shape[0], tagmatrix.shape[1])

    # save results in pkl format
    printStatus(INFO, "Dump results in pkl format at %s" % resultfile)    
    makedirsforfile(resultfile)
    with open(resultfile, 'w') as f:
        pickle.dump({'concepts':concepts, 'id_images':map(int, id_images), 'scores':tagmatrix}, f, pickle.HIGHEST_PROTOCOL)
    
    
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] workingCollection annotationName outputpkl""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--random", default=0, type="int", help="ranking of not assigned tags are randomized for each image (default=0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="(default: %s)" % ROOT_PATH)   
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1], args[2])

if __name__ == "__main__":
    sys.exit(main())
