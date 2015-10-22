#!/usr/bin/env python
# encoding: utf-8

import sys, os, time
import subprocess
import h5py
import cPickle as pickle
import numpy as np
from basic.common import ROOT_PATH, checkToSkip, printStatus, makedirsforfile
from basic.util import readImageSet
from basic.annotationtable import readConcepts

INFO = 'baselines.random'

def process(options, workingCollection, annotationName, outputpkl):
    rootpath = options.rootpath
    overwrite = options.overwrite

    resultfile = os.path.join(outputpkl)
    if checkToSkip(resultfile, overwrite):
        return 0

    concepts = readConcepts(workingCollection, annotationName, rootpath)
    id_images = readImageSet(workingCollection, workingCollection, rootpath)
    tagmatrix = np.random.rand(len(id_images), len(concepts))

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
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="(default: %s)" % ROOT_PATH)   
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1], args[2])

if __name__ == "__main__":
    sys.exit(main())
