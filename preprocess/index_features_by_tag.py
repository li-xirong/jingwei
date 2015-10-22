import sys
import os
import time
import math
import numpy as np

from basic.constant import ROOT_PATH
from basic.common import checkToSkip,printStatus,makedirsforfile,niceNumber
from util.simpleknn.bigfile import BigFile
    
DEFAULT_TPP = 'lemm'
INFO = __file__

def buildHitLists(collection, tpp='lemm', rootpath=ROOT_PATH):
    vobfile = os.path.join(rootpath, collection, 'TextData', 'wn.%s.txt' % collection)
    vob = set(map(str.strip, open(vobfile).readlines()))
    
    printStatus(INFO, '%s, %d unique tags' % (collection, len(vob)))
    
    tagfile = os.path.join(rootpath, collection, 'TextData', 'id.userid.%stags.txt'%tpp) 
    hitlists = {}
    for line in open(tagfile).readlines():
        elems = line.strip().split()
        name = elems[0]
        tagset = set(elems[2:]).intersection(vob)
        for tag in tagset:
            hitlists.setdefault(tag,[]).append(name)
    assert(len(hitlists)<=len(vob))
    return hitlists        
    

def process(options, collection, feature):
    rootpath = options.rootpath 
    tpp = options.tpp 
    k = 1000 #options.k
    numjobs = options.numjobs
    job = options.job
    overwrite = options.overwrite
    
    feat_dir = os.path.join(rootpath, collection, 'FeatureData', feature)
    feat_file = BigFile(feat_dir)
    hitlists = buildHitLists(collection, tpp, rootpath)
    printStatus(INFO, 'nr of tags: %d' % len(hitlists))
    
    vob = sorted(hitlists.keys())
    vob = [vob[i] for i in range(len(vob)) if i%numjobs == job-1]
    printStatus(INFO, 'working on %d-%d: %d tags' % (numjobs, job, len(vob)))
    
    for tag_idx,tag in enumerate(vob):
        resultdir = os.path.join(rootpath, collection, 'FeatureIndex', feature, tag[:2], tag)
        binfile = os.path.join(resultdir, 'feature.bin')
        if checkToSkip(binfile, overwrite):
            continue
            
        hitlist = hitlists[tag]
        hitlist = hitlist[:k] # keep at most 1000 images per tag
        renamed,vecs = feat_file.read(hitlist)
        
        makedirsforfile(binfile)
        np.array(vecs).astype(np.float32).tofile(binfile)
        idfile = os.path.join(resultdir, 'id.txt')
        fw = open(idfile, 'w')
        fw.write(' '.join(renamed))
        fw.close()
        
        shapefile = os.path.join(resultdir, 'shape.txt')
        fw = open(shapefile, 'w')
        fw.write('%d %d' % (len(renamed), len(vecs[0])))
        fw.close()
        
        if tag_idx%1e3 == 0:
            printStatus(INFO, '%d - %s, %d images' % (tag_idx, tag, len(hitlist)))
   
   
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection feature""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    #parser.add_option("--k", default=DEFAULT_K, type="int", help="nr of examples for KDE (default: %d)" % DEFAULT_K)
    parser.add_option("--tpp", default=DEFAULT_TPP, type="string", help="tag preprocess, can be raw, stem, or lemm (default: %s)" % DEFAULT_TPP)
    parser.add_option("--numjobs", default=1, type="int", help="number of jobs")
    parser.add_option("--job", default=1, type="int", help="current job")
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 2:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1])
    

if __name__ == "__main__":
    sys.exit(main())  
    
