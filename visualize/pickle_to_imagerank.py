import sys, os
import numpy as np

import cPickle as pickle

from basic.constant import ROOT_PATH
from basic.common import niceNumber,printStatus, writeRankingResults,checkToSkip
from basic.annotationtable import readConcepts,readAnnotationsFrom
from basic.metric import getScorer
from basic.util import readImageSet


INFO = __file__


def process(options, collection, annotationName, runfile):
    rootpath = options.rootpath
    overwrite = options.overwrite

    resultdir = os.path.join(rootpath, collection, 'SimilarityIndex', collection)

    apscorer = getScorer('AP')
    datafiles = [x.strip() for x in open(runfile).readlines() if x.strip() and not x.strip().startswith('#')]
    nr_of_runs = len(datafiles)
    
    concepts = readConcepts(collection, annotationName, rootpath=rootpath)  
    nr_of_concepts = len(concepts)
    
    printStatus(INFO, 'read annotations from files')
    
    name2label = [{} for i in range(nr_of_concepts)]
    hit_imgset = [[] for i in range(nr_of_concepts)]
    
    for i in range(nr_of_concepts):
        names,labels = readAnnotationsFrom(collection, annotationName, concepts[i], skip_0=False, rootpath=rootpath)
        names = map(int, names)
        name2label[i] = dict(zip(names,labels))

        label_file = os.path.join(rootpath, collection, 'tagged,lemm', '%s.txt'% concepts[i])
        try:
            hit_imgset[i] = set(map(int, open(label_file).readlines()))
        except:
            hit_imgset[i] = set()
        printStatus(INFO, 'readLabeledImageSet for %s-%s -> %d hits' % (collection, concepts[i], len(hit_imgset[i])))
        
    ap_table = np.zeros((nr_of_runs, nr_of_concepts))
    ap2_table = np.zeros((nr_of_runs, nr_of_concepts))
    
    for run_idx in range(nr_of_runs):
        runName = os.path.splitext(os.path.basename(datafiles[run_idx]))[0]
        data = pickle.load(open(datafiles[run_idx],'rb'))
        scores = data['scores']
        assert(scores.shape[1] == nr_of_concepts)
        imset = data['id_images']
        nr_of_images = len(imset)
        #print datafiles[run_idx], imset[:5], imset[-5:]
                   
        for c_idx in range(nr_of_concepts):
            ground_truth = name2label[c_idx]
            ranklist =  zip(imset, scores[:,c_idx])
            ranklist.sort(key=lambda v:(v[1], str(v[0])), reverse=True)
            ranklist = [x for x in ranklist if x[0] in ground_truth]

            resfile = os.path.join(resultdir, runName, '%s.txt' % concepts[c_idx])
            if not checkToSkip(resfile, overwrite):
                writeRankingResults(ranklist, resfile)            
            sorted_labels = [ground_truth[x[0]] for x in ranklist]
            assert(len(sorted_labels)>0)
            #print concepts[c_idx], ranklist[:5], sorted_labels[:5]

            ap_table[run_idx, c_idx] = apscorer.score(sorted_labels)

            resfile = os.path.join(resultdir, 'tagged,lemm', runName, '%s.txt' % concepts[c_idx])
            if not checkToSkip(resfile, overwrite):
                writeRankingResults([x for x in ranklist if x[0] in hit_imgset[c_idx]], resfile)            
            
            sorted_labels = [ground_truth[x[0]] for x in ranklist if x[0] in hit_imgset[c_idx]]
            ap2_table[run_idx, c_idx] = apscorer.score(sorted_labels)
     

    print '#'*100
    print '# untagged-concept', ' '.join([os.path.basename(x) for x in datafiles])
    print '#'*100
            
    for c_idx in range(nr_of_concepts):
        print concepts[c_idx], ' '.join(['%.3f' % x for x in ap_table[:,c_idx]])
    print 'meanAP', ' '.join(['%.3f' % x for x in ap_table.mean(axis=1)])
    
    print '#'*100
    print '# tagged-concept'
    print '#'*100
    
    for c_idx in range(nr_of_concepts):
        print concepts[c_idx], ' '.join(['%.3f' % x for x in ap2_table[:,c_idx]])
    print 'meanAP2', ' '.join(['%.3f' % x for x in ap2_table.mean(axis=1)])
    


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] testCollection testAnnotationName runfile""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 3:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1], args[2])


if __name__ == "__main__":
    sys.exit(main())



