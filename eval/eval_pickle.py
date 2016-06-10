import sys, os
import numpy as np
import string

import cPickle as pickle

from basic.constant import ROOT_PATH
from basic.common import niceNumber,printStatus, writeRankingResults
from basic.annotationtable import readConcepts,readAnnotationsFrom
from basic.metric import getScorer
from basic.util import readImageSet


INFO = __file__


def process(options, collection, annotationName, runfile):
    rootpath = options.rootpath
    
    apscorer = getScorer('AP')
    ndcg = getScorer('NDCG@20')
    ndcg2 = getScorer('NDCG2@20')
    p1scorer = getScorer('P@1')
    p5scorer = getScorer('P@5')

    datafiles = [x.strip() for x in open(runfile).readlines() if x.strip() and not x.strip().startswith('#')]
    nr_of_runs = len(datafiles)
    
    concepts = readConcepts(collection, annotationName, rootpath=rootpath)  
    nr_of_concepts = len(concepts)
    
    printStatus(INFO, 'read annotations from files')
    
    name2label = [{} for i in range(nr_of_concepts)]
    hit_imgset = [[] for i in range(nr_of_concepts)]
    rel_conset = {}
    
    for i in range(nr_of_concepts):
        names,labels = readAnnotationsFrom(collection, annotationName, concepts[i], skip_0=False, rootpath=rootpath)
        #names = map(int, names)
        name2label[i] = dict(zip(names,labels))
        
        for im,lab in zip(names,labels):
            if lab > 0:
                rel_conset.setdefault(im,set()).add(i)

        label_file = os.path.join(rootpath, collection, 'tagged,lemm', '%s.txt'% concepts[i])
        try:
            hit_imgset[i] = set(map(string.strip, map(str, open(label_file).readlines()))) # set(map(int, open(label_file).readlines()))
        except:
            hit_imgset[i] = set()
        printStatus(INFO, 'readLabeledImageSet for %s-%s -> %d hits' % (collection, concepts[i], len(hit_imgset[i])))
        
    ap_table = np.zeros((nr_of_runs, nr_of_concepts))
    ap2_table = np.zeros((nr_of_runs, nr_of_concepts))
    ndcg_table = np.zeros((nr_of_runs, nr_of_concepts))
    ndcg2_table = np.zeros((nr_of_runs, nr_of_concepts))
    
    print '#'*100
    print '# method miap hit1 hit5'
    print '#'*100
    
    for run_idx in range(nr_of_runs):
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
            
            sorted_labels = [ground_truth[x[0]] for x in ranklist]
            assert(len(sorted_labels)>0)
            #print concepts[c_idx], ranklist[:5], sorted_labels[:5]

            ap_table[run_idx, c_idx] = apscorer.score(sorted_labels)

            sorted_labels = [ground_truth[x[0]] for x in ranklist if x[0] in hit_imgset[c_idx]]
            ap2_table[run_idx, c_idx] = apscorer.score(sorted_labels)
            ndcg_table[run_idx, c_idx] = ndcg.score(sorted_labels)
            ndcg2_table[run_idx, c_idx] = ndcg2.score(sorted_labels)

        res = np.zeros((nr_of_images, 3))
        for j in range(nr_of_images):
            ranklist = zip(range(nr_of_concepts), scores[j,:])
            ranklist.sort(key=lambda v:v[1], reverse=True)
            rel_set = rel_conset.get(imset[j], set())
            sorted_labels = [int(x[0] in rel_set) for x in ranklist]
            ap = apscorer.score(sorted_labels)
            hit1 = p1scorer.score(sorted_labels)
            hit5 = p5scorer.score(sorted_labels) > 0.1
            res[j,:] = [ap, hit1, hit5]
        avg_perf = res.mean(axis=0)
        print os.path.split(datafiles[run_idx])[-1], ' '.join(['%.3f' % x for x in avg_perf])
            


    print '#'*100
    print '# untagged-concept', ' '.join([os.path.split(x)[-1] for x in datafiles])
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
    
    print '#'*100
    print '# tagged-concept'
    print '#'*100

    for c_idx in range(nr_of_concepts):
        print concepts[c_idx], ' '.join(['%.3f' % x for x in ndcg_table[:,c_idx]])
    print 'mean%s' % ndcg.name(), ' '.join(['%.3f' % x for x in ndcg_table.mean(axis=1)])

    print '#'*100
    print '# tagged-concept'
    print '#'*100

    for c_idx in range(nr_of_concepts):
        print concepts[c_idx], ' '.join(['%.3f' % x for x in ndcg2_table[:,c_idx]])
    print 'mean%s'%ndcg2.name(), ' '.join(['%.3f' % x for x in ndcg2_table.mean(axis=1)])
    

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection annotationName runfile""")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 3:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1], args[2])


if __name__ == "__main__":
    sys.exit(main())



