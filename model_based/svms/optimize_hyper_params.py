import sys
import os

from basic.constant import ROOT_PATH
from basic.common import makedirsforfile,checkToSkip,printStatus
from basic.annotationtable import readConcepts,readAnnotationsFrom
from basic.metric import getScorer
from simpleknn.bigfile import BigFile
from mlengine_const import *

from probabilistic import sigmoid_train
from fiksvm.trainFikConcepts import DEFAULT_NR_BINS

INFO = __file__


def process(options, trainCollection, trainAnnotationName, valCollection, valAnnotationName, feature, modelName):
    assert(modelName in ['fik', 'fastlinear'])
    rootpath = options.rootpath
    autoweight = 1 #options.autoweight
    beta = 0.5
    metric = options.metric
    scorer = getScorer(metric)
    overwrite = options.overwrite
    numjobs = options.numjobs
    job = options.job

    params = {}
    
    if 'fik' == modelName:
        from fiksvm.svmutil import svm_train as train_model
        from fiksvm.fiksvm import svm_to_fiksvm as compress_model
        from fiksvm.svm import KERNEL_TYPE

        nr_bins = options.nr_bins
        modelName += str(nr_bins)
        params['nr_bins'] = nr_bins
        minmax_file = os.path.join(rootpath, trainCollection, 'FeatureData', feature, 'minmax.txt')
        with open(minmax_file, 'r') as f:
            params['min_vals'] = map(float, str.split(f.readline()))
            params['max_vals'] = map(float, str.split(f.readline()))    
    else:
        from fastlinear.liblinear193.python.liblinearutil import train as train_model
        from fastlinear.fastlinear import liblinear_to_fastlinear as compress_model
        modelName = 'fastlinear'


    
    concepts = readConcepts(trainCollection,trainAnnotationName, rootpath=rootpath)
    valConcepts = readConcepts(valCollection,valAnnotationName, rootpath=rootpath)
    concept_num = len(concepts)
    for i in range(concept_num):
        assert(concepts[i] == valConcepts[i])
    
    resultdir = os.path.join(rootpath, trainCollection, 'Models', trainAnnotationName, '%s,best_params'%modelName, '%s,%s,%s' % (valCollection,valAnnotationName,feature))
    todo = []
    for concept in concepts:
        resultfile = os.path.join(resultdir, concept + '.txt')
        if not checkToSkip(resultfile, overwrite):
            todo.append(concept)
    todo = [todo[i] for i in range(len(todo)) if i%numjobs==(job-1)]
    printStatus(INFO, 'to process %d concepts: %s' % (len(todo), ' '.join(todo)))
    if not todo:
        return 0

    train_feat_file = BigFile(os.path.join(rootpath,trainCollection,'FeatureData',feature))
    val_feat_file = BigFile(os.path.join(rootpath,valCollection,'FeatureData',feature))
    feat_dim = train_feat_file.ndims
    assert(feat_dim == val_feat_file.ndims)

    
    for concept in todo:
        names,labels = readAnnotationsFrom(trainCollection, trainAnnotationName, concept, skip_0=True, rootpath=rootpath)
        name2label = dict(zip(names,labels))
        renamed,vectors = train_feat_file.read(names)
        Ys = [name2label[x] for x in renamed]
        np = len([1 for lab in labels if  1 == lab])
        nn = len([1 for lab in labels if  -1== lab])
        wp = float(beta) * (np+nn) / np
        wn = (1.0-beta) * (np+nn) /nn
    
        names,labels = readAnnotationsFrom(valCollection, valAnnotationName, concept, skip_0=True, rootpath=rootpath)
        val_name2label = dict(zip(names,labels))
        val_renamed, val_vectors = val_feat_file.read(names)
        
        min_perf = 2.0
        worst_C = 1.0
        max_perf = 0.0
        best_C = 1.0
        best_scores = None
        best_labels = None
        for C in [1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]:
            if autoweight:
                svm_params = '-w1 %g -w-1 %g' % (wp*C, wn*C) 
            else:
                svm_params = '-c %g' % C
            
            if modelName.startswith('fik'):
                svm_params += ' -s 0 -t %d' % KERNEL_TYPE.index("HI")
            else:
                svm_params += ' -s 2 -B -1 '
            #print modelName, '>'*20, svm_params
            model = train_model(Ys, vectors, svm_params + ' -q')
            new_model = compress_model([model], [1.0], feat_dim, params)

            ranklist = [(val_renamed[i], new_model.predict(val_vectors[i])) for i in range(len(val_renamed))]
            ranklist.sort(key=lambda v:v[1], reverse=True)
            sorted_labels = [val_name2label[x[0]] for x in ranklist]
            perf = scorer.score(sorted_labels)
            if max_perf < perf:
                max_perf = perf
                best_C = C
                best_scores = [x[1] for x in ranklist]
                best_labels = list(sorted_labels)
            if min_perf > perf:
                min_perf = perf
                worst_C = C
                
        [A,B] = sigmoid_train(best_scores, best_labels)
        resultfile = os.path.join(resultdir, '%s.txt' % concept)
        
        printStatus(INFO, '%s -> worseAP=%g, worst_C=%g, bestAP=%g, best_C=%g, a=%g, b=%g' % (concept, min_perf, worst_C, max_perf, best_C, A, B))
        makedirsforfile(resultfile)
        fw = open(resultfile, 'w')
        fw.write('bestAP=%g, best_C=%g, a=%g, b=%g' % (max_perf, best_C, A, B))
        fw.close()
            

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] trainCollection trainAnnotationName valCollection valAnnotationName feature modelName""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--nr_bins", default=DEFAULT_NR_BINS, type="int", help="nr of bins for quantization (default: %d)" % DEFAULT_NR_BINS)
    #parser.add_option("--autoweight", default=1, type="int", help="overwrite existing file (default=1)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--numjobs", default=1, type="int", help="number of jobs")
    parser.add_option("--job", default=1, type="int", help="current job")
    parser.add_option("--metric", default=DEFAULT_METRIC, type="string", help="performance metric (default: %s)" % DEFAULT_METRIC)

    
    (options, args) = parser.parse_args(argv)
    if len(args) < 6:
        parser.print_help()
        return 1
    
    assert(options.job>=1 and options.numjobs >= options.job)
    
    return process(options, args[0], args[1], args[2], args[3], args[4], args[5])

if __name__ == "__main__":
    sys.exit(main())

    
