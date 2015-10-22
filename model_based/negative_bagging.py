import sys
import os
import time

from basic.constant import ROOT_PATH
from basic.common import makedirsforfile,checkToSkip,printStatus
from basic.annotationtable import readConcepts,readAnnotationsFrom, writeConceptsTo

from util.simpleknn.bigfile import BigFile

from svms.mlengine_const import *
from svms.fiksvm.trainFikConcepts import DEFAULT_NR_BINS

INFO = __file__


def process(options, trainCollection, annotationfile, feature, modelName):
    assert(modelName in ['fik', 'fastlinear'])
    rootpath = options.rootpath
    autoweight = 1 #options.autoweight
    beta = 0.5
    C = 1
    overwrite = options.overwrite
    numjobs = options.numjobs
    job = options.job

    params = {'rootpath': rootpath, 'model': modelName}
    
    if 'fik' == modelName:
        from svms.fiksvm.svmutil import svm_train as train_model
        from svms.fiksvm.fiksvm import svm_to_fiksvm as compress_model
        from svms.fiksvm.fiksvm import fiksvm_save_model as save_model
        from svms.fiksvm.svm import KERNEL_TYPE

        nr_bins = options.nr_bins
        modelName += str(nr_bins)
        params['nr_bins'] = nr_bins
        minmax_file = os.path.join(rootpath, trainCollection, 'FeatureData', feature, 'minmax.txt')
        with open(minmax_file, 'r') as f:
            params['min_vals'] = map(float, str.split(f.readline()))
            params['max_vals'] = map(float, str.split(f.readline()))    
    else:
        from svms.fastlinear.liblinear193.python.liblinearutil import train as train_model
        from svms.fastlinear.fastlinear import liblinear_to_fastlinear as compress_model
        from svms.fastlinear.fastlinear import fastlinear_save_model as save_model
 
    newAnnotationName = os.path.split(annotationfile)[-1]
    trainAnnotationNames = [x.strip() for x in open(annotationfile).readlines() if x.strip() and not x.strip().startswith('#')]
    for annotationName in trainAnnotationNames:
        conceptfile = os.path.join(rootpath, trainCollection, 'Annotations', annotationName)
        if not os.path.exists(conceptfile):
            print '%s does not exist' % conceptfile
            return 0

    concepts = readConcepts(trainCollection, trainAnnotationNames[0], rootpath=rootpath)

    resultdir = os.path.join(rootpath, trainCollection, 'Models', newAnnotationName, feature, modelName)
    todo = []
    for concept in concepts:
        resultfile = os.path.join(resultdir, concept + '.model')
        if not checkToSkip(resultfile, overwrite):
            todo.append(concept)
    todo = [todo[i] for i in range(len(todo)) if i%numjobs==(job-1)]
    printStatus(INFO, 'to process %d concepts: %s' % (len(todo), ' '.join(todo)))
    if not todo:
        return 0

    train_feat_file = BigFile(os.path.join(rootpath,trainCollection,'FeatureData',feature))
    feat_dim = train_feat_file.ndims

    s_time = time.time()

    for concept in todo:
        assemble_model = None
        for t in range(1, len(trainAnnotationNames)+1):
            names,labels = readAnnotationsFrom(trainCollection, trainAnnotationNames[t-1], concept, skip_0=True, rootpath=rootpath)
            name2label = dict(zip(names,labels))
            renamed,vectors = train_feat_file.read(names)
            Ys = [name2label[x] for x in renamed]
            np = len([1 for lab in labels if  1 == lab])
            nn = len([1 for lab in labels if  -1== lab])
            wp = float(beta) * (np+nn) / np
            wn = (1.0-beta) * (np+nn) /nn
    
            if autoweight:
                svm_params = '-w1 %g -w-1 %g' % (wp*C, wn*C) 
            else:
                svm_params = '-c %g' % C
            
            if modelName.startswith('fik'):
                svm_params += ' -s 0 -t %d' % KERNEL_TYPE.index("HI")
            else:
                svm_params += ' -s 2 -B -1 '
           
            g_t = train_model(Ys, vectors, svm_params + ' -q')
            if t == 1:
                assemble_model = compress_model([g_t], [1.0], feat_dim, params)
            else:
                new_model = compress_model([g_t], [1.0], feat_dim, params)
                assemble_model.add_fastsvm(new_model, 1-1.0/t, 1.0/t)

        new_model_file = os.path.join(resultdir, '%s.model' % concept)            
        makedirsforfile(new_model_file)
        printStatus(INFO, 'save model to %s' % new_model_file)
        save_model(new_model_file, assemble_model)
        printStatus(INFO, '%s done' % concept)

        
    timecost = time.time() - s_time
    writeConceptsTo(concepts, trainCollection, newAnnotationName, rootpath)
    printStatus(INFO, 'done for %g concepts: %s' % (len(todo), ' '.join(todo)))
    printStatus(INFO, 'models stored at %s' % resultdir)
    printStatus(INFO, '%g seconds in total' % timecost)
        

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] trainCollection annotationsfile feature modelName""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--nr_bins", default=DEFAULT_NR_BINS, type="int", help="nr of bins for quantization (default: %d)" % DEFAULT_NR_BINS)
    #parser.add_option("--autoweight", default=1, type="int", help="overwrite existing file (default=1)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--numjobs", default=1, type="int", help="number of jobs")
    parser.add_option("--job", default=1, type="int", help="current job")
    parser.add_option("--metric", default=DEFAULT_METRIC, type="string", help="performance metric (default: %s)" % DEFAULT_METRIC)

    
    (options, args) = parser.parse_args(argv)
    if len(args) < 4:
        parser.print_help()
        return 1
    
    assert(options.job>=1 and options.numjobs >= options.job)
    
    return process(options, args[0], args[1], args[2], args[3])

if __name__ == "__main__":
    sys.exit(main())

    
