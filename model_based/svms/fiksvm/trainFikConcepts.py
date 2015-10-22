import sys
import os

from basic.constant import ROOT_PATH
from basic.common import makedirsforfile,checkToSkip,printStatus
from basic.annotationtable import readConcepts,readAnnotationsFrom

from svm import KERNEL_TYPE
from svmutil import svm_train
from fiksvm import svm_to_fiksvm, fiksvm_save_model, fiksvm_load_model
from util.simpleknn.bigfile import BigFile


DEFAULT_NR_BINS = 50
INFO = __file__


def process(options, trainCollection, trainAnnotationName, feature):
    import re
    p = re.compile(r'best_C=(?P<C>[\.\d]+),\sa=(?P<a>[\.\-\d]+),\sb=(?P<b>[\.\-\d]+)')

    rootpath = options.rootpath
    overwrite = options.overwrite
    #autoweight = options.autoweight
    numjobs = options.numjobs
    job = options.job
    nr_bins = options.nr_bins
    best_param_dir = options.best_param_dir
    beta = 0.5
    
    modelName = 'fik%d' % nr_bins
    if best_param_dir:
        modelName += '-tuned'
    
    concepts = readConcepts(trainCollection, trainAnnotationName, rootpath=rootpath)
    resultdir = os.path.join(rootpath, trainCollection, 'Models', trainAnnotationName, feature, modelName)
    todo = []
    for concept in concepts:
        resultfile = os.path.join(resultdir, concept + '.model')
        if not checkToSkip(resultfile, overwrite):
            todo.append(concept)
    todo = [todo[i] for i in range(len(todo)) if i%numjobs==(job-1)]
    printStatus(INFO, 'to process %d concepts: %s' % (len(todo), ' '.join(todo)))
    if not todo:
        return 0
    
    feat_dir = os.path.join(rootpath,trainCollection,'FeatureData',feature)
    feat_file = BigFile(feat_dir)
    params = {'nr_bins': nr_bins}

    with open(os.path.join(feat_dir, 'minmax.txt'), 'r') as f:
        params['min_vals'] = map(float, str.split(f.readline()))
        params['max_vals'] = map(float, str.split(f.readline()))
        
    for concept in todo:
        if best_param_dir:
            param_file = os.path.join(best_param_dir, '%s.txt' % concept)
            m = p.search(open(param_file).readline().strip())
            C = float(m.group('C'))
            A = float(m.group('a'))
            B = float(m.group('b'))
        else:
            C = 1
            A = 0
            B = 0
        printStatus(INFO, '%s, C=%g, A=%g, B=%g' % (concept, C, A, B))
        
        model_file_name = os.path.join(resultdir, concept + '.model')
        
        names,labels = readAnnotationsFrom(trainCollection, trainAnnotationName, concept, skip_0=True, rootpath=rootpath)
        name2label = dict(zip(names,labels))
        renamed,vectors = feat_file.read(names)
        y = [name2label[x] for x in renamed]
        np = len([1 for lab in labels if  1 == lab])
        nn = len([1 for lab in labels if  -1== lab])
        wp = float(beta) * (np+nn) / np
        wn = (1.0-beta) * (np+nn) /nn
    
        svm_params = '-w1 %g -w-1 %g' % (wp*C, wn*C) 
        model = svm_train(y, vectors, svm_params + ' -s 0 -t %d -q' % KERNEL_TYPE.index("HI"))
        newmodel = svm_to_fiksvm([model], [1.0], feat_file.ndims, params)
        newmodel.set_probAB(A, B)
        makedirsforfile(model_file_name)
        printStatus(INFO, '-> %s'%model_file_name)
        fiksvm_save_model(model_file_name, newmodel)

        # reload the model file to do a simple check
        fiksvm_load_model(model_file_name)
        assert(abs(newmodel.get_probAB()[0]-A)<1e-6)
        assert(abs(newmodel.get_probAB()[1]-B)<1e-6)

    return len(todo)



def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] trainCollection trainAnnotationName feature""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--nr_bins", default=DEFAULT_NR_BINS, type="int", help="nr of bins for quantization (default: %d)" % DEFAULT_NR_BINS)
    parser.add_option("--best_param_dir", default="", type="string", help="where the tuned parameters are stored")
    #parser.add_option("--autoweight", default=1, type="int", help="overwrite existing file (default=1)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--numjobs", default=1, type="int", help="number of jobs (default: 1)")
    parser.add_option("--job", default=1, type="int", help="current job (default: 1)")
    
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 3:
        parser.print_help()
        return 1
    
    assert(options.job>=1 and options.numjobs >= options.job)
    
    return process(options, args[0], args[1], args[2])

if __name__ == "__main__":
    sys.exit(main())

    
