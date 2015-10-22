import sys, os, time

from basic.constant import ROOT_PATH
from basic.common import printStatus
from basic.annotationtable import readConcepts, readAnnotationsFrom

from util.simpleknn.bigfile import BigFile
from probabilistic import sigmoid_train
from mlengine_util import classify_large_data

OVERWRITE = 0
DEFAULT_MODEL_CLASS = 'fik50'

INFO = __file__

def process(options, trainCollection, modelAnnotationName, trainAnnotationName, feature):
    rootpath = options.rootpath
    modelName = options.model

    if 'fastlinear' == modelName:
        from fastlinear.fastlinear import fastlinear_load_model as load_model
        from fastlinear.fastlinear import fastlinear_save_model as save_model
    else:
        from fiksvm.fiksvm import fiksvm_load_model as load_model
        from fiksvm.fiksvm import fiksvm_save_model as save_model


    concepts = readConcepts(trainCollection, trainAnnotationName, rootpath)
    concepts = [concepts[i] for i in range(len(concepts)) if (i%options.numjobs + 1) == options.job]

    feat_file = None #BigFile(os.path.join(rootpath, trainCollection, "FeatureData", feature))

    for concept in concepts:
        modelfile = os.path.join(rootpath, trainCollection, 'Models', modelAnnotationName, feature, modelName, '%s.model' % concept)
        model = load_model(modelfile)
        (A0, B0) = model.get_probAB()
        if abs(A0) > 1e-8 and not options.overwrite:
            printStatus(INFO, "old parameters exist as A=%g, B=%g, skip" % (A0, B0))
            continue
        names,labels = readAnnotationsFrom(trainCollection, trainAnnotationName, concept, skip_0=True, rootpath=rootpath)
        name2label = dict(zip(names, labels))

        if not feat_file:
            feat_file = BigFile(os.path.join(rootpath, trainCollection, "FeatureData", feature))

        results = classify_large_data(model, names, feat_file, prob_output=False)
        labels = [name2label[x[0]] for x in results]
        dec_values = [x[1] for x in results]
        printStatus(INFO, "%s +%d -%d" % (concept, len([x for x in labels if x==1]), len([x for x in labels if x==-1])))
        [A,B] = sigmoid_train(dec_values, labels)
        model.set_probAB(A, B)
        save_model(modelfile, model)
        (A1, B1) = model.get_probAB()
        printStatus(INFO, "A: %g -> %g, B: %g -> %g" % (A0, A1, B0, B1))
 


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] trainCollection modelAnnotationName trainAnnotationName feature""")
    parser.add_option("--overwrite", default=OVERWRITE, type="int", help="overwrite existing file (default: %d)"%OVERWRITE)
    parser.add_option("--model", default=DEFAULT_MODEL_CLASS, type="string", help="(default: %s)" % DEFAULT_MODEL_CLASS)
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath where the train and test collections are stored (default: %s)"%ROOT_PATH)
    parser.add_option("--numjobs", default=1, type="int", help="number of jobs (default: 1)")
    parser.add_option("--job", default=1, type="int", help="current job (default: 1)")
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 4:
        parser.print_help()
        return 1
    
    assert(options.job>=1 and options.numjobs >= options.job)
    return process(options, args[0], args[1], args[2], args[3])
 
 

if __name__ == "__main__":
    sys.exit(main())
    
 
        

