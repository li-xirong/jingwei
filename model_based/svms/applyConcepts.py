import sys
import os
import time

from basic.constant import ROOT_PATH
from basic.common import  makedirsforfile, checkToSkip, printStatus, niceNumber
from basic.annotationtable import readConcepts
from basic.util import readImageSet
from util.simpleknn.bigfile import BigFile


from mlengine_const import *
INFO = __file__


def process(options, testCollection, trainCollection, trainAnnotationName, feature, modelName):
    if modelName.startswith('fik'):
        from fiksvm.fiksvm import fiksvm_load_model as load_model
    else:
        from fastlinear.fastlinear import fastlinear_load_model as load_model

    rootpath = options.rootpath
    overwrite = options.overwrite
    prob_output = options.prob_output
    numjobs = options.numjobs
    job = options.job
    blocksize = options.blocksize
    
    outputName = '%s,%s' % (feature,modelName)
    if prob_output:
        outputName += ',prob'

    resultfile = os.path.join(rootpath, testCollection, 'autotagging', testCollection, trainCollection, trainAnnotationName, outputName, 'id.tagvotes.txt')
    if numjobs>1:
        resultfile += '.%d.%d' % (numjobs, job)

    if checkToSkip(resultfile, overwrite):
        return 0

    concepts = readConcepts(trainCollection,trainAnnotationName, rootpath=rootpath)
    nr_of_concepts = len(concepts)

    test_imset = readImageSet(testCollection, testCollection, rootpath)
    test_imset = [test_imset[i] for i in range(len(test_imset)) if i%numjobs+1 == job]
    nr_of_test_images = len(test_imset)
    printStatus(INFO, "working on %d-%d, %d test images -> %s" % (numjobs,job,nr_of_test_images,resultfile))

    models = [None] * nr_of_concepts
    for c in range(nr_of_concepts):
        model_file_name = os.path.join(rootpath,trainCollection,'Models',trainAnnotationName,feature, modelName, '%s.model'%concepts[c])
        models[c] = load_model(model_file_name)
        if models[c] is None:
            return 0
        #(pA,pB) = model.get_probAB()
        

    feat_file = BigFile(os.path.join(rootpath, testCollection, "FeatureData", feature))
    makedirsforfile(resultfile)
    fw = open(resultfile, "w")

    read_time = 0
    test_time = 0
    start = 0
    done = 0

    while start < nr_of_test_images:
        end = min(nr_of_test_images, start + blocksize)
        printStatus(INFO, 'processing images from %d to %d' % (start, end-1))

        s_time = time.time()
        renamed, test_X = feat_file.read(test_imset[start:end])
        read_time += time.time() - s_time
        
        s_time = time.time()
        output = [None] * len(renamed)
        for i in xrange(len(renamed)):
            if prob_output:
                scores = [models[c].predict_probability(test_X[i]) for c in range(nr_of_concepts)]
            else:
                scores = [models[c].predict(test_X[i]) for c in range(nr_of_concepts)]
            #dec_value = sigmoid_predict(dec_value, A=pA, B=pB)
            tagvotes = sorted(zip(concepts, scores), key=lambda v:v[1], reverse=True)
            output[i] = '%s %s\n' % (renamed[i], " ".join(["%s %s" % (tag, niceNumber(vote,6)) for (tag,vote) in tagvotes]))
        test_time += time.time() - s_time
        start = end
        fw.write(''.join(output))
        fw.flush()
        done += len(output)

    # done    
    printStatus(INFO, "%d done. read time %g seconds, test_time %g seconds" % (done, read_time, test_time))
    fw.close()
    return done


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] testCollection trainCollection trainAnnotationName feature modelName""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--prob_output", default=DEFAULT_PROB_OUTPUT, type="int", help="probabilistic output. Only work when sigmoid has been trained (default: %d)" % DEFAULT_PROB_OUTPUT)
    parser.add_option("--numjobs", default=1, type="int", help="number of jobs (default: 1)")
    parser.add_option("--job", default=1, type="int", help="current job (default: 1)")
    parser.add_option("--blocksize", default=DEFAULT_BLOCK_SIZE, type="int", help="nr of feature vectors loaded per time (default: %d)" % DEFAULT_BLOCK_SIZE)
    
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 5:
        parser.print_help()
        return 1
    
    assert(options.job>=1 and options.numjobs >= options.job)

    return process(options, args[0], args[1], args[2], args[3], args[4])


if __name__ == "__main__":
    sys.exit(main())


