import sys
import os
import time

from basic.constant import ROOT_PATH
from basic.common import makedirsforfile, checkToSkip, printStatus, niceNumber
from basic.annotationtable import readConcepts,readAnnotationsFrom
from basic.util import readImageSet
from util.simpleknn.bigfile import StreamFile
from fastlinear.faster import ModelArray

DEFAULT_K = -1
INFO = __file__


def process(options, testCollection, trainCollection, trainAnnotationName, feature, modelName):
    assert(modelName.startswith('fastlinear'))
    
    rootpath = options.rootpath
    overwrite = options.overwrite
    numjobs = options.numjobs
    job = options.job
    topk = options.topk
    
    outputName = '%s,%s' % (feature,modelName)
    
    resultfile = os.path.join(rootpath, testCollection, 'autotagging', testCollection, trainCollection, trainAnnotationName, outputName, 'id.tagvotes.txt')
    if numjobs>1:
        resultfile += '.%d.%d' % (numjobs, job)

    if checkToSkip(resultfile, overwrite):
        return 0

    concepts = readConcepts(trainCollection,trainAnnotationName, rootpath=rootpath)
    nr_of_concepts = len(concepts)

    test_imset = readImageSet(testCollection, testCollection, rootpath)
    test_imset = [test_imset[i] for i in range(len(test_imset)) if i%numjobs+1 == job]
    test_imset = set(test_imset)
    nr_of_test_images = len(test_imset)
    printStatus(INFO, "working on %d-%d, %d test images -> %s" % (numjobs,job,nr_of_test_images,resultfile))

    ma = ModelArray(trainCollection, trainAnnotationName, feature, modelName, rootpath=rootpath)
        
    feat_file = StreamFile(os.path.join(rootpath, testCollection, "FeatureData", feature))
    makedirsforfile(resultfile)
    fw = open(resultfile, "w")

    done = 0

    feat_file.open()
    for _id, _vec in feat_file:
        if _id not in test_imset:
            continue
       
        res = ma.predict([_vec],prob=0)
        tagvotes = res[0]
        if topk>0:
            tagvotes = tagvotes[:topk]
        newline = '%s %s\n' % (_id, " ".join(["%s %s" % (tag, niceNumber(vote,6)) for (tag,vote) in tagvotes]))
        fw.write(newline)
        done += 1
        if done % 1e4  == 0:
            printStatus(INFO, "%d done" % done)

    feat_file.close()
    fw.close()
    printStatus(INFO, "%d done" % (done))
    return done


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] testCollection trainCollection trainAnnotationName feature modelName""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--numjobs", default=1, type="int", help="number of jobs (default: 1)")
    parser.add_option("--job", default=1, type="int", help="current job (default: 1)")
    parser.add_option("--topk", default=DEFAULT_K, type="int", help="top k tags (default: %d)" % DEFAULT_K)
    
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 5:
        parser.print_help()
        return 1
    
    assert(options.job>=1 and options.numjobs >= options.job)

    return process(options, args[0], args[1], args[2], args[3], args[4])


if __name__ == "__main__":
    sys.exit(main())
