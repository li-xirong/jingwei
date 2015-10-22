import sys, os, time

from basic.constant import ROOT_PATH, DEFAULT_TPP
from basic.common import checkToSkip, niceNumber, printStatus, makedirsforfile
from basic.util import readImageSet

from util.simpleknn.bigfile import BigFile
from tagvote import DEFAULT_TAGGER, DEFAULT_K, DEFAULT_DISTANCE, DEFAULT_BLOCK_SIZE, NAME_TO_TAGGER

INFO = __file__


def process(options, testCollection, trainCollection, annotationName, feature):
    rootpath = options.rootpath
    k = options.k
    distance = options.distance
    blocksize = options.blocksize
    donefile = options.donefile
    numjobs = options.numjobs
    job = options.job
    overwrite = options.overwrite
    taggerType = options.tagger
    noise = options.noise
    testset = options.testset
    if not testset:
        testset = testCollection
    
    modelName = taggerType
    if 'pretagvote' == taggerType and noise > 1e-3:
        modelName += '-noise%.2f' % noise
    if 'pqtagvote' == taggerType:
        nnName = "l2knn"
    else:
        nnName = distance + "knn"
    resultfile = os.path.join(rootpath, testCollection, 'autotagging', testset, trainCollection, annotationName, modelName, '%s,%s,%d'%(feature,nnName,k), 'id.tagvotes.txt')
    
    if numjobs>1:
        resultfile += ".%d.%d" % (numjobs,job)
    if checkToSkip(resultfile, overwrite):
        return 0

    if donefile:
        doneset = set([x.split()[0] for x in open(donefile) if x.strip()])
    else:
        doneset = set()
    printStatus(INFO, "%d images have been done already, and they will be ignored" % len(doneset))
        
    workingSet = readImageSet(testCollection, testset, rootpath)
    workingSet = [x for x in workingSet if x not in doneset]
    workingSet = [workingSet[i] for i in range(len(workingSet)) if (i%numjobs+1) == job]
    
    test_feat_dir = os.path.join(rootpath, testCollection, 'FeatureData', feature)
    test_feat_file = BigFile(test_feat_dir)
    
    tagger = NAME_TO_TAGGER[taggerType](trainCollection, annotationName, feature, distance, rootpath=rootpath)
    tagger.k = k
    tagger.noise = noise
    
    printStatus(INFO, "working on %d-%d, %d test images -> %s" % (numjobs,job,len(workingSet),resultfile))
    
    makedirsforfile(resultfile)
    fw = open(resultfile, "w")
    

    read_time = 0.0
    test_time = 0.0
    start = 0
    done = 0

    while start < len(workingSet):
        end = min(len(workingSet), start + blocksize)
        printStatus(INFO, 'tagging images from %d to %d' % (start, end-1))

        s_time = time.time()
        renamed,vectors = test_feat_file.read(workingSet[start:end])
        nr_images = len(renamed)
        read_time += time.time() - s_time
        
        s_time = time.time()
        output = [None] * nr_images
        for i in range(nr_images):
            tagvotes = tagger.predict(content=vectors[i], context='%s,%s' % (testCollection, renamed[i]))
            output[i] = '%s %s\n' % (renamed[i], " ".join(["%s %s" % (tag, niceNumber(vote,6)) for (tag,vote) in tagvotes]))
        test_time += time.time() - s_time
        start = end
        fw.write(''.join(output))
        done += len(output)
            
    fw.close()
    printStatus(INFO, '%d images tagged, read time %g seconds, test time %g seconds' % (done, read_time, test_time))

    
    
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] testCollection trainCollection annotationName feature""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--donefile", default=None, type="string", help="to ignore images that have been done")
    parser.add_option("--tagger", default=DEFAULT_TAGGER, type="string", help="tagger type, which can be tagvote, pretagvote, preknn, and pqtagvote (default: %s)" % DEFAULT_TAGGER)
    parser.add_option("--testset", default=None, type="string", help="process a specified subset")
    parser.add_option("--k", default=DEFAULT_K, type="int", help="number of neighbors (%d)" % DEFAULT_K)
    parser.add_option("--tpp", default="lemm", type="string", help="tag preprocess, can be raw, stem, or lemm (default: %s)" % DEFAULT_TPP)
    parser.add_option("--distance", default=DEFAULT_DISTANCE, type="string", help="visual distance, can be l1, l2, or cosine (default: %s)" % DEFAULT_DISTANCE)
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="(default: %s)" % ROOT_PATH)
    parser.add_option("--numjobs", default=1, type="int", help="number of jobs (default: 1)")
    parser.add_option("--job", default=1, type="int", help="current job (default: 1)")
    parser.add_option("--blocksize", default=DEFAULT_BLOCK_SIZE, type="int", help="nr of feature vectors loaded per time (default: %d)" % DEFAULT_BLOCK_SIZE)
    parser.add_option("--noise", default=0, type="float", help="random noise (default: %g)" % 0)
    
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 4:
        parser.print_help()
        return 1
    
    assert(options.job>=1 and options.numjobs >= options.job)
    return process(options, args[0], args[1], args[2], args[3])

if __name__ == "__main__":
    sys.exit(main())


