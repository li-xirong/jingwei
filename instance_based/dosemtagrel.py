import sys
import os

from basic.common import checkToSkip, niceNumber, printStatus, makedirsforfile
from basic.constant import ROOT_PATH, DEFAULT_TPP
from basic.util import readImageSet
from util.imagesearch.datareader import TagReader
from semantictagrel import SIM_TO_TAGREL
     
INFO = __file__   

def process(options, testCollection, trainCollection, tagsimMethod):
    rootpath = options.rootpath
    overwrite = options.overwrite
    testsetName = options.testset if options.testset else testCollection 
    tpp = options.tpp
    numjobs = options.numjobs
    job = options.job
    useWnVob = 1

    outputName = tagsimMethod + '-wn' if useWnVob else tagsimMethod

    if tagsimMethod == 'wns':
        resultfile = os.path.join(rootpath, testCollection, "tagrel", testsetName, outputName,'id.tagvotes.txt')
    else:    
        resultfile = os.path.join(rootpath, testCollection, "tagrel", testsetName, trainCollection, outputName,'id.tagvotes.txt')
    if numjobs>1:
        resultfile = resultfile.replace("id.tagvotes.txt", "id.tagvotes.%d.%d.txt" % (numjobs,job))

    if checkToSkip(resultfile, overwrite):
        sys.exit(0)

    makedirsforfile(resultfile)

    try:
        doneset = set([str.split(x)[0] for x in open(options.donefile).readlines()[:-1]])
    except:
        doneset = set()
        
    printStatus(INFO, "done set: %d" % len(doneset))

 
    testImageSet = readImageSet(testCollection, testCollection, rootpath)
    testImageSet = [x for x in testImageSet if x not in doneset]
    testImageSet = [testImageSet[i] for i in range(len(testImageSet)) if (i%numjobs+1) == job]
    printStatus(INFO, 'working on %d-%d, %d test images -> %s' % (numjobs,job,len(testImageSet),resultfile) )
    
    testreader = TagReader(testCollection, rootpath=rootpath)    

    if tagsimMethod == "wns":
        tagrel = SIM_TO_TAGREL["wns"](trainCollection, useWnVob, "wup", rootpath)
    else:
        tagrel = SIM_TO_TAGREL[tagsimMethod](trainCollection, useWnVob, rootpath)

 
    done = 0
    fw = open(resultfile, "w")
    
    for qry_id in testImageSet:
        qry_tags = testreader.get(qry_id)    
        tagvotes = tagrel.estimate(qry_tags)
        newline = qry_id + " " + " ".join(["%s %s" % (tag, niceNumber(vote,8)) for (tag,vote) in tagvotes])
        fw.write(newline+"\n")
        done += 1
        if done%1000 == 0:
            printStatus(INFO, "%d done" % done)
    # done    
    fw.close()
    printStatus(INFO, "%d done" % done)
    

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] testCollection trainCollection tagsimMethod""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--tpp", default=DEFAULT_TPP, type="string", help="tag preprocess, can be raw, stem, or lemm (default: %s)" % DEFAULT_TPP)
    parser.add_option("--numjobs", default=1, type="int", help="number of jobs")
    parser.add_option("--job", default=1, type="int", help="current job")
    parser.add_option("--testset", default=None, type="string", help="subset to process. Default is None")
    parser.add_option("--donefile", default=None, type="string", help="to ignore images that have been done")
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 3:
        parser.print_help()
        return 1
    
    assert(args[2] in str.split('wns fcs avgcos mulcos'))
    return process(options, args[0], args[1], args[2])


if __name__ == "__main__":
    sys.exit(main()) 


