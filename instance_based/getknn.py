
import os, sys, time

from basic.constant import ROOT_PATH
from basic.common import checkToSkip, printStatus, writeRankingResults
from basic.util import readImageSet

from util.simpleknn.bigfile import BigFile
from util.simpleknn import simpleknn as imagesearch


DEFAULT_K=1500
DEFAULT_DISTANCE = 'l2'
DEFAULT_UU = 1
DEFAULT_BLOCK_SIZE = 1000

INFO = __file__


def unique_user_constraint(knn, im2user, k):
    res = []
    userSet = set()
    removed = 0
    
    for name,score in knn:
        userid = im2user[name]
        if userid in userSet:
            removed += 1
            continue
        userSet.add(userid)
        res.append((name,score))
        if len(res) == k:
            break
    return removed, res                        

    
def process(options, trainCollection, testCollection, feature):
    rootpath = options.rootpath
    k = options.k
    distance = options.distance
    blocksize = options.blocksize
    uniqueUser = options.uu
    numjobs = options.numjobs
    job = options.job
    overwrite = options.overwrite
    testset = options.testset
    if not testset:
        testset = testCollection

    searchMethod = distance + 'knn'
    if uniqueUser:
        searchMethod += ",uu"
        tagfile = os.path.join(rootpath, trainCollection, 'TextData', 'id.userid.lemmtags.txt')
        im2user = {}
        for line in open(tagfile):
            im,userid,tags = line.split('\t')
            im2user[im] = userid
    
    resultdir = os.path.join(rootpath, testCollection, "SimilarityIndex", testset, trainCollection, "%s,%s,%d" % (feature,searchMethod,k))
    feat_dir = os.path.join(rootpath, trainCollection, 'FeatureData', feature)
    id_file = os.path.join(feat_dir, 'id.txt')
    shape_file = os.path.join(feat_dir, 'shape.txt')
    nr_of_images, feat_dim = map(int, open(shape_file).readline().split())
    nr_of_images = len(open(id_file).readline().strip().split())
    searcher = imagesearch.load_model(os.path.join(feat_dir, 'feature.bin'), feat_dim, nr_of_images, id_file)
    searcher.set_distance(distance)
        
    workingSet = readImageSet(testCollection, testset, rootpath=rootpath)
    workingSet = [workingSet[i] for i in range(len(workingSet)) if (i%numjobs+1) == job]
    printStatus(INFO, "working on %d-%d, %d test images -> %s" % (numjobs,job,len(workingSet),resultdir))
    
    test_feat_dir = os.path.join(rootpath, testCollection, 'FeatureData', feature)
    test_feat_file = BigFile(test_feat_dir)

    read_time = 0
    knn_time = 0
    start = 0
    done = 0
    filtered = 0

    while start < len(workingSet):
        end = min(len(workingSet), start + blocksize)
        printStatus(INFO, 'processing images from %d to %d' % (start, end-1))

        s_time = time.time()
        renamed,vectors = test_feat_file.read(workingSet[start:end])
        read_time += time.time() - s_time
        nr_images = len(renamed)
        
        s_time = time.time()
        for i in range(nr_images):
            resultfile = os.path.join(resultdir, renamed[i][-2:], '%s.txt' % renamed[i])
            if checkToSkip(resultfile, overwrite):
                continue
            knn = searcher.search_knn(vectors[i], max_hits=max(3000,k*3))
            if uniqueUser:
                removed, newknn = unique_user_constraint(knn, im2user, k)
                filtered += removed
                knn = newknn
            else:
                knn = knn[:k]
            assert(len(knn) >= k)
            writeRankingResults(knn, resultfile)
            done += 1
        printStatus(INFO, 'job %d-%d: %d done, filtered neighbors %d' % (numjobs, job, done, filtered))
        start = end

    printStatus(INFO, 'job %d-%d: %d done, filtered neighbors %d' % (numjobs, job, done, filtered))
    
    
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
             
    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] trainCollection testCollection feature""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--uu", default=DEFAULT_UU, type="int", help="unique user constraint (default=%d)" % DEFAULT_UU)
    parser.add_option("--testset", default=None, type="string", help="process a specified subset of $testCollection")
    parser.add_option("--k", default=DEFAULT_K, type="int", help="number of neighbors (%d)" % DEFAULT_K)
    parser.add_option("--distance", default=DEFAULT_DISTANCE, type="string", help="visual distance, can be l1 or l2 (default: %s)" % DEFAULT_DISTANCE)
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="(default: %s)" % ROOT_PATH)
    parser.add_option("--numjobs", default=1, type="int", help="number of jobs (default: 1)")
    parser.add_option("--job", default=1, type="int", help="current job (default: 1)")
    parser.add_option("--blocksize", default=DEFAULT_BLOCK_SIZE, type="int", help="nr of feature vectors loaded per time (default: %d)" % DEFAULT_BLOCK_SIZE)
    
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 3:
        parser.print_help()
        return 1
    
    assert(options.job>=1 and options.numjobs >= options.job)
    return process(options, args[0], args[1], args[2])
    
    
if __name__ == "__main__":
    sys.exit(main())

    

