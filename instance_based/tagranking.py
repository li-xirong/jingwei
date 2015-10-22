import sys
import os
import time
import math
import numpy as np

from basic.constant import ROOT_PATH
from basic.common import checkToSkip,printStatus,makedirsforfile,niceNumber
from basic.util import readImageSet
from util.simpleknn.bigfile import BigFile
from util.tagsim.flickr_similarity import FlickrContextSim
from util.imagesearch.datareader import TagReader
    
    
MEDIAN_DISTANCE = {'color64+dsift':0.121128559979, 'vgg-verydeep-16-fc7relul2':1.27498506919}
FEATURE_TO_DIM = {'color64+dsift':1088, 'vgg-verydeep-16-fc7relul2':4096}

DEFAULT_K = 1000
DEFAULT_TPP = 'lemm'
DEFAULT_UU = 0
DEFAULT_RW = 1
INFO = __file__


class TagRanking:

    def __init__(self, trainCollection, tpp="lemm", feature="color64+dsift",  k=1000, rootpath=ROOT_PATH):
        self.trainCollection = trainCollection
        self.k = k
        self.name = "%s(%s,%s,%s,%d)" % (self.__class__.__name__, self.trainCollection, tpp, feature, k)

        vobfile = os.path.join(rootpath, trainCollection, "TextData", "wn.%s.txt"%trainCollection)
        self.vob = set(map(str.strip, open(vobfile).readlines()))
        printStatus(INFO, 'the vocabulary of %s contains %d tags' % (trainCollection, len(self.vob)))

        self.gamma = (1.0/MEDIAN_DISTANCE[feature])**2
        self.feat_dir = os.path.join(rootpath, trainCollection, 'FeatureIndex', feature)
        self.dim = FEATURE_TO_DIM[feature]
        self.fcs = FlickrContextSim(trainCollection,rootpath)  
        
        printStatus(INFO, self.name + ' okay')
        
        
    def getName(self):
        return self.name


    def computePxt(self, qry_vec, tag, uniqueUser=False):
        assert(self.k<=1000)
        feat_file = os.path.join(self.feat_dir, tag[:2], tag, 'feature.bin')
        vecs = np.fromfile(feat_file, dtype=np.float32)
        nr_of_images = len(vecs)/self.dim
        A = vecs.reshape( (nr_of_images, self.dim))
        weights = []
            
        #s_time = time.time()
        # accelerate pxt computation by matrix operations
        squared_distance = np.linalg.norm(qry_vec)**2 + np.linalg.norm(A,axis=1)**2 - 2*A.dot(qry_vec)
        assert(len(squared_distance) == nr_of_images)
        #print squared_distance
        #squared_distance = squared_distance.tolist()[0]
        #t1 = time.time() - s_time

        weights = [math.exp(-self.gamma*x) for x in squared_distance]
 
        ''' compute distance per pair, much slower than the matrix based approach  
        s_time = time.time()
        old_d = [0] * len(vecs)
        i = 0
        for name,x in zip(renamed, vecs):
            old_d[i] = math.sqrt(sum([(x[j]-qry_vec[j])**2 for j in range(self.dim)]))
            i += 1
        t2 = time.time() - s_time
        
        for i in range(len(vecs)):
            diff = old_d[i]**2 - squared_distance[i]
            assert(abs(diff)<1e-8)
        print '%.6f %.6f' % (t1, t2)    
        '''
        
        return np.mean(weights)


    def estimate(self, qry_vec, qry_tags, uniqueUser=0, doRandomwalk=1):
        alpha = 0.5
        epsilon = 1e-6

        tagSeq = str.split(qry_tags)
        tagSeq = [t for t in tagSeq if t in self.vob]
        n = len(tagSeq)
        v = [self.computePxt(qry_vec,tag,uniqueUser) for tag in tagSeq]

        #print sorted(zip(tagSeq,v), key=lambda v:(v[1]), reverse=True)
        if not doRandomwalk:
             tag2score = dict(zip(tagSeq,v))
             return sorted(tag2score.iteritems(), key=lambda v:(v[1]), reverse=True)   

        '''
        -----------------------------------------
        Compute the nxn transition matrix
        -----------------------------------------
        '''
        p = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    p[i,j] = 1.0
                elif i < j:
                    p[i,j] = self.fcs.compute(tagSeq[i],tagSeq[j],gamma=1)
                else:
                    p[i,j] = p[j,i]

        '''
        -----------------------------------------
        normalize row s.t. row sum is 1
        -----------------------------------------
        '''
        for i in range(n):
            rowsum = p[i,:].sum()
            p[i,:] /= float(rowsum)
        #print qry_tags
        #print p
        '''
        -----------------------------------------
        do random walk
        -----------------------------------------
        '''
        r = [0] * n
        r0 = [(1-alpha)*v[j] for j in range(n)]

        for T in range(1, 1000):
            for j in range(n):
                r[j] = alpha * sum([r0[i]*p[i,j] for i in range(n)]) + (1-alpha)*v[j]
            diff = sum([abs(r[j]-r0[j]) for j in range(n)])
            #print T, diff
            if diff < epsilon:
                break
            r0 = [r[j] for j in range(n)]

        tag2score = dict(zip(tagSeq,r))
       
        return sorted(tag2score.iteritems(), key=lambda v:v[1], reverse=True)

  
def process(options, testCollection, trainCollection, feature):
    rootpath = options.rootpath
    overwrite = options.overwrite
    tpp = options.tpp
    doRandomwalk =  1 #options.doRandomwalk
    uniqueUser = 0 #options.uniqueUser
    k = 1000 #options.k
    numjobs = options.numjobs
    job = options.job
    
    #resultfile = os.path.join(rootpath, testCollection, "tagrel", testCollection, trainCollection, 
    #                          "%s,tagrank%d%d,%d,%s" % (feature,doRandomwalk,uniqueUser,k,tpp), "id.tagvotes.txt")
    
    resultfile = os.path.join(rootpath, testCollection, "tagrel", testCollection, trainCollection, '%s,tagrank,%s' % (feature,tpp), 'id.tagvotes.txt')
        
    if numjobs>1:
        resultfile = resultfile + '.%d.%d' % (numjobs, job)
                              
    if checkToSkip(resultfile, overwrite):
        sys.exit(0)    

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
    test_feat_file = BigFile(os.path.join(rootpath, testCollection, 'FeatureData', feature))
    block_size = 100

    tagranking = TagRanking(trainCollection, feature=feature, k=k, rootpath=rootpath)
    
    makedirsforfile(resultfile)
    fw = open(resultfile, "w")
    
    done = 0
    
    nr_of_blocks = len(testImageSet) / block_size
    if nr_of_blocks * block_size < len(testImageSet):
        nr_of_blocks += 1

    for block_index in range(nr_of_blocks):
        start = block_index * block_size
        end = min(len(testImageSet), start + block_size)
        subset = testImageSet[start:end]
        if not subset:
            break
        renamed, features = test_feat_file.read(subset)
        printStatus(INFO, '%d - %d: %d images' % (start, end, len(subset)))
        
        output = []
        for i in range(len(renamed)):
            qry_id = renamed[i]
            qry_tags = testreader.get(qry_id)
            qry_vec = features[i]
            tagvotes = tagranking.estimate(qry_vec, qry_tags) #, uniqueUser=uniqueUser, doRandomwalk=doRandomwalk)
            newline = "%s %s" % (qry_id, " ".join(["%s %g" % (x[0],x[1]) for x in tagvotes]))
            output.append(newline + "\n")
            done += 1
        
        #printStatus(INFO, '%d %s %s' % (done,qry_id,' '.join(['%s:%g' % (x[0],x[1]) for x in tagvotes[:3]] )))
        fw.write("".join(output))
        fw.flush()
  
    fw.close()
    printStatus(INFO, 'done')
    
    
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] testCollection trainCollection feature""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    #parser.add_option("--k", default=DEFAULT_K, type="int", help="nr of examples for KDE (default: %d)" % DEFAULT_K)
    parser.add_option("--tpp", default=DEFAULT_TPP, type="string", help="tag preprocess, can be raw, stem, or lemm (default: %s)" % DEFAULT_TPP)
    parser.add_option("--numjobs", default=1, type="int", help="number of jobs")
    parser.add_option("--job", default=1, type="int", help="current job")
    #parser.add_option("--doRandomwalk", default=DEFAULT_RW, type="int", help="do random walk (default: %d)" % DEFAULT_RW)
    #parser.add_option("--uniqueUser", default=DEFAULT_UU, type="int", help="unique user constraint (default: %d)" % DEFAULT_UU)
    parser.add_option("--donefile", default=None, type="string", help="to ignore images that have been done")
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 3:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1], args[2])


if __name__ == "__main__":
    sys.exit(main())  
