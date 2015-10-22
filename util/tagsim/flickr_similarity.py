import os,sys
from math import log, exp

from basic.constant import ROOT_PATH
from basic.data import COLLECTION_TO_SIZE,COLLECTION_TO_USERNUM

COLLECTION_TO_GAMMA = {'train10k':(1,1.512), 'train100k':(1,1.262), 'train1m':(1,1.18)}
MASK = (1<<16)-1

def encodeKey(x, y):
    return (x<<16) | y

def decodeKey(key):
    x = key>>16
    y = key & MASK    
    return (x,y)
    
        
def normalized_google_distance(fx, fy, fxy, N):
    assert(fxy > 0)
    logfx  = log(fx, 2)
    logfy  = log(fy, 2)
    logfxy = log(fxy, 2)
    logN   = log(N, 2)
    d = (max(logfx,logfy) - logfxy) / (logN - min(logfx,logfy))

    return d



class CorpusSim:
    def __init__(self, collection, rootpath=ROOT_PATH):
        MIN_USER_FREQ = 10
        self.N = [COLLECTION_TO_USERNUM[collection], COLLECTION_TO_SIZE[collection]]
        #vobfile = os.path.join(rootpath, collection, "TextData", "wn.%s.txt" % collection)
        #vobfile = os.path.join(rootpath, collection, "TextData", "vob.minuser20.%s.txt" % collection)
        #vob = map(str.strip, open(vobfile).readlines())
        tagfreqfile = os.path.join(rootpath, collection, "TextData", "lemmtag.userfreq.imagefreq.txt")
        self.vob = [str.split(x)[0] for x in open(tagfreqfile) if int(str.split(x)[1]) >= MIN_USER_FREQ]
        self.tag2index = dict(zip(self.vob, range(len(self.vob))))
        self.freq = [0] * len(self.vob)
     
        for line in open(tagfreqfile).readlines():
            tag,userfreq,imagefreq = str.split(line)
            idx = self.tag2index.get(tag, -1)
            if idx >= 0:
                self.freq[idx] = (int(userfreq), int(imagefreq))
        print ("[tagsim.flickr_similarity.%s] %d images, %d users, %d tags" % (self.__class__.__name__, self.N[1], self.N[0], len(self.vob)))

    def cleanTags(self, tagSeq):
        return [t for t in tagSeq if self.tag2index.get(t,-1) != -1]
        
    # img=1, return the number of images labeled with the $tag$
    # img=0, return the number of distinct users who have used the $tag$ to label images
    def getFreq(self, tag, img):
        try:
            idx = self.tag2index[tag]
            return self.freq[idx][img]
        except:
            return 0
    
    # img=1, compute IDF using image frequency
    # img=0, compute IDF using user frequency       
    def computeIDF(self, tag, img):
        freq = max(self.getFreq(tag, img), 1)
        return log(float(self.N[img])/freq, 2)


    def getKey(self, tagx, tagy):
        ix = self.tag2index.get(tagx, -1)
        iy = self.tag2index.get(tagy, -1)
        if -1==ix or -1==iy:
            return -1
        if ix < iy:
            return encodeKey(ix, iy)
        else:
            return encodeKey(iy, ix)
            

class JaccardSim (CorpusSim):

    def __init__(self, collection, rootpath=ROOT_PATH):
        CorpusSim.__init__(self, collection, rootpath)
        jointfreqfile =  os.path.join(rootpath, collection, "TextData", "ucij.uuij.icij.iuij.txt")
        #jointfreqfile = os.path.join(rootpath, collection, 'TextData', 'lemmtag.minuser20.joint.union.txt')
        self.jointfreq = {}
        self.unionfreq = {}

        for line in open(jointfreqfile).readlines():
            t1, t2, userjointfreq, userunionfreq, imagejointfreq, imageunionfreq = str.split(line)
            #t1, t2, imagejointfreq, imageunionfreq = str.split(line)
            key = self.getKey(t1, t2)
            if key >= 0:
                #self.jointfreq[key] = (int(userjointfreq), int(imagejointfreq))
                #self.unionfreq[key] = (int(userunionfreq), int(imageunionfreq))
                self.jointfreq[key] = int(imagejointfreq)
                self.unionfreq[key] = int(imageunionfreq)
        print ("[tagsim.flickr_similarity.%s] %d tag pairs" % (self.__class__.__name__, len(self.jointfreq)))


    def getJointFreq(self, tagx, tagy, img):
        try:
            #return self.jointfreq[self.getKey(tagx,tagy)][img]
            return self.jointfreq[self.getKey(tagx,tagy)]
        except:
            return 0

    def getUnionFreq(self, tagx, tagy, img):
        try:
            #return self.unionfreq[self.getKey(tagx,tagy)][img]
            return self.unionfreq[self.getKey(tagx,tagy)]
        except:
            return 0


    def compute(self, tagx, tagy, img=1):
        if tagx == tagy:
            return 1.0
        jointfreq = self.getJointFreq(tagx, tagy, img)
        unionfreq = max(self.getUnionFreq(tagx, tagy, img), 1)
        #print tagx, tagy, jointfreq, unionfreq
        return float(jointfreq) / unionfreq
        

class FlickrContextSim(JaccardSim):

    def __init__(self, collection, rootpath=ROOT_PATH):
        JaccardSim.__init__(self, collection, rootpath)
        self.gamma = COLLECTION_TO_GAMMA[collection]
 
    def computeNGD(self, tagx, tagy, img):
        if tagx == tagy:
            return 0

        fxy = self.getJointFreq(tagx, tagy, img)
        if 0 == fxy:
            return 1e6
        fx = self.getFreq(tagx, img)
        fy = self.getFreq(tagy, img)
        d = normalized_google_distance(fx, fy, fxy, self.N[img])
        #print ("%s.computeNGD(%s, %s, img=%d, fx=%d,fy=%d,fxy=%d)=%g" % (self.__class__.__name__,tagx,tagy,img,fx,fy,fxy,d))
        return d

    def compute(self, tagx, tagy, gamma=None, img=1):
        if gamma:
            score = exp(-gamma*self.computeNGD(tagx,tagy,img))
        else:
            score = exp(-self.gamma[img]*self.computeNGD(tagx,tagy,img))
        #print ("%s.compute(%s, %s, img=%d)=%g" % (self.__class__.__name__,tagx,tagy,img,score)),gamma
        return score
            
if __name__ == "__main__":
    for x,y in [(1,3), (1, 10000), (65100, 65200), (200, 100)]:
        key = encodeKey(x, y)
        print x, y, "->", key, decodeKey(key)

    from basic.common import ROOT_PATH
    collection = sys.argv[1] #train100k

    fcs = FlickrContextSim(collection, rootpath=ROOT_PATH)
    tags = str.split('nature waterfall mountain 2012 airshow bmw beach food dog car cat animal street')
    tags = str.split('happy')
    vob = fcs.vob
    for tagx in tags:
        taglist = [(tagy,fcs.compute(tagx,tagy)) for tagy in vob] #fcs.vob]
        taglist.sort(key=lambda v:v[1], reverse=True)
        print tagx, ' '.join(['%s %g' % (x[0],x[1]) for x in taglist[:10]])
            
