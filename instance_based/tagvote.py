import sys, os, time, random

from basic.constant import ROOT_PATH
from basic.common import printStatus, readRankingResults
from basic.util import readImageSet
from basic.annotationtable import readConcepts
from util.simpleknn import simpleknn
#from sandbox.pquan.pqsearch import load_model

INFO = __file__

DEFAULT_K=1000
DEFAULT_TPP = 'lemm'
DEFAULT_DISTANCE = 'l1'
DEFAULT_BLOCK_SIZE = 1000
DEFAULT_TAGGER = 'tagvote'

class TagVoteTagger:
    def __init__(self, collection, annotationName, feature, distance, tpp=DEFAULT_TPP, rootpath=ROOT_PATH):
        self.concepts = readConcepts(collection, annotationName, rootpath)
        self.nr_of_concepts = len(self.concepts)
        self.concept2index = dict(zip(self.concepts, range(self.nr_of_concepts)))
        
        feat_dir = os.path.join(rootpath, collection, "FeatureData", feature)
        id_file = os.path.join(feat_dir, 'id.txt')
        shape_file = os.path.join(feat_dir, 'shape.txt')
        self.nr_of_images, feat_dim = map(int, open(shape_file).readline().split())

        self.searcher = simpleknn.load_model(os.path.join(feat_dir, 'feature.bin'), feat_dim, self.nr_of_images, id_file)
        self.searcher.set_distance(distance)
        self.k = DEFAULT_K
        
        self._load_tag_data(collection, tpp, rootpath)
        
        printStatus(INFO, "%s, %d images, %d unique tags, %s %d neighbours for voting" % (self.__class__.__name__, self.nr_of_images, len(self.tag2freq), distance, self.k))
    
    
    def _load_tag_data(self, collection, tpp, rootpath):
        tagfile = os.path.join(rootpath, collection, "TextData", "id.userid.%stags.txt" % tpp)
        self.textstore = {}
        self.tag2freq = {}
        for line in open(tagfile):
            imageid, userid, tags = line.split('\t')
            tags = tags.lower()
            self.textstore[imageid] = (userid, tags)
            tagset = set(tags.split())
            for tag in tagset:
                self.tag2freq[tag] = self.tag2freq.get(tag,0) + 1
    
    def tagprior(self, tag):
        return float(self.k) * self.tag2freq.get(tag,0) / self.nr_of_images
    
    
    def _get_neighbors(self, content, context):
        return self.searcher.search_knn(content, max_hits=max(3000,self.k*3))
    
    def _compute(self, content, context=None):
        users_voted = set()
        vote = [0-self.tagprior(c) for c in self.concepts] # vote only on the given concept list
        voted = 0
        skip = 0

        neighbors = self._get_neighbors(content, context)
        
        for (name, dist) in neighbors:
            (userid,tags) = self.textstore.get(name, (None, None))
            if tags is None or userid in users_voted:
                skip += 1
                continue
            users_voted.add(userid)
            tagset = set(tags.split())
            for tag in tagset:
                c_idx = self.concept2index.get(tag, -1)
                if c_idx >= 0:
                    vote[c_idx] += 1
            voted += 1
            if voted >= self.k:
                break
        #assert(voted >= self.k), 'too many skips (%d) in %d neighbors' % (skip, len(neighbors))
        return vote
        
    def predict(self, content, context=None):
        scores = self._compute(content, context)
        return sorted(zip(self.concepts, scores), key=lambda v:v[1], reverse=True)


class PreTagVoteTagger (TagVoteTagger):
    def __init__(self, collection, annotationName, feature, distance, tpp=DEFAULT_TPP, rootpath=ROOT_PATH):
        self.rootpath = rootpath
        self.concepts = readConcepts(collection, annotationName, rootpath)
        self.nr_of_concepts = len(self.concepts)
        self.concept2index = dict(zip(self.concepts, range(self.nr_of_concepts)))
        
        self.imset = readImageSet(collection, collection, rootpath)
        self.nr_of_images = len(self.imset)
        self.knndir = os.path.join(collection, '%s,%sknn,uu,1500' % (feature, distance))
        
        self.k = DEFAULT_K
        self.noise = 0
        
        self._load_tag_data(collection, tpp, rootpath)
        
        printStatus(INFO, "%s, %d images, %d unique tags, %s %d neighbours for voting" % (self.__class__.__name__, self.nr_of_images, len(self.tag2freq), distance, self.k))

    def set_noise(self, noise):
        self.noise = noise
        
    def _get_neighbors(self, content, context):
        testCollection,testid = context.split(',')
        knnfile = os.path.join(self.rootpath, testCollection, 'SimilarityIndex', testCollection, self.knndir, testid[-2:], '%s.txt' % testid)
        knn = readRankingResults(knnfile)
        knn = knn[:self.k]
        if self.noise > 1e-3:
            n = int(len(knn) * self.noise)
            hits = random.sample(xrange(len(knn)), n)
            random_set = random.sample(self.imset, n)
            for i in range(n):
                idx = hits[i]
                knn[idx] = (random_set[i], 1000)
        return knn
        
class PreKnnTagger (PreTagVoteTagger):
    def __init__(self, collection, annotationName, feature, distance, tpp=DEFAULT_TPP, rootpath=ROOT_PATH, k = DEFAULT_K):
        self.rootpath = rootpath
        self.concepts = readConcepts(collection, annotationName, rootpath)
        self.nr_of_concepts = len(self.concepts)
        self.concept2index = dict(zip(self.concepts, range(self.nr_of_concepts)))

        self.imset = readImageSet(collection, collection, rootpath)
        self.nr_of_images = len(self.imset)
        self.knndir = os.path.join(collection, '%s,%sknn,1500' % (feature, distance))

        self.k = k
        self.noise = 0

        self._load_tag_data(collection, tpp, rootpath)

        printStatus(INFO, "%s, %d images, %d unique tags, %s %d neighbours for voting" % (self.__class__.__name__, self.nr_of_images, len(self.tag2freq), distance, self.k))


    def _compute(self, content, context=None):
        vote = [0] * self.nr_of_concepts # vote only on the given concept list
        voted = 0
        skip = 0

        neighbors = self._get_neighbors(content, context)

        for (name, dist) in neighbors:
            (userid,tags) = self.textstore.get(name, (None, None))
            if tags is None:
                skip += 1
                continue
            tagset = set(tags.split())
            for tag in tagset:
                c_idx = self.concept2index.get(tag, -1)
                if c_idx >= 0:
                    vote[c_idx] += 1
            voted += 1
            if voted >= self.k:
                break
        #assert(voted >= self.k), 'too many skips (%d) in %d neighbors' % (skip, len(neighbors))
        return vote



class PqTagVoteTagger (TagVoteTagger):
    def __init__(self, collection, annotationName, feature, distance, tpp=DEFAULT_TPP, rootpath=ROOT_PATH):
        self.rootpath = rootpath
        self.concepts = readConcepts(collection, annotationName, rootpath)
        self.nr_of_concepts = len(self.concepts)
        self.concept2index = dict(zip(self.concepts, range(self.nr_of_concepts)))

        featuredir = os.path.join(rootpath,collection,'FeatureData',feature)
        id_file = os.path.join(featuredir, "id.txt")
        shape_file = os.path.join(feat_dir, 'shape.txt')
        self.nr_of_images, feat_dim = map(int, open(shape_file).readline().split())

        self.searcher = load_model(featuredir, self.nr_of_images, feat_dim,nr_of_segments=512,segmentk=256,coarsek=4096)
        self.k = DEFAULT_K
        self._load_tag_data(collection, tpp, rootpath)
        printStatus(INFO, "%s, %d images, %d unique tags, %s %d neighbours for voting" % (self.__class__.__name__, self.nr_of_images, len(self.tag2freq), distance,  self.k))

    def _get_neighbors(self, content, context):
        return self.searcher.search_knn(content, requested=max(3000, self.k*3))
        
    
NAME_TO_TAGGER = {'tagvote':TagVoteTagger, 'pretagvote':PreTagVoteTagger, 'preknn':PreKnnTagger, 'pqtagvote':PqTagVoteTagger}        
   
        
if __name__ == '__main__':
    feature = 'vgg-verydeep-16-fc7relu'
    tagger = TagVoteTagger('train10k', 'concepts81.txt', feature, 'cosine')
    tagger = PreTagVoteTagger('train10k', 'concepts81.txt', feature, 'cosine')
    tagger = PreKnnTagger('train10k', 'conceptsmir14.txt', feature, 'cosine')
 
    
    
