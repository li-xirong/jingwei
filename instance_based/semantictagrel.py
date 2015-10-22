import sys
import os

from basic.constant import ROOT_PATH
from util.tagsim.flickr_similarity import CorpusSim, JaccardSim, FlickrContextSim
from util.tagsim.wordnet_similarity import WordnetSim
from util.tagsim.combined_similarity import AvgCombinedSim, MulCombinedSim


class SemanticTagrelLearner:

    def __init__(self, collection, useWnVob=1, rootpath=ROOT_PATH):
        self.engine = None
        if useWnVob:
            if collection in str.split('train10k train100k train1m'):
                vobfile = os.path.join(rootpath, collection, "TextData", "wn.%s.txt" % 'train1m')
            else:
                vobfile = os.path.join(rootpath, collection, "TextData", "wn.%s.txt" % collection)
            self.vob = set(map(str.strip, open(vobfile).readlines()))
            print ("[%s] vob size %d" % (self.__class__.__name__, len(self.vob)))
        else:
            self.vob = []
            print ("[%s] unlimited vob" % self.__class__.__name__)
        
        
    def getWeight(self, tag):
        return 1.0
    
    def computeSimilarity(self, tagx, tagy):
        return self.engine.compute(tagx, tagy)
    
    def computeSemanticFieldw(self, concept, weightedTagSeq):
        scores = [w*self.computeSimilarity(concept, tag) for (tag,w) in weightedTagSeq]
        Z = sum([w for (tag,w) in weightedTagSeq])
        return sum(scores)/float(Z)

    def computeSemanticField(self, concept, tagSeq):
        scores = [self.computeSimilarity(concept, tag) for tag in tagSeq]
        #print concept, '->', zip(tagSeq, scores)
        Z = len(tagSeq) + 1e-10
        relscore = sum(scores) / Z
        #print relscore, '\n'
        return relscore
                
    def estimate(self, tags):
        tagSeq = str.split(tags)
        if self.vob:
            tagSeq = [x for x in tagSeq if x in self.vob]
        conceptSet = set(tagSeq)
        #weightedTagSeq = [(tag,self.getWeight(tag)) for tag in tagSeq]
        #tagvotes = [(concept, self.computeSemanticField(concept,weightedTagSeq)) for concept in conceptSet]
        tagvotes = [(concept, self.computeSemanticField(concept, tagSeq)) for concept in conceptSet]
        tagvotes.sort(key=lambda v:(v[1]), reverse=True)
        return tagvotes
                
'''
Wns: Wordnet based similarity
'''
class WnsTagrelLearner (SemanticTagrelLearner):
    
    def __init__(self, collection, useWnVob=1, sim="wup", rootpath=ROOT_PATH):
        SemanticTagrelLearner.__init__(self, collection, useWnVob, rootpath)
        self.engine = WordnetSim(sim)
        

'''
Fcs: Flickr context similarity
'''
class FcsTagrelLearner (SemanticTagrelLearner):
    
    def __init__(self, collection, useWnVob=1, rootpath=ROOT_PATH):
        SemanticTagrelLearner.__init__(self, collection, useWnVob, rootpath)
        self.engine = FlickrContextSim(collection, rootpath)


'''
AvgCos: Average Combined similarity. Given a pair of tags, tagx and tagy, their combination similarity is AvgCombinedSim(tagx,tagy)
'''    
class AvgCosTagrelLearner (SemanticTagrelLearner):
    
    def __init__(self, collection, useWnVob=1, rootpath=ROOT_PATH):
        SemanticTagrelLearner.__init__(self, collection, useWnVob, rootpath)      
        self.engine = AvgCombinedSim(collection, rootpath)


class MulCosTagrelLearner (SemanticTagrelLearner):
    
    def __init__(self, collection, useWnVob=1, rootpath=ROOT_PATH):
        SemanticTagrelLearner.__init__(self, collection, useWnVob, rootpath)      
        self.engine = MulCombinedSim(collection, rootpath)


SIM_TO_TAGREL = {"wns":WnsTagrelLearner, "fcs":FcsTagrelLearner, "avgcos":AvgCosTagrelLearner, "mulcos":MulCosTagrelLearner}

if __name__ == "__main__":
    rootpath=ROOT_PATH
    collection = "train10k"
    
    wnstagrel = WnsTagrelLearner(collection, useWnVob=1, sim="wup", rootpath=rootpath)
    #fcstagrel = FcsTagrelLearner(collection, rootpath)
    avgcostagrel = AvgCosTagrelLearner(collection, rootpath)
    #mulcostagrel = MulCosTagrelLearner(collection, rootpath)
    
    content = 'bird nest swallow barnswallow'
    for tags in [content]:
        print wnstagrel.estimate(tags)
        #print fcstagrel.estimate(tags)
        print avgcostagrel.estimate(tags)
        #print mulcostagrel.estimate(tags)
        print ""

