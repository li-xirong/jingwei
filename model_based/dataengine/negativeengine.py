import sys
import os
import random

from basic.constant import ROOT_PATH
from basic.common import readRankingResults,printError,printStatus
from dataengine import DataEngine
from knowledge import wn_expand

class NegativeEngine (DataEngine):

    def __init__(self, collection, rootpath=ROOT_PATH):
        DataEngine.__init__(self, collection, rootpath)
        tagfile = os.path.join(rootpath, collection, 'TextData', 'id.userid.lemmtags.txt')
        self.data = map(str.strip, open(tagfile).readlines())
         
    def precompute_annotator(self, concept):
        self.annotator = set([concept] + concept.split('-'))
        self.tabooImset = set()


    def precompute(self, concept):
        self.precompute_annotator(concept)
        self.candidateset = []

        for i,line in enumerate(self.data):
            elems = str.split(line)
            imageid = elems[0]
            if imageid in self.tabooImset:
                continue

            negative = 1

            for tag in elems[1:]:
                if tag in self.annotator:
                    negative = 0
                    break

            if negative:
                self.candidateset.append(imageid)

        self.candidateset = [x for x in self.candidateset if x in self.imset]
        self.target = concept
        INFO = 'dataengine.%s' % self.__class__.__name__
        printStatus(INFO, "%d candidates for %s" % (self.getCount(concept), concept))




class WnNegativeEngine (NegativeEngine):

    def precompute_annotator(self, concept):
        NegativeEngine.precompute_annotator(self, concept)
        for subconcept in concept.split('-'):
            expandedTagSet = set([subconcept] + wn_expand(subconcept))
            self.annotator = self.annotator.union(expandedTagSet)
        INFO = 'dataengine.%s' % self.__class__.__name__
        printStatus(INFO, 'precomputing the virtual annotator for %s: %d tags' % (concept, len(self.annotator)))


class CoNegativeEngine (NegativeEngine):
    def precompute_annotator(self, concept):
        INFO = 'dataengine.%s.precompute_annotator'%self.__class__.__name__
        topn = 100
        NegativeEngine.precompute_annotator(self, concept)
        
        for subconcept in concept.split('-'):
            expandedTagSet = set([subconcept] + wn_expand(subconcept))
            try:
                datafile = os.path.join(ROOT_PATH, self.collection, 'SimilarityIndex', 'ngd', '%s.txt' % subconcept)
                rankedtags = readRankingResults(datafile)
                expandedTagSet = expandedTagSet.union(set([x[0] for x in rankedtags[:topn]]))
            except:
                printError(INFO, 'failed to load ranktag file for %s' % subconcept)
            self.annotator = self.annotator.union(expandedTagSet)
        printStatus(INFO, 'precomputing the virtual annotator for %s: %d tags' % (concept, len(self.annotator)))


STRING_TO_NEGATIVE_ENGINE = {'wn':WnNegativeEngine, 'co':CoNegativeEngine}

if __name__ == '__main__':
    collection = 'train10k'
    #wnne = WnNegativeEngine('msr2013train')
    wnne = WnNegativeEngine(collection)
    cone = CoNegativeEngine(collection)
    
    for concept in str.split('dog animal car airplane tvmonitor car-street car-showroom car-snow'):
        print concept
        wnne.sample(concept, 5000)
        cone.sample(concept, 5000)
        
