

import sys
import os
import random

from basic.constant import ROOT_PATH
from basic.common import readRankingResults,printStatus
from dataengine import DataEngine

class PositiveEngine (DataEngine):

    def __init__(self, collection, rootpath=ROOT_PATH):
        DataEngine.__init__(self, collection)
        self.name = '%s.%s' % (self.__class__.__name__, collection)

    def precompute(self, concept):
        datafile = os.path.join(self.datadir, 'tagged,lemm', concept + ".txt")
        newset = map(str.strip, open(datafile).readlines())
        self.candidateset = [x for x in newset if x in self.imset]
        self.target = concept
        print ("[%s] precomputing candidate positive examples for %s: %d instances" % (self.name, concept, len(self.candidateset)))


class SelectivePositiveEngine (PositiveEngine):

    def __init__(self, collection, method, rootpath=ROOT_PATH):
        PositiveEngine.__init__(self, collection)
        self.name = '%s.%s.%s' % (self.__class__.__name__, collection, method)
        self.datadir = os.path.join(rootpath, collection, 'SimilarityIndex', collection, method)
    
    def precompute(self, concept):
    	  print ("[%s] precomputing candidate positive examples for %s" % (self.name, concept))
    	  datafile = os.path.join(self.datadir, '%s.txt' % concept)
    	  ranklist = readRankingResults(datafile)
    	  self.candidateset = [x[0] for x in ranklist]
    	  self.target = concept
    	  
    def sample(self, concept, n):
        if self.target != concept:
            self.precompute(concept)
        
        if len(self.candidateset) <= n:
            print ("[%s] request %d examples of %s, but %d available only :(" % (self.name, n, concept, len(self.candidateset)))
            return list(self.candidateset)          
        return self.candidateset[:n]
        	

if __name__ == "__main__":
    collection = "train10k"
    method = "tagged,lemm/%s/vgg-verydeep-16-fc7relu,cosineknn,1000,lemm" % collection
    pe1 = PositiveEngine(collection)

    pe2 = SelectivePositiveEngine(collection, method)
    for concept in str.split('airplane dog'):
        pe1.sample(concept, 100)
        pe2.sample(concept, 100)
   

    
