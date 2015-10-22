import sys
import os
import random

from basic.constant import ROOT_PATH
from basic.common import printStatus

class DataEngine:

    def __init__(self, collection, rootpath=ROOT_PATH):
        self.name = '%s.%s' % (self.__class__.__name__, collection)
        imsetfile = os.path.join(rootpath, collection, "ImageSets", "%s.txt" % collection) 
        self.imset = set(map(str.strip, open(imsetfile).readlines()))

        holdoutfile = os.path.join(rootpath, collection, "ImageSets", "holdout.txt") 
        holdoutSet = set(map(str.strip, open(holdoutfile).readlines()))
        printStatus(self.name, '%d examples, %d holdout' % (len(self.imset), len(holdoutSet)))

        self.collection = collection
        self.target = None
        self.imset = set([x for x in self.imset if x not in holdoutSet])
        self.candidateset = sorted(list(self.imset))
        self.datadir = os.path.join(rootpath, collection)
        


    def precompute(self, concept):
        self.annotator = set()
        self.target = concept
  
    def getCount(self, concept):
        if self.target != concept:
            self.precompute(concept)
        return len(self.candidateset)      

    def sample(self, concept, n):
        if self.target != concept:
            self.precompute(concept)
        
        if len(self.candidateset) <= n:
            print ("[%s] request %d examples of %s, but %d available only :(" % (self.name, n, concept, len(self.candidateset)))
            return list(self.candidateset)          
        return random.sample(self.candidateset, n)


if __name__ == "__main__":

    collection = "geoflickr1m"
    dataengine = DataEngine(collection)
    for concept in str.split("car horse street"):
        print concept, dataengine.getCount(concept)


