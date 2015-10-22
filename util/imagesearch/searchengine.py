import sys
import os
import random
import numpy as np

from basic.constant import ROOT_PATH
from basic.common import checkToSkip,niceNumber,writeRankingResults, printStatus
from basic.annotationtable import readConcepts,readAnnotationsFrom
from basic.metric import getScorer
from datareader import AttributeReader,TagrelReader,AutotagReader,TagrankReader


class SearchEngine:

    def __init__(self, collection, dataset, rootpath=ROOT_PATH):
        self.collection = collection
        self.dataset = dataset
        self.name = "%s(%s,%s)" % (self.__class__.__name__, self.collection, self.dataset)
        self.datadir = os.path.join(rootpath, collection)
        self.outputdir = os.path.join(rootpath, collection, 'SimilarityIndex', dataset) 
        
        try:
            holdoutfile = os.path.join(rootpath, collection, "ImageSets", "holdout.txt")
            self.holdoutset = set(map(str.strip, open(holdoutfile).readlines()))
            print ('[%s] %d holdout images' % (self.name, len(self.holdoutset)))
        except:
            self.holdoutset = None
            
             
    def getName(self):
        return self.name

    def getOutputdir(self):
        return self.outputdir
   

    '''
    By default, SearchEngine scores the entire collection. 
    However, for tag-based search, it scores images which are labeled with the concept.
    '''        
    def readHitlist(self, concept):
        datafile = os.path.join(self.datadir, 'ImageSets', '%s.txt' % self.dataset)
        dataset = map(str.strip, open(datafile).readlines())
        if self.holdoutset:
            return [x for x in dataset if x not in self.holdoutset]
        else:
            return dataset

    '''
    Implement the computeScore function
    '''
    def computeScore(self, concept, photoid):
        return 0

         
    '''
    Do not modify scoreCollection
    '''
    def scoreCollection(self, concept):
        dataset = self.readHitlist(concept)
        searchresults = [(photoid, self.computeScore(concept, photoid)) for photoid in dataset]
        searchresults.sort(key=lambda v:(v[1],v[0]), reverse=True)
        return searchresults



class DetectorSearchEngine (SearchEngine):
    def __init__(self, collection, dataset, detectorMethod, rootpath=ROOT_PATH):
        SearchEngine.__init__(self, collection, dataset, rootpath)
        self.reader = AutotagReader(collection, dataset, detectorMethod, rootpath=rootpath)
        self.name = "%s(%s)" % (self.__class__.__name__, self.reader.name)
        self.outputdir = os.path.join(self.outputdir, detectorMethod)
        
    def computeScore(self, concept, photoid):
        return self.reader.get(photoid, concept)
                

class TagBasedSearchEngine (SearchEngine):

    def __init__(self, collection, dataset, tpp="lemm", rootpath=ROOT_PATH):
        SearchEngine.__init__(self, collection, dataset, rootpath)
        self.name = "%s(%s,%s)" % (self.__class__.__name__, self.collection, tpp)
        self.datadir = os.path.join(rootpath, collection, "tagged,%s" % tpp)
        self.outputdir = os.path.join(self.outputdir, 'tagged,%s' % tpp)
        
        
    def readHitlist(self, concept):
        datafile = os.path.join(self.datadir, '%s.txt' % concept)
        dataset = map(str.strip, open(datafile).readlines())        
        if self.holdoutset:
            return [x for x in dataset if x not in self.holdoutset]
        else:
            return dataset
    
               

class RandomSearchEngine (TagBasedSearchEngine):
    def __init__(self, collection, dataset, tpp="lemm", rootpath=ROOT_PATH):
        TagBasedSearchEngine.__init__(self, collection, dataset, tpp, rootpath)
        self.outputdir = os.path.join(self.outputdir, 'random')
        
    def computeScore(self, concept, photoid):
        return random.random() # Return a floating point number in the range [0.0, 1.0)
    

class RawtagnumSearchEngine (TagBasedSearchEngine):
    
    def __init__(self, collection, dataset, tpp="lemm", rootpath=ROOT_PATH):
        TagBasedSearchEngine.__init__(self, collection, dataset, tpp, rootpath)
        self.reader = AttributeReader(collection, attr="rawtagnum", rootpath=rootpath)
        self.outputdir = os.path.join(self.outputdir, 'rawtagnum')

    def computeScore(self, concept, photoid):
        return 1.0 / self.reader.get(photoid)


class TagrelSearchEngine (TagBasedSearchEngine):
    def __init__(self, collection, dataset, tagrelMethod, tpp='lemm', rootpath=ROOT_PATH):
        TagBasedSearchEngine.__init__(self, collection, dataset, tpp, rootpath)
        self.reader = TagrelReader(collection, dataset, tagrelMethod, nonnegative=0, rootpath=rootpath)
        self.name = "%s(%s)" % (self.__class__.__name__, self.reader.name)
        self.outputdir = os.path.join(self.outputdir, tagrelMethod)
        
    def computeScore(self, concept, photoid):
        return self.reader.get(photoid, concept)


class TagrankSearchEngine (TagBasedSearchEngine):

    def __init__(self, collection, dataset, tagrelMethod, tpp='lemm', rootpath=ROOT_PATH):
        TagBasedSearchEngine.__init__(self, collection, dataset, tpp, rootpath)
        self.rawtagnumreader = AttributeReader(collection, "rawtagnum", rootpath)
        self.reader = TagrankReader(collection, dataset, tagrelMethod, rootpath)
        self.name = "%s(%s)" % (self.__class__.__name__, self.reader.name)
        self.outputdir = os.path.join(self.outputdir, 'tagrank', tagrelMethod)
        
    def computeScore(self, concept, photoid):
        rawtagnum = self.rawtagnumreader.get(photoid)
        rank = self.reader.get(photoid,concept)
        return -rank + (1.0/rawtagnum)



def submit(searchers, collection,annotationName, rootpath=ROOT_PATH, overwrite=0):
    concepts = readConcepts(collection, annotationName, rootpath=rootpath)
    nr_of_runs = len(searchers)

    for concept in concepts:
        for j in range(nr_of_runs):
            resultfile = os.path.join(searchers[j].getOutputdir(), concept + ".txt")
            if checkToSkip(resultfile, overwrite):
                continue
            searchresults = searchers[j].scoreCollection(concept)
            print ("%s: %s %d -> %s" % (searchers[j].name, concept, len(searchresults), resultfile))
            writeRankingResults(searchresults, resultfile)

    printStatus('%s.submit'%os.path.basename(__file__), "done")


def evaluateSearchEngines(searchers, collection, annotationName, metric, rootpath=ROOT_PATH):
    scorer = getScorer(metric)
    concepts = readConcepts(collection, annotationName, rootpath)
    
    nr_of_runs = len(searchers)
    nr_of_concepts = len(concepts)
    results = np.zeros((nr_of_concepts,nr_of_runs))


    for i in range(nr_of_concepts):
        names, labels = readAnnotationsFrom(collection, annotationName, concepts[i], rootpath)
        name2label = dict(zip(names,labels))
        
        for j in range(nr_of_runs):
            searchresults = searchers[j].scoreCollection(concepts[i])
            sorted_labels = [name2label[name] for (name,score) in searchresults if name in name2label]
            results[i,j] = scorer.score(sorted_labels)

    for i in range(nr_of_concepts):
        print concepts[i], ' '.join([niceNumber(x,3) for x in results[i,:]])
    mean_perf = results.mean(0)
    print 'mean%s'%metric, ' '.join([niceNumber(x,3) for x in mean_perf])

    return concepts,results


def submit_and_evaluate(searchers, collection, annotationName,metric,rootpath=ROOT_PATH,overwrite=0):
    submit(searchers,collection,annotationName)
    evaluateSearchEngines(searchers,collection,annotationName,metric)


def dryrun(collection='flickr20',annotationName="concepts20.txt",metric="AP"):
    searchengines = [] #[x(collection,collection) for x in [RawtagnumSearchEngine]]
    tagrelMethods = ['tagpos,lemm']
    for tagrelMethod in tagrelMethods:
        searchengines += [TagrelSearchEngine(collection, collection, tagrelMethod)]
        #searchengines += [TagrankSearchEngine(collection, collection, tagrelMethod)]

    submit_and_evaluate(searchengines,collection,annotationName,metric)
    #submit(searchengines,collection,annotationName)
    #concepts,results=evaluateSearchEngines(searchengines,collection,annotationName,metric)



NAME_TO_ENGINE = {"rawtagnum":RawtagnumSearchEngine, "detector":DetectorSearchEngine, "tagrel":TagrelSearchEngine, "tagrank":TagrankSearchEngine}

def getSearchEngine(name):
    return NAME_TO_ENGINE[name]


if __name__ == "__main__":
    dryrun(collection='flickr81',annotationName='concepts81.txt',metric='AP')


