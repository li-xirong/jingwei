import sys
import os
from basic.common import ROOT_PATH,readRankingResults
from basic.annotationtable import readConcepts,readAnnotationsFrom,writeAnnotationsTo,writeConceptsTo

if __name__ == '__main__':
    rootpath = ROOT_PATH
    collection = 'geoflickr1m'
    collection = 'flickr1m'
    #collection = 'web13train'
    #collection = 'tentagv10dev'
    collection = sys.argv[1] #'msr2013train'
    
    conceptSetName = 'concepts88'
    conceptSetName = 'biconcepts15'
    conceptSetName = 'concepts11'
    conceptSetName = 'conceptsweb13'
    conceptSetName = 'conceptstentagv10dev'
    conceptSetName = 'concepts100' 
    #conceptSetName = 'conceptsweb15'
    
    T = 10
    numPos = 100
    numNeg = numPos * 10

    sourceAnnotationName = '%s.rand%d.0.randco%d.' % (conceptSetName, numPos, numNeg) + '%d.txt'
    sourceAnnotationName = '%s.rand%d.0.randwn%d.' % (conceptSetName, numPos, numNeg) + '%d.txt'  

    #posName = 'dsift-1000nn' + str(numPos)
    #tagrelMethod = 'dsiftpca225,knn,1000,lemm'
    posName = 'borda-cos-dsiftpca' + str(numPos)
    posName = 'borda-fcsidf-multipca' + str(numPos)
    tagrelMethod = 'borda-cos-dsiftpca'
    tagrelMethod = 'borda-fcsidf-multipca'
    
    removeBatchTagged = 0
    if removeBatchTagged:
        posName = 'multipcanobt' + str(numPos)
        tagrelMethod = 'tentagv10dev/multipca,knn,1000,lemm/nobt' 
    else:
        posName = 'multipca' + str(numPos)
        tagrelMethod = 'tentagv10dev/multipca,knn,1000,lemm'
    
    #sourceAnnotationName = '%s.rand%d.0.randwn%d.' % (conceptSetName, numPos, numNeg) + '%d.txt'
    #posName = 'rgbsift' + str(numPos)
    #tagrelMethod = 'web13train/rgbsift,knn,1000,w'
    #posName = 'txt' + str(numPos)
    #tagrelMethod = 'textual'
    posName = 'clickcount' + str(numPos)
    tagrelMethod = 'clickcount'
    
    #posName = 'ccgd' + str(numPos)
    #tagrelMethod = 'flickr1m/ccgd,knn,1000'

    concepts = readConcepts(collection, sourceAnnotationName%0, rootpath)

    holdoutfile = os.path.join(rootpath, collection, "ImageSets", "holdout.txt") 
    holdoutSet = set(map(str.strip, open(holdoutfile).readlines()))
    print ('%s holdout %d' % (collection,len(holdoutSet)))
 
    for concept in concepts:
        simfile = os.path.join(rootpath, collection, 'SimilarityIndex', collection, 'tagged,lemm', tagrelMethod, '%s.txt' % concept)
        searchresults = readRankingResults(simfile)
        searchresults = [x for x in searchresults if x[0] not in holdoutSet]
        positiveSet = [x[0] for x in searchresults[:numPos]]
                
        for t in range(T):
            newAnnotationName = sourceAnnotationName % t
            newAnnotationName = newAnnotationName.replace('rand%d.0'%numPos, posName)
            names,labels = readAnnotationsFrom(collection,sourceAnnotationName%t,concept,rootpath)
            
            negativeSet = [x[0] for x in zip(names,labels) if -1 == x[1]]
            renamed = positiveSet + negativeSet
            relabeled = [1] * len(positiveSet) + [-1] * len(negativeSet)
            print ('[%s] %s +%d, -%d -> %s' % (concept,sourceAnnotationName % t,len(positiveSet),len(negativeSet),newAnnotationName)) 
            writeAnnotationsTo(renamed, relabeled, collection, newAnnotationName, concept, rootpath)
            
    for t in range(T):
        newAnnotationName = sourceAnnotationName % t
        newAnnotationName = newAnnotationName.replace('rand%d.0'%numPos, posName)
        writeConceptsTo(concepts, collection, newAnnotationName, rootpath)
        
