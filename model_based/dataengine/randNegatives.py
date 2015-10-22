import os
from basic.common import ROOT_PATH,readRankingResults,checkToSkip
from basic.annotationtable import readConcepts,readAnnotationsFrom,writeAnnotationsTo,writeConceptsTo
from negativeengine import WnNegativeEngine

if __name__ == '__main__':
    rootpath = ROOT_PATH
    collection = 'geoflickr1m'
    numPos = 1000
    numNeg = numPos
    T = 10
    overwrite = 0

    sourceAnnotationName = 'concepts88.rand%d.0.randwn%d.0.txt' % (numPos, numPos*5)
    newAnnotationName = 'concepts88.rand%d.0.randwn%d.' % (numPos,numNeg) + '%d.txt'

    concepts = readConcepts(collection, sourceAnnotationName, rootpath)
    ne = WnNegativeEngine(collection)

    for concept in concepts:
        names,labels = readAnnotationsFrom(collection,sourceAnnotationName,concept,rootpath)
        positiveSet = [x[0] for x in zip(names,labels) if 1 == x[1]]
        for t in range(T):
             newfile = os.path.join(rootpath, collection, 'Annotations', 'Image',  newAnnotationName%t, '%s.txt'%concept)
             if checkToSkip(newfile, overwrite):
                 continue
             negativeSet = ne.sample(concept, len(positiveSet))
             renamed = positiveSet + negativeSet
             relabeled = [1] * len(positiveSet) + [-1] * len(negativeSet)
             writeAnnotationsTo(renamed, relabeled, collection, newAnnotationName%t, concept, rootpath)

    for t in range(T):
        writeConceptsTo(concepts, collection, newAnnotationName%t, rootpath)

    
    
