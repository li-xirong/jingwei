
from basic.common import checkToSkip
from basic.annotationtable import readConcepts

testCollections = 'mirflickr08 flickr55 flickr81'.split()
annotationNames = 'conceptsmir14.txt concepts51ms.txt concepts81.txt'.split()

all_concepts = []

for collection,annotationName in zip(testCollections, annotationNames):
    concepts = readConcepts(collection, annotationName)
    all_concepts += concepts

all_concepts = sorted(list(set(all_concepts)))
concepts130 = readConcepts('train10k', 'concepts130.txt')

print ('nr of unique concepts: %d' % len(all_concepts))
print ('exceptions: %s' % ' '.join([x for x in all_concepts if x not in concepts130]))
print ('extra: %s' % ' '.join([x for x in concepts130 if x not in all_concepts]))

