import os, sys
import h5py
import numpy as np
import cPickle as pickle

from basic.common import ROOT_PATH
from basic.util import readImageSet, bisect_index, getVocabMap
from basic.annotationtable import readConcepts

tagmatrix_file = h5py.File(sys.argv[1], 'r')
pkl_file = open(sys.argv[2], 'w')
workingCollection = sys.argv[3]
annotationName = sys.argv[4]
rootpath = ROOT_PATH

id_images = tagmatrix_file['id_images']
concepts = readConcepts(workingCollection, annotationName, rootpath)
testset_id_images = readImageSet(workingCollection.split('+')[1], workingCollection.split('+')[1], rootpath)
testset_id_images.sort()

if not type(id_images[0]) is str:
	id_images = map(str, id_images)

if not type(testset_id_images[0]) is str:
	testset_id_images = map(str, testset_id_images)

mapping = getVocabMap(list(tagmatrix_file['vocab'][:]),concepts)
predicted_tagmatrix = tagmatrix_file['tagmatrix'][:,mapping]

print "predicted_tagmatrix.shape = ", predicted_tagmatrix.shape
print "len(id_images) = ", len(id_images)
print "len(testset_id_images) = ", len(testset_id_images)

idx = np.array([bisect_index(id_images, x) for x in testset_id_images])
final_tagmatrix = predicted_tagmatrix[idx, :]
id_images = testset_id_images

print "dumping %d elements..." % len(id_images)

pickle.dump({'concepts':concepts, 'id_images':map(int, id_images), 'scores':final_tagmatrix}, pkl_file, pickle.HIGHEST_PROTOCOL)
