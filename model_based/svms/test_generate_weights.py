import sys
import os
import time

from basic.common import ROOT_PATH,checkToSkip,makedirsforfile
from basic.util import readImageSet
from simpleknn.bigfile import BigFile, StreamFile
from basic.annotationtable import readConcepts,readAnnotationsFrom


rootpath = ROOT_PATH

trainCollection = 'voc2008train'
trainAnnotationName = 'conceptsvoc2008train.txt'
modelName = 'fik50'
modelName = 'fastlinear'
modelName = sys.argv[1]
feature = 'dsift'
weight_dir = os.path.join(rootpath, trainCollection, 'l2r', modelName)

concepts = readConcepts(trainCollection,trainAnnotationName,rootpath=rootpath)
nr_of_models = 5

for concept in concepts:
    weight_file = os.path.join(weight_dir, '%s.txt' % concept)
    makedirsforfile(weight_file)
    weights = [1.0/nr_of_models] * nr_of_models
    model = os.path.join(trainCollection, 'Models', 'conceptsvoc2008train.txt', feature, modelName)
    models = [model] * nr_of_models
    fw = open(weight_file, 'w')
    fw.write('\n'.join(['%g %s' % (w,m) for w,m in zip(weights, models)]))
    fw.close()

