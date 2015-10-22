
import sys
import os
#import subprocess

pwd = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(pwd)
sys.path.append(parent_dir)

test_tags = str.split('child face insect')
trainCollection = 'train10k'
trainAnnotationName = 'conceptsmm15tut.txt'
feature = "vgg-verydeep-16-fc7relul2"
testCollection = 'mirflickr08'


from basic.constant import ROOT_PATH
rootpath = ROOT_PATH
conceptfile = os.path.join(rootpath, trainCollection, 'Annotations', trainAnnotationName)

from basic.annotationtable import writeConceptsTo
writeConceptsTo(test_tags, trainCollection, trainAnnotationName)

cmd = '%s/util/imagesearch/obtain_labeled_examples.py %s %s' % (parent_dir, trainCollection, conceptfile)
os.system('python ' + cmd)

train_feat_dir = os.path.join(rootpath, trainCollection, 'FeatureData', feature)
from util.simpleknn.bigfile import BigFile
train_feat_file = BigFile(train_feat_dir) 
feat_dim = train_feat_file.ndims

from basic.util import readImageSet
test_imset = readImageSet(testCollection)
test_feat_dir = os.path.join(rootpath, testCollection, 'featureData', feature)
test_feat_file = BigFile(test_feat_dir)
#test_renamed, test_vectors = test_feat_file.read(test_imset)


from model_based.dataengine.positiveengine import PositiveEngine
from model_based.dataengine.negativeengine import NegativeEngine

pe = PositiveEngine(trainCollection)
ne = NegativeEngine(trainCollection)

for tag in test_tags:
    pos_set = pe.sample(tag, 100)
    neg_set = ne.sample(tag, 100)
    names = pos_set + neg_set
    labels = [1] * len(pos_set) + [-1] * len(neg_set)
    name2label = dict(zip(names,labels))
    
    (renamed, vectors) = train_feat_file.read(names)
    y = [name2label[x] for x in renamed]
    
    print 'training %s' % tag
    from model_based.svms.fastlinear.liblinear193.python.liblinearutil import train
    from model_based.svms.fastlinear.fastlinear import liblinear_to_fastlinear
    svm_params = '-s 2 -B -1 -q'
    model = train(y, vectors, svm_params)
    fastmodel = liblinear_to_fastlinear([model], [1.0], feat_dim)

    # optionally save the learned model to disk
    from model_based.svms.fastlinear.fastlinear import fastlinear_save_model
    model_dir = os.path.join(rootpath, trainCollection, 'Models', trainAnnotationName, feature, 'fastlinear')
    model_filename = os.path.join(model_dir, '%s.model' % tag)
    
    from basic.common import makedirsforfile
    makedirsforfile(model_filename)
    fastlinear_save_model(model_filename, fastmodel)

    print 'applying %s' % tag
    from model_based.svms.mlengine_util import classify_large_data
    ranklist = classify_large_data(fastmodel, test_imset, test_feat_file)    
    #predict_scores = [fastmodel.predict(x) for x in test_vectors]
    #ranklist = sorted(zip(test_renamed, predict_scores), key=lambda v:(v[1],v[0]), reverse=True)
    
    from basic.common import writeRankingResults
    simdir = os.path.join(rootpath, testCollection, 'SimilarityIndex', testCollection, trainCollection, 'conceptsmm15tut.txt', '%s,fastlinear'%feature)
    resultfile = os.path.join(simdir, '%s.txt' % tag)
    writeRankingResults(ranklist, resultfile)

    