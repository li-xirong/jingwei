import sys
import os
import time
import numpy as np


from basic.constant import ROOT_PATH
from basic.common import printStatus, makedirsforfile
from basic.annotationtable import readConcepts

from fastlinear import fastlinear_load_model as load_model
from model_based.svms.probabilistic import sigmoid_predict

INFO = __file__

class ModelArray:

    def __init__(self, trainCollection, trainAnnotationName, feature, modelName, rootpath=ROOT_PATH):
        assert(modelName.startswith('fastlinear')), modelName
        self.concepts = readConcepts(trainCollection, trainAnnotationName, rootpath=rootpath)
        self.nr_of_concepts = len(self.concepts)
        modeldir = os.path.join(rootpath, trainCollection, 'Models', trainAnnotationName, feature, modelName)
        model = load_model(os.path.join(modeldir, self.concepts[0]+'.model'))
        self.feat_dim = model.get_feat_dim()
         
        self.W = np.zeros((self.feat_dim, self.nr_of_concepts))
        self.AB = np.zeros((2, self.nr_of_concepts))
        for i in range(self.nr_of_concepts):
            model_file_name = os.path.join(modeldir, "%s.model" % self.concepts[i])
            model = load_model(model_file_name)
            self.W[:,i] = model.get_w()
            [A,B] = model.get_probAB()
            self.AB[:,i] = [A,B] if abs(A)>1e-8 else [-1,0]
            printStatus(INFO, '%s, A=%g, B=%g' % (self.concepts[i], A, B))
        printStatus(INFO, '%s-%s-%s -> %dx%d ModelArray' % (trainCollection,trainCollection,feature,self.feat_dim,self.nr_of_concepts))
        
        
    def predict(self, test_feat_vecs, topk=-1, prob=0):
        #input_feat = np.array(test_feat_vecs)
        nr_of_test = len(test_feat_vecs)
        topk = topk if topk>0 else self.nr_of_concepts
        #threshold = self.nr_of_concepts - topk
        A = np.array(test_feat_vecs).dot(self.W)

        if prob:
            for row in range(nr_of_test):
                A[row,:] = [sigmoid_predict(A[row,j], self.AB[0,j], self.AB[1,j]) for j in range(self.nr_of_concepts)]

        B = np.argsort(A, axis=1)
        res = []
        
        for row in range(nr_of_test):
            ranklist = []
            for column in range(-1, -topk-1, -1):
                idx = B[row,column]
                ranklist.append((self.concepts[idx], A[row,idx]))
            ranklist.sort(key=lambda v:v[1], reverse=True)
            res.append(ranklist)

        return res


if __name__ == '__main__':
    rootpath = '/Users/xirong/VisualSearch'
    rootpath = 'e:/xirong/VisualSearch'
    collection = 'voc2008train'
    annotationName = 'conceptsvoc2008train.txt'
    feature = 'dsift'
    modelName = 'fastlinear'
    #modelName = 'fastlinear-tuned'
    ma = ModelArray(collection, annotationName, feature, modelName, rootpath=rootpath)

    testCollection = 'voc2008val'
    testAnnotationName = 'conceptsvoc2008val.txt'
    from basic.util import readImageSet
    testset = readImageSet(testCollection, testCollection, rootpath)
 
    from simpleknn.bigfile import BigFile
    feat_file = BigFile(os.path.join(rootpath, testCollection, 'FeatureData', feature))
 
    renamed, vectors = feat_file.read(testset)
    res = ma.predict(vectors,prob=0)

    # re-organize the result per concept
    ranklist = {}

    for i in range(len(renamed)):
        test_id = renamed[i]
        for concept,score in res[i]:
            ranklist.setdefault(concept,[]).append((test_id,score))
          

    # evaluation
    concepts = readConcepts(testCollection,testAnnotationName,rootpath=rootpath)
 
    from basic.metric import getScorer
    scorer = getScorer('AP')
    mean_perf = 0.0

    from basic.annotationtable import readAnnotationsFrom
    from basic.common import niceNumber
    for concept in concepts:
        names,labels = readAnnotationsFrom(testCollection,testAnnotationName,concept,skip_0=True,rootpath=rootpath)
        name2label = dict(zip(names,labels))
        imagelist = ranklist[concept]
        imagelist.sort(key=lambda v:(v[1],v[0]), reverse=True)
        #print concept, imagelist[:3], imagelist[-3:]
        sorted_labels = [name2label[_id] for _id,_score in imagelist if _id in name2label]
        perf = scorer.score(sorted_labels)
        print concept, niceNumber(perf,3)
        mean_perf += perf
    mean_perf /= len(concepts)
    print 'MEAN', niceNumber(mean_perf,3)
              

