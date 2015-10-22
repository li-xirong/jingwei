import sys
import os
import time

from basic.constant import ROOT_PATH
from simpleknn.bigfile import BigFile
from basic.data import FEATURE_TO_DIM
from basic.annotationtable import readConcepts, readAnnotationsFrom
from basic.metric import getScorer
#from svmutil import *
#from svm import *
from fastsvm.svmutil import *
from fastsvm.svm import *
from fiksvm import *
from fiksvmutil import *
from fastsvm.fiksvm import svm_to_fiksvm as svm_to_fiksvm0


if __name__ == "__main__":
    rootpath = ROOT_PATH
    trainCollection = "voc2008train"
    testCollection = "voc2008val"
    annotationName = "conceptsvoc2008train.txt"
    #concept = "aeroplane"
    feature = "dsift"

    concepts = readConcepts(testCollection, 'conceptsvoc2008val.txt')
    scorer = getScorer('AP')

    min_vals, max_vals = find_min_max_vals(BigFile(os.path.join(rootpath, trainCollection, 'FeatureData', feature), FEATURE_TO_DIM[feature]))
    featurefile = os.path.join(rootpath, testCollection, "FeatureData", feature, "id.feature.txt")

    feat_dim = 1024
    num_bins = 50

    #fikmodel.set_probAB(-1, 0)
    
    #print "fik model0", fikmodel0.get_nr_svs(), fikmodel0.get_feat_dim(), fikmodel0.get_probAB()
    #print "fik model", fikmodel.get_nr_svs(), fikmodel.get_feat_dim(), fikmodel.get_probAB()
    mAP = [0]*4
    for concept in concepts:
        names,labels = readAnnotationsFrom(testCollection, 'conceptsvoc2008val.txt', concept)
        name2label = dict(zip(names,labels))
        ranklist = []

        modelfile = os.path.join(rootpath, trainCollection, "Models", annotationName, feature, 'hiksvm', "%s.model" % concept)
        #print modelfile
        model = svm_load_model(modelfile)
        #print model.get_svm_type()
        #print model.get_nr_class()
        svm_models = [model, model]
        num_models = len(svm_models)
        fikmodel0 = svm_to_fiksvm0(svm_models, [1.0/num_models]*num_models, num_models, feat_dim, num_bins)
        fikmodel1 = svm_to_fiksvm(svm_models, num_models, [1.0/num_models]*num_models, feat_dim, min_vals, max_vals, num_bins)
        fikmodel2 = svm_to_fiksvm(svm_models, num_models, [1.0/num_models]*num_models, feat_dim, min_vals, max_vals, num_bins)
        fikmodel2.add_new_fikmodel(fikmodel1, 0.5)
        print concept, fikmodel1.get_nr_svs(), fikmodel1.get_nr_svs() + fikmodel1.get_nr_svs()/2,
        fikmodel1.add_new_hikmodel(model, 0.5)
        print fikmodel1.get_nr_svs(), fikmodel2.get_nr_svs()

        for line in open(featurefile):
            elems = str.split(line)
            imageid = elems[0]
            x = map(float, elems[1:])
            s_time = time.time()  
            label, score = svm_predict(svm_models[0], x)
            time1 = time.time() - s_time
            s_time = time.time()
            score0 = fikmodel0.predict(x)
            score1 = fikmodel1.predict(x)
            score2 = fikmodel2.predict(x)
            time2 = time.time() - s_time
            #print score-score0, score-score1, score-score2, time1, time2
            ranklist.append((imageid, score, score0, score1, score2))
            #if len(ranklist) >= 20:
            #    sys.exit(0)
   
        print concept,
        for i in range(1,5):
            ranklist.sort(key=lambda v:(v[i]), reverse=True)
            sorted_labels = [name2label[x[0]] for x in ranklist if x[0] in name2label]
            print '%.3f' % scorer.score(sorted_labels),
            mAP[i-1] += scorer.score(sorted_labels)
        print ""

    print "MEAN", 
    for x in mAP:
        print x/len(concepts),
    print ""
    sys.exit(0)
    print "\n".join(["%s %g %g" % (v[0], v[1], v[2]) for v in ranklist[:20]])

    model_file_name = "%s.model" % concept
    fiksvm_save_model(model_file_name, fikmodel)

    newmodel = fiksvm_load_model(model_file_name)
    print "new model", newmodel.get_nr_svs(), newmodel.get_feat_dim(), newmodel.get_probAB()

    sys.exit(0)

    ranklist = []

    for line in open(featurefile):
        elems = str.split(line)
        imageid = elems[0]
        x = map(float, elems[1:])
        s_time = time.time()
        score = fikmodel.predict(x)
        fikscore = newmodel.predict(x)
        time2 = time.time() - s_time
        print score, fikscore, time1, time2

        ranklist.append((imageid, score, fikscore))
        if len(ranklist) > 10:
            break
    
    ranklist.sort(key=lambda v:(v[2]), reverse=True)
    print "\n".join(["%s %g %g" % (v[0], v[1], v[2]) for v in ranklist[:20]])



