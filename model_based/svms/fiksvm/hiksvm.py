
import numpy as np

from svm import KERNEL_TYPE
from svmutil import svm_train
from fiksvm import svm_to_fiksvm
from util import classifyLargeData


def hiksvm_train(labels, features, beta):
    # calculate class prior
    np = len([1 for lab in labels if  1 == lab])
    nn = len([1 for lab in labels if -1 == lab])
    wp = float(beta)/np
    wn = (1.0-beta)/nn
    wp *= (np+nn)
    wn *= (np+nn)
    parameters = "-s 0 -c 1 -t %d -w-1 %g -w1 %g" % (KERNEL_TYPE.index("HI"), wn, wp)
    model = svm_train(labels, features, parameters)
    return model


def find_best_beta(labels, features, cv, scorer, min_vals, max_vals):
    dim = len(features[0])
    beta_set = [x/10.0 for x in range(1,10)]
    perfs = [[] for i in range(len(beta_set))]

    n = len(labels)

    positive_index = [i for i in range(n) if  1 == labels[i]]
    negative_index = [i for i in range(n) if -1 == labels[i]]
    num_positive = len(positive_index)
    num_negative = len(negative_index)

    if num_positive < cv:
        message = "[find_best_beta] %d positive examples, insufficient for %d-fold cross-validation" % (len(positive_index), cv)
        raise Exception(message)

    for folder in range(cv):
        print ("[find_best_beta] %d <- %s" % (folder, "-".join(map(str, [i for i in range(cv) if i!=folder]))))

        labels_val = [1 for i in range(num_positive) if i%cv == folder] + [-1 for i in range(num_negative) if i%cv == folder]
        features_val = ([features[positive_index[i]] for i in range(num_positive) if i%cv == folder] +
                        [features[negative_index[i]] for i in range(num_negative) if i%cv == folder])

        labels_train = [1 for i in range(num_positive) if i%cv != folder] + [-1 for i in range(num_negative) if i%cv != folder]
        features_train = ([features[positive_index[i]] for i in range(num_positive) if i%cv != folder] +
                          [features[negative_index[i]] for i in range(num_negative) if i%cv != folder])

        assert(len(labels_val) == len(features_val))
        assert(len(labels_train) == len(features_train))
        assert((len(labels_val)+len(labels_train)) == n)

        for index,beta in enumerate(beta_set):
            model = hiksvm_train(labels_train, features_train, beta=beta)
            #fikmodel = svm_to_fiksvm([model], [1.0], 1, dim, 50)
            fikmodel = svm_to_fiksvm([model], 1, [1.0], feat_dim=dim, min_vals=min_vals, max_vals=max_vals, num_bins=50)
            results = [(labels_val[i], fikmodel.predict(features_val[i])) for i in range(len(labels_val))]
            results.sort(key=lambda v:(v[1]), reverse=True)
            sorted_labels = [x[0] for x in results]
            perf = scorer.score(sorted_labels)
            print "[find_best_beta] folder %d, beta %g -> %s=%g" % (folder, beta, scorer.name(), perf)
            perfs[index].append(perf)

    ranklist = [(beta_set[index], np.mean(perfs[index])) for index in range(len(beta_set))]
    ranklist.sort(key=lambda v:(v[1]), reverse=True)
    print "[find_best_beta] done", ranklist
    best_beta = ranklist[0][0]
    cv_score = ranklist[0][1]
    return best_beta, cv_score


def hiksvm_train_cv(labels, features, cv, scorer, min_vals, max_vals):
    if cv < 2:
        best_beta = 0.5
        cv_score = -1
    else:
        best_beta, cv_score = find_best_beta(labels=labels, features=features, cv=cv, scorer=scorer, min_vals=min_vals, max_vals=max_vals)
    model = hiksvm_train(labels, features, best_beta)
    #fikmodel = svm_to_fiksvm([model], [1.0], 1, len(features[0]), 50)
    return best_beta, cv_score, model


if __name__ == "__main__":
    import sys, os, time, random

    from basic.constant import ROOT_PATH
    from basic.metric import getScorer
    from basic.common import writeRankingResults
    from basic.annotationtable import readAnnotationsFrom
    from simpleknn.bigfile import BigFile
    
    ROOT_PATH = '/home/root123/xirong/VisualSearch'
    rootpath = ROOT_PATH
    trainCollection = 'flickr81train'
    trainAnnotationName = 'concepts81train.random50.0.random50.0.txt'
    testCollection = "flickr81test"
    testAnnotationName = 'conceptsflickr81test.txt'
    feature = "dascaffeprob"
    feat_dim = 1000
    scorer = getScorer("AP")
    
    targetConcept = sys.argv[1] #"aeroplane"

    train_feat_file = BigFile(os.path.join(ROOT_PATH, trainCollection, "FeatureData", feature), feat_dim)
    test_feat_file = BigFile(os.path.join(ROOT_PATH, testCollection, "FeatureData", feature), feat_dim)
    testImageSet = test_feat_file.names #random.sample(test_feat_file.names, 10000)
    
    minmax_file = os.path.join(rootpath, trainCollection, 'FeatureData', feature, 'minmax.txt')
    with open(minmax_file, 'r') as f:
        min_vals = map(float, str.split(f.readline()))
        max_vals = map(float, str.split(f.readline()))


    [names,labels] = readAnnotationsFrom(collection=trainCollection, annotationName=trainAnnotationName, concept=targetConcept, rootpath=rootpath)
    name2label = dict(zip(names,labels))
    (renamed, vectors) = train_feat_file.read(names)
    relabeled = [name2label[x] for x in renamed] #label is either 1 or -1
    
    [names,labels] = readAnnotationsFrom(collection=testCollection, annotationName=testAnnotationName, concept=targetConcept, rootpath=rootpath)
    test2label = dict(zip(names,labels))
    

    for beta in [0.5]: #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        #model = hiksvm_train(relabeled, vectors, beta=beta)
        cv = 3
        best_beta, cv_score, model = hiksvm_train_cv(relabeled, vectors, cv, scorer, min_vals, max_vals)
        print best_beta, cv_score
        #fikmodel = svm_to_fiksvm([model], [1.0], 1, dim, 50)
        fikmodel = svm_to_fiksvm([model], 1, [1.0], feat_dim=feat_dim, min_vals=min_vals, max_vals=max_vals, num_bins=50)
        
        results = classifyLargeData(fikmodel, testImageSet, test_feat_file, prob_output=True)
        print results[:5]

        sorted_labels = [test2label[x] for x,y in results]
        score = scorer.score(sorted_labels)
        print "beta", beta, "AP", score
  
            
