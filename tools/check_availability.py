
import sys
import os
import unittest

from basic.constant import ROOT_PATH, MATLAB_PATH, CODE_PATH
from basic.common import printStatus

INFO = __file__
TRAIN_COLLECTION_LIST = 'train10k train100k train1m'.split()
TEST_COLLECTION_LIST = 'mirflickr08 flickr51 flickr81'.split()

TRAIN_COLLECTION_LIST = 'train10k'.split()
TEST_COLLECTION_LIST = 'mirflickr08'.split()

FEATURE_LIST = 'vgg-verydeep-16-fc7relu'.split()



def print_msg(methodName, msg):
    print ('')
    print_separator()
    print ('%s is not available, because\n%s' % (methodName, msg) )
    print_separator()


def print_separator(marker='*'):
    print (marker*70)


def check_rootpath():
    if not os.path.exists(ROOT_PATH):
        printStatus(INFO, 'rootpath %s not exists' % ROOT_PATH)
        return False
    return True


def check_matlab():
    matlab_exe_path = MATLAB_PATH + '/matlab'
    if not os.path.exists(matlab_exe_path):
        return False
    return True


def check_testset(testCollection):
    tag_file = os.path.join(ROOT_PATH, testCollection, 'TextData', 'id.userid.lemmtags.txt')
    imset_file = os.path.join(ROOT_PATH, testCollection, 'ImageSets', '%s.txt' % testCollection)

    datafiles = [tag_file, imset_file]

    for feature in FEATURE_LIST:
        feat_file = os.path.join(ROOT_PATH, testCollection, 'FeatureData', feature)
        datafiles.append(feat_file)

    res = find_missing_files(datafiles)
    if res:
        print_msg('Test set %s' % testCollection, 'the following files or folders are missing:\n%s' % res)
        return False
    return True



    

def find_missing_files(filelist):
    missing_files = []
    for datafile in filelist:
        if not os.path.exists(datafile):
            missing_files.append(datafile)

    missing_files = '\n'.join(['[%d] %s' % (i+1,x) for (i,x) in enumerate(missing_files)])
    return missing_files


def check_tagvote(trainCollection, feature):
    missing_files = []
    datafiles = [os.path.join(ROOT_PATH, trainCollection, 'TextData', 'id.userid.lemmtags.txt'),
                 os.path.join(ROOT_PATH, trainCollection, 'FeatureData', feature)]

    res = find_missing_files(datafiles)

    if res:
        print_msg('TagVote (%s, %s)' % (trainCollection, feature), 'the following files or folders are missing:\n%s' % res)
        return False

    return True



def check_wordnet_similarity():
    try:
        from util.tagsim.wordnet_similarity import WordnetSim
        wnsim = WordnetSim('wup')  
        wnsim.compute('dog', 'pet') 
    except Exception, e:
        print e
        return False

    return True



def check_semanticfield(trainCollection):
    datafiles = [ os.path.join(ROOT_PATH, trainCollection, 'TextData', 'lemmtag.userfreq.imagefreq.txt'),
                  os.path.join(ROOT_PATH, trainCollection, 'TextData', 'ucij.uuij.icij.iuij.txt'),
                  os.path.join(ROOT_PATH, trainCollection, 'TextData', 'wn.train1m.txt')]

    res = find_missing_files(datafiles)

    if res:
        print_msg('SemanticField (%s)' % (trainCollection), 'the following files are missing:\n%s' % res)
        return False
    
    return True


def check_tagcooccur(trainCollection):
    #todo
    return True


def check_tagcooccurplus(trainCollection, feature):
    #todo
    return True


def check_tagfeature(trainCollection, feature):
    ready = True

    # check external dependencies
    try:
        from model_based.svms.fastlinear.fastlinear import liblinear_to_fastlinear
    except Exception, e:
        print e
        ready = False
  
    new_feature = 'tag400-%s+%s' % (trainCollection, feature)
    new_feature = new_feature + 'l2' if feature.startswith('vgg') else new_feature
    annotationName = 'concepts130social.random500.0-0.npr1.0-4.txt'

    # check data files
    datafiles = [os.path.join(ROOT_PATH, trainCollection, 'Anntations', 'concepts130.txt'),
                 os.path.join(ROOT_PATH, trainCollection, 'TextData', 'id.userid.lemmtags.txt'),
                 os.path.join(ROOT_PATH, trainCollection, 'ImageSets', '%s.txt' % trainCollection),
                 os.path.join(ROOT_PATH, trainCollection, 'Models', annotationName, new_feature, 'fastlinear')]
    res = find_missing_files(datafiles)
    if res:
        print_msg('TagFeature (%s, %s)' % (trainCollection, feature), 'the following files or folders are missing:\n%s' % res)
        return False
    return ready


def check_relexample(trainCollection, feature):
    ready = True

    # check external dependencies
    try:
        from model_based.svms.fiksvm.fiksvm import svm_to_fiksvm
    except Exception, e:
        print e
        ready = False

    pos_name = 'fcswndsiftbc500' if feature == 'color64+dsift' else 'fcswncnnbc500'
    annotationName = 'concepts130social.%s.random500.0.fik50.top.npr10.T10.txt' % pos_name

    # check data files
    datafiles = [os.path.join(ROOT_PATH, trainCollection, 'Anntations', 'concepts130.txt'),
                 os.path.join(ROOT_PATH, trainCollection, 'TextData', 'id.userid.lemmtags.txt'),
                 os.path.join(ROOT_PATH, trainCollection, 'ImageSets', '%s.txt' % trainCollection),
                 os.path.join(ROOT_PATH, trainCollection, 'FeatureData', feature),
                 os.path.join(ROOT_PATH, trainCollection, 'Models', annotationName, feature, 'fik50')]
    res = find_missing_files(datafiles)
    if res:
        print_msg('RelExamples (%s, %s)' % (trainCollection, feature), 'the following files or folders are missing:\n%s' % res)
        return False
    return ready


def check_knn(trainCollection, feature):
    #todo
    return True
    
    
def check_tagranking(trainCollection, feature):
    #todo
    return True


def check_tagprop(trainCollection, feature):
    ready = True

    # check matlab
    if not check_matlab():
        print_msg('TagProp (%s, %s)' % (trainCollection, feature), 'Matlab is not available or incorrectly configured.')
        ready = False
    
    # check if knn is available
    if not check_knn(trainCollection, feature):
        print_msg('TagProp (%s, %s)' % (trainCollection, feature), 'KNN is not available.')        
        ready = False
        
    # check downloaded file
    tagprop_files = [os.path.join(CODE_PATH, 'TagProp/logsigmoid.m'), 
                     os.path.join(CODE_PATH, 'TagProp/minimize.m'), 
                     os.path.join(CODE_PATH, 'TagProp/projDistrib.m'), 
                     os.path.join(CODE_PATH, 'TagProp/projGradDescentArmijo.m'), 
                     os.path.join(CODE_PATH, 'TagProp/projNoConstraints.m'), 
                     os.path.join(CODE_PATH, 'TagProp/projNonNegative.m'), 
                     os.path.join(CODE_PATH, 'TagProp/sigmoid.m'), 
                     os.path.join(CODE_PATH, 'TagProp/sigmoids.m'), 
                     os.path.join(CODE_PATH, 'TagProp/tagpropCmt.c'), 
                     os.path.join(CODE_PATH, 'TagProp/tagpropCmt.mexa64'), 
                     os.path.join(CODE_PATH, 'TagProp/tagprop_learn.m'), 
                     os.path.join(CODE_PATH, 'TagProp/tagprop_predict.m')]
    
    res = find_missing_files(tagprop_files)
    if res:
        print_msg('TagProp (%s, %s)' % (trainCollection, feature), 'Run setup.sh. The following files are missing:\n%s' % res)
        ready = False
        
    # check data files
    datafiles = [os.path.join(ROOT_PATH, trainCollection, 'TextData', 'id.userid.lemmtags.txt'),
                 os.path.join(ROOT_PATH, trainCollection, 'FeatureData', feature)]
    res = find_missing_files(datafiles)
    if res:
        print_msg('TagProp (%s, %s)' % (trainCollection, feature), 'the following files or folders are missing:\n%s' % res)
        return False    
        
    # check external dependencies
    try:
        import h5py
        import numpy
    except Exception, e:
        print e
        ready = False

    return ready

def check_robustpca(trainCollection, testCollection, feature):
    ready = True
    
    # check matlab    
    if not check_matlab():
        print_msg('RobustPCA (%s, %s, %s)' % (trainCollection, testCollection, feature), 'Matlab is not available or incorrectly configured.')
        ready = False
    
    # check if knn is available
    if not check_knn(trainCollection, testCollection, feature):
        print_msg('RobustPCA (%s, %s, %s)' % (trainCollection, testCollection, feature), 'KNN is not available.')        
        ready = False

    # check data files
    datafiles = [ os.path.join(ROOT_PATH, trainCollection, 'TextData', 'id.userid.lemmtags.txt'),
                  os.path.join(ROOT_PATH, trainCollection, 'FeatureData', feature)]
    res = find_missing_files(datafiles)
    if res:
        print_msg('RobustPCA (%s, %s, %s)' % (trainCollection, testCollection, feature), 'the following files or folders are missing:\n%s' % res)
        return False    
              
    # check external dependencies  
    try:
        import h5py
        import numpy
        import scipy.io
        import scipy.sparse
        from nltk.corpus import wordnet as wn
        from nltk.corpus import wordnet_ic
        brown_ic = wordnet_ic.ic('ic-brown.dat')
        wn.morphy('cat')
        wn.synsets('cat', pos=wn.NOUN)
    except Exception, e:
        try:
            import nltk
            nltk.download('brown')
            nltk.download('wordnet')
            nltk.download('wordnet_ic')
        except Exception, e:
            print e
            ready = False

    return ready


if __name__ == '__main__':
    if not check_rootpath():
        sys.exit(0)

    okay_testsets = []
    okay_methods = []
    
    matlab_ok = check_matlab()

    for testCollection in TEST_COLLECTION_LIST:
        if check_testset(testCollection):
            okay_testsets.append(testCollection)

    try:
        import h5py
        import numpy
        import scipy.io
        import scipy.sparse
        from nltk.corpus import wordnet as wn
        from nltk.corpus import wordnet_ic
        brown_ic = wordnet_ic.ic('ic-brown.dat')
        wn.morphy('cat')
        wn.synsets('cat', pos=wn.NOUN)
    except Exception, e:
        try:
            import nltk
            nltk.download('brown')
            nltk.download('wordnet')
            nltk.download('wordnet_ic')
        except Exception, e:
            print e
            print_msg('TagProp, RobustPCA', 'dependencies are not meet.')

    if check_wordnet_similarity():
        for trainCollection in TRAIN_COLLECTION_LIST:
            if check_semanticfield(trainCollection):
                okay_methods.append('SemanticField (%s)' % (trainCollection))
    else:
        print_msg('SemanticField', 'the wordnet similarity is unavailable')

    sys.exit(0)
    for trainCollection in TRAIN_COLLECTION_LIST:
        if check_tagcooccur(trainCollection):
                okay_methods.append('TagCooccur (%s)' % (trainCollection))

        for feature in FEATURE_LIST:
            if check_tagcooccurplus(trainCollection, feature):
                okay_methods.append('TagCooccur+ (%s,%s)' % (trainCollection, feature))
            
            if check_knn(trainCollection, feature):
                okay_methods.append('KNN (%s,%s)' % (trainCollection, feature))

            if check_tagvote(trainCollection, feature):
                okay_methods.append('TagVote (%s,%s)' % (trainCollection, feature))

            if check_tagprop(trainCollection, feature):
                okay_methods.append('TagProp (%s,%s)' % (trainCollection, feature))

            if check_tagranking(trainCollection, feature):
                okay_methods.append('TagRanking (%s,%s)' % (trainCollection, feature))

            if check_tagfeature(trainCollection, feature):
                okay_methods.append('TagFeature (%s,%s)' % (trainCollection, feature))

            if check_relexample(trainCollection, feature):
                okay_methods.append('RelExample (%s,%s)' % (trainCollection, feature))

    for trainCollection in TRAIN_COLLECTION_LIST:
        for testCollection in TEST_COLLECTION_LIST:
            if check_robustpca(trainCollection, testCollection, feature):
                okay_methods.append('RobustPCA (%s + %s,%s)' % (trainCollection, testCollection, feature))


    
    print_separator('=')
    print ('Methods available:')
    print ('\n'.join(['[%d] %s' % (i+1, x) for i,x in enumerate(okay_methods)]))
    print_separator('=')

    print ('Test sets available:')
    print ('\n'.join(['[%d] %s' % (i+1, x) for i,x in enumerate(okay_testsets)]))

    print_separator('=')    
    print ('Matlab is %s available.' % ('' if matlab_ok else 'NOT'))
    

