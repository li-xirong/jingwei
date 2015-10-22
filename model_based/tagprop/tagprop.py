#!/usr/bin/env python
# encoding: utf-8

import sys, os, time
import subprocess
import cPickle as pickle
import h5py

from basic.constant import ROOT_PATH, MATLAB_PATH
from basic.common import checkToSkip, niceNumber, printStatus, makedirsforfile
from basic.util import readImageSet, getVocabMap
from basic.annotationtable import readConcepts


INFO = 'tagrelcodebase.tagprop'
DEFAULT_VARIANT = "ranksigmoids"
DEFAULT_K = 1000
DEFAULT_DISTANCE = "cosine"

def call_matlab(script):
    id_file = os.getpid()

    with open("/tmp/script_%d.m" % id_file, 'w') as f:
        f.write(script)

    printStatus(INFO, "Starting MATLAB to run /tmp/script_%d.m" % (id_file))
    result = subprocess.call(MATLAB_PATH + '/bin/matlab -nodesktop -nosplash -nojvm -r "addpath(\'/tmp\'); script_%d"' % id_file, shell=True)

    os.unlink("/tmp/script_%d.m" % id_file)
    if result != 0:
        printStatus(INFO, "Error while calling MATLAB, return value %s. Aborting..." % (str(result)))
        sys.exit(2)
    else:
        printStatus(INFO, "MATLAB return value %s" % (str(result)))


def process(options, testCollection, trainCollection, annotationName, feature, outputpkl):
    rootpath = options.rootpath
    k = options.k
    distance = options.distance
    variant = options.variant
    overwrite = options.overwrite
    testset = testCollection
    forcetrainmodel = options.trainmodel
    modelName = "tagprop"
    nnName = distance + "knn"

    printStatus(INFO, "Starting TagProp %s,%s,%s,%s,%s" % (variant, trainCollection, testCollection, annotationName, feature))

    resultfile = os.path.join(outputpkl)
    resultfile_tagprop = os.path.join(rootpath, testCollection, 'TagProp-Prediction', testset, trainCollection, annotationName, modelName, '%s,%s,%s,%d'%(feature,nnName,variant,k), 'prediction.mat')
    if checkToSkip(resultfile, overwrite) or checkToSkip(resultfile_tagprop, overwrite):
        return 0

    tagmatrix_file = os.path.join(rootpath, trainCollection, 'TextData', 'lemm_wordnet_freq_tags.h5')
    if not os.path.exists(tagmatrix_file):
        printStatus(INFO, "Tag matrix file not found at %s Did you run wordnet_frequency_tags.py?" % (tagmatrix_file))
        sys.exit(1)

    train_neighs_file = os.path.join(rootpath, trainCollection, 'TagProp-data', trainCollection, '%s,%s,%d'%(feature,nnName,k), 'nn_train.h5')
    if not os.path.exists(train_neighs_file):
        printStatus(INFO, "Matlab train neighbors file not found at %s Did you run prepare_tagprop_data.py?" % (train_neighs_file))
        sys.exit(1)

    # do we need to perform learning?
    train_model_file = os.path.join(rootpath, trainCollection, 'TagProp-models', '%s,%s,%s,%d'%(feature,nnName,variant,k), 'model.mat')
    if os.path.exists(train_model_file) and not forcetrainmodel:
        printStatus(INFO, "model for %s available at %s" % (trainCollection, train_model_file))
    else:
        printStatus(INFO, "starting learning model for %s" % (trainCollection))
        makedirsforfile(train_model_file)

        script = """
                tagprop_path = 'model_based/tagprop/TagProp/';
                addpath(tagprop_path);
                tagmatrix = h5read('%s', '/tagmatrix') > 0.5;
                tagmatrix = sparse(tagmatrix);
                NN = h5read('%s', '/NN');
                NN = NN(2:end, :);
                NN = double(NN);
        """ % (tagmatrix_file, train_neighs_file)

        if variant == 'dist' or variant == 'distsigmoids':
            script += """
                NND = h5read('%s', '/NND');
                NND = NND(2:end, :);
                NND = reshape(NND, 1, size(NND,1), size(NND,2));
                NND = double(NND);
            """ % train_neighs_file

        if variant == 'rank':
            script += """
                m = tagprop_learn(NN,[],tagmatrix);
            """
        elif variant == 'ranksigmoids':
            script += """
                m = tagprop_learn(NN,[],tagmatrix,'sigmoids',true);
            """
        elif variant == 'dist':
            script += """
                m = tagprop_learn(NN,NND,tagmatrix,'type','dist');
            """
        elif variant == 'distsigmoids':
            script += """
                m = tagprop_learn(NN,NND,tagmatrix,'type','dist','sigmoids',true);
            """

        script += """
                save('%s', 'm', '-v7.3');
                exit;
        """ % train_model_file

        call_matlab(script)

    # we perform prediction
    printStatus(INFO, "starting prediction")
    test_neighs_file = os.path.join(rootpath, testCollection, 'TagProp-data', testset, trainCollection, annotationName, '%s,%s,%d'%(feature,nnName,k), 'nn_test.h5')
    if not os.path.exists(test_neighs_file):
        printStatus(INFO, "Matlab test neighbors file not found at %s Did you run prepare_tagprop_data.py?" % (test_neighs_file))
        sys.exit(1)

    script = """
            tagprop_path = 'model_based/tagprop/TagProp/';
            addpath(tagprop_path);
            load('%s');
            tagmatrix = h5read('%s', '/tagmatrix') > 0.5;
            tagmatrix = sparse(tagmatrix);
            NNT = h5read('%s', '/NNT');
            NNT = double(NNT);

    """ % (train_model_file, tagmatrix_file, test_neighs_file)

    if variant == 'dist' or variant == 'distsigmoids':
        script += """
            NNDT = h5read('%s', '/NNDT');
            NNDT = reshape(NNDT, 1, size(NNDT,1), size(NNDT,2));
            NNDT = double(NNDT);
        """ % test_neighs_file

    script += """
            P = tagprop_predict(NNT,[],m)';
            save('%s', '-v7.3');
            exit;
    """ % resultfile_tagprop

    makedirsforfile(resultfile_tagprop)
    call_matlab(script)

    # save results in pkl format
    printStatus(INFO, "Dump results in pkl format at %s" % resultfile)

    concepts = readConcepts(testCollection, annotationName, rootpath)
    id_images = readImageSet(testCollection, testset, rootpath)
    id_images.sort()
    id_images = map(int, id_images)

    # concepts mapping
    tagprop_output = h5py.File(resultfile_tagprop, 'r')
    tagprop_input = h5py.File(tagmatrix_file, 'r')
    mapping = getVocabMap(list(tagprop_input['vocab'][:]),concepts)

    final_tagmatrix = tagprop_output['P'][:][:,mapping]

    with open(resultfile, 'w') as f:
        pickle.dump({'concepts':concepts, 'id_images':id_images, 'scores':final_tagmatrix}, f, pickle.HIGHEST_PROTOCOL)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] testCollection trainCollection annotationName feature outputpkl""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--trainmodel", default=0, type="int", help="train the model even if already available (default=0)")
    parser.add_option("--k", default=DEFAULT_K, type="int", help="number of neighbors (%d)" % DEFAULT_K)
    parser.add_option("--variant", default="ranksigmoids", type="string", help="tagprop variant, can be rank, dist, ranksigmoids or distsigmoids (default: %s)" % DEFAULT_VARIANT)
    parser.add_option("--distance", default=DEFAULT_DISTANCE, type="string", help="visual distance, can be l1, l2 or cosine (default: %s)" % DEFAULT_DISTANCE)
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="(default: %s)" % ROOT_PATH)

    (options, args) = parser.parse_args(argv)
    if len(args) < 5:
        parser.print_help()
        return 1

    return process(options, args[0], args[1], args[2], args[3], args[4])

if __name__ == "__main__":
    sys.exit(main())
