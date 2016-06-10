#!/usr/bin/env python
# encoding: utf-8

import sys, os, time
import subprocess
import cPickle as pickle
import h5py
import numpy as np
from basic.constant import ROOT_PATH, MATLAB_PATH
from basic.common import checkToSkip, niceNumber, printStatus, makedirsforfile
from basic.util import readImageSet, getVocabMap, bisect_index
from basic.annotationtable import readConcepts

INFO = 'tagrelcodebase.robustpca'
DEFAULT_LAMBDA1 = 0.
DEFAULT_LAMBDA2 = 0.
DEFAULT_K_RATIO = 0.001
DEFAULT_K_PROP = 0.1
DEFAULT_RATIO_CS = 0.9
DEFAULT_DISTANCE = "cosine"

def call_matlab(script):
    id_file = os.getpid()

    with open("/tmp/script_%d.m" % id_file, 'w') as f:
        f.write(script)

    printStatus(INFO, "Starting MATLAB to run /tmp/script_%d.m" % (id_file))
    result = subprocess.call(MATLAB_PATH + '/bin/matlab -nodesktop -nosplash -r "addpath(\'/tmp\'); script_%d"' % id_file, shell=True)

    os.unlink("/tmp/script_%d.m" % id_file)
    if result != 0:
        printStatus(INFO, "Error while calling MATLAB, return value %s. Aborting..." % (str(result)))
        sys.exit(2)
    else:
        printStatus(INFO, "MATLAB return value %s" % (str(result)))


def process(options, workingCollection, annotationName, feature, outputpkl):
    rootpath = options.rootpath
    distance = options.distance
    overwrite = options.overwrite
    k_ratio = options.kratio
    ratio_cs = options.ratiocs
    lambda1 = options.lambda1
    lambda2 = options.lambda2
    outputonlytest = options.outputonlytest
    rawtagmatrix = options.rawtagmatrix
    modelName = "robustpca"
    nnName = distance + "knn"

    printStatus(INFO, "Starting RobustPCA %s,%s,%s,%s,%f,%f,%f" % (workingCollection, annotationName, feature, nnName, k_ratio, lambda1, lambda2))

    if rawtagmatrix:
        printStatus(INFO, "Using raw tag matrix.")
    else:
        printStatus(INFO, "Using preprocessed tag matrix.")

    resultfile = os.path.join(outputpkl)
    resultfile_robustpca = os.path.join(rootpath, workingCollection, 'RobustPCA-Prediction', '%s,%s,%f,%f,%f,%d'%(feature,nnName,lambda1,lambda2,k_ratio,rawtagmatrix), 'prediction.mat')

    if checkToSkip(resultfile_robustpca, overwrite):
        only_dump = True
    else:
        only_dump = False

    if not rawtagmatrix:
        tagmatrix_file = os.path.join(rootpath, workingCollection, 'RobustPCA', '%s,%s,%f'%(feature,nnName,DEFAULT_K_PROP), 'tagmatrix.h5')
        if not os.path.exists(tagmatrix_file):
            printStatus(INFO, "Tag matrix file not found at %s Did you run robustpca_preprocessing.py?" % (tagmatrix_file))
            sys.exit(1)
    else:
        tagmatrix_file = os.path.join(rootpath, workingCollection, 'TextData', "lemm_wordnet_freq_tags.h5")
        if not os.path.exists(tagmatrix_file):
            printStatus(INFO, 'Tag matrix file not found in %s Did you run wordnet_frequency_tags.py?' % (tagmatrix_file))
            sys.exit(1)

    laplacianI_file = os.path.join(rootpath, workingCollection, 'LaplacianI', workingCollection, '%s,%s,%f'%(feature,nnName,k_ratio), 'laplacianI.mat')
    if not os.path.exists(laplacianI_file):
        printStatus(INFO, "LaplacianI file not found at %s Did you run laplacian_images.py?" % (laplacianI_file))
        sys.exit(1)

    laplacianT_file = os.path.join(rootpath, workingCollection, 'LaplacianT', '%f'%(ratio_cs), 'laplacianT.mat')
    if not os.path.exists(laplacianT_file):
        printStatus(INFO, "LaplacianT file not found at %s Did you run laplacian_tags.py?" % (laplacianT_file))
        sys.exit(1)

    # being learning
    script = """
        rpca_path = 'transduction_based/robustpca/';
        addpath(rpca_path);
        addpath([rpca_path, 'fast_svd/']);
        tagmatrix = sparse(double(h5read('%s', '/tagmatrix')));
        load('%s');
        load('%s');

        lambda1 = %f;
        lambda2 = %f;
        maxIters = 50;
        precision = 1e-4;
        mu_start = 1.;

        parpool('local', 4);
        [P,E]=robustpca(tagmatrix, lambda1, lambda2, tag_similarity, im_similarity, maxIters, precision, mu_start);
        """ % (tagmatrix_file, laplacianI_file, laplacianT_file, lambda1, lambda2)

    script += """
        delete(gcp);
        save('%s', 'P', 'E', 'lambda1', 'lambda2', '-v7.3');
        exit;
    """ % resultfile_robustpca

    if not only_dump:
        printStatus(INFO, "starting learning")
        makedirsforfile(resultfile_robustpca)
        call_matlab(script)

    if checkToSkip(resultfile, overwrite):
        return 0

    # save results in pkl format
    printStatus(INFO, "Dump results in pkl format at %s" % resultfile)
    concepts = readConcepts(workingCollection, annotationName, rootpath)
    if outputonlytest:
        testset_id_images = readImageSet(workingCollection.split('+')[1], workingCollection.split('+')[1], rootpath)
        testset_id_images.sort()

    id_images = readImageSet(workingCollection, workingCollection, rootpath)
    id_images.sort()

    # concepts mapping
    robustpca_output = h5py.File(resultfile_robustpca, 'r')
    tagprop_input = h5py.File(tagmatrix_file, 'r')
    mapping = getVocabMap(list(tagprop_input['vocab'][:]),concepts)

    predicted_tagmatrix = robustpca_output['P'][:,mapping]

    if outputonlytest:
        idx = np.array([bisect_index(id_images, x) for x in testset_id_images])
        final_tagmatrix = predicted_tagmatrix[idx, :]
        assert(final_tagmatrix.shape[0] == idx.shape[0])
        id_images = testset_id_images
    else:
        final_tagmatrix = predicted_tagmatrix

    makedirsforfile(resultfile)
    with open(resultfile, 'w') as f:
        pickle.dump({'concepts':concepts, 'id_images': id_images, 'scores':final_tagmatrix}, f, pickle.HIGHEST_PROTOCOL)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] workingCollection annotationName feature outputpkl""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--kratio", default=DEFAULT_K_RATIO, type="float", help="laplacianI K ratio (%f)" % DEFAULT_K_RATIO)
    parser.add_option("--ratiocs", default=DEFAULT_RATIO_CS, type="float", help="laplacianT ratio cs (%f)" % DEFAULT_RATIO_CS)
    parser.add_option("--lambda1", default=DEFAULT_LAMBDA1, type="float", help="weight parameter of sparseness on error matrix (default: %s)" % DEFAULT_LAMBDA1)
    parser.add_option("--lambda2", default=DEFAULT_LAMBDA2, type="float", help="weight parameter of laplacians terms (default: %s)" % DEFAULT_LAMBDA2)
    parser.add_option("--distance", default=DEFAULT_DISTANCE, type="string", help="visual distance, can be l1,l2 or cosine (default: %s)" % DEFAULT_DISTANCE)
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="(default: %s)" % ROOT_PATH)
    parser.add_option("--outputonlytest", default=0, type="int", help="dump test set results only (works on merged datasets, default: 0)")
    parser.add_option("--rawtagmatrix", default=0, type="int", help="use the raw tag matrix instead of preprocessed one (default: 0)")

    (options, args) = parser.parse_args(argv)
    if len(args) < 4:
        parser.print_help()
        return 1

    return process(options, args[0], args[1], args[2], args[3])

if __name__ == "__main__":
    sys.exit(main())
