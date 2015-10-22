import sys, os, time, random

from basic.constant import ROOT_PATH
from basic.common import  makedirsforfile, checkToSkip, printStatus
from basic.annotationtable import readAnnotationsFrom, writeAnnotationsTo, readConcepts, writeConceptsTo

from util.simpleknn.bigfile import BigFile

from svms.mlengine_util import classify_large_data
from svms.fiksvm.svm import KERNEL_TYPE


DEFAULT_NPR = 10
DEFAULT_NR_ITER = 10
DEFAULT_NR_BINS = 50
DEFAULT_STRATEGY = 'top'
MIN_ERROR_RATE = 1e-4

INFO = __file__


def get_model_name(params):
    name = params['model']
    assert(name in ['fik', 'fastlinear'])
    if 'fik' == name:
        name += str(params['nr_bins'])
    return name
        

def get_new_annotation_name(params):
    assert(params['strategy'] in ['top', 'toprand'])
    startAnnotationName = params['startAnnotationName']
    assert(startAnnotationName.endswith('.txt'))
    startPoint = startAnnotationName[:-4]
    name = get_model_name(params)
    return '%s.%s.%s.npr%d.T%d.txt' % (startPoint, name, params['strategy'], params['npr'], params['iterations'])



class NegativeBootstrap:
    @staticmethod
    def __init__(params):
        rootpath = params['rootpath']
        trainCollection = params['trainCollection']
        feature = params['feature']

        misc = {}
        # the following is required by fiksvm
        minmaxfile = os.path.join(rootpath, trainCollection, 'FeatureData', feature, 'minmax.txt')
        with open(minmaxfile, 'r') as f:
            misc['min_vals'] = map(float, str.split(f.readline()))
            misc['max_vals'] = map(float, str.split(f.readline()))
        return misc    
        

    @staticmethod
    def sampling(predictions, strategy, n):
        printStatus(INFO, '%s sampling: %d out of %d instances' % (strategy, n, len(predictions)))
        if 'toprand' == strategy:
            temp = [(x[0], x[1]*random.uniform(0.9,1)) for x in predictions]   
            temp.sort(key=lambda v:v[1], reverse=True)
            return [x[0] for x in temp[:n]]
        else:
            return [x[0] for x in predictions[:n]]


    @staticmethod
    def learn(concept, params):
        rootpath = params['rootpath']
        trainCollection = params['trainCollection']
        baseAnnotationName = params['baseAnnotationName']
        startAnnotationName = params['startAnnotationName']
        strategy = params['strategy']
        feature = params['feature']
        feat_file = params['feat_file']
        feat_dim = feat_file.ndims
        npr = params['npr']
        iterations = params['iterations']
        beta = 0.5
        
        names,labels = readAnnotationsFrom(trainCollection, startAnnotationName, concept, skip_0=True, rootpath=rootpath)
        positive_bag = [x[0] for x in zip(names,labels) if x[1] > 0]
        negative_bag = [x[0] for x in zip(names,labels) if x[1] < 0]

        names,labels = readAnnotationsFrom(trainCollection, baseAnnotationName, concept, skip_0=True, rootpath=rootpath)
        negative_pool = [x[0] for x in zip(names,labels) if x[1] < 0]

        Usize = max(5000, len(positive_bag) * npr)
        Usize = min(10000, Usize)
        Usize = min(Usize, len(negative_pool))

        new_model = None
         
        for t in range(1, iterations+1):
            printStatus(INFO, 'iter %d (%s)' % (t, concept))
            if t > 1: # select relevant negative examples 
                # check how good at classifying positive training examples
                results = classify_large_data(assemble_model, positive_bag, feat_file)
                pos_error_rate = len([1 for x in results if x[1]<0])/float(len(results))
 
                U = random.sample(negative_pool, Usize)
                predictions = classify_large_data(assemble_model, U, feat_file)
                neg_error_rate = len([1 for x in predictions if x[1]>0])/float(len(predictions))               
               
                error_rate = (pos_error_rate + neg_error_rate)/2.0

                printStatus(INFO, 'iter %d: %s %.3f -> %s %.3f, pe=%.3f, ne=%.3f, error=%.3f' % (t, predictions[-1][0], predictions[-1][1], 
                                                                                                    predictions[0][0], predictions[0][1], 
                                                                                                    pos_error_rate, neg_error_rate, error_rate))
                if error_rate < MIN_ERROR_RATE:
                    printStatus(INFO, 'hit stop criteria: error (%.3f) < MIN_ERROR_RATE (%.3f)' % (error_rate, MIN_ERROR_RATE))
                    break

                # assume that 1% of the randomly sampled set is truely positive, and the classifier will rank them at the top
                # so ignore them
                nr_of_estimated_pos = int(len(predictions)*0.01)
                negative_bag = NegativeBootstrap.sampling(predictions[nr_of_estimated_pos:], strategy, max(1000, len(positive_bag)))

            new_names = positive_bag + negative_bag
            new_labels = [1] * len(positive_bag) + [-1] * len(negative_bag)
            name2label = dict(zip(new_names,new_labels))
            renamed, vectors = feat_file.read(new_names)
            Ys = [name2label[x] for x in renamed] 

            np = len([1 for y in Ys if y>0])
            nn = len([1 for y in Ys if y<0])
            assert(len(positive_bag) == np)
            assert(len(negative_bag) == nn) 
            wp = float(beta) * (np+nn) / np
            wn = (1.0-beta) * (np+nn) /nn
            C = 1
            svm_params = '-w1 %g -w-1 %g' % (wp*C, wn*C) 
            if 'fik' == params['model']:
                svm_params += ' -s 0 -t %d' % KERNEL_TYPE.index("HI")
            else:
                svm_params += ' -s 2'
            g_t = train_model(Ys, vectors, svm_params + ' -q')
            if t == 1:
                assemble_model = compress_model([g_t], [1.0], feat_dim, params)
            else:
                new_model = compress_model([g_t], [1.0], feat_dim, params)
                assemble_model.add_fastsvm(new_model, 1-1.0/t, 1.0/t)

        return assemble_model




def process(options, trainCollection, baseAnnotationName, startAnnotationName, feature, modelName):
    global train_model, compress_model, save_model
    assert(modelName in ['fik', 'fastlinear'])
    if 'fik' == modelName:
        from model_based.svms.fiksvm.svmutil import svm_train as train_model
        from model_based.svms.fiksvm.fiksvm import svm_to_fiksvm as compress_model
        from model_based.svms.fiksvm.fiksvm import fiksvm_save_model as save_model
    else:
        from model_based.svms.fastlinear.liblinear193.python.liblinearutil import train as train_model
        from model_based.svms.fastlinear.fastlinear import liblinear_to_fastlinear as compress_model
        from model_based.svms.fastlinear.fastlinear import fastlinear_save_model as save_model


    rootpath = options.rootpath
    overwrite = options.overwrite
    params = {'rootpath': rootpath, 'trainCollection': trainCollection, 'baseAnnotationName': baseAnnotationName,
              'startAnnotationName': startAnnotationName, 'feature': feature, 'model': modelName, 'strategy': options.strategy,
              'iterations': options.iterations, 'npr': options.npr, 'nr_bins': options.nr_bins}

    concepts = readConcepts(trainCollection, startAnnotationName, rootpath)
    newAnnotationName = get_new_annotation_name(params)
    newModelName = get_model_name(params)
    modeldir = os.path.join(rootpath, trainCollection, 'Models', newAnnotationName, feature, newModelName)
    todo = [concept for concept in concepts if overwrite or os.path.exists(os.path.join(modeldir,'%s.txt'%concept)) is False]
    activeConcepts = [todo[i] for i in range(len(todo)) if (i%options.numjobs+1) == options.job]

    params['feat_file'] = BigFile(os.path.join(rootpath, trainCollection, 'FeatureData', feature))

    if 'fik' == modelName:
        minmax_file = os.path.join(rootpath, trainCollection, 'FeatureData', feature, 'minmax.txt')
        with open(minmax_file, 'r') as f:
            params['min_vals'] = map(float, str.split(f.readline()))
            params['max_vals'] = map(float, str.split(f.readline()))    

        
    s_time = time.time()

    for concept in activeConcepts:
        printStatus(INFO, 'processing %s' % concept)
        modelfile = os.path.join(modeldir, '%s.model'%concept)
        if checkToSkip(modelfile, overwrite):
            continue
        new_model = NegativeBootstrap.learn(concept, params)
        makedirsforfile(modelfile)
        printStatus(INFO, 'save model to %s' % modelfile)
        save_model(modelfile, new_model)
        printStatus(INFO, '%s done' % concept)
        
    timecost = time.time() - s_time
    writeConceptsTo(concepts, trainCollection, newAnnotationName, rootpath)
    printStatus(INFO, 'done for %g concepts: %s' % (len(activeConcepts), ' '.join(activeConcepts)))
    printStatus(INFO, 'models stored at %s' % modeldir)
    printStatus(INFO, '%g seconds in total' % timecost)
    
    


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] trainCollection baseAnnotationName startAnnotationName feature modelName""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--nr_bins", default=DEFAULT_NR_BINS, type="int", help="level of quantization (default: %d)"%DEFAULT_NR_BINS)
    parser.add_option("--npr", default=DEFAULT_NPR, type="int", help="|U|=max(10000, npr*|pos_bag|) (default: %d)"%DEFAULT_NPR)
    parser.add_option("--iterations", default=DEFAULT_NR_ITER, type="int", help="number of iterations (default: %d)"%DEFAULT_NR_ITER)
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath where the train and test collections are stored (default: %s)"%ROOT_PATH)
    parser.add_option("--strategy", default=DEFAULT_STRATEGY, type="string", help="sampling strategy (default: %s)"%DEFAULT_STRATEGY)
    parser.add_option("--numjobs", default=1, type="int", help="number of jobs (default: 1)")
    parser.add_option("--job", default=1, type="int", help="current job (default: 1)")
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 5:
        parser.print_help()
        return 1
    
    assert(options.job>=1 and options.numjobs >= options.job)
    return process(options, args[0], args[1], args[2], args[3], args[4])
 
 

if __name__ == "__main__":
    sys.exit(main())
    

