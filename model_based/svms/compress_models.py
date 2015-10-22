
import sys
import os
import time

from basic.constant import ROOT_PATH
from basic.common import checkToSkip,makedirsforfile, printStatus


INFO = os.path.basename(__file__)


def process(options, model_name, concept_file, weight_dir, result_dir):
    rootpath = options.rootpath
    overwrite = options.overwrite

    if 'fastlinear' == model_name:
        from fastlinear.fastlinear import fastlinear_load_model as load_model
        from fastlinear.fastlinear import fastlinear_save_model as save_model
    else:
        from fiksvm.fiksvm import fiksvm_load_model as load_model
        from fiksvm.fiksvm import fiksvm_save_model as save_model


    concepts = [x.strip() for x in open(concept_file).readlines() if x.strip() and not x.strip().startswith('#')]
    todo = [x for x in concepts if overwrite or not os.path.exists(os.path.join(result_dir, '%s.model'%x))]
    printStatus(INFO, '%d concepts to do' % len(todo))

    for concept in todo:
        weight_file = os.path.join(weight_dir, '%s.txt' % concept)
        weight_data = map(str.strip, open(weight_file).readlines())
        nr_of_models = len(weight_data)
        assert(nr_of_models >= 2)
        weights = [0] * nr_of_models
        models = [None] * nr_of_models

        for i,line in enumerate(weight_data):
            w, model_dir = line.split()
            weights[i] = float(w)
            model_dir =  model_dir if model_dir.startswith(rootpath) else os.path.join(rootpath, model_dir)
            assert (model_dir.find(model_name)>0)
            model_file_name = os.path.join(model_dir, '%s.model' % concept)
            models[i] = load_model(model_file_name)

        new_model = models[0]
        new_model.add_fastsvm(models[1], weights[0], weights[1])
        for i in range(2, len(models)):
            new_model.add_fastsvm(models[i], 1, weights[i])    

        new_model_file = os.path.join(result_dir, '%s.model'%concept)
        makedirsforfile(new_model_file)
        save_model(new_model_file, new_model)



def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] model_name, concept_file weight_dir result_dir""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 4:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1], args[2], args[3])


if __name__ == "__main__":
    sys.exit(main())
    
        
