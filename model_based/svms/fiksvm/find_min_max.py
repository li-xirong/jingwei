import os, sys, array, time
import numpy as np
from optparse import OptionParser
from basic.common import checkToSkip, printStatus


INFO = 'mlengine.fiksvm.find_min_max.py'


def process(options, feat_dir):
    resultfile = os.path.join(feat_dir, 'minmax.txt')
    if checkToSkip(resultfile, options.overwrite):
        sys.exit(0)
 
    nr_of_images, feat_dim = map(int, open(os.path.join(feat_dir,'shape.txt')).readline().split())   
    min_vals = [1e6] * feat_dim
    max_vals = [-1e6] * feat_dim

    offset = np.float32(1).nbytes * feat_dim
    res = array.array('f')
    
    feat_file = os.path.join(feat_dir, 'feature.bin')
    id_file = os.path.join(feat_dir, 'id.txt')
    nr_of_images = len(open(id_file).readline().strip().split())
    printStatus(INFO, 'parsing %s' % feat_file)
    fr = open(feat_file, 'rb')

    s_time = time.time()

    for i in xrange(nr_of_images):
        res.fromfile(fr, feat_dim)
        vec = res
        for d in xrange(feat_dim):
            if vec[d] > max_vals[d]:
                max_vals[d] = vec[d]
            if vec[d] < min_vals[d]:
                min_vals[d] = vec[d]
        del res[:]
    fr.close()

    timecost = time.time() - s_time
    printStatus(INFO, "%g seconds to find min [%g,%g] and max [%g,%g]" % (timecost, min(min_vals), max(min_vals), min(max_vals), max(max_vals)))

    with open(resultfile, 'w') as f:
        f.write('%s\n' % ' '.join(map(str, min_vals)))
        f.write('%s\n' % ' '.join(map(str, max_vals)))
        f.close()



def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = OptionParser(usage="""usage: %prog [options] feat_dir""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1
    
    return process(options, args[0])


if __name__ == "__main__":
    sys.exit(main())

