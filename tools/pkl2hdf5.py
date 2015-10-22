import sys, os
import numpy as np
import cPickle as pkl
import h5py

from basic.common import ROOT_PATH, makedirsforfile, checkToSkip, printStatus
from basic.util import readImageSet
from basic.annotationtable import readConcepts
from simpleknn.bigfile import BigFile

INFO = 'tools.pkl2hdf5.py'

def process(options, pklfile, hdf5file):
    if checkToSkip(hdf5file, options.overwrite):
        return 0

    printStatus(INFO, 'Loading pkl file %s' % pklfile)
    with open(pklfile, 'r') as f:
        data = pkl.load(f)
    printStatus(INFO, 'Found %d elements.' % len(data))

    printStatus(INFO, 'Saving hdf5 file %s' % hdf5file)
    with h5py.File(hdf5file,'w') as f:
        for k,v in data.items():
            printStatus(INFO, 'Dumping %s' % k)
            f[k] = v

    printStatus(INFO, 'Done.')

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] pklfile outhdf5file""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")

    (options, args) = parser.parse_args(argv)
    if len(args) < 2:
        parser.print_help()
        return 1

    return process(options, args[0], args[1])


if __name__ == "__main__":
    sys.exit(main())
