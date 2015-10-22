import sys, os
import numpy as np

from basic.common import ROOT_PATH, makedirsforfile, checkToSkip, printStatus
from basic.data import readImageSet
from basic.annotationtable import readConcepts
from simpleknn.bigfile import BigFile

DEFAULT_BLOCK_SIZE = 1000
INFO = 'surveypaper.code.concat_features'

'''
def get_dim(feature):
    if feature == 'color64+dsift':
        return 1088
    if feature.startswith('tagfea'):
        assert ('tagfea' == feature.split('-')[0])
        trainCollection = feature.split('-')[1]
        concepts = readConcepts(trainCollection, 'conceptstagfea.txt')
        return len(concepts)
    raise Exception('unknown feature %s' % feature)
'''

def process(options, collection, features):
    rootpath = options.rootpath
    blocksize = options.blocksize
    newfeature = options.newfeature

    src_features = features.split(',')    
    
    if not newfeature:
        newfeature = '+'.join(src_features)

    new_feat_dir = os.path.join(rootpath, collection, 'FeatureData', newfeature)
    new_feat_file = os.path.join(new_feat_dir, 'feature.bin')
    if checkToSkip(new_feat_file, options.overwrite):
        return 0

    imset = readImageSet(collection, collection, rootpath)
    nr_to_read = len(imset) / blocksize
    if blocksize*nr_to_read < len(imset):
        nr_to_read += 1

    src_feat_files = [BigFile(os.path.join(rootpath, collection,'FeatureData',feature)) for feature in src_features]
    nr_fea = len(src_feat_files)
    src_feat_dims = [x.ndims for x in src_feat_files]
    new_feat_dim = sum(src_feat_dims)

    printStatus(INFO, '%s -> %s,%d' % (' '.join(['(%s,%d)' % (x[0],x[1]) for x in zip(src_features,src_feat_dims)]), newfeature, new_feat_dim))

    makedirsforfile(new_feat_file)
    fw = open(new_feat_file, 'wb')
    id_images = []

    for t in range(nr_to_read):
        start = t*blocksize
        end = min(len(imset), start + blocksize)
        printStatus(INFO, 'processing images from %d to %d' % (start, end-1))
        todo = imset[start:end]
        nr_images = len(todo)

        if nr_images == 0:
            break

        mapping = dict(zip(todo, range(nr_images)))
        renamed = [None] * nr_fea
        vectors = [None] * nr_fea
        
        for i in range(nr_fea):
            tmp_names, tmp_vecs = src_feat_files[i].read(todo)
            assert(len(tmp_names) == nr_images)
            renamed[i] = [None] * nr_images
            vectors[i] = [None] * nr_images

            for name,vec in zip(tmp_names, tmp_vecs):
                j = mapping[name]
                renamed[i][j] = name
                vectors[i][j] = vec

        for j in range(nr_images):
            new_vec = vectors[0][j]
            for i in range(1,nr_fea):
                assert (renamed[i-1][j] == renamed[i][j]), '%s %s' % (renamed[i-1][j], renamed[i][j])
                new_vec += vectors[i][j]
            assert(len(new_vec) == new_feat_dim)
            
            name = renamed[0][j]
            np.array(new_vec, dtype=np.float32).tofile(fw)
            id_images.append(name)

    fw.close()
    new_id_file = os.path.join(new_feat_dir, 'id.txt')
    fw = open(new_id_file, 'w')
    fw.write(' '.join(id_images))
    fw.close()

    fw = open(os.path.join(new_feat_dir,'shape.txt'), 'w')
    fw.write('%d %d' % (len(id_images), new_feat_dim))
    fw.close()




def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection features""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--blocksize", default=DEFAULT_BLOCK_SIZE, type="int", help="nr of feature vectors loaded per time (default: %d)" % DEFAULT_BLOCK_SIZE)
    parser.add_option("--newfeature", default=None, type="string", help="newfeature (default: None)")
    

    (options, args) = parser.parse_args(argv)
    if len(args) < 2:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1])


if __name__ == "__main__":
    sys.exit(main())
