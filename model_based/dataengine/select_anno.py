import sys, os
from basic.common import checkToSkip, ROOT_PATH, makedirsforfile
from basic.annotationtable import readConcepts, readAnnotationsFrom, writeAnnotationsTo, writeConceptsTo
from basic.data import readImageSet


if __name__ == '__main__':
    args = sys.argv[1:]
    rootpath = '/var/scratch2/xirong/VisualSearch'
    srcCollection = args[0]
    annotationName = args[1]
    dstCollection = args[2]
    overwrite = 0

    concepts = readConcepts(srcCollection, annotationName, rootpath)
    todo = []
    for concept in concepts:
        resfile = os.path.join(rootpath, dstCollection, 'Annotations', 'Image', annotationName, '%s.txt'%concept)
        if checkToSkip(resfile, overwrite):
            continue
        todo.append(concept)
    if not todo:
        print ('nothing to do')
        sys.exit(0)


    imset = set(readImageSet(dstCollection, dstCollection, rootpath))

    for concept in todo:
        names,labels = readAnnotationsFrom(srcCollection, annotationName, concept, rootpath=rootpath)
        selected = [x for x in zip(names,labels) if x[0] in imset]
        print concept, len(selected)
        writeAnnotationsTo([x[0] for x in selected], [x[1] for x in selected], dstCollection, annotationName,  concept, rootpath=rootpath)

    writeConceptsTo(concepts, dstCollection, annotationName, rootpath)
