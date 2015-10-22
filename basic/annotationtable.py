import os


from constant import ROOT_PATH

def readAnnotations(inputfile, skip_0=True):
    data = [(str.split(x)[0], int(str.split(x)[1])) for x in open(inputfile).readlines()]
    names = [x[0] for x in data]
    labels = [x[1] for x in data]
    if skip_0:
        idx = [i for i in range(len(names)) if labels[i] != 0]
        names = [names[x] for x in idx]
        labels = [labels[x] for x in idx]
    return (names, labels)

def readAnnotationsFrom(collection, annotationName, concept, skip_0=True, rootpath=ROOT_PATH):
    annotationfile = os.path.join(rootpath, collection, "Annotations", "Image", annotationName, concept + ".txt")
    return readAnnotations(annotationfile, skip_0)


def annotationsExist(collection,annotationName, concept, rootpath=ROOT_PATH):
    annotationfile = os.path.join(rootpath, collection, "Annotations", "Image", annotationName, concept + ".txt")
    return os.path.exists(annotationfile)

def readConcepts(collection, annotationName, rootpath=ROOT_PATH):
    conceptfile = os.path.join(rootpath, collection, "Annotations",  annotationName)
    return [x.strip() for x in open(conceptfile).readlines() if x.strip()]

def conceptsExist(collection, annotationName, rootpath=ROOT_PATH):
    conceptfile = os.path.join(rootpath, collection, "Annotations",  annotationName)
    return os.path.exists(conceptfile)

def writeConcepts(concepts, resultfile):
    try:
        os.makedirs(os.path.split(resultfile)[0])
    except Exception, e:
        #print e
        pass
    fout = open(resultfile, "w")
    fout.write("\n".join(concepts) + "\n")
    fout.close()

def writeConceptsTo(concepts, collection, annotationName, rootpath=ROOT_PATH):
    resultfile = os.path.join(rootpath, collection, "Annotations", annotationName)
    writeConcepts(concepts, resultfile)


def writeAnnotations(names, labels, resultfile):
    try:
        os.makedirs(os.path.split(resultfile)[0])
    except:
        pass
    fout = open(resultfile, "w")
    fout.write("".join(["%s %g\n" % (im,lab) for (im,lab) in zip(names,labels)]))
    fout.close()
    
def writeAnnotationsTo(names, labels, collection, annotationName, concept, rootpath=ROOT_PATH):
    annotationfile = os.path.join(rootpath, collection, "Annotations", "Image", annotationName, concept + ".txt")
    writeAnnotations(names, labels, annotationfile)


if __name__ == '__main__':
    collection = 'mirflickr08'
    annotationName = 'conceptsmir14.txt'
    concepts = readConcepts(collection, annotationName)
    for concept in concepts:
        names,labels = readAnnotationsFrom(collection, annotationName, concept, skip_0=True)
        pos_set = [names[i] for i in range(len(names)) if labels[i]>0]
        neg_set = [names[i] for i in range(len(names)) if labels[i]<0]
        print concept, len(pos_set), len(neg_set)

    
