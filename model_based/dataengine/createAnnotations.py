import sys
import os


from basic.common import ROOT_PATH,checkToSkip,CmdOptions,printError,printStatus,makedirsforfile
from basic.annotationtable import readConcepts,writeConcepts,writeAnnotations,annotationsExist,conceptsExist

from positiveengine import PositiveEngine,SelectivePositiveEngine
from negativeengine import STRING_TO_NEGATIVE_ENGINE 


class CreateAnnotationsOptions (CmdOptions):
    def __init__(self):
        CmdOptions.__init__(self)
        self.addOption("collection", "")
        self.addOption("annotationName","")
        self.addOption("pos_source", "tagged/lemm")
        self.addOption("select_pos", "random")
        self.addOption("nr_pos", "")
        self.addOption("neg_filter", "wn")
        self.addOption("neg_pos_ratio", 5)
        self.addOption("nr_pos_bags", 1)
        self.addOption("nr_neg_bags", 1)
        self.addOption("tpp","lemm")
        

    def printHelp(self):
        CmdOptions.printHelp(self)
        print """
              --collection
              --annotationName
              --pos_source [default:tagged/lemm]
              --select_pos [default: random]
              --nr_pos number of positive training examples per bag
              --neg_filter [default: wn]
              --neg_pos_ratio [default: 5]
              --nr_pos_bags [default: 1]
              --nr_neg_bags [default: 5]
              --tpp [default: lemm]
              """


    def checkArgs(self):
        if not CmdOptions.checkArgs(self):
            return False
        if self.getString('select_pos') != 'random' and self.getInt('nr_pos_bags') > 1:
            printError(self.__class__.__name__, "given select_pos=random, nr_pos_bags shall be 1")
            return False
        return True


def generate_new_annotation_template(cmdOpts):
    newName = os.path.splitext(cmdOpts.getString('annotationName'))[0]
    newName += '.' + cmdOpts.getString('select_pos') + cmdOpts.getString('nr_pos') + '.%d'
    nr_neg = cmdOpts.getInt('nr_pos') * cmdOpts.getInt('neg_pos_ratio')   
    newName += ('.random%s%d' % (cmdOpts.getString('neg_filter'),nr_neg)) + '.%d.txt' 
    return newName


if __name__ == "__main__":
    cmdOpts = CreateAnnotationsOptions()
    if not cmdOpts.parseArgs(sys.argv[1:]):
        sys.exit(0)
                    
    overwrite = cmdOpts.getInt('overwrite')                
    rootpath = cmdOpts.getString('rootpath')
    collection = cmdOpts.getString('collection')
    annotationName = cmdOpts.getString('annotationName')
    tpp = cmdOpts.getString('tpp')
    nr_pos = cmdOpts.getInt('nr_pos')
    pos_source = cmdOpts.getString('pos_source')
    select_pos = cmdOpts.getString('select_pos')
    neg_filter = cmdOpts.getString('neg_filter')
    neg_pos_ratio = cmdOpts.getInt('neg_pos_ratio')
    nr_pos_bags = cmdOpts.getInt('nr_pos_bags')
    nr_neg_bags = cmdOpts.getInt('nr_neg_bags')
    nr_neg = nr_pos * neg_pos_ratio

    concepts = readConcepts(collection, annotationName)
    annotationNameStr = generate_new_annotation_template(cmdOpts)

    nr_skipped = 0
    newAnnotationNames = [None] * (nr_pos_bags * nr_neg_bags)

    for idxp in range(nr_pos_bags):
        for idxn in range(nr_neg_bags):
            anno_idx = idxp * nr_neg_bags + idxn
            newAnnotationNames[anno_idx] = annotationNameStr % (idxp, idxn)
            resultfile = os.path.join(rootpath,collection,'Annotations',newAnnotationNames[anno_idx])
            if checkToSkip(resultfile,overwrite):
                nr_skipped += 1
                continue
            writeConcepts(concepts,resultfile)

    first,second,last = annotationNameStr.split('%d')
    scriptfile = os.path.join(rootpath,collection,'annotationfiles',first + '0-%d'%(nr_pos_bags-1) + second + '0-%d'%(nr_neg_bags-1) + last)
    makedirsforfile(scriptfile)
    fout = open(scriptfile,'w')
    fout.write('\n'.join(newAnnotationNames) + '\n')
    fout.close()

    if nr_skipped == (nr_pos_bags * nr_neg_bags):
        sys.exit(0)

        
    if select_pos == 'random':
        pe = PositiveEngine(collection)
    else:
        pe = SelectivePositiveEngine(collection, pos_source)
    ne = STRING_TO_NEGATIVE_ENGINE[neg_filter](collection)


    for concept in concepts:
        for idxp in range(nr_pos_bags):
            pos_set = pe.sample(concept, nr_pos)
            for idxn in range(nr_neg_bags):
                anno_idx = idxp * nr_neg_bags + idxn
                newAnnotationName = newAnnotationNames[anno_idx]
                resultfile = os.path.join(rootpath,collection,'Annotations','Image',newAnnotationName,'%s.txt'%concept)
                if checkToSkip(resultfile,overwrite):
                    break 
                neg_set = ne.sample(concept, nr_neg)
                assert(len(set(pos_set).intersection(set(neg_set))) == 0)
                newlabels = [1] * len(pos_set) + [-1] * len(neg_set)
                newnames = pos_set + neg_set
                printStatus('dataengine.createAnnotations', "anno(%s,%d) %d pos %d neg -> %s" % (concept,anno_idx,len(pos_set),len(neg_set),resultfile))
                writeAnnotations(newnames, newlabels, resultfile)

