
@echo off

call ..\start.bat

set rootpath=%SURVEY_DATA%
set trainCollection=train10k
set testCollection=mirflickr08
set feature=vgg-verydeep-16-fc7relu
set distance=cosine
set k=1000
set tagger=tagvote
set overwrite=0


set annotationName=concepts130.txt
python %SURVEY_CODE%/instance_based/apply_tagger.py %testCollection% %trainCollection% %annotationName% %feature% --tagger %tagger% --distance %distance% --k %k% --overwrite %overwrite%


if "%testCollection%" == "mirflickr08" (
    set testAnnotationName=conceptsmir14.txt
)
if "%testCollection%" == "flickr51" (
    set testAnnotationName=concepts51ms.txt
)
if "%testCollection%" == "flickr81" (
    set testAnnotationName=concepts81.txt
)

set conceptfile=%rootpath%/%testCollection%/Annotations/%testAnnotationName%
set tagvotesfile="%rootpath%/%testCollection%/autotagging/%testCollection%/%trainCollection%/%annotationName%/%tagger%/%feature%,%distance%knn,%k%/id.tagvotes.txt"
set resultfile=%SURVEY_DB%/%trainCollection%_%testCollection%_%feature%,tagvote.pkl

if NOT EXIST %tagvotesfile% (
    echo "%tagvotesfile% does not exist"
) else (
    python %SURVEY_CODE%/postprocess/pickle_tagvotes.py %conceptfile% %tagvotesfile% %resultfile% --overwrite %overwrite%
)

@pause

