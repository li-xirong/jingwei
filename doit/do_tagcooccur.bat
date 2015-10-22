@echo off

call ..\start.bat

set rootpath=%SURVEY_DATA%
set trainCollection=train10k
set testCollection=mirflickr08
set tagger=tagcooccurplus
set feature=vgg-verydeep-16-fc7relu
set overwrite=0



if "%testCollection%" == "mirflickr08" (
    set conceptset=conceptsmir14
)
if "%testCollection%" == "flickr51" (
    conceptset=concepts51ms
)
if "%testCollection%" == "flickr81" (
    set conceptset=concepts81
)

set annotationName=%conceptset%.txt
set conceptfile=%rootpath%/%trainCollection%/Annotations/%annotationName%


if "%tagger%" == "tagcooccur"  (
    set kc=0
    ::set tagvotesfile=%rootpath%/%testCollection%/autotagging/%testCollection%/%trainCollection%/%annotationName%/cotag,m25,kr4,kd11,ks9,kc%kc%,bonus0/id.tagvotes.txt
    set tagvotesfile=%rootpath%/%testCollection%/autotagging/%testCollection%/%trainCollection%/%annotationName%/cotag/id.tagvotes.txt
    set pklfile=%SURVEY_DB%/%trainCollection%_%testCollection%_%tagger%.pkl
) else (
    set kc=1
    ::set tagvotesfile=%rootpath%/%testCollection%/autotagging/%testCollection%/%trainCollection%/%annotationName%/cotag,m25,kr4,kd11,ks9,kc%kc%,bonus0/%feature%/id.tagvotes.txt
    set tagvotesfile=%rootpath%/%testCollection%/autotagging/%testCollection%/%trainCollection%/%annotationName%/cotag/%feature%/id.tagvotes.txt
    set pklfile=%SURVEY_DB%/%trainCollection%_%testCollection%_%feature%,%tagger%.pkl
)

python %SURVEY_CODE%/instance_based/compute_concept_rank_based_on_tagcooccur.py %trainCollection% %annotationName%
    
if "%tagger%" == "tagcooccurplus" (
    python %SURVEY_CODE%/instance_based/tagrel_to_concept_rank.py %SURVEY_DB%/%trainCollection%_%testCollection%_%feature%,tagvote.pkl %rootpath%/%testCollection%/autotagging/%trainCollection%_%testCollection%_tagvote,%feature%_%conceptset%_rank.pkl
)

python %SURVEY_CODE%/instance_based/apply_tagcooccur.py %trainCollection% %annotationName% %testCollection% --kc %kc% --bonus 0 --feature %feature%

if EXIST %tagvotesfile% (
    python %SURVEY_CODE%/postprocess/pickle_tagvotes.py %conceptfile% %tagvotesfile% %pklfile%
)

@PAUSE
