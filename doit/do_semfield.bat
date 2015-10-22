
@echo off

call ..\start.bat

set rootpath=%SURVEY_DATA%
set trainCollection=train10k
set testCollection=mirflickr08
set tagsimMethod=avgcos
set overwrite=0

python %SURVEY_CODE%/instance_based/dosemtagrel.py %testCollection% %trainCollection% %tagsimMethod% --overwrite %overwrite%


if "%testCollection%" == "mirflickr08" (
    set annotationName=conceptsmir14.txt
)
if "%testCollection%" == "flickr51" (
    set annotationName=concepts51ms.txt
)
if "%testCollection%" == "flickr81" (
    set annotationName=concepts81.txt
)

set conceptfile=%rootpath%/%testCollection%/Annotations/%annotationName%
set tagvotesfile="%rootpath%/%testCollection%/tagrel/%testCollection%/%trainCollection%/%tagsimMethod%-wn/id.tagvotes.txt"
set resultfile=%SURVEY_DB%/%trainCollection%_%testCollection%_semfield.pkl


if NOT EXIST %tagvotesfile% (
    echo "%tagvotesfile% does not exist"
) else (
    python %SURVEY_CODE%/postprocess/pickle_tagvotes.py %conceptfile% %tagvotesfile% %resultfile% --overwrite %overwrite%
)

@pause
