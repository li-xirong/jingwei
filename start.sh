#!/bin/bash
# use as 'source start.sh'

export SURVEY_CODE=$HOME/github/jingwei
export PYTHONPATH=$SURVEY_CODE:$PYTHONPATH
export SURVEY_DATA=/local/xirong/VisualSearch

if [ ! -d "$SURVEY_DATA" ]; then
    export SURVEY_DATA=$HOME/VisualSearch
fi

export SURVEY_DB="$SURVEY_DATA/surveyruns"
# matlab start script should be in this path
export MATLAB_PATH=/usr/local/

