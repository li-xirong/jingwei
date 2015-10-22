#!/bin/bash
# use as 'source start.sh'

export SURVEY_CODE=/home/urix/mm2015-tut/tagrelcodebase
#export SURVEY_CODE=/Users/xirong/Documents/bitbucket/tagrelcodebase
export PYTHONPATH=$SURVEY_CODE:$PYTHONPATH
export SURVEY_DATA="$(dirname "$SURVEY_CODE")/data"
#export SURVEY_DATA=/Users/xirong/mm15tut
export SURVEY_DB="$SURVEY_DATA/surveyruns"

# matlab start script should be in this path
export MATLAB_PATH=/usr/local/

