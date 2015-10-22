#!/bin/bash
source start.sh

# setup simpleknn
# we now provide pre-compiled dll/so files for linux/mac/win32/win64
#if [ ! -f simpleknn/cpp/libsearch.so ]; then
#	cd simpleknn/cpp
#	make clean
#	make
#fi

# setup tagprop
if [ ! -f model_based/tagprop/TagProp/tagprop_learn.m ]; then
	cd model_based/tagprop
	./setup-tagprop.sh
fi

cd $SURVEY_CODE

# check the status of the required files and libraries
python tools/check_availability.py


