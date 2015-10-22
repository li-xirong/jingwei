#!/bin/bash

# Get and setup Tagprop

wget http://lear.inrialpes.fr/people/guillaumin/code/TagProp_0.2.tar.gz
tar xvzf TagProp_0.2.tar.gz
rm TagProp_0.2.tar.gz

echo "Patching for efficiency..."
patch TagProp/sigmoids.m < sigmoids.m.patch
patch TagProp/tagprop_learn.m < tagprop_learn.m.patch
patch TagProp/tagprop_predict.m < tagprop_predict.m.patch

echo "Compiling mex..."
cd TagProp
$MATLAB_PATH/bin/matlab -nodesktop -nosplash -nojvm -r "mex tagpropCmt.c; exit"
cd ..


