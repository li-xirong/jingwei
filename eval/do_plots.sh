codepath=/home/urix/shared/tagrelcodebase
survedydb=/home/urix/surveyruns

echo "Converting pkl file into hdf5 for MATLAB..."
mkdir $surveydb/hdf5
for i in $surveydb/*.pkl
do
  python tools/pkl2hdf5.py $i $i.h5
done
mv $surveydb/*.h5 $surveydb/hdf5/

echo "Starting MATLAB to draw nice plots..."
matlab -nodesktop -nosplash -r "do_plots('$surveydb/hdf5/'); exit;"

echo "Plots should be saved as pdf in $codepath/eval/plots/"
