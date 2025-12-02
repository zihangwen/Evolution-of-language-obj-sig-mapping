#!/bin/bash

i=$1
param_file=$2
package_name=$3
results_dir=$4

##### ----- Virtual environment ----- #####
# have job exit if any command returns with non-zero exit status (aka failure)
set -e
# replace env-name on the right hand side of this line with the name of your conda environment
ENVNAME=$package_name
# if you need the environment directory to be named something other than the environment name, change this line
export ENVDIR=$ENVNAME
# these lines handle setting up the environment; you shouldn't have to modify them
export PATH
mkdir -p $ENVDIR
tar -xzf ${ENVNAME}.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate


##### ----- Parameter ----- #####
line_count=$(wc -l < "${param_file}")
line=$((i % line_count + 1))
rep=$((i / line_count))
# line=$((i + 1))

num_objects=$(sed -n "${line}p" ${param_file} | awk '{print $1}')
num_sounds=$(sed -n "${line}p" ${param_file} | awk '{print $2}')
graph_path=$(sed -n "${line}p" ${param_file} | awk '{print $3}')
num_trials=$(sed -n "${line}p" ${param_file} | awk '{print $4}')


##### ----- Run ----- #####
echo  code/sim_invade.py ${num_objects} ${num_sounds} ${graph_path} ${num_trials} ${results_dir} ${rep}
python code/sim_invade.py ${num_objects} ${num_sounds} ${graph_path} ${num_trials} ${results_dir} ${rep}

exit $?
