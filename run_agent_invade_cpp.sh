#!/bin/bash

i=$1
param_file=$2
results_dir=$3


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
command="code_c/run_invade  ${num_objects} ${num_sounds} ${graph_path} ${num_trials} ${results_dir} ${rep}"
echo ${command}
${command}

exit $?
