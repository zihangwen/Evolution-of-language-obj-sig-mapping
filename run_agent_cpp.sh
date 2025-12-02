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
model_name=$(sed -n "${line}p" ${param_file} | awk '{print $3}')
graph_path=$(sed -n "${line}p" ${param_file} | awk '{print $4}')
num_runs=$(sed -n "${line}p" ${param_file} | awk '{print $5}')
sample_times=$(sed -n "${line}p" ${param_file} | awk '{print $6}')
temperature=$(sed -n "${line}p" ${param_file} | awk '{print $7}')


##### ----- Run ----- #####
command="code_c/run  ${num_objects} ${num_sounds} ${model_name} ${graph_path} ${num_runs} ${results_dir} ${sample_times} ${temperature} ${rep}"
echo ${command}
${command}

exit $?
