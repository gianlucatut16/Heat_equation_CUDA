#!/bin/bash

echo "Compile program"
nvcc HeatEquation.cu -o HeatEquation
echo 


echo "Execute program"
srun -N 1 -n 1 --gpus-per-task=1 -p gpus --reservation=maintenance ./HeatEquation
echo 

echo "Clear temporary file"
rm HeatEquation
echo 
