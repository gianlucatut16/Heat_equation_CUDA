 In this project, I use a parallel algorithm to solve the heat equation in two dimensions on University of Naples Federico II’s HPC cluster I.Bi.S.Co. (Infrastructure for Big data and Scientific Computing).
 First of all there's the use of the finite difference method that allows to solve the heat equation.
 This method divide the spatial domain into a grid, and discretize the time domain into discrete time steps.
 In order to do this, I used CUDA, which involves parallelizing the computation on the GPU using CUDA programming techniques. The heat equation solver can be implemented using CUDA in different ways, for example through the domain decomposition, data allocation, data initialization, synchronization, memory access and memory transfer etc.
 The time iterations are 1000 steps so the heat is allowed to move on all the grid. The grid state is monitored every 200 time steps. 
 In the project i used different block size for the parallel computing and save the execution time in order to find out the trend the latter follows incresing the former.
 
