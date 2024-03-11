
## Introduction
Here is the code of paper "Out-of-Core GPU Memory Management for Incremental Graph Processing".
## Software Requirements
* CUDA == 11.4
* GCC == 7.5.0
* CMake == 3.18.5
* Linux/Unix

## Hardware Requirements
* Intel/AMD X64 CPUs
* 128GB RAM (at least)
* NVIDIA GTX A100 or A5000
* 24GB Storage space (at least)

## Requirements and Compilation

Under the root directory of the project, execute the following commands to compile the project.

```zsh
mkdir build && cd build
cmake ..
make
```
## Get Started
The include/framework/ directory contains the incremental graph computation engine and static graph computation engine. The application codes (e.g., PageRank, SSSP, etc.) can be found in the samples/ directory. Useful helper files for generating graph data structures ), reading the graph inputs in the correct format ((utils/)are also provided.

## Execute Algorithms(for example pagerank)
Execute the following commands to test the pagerank algorithm of the binary file.You can switch the binary file name to other algorithms like hybrid_sssp, hybrid_bfs,hybrid_cc.
The executable takes the following command-line parameters:

-graphfile : Graphfile is used.
-format : Large graph which size is out of gpu memory.
-cache : Optional parameter to specify the cache size. Default is 0.
-updatesize : Total stream size of updatefile.
-updatefile : The batch information file.

```zsh
cd build
./hybrid_pr -graphfile {input graph path} -format market_big -hybrid 0 -SEGMENT 512 -weight_num 1 -weight 1 -update_size {user_stream_size} -updatefile {update graph path} -cache 0
```



## Input
Both the input graph and update graph are edge lists.
Each input graph starts with 'source destination' where source is the source vertex and destination is the destination vertex of an edge. 
Each update graph starts with 'operation source destination' where source is the source vertex and destination is the destination vertex of an edge,the operation "a" means insert edges,"d" means delete edges. 
For complete information about the "update_stream_size" file, please refer to the Stream size example below.

Input Graph Example:

```zsh
0 1
0 2
1 2
1 3
2 4
3 4
```
Update Graph Example:

```zsh
a 0 3
a 0 4
a 1 4
a 2 3
d 0 1
d 0 2
d 1 2
d 1 3
d 2 4
d 3 4
```

Stream Size Example:
The following is “Stream size example” file containing two snapshots, each snapshot contains two edges inserted and two edges deleted.
```zsh
2 3 # "2" is the number of added edges, "3" is the number of deleted edges.
2 3
```



