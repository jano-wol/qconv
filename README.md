# qconv
qconv is a performant C++ library implementing fundamental algorithms like: minimum search, relu, vector addition, fully connected layer, quantitized convolutional layer (kernel 3x3) with AVX2 instructions. All of the implemented algorithms are tested by google tests, and benchmarked by google benchmark.

# Setup repo
Supported os: Linux  

Known dependencies:  
cmake  
ninja  
clang  

# Check setup
To check that the repo is working, run the followings from the root of the repo:  
./source/scripts/configure.sh release  
./source/scripts/build.sh release  
./source/scripts/test.sh release  
./source/scripts/bench.sh release
  
# Compiler
The default compiler of qconv is clang. Gcc is also perfectly fine to compile qconv, 
the only adventage of clang is that experience shows that the google benchmark results are more consistent with the clang compiled binaries than with the gcc compiled binaries. 

