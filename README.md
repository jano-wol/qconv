## qconv
qconv is a performant C++ library using AVX2 instructions to implement fundamental algorithms like: minimum search, relu, vector addition, quantitized fully connected layers, quantitized convolutional layers with kernel 3x3. (These algorithms and layers are the buiding blocks of neural networks.) All of the implemented algorithms are tested by google tests, and benchmarked by google benchmark.

## Setup repo
Supported os: Linux  

Known dependencies:  
cmake  
ninja  
clang  

#Method 1 (tested on Ubuntu 22.04):  
sudo snap install cmake --classic  
sudo apt-get -y install ninja-build  
sudo apt install clang  
sudo apt install libstdc++-12-dev  
sudo apt-get install libc++-dev  
sudo apt install libc++abi-dev  

# Method 2 (docker):


## Build and test the repo
After setup the repo can be configured, build, tested and benchmarked with the following commands:  
./source/scripts/configure.sh release  
./source/scripts/build.sh release  
./source/scripts/test.sh release  
./source/scripts/bench.sh release
  
## Compiler
The default compiler of qconv is clang. gcc is also perfectly fine to compile qconv, 
the only advantage of clang is that experience shows that the google benchmark results are more consistent with the clang compiled binaries than with the gcc compiled binaries. 

