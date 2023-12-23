# qconv
qconv is a performant C++ library using AVX2 instructions to implement fundamental algorithms like: minimum search, relu, vector addition, quantitized fully connected layers, quantitized convolutional layers with kernel 3x3. (These algorithms and layers are the buiding blocks of neural networks.) All of the implemented algorithms are tested by google tests, and benchmarked by google benchmark.

# Setup qconv
Supported os: Linux  

Known dependencies:  
cmake  
ninja  
clang  

### Setup Method 1 (tested on Ubuntu 22.04):  
sudo snap install cmake --classic  
sudo apt-get -y install ninja-build  
sudo apt install clang  
sudo apt install libstdc++-12-dev  
sudo apt-get install libc++-dev  
sudo apt install libc++abi-dev  

### Setup Method 2 (Docker):
A Docker image can be builded with the following command (casted from the root of the repo):  
sudo ./source/scripts/misc/docker/build.sh 

# Build, test and benchmark qconv
After the setup is done, the repo can be configured, builded, tested and benchmarked with the following commands (casted from the root of the repo):  
[OPTIONAL] ./source/scripts/misc/docker/run.sh (this command is needed only in case setup was done by Docker)  
./source/scripts/configure.sh release  
./source/scripts/build.sh release  
./source/scripts/test.sh release  
./source/scripts/bench.sh release
  
# Compiler
The default compiler of qconv is clang. gcc is also perfectly fine to compile qconv, 
the only advantage of clang is that experience shows that the google benchmark results are more consistent with the clang compiled binaries than with the gcc compiled binaries. 

