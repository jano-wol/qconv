FROM ubuntu:22.04
RUN apt -qq update && DEBIAN_FRONTEND=noninteractive apt -qq install build-essential cmake git -y
RUN apt-get install -y ninja-build 
RUN apt install -y clang  
RUN apt install -y libstdc++-12-dev  
RUN apt-get install -y libc++-dev  
RUN apt install -y libc++abi-dev   
WORKDIR /qconv

