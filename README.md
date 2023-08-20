# DMS
# ncnn creat
$ git clone https://github.com/Tencent/ncnn.git
$ cd ncnn
$ git submodule update --init
$ mkdir build && cd build
$ cmake ..
$ make -j`nproc` && make install

move install/include and lib folder to DMS folder

└─DMS
    ├─include
    │  └─ncnn
    ├─lib
    │  └─libncnn.a
    └─src

# Build DMS
remenber to change libncnn.a path in CMakeLists.txt

$ mkdir build && cd build
$ cmake ..
$ make -j`nproc`