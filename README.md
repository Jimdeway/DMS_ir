# DMS
# ncnn creat
$ git clone https://github.com/Tencent/ncnn.git
$ cd ncnn
$ git submodule update --init
$ mkdir build && cd build
$ . /opt/bsp-5.4.70-2.3.3/environment-setup-aarch64-poky-linux # You can refer to it from doc/BSP.md
$ cmake ..
$ make -j`nproc` && make install
