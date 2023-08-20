# DMS
# ncnn creat
```
$ git clone https://github.com/Tencent/ncnn.git
$ cd ncnn
$ git submodule update --init
$ mkdir build && cd build
$ cmake ..
$ make -j`nproc` && make install
```
move install/include and lib folder to DMS folder
```
└─DMS
    ├─include
    │  └─ncnn
    ├─lib
    │  └─libncnn.a
    └─src
```
# Build DMS
remenber to change libncnn.a path in CMakeLists.txt
```
$ cmake .
$ make -j`nproc`
```
# Reference
PFLD_GhostOne:https://github.com/AnthonyF333/PFLD_GhostOne
PFLD-ncnn:https://github.com/Brightchu/pfld-ncnn
mobilefacenet:https://github.com/liguiyuan/mobilefacenet-ncnn
YOLOv5:https://github.com/ultralytics/yolov5
YOLOv5 lite:https://github.com/ppogg/YOLOv5-Lite
ncnn:https://github.com/Tencent/ncnn
alsa:https://gist.github.com/ghedo/963382
ear:https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf?spm=5176.100239.blogcont336184.8.b7697a07zKT7r&file=05.pdf
Eular_angle:https://blog.csdn.net/u014090429/article/details/100762308
Fatigue_detect:https://blog.csdn.net/haiyangyunbao813/article/details/105760197
