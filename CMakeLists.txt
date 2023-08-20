cmake_minimum_required(VERSION 3.10)
project( DMS )

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_STANDARD 11)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Debug)" FORCE)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

aux_source_directory(./src/ SRC_LIST)
 

add_executable(${PROJECT_NAME} ${SRC_LIST})

include_directories( include/ncnn )
find_package( OpenCV REQUIRED )
find_package(Threads REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} asound /home/es912-23/DMSD/lib/libncnn.a  Threads::Threads)

