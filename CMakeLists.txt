# ibmGBT
if(USE_CUDA)
  cmake_minimum_required(VERSION 3.11)
  enable_language(CUDA)
elseif(USE_GPU OR APPLE)
  cmake_minimum_required(VERSION 3.2)
else()
  cmake_minimum_required(VERSION 2.8)
endif()

PROJECT(lightgbm LANGUAGES C CXX CUDA)

OPTION(USE_MPI "Enable MPI-based parallel learning" OFF)
OPTION(USE_OPENMP "Enable OpenMP" ON)
OPTION(USE_CUDA "Enable CUDA-accelerated training" ON) # ibmGBT
OPTION(USE_GPU "Enable GPU-accelerated training" ON) # ibmGBT
OPTION(USE_SWIG "Enable SWIG to generate Java API" OFF)
OPTION(USE_HDFS "Enable HDFS support (EXPERIMENTAL)" OFF)
OPTION(USE_R35 "Set to ON if your R version is not earlier than 3.5" OFF)
OPTION(USE_CH "Enable the CHAllocator" ON)

if(APPLE)
    OPTION(APPLE_OUTPUT_DYLIB "Output dylib shared library" OFF)
endif(APPLE)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.8.2")
    message(FATAL_ERROR "Insufficient gcc version")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "3.8")
    message(FATAL_ERROR "Insufficient Clang version")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "8.1.0")
    message(FATAL_ERROR "Insufficient AppleClang version")
  endif()
  cmake_minimum_required(VERSION 3.12)
elseif(MSVC)
  if(MSVC_VERSION LESS 1900)
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} doesn't support required C++11 features. Please use a newer MSVC.")
  endif()
  cmake_minimum_required(VERSION 3.8)
endif()

if(USE_SWIG)
  find_package(SWIG REQUIRED)
  find_package(Java REQUIRED)
  find_package(JNI REQUIRED)
  include(UseJava)
  include(UseSWIG)
  set(SWIG_CXX_EXTENSION "cxx")
  set(SWIG_EXTRA_LIBRARIES "")
  set(SWIG_JAVA_EXTRA_FILE_EXTENSIONS ".java" "JNI.java")
  set(SWIG_MODULE_JAVA_LANGUAGE "JAVA")
  set(SWIG_MODULE_JAVA_SWIG_LANGUAGE_FLAG "java")
  set(CMAKE_SWIG_OUTDIR "${CMAKE_CURRENT_BINARY_DIR}/java")
  include_directories(Java_INCLUDE_DIRS)
  include_directories(JNI_INCLUDE_DIRS)
  include_directories($ENV{JAVA_HOME}/include)
  if(WIN32)
      FILE(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/com/microsoft/ml/lightgbm/windows/x86_64")
      include_directories($ENV{JAVA_HOME}/include/win32)
  elseif(APPLE)
      FILE(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/com/microsoft/ml/lightgbm/osx/x86_64")
      include_directories($ENV{JAVA_HOME}/include/darwin)
  else()
      FILE(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/com/microsoft/ml/lightgbm/linux/x86_64")
      include_directories($ENV{JAVA_HOME}/include/linux)
  endif()
endif(USE_SWIG)

if(USE_R35)
    ADD_DEFINITIONS(-DR_VER_ABOVE_35)
endif(USE_R35)

if(USE_MPI)
    find_package(MPI REQUIRED)
    ADD_DEFINITIONS(-DUSE_MPI)
    MESSAGE(STATUS "MPI libraries: " ${MPI_LIBRARIES})
    MESSAGE(STATUS "MPI C++ libraries: " ${MPI_CXX_LIBRARIES})
else()
    ADD_DEFINITIONS(-DUSE_SOCKET)
endif(USE_MPI)

if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    # Ignore unknown #pragma warning
    if((CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    endif()
endif(USE_OPENMP)

if(USE_GPU)
    # ibmGBT
    #SET(OpenCL_LIBRARY /usr/local/cuda/lib64/libOpenCL.so)
    #SET(OpenCL_INCLUDE_DIR /usr/local/cuda/include)

    #SET(BOOST_COMPUTE_HEADER_DIR ${PROJECT_SOURCE_DIR}/compute/include)
    #include_directories(${BOOST_COMPUTE_HEADER_DIR})
    # find_package(OpenCL REQUIRED)
    # include_directories(${OpenCL_INCLUDE_DIRS})
    #MESSAGE(STATUS "OpenCL include directory: " ${OpenCL_INCLUDE_DIRS})
    #if (WIN32)
    #    set(Boost_USE_STATIC_LIBS ON)
    # endif()

    #find_package(Boost 1.56.0 COMPONENTS filesystem system REQUIRED)
    #if (WIN32)
    #    # disable autolinking in boost
    #    add_definitions(-DBOOST_ALL_NO_LIB)
    #endif()
    #include_directories(${Boost_INCLUDE_DIRS})
    ADD_DEFINITIONS(-DUSE_GPU)
endif(USE_GPU)

# ibmGBT
if(USE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    LIST(APPEND CMAKE_CUDA_FLAGS -Xcompiler=-fopenmp -Xcompiler=-fPIC -Xcompiler=-Wall -lineinfo)
    CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS 7.0)

    LIST(APPEND CMAKE_CUDA_FLAGS ${CUDA_ARCH_FLAGS})
    if(CMAKE_BUILD_TYPE MATCHES Release)
      LIST(APPEND CMAKE_CUDA_FLAGS -03)
    endif()

    message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
    string(REPLACE ";" " " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
    message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
    set(CMAKE_CUDA_FLAGS_DEBUG -G)
    set(CMAKE_CUDA_FLAGS_RELEASE -lineinfo)

    ADD_DEFINITIONS(-DUSE_CUDA)
    ADD_DEFINITIONS(-DTIMETAG)
    if (NOT DEFINED CMAKE_CUDA_STANDARD)
      set(CMAKE_CUDA_STANDARD 11)
      set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
#    find_package(Boost 1.56.0 COMPONENTS filesystem system REQUIRED)
#    include_directories(${Boost_INCLUDE_DIRS})

    set(BASE_DEFINES
     -DPOWER_FEATURE_WORKGROUPS=12
     -DUSE_CONSTANT_BUF=0
    )

    set(ALLFEATS_DEFINES
     ${BASE_DEFINES}
     -DENABLE_ALL_FEATURES
    )

    set(FULLDATA_DEFINES
     ${ALLFEATS_DEFINES}
     -DIGNORE_INDICES
    )

    #string(REPLACE ";" " " BASE_DEFINES "${BASE_DEFINES}")
    #string(REPLACE ";" " " ALLFEATS_DEFINES "${ALLFEATS_DEFINES}")
    #string(REPLACE ";" " " FULLDATA_DEFINES "${FULLDATA_DEFINES}")

    message(STATUS ALLFEATS_DEFINES: ${ALLFEATS_DEFINES})
    message(STATUS FULLDATA_DEFINES: ${FULLDATA_DEFINES})

    add_library(histo256_sp_const OBJECT src/treelearner/kernels/histogram256.cu)
    set_target_properties(histo256_sp_const PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(
      histo256_sp_const PRIVATE
      -DCONST_HESSIAN=1
      ${BASE_DEFINES}
    )

    add_library(histo256_sp OBJECT src/treelearner/kernels/histogram256.cu)
    set_target_properties(histo256_sp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(
      histo256_sp PRIVATE
      -DCONST_HESSIAN=0
      ${BASE_DEFINES}
    )

    add_library(histo256-allfeats_sp_const OBJECT src/treelearner/kernels/histogram256.cu)
    set_target_properties(histo256-allfeats_sp_const PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(
      histo256-allfeats_sp_const PRIVATE
      -DCONST_HESSIAN=1
      ${ALLFEATS_DEFINES}
    )

    add_library(histo256-allfeats_sp OBJECT src/treelearner/kernels/histogram256.cu)
    set_target_properties(histo256-allfeats_sp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(
      histo256-allfeats_sp PRIVATE
      -DCONST_HESSIAN=0
      ${ALLFEATS_DEFINES}
    )

    add_library(histo256-fulldata_sp_const OBJECT src/treelearner/kernels/histogram256.cu)
    set_target_properties(histo256-fulldata_sp_const PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(
      histo256-fulldata_sp_const PRIVATE
      -DCONST_HESSIAN=1
      ${FULLDATA_DEFINES}
    )

    add_library(histo256-fulldata_sp OBJECT src/treelearner/kernels/histogram256.cu)
    set_target_properties(histo256-fulldata_sp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(
      histo256-fulldata_sp PRIVATE
      -DCONST_HESSIAN=0
      ${FULLDATA_DEFINES}
    )

endif(USE_CUDA)

if(USE_HDFS)
    find_package(JNI REQUIRED)
    find_path(HDFS_INCLUDE_DIR hdfs.h REQUIRED)
    find_library(HDFS_LIB NAMES hdfs REQUIRED)
    include_directories(${HDFS_INCLUDE_DIR})
    ADD_DEFINITIONS(-DUSE_HDFS)
    SET(HDFS_CXX_LIBRARIES ${HDFS_LIB} ${JAVA_JVM_LIBRARY})
endif(USE_HDFS)

if(UNIX OR MINGW OR CYGWIN)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -O3 -g -DUSE_CH -Wextra -Wall -Wno-unknown-pragmas -Wno-return-type")
    if(USE_SWIG)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing")
    endif()
endif()

if(WIN32 AND MINGW)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++")
endif()

if(MSVC)
    SET(variables
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_MINSIZEREL
        CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_MINSIZEREL
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
    )
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /O2 /Ob2 /Oi /Ot /Oy /GL /MP")
else()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")
endif(MSVC)

SET(LightGBM_HEADER_DIR ${PROJECT_SOURCE_DIR}/include)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

include_directories(${LightGBM_HEADER_DIR})

if(APPLE)
  if(APPLE_OUTPUT_DYLIB)
    SET(CMAKE_SHARED_LIBRARY_SUFFIX ".dylib")
  else()
    SET(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  endif()
endif(APPLE)

if(USE_MPI)
  include_directories(${MPI_CXX_INCLUDE_PATH})
endif(USE_MPI)

file(GLOB SOURCES
    src/application/*.cpp
    src/boosting/*.cpp
    src/io/*.cpp
    src/metric/*.cpp
    src/objective/*.cpp
    src/network/*.cpp
    src/treelearner/*.cpp
    src/treelearner/*.cu
)

add_executable(lightgbm src/main.cpp ${SOURCES})
add_library(_lightgbm SHARED src/c_api.cpp src/lightgbm_R.cpp ${SOURCES})

if(MSVC)
    set_target_properties(_lightgbm PROPERTIES OUTPUT_NAME "lib_lightgbm")
endif(MSVC)

if(USE_SWIG)
  set_property(SOURCE swig/lightgbmlib.i PROPERTY CPLUSPLUS ON)
  LIST(APPEND swig_options -package com.microsoft.ml.lightgbm)
  set_property(SOURCE swig/lightgbmlib.i PROPERTY SWIG_FLAGS "${swig_options}")
  swig_add_module(_lightgbm_swig java swig/lightgbmlib.i)
  swig_link_libraries(_lightgbm_swig _lightgbm)
  # needed to ensure Linux build does not have lib prefix specified twice, e.g. liblib_lightgbm_swig
  set_target_properties(_lightgbm_swig PROPERTIES PREFIX "")
  # needed in some versions of CMake for VS and MinGW builds to ensure output dll has lib prefix
  set_target_properties(_lightgbm_swig PROPERTIES OUTPUT_NAME "lib_lightgbm_swig")
  if(WIN32)
    if(MINGW OR CYGWIN)
        add_custom_command(TARGET _lightgbm_swig POST_BUILD
            COMMAND "${Java_JAVAC_EXECUTABLE}" -d . java/*.java
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${PROJECT_SOURCE_DIR}/lib_lightgbm.dll" com/microsoft/ml/lightgbm/windows/x86_64
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${PROJECT_SOURCE_DIR}/lib_lightgbm_swig.dll" com/microsoft/ml/lightgbm/windows/x86_64
            COMMAND "${Java_JAR_EXECUTABLE}" -cf lightgbmlib.jar com)
    else()
        add_custom_command(TARGET _lightgbm_swig POST_BUILD
            COMMAND "${Java_JAVAC_EXECUTABLE}" -d . java/*.java
            COMMAND cp "${PROJECT_SOURCE_DIR}/Release/*.dll" com/microsoft/ml/lightgbm/windows/x86_64
            COMMAND "${Java_JAR_EXECUTABLE}" -cf lightgbmlib.jar com)
    endif()
  elseif(APPLE)
    add_custom_command(TARGET _lightgbm_swig POST_BUILD
            COMMAND "${Java_JAVAC_EXECUTABLE}" -d . java/*.java
            COMMAND cp "${PROJECT_SOURCE_DIR}/*.dylib" com/microsoft/ml/lightgbm/osx/x86_64
            COMMAND cp "${PROJECT_SOURCE_DIR}/lib_lightgbm_swig.jnilib" com/microsoft/ml/lightgbm/osx/x86_64/lib_lightgbm_swig.dylib
            COMMAND "${Java_JAR_EXECUTABLE}" -cf lightgbmlib.jar com)
  else()
    add_custom_command(TARGET _lightgbm_swig POST_BUILD
	    COMMAND "${Java_JAVAC_EXECUTABLE}" -d . java/*.java
	    COMMAND cp "${PROJECT_SOURCE_DIR}/*.so" com/microsoft/ml/lightgbm/linux/x86_64
	    COMMAND "${Java_JAR_EXECUTABLE}" -cf lightgbmlib.jar com)
  endif()
endif(USE_SWIG)

if(USE_MPI)
  TARGET_LINK_LIBRARIES(lightgbm ${MPI_CXX_LIBRARIES})
  TARGET_LINK_LIBRARIES(_lightgbm ${MPI_CXX_LIBRARIES})
endif(USE_MPI)

if(USE_OPENMP)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        TARGET_LINK_LIBRARIES(lightgbm OpenMP::OpenMP_CXX)
        TARGET_LINK_LIBRARIES(_lightgbm OpenMP::OpenMP_CXX)
    endif()
endif(USE_OPENMP)

if(USE_GPU)
  TARGET_LINK_LIBRARIES(lightgbm ${OpenCL_LIBRARY} ${Boost_LIBRARIES})
  TARGET_LINK_LIBRARIES(_lightgbm ${OpenCL_LIBRARY} ${Boost_LIBRARIES})
endif(USE_GPU)

if(USE_CUDA)
  TARGET_LINK_LIBRARIES(
    lightgbm
    histo256_sp_const
    histo256_sp
    histo256-fulldata_sp_const
    histo256-fulldata_sp
    ${Boost_LIBRARIES}
    nvToolsExt
  )
  TARGET_LINK_LIBRARIES(
    _lightgbm
    histo256_sp_const
    histo256_sp
    histo256-fulldata_sp_const
    histo256-fulldata_sp
    ${Boost_LIBRARIES}
    nvToolsExt
  )
endif(USE_CUDA)

if(USE_HDFS)
  TARGET_LINK_LIBRARIES(lightgbm ${HDFS_CXX_LIBRARIES})
  TARGET_LINK_LIBRARIES(_lightgbm ${HDFS_CXX_LIBRARIES})
endif(USE_HDFS)

if(WIN32 AND (MINGW OR CYGWIN))
    TARGET_LINK_LIBRARIES(lightgbm Ws2_32)
    TARGET_LINK_LIBRARIES(_lightgbm Ws2_32)
    TARGET_LINK_LIBRARIES(lightgbm IPHLPAPI)
    TARGET_LINK_LIBRARIES(_lightgbm IPHLPAPI)
endif()

install(TARGETS lightgbm _lightgbm
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

install(DIRECTORY ${LightGBM_HEADER_DIR}/LightGBM DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
