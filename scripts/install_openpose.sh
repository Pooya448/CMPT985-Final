module load cuda/10.2

mkdir -p ../../OpenPose
cd ../../OpenPose
# install new CMake becaue of CUDA10
wget -q https://cmake.org/files/v3.13/cmake-3.13.0-Linux-x86_64.tar.gz
tar xfz cmake-3.13.0-Linux-x86_64.tar.gz --strip-components=1 -C /u/gtiwari/CMake
# clone openpose
git clone -q --depth 1 'https://github.com/CMU-Perceptual-Computing-Lab/openpose.git'
sed -i 's/execute_process(COMMAND git checkout master WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}\/3rdparty\/caffe)/execute_process(COMMAND git checkout f019d0dfe86f49d1140961f8c7dec22130c83154 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}\/3rdparty\/caffe)/g' openpose/CMakeLists.txt
# install system dependencies
apt-get -qq install -y libatlas-base-dev libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libgflags-dev libgoogle-glog-dev liblmdb-dev opencl-headers ocl-icd-opencl-dev libviennacl-dev
# build openpose
cd openpose && rm -rf build || true && mkdir build && cd build && cmake .. && make -j`nproc`
