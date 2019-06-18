apt-get update
apt-get upgrade
apt install -y unzip make cmake
apt-get install -y build-essential cmake pkg-config libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev libv4l-dev v4l-utils libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libqt4-dev mesa-utils libgl1-mesa-dri libqt4-opengl-dev libatlas-base-dev gfortran libeigen3-dev python2.7-dev python3-dev python-numpy python3-numpy
mkdir opencv && cd opencv
wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.4.2.zip && unzip opencv.zip
wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.4.2.zip && unzip opencv_contrib.zip
cd opencv-3.4.2
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_TBB=OFF \
-D WITH_IPP=OFF \
-D WITH_1394=OFF \
-D BUILD_WITH_DEBUG_INFO=OFF \
-D BUILD_DOCS=OFF \
-D INSTALL_C_EXAMPLES=ON \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D BUILD_EXAMPLES=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D WITH_QT=ON \
-D WITH_OPENGL=ON \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.2/modules \
-D WITH_V4L=ON  \
-D WITH_FFMPEG=ON \
-D WITH_XINE=ON \
-D BUILD_NEW_PYTHON_SUPPORT=ON \
../
make -j4
#make -j $(nproc)
make install

# For virtualenv
# sudo make install
# sudo ldconfig
# cd /usr/local/lib/python3.6/site-packages/
# sudo mv cv2.cpython-36m-x86_64-linux-gnu.so cv2.so
# cd ~/.virtualenvs/[virtualenv_name]/lib/python3.6/site-packages/
# ln -s /usr/local/lib/python3.6/site-packages/cv2.so cv2.so
