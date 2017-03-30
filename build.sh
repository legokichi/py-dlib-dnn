# http://myenigma.hatenablog.com/entry/2016/12/17/075812
# https://pybind11.readthedocs.io/en/master/advanced/cast/stl.html
g++ \
  -std=gnu++11 -Wall -v \
  -O3 \
  -DDLIB_JPEG_SUPPORT -DDLIB_PNG_SUPPORT -DDLIB_NO_GUI_SUPPORT -DENABLE_ASSERTS -DNO_MAKEFILE -DDLIB_USE_BLAS -DDLIB_USE_LAPACK  \
  -I./dlib \
  -I./pybind11/include \
  -lpthread -ljpeg -lpng -llapack -lopenblas \
  `python-config --includes` \
  -fPIC \
  -o dnn.o \
  -c dnn.cpp
g++ \
  -std=gnu++11 -Wall -v \
  -O3 \
  dnn.o \
  -I./dlib \
  -lpthread -ljpeg -lpng -llapack -lopenblas \
  `python-config --includes` \
  -shared -Wl,-soname,dnn.so \
  -o dnn.so
python ./example.py
