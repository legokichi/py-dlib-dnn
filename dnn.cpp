#include<pybind11/pybind11.h>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>


using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;
                           
template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;

using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

// ----------------------------------------------------------------------------------------


class DNN {
public:
    DNN(const std::string mmod_file_path) {
       deserialize(mmod_file_path) >> net;  
    }
    net_type net;
    const std::string detect(const std::string img_file_path){
        try {
            matrix<rgb_pixel> img;
            load_image(img, img_file_path);

            // Upsampling the image will allow us to detect smaller faces but will cause the
            // program to use more RAM and run longer.
            while(img.size() < 1800*1800) pyramid_up(img);
            
            auto dets = net(img);

            std::stringstream ss;
            for (auto&& d : dets){
                auto rect = d.rect;
                ss
                  << "["
                  << "[" << rect.left() << "," << rect.top() << "]"
                  << ","
                  << "[" << rect.right() << "," << rect.bottom() << "]"
                  << "]"
                  << '\n';
            }
            string s = ss.str();
            return s;
        }catch(std::exception& e){
            return e.what();
        }
    }
};

namespace py = pybind11;
PYBIND11_PLUGIN(dnn)
{
    py::module m("dnn", "dlib dnn");
    py::class_<DNN>(m, "DNN")
        .def(py::init<const std::string>())
        .def("detect", &DNN::detect);

    return m.ptr();
}

