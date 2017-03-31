#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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
    DNN(const string mmod_file_path) {
       deserialize(mmod_file_path) >> net;  
    }
    net_type net;
    std::vector<pair<pair<long, long>, std::vector<tuple<long, long, long, long>>>> detect(std::vector<string> img_file_paths, uint32_t upsampling=1000){

        std::vector<matrix<rgb_pixel>> imgs{};

        for (auto&& path : img_file_paths){

            matrix<rgb_pixel> img;

            load_image(img, path);

            while(img.size() < upsampling*upsampling) pyramid_up(img);

            imgs.push_back(img);
        }
        
        std::vector<std::vector<mmod_rect>> detss = net(imgs);

        std::vector<pair<pair<long, long>, std::vector<tuple<long, long, long, long>>>> rets;
        int i = 0;
        for (auto&& dets : detss){

            pair<long, long> size(imgs[i].nc(), imgs[i].nr());
            i++;

            std::vector<tuple<long, long, long, long>> rects;

            for (auto&& d : dets){
                auto rect = d.rect;
                tuple<long, long, long, long> _rect{
                    rect.left(),
                    rect.top(),
                    rect.right(),
                    rect.bottom()
                };
                rects.push_back(_rect);
            }

            pair<pair<long, long>, std::vector<tuple<long, long, long, long>>> ret(size, rects);
            rets.push_back(ret);
        }
        return rets;
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

