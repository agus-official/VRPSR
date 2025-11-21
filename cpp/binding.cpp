#include <bits/stdc++.h>
#include <pybind11/pybind11.h>
#include <x264Codec.hpp>
#include <x265Codec.hpp>
#include <x266Codec.hpp>

using namespace std;
using namespace pybind11;

class PYRGBx264Codec: public RGBx264Codec{
  public:
    PYRGBx264Codec(int height, int width) : RGBx264Codec(height, width){}
    ~PYRGBx264Codec(){}
    int Encode(buffer &in_buffer, buffer &out_buffer, int qp){
        buffer_info in_info = in_buffer.request();
        buffer_info out_info = out_buffer.request();
        return RGBx264Codec::Encode(
            reinterpret_cast<uint8_t*> (in_info.ptr),
            reinterpret_cast<uint8_t*> (out_info.ptr),
            qp
        );
    }
};

class PYRGBx265Codec: public RGBx265Codec{
    public:
    PYRGBx265Codec(int height, int width) : RGBx265Codec(height, width){}
    ~PYRGBx265Codec(){}
    int Encode(buffer &in_buffer, buffer &out_buffer, int qp){
        buffer_info in_info = in_buffer.request();
        buffer_info out_info = out_buffer.request();
        return RGBx265Codec::Encode(
            reinterpret_cast<uint8_t*> (in_info.ptr),
            reinterpret_cast<uint8_t*> (out_info.ptr),
            qp
        );
    }
};

class PYRGBx266Codec: public RGBx266Codec{
    public:
    PYRGBx266Codec(int height, int width) : RGBx266Codec(height, width){}
    ~PYRGBx266Codec(){}
    int Encode(buffer &in_buffer, buffer &out_buffer, int qp){
        buffer_info in_info = in_buffer.request();
        buffer_info out_info = out_buffer.request();
        return RGBx266Codec::Encode(
            reinterpret_cast<uint8_t*> (in_info.ptr),
            reinterpret_cast<uint8_t*> (out_info.ptr),
            qp
        );
    }
};
  

PYBIND11_MODULE(codecsimulator, m) {
    m.doc() = "codecsimulator python library";
    class_<PYRGBx264Codec>(m, "PYRGBx264Codec")
        .def(init<int, int>())
        .def("Encode", &PYRGBx264Codec::Encode);
    class_<PYRGBx265Codec>(m, "PYRGBx265Codec")
        .def(init<int, int>())
        .def("Encode", &PYRGBx265Codec::Encode);
    class_<PYRGBx266Codec>(m, "PYRGBx266Codec")
        .def(init<int, int>())
        .def("Encode", &PYRGBx266Codec::Encode);
}