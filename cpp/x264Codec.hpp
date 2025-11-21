#include <bits/stdc++.h>
#include <x264.h>
#include <libyuv.h>

using namespace std;
using namespace libyuv;

class x264Codec {
    public:
      int _width, _height, _lsize, _csize, _cwidth, _cheight, _psize;
      x264_param_t _param;
      x264_picture_t _pic;
      x264_picture_t _pic_out;
      x264_t *_h;
      int _i_frame = 0;
      int _i_frame_size;
      x264_nal_t *_nal;
      int _i_nal;
      x264Codec(int height, int width);
      ~x264Codec();
      void Buf2pic(uint8_t* in_buffer);
      void Pic2buf(uint8_t* out_buffer);
      int Encode(uint8_t* in_buffer, uint8_t* out_buffer, int qp);
  };
  
class RGBx264Codec{
  public:
    x264Codec* _x264codec;
    uint8_t* _in_buffer; 
    uint8_t* _out_buffer; 
    RGBx264Codec(int height, int width);
    ~RGBx264Codec();
    int Encode(uint8_t* in_buffer, uint8_t* out_buffer, int qp);
};
    