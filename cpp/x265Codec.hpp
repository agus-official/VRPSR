#include <bits/stdc++.h>
#include <x265.h>
#include <libyuv.h>

using namespace std;
using namespace libyuv;

class x265Codec {
    public:
      int _width, _height, _lsize, _csize, _cwidth, _cheight, _psize;
      x265_param* _pParam;
      x265_encoder* _pHandle;
      x265_picture* _pPic_in;
      x265_picture* _pPic_out;
      x265_nal* _pNals;
      uint32_t _iNal;
    
      int _i_nal;
      x265Codec(int height, int width);
      ~x265Codec();
      void Buf2pic(uint8_t* in_buffer);
      void Pic2buf(uint8_t* out_buffer);
      int Encode(uint8_t* in_buffer, uint8_t* out_buffer, int qp);
};
  

class RGBx265Codec{
  public:
    x265Codec* _x265codec;
    uint8_t* _in_buffer; 
    uint8_t* _out_buffer; 
    RGBx265Codec(int height, int width);
    ~RGBx265Codec();
    int Encode(uint8_t* in_buffer, uint8_t* out_buffer, int qp);
};
