#pragma once
#include <cstdint>
#include <vvenc/vvenc.h>
#include <libyuv.h>
#include <iostream>
#include <vector>

class x266Codec {
public:
    x266Codec(int height, int width);
    ~x266Codec();

    int Encode(uint8_t* in_buffer, uint8_t* out_buffer, int qp);

    int getWidth()   const { return _width; }
    int getHeight()  const { return _height; }
    int getPSize()   const { return _psize; }
    int getLSize()   const { return _lsize; }
    int getCSize()   const { return _csize; }

private:
    vvencEncoder*      _encoder;
    vvenc_config       _params;
    vvencYUVBuffer*    _in_yuvbuf;
    vvencAccessUnit*   _au;
    int16_t*           _internalYUVBuffer;

    int _width, _height;
    int _cwidth, _cheight;
    int _lsize, _csize;
    int _psize;

    uint8_t* _curOutBuf;

private:
    static void RecYuvCallback(void* pCtx, vvencYUVBuffer* reconYuv);

    void Convert10BitYUVToRGB(vvencYUVBuffer* reconYuv);

    void Buf2pic(uint8_t* in_buffer);
    void Pic2buf(uint8_t* out_buffer, int usedSize);
};


class RGBx266Codec {
public:
    RGBx266Codec(int height, int width);
    ~RGBx266Codec();

    int Encode(uint8_t* in_buffer, uint8_t* out_buffer, int qp);

private:
    x266Codec* _x266codec;
    uint8_t*   _in_buffer;
    uint8_t*   _out_buffer;
};
