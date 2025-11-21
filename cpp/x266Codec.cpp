#include "x266Codec.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <libyuv.h>
#include <vector>

using namespace libyuv;

//////////////////////////////////////////////////////////////////////////
/// x266Codec
//////////////////////////////////////////////////////////////////////////

x266Codec::x266Codec(int height, int width)
{
    _width  = width;
    _height = height;
    _cwidth  = _width  / 2;
    _cheight = _height / 2;
    _lsize   = _width  * _height;
    _csize   = _cwidth * _cheight;
    _psize   = _lsize + 2 * _csize;

    vvenc_config_default(&_params);

    vvenc_init_default(&_params, _width, _height, 30,
                       VVENC_RC_OFF, VVENC_AUTO_QP, vvencPresetMode::VVENC_FASTER);

    _params.m_internChromaFormat  = VVENC_CHROMA_420;
    _params.m_inputBitDepth[0]    = 8;
    _params.m_internalBitDepth[0] = 10;
    _params.m_QP                  = 32;
    _params.m_FrameRate           = 30;
    _params.m_FrameScale          = 1;

    _encoder = vvenc_encoder_create();
    if (!_encoder) {
        std::cerr << "[ERROR] vvenc_encoder_create() failed.\n";
        assert(0);
    }

    int ret = vvenc_encoder_open(_encoder, &_params);
    if (ret != 0) {
        std::cerr << "[ERROR] vvenc_encoder_open() failed, ret=" << ret << std::endl;
        assert(0);
    }

    vvenc_encoder_set_RecYUVBufferCallback(_encoder, this, &x266Codec::RecYuvCallback);

    _in_yuvbuf = vvenc_YUVBuffer_alloc();
    _au = vvenc_accessUnit_alloc();
    vvenc_accessUnit_alloc_payload(_au, 3 * _width * _height + 1024);

    _internalYUVBuffer = new int16_t[_psize];

    _curOutBuf = nullptr;
}


x266Codec::~x266Codec()
{
    if (_encoder) {
        vvenc_encoder_close(_encoder);
        _encoder = nullptr;
    }

    if (_in_yuvbuf) {
        _in_yuvbuf->planes[0].ptr = nullptr;
        _in_yuvbuf->planes[1].ptr = nullptr;
        _in_yuvbuf->planes[2].ptr = nullptr;
        vvenc_YUVBuffer_free(_in_yuvbuf, false);
        _in_yuvbuf = nullptr;
    }

    if (_au) {
        vvenc_accessUnit_free(_au, true);
        _au = nullptr;
    }

    if (_internalYUVBuffer) {
        delete[] _internalYUVBuffer;
        _internalYUVBuffer = nullptr;
    }
}



void x266Codec::Buf2pic(uint8_t* in_buffer)
{
    for (int i = 0; i < _psize; ++i) {
        _internalYUVBuffer[i] = static_cast<int16_t>(in_buffer[i]) << 2;
    }

    _in_yuvbuf->planes[0].ptr    = _internalYUVBuffer;
    _in_yuvbuf->planes[1].ptr    = _internalYUVBuffer + _lsize;
    _in_yuvbuf->planes[2].ptr    = _internalYUVBuffer + _lsize + _csize;

    _in_yuvbuf->planes[0].width  = _width;
    _in_yuvbuf->planes[0].height = _height;
    _in_yuvbuf->planes[0].stride = _width;

    _in_yuvbuf->planes[1].width  = _cwidth;
    _in_yuvbuf->planes[1].height = _cheight;
    _in_yuvbuf->planes[1].stride = _cwidth;

    _in_yuvbuf->planes[2].width  = _cwidth;
    _in_yuvbuf->planes[2].height = _cheight;
    _in_yuvbuf->planes[2].stride = _cwidth;

    _in_yuvbuf->cts      = 0;
    _in_yuvbuf->ctsValid = true;
}


void x266Codec::Pic2buf(uint8_t* out_buffer, int usedSize)
{

}


int x266Codec::Encode(uint8_t* in_buffer, uint8_t* out_buffer, int qp)
{
    if (qp < 0 || qp > 51) {
        std::cerr << "[ERROR] QP " << qp << " invalid.\n";
        return -1;
    }
    
    if (_params.m_QP != qp) {
        _params.m_QP = qp;
        if (_encoder) {
            vvenc_encoder_close(_encoder);
        }
        _encoder = vvenc_encoder_create();
        if (!_encoder) {
            std::cerr << "[ERROR] vvenc_encoder_create() failed during reinitialization.\n";
            return -1;
        }
        int ret = vvenc_encoder_open(_encoder, &_params);
        if (ret != 0) {
            std::cerr << "[ERROR] vvenc_encoder_open() failed during reinitialization, ret=" << ret << std::endl;
            vvenc_encoder_close(_encoder);
            _encoder = nullptr;
            return -1;
        }
        vvenc_encoder_set_RecYUVBufferCallback(_encoder, this, &x266Codec::RecYuvCallback);
    }

    _curOutBuf = out_buffer;
    
    Buf2pic(in_buffer);
    
    vvenc_accessUnit_reset(_au);
    
    bool encodeDone = false;
    int ret = vvenc_encode(_encoder, _in_yuvbuf, _au, &encodeDone);
    if (ret < 0) {
        std::cerr << "[ERROR] vvenc_encode() failed, ret=" << ret << std::endl;
        return -1;
    }
    
    Pic2buf(out_buffer, _au->payloadUsedSize);
    
    while (!encodeDone) {
        vvenc_accessUnit_reset(_au);
        ret = vvenc_encode(_encoder, nullptr, _au, &encodeDone);
        if (ret < 0) {
            std::cerr << "[ERROR] vvenc_encode() flush failed, ret=" << ret << std::endl;
            return -1;
        }
        Pic2buf(out_buffer, _au->payloadUsedSize);
    }
    
    return _au->payloadUsedSize;
}

void x266Codec::RecYuvCallback(void* pCtx, vvencYUVBuffer* reconYuv)
{
    // ctx 转 x266Codec*
    x266Codec* thiz = reinterpret_cast<x266Codec*>(pCtx);
    if (!thiz || !reconYuv) return;

    thiz->Convert10BitYUVToRGB(reconYuv);
}

void x266Codec::Convert10BitYUVToRGB(vvencYUVBuffer* reconYuv)
{    
    if (!_curOutBuf || !reconYuv) {
        return;
    }

    int realWidth  = reconYuv->planes[0].width;
    int realHeight = reconYuv->planes[0].height;

    int planeSizeY = realWidth * realHeight;
    int planeSizeC = planeSizeY / 4;

    std::vector<uint8_t> tempI420( planeSizeY + 2*planeSizeC );

    {
        auto* src16 = reinterpret_cast<int16_t*>( reconYuv->planes[0].ptr );
        for( int r = 0; r < realHeight; r++ )
        {
            for( int c = 0; c < realWidth; c++ )
            {
                tempI420[c + r*realWidth] = (src16[c] >> 2) & 0xFF;
            }
            src16 += reconYuv->planes[0].stride; // stride in samples
        }
    }

    // U plane
    {
        auto* src16 = reinterpret_cast<int16_t*>( reconYuv->planes[1].ptr );
        int halfW = realWidth >> 1;
        int halfH = realHeight >> 1;
        uint8_t* dstU = tempI420.data() + planeSizeY; 
        for( int r = 0; r < halfH; r++ )
        {
            for( int c = 0; c < halfW; c++ )
            {
                dstU[c + r*halfW] = (src16[c] >> 2) & 0xFF;
            }
            src16 += reconYuv->planes[1].stride;
        }
    }

    // V plane
    {
        auto* src16 = reinterpret_cast<int16_t*>( reconYuv->planes[2].ptr );
        int halfW = realWidth >> 1;
        int halfH = realHeight >> 1;
        uint8_t* dstV = tempI420.data() + planeSizeY + planeSizeC; 
        for( int r = 0; r < halfH; r++ )
        {
            for( int c = 0; c < halfW; c++ )
            {
                dstV[c + r*halfW] = (src16[c] >> 2) & 0xFF;
            }
            src16 += reconYuv->planes[2].stride;
        }
    }

    I420ToRGB24(
        tempI420.data(),                   realWidth,
        tempI420.data() + planeSizeY,      realWidth/2,
        tempI420.data() + planeSizeY + planeSizeC, realWidth/2,
        _curOutBuf,                        realWidth * 3,
        realWidth, realHeight
    );
}
//////////////////////////////////////////////////////////////////////////
/// RGBx266Codec
//////////////////////////////////////////////////////////////////////////

RGBx266Codec::RGBx266Codec(int height, int width)
{
    _x266codec = new x266Codec(height, width);

    _in_buffer  = new uint8_t[_x266codec->getPSize()];
    _out_buffer = new uint8_t[_x266codec->getPSize()];
}

RGBx266Codec::~RGBx266Codec()
{
    delete[] _in_buffer;
    delete[] _out_buffer;
    delete   _x266codec;
    // std::cout << "[DEBUG] 266deconstruction" << std::endl;
}

int RGBx266Codec::Encode(uint8_t* in_buffer, uint8_t* out_buffer, int qp)
{
    // 1) RGB24 -> I420
    RGB24ToI420(
        in_buffer, _x266codec->getWidth() * 3,
        _in_buffer,                          _x266codec->getWidth(),
        _in_buffer + _x266codec->getLSize(), _x266codec->getWidth() / 2,
        _in_buffer + _x266codec->getLSize()
                  + _x266codec->getCSize(), _x266codec->getWidth() / 2,
        _x266codec->getWidth(),
        _x266codec->getHeight()
    );

    // 2) encode
    int encoded_size = _x266codec->Encode(_in_buffer, out_buffer, qp);
    if (encoded_size < 0)
    {
        std::cerr << "[ERROR] x266Codec->Encode failed.\n";
        return -1;
    }
    return encoded_size;
}
