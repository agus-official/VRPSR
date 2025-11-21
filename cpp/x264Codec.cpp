#include <x264Codec.hpp>

x264Codec::x264Codec(int height, int width){
    _width = width;
    _height = height;
    _cwidth = _width / 2;
    _cheight = _height / 2;
    _lsize = _width * _height;
    _csize = _cwidth * _cheight;
    _psize = _lsize + 2 * _csize;
    if( x264_param_default_preset( &_param, "medium", NULL ) < 0 )
        assert(0);
    _param.i_csp = X264_CSP_I420;
    _param.i_width  = _width;
    _param.i_height = _height;
    _param.b_vfr_input = 0;
    _param.b_repeat_headers = 1;
    _param.b_annexb = 1;
    _param.b_full_recon = 1;
    _param.i_log_level = X264_LOG_NONE;

    if(x264_param_apply_profile( &_param, "high" ) < 0)
        assert(0);
    if( x264_picture_alloc( &_pic, _param.i_csp, _param.i_width, _param.i_height ) < 0 )
        assert(0);
    _h = x264_encoder_open( &_param );
    if( !_h )
        assert(0);
}

x264Codec::~x264Codec(){
    x264_encoder_close( _h );
    x264_picture_clean( &_pic );    
}

void x264Codec::Buf2pic(uint8_t* in_buffer){
    memcpy(_pic.img.plane[0], in_buffer, _lsize);
    memcpy(_pic.img.plane[1], in_buffer + _lsize, _csize);
    memcpy(_pic.img.plane[2], in_buffer + _lsize + _csize, _csize);
}

void x264Codec::Pic2buf(uint8_t* out_buffer){
    assert(_pic_out.img.i_csp == 4);
    NV12ToI420(_pic_out.img.plane[0],
        _pic_out.img.i_stride[0],
        _pic_out.img.plane[1],
        _pic_out.img.i_stride[1],
        out_buffer, _width,
        out_buffer + _lsize, _cwidth,
        out_buffer + _lsize + _csize, _cwidth,
        _width, _height);       
}

int x264Codec::Encode(uint8_t* in_buffer, uint8_t* out_buffer, int qp){
    Buf2pic(in_buffer);
    _pic.i_pts = _i_frame;
    _pic.i_type = X264_TYPE_IDR;
    _pic.i_qpplus1 = qp;
    int i_frame_size = x264_encoder_encode( _h, &_nal, &_i_nal, &_pic, &_pic_out );
    if( i_frame_size < 0 )
        assert(0);
    else if( i_frame_size )
    {
        Pic2buf(out_buffer);
    } else {
        while ( x264_encoder_delayed_frames( _h ) ) {
            i_frame_size = x264_encoder_encode( _h, &_nal, &_i_nal, NULL, &_pic_out );
            if (i_frame_size < 0){
                assert(0);
            } else if (i_frame_size){
                Pic2buf(out_buffer);
            }
        }
    }
    _i_frame++;
    return i_frame_size;
}

RGBx264Codec::RGBx264Codec(int height, int width){
    _x264codec = new x264Codec(height, width);
    _in_buffer = new uint8_t[_x264codec->_psize];
    _out_buffer = new uint8_t[_x264codec->_psize];
}
RGBx264Codec::~RGBx264Codec(){
    delete _x264codec;
    delete _in_buffer;
    delete _out_buffer;
}
int RGBx264Codec::Encode(uint8_t* in_buffer, uint8_t* out_buffer, int qp){
    RGB24ToI420(
        in_buffer, _x264codec->_width * 3, 
        _in_buffer, _x264codec->_width,
        _in_buffer + _x264codec->_lsize, _x264codec->_cwidth,
        _in_buffer + _x264codec->_lsize + _x264codec->_csize, _x264codec->_cwidth,
        _x264codec->_width, _x264codec->_height
    );
    int size = _x264codec->Encode(_in_buffer, _out_buffer, qp);
    I420ToRGB24(
        _out_buffer, _x264codec->_width,
        _out_buffer + _x264codec->_lsize, _x264codec->_cwidth,
        _out_buffer + _x264codec->_lsize + _x264codec->_csize, _x264codec->_cwidth,
        out_buffer, _x264codec->_width * 3,
        _x264codec->_width, _x264codec->_height
    );
    return size;
}
