#include <x265Codec.hpp>

x265Codec::x265Codec(int height, int width){
    _width = width;
    _height = height;
    _cwidth = _width / 2;
    _cheight = _height / 2;
    _lsize = _width * _height;
    _csize = _cwidth * _cheight;
    _psize = _lsize + 2 * _csize;

	_pParam=x265_param_alloc();
    x265_param_default_preset(_pParam, "placebo", "psnr");
    _pParam->logLevel=X265_LOG_NONE;
	_pParam->bRepeatHeaders=1;
	_pParam->internalCsp=X265_CSP_I420;
	_pParam->sourceWidth=_width;
	_pParam->sourceHeight=_height;
	_pParam->fpsNum=25;
	_pParam->fpsDenom=1;

    _pHandle=x265_encoder_open(_pParam);

	if(_pHandle==NULL)
        assert(0);

	_pPic_in = x265_picture_alloc();
	x265_picture_init(_pParam, _pPic_in);
	_pPic_out = x265_picture_alloc();
	x265_picture_init(_pParam, _pPic_out);
}

x265Codec::~x265Codec(){
	x265_encoder_close(_pHandle);
	x265_picture_free(_pPic_in);
	x265_param_free(_pParam);
}

void x265Codec::Buf2pic(uint8_t* in_buffer){
    _pPic_in->width = _width;
    _pPic_in->height = _height;
    _pPic_in->planes[0] = in_buffer;
    _pPic_in->planes[1] = in_buffer + _lsize;
    _pPic_in->planes[2] = in_buffer + _lsize + _csize;
    _pPic_in->stride[0] = _width;
    _pPic_in->stride[1] = _cwidth;
    _pPic_in->stride[2] = _cwidth;
}

void x265Codec::Pic2buf(uint8_t* out_buffer){
    I420Copy(
        (const uint8_t*)_pPic_out->planes[0], _pPic_out->stride[0],
        (const uint8_t*)_pPic_out->planes[1], _pPic_out->stride[1],
        (const uint8_t*)_pPic_out->planes[2], _pPic_out->stride[2],
        out_buffer, _width,
        out_buffer + _lsize, _cwidth,
        out_buffer + _lsize + _csize, _cwidth,
        _width,
        _height
    );
}


int x265Codec::Encode(uint8_t* in_buffer, uint8_t* out_buffer, int qp){
    Buf2pic(in_buffer);
    int i_size = 0;
    _pPic_in->forceqp = qp;
    _pPic_in->sliceType = X265_TYPE_IDR;
    int ret=x265_encoder_encode(_pHandle,&_pNals,&_iNal,_pPic_in,_pPic_out);	

    for(int j=0;j<_iNal;j++){
        i_size += _pNals[j].sizeBytes;
    }
    if (ret)
        Pic2buf(out_buffer);
    
	while(1){
		ret=x265_encoder_encode(_pHandle,&_pNals,&_iNal,NULL,_pPic_out);
		if(ret==0){
			break;
		}
		for(int j=0;j<_iNal;j++){
            i_size += _pNals[j].sizeBytes;
		}
        if (ret)
            Pic2buf(out_buffer);

	}
    return i_size;
}


RGBx265Codec::RGBx265Codec(int height, int width){
    _x265codec = new x265Codec(height, width);
    _in_buffer = new uint8_t[_x265codec->_psize];
    _out_buffer = new uint8_t[_x265codec->_psize];
}
RGBx265Codec::~RGBx265Codec(){
    delete _x265codec;
    delete _in_buffer;
    delete _out_buffer;
}
int RGBx265Codec::Encode(uint8_t* in_buffer, uint8_t* out_buffer, int qp){
    RGB24ToI420(
        in_buffer, _x265codec->_width * 3, 
        _in_buffer, _x265codec->_width,
        _in_buffer + _x265codec->_lsize, _x265codec->_cwidth,
        _in_buffer + _x265codec->_lsize + _x265codec->_csize, _x265codec->_cwidth,
        _x265codec->_width, _x265codec->_height
    );
    int size = _x265codec->Encode(_in_buffer, _out_buffer, qp);
    I420ToRGB24(
        _out_buffer, _x265codec->_width,
        _out_buffer + _x265codec->_lsize, _x265codec->_cwidth,
        _out_buffer + _x265codec->_lsize + _x265codec->_csize, _x265codec->_cwidth,
        out_buffer, _x265codec->_width * 3,
        _x265codec->_width, _x265codec->_height
    );
    return size;
}
