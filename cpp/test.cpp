#include <bits/stdc++.h>
#include <x264.h>
#include <libyuv.h>
#include <x264Codec.hpp>
#include <x265Codec.hpp>
#include <x266Codec.hpp>

using namespace std;
using namespace libyuv;

void test_264(){
    int h = 144;
    int w = 176;
    int rgb_sz = h * w * 3;
    x264Codec codec = x264Codec(h, w);
    int i_frame = 0;
    uint8_t* in_buffer = new uint8_t[codec._psize]; 
    uint8_t* out_buffer = new uint8_t[codec._psize]; 

    FILE *f_in = fopen("../akiyo_qcif.yuv", "r");
    int sz = fread(in_buffer, codec._psize, 1, f_in);
    fclose(f_in);

    int fsz = codec.Encode(in_buffer, out_buffer, 51);
    printf("frame size %d, encode done\n", fsz);
    FILE *f_out = fopen("../akiyo_qcif.decode_264.yuv", "w");
    fwrite(out_buffer, codec._psize, 1, f_out);
    fclose(f_out);

    delete in_buffer;
    delete out_buffer;
}

void test_265(){
    int h = 144;
    int w = 176;
    int rgb_sz = h * w * 3;
    x265Codec codec = x265Codec(h, w);
    int i_frame = 0;
    uint8_t* in_buffer = new uint8_t[codec._psize]; 
    uint8_t* out_buffer = new uint8_t[codec._psize]; 

    FILE *f_in = fopen("../akiyo_qcif.yuv", "r");
    int sz = fread(in_buffer, codec._psize, 1, f_in);
    fclose(f_in);

    int fsz = codec.Encode(in_buffer, out_buffer, 51);
    printf("frame size %d, encode done\n", fsz);
    FILE *f_out = fopen("../akiyo_qcif.decode_265.yuv", "w");
    fwrite(out_buffer, codec._psize, 1, f_out);
    fclose(f_out);

    delete in_buffer;
    delete out_buffer;
}

void test_266(){
    int h = 144;
    int w = 176;
    int rgb_sz = h * w * 3;
    x266Codec codec = x266Codec(h, w);
    int i_frame = 0;
    uint8_t* in_buffer = new uint8_t[codec.getPSize() * 3]; 
    uint8_t* out_buffer = new uint8_t[codec.getPSize() * 3]; 

    FILE *f_in = fopen("../akiyo_qcif.yuv", "r");
    int sz = fread(in_buffer, codec.getPSize(), 1, f_in);
    fclose(f_in);

    int fsz = codec.Encode(in_buffer, out_buffer, 51);
    printf("frame size %d, encode done\n", fsz);
    FILE *f_out = fopen("../akiyo_qcif.decode_265.yuv", "w");
    fwrite(out_buffer, codec.getPSize(), 1, f_out);
    fclose(f_out);

    delete in_buffer;
    delete out_buffer;
}

int main(){
    test_266();
}