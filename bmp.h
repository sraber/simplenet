#pragma once
void generateBitmapImage(unsigned char* image, int height, int width, int pitch, std::string imageFileName);

struct pixel_data {
   unsigned char r;
   unsigned char g;
   unsigned char b;
};
