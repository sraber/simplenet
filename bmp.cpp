#include <cstdio>
#include <string>
#include <cassert>

// REVIEW: For a 3 class problem each of output probabilities can be scaled to 255 (R, G, B) to
//         give a real picture ot the class boundaries.

const int bytesPerPixel = 3; /// red, green, blue
const int fileHeaderSize = 14;
const int infoHeaderSize = 40;

unsigned char* createBitmapFileHeader(int height, int width, int pitch, int paddingSize);
unsigned char* createBitmapInfoHeader(int height, int width);

void generateBitmapImage(unsigned char* image, int height, int width, int pitch, std::string imageFileName) {

   unsigned char padding[3] = { 0, 0, 0 };
   int paddingSize = (4 - (/*width*bytesPerPixel*/ pitch) % 4) % 4;

   unsigned char* fileHeader = createBitmapFileHeader(height, width, pitch, paddingSize);
   unsigned char* infoHeader = createBitmapInfoHeader(height, width);

   FILE* imageFile;
   errno_t res = fopen_s(&imageFile, imageFileName.c_str(), "wb");
   assert( ("Could'nt open file." , res) == 0 );

   fwrite(fileHeader, 1, fileHeaderSize, imageFile);
   fwrite(infoHeader, 1, infoHeaderSize, imageFile);

   int i;
   for (i = 0; i < height; i++) {
      fwrite(image + (i * pitch /*width*bytesPerPixel*/), bytesPerPixel, width, imageFile);
      fwrite(padding, 1, paddingSize, imageFile);
   }

   fclose(imageFile);
   //free(fileHeader);
   //free(infoHeader);
}

unsigned char* createBitmapFileHeader(int height, int width, int pitch, int paddingSize) {
   int fileSize = fileHeaderSize + infoHeaderSize + (/*bytesPerPixel*width*/pitch + paddingSize) * height;

   static unsigned char fileHeader[] = {
       0,0, /// signature
       0,0,0,0, /// image file size in bytes
       0,0,0,0, /// reserved
       0,0,0,0, /// start of pixel array
   };

   fileHeader[0] = (unsigned char)('B');
   fileHeader[1] = (unsigned char)('M');
   fileHeader[2] = (unsigned char)(fileSize);
   fileHeader[3] = (unsigned char)(fileSize >> 8);
   fileHeader[4] = (unsigned char)(fileSize >> 16);
   fileHeader[5] = (unsigned char)(fileSize >> 24);
   fileHeader[10] = (unsigned char)(fileHeaderSize + infoHeaderSize);

   return fileHeader;
}

unsigned char* createBitmapInfoHeader(int height, int width) {
   static unsigned char infoHeader[] = {
       0,0,0,0, /// header size
       0,0,0,0, /// image width
       0,0,0,0, /// image height
       0,0, /// number of color planes
       0,0, /// bits per pixel
       0,0,0,0, /// compression
       0,0,0,0, /// image size
       0,0,0,0, /// horizontal resolution
       0,0,0,0, /// vertical resolution
       0,0,0,0, /// colors in color table
       0,0,0,0, /// important color count
   };

   infoHeader[0] = (unsigned char)(infoHeaderSize);
   infoHeader[4] = (unsigned char)(width);
   infoHeader[5] = (unsigned char)(width >> 8);
   infoHeader[6] = (unsigned char)(width >> 16);
   infoHeader[7] = (unsigned char)(width >> 24);
   infoHeader[8] = (unsigned char)(height);
   infoHeader[9] = (unsigned char)(height >> 8);
   infoHeader[10] = (unsigned char)(height >> 16);
   infoHeader[11] = (unsigned char)(height >> 24);
   infoHeader[12] = (unsigned char)(1);
   infoHeader[14] = (unsigned char)(bytesPerPixel * 8);

   return infoHeader;
}