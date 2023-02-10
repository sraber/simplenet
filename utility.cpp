#include <utility.h>

void MakeMatrixRGBImage(string file, Matrix rm, Matrix gm, Matrix bm)
{
   pixel_data pixel;
   int rows = (int)rm.rows();
   int cols = (int)rm.cols();

   runtime_assert( gm.rows()==rows && bm.rows()==rows)
   runtime_assert( gm.cols()==cols && bm.cols()==cols)


   unsigned char* pbytes = new unsigned char[rows * cols * sizeof(pixel_data)]; // 24 bit BMP
   unsigned char* pbs = pbytes;
   for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
         pixel.r = static_cast<unsigned char>(rm(r, c) * 254);
         pixel.g = static_cast<unsigned char>(gm(r, c) * 254);
         pixel.b = static_cast<unsigned char>(bm(r, c) * 254);

         std::memcpy(pbs, &pixel, sizeof(pixel_data));
         pbs += sizeof(pixel_data);
      }
   }

   generateBitmapImage(pbytes, rows, cols, cols * sizeof(pixel_data), file);
}

void MakeMatrixImage(string file, Matrix m)
{
   pixel_data pixel;
   int rows = (int)m.rows();
   int cols = (int)m.cols();

   unsigned char* pbytes = new unsigned char[rows * cols * sizeof(pixel_data)]; // 24 bit BMP
   unsigned char* pbs = pbytes;
   for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
         double v = m(r, c) * 254;
         pixel.r = static_cast<unsigned char>(v);
         pixel.g = static_cast<unsigned char>(v);
         pixel.b = static_cast<unsigned char>(v);

         std::memcpy(pbs, &pixel, sizeof(pixel_data));
         pbs += sizeof(pixel_data);
      }
   }

   generateBitmapImage(pbytes, rows, cols, cols * sizeof(pixel_data), file);
}

void ScaleToOne(double* pdata, int size)
{
   double max = 0.0;
   double min = 0.0;
   double* pd = pdata;
   double* pde = pd + size;
   for (; pd < pde; pd++) {
      if (max < *pd) { max = *pd; }
      if (min > * pd) { min = *pd; }
   }
   for (pd = pdata; pd < pde; pd++) {
      *pd = (*pd - min) / (max - min);
   }
}