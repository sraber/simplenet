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

void resize_image(const Matrix& input_image, const float scale_factor, Matrix& output_image)
{
   // Calculate the new image dimensions based on the scale factor
   const int new_width = static_cast<int>(scale_factor * input_image.cols());
   const int new_height = static_cast<int>(scale_factor * input_image.rows());

   // Create a new output image matrix with the desired dimensions
   output_image.resize(new_height, new_width);

   // Use bilinear interpolation to scale the image
   for (int y = 0; y < new_height; ++y)
   {
      for (int x = 0; x < new_width; ++x)
      {
         // Calculate the corresponding pixel position in the input image
         const double input_x = static_cast<float>(x) / scale_factor;
         const double input_y = static_cast<float>(y) / scale_factor;

         // Compute the pixel values of the four nearest neighbors
         const int x0 = static_cast<int>(input_x);
         const int y0 = static_cast<int>(input_y);
         const int x1 = std::min(x0 + 1, static_cast<int>(input_image.cols() - 1));
         const int y1 = std::min(y0 + 1, static_cast<int>(input_image.rows() - 1));

         const double p00 = input_image(y0, x0);
         const double p01 = input_image(y0, x1);
         const double p10 = input_image(y1, x0);
         const double p11 = input_image(y1, x1);

         // Compute the bilinear interpolation weights
         const double weight_x1 = input_x - static_cast<float>(x0);
         const double weight_x0 = 1.0f - weight_x1;
         const double weight_y1 = input_y - static_cast<float>(y0);
         const double weight_y0 = 1.0f - weight_y1;

         // Compute the interpolated pixel value
         const double interpolated_value = weight_x0 * (weight_y0 * p00 + weight_y1 * p10) + weight_x1 * (weight_y0 * p01 + weight_y1 * p11);

         // Set the output pixel value
         output_image(y, x) = interpolated_value;
      }
   }
}

