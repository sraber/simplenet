// algors.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <Eigen>
#include <iostream>
#include <iomanip>
#include <MNISTReader.h>
#include <Layer.h>
#include <bmp.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <amoeba.h>

#include <time.h>

#include "fft1.h"
#include "fftn.h"

string path = "C:\\projects\\neuralnet\\simplenet\\algors\\results";

void MakeMNISTImage(string file, Matrix m)
{
   pixel_data pixel;
   int rows = m.rows();
   int cols = m.cols();

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

   generateBitmapImage(pbytes, 28, 28, 28 * sizeof(pixel_data), file);
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
   if (max == min) {
      for (pd = pdata; pd < pde; pd++) {
         *pd = 0.0;
      }
   }
   else {
      for (pd = pdata; pd < pde; pd++) {
         *pd = (*pd - min) / (max - min);
      }
   }
}

void TrasformMNISTtoMatrix(Matrix& m, ColVector& x)
{
   int i = 0;
   for (int r = 27; r >= 0; r--) {
      for (int c = 0; c < 28; c++) {
         m(r, c) = x(i);
         i++;
      }
   }
}

int GetMNISTLabel(ColVector& lv)
{
   for (int i = 0; i < 10; i++) {
      if (lv(i) > 0.5) {
         return i;
      }
   }
   assert(false);
   return 0;
}

void LinearConvolution( Matrix& m, Matrix& h, Matrix& out )
{
   const int hrows = h.rows();
   const int hcols = h.cols();
   const int mrows = m.rows();
   const int mcols = m.cols();
   const int orows = out.rows();
   const int ocols = out.cols();
   int mr2 = mrows >> 1; if (!(mrows % 2)) { mr2--; }  
   int mc2 = mcols >> 1; if (!(mcols % 2)) { mc2--; }  
   int hr2 = hrows >> 1; if (!(hrows % 2)) { hr2--; }
   int hc2 = hcols >> 1; if (!(hcols % 2)) { hc2--; }
   int or2 = orows >> 1; if (!(orows % 2)) { or2--; }
   int oc2 = ocols >> 1; if (!(ocols % 2)) { oc2--; }

   for (int r = 0; r < out.rows(); r++) {     // 0 to rows --> -On/2+1 to On/2 if On is even, else -On/2 to On/2.
      for (int c = 0; c < out.cols(); c++) {
         double sum = 0.0;
         for (int rr = 0; rr < h.rows(); rr++) {      // 0 to rows --> -N/2 to N/2 
            for (int cc = 0; cc < h.cols(); cc++) {   // 0 to cols --> -M to M --> odd numbered kernels work well becuse they are 2*M+1.
               int mr = r - rr + mr2 - or2 + hr2;                 // r and c are the correlation plane
               int mc = c - cc + mc2 - oc2 + hc2;                 // coordinates.
               if (mr >= 0 && mr < m.rows() && 
                     mc >= 0 && mc < m.cols()) {
                  sum += m(mr, mc) * h(rr, cc);
               }
            }
         }
         out(r, c) = sum;
      }
   }
}

double MultiplyReverseBlock( Matrix& m, int mr, int mc, Matrix& h, int hr, int hc, int size_r, int size_c )
{
   double sum = 0.0;
   for (int r = 0; r < size_r; r++) {
      for (int c = 0; c < size_c; c++) {
         sum += m(mr-r, mc-c) * h(hr+r, hc+c);
      }
   }
   return sum;
}

void LinearConvolution3( Matrix& m, Matrix& h, Matrix& out )
{
   const int mrows = m.rows();
   const int mcols = m.cols();
   const int hrows = h.rows();
   const int hcols = h.cols();
   const int orows = out.rows();
   const int ocols = out.cols();
   int mr2 = mrows >> 1; if (!(mrows % 2)) { mr2--; }
   int mc2 = mcols >> 1; if (!(mcols % 2)) { mc2--; }
   int hr2 = hrows >> 1; if (!(hrows % 2)) { hr2--; }
   int hc2 = hcols >> 1; if (!(hcols % 2)) { hc2--; }
   int hr2p = hrows >> 1; // The complement of hr2
   int hc2p = hcols >> 1;
   int or2 = orows >> 1; if (!(orows % 2)) { or2--; }
   int oc2 = ocols >> 1; if (!(ocols % 2)) { oc2--; }

   for (int r = 0; r < out.rows(); r++) {     // Scan through the Correlation surface.
      for (int c = 0; c < out.cols(); c++) {
         int h1r, h1c;
         int m2r, m2c;
         int m1r, m1c;
         int cr, cc;
         cr = r + mr2 - or2;
         cc = c + mc2 - oc2;
         m2r = cr + hr2;  // Use h2 to the positive side because it is the negitive side
         m2c = cc + hc2;  // relitive to the way convolution is performed.
         m1r = cr - hr2p; // Similarly the negitive side physically is the positive
         m1c = cc - hc2p; // side relitive to the convolution algorithm.
         h1r = 0;
         h1c = 0;

         int shr = hrows;
         if (m2r >= mrows) {
            int d = m2r - mrows + 1;
            m2r = mrows - 1;
            h1r += d;
            shr -= d;
         }
         if (m1r < 0) {
            shr += m1r;
            m1r = 0;
         }

         int shc = hcols;
         if (m2c >= mcols) {
            int d = m2c - mcols + 1;
            m2c = mcols - 1;
            h1c += d;
            shc -= d;
         }
         if (m1c < 0) {
            shc += m1c;
            m1c = 0;
         }

         if (shr <= 0 || shc <= 0) {
            out(r, c) = 0.0;
         }
         else {
            //cout << m1r << "," << m1c << "," << h1r << "," << h1c << "," << shr << "," << shc << "," << endl;
            out(r, c) = MultiplyReverseBlock(m, m2r, m2c, h, h1r, h1c, shr, shc);
         }
      }
   }
}

void LinearCorrelate0( Matrix& g, Matrix& h, Matrix& out )
{
   for (int r = 0; r < out.rows(); r++) {     // 0 to rows --> -On to On
      for (int c = 0; c < out.cols(); c++) {
         double sum = 0.0;
         for (int rr = 0; rr < h.rows(); rr++) {      // 0 to rows --> -N to N 
            for (int cc = 0; cc < h.cols(); cc++) {   // 0 to cols --> -M to M --> odd numbered kernels work well becuse they are 2*M+1.
               int gr = r + rr;                       // r and c are the correlation plane
               int gc = c + cc;                       // coordinates.
               if (gr >= 0 && gr < g.rows() && 
                     gc >= 0 && gc < g.cols()) {
                  sum += g(gr, gc) * h(rr, cc);
               }
            }
         }
         out(r, c) = sum;
      }
   }
}


void LinearCorrelate( Matrix& m, Matrix& h, Matrix& out )
{
   const int mrows = m.rows();
   const int mcols = m.cols();
   const int hrows = h.rows();
   const int hcols = h.cols();
   const int orows = out.rows();
   const int ocols = out.cols();
   int mr2 = mrows >> 1; if (!(mrows % 2)) { mr2--; }
   int mc2 = mcols >> 1; if (!(mcols % 2)) { mc2--; }
   int hr2 = hrows >> 1; if (!(hrows % 2)) { hr2--; }
   int hc2 = hcols >> 1; if (!(hcols % 2)) { hc2--; }
   int or2 = orows >> 1; if (!(orows % 2)) { or2--; }
   int oc2 = ocols >> 1; if (!(ocols % 2)) { oc2--; }

   for (int r = 0; r < out.rows(); r++) {     // 0 to rows --> -On to On
      for (int c = 0; c < out.cols(); c++) {
         double sum = 0.0;
         for (int rr = 0; rr < h.rows(); rr++) {      // 0 to rows --> -N to N 
            for (int cc = 0; cc < h.cols(); cc++) {   // 0 to cols --> -M to M --> odd numbered kernels work well becuse they are 2*M+1.
               int mr = r + rr + mr2 - or2 - hr2;                 // r and c are the correlation plane
               int mc = c + cc + mc2 - oc2 - hc2;                 // coordinates.
               if (mr >= 0 && mr < m.rows() && 
                     mc >= 0 && mc < m.cols()) {
                  sum += m(mr, mc) * h(rr, cc);
               }
            }
         }
         out(r, c) = sum;
      }
   }
}

double MultiplyBlock( Matrix& m, int mr, int mc, Matrix& h, int hr, int hc, int size_r, int size_c )
{
   double sum = 0.0;
   for (int r = 0; r < size_r; r++) {
      for (int c = 0; c < size_c; c++) {
         sum += m(mr+r, mc+c) * h(hr+r, hc+c);
      }
   }
   return sum;
}

double MultiplyBlock1( Matrix& m, int mr, int mc, Matrix& h, int hr, int hc, int size_r, int size_c )
{
   double sum = (m.array().block(mr, mc, size_r, size_c) * h.array().block(hr, hc, size_r, size_c)).sum();

   return sum;
}

void LinearCorrelate3( Matrix& m, Matrix& h, Matrix& out )
{
   const int mrows = m.rows();
   const int mcols = m.cols();
   const int hrows = h.rows();
   const int hcols = h.cols();
   const int orows = out.rows();
   const int ocols = out.cols();
   int mr2 = mrows >> 1; if (!(mrows % 2)) { mr2--; } 
   int mc2 = mcols >> 1; if (!(mcols % 2)) { mc2--; } 
   int hr2 = hrows >> 1; if (!(hrows % 2)) { hr2--; } 
   int hc2 = hcols >> 1; if (!(hcols % 2)) { hc2--; } 
   int or2 = orows >> 1; if (!(orows % 2)) { or2--; } 
   int oc2 = ocols >> 1; if (!(ocols % 2)) { oc2--; } 

   for (int r = 0; r < orows; r++) {     // Scan through the Correlation surface.
      for (int c = 0; c < ocols; c++) {
         int h1r, h1c;
         int m1r, m1c;
         m1r = r + mr2 - or2 - hr2;
         m1c = c + mc2 - oc2 - hc2;

         int shr = hrows;
         if (m1r < 0) {
            shr += m1r;
            m1r = 0;
            h1r = hrows - shr;
         }
         else {
            h1r = 0;
            shr = hrows;
         }
         if (m1r + shr > mrows) {
            shr = mrows - m1r;
         }

         int shc = hcols;
         if (m1c < 0) {
            shc += m1c;
            m1c = 0;
            h1c = hcols - shc;
         }
         else {
            h1c = 0;
            shc = hcols;
         }
         if (m1c + shc > mcols) {
            shc = mcols - m1c;
         }

         if (shr <= 0 || shc <= 0) {
            out(r, c) = 0.0;
         }
         else {
            //cout << m1r << "," << m1c << "," << h1r << "," << h1c << "," << shr << "," << shc << "," << endl;
            out(r, c) = MultiplyBlock(m, m1r, m1c, h, h1r, h1c, shr, shc);
         }
      }
   }
}

// Use Eigen blocks to do the correlation.
/*
void LinearCorrelate1( int pr, int pc, Matrix& m, Matrix& h, Matrix& out )
{
   const int hrows = h.rows();
   const int hcols = h.cols();
   const int mrows = m.rows();
   const int mcols = m.cols();
   for (int r = 0; r < out.rows(); r++) {     // Scan through the Correlation surface.
      for (int c = 0; c < out.cols(); c++) {
         int h1r, h1c;
         int m1r, m1c;

         int shr = hrows;
         if (r < pr) {
            h1r = pr - r;
            m1r = 0;
            shr = hrows - h1r;
         }
         else {
            h1r = 0;
            m1r = r - pr;
            if (m1r + shr > mrows) {
               shr = mrows - m1r;
            }
         }

         int shc = hcols;
         if (c < pc) {
            h1c = pc - c;
            m1c = 0;
            shc = hcols - h1c;
         }
         else {
            h1c = 0;
            m1c = c - pc;
            if (m1c + shc > mcols) {
               shc = mcols - m1c;
            }
         }
         //cout << m1r << "," << m1c << "," << h1r << "," << h1c << "," << shr << "," << shc << "," << endl;
         out(r, c) = MultiplyBlock(m,m1r,m1c,h,h1r,h1c,shr,shc);
      }
   }
}

void LinearCorrelate2( int pr, int pc, Matrix& m, Matrix& h, Matrix& out )
{
   const int hrows = h.rows();
   const int hcols = h.cols();
   const int mrows = m.rows();
   const int mcols = m.cols();
   for (int r = 0; r < out.rows(); r++) {     // Scan through the Correlation surface.
      for (int c = 0; c < out.cols(); c++) {
         int h1r, h1c;
         int m1r, m1c;

         int shr = hrows;
         if (r < pr) {
            h1r = pr - r;
            m1r = 0;
            shr = hrows - h1r;
         }
         else {
            h1r = 0;
            m1r = r - pr;
            if (m1r + shr > mrows) {
               shr = mrows - m1r;
            }
         }

         int shc = hcols;
         if (c < pc) {
            h1c = pc - c;
            m1c = 0;
            shc = hcols - h1c;
         }
         else {
            h1c = 0;
            m1c = c - pc;
            if (m1c + shc > mcols) {
               shc = mcols - m1c;
            }
         }
         //cout << m1r << "," << m1c << "," << h1r << "," << h1c << "," << shr << "," << shc << "," << endl;
         out(r, c) = MultiplyBlock1(m,m1r,m1c,h,h1r,h1c,shr,shc);
      }
   }
}
*/

void Rotate180(Matrix& k)
{
   assert(k.rows() == k.cols());  // No reason for this.
                                    // The algor could handle rows != cols.
   int kn = k.rows();
   // rotate k by 180 degrees ------------
   int kn2 = kn / 2;
   for (int i = 0; i < kn2; i++) {
      int j = kn - i - 1;
      for (int c1 = 0; c1 < kn; c1++) {
         int c2 = kn - c1 - 1;
         double temp = k(i, c1);
         k(i, c1) = k(j, c2);
         k(j, c2) = temp;
      }
   }
   if (kn % 2) {
      int j = kn / 2;  // Don't add 1.  The zero offset compensates.
      for (int c1 = 0; c1 < kn2; c1++) {
         int c2 = kn - c1 - 1;
         double temp = k(j, c1);
         k(j, c1) = k(j, c2);
         k(j, c2) = temp;
      }
   }
   //------------------------------------------
}

void TestKernelFlipper()
{
   // Input matrix size.
   const int n = 28;
   // Kernel matrix size.
   const int kn = 28;
   // Padding of input matrix.
   //const int p = (int)floor((double)kn/2.0);
   const int p = 27;
   // Convolution / back propagation delta size.
   // cn max = n + p - kn
   const int cn = 19; // 29;
   // Input delta padding is a function of the kernel size and input padding.
   int dp = kn - p - 1;
   // Input matrix padding.
   int nn = n + 2 * p;
   // Delta matrix size with padding.
   int dd = cn + 2 * dp;

   // Make a kernel matrix.  Unroll it into another matrix.
   // Make a grad matrix.  Flatten that matrix and multiple by the unrolled matrix.
   // Correlate the matrix with the 180 flip kernel.  The two methods should agree.

   Matrix k(kn, kn);    // Kernel
   // The incomming delta gradient.  This is the gradient that is formed by multiplying
   // the child gradient with the activation jacobian.  It is the same size as the convolution
   // that is output by the current layer.
   // Here we just generate some random numbers to represent it.
   Matrix g(cn, cn);     

   
   Matrix gp(dd, dd);   // Padded Image
   // Rows is the length of the convolution matrix.
   // Columns is the length of the Input matrix.
   Matrix w(cn * cn, n * n); // Unrolled correlation.

   // Backprop result.  Sizeof Input image.  The point of this operation is to
   // map the back propagation delta of this layer to the size of the input matrix to this layer.
   Matrix dc1(n,n);  
   Matrix dc2(n,n);

   k.setRandom();
   //g.setOnes();  // Either works.
   g.setRandom();  //
   gp.setZero();
   w.setZero();

   // Could just build the g matrix right into the padded matrix, but for this
   // code might as well keep them seperate.
   gp.block(dp, dp, cn, cn) = g;
   //WriteWeightsCSV wgp("C:\\projects\\neuralnet\\simplenet\\SNCVMNIST\\gp.csv");
   //wgp.Write(gp);

   // Itterate through the rows and columns of the correlation plane.
   for (int r = 0; r < cn; r++) {
      for (int c = 0; c < cn; c++) {
         // Each point in the correlation plane involves a summation over 
         // the target (image) surface and kernel.
         // Apply the kernel to the input image.
         for (int rr = 0; rr < kn; rr++) {
            for (int cc = 0; cc < kn; cc++) {
               // In this code we don't use the padded image.  In fact we don't use
               // any image.  We are just coping the kernel value to the correct position
               // in the unwrapped W matrix.
               int gr = r + rr - p;
               int gc = c + cc - p;
               if (gr >= 0 && gr < n && 
                   gc >= 0 && gc < n ) {
                  //int wc = gc * n + gr;
                  //int wr = c * cn + r;
                  // Unroll row wise
                  int wc = gr * n + gc;
                  int wr = r * cn + c;
                  w(wr, wc) = k(rr, cc);
               }
            }
         }
      }
   }

   OWeightsCSVFile writer(path,"unwrapmat");
   writer.Write(w,1);

   //Eigen::Map<ColVector> cv(g.data(), g.size());
   //Eigen::Map<ColVector> ov1(dc1.data(), dc1.size());

   // Multiply the flattened grad matrix by the unrolled kernel.
   // This is just to check normal convolution.
   //ov1 = w * cv;


   //WriteWeightsCSV mov1("C:\\projects\\neuralnet\\simplenet\\SNCVMNIST\\mov1.csv");
   //mov1.Write(dc1);

   // Make a row vector and perform the flattened backprop operation.
   Eigen::Map<RowVector> rv(g.data(), g.size());
   Eigen::Map<RowVector> rv1(dc1.data(), dc1.size());

   ofstream fov1("C:\\projects\\neuralnet\\simplenet\\SNCVMNIST\\grv.csv", ios::trunc);
   fov1 << rv;
   fov1.close();

   rv1 = rv * w;
   ofstream frv1("C:\\projects\\neuralnet\\simplenet\\SNCVMNIST\\rv1.csv", ios::trunc);
   frv1 << rv1;
   frv1.close();
   OWeightsCSVFile mdc1(path,"dc1");
   mdc1.Write(dc1,1);

   // rotate k by 180 degrees ------------
      int kn2 = kn / 2;
      for (int i = 0; i < kn2; i++) {
         int j = kn - i - 1;
         for (int c1 = 0; c1 < kn; c1++) {
            int c2 = kn - c1 - 1;
            double temp = k(i, c1);
            k(i, c1) = k(j, c2);
            k(j, c2) = temp;
         }
      }
      if (kn % 2) {
         int j = kn / 2;  // Don't add 1.  The zero offset compensates.
         for (int c1 = 0; c1 < kn2; c1++) {
            int c2 = kn - c1 - 1;
            double temp = k(j, c1);
            k(j, c1) = k(j, c2);
            k(j, c2) = temp;
         }
      }
   //------------------------------------------

   //WriteWeightsCSV mrk("C:\\projects\\neuralnet\\simplenet\\SNCVMNIST\\mrk.csv");
   //mrk.Write(k);

   LinearCorrelate0(gp, k, dc2);

   OWeightsCSVFile mdc2(path,"dc2");
   mdc2.Write(dc2,1);

   Matrix dif(dc2.rows(), dc2.cols());
   dif = dc1 - dc2;
   dif.cwiseAbs();

   cout << "Max dc1: " << dc1.cwiseAbs().maxCoeff() <<  " Max dc2: " << dc2.cwiseAbs().maxCoeff() << " Max dif: " << dif.maxCoeff() << endl;

   cout << "enter a key and press Enter" << endl;
   char c;
   cin >> c;
}

// This test is to determine how to use the centered
// correlator and convolver algorithms.
void TestKernelFlipper1()
{
   // Input matrix size.
   const int n = 32;          // Layer input size.
   // Kernel matrix size.
   const int kn = 16;

   const int cn = 16;         // Layer output size.  This is input matrix during backprop.

   // The padding is now implied.
   //int n2 = n >> 1;  if (!(n2 % 2)) { n2--; }
   //int k2 = kn >> 1; if (!(k2 % 2)) { k2--; }
   //int c2 = cn >> 1; if (!(c2 % 2)) { c2--; }
   //int ip = n2 - c2 - k2;
   //int p = ip < 0 ? -ip : 0;
   // Convolution / back propagation delta size.
   // cn max = n + p - kn
   // Input delta padding is a function of the kernel size and input padding.
   //int dp = kn - p - 1;
   // Input matrix padding.
   //int nn = n + 2 * p;
   // Delta matrix size with padding.
   //int dd = cn + 2 * dp;

   // Make a kernel matrix.  Unroll it into another matrix.
   // Make a grad matrix.  Flatten that matrix and multiple by the unrolled matrix.
   // Correlate the matrix with the 180 flip kernel.  The two methods should agree.

   Matrix k(kn, kn);    // Kernel
   // The incomming delta gradient.  This is the gradient that is formed by multiplying
   // the child gradient with the activation jacobian.  It is the same size as the convolution
   // that is output by the current layer.
   // Here we just generate some random numbers to represent it.
   Matrix g(cn, cn);     

   
   //Matrix gp(dd, dd);   // Padded Image
   // Rows is the length of the convolution matrix.
   // Columns is the length of the Input matrix.
   Matrix w(cn * cn, n * n); // Unrolled correlation.

   // Backprop result.  Sizeof Input image.  The point of this operation is to
   // map the back propagation delta of this layer to the size of the input matrix to this layer.
   Matrix dc1(n,n);  
   Matrix dc2(n,n);

   k.setRandom();
   g.setOnes();
   //gp.setZero();
   w.setZero();

   // Could just build the g matrix right into the padded matrix, but for this
   // code might as well keep them seperate.
   //gp.block(dp, dp, cn, cn) = g;
   //WriteWeightsCSV wgp("C:\\projects\\neuralnet\\simplenet\\SNCVMNIST\\gp.csv");
   //wgp.Write(gp);

   // Itterate through the rows and columns of the correlation plane.
   int gr2 = n >> 1; if (!(n % 2)) { gr2--; }
   int gc2 = n >> 1; if (!(n % 2)) { gc2--; }
   int hr2 = kn >> 1; if (!(kn % 2)) { hr2--; }  //
   int hc2 = kn >> 1; if (!(kn % 2)) { hc2--; }  //
   int or2 = cn >> 1; if (!(cn % 2)) { or2--; }  //
   int oc2 = cn >> 1; if (!(cn % 2)) { oc2--; }  //
   for (int r = 0; r < cn; r++) {
      for (int c = 0; c < cn; c++) {


         // Each point in the correlation plane involves a summation over 
         // the target (image) surface and kernel.
         // Apply the kernel to the input image.
         for (int rr = 0; rr < kn; rr++) {
            for (int cc = 0; cc < kn; cc++) {
               // In this code we don't use the padded image.  In fact we don't use
               // any image.  We are just coping the kernel value to the correct position
               // in the unwrapped W matrix.
               //int gr = r + rr - p;  // no padding needed.  Happens naturally.
               //int gc = c + cc - p;

               int gr = r + rr + gr2 - hr2 - or2;                 // r and c are the correlation plane
               int gc = c + cc + gc2 - hc2 - oc2;                 // coordinates.

               if (gr >= 0 && gr < n && 
                   gc >= 0 && gc < n ) {
                  //int wc = gc * n + gr;
                  //int wr = c * cn + r;
                  // Unroll row wise.  Do this to align with the Matrix construction.
                  int wc = gr * n + gc;
                  int wr = r * cn + c;
                  w(wr, wc) = k(rr, cc);
               }
            }
         }
      }
   }

   OWeightsCSVFile writer(path,"unwrapmat");
   writer.Write(w,1);

   //Eigen::Map<ColVector> cv(g.data(), g.size());
   //Eigen::Map<ColVector> ov1(dc1.data(), dc1.size());

   // Multiply the flattened grad matrix by the unrolled kernel.
   // This is just to check normal convolution.
   //ov1 = w * cv;


   //WriteWeightsCSV mov1("C:\\projects\\neuralnet\\simplenet\\SNCVMNIST\\mov1.csv");
   //mov1.Write(dc1);

   // Make a row vector and perform the flattened backprop operation.
   Eigen::Map<RowVector> rv(g.data(), g.size());
   Eigen::Map<RowVector> rv1(dc1.data(), dc1.size());

   // This is backprop the long way.  W represents the unwrapped correlation coefficients
   // and rv is the L-1 delta row vector.  The result is L delta vector.
   rv1 = rv * w;
   // rv1 is mapped onto the dc1 matrix.
   OWeightsCSVFile mdc1(path,"dc1");
   mdc1.Write(dc1,1);

//#define BY_ROTATION
#ifdef BY_ROTATION
   // rotate k by 180 degrees ------------
      int kn2 = kn / 2;
      for (int i = 0; i < kn2; i++) {
         int j = kn - i - 1;
         for (int c1 = 0; c1 < kn; c1++) {
            int c2 = kn - c1 - 1;
            double temp = k(i, c1);
            k(i, c1) = k(j, c2);
            k(j, c2) = temp;
         }
      }
      if (kn % 2) {
         int j = kn / 2;  // Don't add 1.  The zero offset compensates.
         for (int c1 = 0; c1 < kn2; c1++) {
            int c2 = kn - c1 - 1;
            double temp = k(j, c1);
            k(j, c1) = k(j, c2);
            k(j, c2) = temp;
         }
      }
   //------------------------------------------

   //WriteWeightsCSV mrk("C:\\projects\\neuralnet\\simplenet\\SNCVMNIST\\mrk.csv");
   //mrk.Write(k);

   LinearCorrelate3(g, k, dc2);
#else
   LinearConvolution3(g, k, dc2);
#endif
   OWeightsCSVFile mdc2(path,"dc2");
   mdc2.Write(dc2,1);

   Matrix dif(dc2.rows(), dc2.cols());
   dif = dc1 - dc2;
   dif.cwiseAbs();

   cout << "Max dc1: " << dc1.cwiseAbs().maxCoeff() <<  " Max dc2: " << dc2.cwiseAbs().maxCoeff() << " Max dif: " << dif.maxCoeff() << endl;

   cout << "enter a key and press Enter" << endl;
   char c;
   cin >> c;
}

void MakeCenterCircle(Matrix& m, double rad)
{
   int n = m.rows();
   int n2 = n >> 1; if (!(n % 2)) { n2--; }
   m.setConstant(0.0);
   for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
         int x = n2 - j;
         int y = n2 - i;
         double r = sqrt(x * x + y * y);
         m(i, j) = r < rad ? 1.0 : 0.0;
      }
   }
}

typedef void (*filter)(Matrix&, Matrix&, Matrix&);


void TestFilter( int sm, int sh, int so, string name, filter fn )
{
      MNISTReader::MNIST_list dl;
      int itodl[10];
      /*
      MNISTReader reader("C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data\\train\\train-images-idx3-ubyte",
         "C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data\\train\\train-labels-idx1-ubyte");

      dl = reader.read_batch(20);
      itodl[0] = 1;
      itodl[1] = 14;
      itodl[2] = 5;
      itodl[3] = 12;
      itodl[4] = 2;
      itodl[5] = 0;
      itodl[6] = 13;
      itodl[7] = 15;
      itodl[8] = 17;
      itodl[9] = 4;

      Matrix h1(28, 28);
      TrasformMNISTtoMatrix(h1, dl[ itodl[4] ].x );
      ScaleToOne(h1.data(), h1.rows() * h1.cols());

      Matrix h(14, 14);
      h = h1.block(7, 7, 14, 14);

      Matrix sx(3, 3);
      sx.setZero();
      sx(0, 0) = 1;  sx(0, 2) = -1;
      sx(1, 0) = 2;  sx(1, 2) = -2;
      sx(2, 0) = 1;  sx(2, 2) = -1;

      Matrix sy(3, 3);
      sy.setZero();
      sy(0, 0) = 1;  sy(2, 0) = -1;
      sy(0, 1) = 2;  sy(2, 1) = -2;
      sy(0, 2) = 1;  sy(2, 2) = -1;

      Matrix s(3, 3);
      //LinearConvolution(sx, sy, s);
      s.setConstant(-1);
      s(1, 1) = 4; 

      Matrix m(28, 28);
      TrasformMNISTtoMatrix(m, dl[ itodl[4] ].x );
      ScaleToOne(m.data(), m.rows() * m.cols());


      Matrix g(56,56);
      g.setZero();
      g.block(14, 14,28,28) = m;
      */

      Matrix m(sm, sm);  MakeCenterCircle(m, 5);
      Matrix h(sh, sh);  MakeCenterCircle(h, 5);
      Matrix o(so, so);  o.setZero();

      //Rotate180(h);
      fn(m, h, o);

      ScaleToOne(o.data(), o.rows() * o.cols());

      string fileroot = "C:\\projects\\neuralnet\\simplenet\\algors\\results\\";

      MakeMatrixImage(fileroot + name + ".bmp", o);

      ofstream owf(fileroot + name + ".csv", ios::trunc);

      // octave file format
      const static Eigen::IOFormat OctaveFmt(6, 0, ", ", ";\n", "", "", "", "");
      owf << o.format(OctaveFmt);
      owf.close();
}

double sqr(ColVector p)
{
   double x = p[0];
   double y = p(1);
      
   return pow( (x - 5.0), 2.0) + y * y - 100.0;
}

double sqr4(ColVector p)
{
   double w = p[0];
   double x = p(1);
   double y = p[2];
   double z = p(3);
      
   return pow( (x-5), 2.0) + y*y + 2 * w*w + z*z*z*z;
}

void test_amoeba() 
{
   AmoFunc func = sqr4;
   Amoeba aba(1.0e-5);
   Matrix p(5,4);

   p << 0.0, 0.0, 0.0, 0.0,
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0;

   ColVector res = aba.minimize<AmoFunc>(p, func);

   cout << aba.nfunc << endl;
   cout << res;
}

void rlft2(Doub *data, ColVector& speq,
	       const Int nn2, const Int nn3, const Int isign) {
	Int i1,i2,i3,j1,j2,j3,k1,k2,k3,k4;
	Doub theta,wi,wpi,wpr,wr,wtemp;
	Doub c1,c2,h1r,h1i,h2r,h2i;
   const int nn1 = 1;
	//VecInt nn(3);
	//VecDoubp spq(nn1);
	unsigned long nn[2];

	//std::unique_ptr<double*[]> manage_spq = make_unique<double*[]>(nn1);
	//double** spq = manage_spq.get();
	//for (i1 = 0; i1 < nn1; i1++) { spq[i1] = speq + 2 * nn2 * i1; }

	c1 = 0.5;
	c2 = -0.5*isign;
	theta = isign*(6.28318530717959/(Doub)nn3);
	wtemp = sin(0.5*theta);
	wpr = -2.0*wtemp*wtemp;
	wpi = sin(theta);
	//nn[0] = nn1;
	nn[0] = nn2;
	nn[1] = nn3 >> 1;
   if (isign == 1) {
      fourn(data, nn, 2, isign);
      k1 = 0;
      //for (i1=0;i1<nn1;i1++){
      for (i2 = 0, j2 = 0; i2 < nn2; i2++, k1 += nn3) {
         //if (j2 > 61) {
         //   exit(0);
         //}
         speq[j2++] = data[k1];
         speq[j2++] = data[k1 + 1];
      }
      //}
	}
	//for (i1=0;i1<nn1;i1++) {
	//	j1=(i1 != 0 ? nn1-i1 : 0);
   j1 = 0;
		wr=1.0;
		wi=0.0;
		for (i3=0;i3<=(nn3>>1);i3+=2) {
			//k1=i1*nn2*nn3;
			//k3=j1*nn2*nn3;
         k1 = 0;
         k3 = 0;
			for (i2=0;i2<nn2;i2++,k1+=nn3) {
				if (i3 == 0) {
					j2=(i2 != 0 ? ((nn2-i2)<<1) : 0);
					h1r=c1*(data[k1]+speq[j2]);
					h1i=c1*(data[k1+1]-speq[j2+1]);
					h2i=c2*(data[k1]-speq[j2]);
					h2r= -c2*(data[k1+1]+speq[j2+1]);
					data[k1]=h1r+h2r;
					data[k1+1]=h1i+h2i;
					speq[j2]=h1r-h2r;
					speq[j2+1]=h2i-h1i;
				} else {
					j2=(i2 != 0 ? nn2-i2 : 0);
					j3=nn3-i3;
					k2=k1+i3;
					k4=k3+j2*nn3+j3;
					h1r=c1*(data[k2]+data[k4]);
					h1i=c1*(data[k2+1]-data[k4+1]);
					h2i=c2*(data[k2]-data[k4]);
					h2r= -c2*(data[k2+1]+data[k4+1]);
					data[k2]=h1r+wr*h2r-wi*h2i;
					data[k2+1]=h1i+wr*h2i+wi*h2r;
					data[k4]=h1r-wr*h2r+wi*h2i;
					data[k4+1]= -h1i+wr*h2i+wi*h2r;
				}
			}
			wr=(wtemp=wr)*wpr-wi*wpi+wr;
			wi=wi*wpr+wtemp*wpi+wi;
		}
	//}
	if (isign == -1) fourn(data,nn,2,isign);
}

int main()
{
    std::cout << "Algorithm Tester!\n";
    TestKernelFlipper1();

    //REVIEW:
    // The test image was tainting the comparisons.  Fix the test image
    // and now the correlation and convolution algorithms produce same
    // result for a symetric test pattern.
    // Now need to reconcile this with the Kernel Flipping algor

    //TestFilter(32,16,16,"con_old", LinearCorrelate3);
    //TestFilter(32,16,16,"con_new", LinearConvolution3);

    // Combining Signals in the Time or Frequency domain results in the same outcome/
    /*
    ColVector vec1(256);
    vec1.setZero();
    for (int i = 0; i < 256; i++) {
       // Add some harmonics
       for (int h = 6; h <= 64; h += 6) {
          vec1(i) += 0.1 * sin((double)h * M_PI * (double)i / 256.0);
       }

       // Add a high frequency component for fun
       vec1(i) += 0.05 * sin(128.0 * M_PI * (double)i / 256.0);
    }

    ColVector vec2(256);
    vec2.setZero();
    for (int i = 0; i < 256; i++) {
       // Add some odd harmonics
       for (int h = 9; h <= 64; h += 9) {
          vec2(i) += 0.1 * sin((double)h * M_PI * (double)i / 256.0);
       }

       // Add a different high frequency component for fun
       vec2(i) += 0.05 * sin(100.0 * M_PI * (double)i / 256.0);
    }

    ColVector a1(256);
    a1 = vec1 + vec2;

    rfftsine(vec1.data(), vec1.size(), 1);
    rfftsine(vec2.data(), vec2.size(), 1);

    vec1 *= (1.0 / 128.0);
    vec2 *= (1.0 / 128.0);

    ColVector s(128);
    int j = 1;
    for (int i = 2; i < 256; i += 2, j++) {
       double re = vec1(i);
       double im = vec1(i + 1);
       s(j) = sqrt(re * re + im * im);
    }
    s(0) = vec1(0);
    s(127) = vec1(1);

    ColVector a2(256);
    a2 = vec1 + vec2;
    rfftsine(a2.data(), a2.size(), -1);

   ofstream owf(path + "\\vec1.spectrum.csv", ios::trunc);
   assert(owf.is_open());
   owf << s;
   owf.close();

   owf.open(path + "\\a1.csv", ios::trunc);
   assert(owf.is_open());
   owf << a1;
   owf.close();

   owf.open(path + "\\a2.csv", ios::trunc);
   assert(owf.is_open());
   owf << a2;

   exit(0);
   */
    /*
   Matrix t(64, 64);
   const double center = 32.5;
   for (int row = 0; row < 64; row++) {
      for (int col = 0; col < 64; col++) {
         double x = center - (double)col;
         double y = center - (double)row;
         double r = sqrt(x * x + y * y);
         double f = 0.0;
         if (r < 4.0) {
            f = 1.0;
         }
         t(row, col) = f;
      }
   }

   ofstream owf(path + "\\t.csv", ios::trunc);
   assert(owf.is_open());
   owf << t;
   owf.close();

   //unsigned long dims[] = { 64, 64 };
   ColVector speq(2*64);
   rlft2(t.data(), speq, 64, 64, 1);

   Matrix rel(32, 33);
   Matrix img(32, 33);
   Matrix spc(32, 33);

   for (int row = 0; row < 32; row++) {
      for (int col = 0; col < 32; col++) {
         double r = t(row, 2 * col);
         double i = t(row, 2 * col + 1);
         rel(row, col) = r;
         img(row, col) = i;
         spc(row, col) = sqrt(r * r + i * i);
      }
   }

      for (int row = 0; row < 32; row++) {
         double r = speq( 2 * row);
         double i = speq( 2 * row + 1);
         rel(row, 32) = r;
         img(row, 32) = i;
         spc(row, 32) = sqrt(r * r + i * i);
      }

   owf.open(path + "\\real.csv", ios::trunc);
   assert(owf.is_open());
   owf << rel;
   owf.close();

   owf.open(path + "\\imag.csv", ios::trunc);
   assert(owf.is_open());
   owf << img;
   owf.close();

   owf.open(path + "\\spc.csv", ios::trunc);
   assert(owf.is_open());
   owf << spc;
   owf.close();
   
   exit(0);
   */
   //test_amoeba();
   //exit(0);
}

   /*
   // ---- Test Gradient ---
   ColVector X(2);
   ColVector Y(1);
   X(0) = 1.5;
   X(1) = 1.0;
   Y(0) = 0.0;

   int r = 5;
   int c = 2;
   ColVector g;
   double e = 1.0e-12;
   double w = LayerList[0]->W(r,c);
   LayerList[0]->W(r,c) = w + e;
   g = X;
   for (const auto& lit : LayerList) { g = lit->Eval(g); }
   double g1 = loss.Eval(g, Y);


   LayerList[0]->W(r,c) = w - e;
   g = X;
   for (const auto& lit : LayerList) { g = lit->Eval(g); }
   double g2 = loss.Eval(g, Y);

   cout << (g1 - g2) / (2.0 * e) << endl;

   LayerList[0]->W(r,c) = w;
   g = X;
   for (const auto& lit : LayerList) { g = lit->Eval(g); }
   loss.Eval(g, Y);

   RowVector bp = loss.LossGradient();
   for (layer_list::reverse_iterator riter = LayerList.rbegin();
      riter != LayerList.rend();
      riter++) {
      bp = (*riter)->BackProp(bp);
   }

   cout << LayerList[0]->dW.transpose()(r,c) << endl;

   exit(0);
   */
