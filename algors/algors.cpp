// algors.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
//#define USE_THREAD_POOL
//#define EIGEN_DONT_PARALLELIZE
#include <Eigen>
#ifdef USE_THREAD_POOL
   #include <CXX11\\Tensor>
   #include <CXX11\\ThreadPool>
#endif
#include <iostream>
#include <iomanip>
#include <MNISTReader.h>
#include <Layer.h>
#include <bmp.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <amoeba.h>
#include <random>
#include <functional>

#include <time.h>

#include "fft1.h"
#include "fftn.h"
#include <complex>

#include <opencv2\\core.hpp>
#include <opencv2\\highgui.hpp>
#include <opencv2\\imgproc.hpp>

#include <SpacialTransformer.h>
#include <CyclicVectorTransformer.h>

string path = "C:\\projects\\neuralnet\\simplenet\\algors\\results";

void fft2ConCor(Matrix& m, Matrix& h, Matrix& out, int sign);

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

double MultiplyReverseCol(ColVector& m, int mr, ColVector& h, int hr, int size_r)
{
   double sum = 0.0;
   for (int r = 0; r < size_r; r++) {
      sum += m(mr - r) * h(hr + r);
   }
   return sum;
}

void LinearConvolve1D(ColVector& m, ColVector& h, ColVector& out)
{
   const int mrows = m.rows();
   const int hrows = h.rows();
   const int orows = out.rows();
   int mr2 = mrows >> 1; if (!(mrows % 2)) { mr2--; }
   int hr2 = hrows >> 1; if (!(hrows % 2)) { hr2--; }
   int hr2p = hrows >> 1; // The complement of hr2
   int or2 = orows >> 1; if (!(orows % 2)) { or2--; }

   for (int r = 0; r < out.rows(); r++) {     // Scan through the Correlation surface.
      int h1r;
      int m2r;
      int m1r;
      int cr, cc;
      cr = r + mr2 - or2;
      m2r = cr + hr2;  // Use h2 to the positive side because it is the negitive side
      m1r = cr - hr2p; // Similarly the negitive side physically is the positive
      h1r = 0;

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

      if (shr <= 0 ) {
         out(r) = 0.0;
      }
      else {
         out(r) = MultiplyReverseCol(m, m2r, h, h1r, shr);
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

// This is the version that gets used.  Because it is based on input
// centers it requres no padding.
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

void LinearCorrelate1D(ColVector& m, ColVector& h, ColVector& out)
{
   const int mrows = m.rows();
   const int hrows = h.rows();
   const int orows = out.rows();
   int mr2 = mrows >> 1; if (!(mrows % 2)) { mr2--; }
   int hr2 = hrows >> 1; if (!(hrows % 2)) { hr2--; }
   int or2 = orows >> 1; if (!(orows % 2)) { or2--; }

   for (int r = 0; r < orows; r++) {     // Scan through the Correlation surface.
      int h1r, h1c;
      int m1r, m1c;
      m1r = r + mr2 - or2 - hr2;

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

      if (shr <= 0) {
         out(r) = 0.0;
      }
      else {
         out(r) = (m.array().block(m1r, 0, shr, 1) * h.array().block(h1r, 0, shr, 1)).sum();
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

void TestKernelFlipper2()
{
   // Input matrix size.
   const int n = 32;          // Layer input size.
   // Kernel matrix size.
   const int kn = 5;

   const int cn = 28;         // Layer output size.  This is input matrix during backprop.

   // Make a kernel matrix.  Unroll it into another matrix.
   // Make a grad matrix.  Flatten that matrix and multiple by the unrolled matrix.
   // Correlate the matrix with the 180 flip kernel.  The two methods should agree.

   Matrix k(kn, kn);    // Kernel
   // The incomming delta gradient.  This is the gradient that is formed by multiplying
   // the child gradient with the activation jacobian.  It is the same size as the convolution
   // that is output by the current layer.
   // Here we just generate some random numbers to represent it.
   Matrix g(n, n);

   Matrix dc1(cn,cn);  
   Matrix dc2(cn,cn);

   k.setRandom();
   g.setOnes();

   fft2convolve(g, k, dc1, -1);

   Rotate180(k);

   fft2convolve(g, k, dc2, 1);

   //OWeightsCSVFile mdc2(path,"dc2");
   //mdc2.Write(dc2,1);

   Matrix dif(cn, cn);
   dif = dc1 - dc2;
   dif.cwiseAbs();

   cout << "Max dc1: " << dc1.cwiseAbs().maxCoeff() <<  " Max dc2: " << dc2.cwiseAbs().maxCoeff() << " Max dif: " << dif.maxCoeff() << endl;

   cout << "enter a key and press Enter" << endl;
   char c;
   cin >> c;
}

void MakeCenterPlus(Matrix& m, int size)
{
   // Is this a better way to arrange shapes around zero?
   // Start with range of -1 to 1 over the number of points in the matrix row or col.  Zero may or may not be one of the elements.
   // Then multiply by half the row or col value.
   //const Eigen::ArrayXf grid_x = Eigen::ArrayXf::LinSpaced(output_width, -1.0f, 1.0f) * static_cast<float>(input_width_ - 1) / 2.0f;
   //const Eigen::ArrayXf grid_y = Eigen::ArrayXf::LinSpaced(output_height, -1.0f, 1.0f) * static_cast<float>(input_height_ - 1) / 2.0f;

   int nn = m.rows();
   int mm = m.cols();
   runtime_assert(size <= nn && size <= mm);

   m.setConstant(0.0);

   int n2 = nn >> 1; if (!(nn % 2)) { n2--; }
   int m2 = mm >> 1; if (!(mm % 2)) { m2--; }

   int no = (nn - size) >> 1;
   int mo = (mm - size) >> 1;
   m.block(no, m2, size, 1).setConstant(1.0);
   m.block(n2, mo, 1, size).setConstant(1.0);
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

void MakeCenterGaussian(Matrix& m, double t)
{
   int rows = m.rows();
   int cols = m.cols();
   int n = cols > rows ? cols : rows;
   int n2 = n >> 1; if (!(n % 2)) { n2--; }
   m.setConstant(0.0);
   double c = 1.0 / (2.0 * M_PI * t);
   double c1 = 0.25 / t;
   for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
         double x = (double)(n2 - j);
         double y = (double)(n2 - i);
         //double r = sqrt(x * x + y * y);
         m(i, j) = exp( -c1 * (x * x + y * y) );
      }
   }
}

void TestMomentum()
{
   std::default_random_engine generator;
   std::normal_distribution<double> distribution(0.0, 1.0);

   ofstream owf(path +  "\\" + "momentum" + ".csv", ios::trunc);

   double v = 0.0;
   double avg_v = 0.0;
   double avg_x = 0.0;
   for (int c = 1; c <= 100; c++) {
      double a = 1.0 / (double)c;
      double b = 1.0 - a;
      double x = distribution(generator);

      double mA = 0.8;  // Momentum parameter.  valid range 0 to 1.

      // Momentum is just a simple 1st-order IIR low pass filter.
      // v = (1.0 - mA) * v + mA * x;
      // Can be written as:
      v = v + mA * (x - v);
      owf << x << " , " << v << endl;
      avg_v = b * avg_v + a * v;
      avg_x = b * avg_x + a * x;
   }

   cout << "avg_v: " << avg_v << " avg_x: " << avg_x << endl;

   char c;
   cin >> c;
}

void SobelXY(Matrix& m)
{
   Matrix hx(3,3);
   Matrix hy(3,3);

   hx << 1, 0, -1,
         2, 0, -2,
         1, 0, -1;

   hy << 1, 2, 1,
         0, 0, 0,
        -1,-2,-1;

   Matrix sx(m.rows(),m.cols());
   Matrix sy(m.rows(),m.cols());

   LinearConvolution3(m, hx, sx);
   LinearConvolution3(m, hy, sy);

   sx.array() *= sx.array();
   sy.array() *= sy.array();
   m = sx + sy;
   m.array().cwiseSqrt();
}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

void TestObjectDetection()
{
   cv::Mat image;
   image = cv::imread( path + "\\" + "objects.bmp", cv::IMREAD_GRAYSCALE);

   //cout << type2str(image.type());
   //cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
   //imshow("objs", image);
   int rows = image.rows;
   int cols = image.cols;
   //Eigen::Map<Matrix> cvm(image.ptr<double>(), rows, cols);
   Matrix m(rows, cols);
   for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
         m(r,c) = (double)image.at<unsigned char>(r, c);
         //cout << (int)image.at<unsigned char>(r, c) << ",";
      }
      //cout << endl;
   }

   //SobelXY(m);

   Matrix g(64, 64);
   MakeCenterGaussian(g, 24.0);

   Matrix out(rows, cols);
   LinearConvolution3(m, g, out);

   //ofstream owf(path +  "\\" + "gauss" + ".csv", ios::trunc);
   //owf << g;

   ofstream owf(path +  "\\" + "objs" + ".csv", ios::trunc);
   owf << out;

   //char c = cv::waitKey(0);
}

typedef function<void(Matrix&, Matrix&, Matrix&)> fn_filter;
typedef function<void(Matrix&)> fn_mat_shape;

typedef function<void(ColVector&, ColVector&, ColVector&)> fn_filter1D;
typedef function<void(ColVector&)> fn_mat_shape1D;

void TestEdgeDetect()
{
      MNISTReader::MNIST_list dl;
      int itodl[10];
      
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

      Matrix g(28, 28);
      TrasformMNISTtoMatrix(g, dl[ itodl[4] ].x );
      ScaleToOne(g.data(), g.rows() * g.cols());

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

      Matrix lap(3, 3);
      lap.setZero();
                      lap(1, 0) =  1;  
      lap(0, 1) = 1;  lap(1, 1) = -4;  lap(2, 1) =  1;
                      lap(1, 2) =  1;

      Matrix o(28, 28);

      LinearConvolution3(g, lap, o);

      ScaleToOne(o.data(), o.rows() * o.cols());

      string name = "lap_edge";

      MakeMatrixImage(path + "\\" + name + ".bmp", o);

      ofstream owf(path +  "\\" + name + ".csv", ios::trunc);

      // octave file format
      const static Eigen::IOFormat OctaveFmt(6, 0, ", ", ";\n", "", "", "", "");
      owf << o.format(OctaveFmt);
      owf.close();
}


void TestFilterFunction( int sm, int sh, int so, string name, fn_filter fn, fn_mat_shape shp1, fn_mat_shape shp2)
{
   Matrix m(sm, sm);  shp1(m);
   Matrix h(sh, sh);  shp2(h);
   Matrix o(so, so);  o.setZero();

   //Rotate180(h);
   fn(m, h, o);

   unsigned int mr = 0;
   unsigned int mc = 0;
   double max = 0.0;
   for (int r = 0; r < so; r++) {
      for (int c = 0; c < so; c++) {
         if (max < o(r, c)) {
            max = o(r, c);
            mr = r;
            mc = c;
         }
      }
   }

   cout << "max row: " << mr << " col:" << mc << endl;

   // octave file format
   ofstream owf(path +  "\\" + name + ".csv", ios::trunc);
   const static Eigen::IOFormat OctaveFmt(6, 0, ", ", ";\n", "", "", "", "");
   owf << o.format(OctaveFmt);
   owf.close();

   ScaleToOne(o.data(), o.rows() * o.cols());
   MakeMatrixImage(path + "\\" + name + ".bmp", o);
}

void TestFilterFunction1D(int sm, int sh, int so, string name, fn_filter1D fn, fn_mat_shape1D shp1, fn_mat_shape1D shp2)
{


   ColVector m(sm);  shp1(m);
   ColVector h(sh);  shp2(h);
   ColVector o(so);  o.setZero();

   fn(m, h, o);

   ScaleToOne(o.data(), o.rows() * o.cols());

   ofstream owf(path + "\\" + name + ".csv", ios::trunc);

   unsigned int mr = 0;
   double max = 0.0;
   for (int r = 0; r < so; r++) {
      if (max < o(r)) {
         max = o(r);
         mr = r;
      }
   }

   cout << "max row: " << mr << endl;

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


void TestFilterFunctionDriver(int sm, int sh, int so, fn_filter fn, fn_filter ln, fn_mat_shape shp1, fn_mat_shape shp2)
{
   char buf[255];
   std::snprintf(buf, 255, "fc_%d_%d_%d", sm, sh, so);
   cout << "FFT Filter" << endl;
   TestFilterFunction(sm, sh, so, buf, fn, shp1, shp2 );
   cout << endl << "Linear Filter" << endl;
   std::snprintf(buf, 255, "lc_%d_%d_%d", sm, sh, so);
   TestFilterFunction(sm, sh, so, buf, ln, shp1, shp2);
   cout << endl;
}

void TestFilter1Driver(int sm, int sh, int so, fn_filter1D fn, fn_filter1D ln, fn_mat_shape1D shp1, fn_mat_shape1D shp2)
{
   char buf[255];
   std::snprintf(buf, 255, "fc1D_%d_%d_%d", sm, sh, so);
   cout << "FFT Filter" << endl;
   TestFilterFunction1D(sm, sh, so, buf, fn, shp1, shp2);
   cout << endl << "Linear Filter" << endl;
   std::snprintf(buf, 255, "lc1D_%d_%d_%d", sm, sh, so);
   TestFilterFunction1D(sm, sh, so, buf, ln, shp1, shp2);
   cout << endl;
}

void ShiftExperiment()
{
   auto gauss2d = [](Matrix& m, double sigma, int rc, int cc) {
      const unsigned int rows = m.rows();
      const unsigned int cols = m.cols();
      for (int r = 0; r < rows; r++) {
         for (int c = 0; c < cols; c++) {
            double e = (std::pow(r - rc, 2.0) + std::pow(c - cc, 2.0)) / (2.0 * sigma);
            m(r, c) = std::exp(-e);
         }
      }
   };

   auto square = [](Matrix& m, double sigma, int rc, int cc) {
      const unsigned int rows = m.rows();
      const unsigned int cols = m.cols();
      for (int r = 0; r < rows; r++) {
         for (int c = 0; c < cols; c++) {
            if (std::abs(r - rc) <= sigma && std::abs(c - cc) <= sigma) {
               m(r, c) = 1.0;
            }
            else {
               m(r, c) = 0.0;
            }
         }
      }
   };

   const double DX = 3.0;
   const double DY = 5.0;

   const unsigned int R = 64;
   const unsigned int C = 64;

   Matrix m(R, C);
   Matrix mn(R,2);
   //gauss2d(m, 8.0, 31, 31);
   square(m, 8.0, 31, 31);

   ofstream owf(path + "\\gauss.csv", ios::trunc);
   const static Eigen::IOFormat OctaveFmt(6, 0, ", ", ";\n", "", "", "", "");
   owf << m.format(OctaveFmt);
   owf.close();

   // rlft3 must be given a row-major matrix.
   runtime_assert(Eigen::MatrixBase<decltype(m)>::IsRowMajor );
   rlft3(m.data(), mn.data(), 1, R, C, 1);

   double fac = 2.0 / (R * C);

   const unsigned int C2 = C >> 1;

   Matrix real(R, C2);
   Matrix imag(R, C2);
   Matrix m_real(R, C2 + 1);
   Matrix m_imag(R, C2 + 1);

   // There are +- R/2 frequencies.

   // There are zero to ocols frequencies in the column direction
   // so we can simply use c.

   double* pdat = m.data();
   for (unsigned int r = 0; r < R; r++) {
      for (unsigned int c = 0; c < C2; c++) {
         unsigned int cc = c << 1;
         double a = m(r, cc);
         double b = m(r, cc + 1);

         // So, the result is stored in row-major.  This is complex pairs stored
         // in row-major order.  So that means that the resultant matrix is 64 (in this example)
         // wide or C wide.  The columns are associated with the real valued first FFT and is
         // 32 or C/2 complex pairs, thus the origional C wide, however it is indexed in steps of 2
         // since the complex pairs are lined up in the row.
         // There are simply R rows containing the full positive and negitive spectrum, but since
         // the complex pairs are across the row (spans 2 columns) there are simply R rows.
         // So the shape of the matrix turns out to be the origional R x C.


         //unsigned int i = r * 2 * ocols + 2 * c;
         //double a = pdat[i];
         //double b = pdat[i+1];
         m_real(r, c) = a;
         m_imag(r, c) = b;

         double o = fac * a;
         double p = fac * b;
         // The shift terms.
         double e = cos(-2.0 * EIGEN_PI * (DX * (double)r / (double)R + DY * (double)c / (double)C));
         double f = sin(-2.0 * EIGEN_PI * (DX * (double)r / (double)R + DY * (double)c / (double)C));

         m(r, cc)     = e * o + f * p;
         m(r, cc + 1) = e * p - f * o;

         real(r, c) = e;
         imag(r, c) = -f;
      }
   }

   cout << "\n" << mn.maxCoeff() << endl;

   for (unsigned int r = 0; r < R; r++) {
      // Introduce a shift in the result of -sign.
      //
      double o = fac * mn(r, 0);
      double p = fac * mn(r, 1);

      m_real(r, C2) = mn(r, 0);
      m_imag(r, C2) = mn(r, 1);

      // The shift terms.
      double e = cos(-2.0 * EIGEN_PI * (DX * (double)r / (double)R + DY * 0.5 ));
      double f = sin(-2.0 * EIGEN_PI * (DX * (double)r / (double)R + DY * 0.5 ));

      // Compute modified result.
      mn(r, 0) = e * o + f * p;
      mn(r, 1) = e * p - f * o;
   }

   rlft3(m.data(), mn.data(), 1, R, C, -1);

   owf.open(path + "\\gaussout.csv", ios::trunc);
   owf << m.format(OctaveFmt);
   owf.close();

   owf.open(path + "\\greal.csv", ios::trunc);
   owf << m_real.format(OctaveFmt);
   owf.close();

   owf.open(path + "\\gimag.csv", ios::trunc);
   owf << m_imag.format(OctaveFmt);
   owf.close();

   owf.open(path + "\\real.csv", ios::trunc);
   owf << real.format(OctaveFmt);
   owf.close();

   owf.open(path + "\\imag.csv", ios::trunc);
   owf << imag.format(OctaveFmt);
   owf.close();
}
/*
#include <vector>
#include <thread>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>

class ThreadPool {
public:
   ThreadPool(size_t num_threads) {
      // Create worker threads and start them
      for (size_t i = 0; i < num_threads; ++i) {
         workers.emplace_back([this] {
            while (true) {
               // Get the next task from the queue
               std::function<void()> task;
               {
                  std::unique_lock<std::mutex> lock(queue_mutex);
                  condition.wait(lock, [this] {
                     return stop || !tasks.empty();
                     });
                  if (stop && tasks.empty()) {
                     // Stop requested and no more tasks, so exit the thread
                     return;
                  }
                  task = std::move(tasks.front());
                  tasks.pop();
               }
               // Execute the task
               task();
            }
            });
      }
   }

   ~ThreadPool() {
      // Request stop and notify all threads to wake up
      {
         std::unique_lock<std::mutex> lock(queue_mutex);
         stop = true;
      }
      condition.notify_all();
      // Join all threads
      for (auto& worker : workers) {
         worker.join();
      }
   }

   // ParallelFor method that executes a function in parallel for a given index range
   template<typename Func>
   void ParallelFor(int start_index, int end_index, Func func) {
      // Divide the range into chunks based on the number of threads
      int chunk_size = (end_index - start_index + 1) / workers.size();
      if (chunk_size < 1) {
         chunk_size = 1;
      }
      // Add tasks to the queue for each chunk
      std::vector<std::future<void>> futures;
      for (int i = start_index; i <= end_index; i += chunk_size) {
         int chunk_end = std::min(i + chunk_size - 1, end_index);
         futures.emplace_back(std::async(std::launch::async, [i, chunk_end, func] {
            for (int j = i; j <= chunk_end; ++j) {
               func(j);
            }
            }));
      }
      // Wait for all tasks to complete
      for (auto& future : futures) {
         future.wait();
      }
   }

private:
   std::vector<std::thread> workers;
   std::queue<std::function<void()>> tasks;
   std::mutex queue_mutex;
   std::condition_variable condition;
   bool stop = false;
};
*/

void EnviOut(char* name)
{
   char* value = nullptr;
   std::size_t size;
   errno_t error = _dupenv_s(&value, &size, name);
   if (error == 0 && value != nullptr) {
      std::cout << "The value of " << name << " is: " << value << std::endl;
      free(value);
   }
   else {
      std::cout << name << " is not set." << std::endl;
   }
}

void FastMath()
{
   ColVector z(1000);
   ColVector x(1000);
   Matrix m(1000, 1000);

   EnviOut("OMP_NUM_THREADS");
   EnviOut("OMP_PROC_BIND");
   EnviOut("OMP_SCHEDULE");
   EnviOut("OMP_NESTED");
   cout << "Eigen threads:" << Eigen::nbThreads() << endl;

   char c;
   cout << "Ready? Hit a key." << endl;
   cin >> c;

   x.setConstant(1);
   m.setZero();
   for (int i = 0; i < 100; i++) {
      m(i, i) = i;
   }




   //cout << "Using " << omp_get_max_threads() << " threads." << endl;

   auto start_time = std::chrono::high_resolution_clock::now();
#ifdef USE_THREAD_POOL
   Eigen::ThreadPool pool(4);
   Eigen::ThreadPoolDevice dev(&pool, 4);


   Eigen::TensorOpCost toc = Eigen::TensorOpCost(m.size(), m.size(), m.size());
   dev.parallelFor(m.rows(), toc, [&](Eigen::Index a, Eigen::Index b) {
      //cout << a << "," << b << endl;
      z.block(a, 0, b - a, 1) = m.block(a, 0, b - a, m.cols()) * x;
      });
#else
   for (int i = 0; i < 1000; i++) {
      //cout << Eigen::nbThreads() << endl;
      z = m * x;
   }
#endif
   auto end_time = std::chrono::high_resolution_clock::now();
   auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
   std::cout << "Duration: " << duration_ms.count() << " ms" << std::endl;

   //cout << z.block(0,0,10,1) << endl;

   // Use the thread pool to parallelize the multiplication
   //pool.(0, matrix.rows(), [&](int i) {
   //   result(i) = (matrix.row(i) * colVector)(0);
   //   });


}

// Test spacial transformer
void test_transformer()
{
   const Size size_in(28, 28);
   //const Size size_out(28, 28);
   int waste = 0;
   const Size size_out(32, 32 - waste);
   Matrix shape(size_in.rows, size_in.cols);
   Matrix out(size_out.rows, size_out.cols);

   MakeCenterPlus(shape, 10.0);

   ColVector t(6);
   t.setZero();

   // Layout
   //   0     1      4
   //   2     3      5
   /*
   t(0) = 1.25;
   t(1) = 0.0;
   t(2) = 0.0;
   t(3) = 1.25;
   */

   t(0) = cos(M_PI * 45.0 / 180.0); // 1;
   t(1) = -sin(M_PI * 45.0 / 180.0);
   t(2) = sin(M_PI * 45.0 / 180.0);
   t(3) = cos(M_PI * 45.0 / 180.0); // 1;

   t(4) = 0.5;  // Traslate in the colunm direction.
   t(5) = 0.5;
   gridAffine grd(size_in, size_out);

/*
   ColVector t(2);
   t.setZero();

   t(0) = 13.0; 
   t(1) = 14.0;
   gridLogPolar grd(size_in, size_out, waste);
   */
   SampleMatrix sm;
   samplerBiLinear smp(size_in, size_out);

   vector_of_matrix vom_shape;
   vom_shape.push_back(shape);
   vector_of_matrix vom_out;
   vom_out.push_back(out);
   grd.Eval(sm, t);
   smp.Eval(vom_shape, vom_out, sm);
   out = vom_out[0];

   OWeightsCSVFile fcsv(path, "transformer");
   MakeMatrixImage(path + "\\transformer_in.bmp", shape);
   MakeMatrixImage(path + "\\transformer_out.bmp", out);
   fcsv.Write(shape, 0);
   fcsv.Write(out, 1);

   // Test Backprop.
   // Using out as dV.
   // out.setConstant(1.0);
   // RowVector dt = st.BackpropGrid(out);
   // cout << dt << endl;

   //out.setZero();
   //out(14, 14) = 1.0;

   //shape = st.BackpropSampler(out);
   //cout << shape << endl;
}

   #define COMPUTE_LOSS {\
      grd.Eval(sm, t);\
      vom_out[0].resize(size_out.rows, size_out.cols);\
      smp.Eval(vom_in, vom_out, sm);\
      vom_out = flat.Eval(vom_out);\
      e = loss.Eval(vom_out[0].col(0), label);\
   }



//#define AFFINE

void TestTransformerGradComp()
{
   const Size size_in(28, 28);
   const int waste = 0;
   //const Size size_out(32, 32 - waste);
   const Size size_out(28, 28);

   Flatten2D flat(Size(size_out.rows, size_out.cols), 1);
   LossL2 loss(size_out.rows * size_out.cols, 1);

   ColVector label(size_out.rows * size_out.cols);
   label.setZero();

   vector_of_matrix vom_in(1);
   vom_in[0].resize(size_in.rows, size_in.cols);
   //vom_in[0].setRandom();
   MakeCenterPlus(vom_in[0], 10.0);
  // MakeCenterCircle(vom_in[0], 10.0);
   //MakeCenterGaussian(vom_in[0], 10.0);

   vector_of_matrix vom_out(1);

   ColVector vv;

   double e;
   double f1, f2;
   
#ifdef AFFINE
   ColVector t(6);
   t.setZero();

   // Layout
   //   0     1      4
   //   2     3      5
   t(0) = cos(M_PI * 45.0 / 180.0); // 1;
   t(1) = -sin(M_PI * 45.0 / 180.0); // 0;
   t(2) = sin(M_PI * 45.0 / 180.0); // 0;
   t(3) = cos(M_PI * 45.0 / 180.0); // 1;

   //t(0) =  1;
   //t(1) =  0;
   //t(2) =  0;
   //t(3) =  1;   
   t(4) = 1.0;  // Traslate in the col direction.
   t(5) = 1.0;

   gridAffine grd(size_in, size_out);
#else  
   ColVector t(2);
   t.setZero();

   t(0) = 13.0;
   t(1) = 14.0;
   gridLogPolar grd(size_in, size_out, waste);
#endif

   SampleMatrix sm;
   samplerBiLinear smp(size_in, size_out);

   /*
   // Test Sample Gradient  (U to V)
   
   Matrix dif(size_in.rows, size_in.cols);
   dif.setZero();

   for (int r = 0; r < size_in.rows; r++) {
      std::cout << ".";
      for (int c = 0; c < size_in.cols; c++) {
         double eta = 1.0e-5;

         double w1 = vom_in[0](r, c);
         //----- Eval ------
         vom_in[0](r, c) = w1 - eta;
         COMPUTE_LOSS
         f1 = e;

         vom_in[0](r, c) = w1 + eta;
         COMPUTE_LOSS
         f2 = e;

         vom_in[0](r, c) = w1;

         dif(r, c) = (f2 - f1) / (2.0 * eta);
      }
   }

   COMPUTE_LOSS
   vector_of_matrix vb(1);
   RowVector gg = loss.LossGradient();
   vb[0] = gg;
   vb = flat.BackProp(vb);

   vom_out = smp.Backprop(vb, sm);

   OWeightsCSVFile fcsv(path, "samp_grad");
   fcsv.Write(dif, 0);
   fcsv.Write(vom_out[0], 1);

   cout << "Du//Dv max coeff: " << (dif - vom_out[0]).maxCoeff() << endl;
*/   
   /*
   ColVector tdif(t.size());

   for (int i = 0; i < t.size(); i++) {
      double eta = 1.0e-5;

      double w1 = t(i);
      //----- Eval ------
      t(i) = w1 - eta;
      COMPUTE_LOSS
      f1 = e;

      t(i) = w1 + eta;
      COMPUTE_LOSS
      f2 = e;

      t(i) = w1;

      tdif(i) = (f2 - f1) / (2.0 * eta);

   }

   COMPUTE_LOSS

   vector_of_matrix vm_backprop(1);
   RowVector g = loss.LossGradient();
   vm_backprop[0] = g;
   vm_backprop = flat.BackProp(vm_backprop);

   Matrix dldr;
   Matrix dldc;

   smp.ComputeGridGradients(dldr, dldc, vm_backprop, sm);
   g = grd.Backprop(dldr,dldc,sm);

   cout << "fintie dif: " << tdif.col(0).transpose() << endl
      << "analog dif: " << g << endl;

   tdif -= g.transpose();
   cout << "differ: " << tdif.col(0).transpose() << endl;
*/


   ColVector l(100);
   ColVector dl(100);

   for (int i = 0; i < 100; i++) {
      t(1) = 14.0 + 0.01 * (double)i;

      COMPUTE_LOSS
      l(i) = e;

      vector_of_matrix vm_backprop(1);
      RowVector g = loss.LossGradient();
      vm_backprop[0] = g;
      vm_backprop = flat.BackProp(vm_backprop);

      Matrix dldr;
      Matrix dldc;

      smp.ComputeGridGradients(dldr, dldc, vm_backprop, sm);
      g = grd.Backprop(dldr, dldc, sm);

      dl(i) = g(1);
   }

   OWeightsCSVFile lcsv(path, "loss");
   lcsv.Write(l, 0);
   lcsv.Write(dl, 1);

   cout << "enter a key and press Enter" << endl;
   char c;
   cin >> c;
}
#undef COMPUTE_LOSS

#define COMPUTE_LOSS {\
   st.Eval(shape, out, t);\
   e = loss.Eval(out, label);\
}

void TestCyclicTransformerGradComp()
{
   int size_in = 28;
   int size_out = 32;
   ColVector shape(size_in);
   ColVector out(size_out);

   LossL2 loss(size_out, 1);

   ColVector label(size_out);
   label.setZero();

   double e;
   double f1, f2;

   shape.setRandom();

   double t = 2.5;

   CyclicVectorTransformer st(size_in, size_out);

   // Test Sample Gradient  (U to V)

   RowVector dif(size_in);
   dif.setZero();

   for (int r = 0; r < size_in; r++) {
      std::cout << ".";

      double eta = 1.0e-5;

      double w1 = shape(r);
      //----- Eval ------
      shape(r) = w1 - eta;
      COMPUTE_LOSS
      f1 = e;

      shape(r) = w1 + eta;
      COMPUTE_LOSS
      f2 = e;

      shape(r) = w1;

      dif(r) = (f2 - f1) / (2.0 * eta);
   }

   COMPUTE_LOSS
   RowVector gg = loss.LossGradient();

   gg = st.BackpropSampler(gg);

   OWeightsCSVFile fcsv(path, "samp_grad");
   fcsv.Write(dif, 0);
   fcsv.Write(gg, 1);


   cout << "Du//Dv error max coeff: " << (dif - gg).maxCoeff() << endl;


   double tdif;

   double eta = 1.0e-5;

   double w1 = t;
   //----- Eval ------
   t = w1 - eta;
   COMPUTE_LOSS
   f1 = e;

   t = w1 + eta;
   COMPUTE_LOSS
   f2 = e;

   t = w1;

   tdif = (f2 - f1) / (2.0 * eta);


   COMPUTE_LOSS

   RowVector g = loss.LossGradient();
   g = st.BackpropGrid(g);

   cout << "fintie dif: " << tdif << endl
      << "analog dif: " << g(0) << endl;

   cout << "enter a key and press Enter" << endl;
   char c;
   cin >> c;
}

void TestCyclicTransform()
{
   int size_in = 28;
   int size_out = 16;
   ColVector shape(size_in);
   ColVector out(size_out);

   Matrix temp(size_in, size_in);
   MakeCenterGaussian(temp, 5);
   shape = temp.col(size_in >> 1);


   CyclicVectorTransformer st(size_in, size_out);


   OWeightsCSVFile fcsv(path, "cyc_xform");
   fcsv.Write(shape, 0);

   st.Eval(shape, out, 3.5);
   fcsv.Write(out, 1);

   st.Eval(shape, out, 5.75);
   fcsv.Write(out, 2);

   st.Eval(shape, out, 10.35);
   fcsv.Write(out, 3);

   st.Eval(shape, out, -5.75);
   fcsv.Write(out, 4);

   st.Eval(shape, out, -49.0);
   fcsv.Write(out, 5);
}

void TestFatner()
{
   MNISTReader::MNIST_list dl;
   int itodl[10];

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

   Matrix g(28, 28);
   TrasformMNISTtoMatrix(g, dl[itodl[4]].x);
   ScaleToOne(g.data(), g.rows() * g.cols());

   MakeMatrixImage(path + "\\origional.bmp", g);

   Matrix h(28, 28);
   double r_c = 13.5;
   double c_c = 13.5;
   double sig = 2.0;
   double norm = 1.0 / (2.0 * M_PI * sig);

   for (int r = 0; r < 28; r++) {
      for (int c = 0; c < 28; c++) {
         double rr = r - r_c;
         double cc = c - c_c;
         double e = (rr * rr + cc * cc) / (2.0 * sig);
         h(r, c) = norm * std::exp(-e);
      }
   }

   Matrix o(28, 28);

   //LinearConvolution3(g, h, o);
   fft2convolve(g, h, o, 1, true, true);

   //ScaleToOne(o.data(), o.rows() * o.cols());
   double t = 0.15;
   for (int r = 0; r < 28; r++) {
      for (int c = 0; c < 28; c++) {
         if (o(r, c) >= t) {
            o(r, c) = 1.0;
         }
         else {
            o(r, c) = 0.0;
         }
      }
   }

   string name = "fat_four";

   MakeMatrixImage(path + "\\" + name + ".bmp", o);

   ofstream owf(path + "\\" + name + ".csv", ios::trunc);

   // octave file format
   const static Eigen::IOFormat OctaveFmt(6, 0, ", ", ";\n", "", "", "", "");
   owf << o.format(OctaveFmt);
   owf.close();
}

void ComputeCentroid(double& x, double& y, Matrix& m)
{
   int rows = m.rows();
   int cols = m.cols();
   double total_x = 0;
   double total_y = 0;
   double weight = 0;

   // Iterate through the matrix to calculate the centroid
   for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
         double value = m(i, j);
         if (value > 0.0) {
            total_x += j * value;
            total_y += i * value;
            weight += value;
         }
      }
   }

   // Calculate the centroid
   if (weight > 0.0) {
      x = total_x / weight;
      y = total_y / weight;
   }
}

void TestGauss2DFitter()
{
   // ToDo:
   // Generate a matrix to fit.
   // Write the Gauss function and deravitive.
   // Itterate to find the fit.
   // Output the result.
   // Output the input matrix and the fit matrix to CSV.

   struct grad3 {
      double dgdrc;
      double dgdcc;
      double dgdlam;
      grad3(double drc, double dcc, double dlam) : dgdrc(drc), dgdcc(dcc), dgdlam(dlam) {}
      grad3() : dgdrc(0.0), dgdcc(0.0), dgdlam(0.0) {}
   };

   auto g = [](double rc,double cc,double lam,double r, double c) {
      double dr = r - rc;
      double dc = c - cc;
      double sig = 2.0 * lam;
      return (exp(-(dr * dr + dc * dc) / sig));
   };

   auto grad_g = [g](double rc, double cc, double lam, double r, double c) {
      double dr = r - rc;
      double dc = c - cc;
      double dgdrc = g(rc, cc, lam, r, c) * (r - rc) / lam;
      double dgdcc = g(rc, cc, lam, r, c) * (c - cc) / lam;
      double dgdl = 0.5 * g(rc, cc, lam, r, c) * (dr * dr + dc * dc) / (lam * lam);

      return grad3(dgdrc,dgdcc,dgdl);
   };

   /*
   const double RC = 16.0;
   const double CC = 14.0;
   const double LAM = 10.0;

   ColVector gauss(32);
   ColVector dgdr(32);
   ColVector dgdc(32);
   ColVector dgdl(32);

   ColVector g1(32);
   ColVector g2(32);
   double h = 0.001;

   for (int r = 0; r < 32; r++) {
      g1(r) = g(RC, CC, LAM - h, (double)r, CC);
      g2(r) = g(RC, CC, LAM + h, (double)r, CC);
   }

   g2 -= g1;
   g2 /= (2.0 * h);



   for (int r = 0; r < 32; r++) {
      gauss(r) = g(RC, CC, LAM, (double)r, CC);
      grad3 g3 = grad_g(RC, CC, LAM, (double)r, CC);
      dgdr(r) = g3.dgdrc;
      dgdc(r) = g3.dgdcc;
      dgdl(r) = g3.dgdlam;
   }

   ofstream owf(path + "\\gauss.csv", ios::trunc);
   assert(owf.is_open());
   owf << gauss;
   owf.close();

   owf.open(path + "\\gauss_dr.csv", ios::trunc);
   assert(owf.is_open());
   owf << dgdr;
   owf.close();

   owf.open(path + "\\gauss_test_dl.csv", ios::trunc);
   assert(owf.is_open());
   owf << g2;
   owf.close();

   owf.open(path + "\\gauss_dc.csv", ios::trunc);
   assert(owf.is_open());
   owf << dgdc;
   owf.close();

   owf.open(path + "\\gauss_dl.csv", ios::trunc);
   assert(owf.is_open());
   owf << dgdl;
   owf.close();

   return;
   */

   const int rows = 32;
   const int cols = 28;
   Matrix f(rows, cols);
   f.setRandom();
   f *= 0.1;

   for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
         f(r, c) += 0.25 * g(9.0, 10.0, 4.0, r, c);
         f(r, c) += 0.25 * g(10.0, 18.0, 6.0, r, c);
         f(r, c) += 0.25 * g(18.0, 12.0, 7.0, r, c);
         f(r, c) += 0.25 * g(19.0, 11.0, 3.0, r, c);
      }
   }

   double step = 0.05;
   double mc = 16.0;
   double nc = 14.0;
   double lam = 6.0;

   ComputeCentroid(mc, nc, f);

   for (int n = 0; n < 10000; n++) {
      // Compute E and the grad of E
      grad3 grad_E;
      double err = 0;

      for (int r = 0; r < rows; r++) {
         for (int c = 0; c < cols; c++) {
            double dr = r - mc;
            double dc = c - nc;
            double gg = g(mc, nc, lam, r, c);
            double zz = f(r,c) - gg;
            grad3 g3 = grad_g(mc, nc, lam, r, c);


            // Accumulate the error term
            err += (zz * zz);

            // Compute the grad of E
            grad_E.dgdrc += (-2.0 * zz * g3.dgdrc);
            grad_E.dgdcc += (-2.0 * zz * g3.dgdcc);
            grad_E.dgdlam += (-2.0 * zz * g3.dgdlam);
         }
      }
      if (!(n % 100)) {
         step *= 0.99;
         cout << "step: " << step << " err: " << err << " mc: " << mc << " nc: " << nc << " lam: " << lam << endl;
      }

      mc = mc - step * grad_E.dgdrc;
      nc = nc - step * grad_E.dgdcc;
      lam = lam - step * grad_E.dgdlam;
      if (lam < 1.0) {
         lam = 1.0;
      }
   }

   cout << "mc: " << mc << " nc: " << nc << " lam: " << lam << endl;

   Matrix g1(rows,cols);

   for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
         g1(r,c) = g(mc, nc, lam, (double)r, c);
      }
   }

   ofstream owf(path + "\\gauss.csv", ios::trunc);
   assert(owf.is_open());
   owf << g1;
   owf.close();

   owf.open(path + "\\tofit.csv", ios::trunc);
   assert(owf.is_open());
   owf << f;
   owf.close();
}

int main()

{
    std::cout << "Algorithm Tester!\n";

    TestGauss2DFitter();

    //TestFatner();

    //test_transformer();
    //TestTransformerGradComp();
    //TestCyclicTransformerGradComp();
    //TestCyclicTransform();
    //return 0;
    //ColVector test;
    //test = ColVector::LinSpaced(11, -1.0, 1.0) * ((11 - 1) / 2.0);

    //ShiftExperiment();
    //return 1;

    //TestKernelFlipper1();
    //TestKernelFlipper2();
    //TestObjectDetection();
    //TestEdgeDetect();

    /*
    Matrix m(32,32);
    m.setZero();
    m(15, 15) = 1.0;

    unsigned int rows = 64;
    unsigned int cols = 64;
    unsigned int mrows = 32;
    unsigned int mcols = 32;
    // Copy the m matrix to temporary matrix pm.
    Matrix pm(rows, cols);
    unsigned int sr = rows - mrows; sr >>= 1;
    unsigned int sc = cols - mcols; sc >>= 1;
    pm.setZero();
    pm.block(sr, sc, mrows, mcols) = m;

    unsigned int mr = 0;
    unsigned int mc = 0;
    double max = 0.0;
    for (int r = 0; r < rows; r++) {
       for (int c = 0; c < cols; c++) {
          if (max < pm(r, c)) {
             max = pm(r, c);
             mr = r;
             mc = c;
          }
       }
    }
    cout << "max row: " << mr << " col:" << mc << endl;
    */

   // FastMath();

/*
    auto fft1filter = [](Matrix& m, Matrix& h, Matrix& o) { fft2convolve(m, h, o, -1, true, true); };
    auto fft2filter = [](Matrix& m, Matrix& h, Matrix& o) { fft2convolve(m, h, o, 1, true, true); };


    auto shp = [](Matrix& m) { MakeCenterCircle(m, 5); };
    auto shp_pt = [](Matrix& m) { m.setZero(); m(15, 15) = 1.0; };
    auto shp_edg = [](Matrix& m) { 
       m.setZero(); 
       int col = 0;
       m.col(col).setConstant(1.0); 
       m(0, col) = 0;
       m(1, col) = 0;
       m(30,col) = 0;
       m(31,col) = 0;
    };
    auto shp_sploch = [](Matrix& m) {
       m.setZero();
       int col = 31;
       m(3, col) = 1;
       m(4, col) = 1;
       m(15, col) = 1;
       m(16, col) = 1;

       col = 30;
       m(3, col) = 1;
       m(4, col) = 1;
       m(15, col) = 1;
       m(16, col) = 1;
    };

    auto shp_gauss = [](Matrix& m) {
       double sig = 4.0;
       int ksr = m.rows();
       int ksc = m.cols();
       double r_c = ((double)ksr) / 2.0;
       double c_c = ((double)ksr) / 2.0;

       double norm = 1.0 / (2.0 * M_PI * sig);

       for (int r = 0; r < ksr; r++) {
          for (int c = 0; c < ksc; c++) {
             double rr = r - r_c;
             double cc = c - c_c;
             double e = (rr * rr + cc * cc) / (2.0 * sig);
             m(r, c) = norm * std::exp(-e);
          }
       }
    };


    //TestFilterFunctionDriver(32, 32, 32, fft1filter, LinearCorrelate3, shp_edg, shp_pt);

    TestFilterFunctionDriver(32, 32, 32, fft2filter, LinearConvolution3, shp_sploch, shp_gauss);

    /*
    auto fft1filter = [](ColVector& m, ColVector& h, ColVector& o) { fft1convolve(m, h, o, -1, true); };
    auto fft2filter = [](ColVector& m, ColVector& h, ColVector& o) { fft1convolve(m, h, o, 1, true); };

    auto l1filter = [](ColVector& m, ColVector& h, ColVector& o) { LinearCorrelate1D(m, h, o); };
    auto l2filter = [](ColVector& m, ColVector& h, ColVector& o) { LinearConvolve1D(m, h, o); };


    auto MakeCenterPatch = [](ColVector& m )
    {
       const double rad = 5;
       int n = m.size();
       int n2 = n >> 1; if (!(n % 2)) { n2--; }
       m.setConstant(0.0);
       for (int j = 0; j < n; j++) {
          int x = n2 - j;
          double r = sqrt(x * x);
          m(j) = r < rad ? 1.0 : 0.0;
       }
    };
    auto shp_pt1 = [](ColVector& m) { m.setZero(); m(15) = 1.0; };
    auto shp_gauss = [](ColVector& m) {
       double sig = 2.0;
       double norm = 1.0 / (2.0 * M_PI * sig);
       int ksr = m.rows();
       double r_c = ((double)ksr) / 2.0;

       for (int r = 0; r < ksr; r++) {
          double rr = r - r_c;
          double e = (rr * rr) / (2.0 * sig);
          m(r) = norm * std::exp(-e);
       }
    };

    auto shp_half_gauss = [](ColVector& m) {
       double sig = 2.0;
       double norm = 1.0 / (2.0 * M_PI * sig);
       int ksr = m.rows();
       double r_c = ((double)ksr) / 2.0;

       m.setZero();
       for (int r = r_c; r < ksr; r++) {
          double rr = r - r_c;
          double e = (rr * rr) / (2.0 * sig);
          m(r) = norm * std::exp(-e);
       }
    };
    auto shp_edg1 = [](ColVector& m) { m.setZero(); m(0) = 1.0; };
    auto shp_edg2 = [](ColVector& m) { m.setZero(); m(31) = 1.0; };

    TestFilter1Driver(32, 32, 32, fft1filter, l1filter, shp_pt1, shp_half_gauss);
    */
    char c;
    cout << "hit a key and press Enter";
    cin >> c;
    // Try this
    //system("pause");
    //
    //REVIEW:
    // The test image was tainting the comparisons.  Fixed the test image
    // and now the correlation and convolution algorithms produce same
    // result for a symetric test pattern.
    // Now need to reconcile this with the Kernel Flipping algor

    //TestFilterFunction(32,16,16,"con_old", LinearCorrelate3);
    //TestFilterFunction(32,16,16,"con_new", LinearConvolution3);

    // Combining Signals in the Time or Frequency domain results in the same outcome
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
   const double center = 31.5;
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

   Matrix tt(64, 128);
   tt.setZero();
   for (int row = 0; row < 64; row++) {
      for (int col = 0; col < 64; col++) {
         int cc = 2 * col;

         tt(row, cc) = t(row,col);
      }
   }
   /*
   unsigned long nn[2];
   nn[0] = 64;
   nn[1] = 64;
   fourn(tt.data(), nn, 2, 1);

   Matrix rel(64, 64);
   Matrix img(64, 64);

   for (int row = 0; row < 64; row++) {
      for (int col = 0; col < 64; col++) {
         int cc = 2 * col;

         double a = tt(row,cc);
         double b = tt(row,cc + 1);

         tt(row, cc) = a*a - b*b;
         tt(row, cc+1) = 2.0 * a * b;
      }
   }

   fourn(tt.data(), nn, 2, -1);

   for (int row = 0; row < 64; row++) {
      for (int col = 0; col < 64; col++) {
         int cc = 2 * col;

         t(row, col) = tt(row,cc);
      }
   }

   */
/*
   Matrix h = t;
   Matrix o(64, 64);
   fft2ConCor(t, h, o, -1);
   */
/*
   //LinearCorrelate3(t, h, o);
   Matrix real(32, 32);
   for (int row = 0; row < 32; row++) {
      for (int col = 0; col < 32; col++) {
         double a = o(row, 2 * col);
         double b = o(row, 2 * col + 1);

         real(row, col) = a;
      }
   }
 */  
   //ofstream owf(path + "\\c.csv", ios::trunc);
   //owf << o;
   //owf.close();
   /*
   owf.open(path + "\\real.csv", ios::trunc);
   assert(owf.is_open());
   owf << rel;
   owf.close();

   owf.open(path + "\\imag.csv", ios::trunc);
   owf << img;
   owf.close();
   */
 
    
  /*
   //!!!!!!!!!!!!!!!!!!  2D FFT Test  !!!!!!!!!!!!!!!!!!!
   //unsigned long dims[] = { 64, 64 };
   ColVector speq(2*64);
   rlft2(t.data(), speq, 64, 64, 1);
   double fac = 2.0 / (64 * 64);
   t.array() *= fac;
   speq.array() *= fac;

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
   */
  // exit(0);
 
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
