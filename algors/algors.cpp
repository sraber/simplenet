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

void LinearConvolution( Matrix g, Matrix h, Matrix& out )
{
   int gr2 = g.rows() >> 1; if (!(gr2 % 2)) { gr2--; }
   int gc2 = g.cols() >> 1; if (!(gc2 % 2)) { gc2--; }
   int hr2 = h.rows() >> 1; if (!(hr2 % 2)) { hr2--; }
   int hc2 = h.cols() >> 1; if (!(hc2 % 2)) { hc2--; }
   int or2 = out.rows() >> 1; if (!(or2 % 2)) { or2--; }
   int oc2 = out.cols() >> 1; if (!(oc2 % 2)) { oc2--; }

   for (int r = 0; r < out.rows(); r++) {     // 0 to rows --> -On/2+1 to On/2 if On is even, else -On/2 to On/2.
      for (int c = 0; c < out.cols(); c++) {
         double sum = 0.0;
         for (int rr = 0; rr < h.rows(); rr++) {      // 0 to rows --> -N/2 to N/2 
            for (int cc = 0; cc < h.cols(); cc++) {   // 0 to cols --> -M to M --> odd numbered kernels work well becuse they are 2*M+1.
               int gr = r - rr + gr2 + hr2 - or2;                 // r and c are the correlation plane
               int gc = c - cc + gc2 + hc2 - oc2;                 // coordinates.
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

//void LinearCorrelate( Matrix& g, Matrix& h, Matrix& out )
//{
//   for (int r = 0; r < out.rows(); r++) {     // 0 to rows --> -On to On
//      for (int c = 0; c < out.cols(); c++) {
//         double sum = 0.0;
//         for (int rr = 0; rr < h.rows(); rr++) {      // 0 to rows --> -N to N 
//            for (int cc = 0; cc < h.cols(); cc++) {   // 0 to cols --> -M to M --> odd numbered kernels work well becuse they are 2*M+1.
//               int gr = r + rr;                       // r and c are the correlation plane
//               int gc = c + cc;                       // coordinates.
//               if (gr >= 0 && gr < g.rows() && 
//                     gc >= 0 && gc < g.cols()) {
//                  sum += g(gr, gc) * h(rr, cc);
//               }
//            }
//         }
//         out(r, c) = sum;
//      }
//   }
//}

void LinearCorrelate( Matrix& g, Matrix& h, Matrix& out )
{
   int gr2 = g.rows() >> 1; if (!(gr2 % 2)) { gr2--; }
   int gc2 = g.cols() >> 1; if (!(gc2 % 2)) { gc2--; }
   int hr2 = h.rows() >> 1; if (!(hr2 % 2)) { hr2--; }
   int hc2 = h.cols() >> 1; if (!(hc2 % 2)) { hc2--; }
   int or2 = out.rows() >> 1; if (!(or2 % 2)) { or2--; }
   int oc2 = out.cols() >> 1; if (!(oc2 % 2)) { oc2--; }

   for (int r = 0; r < out.rows(); r++) {     // 0 to rows --> -On to On
      for (int c = 0; c < out.cols(); c++) {
         double sum = 0.0;
         for (int rr = 0; rr < h.rows(); rr++) {      // 0 to rows --> -N to N 
            for (int cc = 0; cc < h.cols(); cc++) {   // 0 to cols --> -M to M --> odd numbered kernels work well becuse they are 2*M+1.
               int gr = r + rr + gr2 - hr2 - or2;                 // r and c are the correlation plane
               int gc = c + cc + gc2 - hc2 - oc2;                 // coordinates.
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
   const int hrows = h.rows();
   const int hcols = h.cols();
   const int mrows = m.rows();
   const int mcols = m.cols();
   int mr2 = mrows >> 1; if (!(mr2 % 2)) { mr2--; }
   int mc2 = mcols >> 1; if (!(mc2 % 2)) { mc2--; }
   int hr2 = hrows >> 1; if (!(hr2 % 2)) { hr2--; }
   int hc2 = hcols >> 1; if (!(hc2 % 2)) { hc2--; }
   int or2 = out.rows() >> 1; if (!(or2 % 2)) { or2--; }
   int oc2 = out.cols() >> 1; if (!(oc2 % 2)) { oc2--; }

   for (int r = 0; r < out.rows(); r++) {     // Scan through the Correlation surface.
      for (int c = 0; c < out.cols(); c++) {
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
            if (m1r + shr > mrows) {
               shr = mrows - m1r;
            }
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
            if (m1c + shc > mcols) {
               shc = mcols - m1c;
            }
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
   const int kn = 7;
   // Padding of input matrix.
   //const int p = (int)floor((double)kn/2.0);
   const int p = 6; // 14;
   // Convolution / back propagation delta size.
   // cn max = n + p - kn
   const int cn = 29; // 29;
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
   g.setOnes();
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
         // the entire target (image) surface.
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
                  int wc = gc * n + gr;
                  int wr = c * cn + r;
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

   LinearCorrelate(gp, k, dc2);

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

void TestFilter(bool b_convolution, string name)
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

      Matrix h(28, 28);
      TrasformMNISTtoMatrix(h, dl[ itodl[4] ].x );
      ScaleToOne(h.data(), h.rows() * h.cols());

      Matrix h1(7, 7);
      h1 = h.block(14, 14, 7, 7);

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

      Matrix o(64,64);
      if (b_convolution) {
         // This shows that rotating the filter 180 deg turns the convolution 
         // into a correlation.  Why might we prefer convolution to correlation?
         // One reason is that convolution is associative and correlation is not.
         // If we have two filters h and d and need to apply both of them to matrix m
         // the following is true:
         //    h * (d * m) = (h * d) * m
         // Not true for correlation.
         //Rotate180(h);
         LinearConvolution(g, s, o);
         //LinearConvolution(g, sx, o);
         //LinearConvolution(g, sy, o);
      }
      else {
         //LinearCorrelate(g, h1, o);
         LinearCorrelate3(m, h, o);
         //LinearCorrelate2(14, 14, m, h, o);
         //clock_t start, end;
         //start=clock();
         //for (int i = 0; i < 5000; i++) {
         //   LinearCorrelate(g, h, o);
         //}
         //end=clock();
         //cout << "LinCor:" << (double)(end - start) / double(CLOCKS_PER_SEC) << endl;

         //start=clock();
         //for (int i = 0; i < 5000; i++) {
         //   LinearCorrelate1(14, 14, m, h, o);
         //}
         //end=clock();
         //cout << "LinCor1:" << (double)(end - start) / double(CLOCKS_PER_SEC) << endl;

         //start=clock();
         //for (int i = 0; i < 5000; i++) {
         //   LinearCorrelate2(14, 14, m, h, o);
         //}
         //end=clock();
         //cout << "LinCor2:" << (double)(end - start) / double(CLOCKS_PER_SEC) << endl;

         //char c;
         //cin >> c;
         //exit(0);
      }
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

int main()
{
    std::cout << "Hello World!\n";
    //TestFilter(false , "cor3");

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
