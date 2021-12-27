// algors.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <Eigen>
#include <iostream>
#include <iomanip>
#include <MNISTReader.h>
#include <Layer.h>
#include <bmp.h>

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

void LinearCorrelate( Matrix g, Matrix h, Matrix& out )
{
   for (int r = 0; r < out.rows(); r++) {
      for (int c = 0; c < out.cols(); c++) {
         double sum = 0.0;
         for (int rr = 0; rr < h.rows(); rr++) {
            for (int cc = 0; cc < h.cols(); cc++) {
               int gr = r + rr;
               int gc = c + cc;
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

void LinearConvolution( Matrix g, Matrix h, Matrix& out )
{
   int hr2 = h.rows() >> 1;
   int hc2 = h.cols() >> 1;
   for (int r = 0; r < out.rows(); r++) {
      for (int c = 0; c < out.cols(); c++) {
         double sum = 0.0;
         for (int rr = 0; rr < h.rows(); rr++) {
            for (int cc = 0; cc < h.cols(); cc++) {
               int gr = r - rr + hr2; // not sure the signs are right or the order is correct.  rr - r ??
               int gc = c - cc + hc2;
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

   WriteWeightsCSV writer("C:\\projects\\neuralnet\\simplenet\\SNCVMNIST\\unwrapmat.csv");
   writer.Write(w);

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
   WriteWeightsCSV mdc1("C:\\projects\\neuralnet\\simplenet\\SNCVMNIST\\dc1.csv");
   mdc1.Write(dc1);

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

   WriteWeightsCSV mdc2("C:\\projects\\neuralnet\\simplenet\\SNCVMNIST\\dc2.csv");
   mdc2.Write(dc2);

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
      TrasformMNISTtoMatrix(h, dl[ itodl[2] ].x );
      ScaleToOne(h.data(), h.rows() * h.cols());


      Matrix m(28, 28);
      TrasformMNISTtoMatrix(m, dl[ itodl[2] ].x );
      ScaleToOne(m.data(), m.rows() * m.cols());


      Matrix g(56,56);
      g.setZero();
      g.block(14, 14,28,28) = m;

      Matrix o(28,28);
      if (b_convolution) {
         // This shows that rotating the filter 180 deg turns the convolution 
         // into a correlation.  Why might we prefer convolution to correlation?
         // One reason is that convolution is associative and correlation is not.
         // If we have two filters h and d and need to apply both of them to matrix m
         // the following is true:
         //    h * (d * m) = (h * d) * m
         // Not true for correlation.
         //Rotate180(h);
         LinearConvolution(m, h, o);
      }
      else {
         LinearCorrelate(m, h, o);
      }
      ScaleToOne(o.data(), o.rows() * o.cols());

      MakeMNISTImage("C:\\projects\\neuralnet\\simplenet\\algors\\results\\" + name + ".bmp", o);

}

int main()
{
    std::cout << "Hello World!\n";
    TestFilter(true, "convo3");
}

