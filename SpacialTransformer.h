#pragma once
#include <Layer.h>

class IWeightsForStAffine : public iGetWeights {
   double StdDev;
public:

   IWeightsForStAffine(double std_dev) : StdDev(std_dev) {}

   void ReadConvoWeight(Matrix& w, int k) {
      throw runtime_error("Not implemented.  This initializer is for the last layer of a "
                           "localizer network for a Spacial Transformer that uses an Affine transform (6 parameters).");
   }
   void ReadConvoBias(Matrix& w, int k) {
      throw runtime_error("Not implemented.  This initializer is for the last layer of a "
         "localizer network for a Spacial Transformer that uses an Affine transform (6 parameters).");
   }
   void ReadFC(Matrix& w) {
      runtime_assert(w.rows()==6)
      std::random_device rd;
      std::default_random_engine generator(rd());
      std::normal_distribution<double> distribution(0.0, StdDev);
      for (int r = 0; r < w.rows(); r++) {
         for (int c = 0; c < w.cols(); c++) {
            w(r, c) = distribution(generator);
         }
      }
      // The bias is initialized to the Affine Identity transform.
      // Layout of the column vector indexes for the Affine transform:
      //    1        2        5
      //    3        4        6
      //
      Eigen::Index p = w.cols() - 1;
      w(0, p) = 1.0;
      w(1, p) = 0.0;
      w(2, p) = 0.0;
      w(3, p) = 1.0;
      w(4, p) = 0.0;
      w(5, p) = 0.0;
   }
};

class actLinearForStAffine : public iActive {
   int Size;
   ColVector J;
public:
   // Leave input size for compatibility.
   actLinearForStAffine(int size) : Size(6), J(6) {
      J.setOnes();
   }
   actLinearForStAffine() : Size(6), J(6) {
      J.setOnes();
   }
   void Resize(int size) {
      runtime_assert(size == 6);
   }
   ColVector Eval(ColVector& x) {
      // Assumes the layer that uses this activation function initializes
      // bias vector to zero.  We turn the zero vector to Identity transform
      // vector here.
      x(0) += 1.0;
      x(3) += 1.0;

      return x;
   }
   ColVector Eval(Eigen::Map<ColVector>& x) {
      x(0) += 1.0;
      x(3) += 1.0;

      return x;
   }
   Matrix Jacobian(const ColVector& x) {
      return J.asDiagonal();
   }

   int Length() { return Size; }
};

class actTanhForStAffine : public iActive {
   int Size;
   actTanh ATH;
   ColVector T;

   void ComputeT(const ColVector& q)
   {
      T(0) = 0.50 * q(0) + 1.0;
      T(1) = 0.25 * q(1);
      T(2) = 0.25 * q(2);
      T(3) = 0.50 * q(3) + 1.0;
      T(4) = 4.00 * q(4);
      T(5) = 4.00 * q(5);
   }

public:
   actTanhForStAffine(int size) : Size(6), ATH(6), T(6) {}
   actTanhForStAffine() : Size(6), ATH(6), T(6) {}
   void Resize(int size) {
      runtime_assert(size == 6);
   }
   // Z = W x + b.  Z is what is passed into the activation function.
   ColVector Eval(ColVector& q) {
      // Don't mess with q.
      ATH.Eval(q);
      ComputeT(q);

      return T;
   }
   ColVector Eval(Eigen::Map<ColVector>& q) {
      // Don't mess with q.
      ATH.Eval(q);
      ComputeT(q);

      return T;
   }
   Matrix Jacobian(const ColVector& q) {

      return ATH.Jacobian(q);;
   }

   int Length() { return Size; }
};

class actCubeForAffine : public iActive {
   int Size;
   ColVector J;

public:
   actCubeForAffine(int size) : Size(6), J(6) {}
   actCubeForAffine() : Size(0), J(6) {}
   void Resize(int size) {
      runtime_assert(size == 6);
   }
   // Z = W x + b.  Z is what is passed into the activation function.
   ColVector Eval(ColVector& q) {
      // Vector q carries the z vector on the way in, then is trasformed
      // to the values of the special function for the way out.  It will later
      // be passed into the Jacobian method and used to compute the Jacobian.
      // In this case it should simple retain the value of z.

      // Use J for the return.
      J(0) = q(0) * q(0) * q(0) + 1.0;  // q^3 + 1.0
      J(1) = q(1) * q(1) * q(1); // q^3
      J(2) = q(2) * q(2) * q(2); // q^3
      J(3) = q(3) * q(3) * q(3) + 1.0;  // q^3 + 1.0
      J(4) = q(4) * q(4) * q(4); // q^3
      J(5) = q(5) * q(5) * q(5); // q^3

      return J;
   }
   ColVector Eval(Eigen::Map<ColVector>& q) {
      // Use J for the return.
      J(0) = 0.01 * q(0) * q(0) * q(0) + 1.0;  // q^3 + 1.0
      J(1) = 0.01 * q(1) * q(1) * q(1); // q^3
      J(2) = 0.01 * q(2) * q(2) * q(2); // q^3
      J(3) = 0.01 * q(3) * q(3) * q(3) + 1.0;  // q^3 + 1.0
      J(4) = 0.01 * q(4) * q(4) * q(4); // q^3
      J(5) = 0.01 * q(5) * q(5) * q(5); // q^3

      return J;
   }
   Matrix Jacobian(const ColVector& q) {
      //                derivitive of
      J(0) = 0.01 * 3.0 * q(0) * q(0);        // q^3 + 1.0
      J(1) = 0.01 * 3.0 * q(1) * q(1); // q^3
      J(2) = 0.01 * 3.0 * q(2) * q(2); // q^3
      J(3) = 0.01 * 3.0 * q(3) * q(3);        // q^3 + 1.0
      J(4) = 0.01 * 3.0 * q(4) * q(4); // q^3
      J(5) = 0.01 * 3.0 * q(5) * q(5); // q^3

      return J.asDiagonal();
   }

   int Length() { return Size; }
};

class SpacialTransformer
{
public:
   struct Size {
      int cols;
      int rows;
      Size() : cols(0), rows(0) {}
      Size(int r, int c) : cols(c), rows(r) {}
      inline void Resize(int r, int c) { rows = r; cols = c; }
   };

private:
   typedef std::vector < std::vector< std::tuple<int, int, int, int, double, double>>> SampleMatrix;

   Size InputSize;
   Size OutputSize;
   ColVector GridC;
   ColVector GridR;

   SampleMatrix SM;

   // Input Matrix.
   Matrix U;
public:

   SpacialTransformer(const Size input_size, const Size output_size ) :
      InputSize(input_size),
      OutputSize(output_size)
   {
      GridR = ColVector::LinSpaced(OutputSize.rows, -1.0, 1.0) * static_cast<double>(InputSize.rows - 1) / 2.0f;
      GridC = ColVector::LinSpaced(OutputSize.cols, -1.0, 1.0) * static_cast<double>(InputSize.cols - 1) / 2.0f;

      SM.resize(OutputSize.rows, std::vector<std::tuple<int, int, int, int, double, double>>(OutputSize.cols));
   }

   void Eval(const Matrix& UU, Matrix& V, const ColVector& T)
   {
      U = UU;
      V.resize(OutputSize.rows, OutputSize.cols);


      GenerateAffineGrid(T);
      //cout << T.transpose() << endl;
      // Sample the input image
      BiLinearSampler(U, V);
   }

   RowVector BackpropGrid(const Matrix& dV)
   {
      Matrix dLdR(OutputSize.rows, OutputSize.cols);
      Matrix dLdC(OutputSize.rows, OutputSize.cols);
      RowVector dT(6);

      ComputeGridGradients(dV, dLdR, dLdC, SM);

      //cout << dLdR << endl << endl << dLdC << endl;

      int rows = dV.rows();
      int cols = dV.cols();

      double dt0 = 0;
      double dt1 = 0;
      double dt2 = 0;
      double dt3 = 0;
      double dt4 = 0;
      double dt5 = 0;

      for (int r = 0; r < OutputSize.rows; r++) {
         for (int c = 0; c < OutputSize.cols; c++) {
            double dldr = dLdR(r, c);
            double dldc = dLdC(r, c);
            double rr = GridR(r);
            double cc = GridC(c);
            dt0 += cc * dldc;
            dt1 += rr * dldc;
            dt4 +=      dldc;
            dt2 += cc * dldr;
            dt3 += rr * dldr;
            dt5 +=      dldr;
         }
      }

      dT(0) = dt0;
      dT(1) = dt1;
      dT(2) = dt2;
      dT(3) = dt3;
      dT(4) = dt4;
      dT(5) = dt5;

      return dT;
   }

   Matrix BackpropSampler(const Matrix& dV) 
   {
      Matrix dU(InputSize.rows, InputSize.cols);
      dU.setZero();

      for (int r = 0; r < OutputSize.rows; r++) {
         for (int c = 0; c < OutputSize.cols; c++) {
            const tuple<int, int, int, int, double, double>& t = SM[r][c];
            const int rl = std::get<0>(t);
            const int rh = std::get<1>(t);
            const int cl = std::get<2>(t);
            const int ch = std::get<3>(t);
            const double rr = std::get<4>(t);
            const double cc = std::get<5>(t);
            const double dr = rh - rr;
            const double dc = ch - cc;

            Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> corners(2,4);
            // upper left
            corners(0, 0) = rh; corners(1, 0) = cl;
            corners(0, 1) = rh; corners(1, 1) = ch;
            corners(0, 2) = rl; corners(1, 2) = cl;
            corners(0, 3) = rl; corners(1, 3) = ch;

            // Compute the weights for the bilinear interpolation
            ColVector weights(4);
            weights(0) = dr * dc;
            weights(1) = dr * (1.0 - dc);
            weights(2) = (1.0 - dr)  * dc;
            weights(3) = (1.0 - dr) * (1.0 - dc);

            // Compute the gradient for the input patch
            for (int i = 0; i < 4; ++i) {
               if (corners(0, i) < 0 || corners(0, i) >= InputSize.rows ||
                  corners(1, i) < 0 || corners(1, i) >= InputSize.cols) {
                  // The corner is out of bounds, skip it
                  continue;
               }

               dU(corners(0, i), corners(1, i)) += weights(i, 0) * dV(r,c);
            }
         }
      }

      return dU;
   }

private:
   // The Sample Matrix generator for Affine transform.
   void GenerateAffineGrid(const ColVector& T)
   {
      //REVIEW: Do some range checking.
      SampleMatrix& sm = SM;

      // Compute the transform matrix
      Eigen::Affine2d transform = Eigen::Affine2d::Identity();
      transform.linear()(0, 0) = T(0);
      transform.linear()(0, 1) = T(1);
      transform.linear()(1, 0) = T(2);
      transform.linear()(1, 1) = T(3);
      transform.translation()(0) = T(4);
      transform.translation()(1) = T(5);

      // Compute the grid coordinates
 // grid_c is in the range of –1 to 1 * (output_width-1)/2
 // grid_r is in the range of –1 to 1 * (output_height-1)/2
 // The value of grid_r or grid_c is in the range of the input matrix.
 // This feature alone could be used for upsampling or downsampling.

      for (int r = 0; r < OutputSize.rows; ++r)
      {
         for (int c = 0; c < OutputSize.cols; ++c)
         {
            // NOTE: I think that traslation alone could be accomplished in the frequency domain
            //       if the data were 2X padded out, then the inversion issue that I saw
            //       when non-zero data crossed a boundry edge would not occur.

            //const Eigen::Vector2d point = transform * Eigen::Vector2d(GridR(r), GridC(c));
            // The result is {c, r}
            const Eigen::Vector2d point = transform * Eigen::Vector2d(GridC(c), GridR(r));

            // Below is equivilant output.
            //ColVector point(2);
            //point(1) = T(0) * GridC(c) + T(1) * GridR(r) + T(4);
            //point(0) = T(2) * GridC(c) + T(3) * GridR(r) + T(5);

            // The sample grid is centered about zero so the image can (potentially) be rotated about zero,
            // but we have to shift the points back into actual coordinate range before they are stored.
            //
            // Note that there is no range checking.  rr and cc can go outside of the input range.
            // The sampler must be able to handle this.

            double rr = point(1) + static_cast<double>(InputSize.rows - 1) / 2.0;
            double cc = point(0) + static_cast<double>(InputSize.cols - 1) / 2.0;

            int rl = static_cast<int>(floor(rr));
            int rh = static_cast<int>(ceil(rr));// rl + 1;
            int cl = static_cast<int>(floor(cc));
            int ch = static_cast<int>(ceil(cc));// cl + 1;

            sm[r][c] = std::tuple<int, int, int, int, double, double>(rl, rh, cl, ch, rr, cc);
         }
      }
   }

   // REVIEW: There should be a parameter such as "padding"
   //          It tells what the value should be outside the boundry.
#define BndValue( R, C )  (( R >= 0 && R <= (srows - 1) && \
                                C >= 0 && C <= (scols - 1) ) ? m(R,C) : 0.0)

   void BiLinearSampler(const Matrix& m, Matrix& out)
   {
      SampleMatrix& sm = SM;

      int rows = out.rows();
      int cols = out.cols();
      int srows = m.rows();
      int scols = m.cols();

      for (int r = 0; r < rows; r++) {
         for (int c = 0; c < cols; c++) {
            const tuple<int, int, int, int, double, double>& t = sm[r][c];
            const double rr = std::get<4>(t);
            const double cc = std::get<5>(t);

            const int rl = std::get<0>(t);
            const int rh = std::get<1>(t);
            const int cl = std::get<2>(t);
            const int ch = std::get<3>(t);

            const double dr = static_cast<double>(rh) - rr;
            const double dc = static_cast<double>(ch) - cc;

            out(r, c) =   BndValue(rl, cl) * dr * dc 
                        + BndValue(rl, ch) * dr * (1.0 - dc) 
                        + BndValue(rh, cl) * (1.0 - dr) * dc 
                        + BndValue(rh, ch) * (1.0 - dr) * (1.0 - dc);
         }
      }
   }
#undef BiLinearSampler

#define ValueInRange( R, C )  (( R >= 0 && R <= (InputSize.rows - 1) && \
                                C >= 0 && C <= (InputSize.cols - 1) ) ? U(R,C) : 0.0)

   void ComputeGridGradients( const Matrix& dLdV, Matrix& dLdR, Matrix& dLdC, SampleMatrix& sm )
   {
      int rows = dLdV.rows();
      int cols = dLdV.cols();

      // REVIEW: Do some bounds checking.

      for (int r = 0; r < OutputSize.rows; r++) {
         for (int c = 0; c < OutputSize.cols; c++) {

            // The range of the element values of grid_xand grid_y is the range of the 
            // input matrix but are not necessarily whole numbers as there can be a 
            // dissimilar output matrix range that is larger or smaller.

            const tuple<int, int, int, int, double, double>& t = sm[r][c];

            // These values are all in the range of the input matrix U.
            const int rl = std::get<0>(t);
            const int rh = std::get<1>(t);
            const int cl = std::get<2>(t);
            const int ch = std::get<3>(t);
            const double rr = std::get<4>(t);
            const double cc = std::get<5>(t);

            const double dr = static_cast<double>(rh) - rr;
            const double dc = static_cast<double>(ch) - cc;

            double dvdc = (1.0 - dr) * (ValueInRange(rh, ch) - ValueInRange(rh, cl)) + dr * (ValueInRange(rl, ch) - ValueInRange(rl, cl));
            double dvdr = (1.0 - dc) * (ValueInRange(rh, ch) - ValueInRange(rl, ch)) + dc * (ValueInRange(rh, cl) - ValueInRange(rl, cl));
            //if (std::isnan(dvdr) || std::isnan(dvdc)) {
            //   cout << "nan" << endl;
            //}
            dLdR(r, c) = dvdr * dLdV(r, c);
            dLdC(r, c) = dvdc * dLdV(r, c);
         }
      }
   }

#undef ValueInRange
};