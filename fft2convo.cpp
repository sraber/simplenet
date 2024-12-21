#include <Eigen>
#include <Layer.h>
#include <fftn.h>



//--------------------------------
// REVIEW: It all should be in a namespace.
namespace fftn {
// NOTE: See std::bit_floor( T x) implemented in C++20
   unsigned nearest_power_floor(unsigned x) {
      if (std::_Is_pow_2(x)) {
         return x;
      }
      int power = 1;
      while (x >>= 1) power <<= 1;
      return power;
   }

   // NOTE: See std::bit_ceil( T x) implemented in C++20
   unsigned nearest_power_ceil(unsigned x) {
      if (std::_Is_pow_2(x)) {
         return x;
      }
      if (x <= 1) return 1;
      int power = 2;
      x--;
      while (x >>= 1) power <<= 1;
      return power;
   }
}
//---------------------------------

void fft2ConCor(Matrix& m, Matrix& h, Matrix& out, int sign)
{
   runtime_assert(sign == 1 || sign == -1);

   const unsigned int mrows = (int)m.rows();
   const unsigned int mcols = (int)m.cols();
   const unsigned int hrows = (int)h.rows();
   const unsigned int hcols = (int)h.cols();
   const unsigned int orows = (int)out.rows();
   const unsigned int ocols = (int)out.cols();

   // The inputs and ouput must all be a power of 2.  The
   // rows and columns don't need to be the same but must be
   // a power of 2.
   // The inputs and output must all be the same dimensions.
   runtime_assert(_Is_pow_2(mrows));
   runtime_assert(_Is_pow_2(mcols));
   runtime_assert(mrows == hrows);
   runtime_assert(mcols == hcols);
   runtime_assert(mrows == orows);
   runtime_assert(mcols == ocols);

   const unsigned int rows = mrows;
   const unsigned int cols = mcols;
   const unsigned int cols2 = cols >> 1;

   const double DX = -1.0;
   const double DY = -1.0;

   // These vectors hold the coefficient of the Nyquist Frequency
   // for each row.  As said in NR, this algorithm is an "almost"
   // in-place transform and the vectors below provide the additional
   // storage space required.  In this algorithm we are utilizing a 2D
   // FFT and the additional storage is a vector for each input and the
   // resulting output.  If it were a 3D trasform the additional storage
   // would be a plane (Matrix).
   // It is 2 * cols to make room for the complex pairs.
   Matrix mnqrow(rows,2);
   Matrix hnqrow(rows,2);
   Matrix onqrow(rows,2);

   rlft3(m.data(), mnqrow.data(), 1, rows, cols, 1);
   rlft3(h.data(), hnqrow.data(), 1, rows, cols, 1);

   double fac = 2.0 / (rows * cols);

   for (unsigned int row = 0; row < rows; row++) {
      for (unsigned int col = 0; col < cols2; col ++) {
         unsigned int cc = col << 1;
         double a = m(row, cc);
         double b = m(row, cc + 1);
         double c = h(row, cc);
         double d = sign * h(row, cc + 1); // Conjugate

         // Introduce a shift in the result of -sign (built into the value of e and f).
         //
         double o = fac * (a * c - d * b);
         double p = fac * (c * b + a * d);

         // The shift terms.
         double e = cos(sign * 2.0 * EIGEN_PI * (DX * (double)row / (double)rows + DY * (double)col / (double)cols));
         double f = sin(sign * 2.0 * EIGEN_PI * (DX * (double)row / (double)rows + DY * (double)col / (double)cols));

         // Compute modified result.
         out(row, cc) = e * o + f * p;
         out(row, cc + 1) = e * p - f * o;
      }
   }

   for (unsigned int row = 0; row < rows; row++) {
      double a = mnqrow(row, 0);
      double b = mnqrow(row, 1);
      double c = hnqrow(row, 0);
      double d = sign * hnqrow(row, 1);  // Conjugate


      double o = fac * (a * c - d * b);
      double p = fac * (c * b + a * d);

      // Introduce a shift in the result of sign.
      double e = cos(sign * 2.0 * EIGEN_PI * (DX * (double)row / (double)rows + DY * 0.5));
      double f = sin(sign * 2.0 * EIGEN_PI * (DX * (double)row / (double)rows + DY * 0.5));

      // Compute modified result.
      onqrow(row, 0) = e * o + f * p;
      onqrow(row, 1) = e * p - f * o;
   }

   // Transform back to time/spacial domain.
   rlft3(out.data(), onqrow.data(), 1, rows, cols, -1);

   // REVIEW: This shift is mearly a convenience.  If the downstream
   //         code were adjusted to utilize the result in it's natural
   //         state this overhead could be avoided.

   // Used for the FFT shift.
   unsigned int or2 = orows >> 1;
   unsigned int oc2 = ocols >> 1;

   // This is FFT shift.
   Matrix t(or2, oc2);
   t = out.block(or2, oc2, or2, oc2);
   out.block(or2, oc2, or2, oc2) = out.block(0, 0, or2, oc2);
   out.block(0, 0, or2, oc2) = t;

   t = out.block(or2, 0, or2, oc2);
   out.block(or2, 0, or2, oc2) = out.block(0, oc2, or2, oc2);
   out.block(0, oc2, or2, oc2) = t;
}

// REVIEW: 1) Hate the name.
//         2) force pad isn't enough.  Minimum pad is a better way to go.
//
// NOTE: Use force_X_pad to insure linear convolution or correlation.
void fft2convolve( const Matrix& m, const Matrix& h, Matrix& o, int con_cor,
                  bool force_row_pad, bool force_col_pad,
                  bool sum_into_output )
{
   const unsigned int mrows = (unsigned int)m.rows();
   const unsigned int mcols = (unsigned int)m.cols();
   const unsigned int hrows = (unsigned int)h.rows();
   const unsigned int hcols = (unsigned int)h.cols();
   const unsigned int orows = (unsigned int)o.rows();
   const unsigned int ocols = (unsigned int)o.cols();

   Matrix* ow = &o;

   Matrix po;

   int unsigned rows = 0;
   if (rows < mrows) { rows = mrows; }
   if (rows < hrows) { rows = hrows; }
   if (rows < orows) { rows = orows; }

   int unsigned cols = 0;
   if (cols < mcols) { cols = mcols; }
   if (cols < hcols) { cols = hcols; }
   if (cols < ocols) { cols = ocols; }

   if (force_row_pad) { rows <<= 1; }
   if (force_col_pad) { cols <<= 1; }

   if (!std::_Is_pow_2(rows)) {
      rows = fftn::nearest_power_ceil(rows);
   }

   if (!std::_Is_pow_2(cols)) {
      cols = fftn::nearest_power_ceil(cols);
   }

   // The FFT is done in-place and this function is designed to
   // preserve the value of the input matricies so we must make
   // a copy of them.  The copy may be padded.

   // Copy the m matrix to temporary matrix pm.
   Matrix pm(rows, cols);
   unsigned int sr = rows - mrows; sr >>= 1;
   unsigned int sc = cols - mcols; sc >>= 1;
   pm.setZero();
   pm.block(sr, sc, mrows, mcols) = m;

   // Copy the h matrix to temporary matrix ph.
   Matrix ph(rows, cols);
   sr = rows - hrows; sr >>= 1;
   sc = cols - hcols; sc >>= 1;
   ph.setZero();
   ph.block(sr, sc, hrows, hcols) = h;

   bool b_output_padded = false;

   // Allocating a different output matrix if the size
   // of the desired output matrix is different from the
   // required resultant output matrix.
   if (orows != rows || ocols != cols || sum_into_output) {
      b_output_padded = true;
      po.resize(rows, cols);
      ow = &po;
   }

   fft2ConCor(pm, ph, *ow, con_cor);

   if (b_output_padded) {
      sr = rows - orows; sr >>= 1;
      sc = cols - ocols; sc >>= 1;
      if (sum_into_output) {
         o += ow->block(sr, sc, orows, ocols);
      }
      else {
         o = ow->block(sr, sc, orows, ocols);
      }
   }
}
