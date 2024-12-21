#include <Eigen>
#include <Layer.h>
#include <fft1.h>

//--------------------------------
// REVIEW: It all should be in a namespace.

void fft1ConCor(ColVector& m, ColVector& h, ColVector& out, int sign)
{
   runtime_assert(sign == 1 || sign == -1);

   const unsigned int mrows = (int)m.rows();
   const unsigned int hrows = (int)h.rows();
   const unsigned int orows = (int)out.rows();

   // The inputs and ouput must all be a power of 2.  The
   // rows and columns don't need to be the same but must be
   // a power of 2.
   // The inputs and output must all be the same dimensions.
   runtime_assert(_Is_pow_2(mrows));
   runtime_assert(mrows == hrows);
   runtime_assert(mrows == orows);

   const unsigned int rows = mrows;

   rfftsine(m.data(), rows, 1);
   rfftsine(h.data(), rows, 1);

   double fac = 2.0 / rows;
   double w = -sign * 2 * EIGEN_PI  / rows;  // N = rows.

   // The Correlation computed in the frequency domain returns a
   // result that is 1 unit off from the correct result.  It must have
   // to do with something very deep but I can't find it.
   // I resorted to introducing a shift in the opposite direction.
   // To shift the signal in the frequency domain multiply the signal
   // fg by an exponential like so (using Matlab script):
   //   fh = fg.*exp(-i * (2 * pi / N) * S * m);
   // where S is the desired shift in points and m is the frequency 
   // number and N is the number of points.  N/2 is the number of frequency
   // coefficients.

   // Handle real valued first and last.
   // Frequency number n = 0 leaves the DC offset unaltered.
   out(0) = fac * (m(0) * h(0));
   // This is the unaltered origional computation for the Nyquist frequency.
   // The frequency number is N/2.
   // out(1) = fac * (m(1) * h(1));
   //
   // This is the shifted adjustment.
   // If w = 2 * PI  / N and the shift S is -sign and m = N/2
   // then the above leads to:
   //    Q = -sign * 2 * PI * (N/2) / N = -sign * PI, and 
   //    cos( Q ) + i sin( Q ).
   //    sin( PI ) or sin( -PI ) is zero.
   //    cos( PI ) or cos( -PI ) is -1.
   // The result leads to the following modified Nyquist term:
   out(1) = -fac * (m(1) * h(1));

   // Initialize the frequency number to 1.
   int n = 1;
   for (int unsigned row = 2; row < rows; row+=2, n++ ) {
      double a = m(row);
      double b = m(row + 1);
      double c = h(row);
      double d = sign * h(row + 1); // Conjugate

      // This is the normal way to compute correlation or convolution.
      //out(row) = fac * (a * c - d * b);
      //out(row + 1) = fac * (c * b + a * d);

      // Introduce a shift in the result of -sign.
      //
      double o = fac * (a * c - d * b);
      double p = fac * (c * b + a * d);
      // The shift terms.
      double e = cos(w * n);
      double f = sin(w * n);

      // Compute modified result.
      out(row) = e * o + f * p;
      out(row + 1) = e * p - o * f;
   }

   // Transform back to time/spacial domain.
   rfftsine(out.data(), rows, -1);

   // Used for the FFT shift.
   int unsigned or2 = orows >> 1;

   // This is FFT shift.
   ColVector t(or2);
   t = out.block(or2, 0, or2, 1);
   out.block(or2, 0, or2, 1) = out.block(0, 0, or2, 1);
   out.block(0, 0, or2, 1) = t;
}

// REVIEW: 1) Hate the name.
//         2) force pad isn't enough.  Minimum pad is a better way to go.
void fft1convolve(const ColVector& m, const ColVector& h, ColVector& o, int con_cor,
   bool force_row_pad,
   bool sum_into_output)
{
   const unsigned int mrows = (unsigned int)m.rows();
   const unsigned int hrows = (unsigned int)h.rows();
   const unsigned int orows = (unsigned int)o.rows();

   ColVector* ow = &o;

   ColVector po;

   unsigned int rows = 0;
   if (rows < mrows) { rows = mrows; }
   if (rows < hrows) { rows = hrows; }
   if (rows < orows) { rows = orows; }

   if (force_row_pad) { rows <<= 1; }

   if (!std::_Is_pow_2(rows)) {
      rows = nearest_power_ceil(rows);
   }

   // The FFT is done in-place and this function is designed to
   // preserve the value of the input matricies so we must make
   // a copy of them.  The copy may be padded.

   //!!!!!!!!!!!!  WARNING  !!!!!!!!!!!!!!!!!
   //!     Experimental code running     !!!!
   //!   Some vectors copied to begining of padding.


   // Copy the m matrix to temporary matrix pm.
   ColVector pm(rows);
   unsigned int sr = rows - mrows; sr >>= 1;
   //if (mrows != 0 && !(mrows% 2)) { sr--; }
   pm.setZero();
   pm.block(sr, 0, mrows, 1) = m;
   //pm.block(0, 0, mrows, 1) = m;

   // Copy the h matrix to temporary matrix ph.
   ColVector ph(rows);
   sr = rows - hrows; sr >>= 1;
   ph.setZero();
   ph.block(sr, 0, hrows, 1) = h;
   //ph.block(0, 0, hrows, 1) = h;

   bool b_output_padded = false;

   // Allocating a different output matrix if the size
   // of the desired output matrix is different from the
   // required resultant output matrix.
   if (orows != rows || sum_into_output) {
      b_output_padded = true;
      po.resize(rows);
      ow = &po;
   }

   fft1ConCor(pm, ph, *ow, con_cor);

   if (b_output_padded) {
      sr = rows - orows; sr >>= 1;
      if (sum_into_output) {
         o += ow->block(sr, 0, orows, 1);
      }
      else {
         o = ow->block(sr, 0, orows, 1);
         //o = ow->block(0, 0, orows, 1);
      }
   }
}
