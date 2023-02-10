#include <Eigen>
#include <Layer.h>
#include <fftn.h>
#include <fft1.h>

//--------------------------------

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
   const int mrows = (int)m.rows();
   const int mcols = (int)m.cols();
   const int hrows = (int)h.rows();
   const int hcols = (int)h.cols();
   const int orows = (int)out.rows();
   const int ocols = (int)out.cols();

   assert(_Is_pow_2(mrows));
   assert(_Is_pow_2(mcols));
   assert(mrows == hrows);
   assert(mcols == hcols);
   assert(mrows == orows);
   assert(mcols == ocols);

   const int rows = mrows;
   const int cols = mcols;

   ColVector mnqrow(2 * cols);
   ColVector hnqrow(2 * cols);
   ColVector onqrow(2 * cols);

   rlft3(m.data(), mnqrow.data(), 1, rows, cols, 1);
   rlft3(h.data(), hnqrow.data(), 1, rows, cols, 1);

   double fac = 2.0 / (rows * cols);

   const int OROWS = rows >> 1;
   const int OCOLS = cols << 1;

   Eigen::Map<Matrix> mo(m.data(), OROWS, OCOLS);
   Eigen::Map<Matrix> ho(h.data(), OROWS, OCOLS);
   Eigen::Map<Matrix> oo(out.data(), OROWS, OCOLS);

   for (int row = 0; row < OROWS; row++) {
      for (int col = 0; col < OCOLS; col += 2) {
         double a = mo(row, col);
         double b = mo(row, col + 1);
         double c = ho(row, col);
         double d = sign * ho(row, col + 1); // Conjugate

         oo(row, col) = fac * (a * c - d * b);
         oo(row, col + 1) = fac * (c * b + a * d);
      }
   }

   for (int col = 0; col < OCOLS; col += 2) {
      double a = mnqrow(col);
      double b = mnqrow(col + 1);
      double c = hnqrow(col);
      double d = sign * hnqrow(col + 1);  // Conjugate

      onqrow(col) = fac * (a * c - d * b);
      onqrow(col + 1) = fac * (c * b + a * d);
   }

   rlft3(out.data(), onqrow.data(), 1, rows, cols, -1);

   int or2 = orows >> 1;
   int oc2 = ocols >> 1;

   // This is FFT shift.
   Matrix t(or2, oc2);
   t = out.block(or2, oc2, or2, oc2);
   out.block(or2, oc2, or2, oc2) = out.block(0, 0, or2, oc2);
   out.block(0, 0, or2, oc2) = t;

   t = out.block(or2, 0, or2, oc2);
   out.block(or2, 0, or2, oc2) = out.block(0, oc2, or2, oc2);
   out.block(0, oc2, or2, oc2) = t;
}


#define padm 0x0001
#define padh 0x0002
#define pado 0x0004
#define padall (padm | padh | pado)

void fft2convolve(const Matrix& m, const Matrix& h, Matrix& o, int con_cor,
                  bool force_row_pad, bool force_col_pad,
                  bool sum_into_output )
{
   const int mrows = (int)m.rows();
   const int mcols = (int)m.cols();
   const int hrows = (int)h.rows();
   const int hcols = (int)h.cols();
   const int orows = (int)o.rows();
   const int ocols = (int)o.cols();

   //const Matrix* mw = &m;
   //const Matrix* hw = &h;
   Matrix* ow = &o;

   Matrix pm;
   Matrix ph;
   Matrix po;

   unsigned int pad = 0;

   int rows = 0;
   if (rows < mrows) { rows = mrows; }
   if (rows < hrows) { rows = hrows; }
   if (rows < orows) { rows = orows; }

   int cols = 0;
   if (cols < mcols) { cols = mcols; }
   if (cols < hcols) { cols = hcols; }
   if (cols < ocols) { cols = ocols; }

   if (std::_Is_pow_2(rows)) {
      if (force_row_pad) { rows <<= 1; }
   }
   else{ 
      rows = fftn::nearest_power_ceil(rows); pad = padall; 
   }
   if (std::_Is_pow_2(cols)) {
      if (force_col_pad) { cols <<= 1; }
   }
   else{
      cols = fftn::nearest_power_ceil(cols); pad = padall; 
   }

   int cr = rows >> 1; cr--;
   int cc = cols >> 1; cc--;

   //if (mrows != rows || mcols != cols) {
   pad |= padm;
   pm.resize(rows, cols);

   int cmr = mrows >> 1; if (!(mrows % 2)) { cmr--; }
   int cmc = mcols >> 1; if (!(mcols % 2)) { cmc--; }
   int sr = cr - cmr;
   int sc = cc - cmc;

   pm.setZero();
   pm.block(sr, sc, mrows, mcols) = m;
   //mw = &pm;
//}
//if (hrows != rows || hcols != cols) {
   pad |= padh;
   ph.resize(rows, cols);

   int chr = hrows >> 1; if (!(hrows % 2)) { chr--; }
   int chc = hcols >> 1; if (!(hcols % 2)) { chc--; }
   sr = cr - chr;
   sc = cc - chc;

   ph.setZero();
   ph.block(sr, sc, hrows, hcols) = h;
   //hw = &ph;
   //}
   if (orows != rows || ocols != cols) {
      pad |= pado;
      po.resize(rows, cols);
      //po.setZero();
      //po.block(0, 0, orows, ocols) = o;
      ow = &po;
   }

   //fft2ConCor(*mw, *hw, *ow, con_cor);
   fft2ConCor(pm, ph, *ow, con_cor);

   if (pad & pado) {
      int cor = orows >> 1; if (!(orows % 2)) { cor--; }
      int coc = ocols >> 1; if (!(ocols % 2)) { coc--; }
      int sr = cr - cor;
      int sc = cc - coc;
      if (sum_into_output) {
         o += ow->block(sr, sc, orows, ocols);
      }
      else {
         o = ow->block(sr, sc, orows, ocols);
      }
   }
}

#undef padm
#undef padh
#undef pado
#undef padall