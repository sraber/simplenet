#include <Eigen>
#include <Layer.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <LogPolar.h>

LogPolarSupportMatrix PrecomputeLogPolarSupportMatrix(int in_rows, int in_cols, int out_rows, int out_cols, LogPolarSupportMatrixCenter* pcenter)
{
   LogPolarSupportMatrix lpsm;
   lpsm.resize(out_rows, std::vector<std::tuple<int, int, int, int, double, double>>(out_cols));

   //                   if in_rows      odd              even
   double r_center = in_rows % 2 ? in_rows >> 1 : (in_rows >> 1) - 0.5;
   double c_center = in_cols % 2 ? in_cols >> 1 : (in_cols >> 1) - 0.5;

   // NOTE: The radius will fit in the smaller of the two dimentions.
   //       if the shape is rectangular then some of the shape will not be sampled.
   //       To do it the other way the code must check for less than zero and greater then rows or cols
   double rad_max = r_center < c_center ? r_center : c_center;
   const double da = 2.0 * M_PI / (double)out_rows; // Here, we don't want to reach 2 PI

   if (pcenter != NULL) {
      pcenter->row_center = r_center;
      pcenter->col_center = c_center;
      pcenter->rad_max = rad_max;
   }

   // at x = 0, y = log(0.5)
   // at x = rows - 1, y = log(rad)
   // dy / dx = (log(rad) - log(0.5)) / rows-1
   // b = log(0.5)
   // NOTE: It doesn't matter what log base is used.  The sample points on the linear scale come out the same.
   //       Different log bases give different scale curves but of the same shape and it is this shape that
   //       is mapped to the range rad_max.  When the log scale points are trasformed to linear positions
   //       the sample positions are the same no matter the base of the log.
   const double dp = (std::log(rad_max) - std::log(0.5)) / (double)(out_cols - 1); // Here we do want to reach log(rad_max).
   const double  b = std::log(0.5);

   ColVector rd(out_cols);
   for (int c = 0; c < out_cols; c++) {
      double p = (double)c * dp + b;
      rd(c) = exp(p);
   }

   for (int r = 0; r < out_rows; r++) {
      double a = (double)r * da; // Angle
      for (int c = 0; c < out_cols; c++) {
         // REVIEW: Could use trig recursion formula to speed 
         //         sin and cos computations.
         double rr = -rd(c) * sin(a) + r_center;
         double cc = rd(c) * cos(a) + c_center;
         //cout << p << "," << a << "," << x << "," << y << endl;

         //--------------------------------------------------
         // NOTE: Moved range check to sampler.
         //
         //if (rr < 0.0) { rr = 0.0; }
         //if (cc < 0.0) { cc = 0.0; }
         //
         //if (rr > (in_rows - 1)) { rr = in_rows - 1; }
         //if (cc > (in_cols - 1)) { cc = in_cols - 1; }
         //
         //--------------------------------------------------

         //runtime_assert(rr >= 0.0 && cc >= 0.0);

         int rl = static_cast<int>(floor(rr));
         int rh = static_cast<int>(ceil(rr));
         int cl = static_cast<int>(floor(cc));
         int ch = static_cast<int>(ceil(cc));

         double rp = rr - (double)rl;
         double cp = cc - (double)cl;

         lpsm[r][c] = std::tuple<int, int, int, int, double, double>(rl, rh, cl, ch, rp, cp);
      }
   }
   return lpsm;
}


LogPolarSupportMatrix PrecomputeLogPolarSupportMatrix1(int in_rows, int in_cols, int out_rows, int out_cols, LogPolarSupportMatrixCenter* pcenter)
{
   LogPolarSupportMatrix lpsm;
   lpsm.resize(out_rows, std::vector<std::tuple<int, int, int, int, double, double>>(out_cols));

   //                   if in_rows      odd              even
   double r_center = in_rows % 2 ? in_rows >> 1 : (in_rows >> 1) - 0.5;
   double c_center = in_cols % 2 ? in_cols >> 1 : (in_cols >> 1) - 0.5;

   // NOTE: The radius will fit in the smaller of the two dimentions.
   //       if the shape is rectangular then some of the shape will not be sampled.
   //       To do it the other way the code must check for less than zero and greater then rows or cols
   double rad_max = r_center < c_center ? r_center : c_center;

   if (pcenter != NULL) {
      pcenter->row_center = r_center;
      pcenter->col_center = c_center;
      pcenter->rad_max = rad_max;
   }

   // For angle we want to stop one bin short of full circle to make the transform perfectly circular.
   ColVector cs = ColVector::LinSpaced(out_rows, 0.0, static_cast<double>(out_rows - 1)) * (2.0 * M_PI / (out_rows));
   ColVector sn(out_rows);
   ColVector rd = ColVector::LinSpaced(out_cols, 0.0, static_cast<double>(out_cols - 1)) * (1.0 / (out_cols - 1));

   // r ^ (x/W)
   const double a = 0.5;
   for (int c = 0; c < out_cols; c++) {
      rd(c) = std::pow(a, (1.0 - rd(c))) * std::pow(rad_max, rd(c));
   }

   for (int r = 0; r < out_rows; r++) {
      double v = cs(r);
      cs(r) = cos(v);
      sn(r) = sin(v);
   }

   for (int r = 0; r < out_rows; r++) {
      for (int c = 0; c < out_cols; c++) {
         // REVIEW:  was a (-) in front of rd.
         double rr = -rd(c) * sn(r) + r_center;
         double cc = rd(c) * cs(r) + c_center;

         //--------------------------------------------------
         // NOTE: Moved range check to sampler.
         //
         //if (rr < 0.0) { rr = 0.0; }
         //if (cc < 0.0) { cc = 0.0; }
         //
         //if (rr > (in_rows - 1)) { rr = in_rows - 1; }
         //if (cc > (in_cols - 1)) { cc = in_cols - 1; }
         //
         //--------------------------------------------------

         //runtime_assert(rr >= 0.0 && cc >= 0.0);

         int rl = static_cast<int>(floor(rr));
         int rh = static_cast<int>(ceil(rr));
         int cl = static_cast<int>(floor(cc));
         int ch = static_cast<int>(ceil(cc));

         double rp = rr - (double)rl;
         double cp = cc - (double)cl;

         lpsm[r][c] = std::tuple<int, int, int, int, double, double>(rl, rh, cl, ch, rp, cp);
      }
   }
   return lpsm;
}

#define BndValue( R, C )  (( R >= 0 && R <= (srows - 1) && \
                                C >= 0 && C <= (scols - 1) ) ? m(R,C) : 0.0)

void ConvertToLogPolar(Matrix& m, Matrix& out, LogPolarSupportMatrix& lpsm)
{
   const int rows = out.rows();
   const int cols = out.cols();
   const int srows = m.rows();
   const int scols = m.cols();

   for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
         const tuple<int, int, int, int, double, double>& t = lpsm[r][c];
         const int rl = std::get<0>(t);
         const int rh = std::get<1>(t);
         const int cl = std::get<2>(t);
         const int ch = std::get<3>(t);
         const double rr = std::get<4>(t);
         const double cc = std::get<5>(t);

         // This Bilinear interpolation formula uses
         // rr = rp - rl  where rp,cp is interpolation point.  (variable meanings are reversed above) 
         // cc = cp - cl

         double a00 = BndValue(rl, cl);
         double a10 = BndValue(rh, cl) - a00;
         double a01 = BndValue(rl, ch) - a00;
         double a11 = BndValue(rh, ch) - a00 - a10 - a01;
         out(r, c) = a00 + a10 * rr + a01 * cc + a11 * rr * cc;

         // The interpolation used by SpacialTrasnformer expects rr,cc to be the interpolation point and
         // uses dr and dc.
         //const double dr = static_cast<double>(rh) - rr;
         //const double dc = static_cast<double>(ch) - cc;

         //out(r, c) = m(rl, cl) * dr * dc + m(rl, ch) * dr * (1.0 - dc) + m(rh, cl) * (1.0 - dr) * dc + m(rh, ch) * (1.0 - dr) * (1.0 - dc);
      }
   }
}

#undef BndValue

void ConvertLogPolarToCart(const Matrix& m, Matrix& out, LogPolarSupportMatrixCenter lpsmc)
{
   int out_rows = out.rows();
   int out_cols = out.cols();
   int in_rows = m.rows();
   int in_cols = m.cols();

   double row_cen = lpsmc.row_center;
   double col_cen = lpsmc.col_center;
   double p_max = lpsmc.rad_max;

   const double da = 2.0 * M_PI / (double)in_rows; // Here, we don't want to reach 2 PI
   // at x = 0, y = log(0.5)
   // at x = rows - 1, y = log(rad)
   // dy / dx = (log(rad) - log(0.5)) / rows-1
   // b = log(0.5)
   const double dp = (std::log(p_max) - std::log(0.5)) / (double)(in_cols - 1); // Here we do want to reach log(dia).
   const double  b = std::log(0.5);

   out.setZero();

   for (int r = 0; r < out_rows; r++) {
      //double rr = r - row_cen - 1;
      double rr = row_cen - r;
      for (int c = 0; c < out_cols; c++) {
         double cc = c - col_cen;
         double p = std::sqrt(rr * rr + cc * cc);
         if (p < 1) {
            p = 0.5;
         }
         double a = atan2(rr, cc);
         if (a < 0) {
            a = 2.0 * M_PI + a;
         }

         double in_r = a / da;
         // this is how radius was computed going from
         // cartesian to LP
         // rad = (double)c * dp + b;
         //
         // Now, given rad (p) find c.
         double in_c = (std::log(p) - b) / dp;

         //cout << p << "," << a << " | " << in_c << "," << in_r << endl;

         int rl = floor(in_r);     if (rl >= in_rows) { rl = in_rows - 1; }
         int rh = ceil(in_r);      if (rh >= in_rows) { rh = in_rows - 1; }
         int cl = floor(in_c);     if (cl >= in_cols) { cl = in_cols - 1; }
         int ch = ceil(in_c);      if (ch >= in_cols) { ch = in_cols - 1; }

         double rp = in_r - (double)rl;
         double cp = in_c - (double)cl;

         double a00 = m(rl, cl);
         double a10 = m(rh, cl) - a00;
         double a01 = m(rl, ch) - a00;
         double a11 = m(rh, ch) - a00 - a10 - a01;

         out(r, c) = a00 + a10 * rp + a01 * cp + a11 * rp * cp;
      }
   }

}