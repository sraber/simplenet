#pragma once
#include <Layer.h>

class Filter :
   public iConvoLayer
{
public:
   Size InputSize;
   Size OutputSize;
   Size KernelSize;
   int Channels;

   // Vector of input matrix.  One per channel.
   vector_of_matrix X;

   // Vector of output matrix.  One per channel.
   vector_of_matrix Z;

   // Kernel matrix.  There is one kernel and it is initialized not learned.
   Matrix W;

   enum CallBackID { EvalPreActivation, Backprop1, Backprop2 };
private:
   shared_ptr<iCallBackSink> EvalPostActivationCallBack;
   shared_ptr<iCallBackSink> BackpropCallBack;

public:

   Filter(Size input_size, int input_channels, Size output_size, Size kernel_size ) :
      X(input_channels),
      Z(input_channels),
      W(kernel_size.rows, kernel_size.cols),
      InputSize(input_size),
      OutputSize(output_size),
      Channels(input_channels)
   {
      for (Matrix& m : X) { m.resize(input_size.rows, input_size.cols); }
      for (Matrix& m : Z) { m.resize(output_size.rows, output_size.cols); }

      double t = 1.0 / W.size();
      W.setConstant(t);
   }

   ~Filter() {}


#ifdef FFT
   void LinearConvolution(const Matrix& m, Matrix& h, Matrix& out)
   {
      // -1 = correlate | 1 = convolution
#ifdef CYCLIC
      fft2convolve(m, h, out, 1, false, true, false);
#else
      fft2convolve(m, h, out, 1, true, true, false);
#endif
   }
#else
   double MultiplyReverseBlock(const Matrix& m, int mr, int mc, Matrix& h, int hr, int hc, int size_r, int size_c)
   {
      double sum = 0.0;
      for (int r = 0; r < size_r; r++) {
         for (int c = 0; c < size_c; c++) {
            sum += m(mr - r, mc - c) * h(hr + r, hc + c);
         }
      }
      return sum;
   }

   void LinearConvolution(const Matrix& m, Matrix& h, Matrix& out)
   {
      const int mrows = (int)m.rows();
      const int mcols = (int)m.cols();
      const int hrows = (int)h.rows();
      const int hcols = (int)h.cols();
      const int orows = (int)out.rows();
      const int ocols = (int)out.cols();
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
               out(r, c) = MultiplyReverseBlock(m, m2r, m2c, h, h1r, h1c, shr, shc);
            }
         }
      }
   }
#endif

#ifdef FFT
   void LinearCorrelate(const Matrix& m, Matrix& h, Matrix& out)
   {
#ifdef CYCLIC
      fft2convolve(m, h, out, -1, false, true, false);
#else
      fft2convolve(m, h, out, -1, true, true, false);
#endif
   }
#else
   double MultiplyBlock(Matrix& m, int mr, int mc, Matrix& h, int hr, int hc, int size_r, int size_c)
   {
      double sum = 0.0;
      for (int r = 0; r < size_r; r++) {
         for (int c = 0; c < size_c; c++) {
            sum += m(mr + r, mc + c) * h(hr + r, hc + c);
         }
      }
      return sum;
   }

   double MultiplyBlockWithEigen(Matrix& m, int mr, int mc, Matrix& h, int hr, int hc, int size_r, int size_c)
   {
      double sum = (m.array().block(mr, mc, size_r, size_c) * h.array().block(hr, hc, size_r, size_c)).sum();

      return sum;
   }
   void LinearCorrelate(const Matrix& m, Matrix& h, Matrix& out, double bias = 0.0)
   {
      const int mrows = (int)m.rows();
      const int mcols = (int)m.cols();
      const int hrows = (int)h.rows();
      const int hcols = (int)h.cols();
      const int orows = (int)out.rows();
      const int ocols = (int)out.cols();
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
               out(r, c) = bias;
            }
            else {
               //cout << m1r << "," << m1c << "," << h1r << "," << h1c << "," << shr << "," << shc << "," << endl;
               out(r, c) = bias + MultiplyBlockWithEigen(m, m1r, m1c, h, h1r, h1c, shr, shc);
            }
         }
      }
   }
#endif

   vector_of_matrix Eval(const vector_of_matrix& _x)
   {
      vector_of_matrix::const_iterator is = _x.begin();
      vector_of_matrix::iterator iz = Z.begin();
      for (; is != _x.end(); ++is, ++iz) {
         LinearConvolution(*is, W, *iz);
      }

      if (EvalPostActivationCallBack != nullptr) {
         map<string, CBObj> props;
         int id = EvalPreActivation;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "X", CBObj(_x) });
         props.insert({ "W", CBObj(W) });
         props.insert({ "Z", CBObj(Z) });
         EvalPostActivationCallBack->Properties(props);
      }

      return Z;
   }

   // Figure out how many output gradiens there are.  There will be the same or less out going 
   // than in-comming.  Accumulate in-comming gradients into outgoing.
   // There is input_channels*kernel_number in-comming.
   // There are input_channels out-going.
   vector_of_matrix BackProp(vector_of_matrix& child_grad, bool want_backprop_grad = true)
   {
      // layer_grad corr=> rotated kernel to size of InputSize
      // sum results according to kernels per channel.
      const int incoming_channels = Channels;
      assert(child_grad.size() == incoming_channels);

      vector_of_matrix::const_iterator is = child_grad.begin();
      vector_of_matrix::iterator ix = X.begin();
      for (; is != child_grad.end(); ++is, ++ix) {
         LinearCorrelate(*is, W, *ix);
      }

      if (BackpropCallBack != nullptr) {
         map<string, CBObj> props;
         int id = Backprop2;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "X", CBObj(X) });
         props.insert({ "W", CBObj(W) });
         props.insert({ "Z", CBObj(Z) });
         BackpropCallBack->Properties(props);
      }

      return X;
   }

   void Update(double eta) {}

   void Save(shared_ptr<iPutWeights> _pOut) {}

   void SetEvalPostActivationCallBack(shared_ptr<iCallBackSink> icb) { EvalPostActivationCallBack = icb; }
   void SetBackpropCallBack(shared_ptr<iCallBackSink> icb) { BackpropCallBack = icb; }
};
