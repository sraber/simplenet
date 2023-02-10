// SNCVMNIST.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#define LOGPOLAR
#ifdef LOGPOLAR
   const int INPUT_ROWS = 32;
   const int INPUT_COLS = 32;
#else
   const int INPUT_ROWS = 28;
   const int INPUT_COLS = 28;
#endif
#define FFT
#define CYCLIC
//#define RETRY 1
#define SGD

// Use MOMENTUM 0.0 for grad comp testing.
#define MOMENTUM 0.6

#include <Eigen>
#include <iostream>
#include <iomanip>
#include <MNISTReader.h>
#include <Layer.h>
#include <bmp.h>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define _USE_MATH_DEFINES
#include <math.h>
#include <fft2convo.h>
#include <fft1.h>

#include <map>

#include <utility.h>
//#include <Windows.h>

typedef vector< shared_ptr<Layer> > layer_list;
typedef vector< shared_ptr<iConvoLayer> > convo_layer_list;
convo_layer_list ConvoLayerList;
layer_list LayerList;

shared_ptr<iLossLayer> loss;
string path = "C:\\projects\\neuralnet\\simplenet\\SNCVMNIST\\weights";
string model_name = "layer";

class myAveFlattenCallBack : public iCallBackSink
{
   Matrix avg;
   bool bFirst = true;
   iColVector counts;
   string file_path;  // Need this becuse the global is gone by the time destructor is called.
public:
   static int label;
   myAveFlattenCallBack( string fp ) : counts(10), file_path(fp) {
      counts.setConstant(1);
   }
   ~myAveFlattenCallBack() {
      OWeightsCSVFile o(file_path, "avg");
      o.Write(avg, 0);
   }
   // Inherited via iCallBackSink
   void Properties(std::map<string, CallBackObj>& props) override
   {
      const vector_of_matrix& z = props["Z"].vm.get();
      if (bFirst) {
         bFirst = false;
         int rows = z[0].rows();
         avg.resize(rows, 10);
         avg.setZero();
      }
      // REVIEW: One problem now is that all patterns are averaged, even ones that don't lead to
      //         a correct answer.  Hard to stop this using this debugging mechanism becuse here
      //         we do not yet know if the answer is correct or not.
      double a = 1.0 / counts[label];
      double b = 1.0 - a;
      counts[label] += 1;

      avg.col(label) = a * z[0].col(0) + b * avg.col(label);
   }
};
int myAveFlattenCallBack::label = 0;
shared_ptr<iCallBackSink> ACB = make_shared<myAveFlattenCallBack>(path);

class myVMCallBack : public iCallBackSink
{
   string w;
public:
   myVMCallBack(string what) { w = what; }
   // Inherited via iCallBackSink
   void Properties(std::map<string, CallBackObj>& props) override
   {
      const vector_of_matrix& z = props[w].vm.get();
      cout << w << ":" << endl;
      int count = 0;
      for (auto m : z) {
         cout << count++ << ":\n" << m << endl;
      }
      // For examining the Jacobian of the Spectral Pool.
      //cout << "Norm J:" << endl << props["J"].m.get() << endl;
   }
};

class myMCallBack : public iCallBackSink
{
   string w;
public:
   myMCallBack(string what) { w = what; }
   // Inherited via iCallBackSink
   void Properties(std::map<string, CallBackObj>& props) override
   {
      const Matrix& z = props[w].m.get();
      cout << w << ":" << endl;

         cout << z << endl;

      // For examining the Jacobian of the Spectral Pool.
      //cout << "Norm J:" << endl << props["J"].m.get() << endl;
   }
};

shared_ptr<iCallBackSink> MCB;// = make_shared<myVMCallBack>("X");
shared_ptr<iCallBackSink> MCB1;// = make_shared<myVMCallBack>("Z");
shared_ptr<iCallBackSink> MCB2;// = make_shared<myMCallBack>("J");

// ---------------------------------------------------
// Special layer to remove mean.
//

class RemMeanLayer2D : 
   public iConvoLayer {
public:
   Size InputSize;
   Size KernelSize;
   int KernelPerChannel;
   int Channels;
   // Vector of input matrix.  One per channel.
   vector_of_matrix X;

   // Vector of activation complementary matrix.  There are KernelPerChannel * input channels.
   // The values may be the convolution prior to activation or something else.  The activation
   // objects use the fly-weight pattern and this is the storage for that.
   vector_of_matrix Z;
   unique_ptr<iActive> pActive;

private:
   shared_ptr<iCallBackSink> EvalPreActivationCallBack;
   shared_ptr<iCallBackSink> EvalPostActivationCallBack;
   shared_ptr<iCallBackSink> BackpropCallBack;

   inline void callback(shared_ptr<iCallBackSink> icb, int id)
   {
      if (icb != nullptr) {
         map<string, CBObj> props;   
         props.insert({ "ID", CBObj(id) });
         props.insert({ "X", CBObj(X) });
         props.insert({ "Z", CBObj(Z) });
         EvalPreActivationCallBack->Properties( props );
      }
   }

public:

   RemMeanLayer2D(Size input_size, int input_channels, unique_ptr<iActive> _pActive ) :
      X(input_channels), 
      Z(input_channels),

      pActive(move(_pActive)),
      InputSize(input_size),
      Channels(input_channels)
   {
      pActive->Resize(input_size.rows * input_size.cols); 

      for (Matrix& m : X) { 
         m.resize(input_size.rows, input_size.cols ); 
         m.setZero();
      }

      for (Matrix& m : Z) { m.resize(input_size.rows,input_size.cols); }
   }

   ~RemMeanLayer2D() {}

   vector_of_matrix Eval(const vector_of_matrix& _x) 
   {
      {
         vector_of_matrix::const_iterator _ix = _x.begin();
         vector_of_matrix::iterator ix = X.begin();
         vector_of_matrix::iterator iz = Z.begin();
         for (; ix != X.end(); ++ix, ++_ix, iz++) {
            // Copy from matrix of source (_x) to matrix of target (X).
            *ix = *_ix;
            double avg = ix->array().mean();
            *iz = ix->array() - avg;
         }
      }

      vector_of_matrix vecOut(Z.size());

      callback(EvalPreActivationCallBack, 1);

      for (Matrix& mm : vecOut) { mm.resize(InputSize.rows, InputSize.cols); }
      vector_of_matrix::iterator iz = Z.begin();
      vector_of_matrix::iterator iv = vecOut.begin();
      for (; iz != Z.end(); ++iz, ++iv) {
         Eigen::Map<ColVector> z(iz->data(), iz->size());
         Eigen::Map<ColVector> v(iv->data(), iv->size());
         v = pActive->Eval(z);
      }

      callback(EvalPostActivationCallBack, 2);

      return vecOut;
   }

   // Figure out how many output gradiens there are.  There will be the same or less out going 
   // than in-comming.  Accumulate in-comming gradients into outgoing.
   // There is input_channels*kernel_number in-comming.
   // There are input_channels out-going.
   vector_of_matrix BackProp(vector_of_matrix& child_grad, bool want_backprop_grad = true ) 
   {
      assert(child_grad.size() == Channels);

      // child_grad * Jacobian is stored in m_delta_grad.  The computation is made on
      // a row vector map onto m_delta_grad.
      Matrix m_delta_grad(InputSize.rows, InputSize.cols);
      Eigen::Map<RowVector> rv_delta_grad(m_delta_grad.data(), m_delta_grad.size());

      // Allocate the vector of matrix for the return.
      vector_of_matrix vm_backprop_grad(Channels);

      for (Matrix& mm : vm_backprop_grad) { 
         mm.resize(InputSize.rows, InputSize.cols); 
      }

      int i = 0;
      for (Matrix& mm : child_grad ) {
         Eigen::Map<RowVector> rv_child_grad(mm.data(), mm.size());
         Eigen::Map<ColVector> cv_z(Z[i].data(), Z[i].size());
         rv_delta_grad = rv_child_grad * pActive->Jacobian(cv_z);

         double grad_avg = m_delta_grad.array().mean();
         vm_backprop_grad[i] = m_delta_grad.array() - grad_avg;

         i++;
      }
      callback(BackpropCallBack,3);
      return vm_backprop_grad;
   }

   void Update(double eta) {}

   void Save(shared_ptr<iPutWeights> _pOut) {}

   void SetEvalPreActivationCallBack(shared_ptr<iCallBackSink> icb) {  EvalPreActivationCallBack = icb; }
   void SetEvalPostActivationCallBack(shared_ptr<iCallBackSink> icb) {  EvalPostActivationCallBack = icb; }
   void SetBackpropCallBack(shared_ptr<iCallBackSink> icb) {  BackpropCallBack = icb; }
};
//----------------------------------------------------

//----------------------------------------------------
// Spectral XForm Layer
//
class poolColSpec : public iConvoLayer {
   shared_ptr<iCallBackSink> EvalPostActivationCallBack;
   shared_ptr<iCallBackSink> BackpropCallBack;
   shared_ptr<iCallBackSink> JacobianCallBack;

   Size InputSize;
   Size OutputSize;
   int Channels;
   // Vector of input matrix.  One per channel.
   vector_of_matrix F;
   vector_of_matrix X;

   // Used for output.
   vector_of_matrix Z;

   // Work.
   Matrix J;
public:
   poolColSpec(Size input_size, int input_channels ) :
      F(input_channels),
      X(input_channels),
      Z(input_channels),
      Channels(input_channels)
   {
      runtime_assert(_Is_pow_2(input_size.rows))
      // This restriction could be relaxed with the proper provisions.
      runtime_assert( input_size.cols==1 )

      InputSize = input_size;
      OutputSize.rows = input_size.rows >> 1;
      OutputSize.cols = input_size.cols;

      J.resize(InputSize.rows, OutputSize.rows);

      for (Matrix& m : F) { m.resize(input_size.rows, input_size.cols); }
      for (Matrix& m : X) { m.resize(input_size.rows, input_size.cols); }
      for (Matrix& m : Z) { m.resize(OutputSize.rows, OutputSize.cols); }
   }
   ~poolColSpec() {
   }

   void ColSpecPool(Matrix& x, Matrix& out)
   {
      int in_rows = x.rows();
      int out_rows = out.rows();

      // out_rows = in_rows/2
      runtime_assert( out_rows==(in_rows>>1) )
      /*
      // This is ugly.  The Matrix is stored Row major to conform with
      // native C matricies.  If there are multiple columns in the matrix
      // this code wants to do 1D transforms on each column which cuts across
      // the rows.  Hows that work?  Eigen would have to return a copy of the
      // slice to do this correctly.  Yuck!
      for (int col = 0; col < x.cols(); col++) {
         rfftsine(x.col(col).data(), x.cols(), 1);
      }
      */

      // For now enforce 1 column.
      runtime_assert( x.cols()==1 )

      // NOTE: If the above constraint is removed then the
      //       input parameters below need to change.
      rfftsine(x.data(), in_rows, 1);
      x.array() /= (double)out_rows;  // !!!!!          REVIEW:  in_rows or out_rows...    not sure

      // Real valued 1st and last complex pair.
      out(0, 0) = fabs(x(0, 0));
      //out(out_rows - 1, 0) = fabs(x(1, 0));
      //for (int c = 1; c < out_rows - 1; c++) {
      // Ignore Nyquist frequency.
      for (int c = 1; c < out_rows; c++) {
         int p = c << 1;
         double r = x(p, 0);
         double i = x(p + 1, 0);
         out(c, 0) = sqrt(r * r + i * i);
      }

   }

   // The xout parameter is an in/out parameter.
   void BackPool(Matrix& xout, Matrix& f, Matrix& z, const Matrix& dw)
   {
      runtime_assert(xout.cols() == 1);
      runtime_assert(dw.cols() == 1);
      runtime_assert(xout.rows() == InputSize.rows);
      runtime_assert(dw.rows() == OutputSize.rows);
      const double Q = 2.0 * M_PI / InputSize.rows;
      /*
      {
         int r = 0;
         for (int c = 0; c < J.cols(); c++) {
            int k = r; // k the number of input values
            int n = c; // n frequency components
            // double w = Q * k * n; // k is zero so cos(w) is 1.

            J(r, c) = f(2 * n, 0) / (z(n, 0) + std::numeric_limits<double>::min() );
         }
      }
      {
         int r = J.rows() - 1;
         for (int c = 0; c < J.cols(); c++) {
            int k = r; // k the number of input values
            int n = c; // n frequency components
            double w = Q * k * n;

            J(r, c) = f(2 * n, 0) * cos(w) / (z(n, 0) + std::numeric_limits<double>::min() );
         }
      }
      for (int r = 1; r < (J.rows() - 1); r++) {
         for (int c = 0; c < J.cols(); c++) {
            int k = r; // k the number of input values
            int n = c; // n frequency components
            double w = Q * k * n - f(2 * n + 1, 0);

            J(r, c) = f(2 * n, 0) * cos(w) / (z(n,0) + std::numeric_limits<double>::min() );
         }
      }
      */

/*

      {
         int c = 0;
         for (int r = 0; r < J.rows(); r++) {
            int k = r; // k the number of input values
            int n = c; // n frequency components
            // double w = Q * k * n; // n is zero so cos(w) is 1.

            // f uses fft of real function compact packing.  Real valued 1st and last are
            // packed into the 0 and 1 index.
            J(r, c) = 1.0 / InputSize.rows;
         }
      }
    
      {
         int c = J.cols() - 1;
         for (int r = 0; r < J.rows(); r++) {
            int k = r; // k the number of input values
            //int n = c; // n frequency components
            //double w = Q * k * n;

            // f uses fft of real function compact packing.  Real valued 1st and last are
            // packed into the 0 and 1 index.
            J(r, c) = ((k % 2) ? 1.0 : -1.0) / InputSize.rows;
         }
      }
 
      // Ignore Nyquist frequency.
      for (int c = 1; c < J.cols(); c++) {
         for (int r = 0; r < J.rows(); r++) {
            int k = r; // k the number of input values
            int n = c; // n frequency components
            double a = f(2 * n, 0);
            double b = f(2 * n + 1, 0);
            double w = Q * k * n;

            J(r, c) = (a * cos(w) + b * sin(w)) / (InputSize.rows * sqrt(a * a + b * b));  // could replace sqrt with z(n,0)
         }
      }



      xout = J * dw;
      */

      xout(0,0) = std::signbit(f(0, 0)) * dw(0,0);
      xout(1,0) = 0.0;
      for (int n = 1; n < OutputSize.rows; n++) {
         int r = n << 1;
         int i = r + 1;
         double a = f(r, 0);
         double b = f(i, 0);

         xout(r,0) =  a * dw(n,0) / z(n, 0);
         xout(i,0) = -b * dw(n,0) / z(n, 0);
      }

      if (JacobianCallBack != nullptr) {
         map<string, CBObj> props;
         int id = 4;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "J", CBObj(J) });
         JacobianCallBack->Properties(props);
      }

      rfftsine(xout.data(), InputSize.rows, -1);
   }

   vector_of_matrix Eval(const vector_of_matrix& _x)
   {
      assert(_x.size() == Channels);
      for (int c = 0; c < Channels; c++) { 
         F[c] = _x[c];
         X[c] = _x[c];
         // The output Z will contain the 1D spectra.
         ColSpecPool(F[c], Z[c]); 
      }

      if (EvalPostActivationCallBack != nullptr) {
         map<string, CBObj> props;
         int id = 1;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "X", CBObj(_x) });
         props.insert({ "Z", CBObj(Z) });
         EvalPostActivationCallBack->Properties(props);
      }
      return Z;
   }

   vector_of_matrix BackProp(vector_of_matrix& child_grad, bool want_backprop_grad = true)
   {
      for (int c = 0; c < Channels; c++)
      {
         BackPool(X[c], F[c], Z[c], child_grad[c]);
      }
      if (BackpropCallBack != nullptr) {
         map<string, CBObj> props;
         int id = 3;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "G", CBObj(child_grad) });
         props.insert({ "X", CBObj(X) });
         BackpropCallBack->Properties(props);
      }
      return X;
   }
   void Update(double eta)
   {
   }
   void Save(shared_ptr<iPutWeights> _pOut)
   {
   }
   void SetEvalPostActivationCallBack(shared_ptr<iCallBackSink> icb) { EvalPostActivationCallBack = icb; }
   void SetBackpropCallBack(shared_ptr<iCallBackSink> icb) { BackpropCallBack = icb; }
   void SetJacobianCallBack(shared_ptr<iCallBackSink> icb) { JacobianCallBack = icb; }
};

//----------------------------------------------------

double MultiplyBlock1( Matrix& m, int mr, int mc, Matrix& h, int hr, int hc, int size_r, int size_c )
{
   double sum = (m.array().block(mr, mc, size_r, size_c) * h.array().block(hr, hc, size_r, size_c)).sum();

   return sum;
}

void LinearCorrelate3( Matrix& m, Matrix& h, Matrix& out )
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
            out(r, c) = 0.0;
         }
         else {
            //cout << m1r << "," << m1c << "," << h1r << "," << h1c << "," << shr << "," << shc << "," << endl;
            out(r, c) = MultiplyBlock1(m, m1r, m1c, h, h1r, h1c, shr, shc);
         }
      }
   }
}


/*
// REVIEW:  Just thinking about a model class.
struct SNModel {
   layer_list LayerList;
   convo_layer_list ConvoLayerList;
   SNModel() {}

};
*/
class StatsOutput
{
   ofstream owf;
public:
   StatsOutput(string name) : owf(path + "\\"  + name + ".csv", ios::trunc)
   {
   }
   void Write(shared_ptr<FilterLayer2D> fl)
   {
      assert(owf.is_open());
      assert(fl);

      //Matrix& dw = fl->dW[0];
      //Eigen::Map<RowVector> rvw(dw.data(), dw.size());
      //const static Eigen::IOFormat OctaveFmt(6, 0, ", ", "\n", "", "", "", "");
      //owf << rvw.format(OctaveFmt) << endl;

      for (int i = 0; i < fl->dW.size();i++) {
         owf << fl->dW[i].blueNorm();
         if (i < fl->dW.size() - 1) {
            owf << ",";
         }
      }
      owf << endl;

   }
};

class GradOutput
{
   ofstream owf;
public:
   GradOutput(string name) : owf(path + "\\"  + name + ".csv", ios::trunc){}
   GradOutput() {}
   void Init(string name) {
      owf.open(path + "\\" + name + ".csv", ios::trunc);
   }
   void Write(RowVector& g)
   {
      assert(owf.is_open());
      owf << g.blueNorm() << endl;
   }
   void Write(ColVector& g)
   {
      assert(owf.is_open());
      owf << g.blueNorm() << endl;
   }
   void Write(vector_of_matrix& vg)
   {
      assert(owf.is_open());

      for (int i = 0; i < vg.size();i++) {
         owf << vg[i].blueNorm();
         if (i < vg.size() - 1) {
            owf << ",";
         }
      }
      owf << endl;
   }
};

void ScalePerLeNet98(double* pdata, int size)
{
   // According to LeCun's 1998 paper, map the extrema of the image
   // to the range -0.1 to 1.175.
   const double y1 = -0.1;
   const double y2 = 1.175;
   double x2 = 0.0;
   double x1 = 0.0;
   double* pd = pdata;
   double* pde = pd + size;
   for (; pd < pde; pd++) {
      if (x2 < *pd) { x2 = *pd; }
      if (x1 > * pd) { x1 = *pd; }
   }

   double m = (y2 - y1) / (x2 - x1);
   double b = y1 - m * x1;

   for (pd = pdata; pd < pde; pd++) {
      *pd = *pd * m + b;
   }
}

typedef std::vector < std::vector< std::tuple<int, int, int, int, double, double>>> LogPolarSupportMatrix;

struct LogPolarSupportMatrixCenter
{
   double row_center;
   double col_center;
   double rad_max;
};

LogPolarSupportMatrix PrecomputeLogPolarSupportMatrix(int in_rows, int in_cols, int out_rows, int out_cols, LogPolarSupportMatrixCenter* pcenter = NULL) 
{
   LogPolarSupportMatrix lpsm;
   lpsm.resize(out_rows, std::vector<std::tuple<int, int, int, int, double, double>>(out_cols));
   
   // For now restrict to square matrix.
   runtime_assert(out_rows == out_cols)
   runtime_assert(in_rows == in_cols )
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
   const double dp = (std::log(rad_max) - std::log(0.5)) / (double)(out_cols-1); // Here we do want to reach log(rad_max).
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
         double cc =  rd(c) * cos(a) + c_center;
         //cout << p << "," << a << "," << x << "," << y << endl;
         if (rr < 0.0) { rr = 0.0; }
         if (cc < 0.0) { cc = 0.0; }

         if (rr > (in_rows - 1)) { rr = in_rows - 1; }
         if (cc > (in_cols - 1)) { cc = in_cols - 1; }

         //runtime_assert(rr >= 0.0 && cc >= 0.0);

         int rl = floor(rr);
         int rh = ceil(rr);
         int cl = floor(cc);
         int ch = ceil(cc);

         /*
         if (xl > 27) { cout << "xl: " << xl; xl = 27; }
         if (xh > 27) { cout << "xh: " << xh; xh = 27; }
         if (yl > 27) { cout << "yl: " << yl; yl = 27; }
         if (yh > 27) { cout << "yh: " << yh; yh = 27; }
         */

         double rp = rr - (double)rl;
         double cp = cc - (double)cl;

         lpsm[r][c] = std::tuple<int, int, int, int, double, double>(rl, rh, cl, ch, rp, cp);
      }
   }
   return lpsm;
}


void ConvertToLogPolar(Matrix& m, Matrix& out, LogPolarSupportMatrix& lpsm )
{
   int rows = out.rows();
   int cols = out.cols();
   // Not right.
   // double rad = cols > rows ? (double)rows : (double)cols;
   
   // For now restrict to square matrix.
   runtime_assert(rows == cols)
   runtime_assert(m.rows() == m.cols() )

   for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
         const tuple<int, int, int, int, double, double>& t = lpsm[r][c];
         const int rl = std::get<0>(t);
         const int rh = std::get<1>(t);
         const int cl = std::get<2>(t);
         const int ch = std::get<3>(t);
         const double rr = std::get<4>(t);
         const double cc = std::get<5>(t);

         double a00 = m(rl, cl);
         double a10 = m(rh, cl) - a00;
         double a01 = m(rl, ch) - a00;
         double a11 = m(rh, ch) - a00 - a10 - a01;
         out(r, c) = a00 + a10 * rr + a01 * cc + a11 * rr * cc;
      }
   }

}

void ConvertLogPolarToCart(const Matrix& m, Matrix& out, LogPolarSupportMatrixCenter lpsmc )
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
   const double dp = (std::log(p_max) - std::log(0.5)) / (double)(in_cols-1); // Here we do want to reach log(dia).
   const double  b = std::log(0.5);

   out.setZero();

   for (int r = 0; r < out_rows; r++) {
      //double rr = r - row_cen - 1;
      double rr =  row_cen - r;
      for (int c = 0; c < out_cols; c++) {
         double cc = c - col_cen;
         double p = std::sqrt(rr*rr+cc*cc);
         if (p < 1) {
            p = 0.5;
         }
         double a = atan2(rr,cc);
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

int GetLabel(ColVector& lv)
{
   for (int i = 0; i < 10; i++) {
      if (lv(i) > 0.5) {
         return i;
      }
   }
   assert(false);
   return 0;
}



//---------------------------------------------------------------------------------
//                         Big Kernel Method
//

class InitBigKernelConvoLayer : public iGetWeights{
   MNISTReader::MNIST_list dl;
   int itodl[10];
public:
   InitBigKernelConvoLayer(){
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
   }
   void ReadConvoWeight(Matrix& m, int k) {
      if (k > 9) {
         m.setZero();
         return;
      }
      assert(m.rows() == 28 && m.cols() == 28);
      TrasformMNISTtoMatrix(m, dl[ itodl[k] ].x );
      ScaleToOne(m.data(), (int)(m.rows() * m.cols()));

      //cout << "Read " << pathname << endl;
      //cout << m << endl;
   }
   void ReadConvoBias(Matrix& w, int k) {
      w.setZero();
   }
   void ReadFC(Matrix& m) {
      throw runtime_error("InitBigKernelConvoLayer::ReadFC not implemented");
   }
};

class InitBigKernelFCLayer : public iGetWeights{

public:
   InitBigKernelFCLayer(){
   }

   void ReadConvoWeight(Matrix& w, int k) {
      throw runtime_error("InitBigKernelFCLayer::ReadConvoWeight not implemented");
   }
   void ReadConvoBias(Matrix& w, int k) {
      throw runtime_error("InitBigKernelFCLayer::ReadConvoBias not implemented");
   }
   void ReadFC(Matrix& m) {
      assert(m.rows() == 10);
      int step = (int)m.cols() / 10;
      assert(step == 14 * 14);
      Matrix pass_field(14, 14);
      pass_field.setZero();
      int pass_rad2 = 4 * 4;
      for (int r = 0; r < 14; r++) {
         int r2 = r * r;
         for (int c = 0; c < 14; c++) {
            int c2 = c * c;
            if ((r2 + c2) <= pass_rad2) {
               pass_field(r, c) = 0.01;
            }
         }
      }
      Eigen::Map<RowVector> rv_pass_field(pass_field.data(), pass_field.size());
      for (int i = 0; i < 10; i++) {
         int pos = i * step;
         m.row(i).setZero();
         m.block(i, pos, 1, step) = rv_pass_field;
      }
      
      // Make sure the bias row is zero.
      m.rightCols(1).setConstant(0.0);
   }
};

//---------------------------------------------------------------------------------
void InitBigKernelModel(bool restore)
{
   model_name = "BKM5";
   ConvoLayerList.clear();
   LayerList.clear();

   // Convolution Layer 1 -----------------------------------------
   // Type: FilterLayer2D
   int size_in  = 28;
   int size_out = 14;
   int kern = 28;
   int pad = 7;
   int kern_per_chn = 5;
   int chn_in = 1;
   int chn_out = kern_per_chn * chn_in;

   int l = 1; // Layer counter
   //                 Size input_size, int input_padding, int input_channels, Size output_size, Size kernel_size, int kernel_number, iActive* _pActive, shared_ptr<iInitWeights> _pInit 
   ConvoLayerList.push_back( make_shared<FilterLayer2D>(clSize(size_in, size_in), chn_in, clSize(size_out, size_out), clSize(kern, kern), kern_per_chn, 
                           make_unique<actReLU>(), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l++))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, chn_in) ),
                           true) ); // No bias
                      //               dynamic_pointer_cast<iGetWeights>( make_shared<InitBigKernelConvoLayer>()),

   // Flattening Layer 2 --------------------------------------------
   // Type: Flatten2D
   size_in  = size_out;
   chn_in = chn_out;
   size_out = size_in * size_in * chn_in;
   chn_out = 1;
   ConvoLayerList.push_back( make_shared<Flatten2D>(clSize(size_in, size_in), chn_in) );
   l++;
   //--------- setup the fully connected network -----------------

   // Fully Connected Layer 1
   // Type: SoftMAX
   size_in = size_out;
   size_out = 10;
   LayerList.push_back(make_shared<Layer>(size_in, size_out,make_unique<actSoftMax>(), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l++))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, chn_in))) );
                                //     dynamic_pointer_cast<iGetWeights>( make_shared<InitBigKernelFCLayer>())) );
   l++;

   // Loss Layer - Not part of network, must be called seperatly.
   // Type: LossCrossEntropy
   loss = make_shared<LossL2>(size_out, 1);   
   //--------------------------------------------------------------

}
//---------------------------- End Big Kernel Method -------------------------------------
//---------------------------- LeNet-5 -------------------------------------------
// LeNet 5 1 chn --> 6 chn -- pool 2 --> (14,14) --> convo 16 (16 * 6 in)[I think] --> pool 2 --> 7
// 
void InitLeNet5Model( bool restore )
{
   model_name = "LeNet5";
   ConvoLayerList.clear();
   LayerList.clear();

   // Convolution Layer 1
   // Type: FilterLayer2D

   int size_in  = 28;
   int size_out = 28;
   int kern_per_chn = 6;
   int chn_in = 1;
   int chn_out = kern_per_chn * chn_in;
   int kern = 5;

   int l = 1; // Layer counter
   //                 Size input_size, int input_padding, int input_channels, Size output_size, Size kernel_size, int kernel_number, iActive* _pActive, shared_ptr<iInitWeights> _pInit 
   ConvoLayerList.push_back( make_shared<FilterLayer2D>( clSize(size_in, size_in), chn_in, clSize(size_out, size_out), clSize(kern, kern), kern_per_chn, 
//                          make_unique<actLeakyReLU(size_out * size_out, 0.01), 
                          make_unique<actTanh>(), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, chn_in))) );
   l++;
   //----------------------------------------------------------------------------------
   // Pooling Layer 2
   // Type: poolAvg3D
   size_in  = size_out;
   size_out = 14;
   chn_in = chn_out;
   chn_out = 16;
   assert(!(size_in % size_out));
   vector_of_colvector_i maps(chn_out);
   int k;
   k = 0;  maps[k].resize(3); maps[k](0) = 0; maps[k](1) = 1; maps[k](2) = 2;
   k = 1;  maps[k].resize(3); maps[k](0) = 1; maps[k](1) = 2; maps[k](2) = 3;
   k = 2;  maps[k].resize(3); maps[k](0) = 2; maps[k](1) = 3; maps[k](2) = 4;
   k = 3;  maps[k].resize(3); maps[k](0) = 3; maps[k](1) = 4; maps[k](2) = 5;
   k = 4;  maps[k].resize(3); maps[k](0) = 0; maps[k](1) = 4; maps[k](2) = 5;
   k = 5;  maps[k].resize(3); maps[k](0) = 0; maps[k](1) = 1; maps[k](2) = 5;
   k = 6;  maps[k].resize(4); maps[k](0) = 0; maps[k](1) = 1; maps[k](2) = 2;  maps[k](3) = 3;
   k = 7;  maps[k].resize(4); maps[k](0) = 1; maps[k](1) = 2; maps[k](2) = 3;  maps[k](3) = 4;
   k = 8;  maps[k].resize(4); maps[k](0) = 2; maps[k](1) = 3; maps[k](2) = 4;  maps[k](3) = 5;
   k = 9;  maps[k].resize(4); maps[k](0) = 0; maps[k](1) = 3; maps[k](2) = 4;  maps[k](3) = 5;
   k = 10;  maps[k].resize(4); maps[k](0) = 0; maps[k](1) = 1; maps[k](2) = 4;  maps[k](3) = 5;
   k = 11;  maps[k].resize(4); maps[k](0) = 0; maps[k](1) = 1; maps[k](2) = 2;  maps[k](3) = 5;
   k = 12;  maps[k].resize(4); maps[k](0) = 0; maps[k](1) = 1; maps[k](2) = 3;  maps[k](3) = 4;
   k = 13;  maps[k].resize(4); maps[k](0) = 1; maps[k](1) = 2; maps[k](2) = 4;  maps[k](3) = 5;
   k = 14;  maps[k].resize(4); maps[k](0) = 0; maps[k](1) = 2; maps[k](2) = 3;  maps[k](3) = 5;
   k = 15;  maps[k].resize(6); maps[k](0) = 0; maps[k](1) = 1; maps[k](2) = 2;  maps[k](3) = 3;  maps[k](4) = 4;  maps[k](5) = 5;

   //                                    poolMax3D(Size input_size, int input_channels, Size output_size, int output_channels, vector_of_colvector_i& output_map) 
   ConvoLayerList.push_back(make_shared<poolMax3D>(clSize(size_in, size_in), chn_in, clSize(size_out, size_out), chn_out, maps));
   l++;  //Need to account for each layer when restoring.
   //--------------------------------------------------------------------------------------
   // Convolution Layer 3
   // Type: FilterLayer2D
   size_in  = size_out;
   size_out = 10; // due to zero padding.
   kern_per_chn = 1;
   chn_in = chn_out;
   chn_out = kern_per_chn * chn_in;
   kern = 5;
   //                 Size input_size, int input_padding, int input_channels, Size output_size, Size kernel_size, int kernel_number, iActive* _pActive, shared_ptr<iInitWeights> _pInit 
   ConvoLayerList.push_back( make_shared<FilterLayer2D>(clSize(size_in, size_in), chn_in, clSize(size_out, size_out), clSize(kern, kern), kern_per_chn, 
//                          make_unique<actLeakyReLU(size_out * size_out, 0.01), 
                          make_unique<actTanh>(), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, chn_in))) );
   l++;
   //----------------------------------------------------------------------------------

   // Pooling Layer 4
   // Type: poolAvg3D
    chn_in = chn_out;
    chn_out = 1;  // Pool all layers into one layer.
   vector_of_colvector_i maps4(1);
   maps4[0].resize(chn_in); 
   for (int i = 0; i < chn_in; i++) { maps4[0](i) = i; }
   size_in  = size_out;
   size_out = 5;

   assert(!(size_in % size_out));
   ConvoLayerList.push_back(make_shared<poolMax3D>(clSize(size_in, size_in), chn_in, clSize(size_out, size_out), chn_out, maps4));
   l++;
   //----------------------------------------------------------------------------------

   // Convolution Layer 5
   // Type: FilterLayer2D
   size_in  = size_out;
   size_out = 1;
   kern_per_chn = 120;
   chn_in = chn_out;
   chn_out = kern_per_chn * chn_in;
   kern = 5;
   //                 Size input_size, int input_padding, int input_channels, Size output_size, Size kernel_size, int kernel_number, iActive* _pActive, shared_ptr<iInitWeights> _pInit 
   ConvoLayerList.push_back( make_shared<FilterLayer2D>(clSize(size_in, size_in), chn_in, clSize(size_out, size_out), clSize(kern, kern), kern_per_chn, 
//                          make_unique<actLeakyReLU(size_out * size_out, 0.01), 
                          make_unique<actTanh>(), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, chn_in))) );
   l++;
   //----------------------------------------------------------------------------------
   // Flattening Layer 6
   // Type: Flatten2D
   size_in  = size_out;
   chn_in = chn_out;
   size_out = size_in * size_in * chn_in;
   chn_out = 1;
   ConvoLayerList.push_back( make_shared<Flatten2D>(clSize(size_in, size_in), chn_in) );
   l++;  //Need to account for each layer when restoring.

   //--------- setup the fully connected network -----------------

   // Fully Connected Layer 1
   // Type: ReLU
   size_in = size_out;
   size_out = 84;
   LayerList.push_back(make_shared<Layer>(size_in, size_out,make_unique<actTanh>(), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, 1))) );
   l++;
   //----------------------------------------------------------------------------------

   // Fully Connected Layer 2
   // Type: SoftMAX
   size_in = size_out;
   size_out = 10;
   LayerList.push_back(make_shared<Layer>(size_in, size_out,make_unique<actSoftMax>(), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l++))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, 1))) );

   // Loss Layer - Not part of network, must be called seperatly.
   loss = make_shared<LossCrossEntropy>(size_out, 1);   
   //--------------------------------------------------------------

}
//------------------------------- End LeNet-5 -------------------------------------
//---------------------------------------------------------------------------------
void InitLeNet5AModel(bool restore)
{
   model_name = "LeNet5A";
   ConvoLayerList.clear();
   LayerList.clear();

   // Convolution Layer 1 -----------------------------------------
   // Type: FilterLayer2D
   int size_in  = 28;
   int size_out = 28;
   int kern = 5;
   int kern_per_chn = 32;
   int chn_in = 1;
   int chn_out = kern_per_chn * chn_in;
   int l = 1; // Layer counter
   ConvoLayerList.push_back( make_shared<FilterLayer2D>(clSize(size_in, size_in), chn_in, clSize(size_out, size_out), clSize(kern, kern), kern_per_chn, 
                          make_unique<actTanh>(), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToRandom>(0.01,0.0)),
                           false) ); // No bias - false -> use bias
   l++;
   //---------------------------------------------------------------
 
   // Pooling Layer 2 ----------------------------------------------
   // Type: poolMax2D
   size_in  = size_out;
   size_out = 14;
   chn_in = chn_out;

   assert(!(size_in % size_out));
   ConvoLayerList.push_back(make_shared<poolAvg2D>(clSize(size_in, size_in), chn_in, clSize(size_out, size_out)));
   l++;  //Need to account for each layer when restoring.
   //---------------------------------------------------------------

   // Convolution Layer 3 -----------------------------------------
   // Type: FilterLayer2D
   size_in  = size_out;
   size_out = 7;
   kern = 5;
   kern_per_chn = 2;
   chn_in = chn_out;
   chn_out = kern_per_chn * chn_in;
   ConvoLayerList.push_back( make_shared<FilterLayer2D>(clSize(size_in, size_in), chn_in, clSize(size_out, size_out), clSize(kern, kern), kern_per_chn, 
                          make_unique<actTanh>(), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToRandom>(0.01,0.0)) ));
   l++;
   //---------------------------------------------------------------

   // Pooling Layer 4 ----------------------------------------------
   // Type: poolMax2D
   size_in  = size_out;
   size_out = 1;
   chn_in = chn_out;

   assert(!(size_in % size_out));
   ConvoLayerList.push_back(make_shared<poolAvg2D>(clSize(size_in, size_in), chn_in, clSize(size_out, size_out)));
   l++;  //Need to account for each layer when restoring.
   //---------------------------------------------------------------      

   // Flattening Layer 5 --------------------------------------------
   // Type: Flatten2D
   size_in  = size_out;
   chn_in = chn_out;
   size_out = size_in * size_in * chn_in;
   chn_out = 1;
   ConvoLayerList.push_back( make_shared<Flatten2D>(clSize(size_in, size_in), chn_in) );
   l++;
   //---------------------------------------------------------------      

   //--------- setup the fully connected network -------------------------------------------------------------------------

   // Fully Connected Layer 6 ---------------------------------------
   // Type: ReLU
   size_in = size_out;
   size_out = 512;
   LayerList.push_back(make_shared<Layer>(size_in, size_out,make_unique<actReLU>(), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToRandom>(0.01,0.0)) ));
   l++;
   //---------------------------------------------------------------      

   // Fully Connected Layer 7 ---------------------------------------
   // Type: SoftMAX
   size_in = size_out;
   size_out = 10;
   LayerList.push_back(make_shared<Layer>(size_in, size_out,make_unique<actSoftMax>(), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToRandom>(0.01,0.0)) ));
   l++;
   //---------------------------------------------------------------      

   // Loss Layer - Not part of network, must be called seperatly.
   // Type: LossCrossEntropy
   loss = make_shared<LossL2>(size_out, 1);   
   //--------------------------------------------------------------

}
//---------------------------- End LeNet 5A ---------------------------------------
//---------------------------------------------------------------------------------
void InitLeNet5BModel(bool restore)
{
   model_name = "LeNet5B";
   ConvoLayerList.clear();
   LayerList.clear();
   cout << "Initializing model: " << model_name << " , restore = " << restore << endl;

   // Convolution Layer 1 -----------------------------------------
   // Type: FilterLayer2D
   int size_in  = 28;
   int size_out = 28;
   int kern = 7;
   int kern_per_chn = 16;
   int chn_in = 1;
   int chn_out = kern_per_chn * chn_in;
   int l = 1; // Layer counter
   ConvoLayerList.push_back( make_shared<FilterLayer2D>(clSize(size_in, size_in), chn_in, clSize(size_out, size_out), clSize(kern, kern), kern_per_chn, 
                          make_unique<actLeakyReLU>(0.01), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, chn_in))) );
   l++;
   //---------------------------------------------------------------
 
   // Pooling Layer  ----------------------------------------------
   // Type: PoolAvg2D
   size_in  = size_out;
   size_out = 14;
   chn_in = chn_out;

   assert(!(size_in % size_out));
   ConvoLayerList.push_back(make_shared<poolAvg2D>(clSize(size_in, size_in), chn_in, clSize(size_out, size_out)));
   l++;  //Need to account for each layer when restoring.
   //---------------------------------------------------------------

   // Convolution Layer  -----------------------------------------
   // Type: FilterLayer2D
   size_in  = size_out;
   size_out = 10;
   kern = 5;
   kern_per_chn = 2;
   chn_in = chn_out;
   chn_out = kern_per_chn * chn_in;
   ConvoLayerList.push_back( make_shared<FilterLayer2D>(clSize(size_in, size_in), chn_in, clSize(size_out, size_out), clSize(kern, kern), kern_per_chn, 
                          make_unique<actLeakyReLU>(0.01), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, chn_in))) );
   l++;
   //---------------------------------------------------------------

   // Pooling Layer 4 ----------------------------------------------
   // Type: poolAvg2D
   size_in  = size_out;
   size_out = 5;
   chn_in = chn_out;

   assert(!(size_in % size_out));
   ConvoLayerList.push_back(make_shared<poolAvg2D>(clSize(size_in, size_in), chn_in, clSize(size_out, size_out)));
   l++;
   //---------------------------------------------------------------      
   
   // Convolution Layer  -----------------------------------------
   // Type: FilterLayer2D
   size_in  = size_out;
   size_out = 1;
   kern = 5;
   kern_per_chn = 2;
   chn_in = chn_out;
   chn_out = kern_per_chn * chn_in;
   ConvoLayerList.push_back( make_shared<FilterLayer2D>(clSize(size_in, size_in), chn_in, clSize(size_out, size_out), clSize(kern, kern), kern_per_chn, 
                          make_unique<actLeakyReLU>(0.01), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, chn_in))) );
   l++;
   //---------------------------------------------------------------
   // Flattening Layer 5 --------------------------------------------
   // Type: Flatten2D
   size_in  = size_out;
   chn_in = chn_out;
   size_out = size_in * size_in * chn_in;
   chn_out = 1;
   ConvoLayerList.push_back( make_shared<Flatten2D>(clSize(size_in, size_in), chn_in) );
   l++;
   //---------------------------------------------------------------      

   //--------- setup the fully connected network -------------------------------------------------------------------------

   // Fully Connected Layer 6 ---------------------------------------
   // Type: ReLU
   size_in = size_out;
   size_out = 32;
   LayerList.push_back(make_shared<Layer>(size_in, size_out,make_unique<actLeakyReLU>(0.01), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, 1))) );   l++;
   //---------------------------------------------------------------      

   // Fully Connected Layer 7 ---------------------------------------
   // Type: SoftMAX
   size_in = size_out;
   size_out = 10;
   LayerList.push_back(make_shared<Layer>(size_in, size_out,make_unique<actSoftMax>(), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, 1))) );   l++;
   //---------------------------------------------------------------      

   // Loss Layer - Not part of network, must be called seperatly.
   loss = make_shared<LossL2>(size_out, 1);   
   //--------------------------------------------------------------

}
//---------------------------- End LeNet 5B ---------------------------------------
//---------------------------------------------------------------------------------

void InitLPModel1(bool restore)
{
   // LP1B - 64 x 64
   //model_name = "LP1C\\LP1C";
   // Adding one kernel, making 5.
   //model_name = "LP1C2\\LP1C2";
   // 6 kernels.
   model_name = "LP1C4\\LP1C3";
   ConvoLayerList.clear();
   LayerList.clear();
   cout << "Initializing model: " << model_name << " , restore = " << restore << endl;

   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   clSize size_in(INPUT_ROWS, INPUT_COLS);
   clSize size_out(INPUT_ROWS, 4);
   clSize size_kern(INPUT_ROWS, INPUT_COLS);
   int kern_per_chn = 6;
   int chn_in = 1;
   int chn_out = kern_per_chn * chn_in;
   int l = 1; // Layer counter
   ConvoLayerList.push_back( make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, kern_per_chn, 
                          //make_unique<actLeakyReLU>(0.04), 
                          make_unique<actReLU>(), 
                          //make_unique<actLinear>(), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, chn_in)),
                           true ) ); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
   l++;
   //---------------------------------------------------------------
   // Pooling Layer ----------------------------------------------
   // Type: poolAvg2D
   size_in  = size_out;
   size_out.Resize(INPUT_ROWS>>1, 1);
   chn_in = chn_out;

   assert(!(size_in.rows % size_out.rows));
   assert(!(size_in.cols % size_out.cols));
   ConvoLayerList.push_back(make_shared<poolAvg2D>(size_in, chn_in, size_out) );
   //dynamic_pointer_cast<poolAvg2D>( ConvoLayerList.back() )->SetEvalPostActivationCallBack( MCB );
   l++;
   //---------------------------------------------------------------
   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in  = size_out;
   chn_in = chn_out;
   int len_out = size_in.rows * size_in.cols * chn_in;
   chn_out = 1;
   ConvoLayerList.push_back( make_shared<Flatten2D>(size_in, chn_in) );
   l++;
   //---------------------------------------------------------------      
   // 
   //--------- setup the fully connected network -------------------------------------------------------------------------
   // 
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   int len_in = len_out;
   len_out = 32;
   LayerList.push_back(make_shared<Layer>(len_in, len_out, make_unique<actLeakyReLU>(0.01), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, 1))) );
   l++;
   //---------------------------------------------------------------  
   // Fully Connected Layer ---------------------------------------
   // Type: SoftMAX
   len_in = len_out;
   len_out = 10;
   LayerList.push_back(make_shared<Layer>(len_in, len_out,make_unique<actSoftMax>(), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, 1))) );
   l++;
   //---------------------------------------------------------------      

   // Loss Layer - Not part of network, must be called seperatly.
   // Type: LossCrossEntropy
   loss = make_shared<LossCrossEntropy>(len_out, 1);   
   //--------------------------------------------------------------

}
//---------------------------- End InitLPModel1a ---------------------------------------
void InitLPModel1a(bool restore)
{
   // LP1B - 64 x 64
   //model_name = "LP1C\\LP1C";
   // Adding one kernel, making 5.
   //model_name = "LP1C2\\LP1C2";
   // 6 kernels.
   model_name = "LP1C4A\\LP1C3A";
   ConvoLayerList.clear();
   LayerList.clear();
   cout << "Initializing model: " << model_name << " , restore = " << restore << endl;

   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   clSize size_in(INPUT_ROWS, INPUT_COLS);
   clSize size_out(INPUT_ROWS, 4);
   clSize size_kern(INPUT_ROWS, INPUT_COLS);
   int kern_per_chn = 6;
   int chn_in = 1;
   int chn_out = kern_per_chn * chn_in;
   int l = 1; // Layer counter
   ConvoLayerList.push_back(make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, kern_per_chn,
      //make_unique<actLeakyReLU>(0.04), 
      make_unique<actReLU>(),
      //make_unique<actLinear>(), 
      restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
      dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, chn_in)),
      true)); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
   l++;
   //---------------------------------------------------------------
   // Pooling Layer ----------------------------------------------
   // Type: poolMax2D
   size_in = size_out;
   size_out.Resize(INPUT_ROWS, 1);
   chn_in = chn_out;

   assert(!(size_in.rows % size_out.rows));
   assert(!(size_in.cols % size_out.cols));
   ConvoLayerList.push_back(make_shared<poolMax2D>(size_in, chn_in, size_out));
   //dynamic_pointer_cast<poolAvg2D>( ConvoLayerList.back() )->SetEvalPostActivationCallBack( MCB );
   l++;
   //---------------------------------------------------------------
   // Pooling Layer ----------------------------------------------
   // Type: poolAvg2D
   size_in = size_out;
   size_out.Resize(INPUT_ROWS >> 1, 1);
   chn_in = chn_out;

   assert(!(size_in.rows % size_out.rows));
   assert(!(size_in.cols % size_out.cols));
   ConvoLayerList.push_back(make_shared<poolAvg2D>(size_in, chn_in, size_out));
   //dynamic_pointer_cast<poolAvg2D>( ConvoLayerList.back() )->SetEvalPostActivationCallBack( MCB );
   l++;
   //---------------------------------------------------------------
   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in = size_out;
   chn_in = chn_out;
   int len_out = size_in.rows * size_in.cols * chn_in;
   chn_out = 1;
   ConvoLayerList.push_back(make_shared<Flatten2D>(size_in, chn_in));
   l++;
   //---------------------------------------------------------------      
   // 
   //--------- setup the fully connected network -------------------------------------------------------------------------
   // 
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   int len_in = len_out;
   len_out = 32;
   LayerList.push_back(make_shared<Layer>(len_in, len_out, make_unique<actLeakyReLU>(0.01),
      restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
      dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, 1))));
   l++;
   //---------------------------------------------------------------  
   // Fully Connected Layer ---------------------------------------
   // Type: SoftMAX
   len_in = len_out;
   len_out = 10;
   LayerList.push_back(make_shared<Layer>(len_in, len_out, make_unique<actSoftMax>(),
      restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
      dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, 1))));
   l++;
   //---------------------------------------------------------------      

   // Loss Layer - Not part of network, must be called seperatly.
   // Type: LossCrossEntropy
   loss = make_shared<LossCrossEntropy>(len_out, 1);
   //--------------------------------------------------------------

}
//---------------------------- End InitLPModel1a ---------------------------------------

void InitLPModel2(bool restore)
{
   model_name = "LP2\\LP2C1";
   ConvoLayerList.clear();
   LayerList.clear();
   cout << "Initializing model: " << model_name << " , restore = " << restore << endl;

   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   clSize size_in(INPUT_ROWS, INPUT_COLS);
   clSize size_out(INPUT_ROWS, 4);
   clSize size_kern(INPUT_ROWS, INPUT_COLS);
   int kern_per_chn = 4;
   int chn_in = 1;
   int chn_out = kern_per_chn * chn_in;
   int l = 1; // Layer counter
   ConvoLayerList.push_back(make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, kern_per_chn,
      //make_unique<actLeakyReLU>(0.04), 
      make_unique<actReLU>(),
      //make_unique<actLinear>(), 
      restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
      dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, chn_in)),
      true)); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
   l++;
   //---------------------------------------------------------------
   // Pooling Layer ----------------------------------------------
   // Type: poolAvg2D
   size_in = size_out;
   size_out.Resize(INPUT_ROWS >> 1, 1);
   chn_in = chn_out;

   assert(!(size_in.rows % size_out.rows));
   assert(!(size_in.cols % size_out.cols));
   ConvoLayerList.push_back(make_shared<poolAvg2D>(size_in, chn_in, size_out));
   //dynamic_pointer_cast<poolAvg2D>( ConvoLayerList.back() )->SetEvalPostActivationCallBack( MCB );
   l++;
   //---------------------------------------------------------------
   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in = size_out;
   chn_in = chn_out;
   int len_out = size_in.rows * size_in.cols * chn_in;
   chn_out = 1;
   ConvoLayerList.push_back(make_shared<Flatten2D>(size_in, chn_in));
   dynamic_pointer_cast<Flatten2D>( ConvoLayerList.back() )->SetEvalPostActivationCallBack( ACB );
   l++;
   //---------------------------------------------------------------      
   // 
   //--------- setup the fully connected network -------------------------------------------------------------------------
   // 
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   int len_in = len_out;
   len_out = 32;
   LayerList.push_back(make_shared<Layer>(len_in, len_out, make_unique<actLeakyReLU>(0.01),
      restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
      dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, 1))));
   l++;
   //---------------------------------------------------------------  
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   len_in = len_out;
   len_out = 16;
   LayerList.push_back(make_shared<Layer>(len_in, len_out, make_unique<actLeakyReLU>(0.01),
      restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
      dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, 1))));
   l++;
   //---------------------------------------------------------------  
   // Fully Connected Layer ---------------------------------------
   // Type: SoftMAX
   len_in = len_out;
   len_out = 10;
   LayerList.push_back(make_shared<Layer>(len_in, len_out, make_unique<actSoftMax>(),
      restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
      dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, 1))));
   l++;
   //---------------------------------------------------------------      

   // Loss Layer - Not part of network, must be called seperatly.
   // Type: LossCrossEntropy
   loss = make_shared<LossCrossEntropy>(len_out, 1);
   //--------------------------------------------------------------

}
//---------------------------- End InitLPModel2 ---------------------------------------

void InitLPModel3(bool restore)
{
   model_name = "LP3\\LP3C1";
   ConvoLayerList.clear();
   LayerList.clear();
   cout << "Initializing model: " << model_name << " , restore = " << restore << endl;

   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   clSize size_in(INPUT_ROWS, INPUT_COLS);
   clSize size_out(INPUT_ROWS, 4);
   clSize size_kern(INPUT_ROWS, INPUT_COLS);
   int kern_per_chn = 4;
   //int kern_per_chn = 1;
   int chn_in = 1;
   int chn_out = kern_per_chn * chn_in;
   int l = 1; // Layer counter
   ConvoLayerList.push_back(make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, kern_per_chn,
      //make_unique<actLeakyReLU>(0.04), 
      make_unique<actReLU>(),
      //make_unique<actLinear>(), 
      restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
      dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, chn_in)),
      true)); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
   l++;
   //---------------------------------------------------------------
   // Pooling Layer ----------------------------------------------
   // Type: poolMax2D
   size_in = size_out;
   size_out.Resize(INPUT_ROWS, 1);
   chn_in = chn_out;

   assert(!(size_in.rows % size_out.rows));
   assert(!(size_in.cols % size_out.cols));
   ConvoLayerList.push_back(make_shared<poolMax2D>(size_in, chn_in, size_out));
   //ConvoLayerList.push_back(make_shared<poolAvg2D>(size_in, chn_in, size_out));
   //dynamic_pointer_cast<poolMax2D>( ConvoLayerList.back() )->SetEvalPostActivationCallBack( MCB );
   l++;
   //---------------------------------------------------------------
  
   // Pooling Layer ----------------------------------------------
   // Type: poolColSpec
   size_in = size_out;
   // This is enforced by poolColSpec.  It does not take a size_out parameter,
   // this is just what you get.
   size_out.Resize(INPUT_ROWS>>1, 1);
   chn_in = chn_out;

   assert(!(size_in.rows % size_out.rows));
   assert(!(size_in.cols % size_out.cols));
   ConvoLayerList.push_back(make_shared<poolColSpec>(size_in, chn_in));

   dynamic_pointer_cast<poolColSpec>(ConvoLayerList.back())->SetBackpropCallBack(MCB);
   dynamic_pointer_cast<poolColSpec>(ConvoLayerList.back())->SetJacobianCallBack(MCB2);

   l++;

   //---------------------------------------------------------------
   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in = size_out;
   chn_in = chn_out;
   int len_out = size_in.rows * size_in.cols * chn_in;
   chn_out = 1;
   ConvoLayerList.push_back(make_shared<Flatten2D>(size_in, chn_in));
   dynamic_pointer_cast<Flatten2D>(ConvoLayerList.back())->SetBackpropCallBack(MCB1);

   l++;
   //---------------------------------------------------------------      
   // 
   //--------- setup the fully connected network -------------------------------------------------------------------------
   // 
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   int len_in = len_out;
   len_out = 32;
   LayerList.push_back(make_shared<Layer>(len_in, len_out, make_unique<actLeakyReLU>(0.01),
      restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
      dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, 1))));
   l++;
   //---------------------------------------------------------------   
   // Fully Connected Layer ---------------------------------------
   // Type: SoftMAX
   len_in = len_out;
   len_out = 10;
   LayerList.push_back(make_shared<Layer>(len_in, len_out, make_unique<actSoftMax>(),
      restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
      dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, 1))));
   l++;
   //---------------------------------------------------------------      

   // Loss Layer - Not part of network, must be called seperatly.
   // Type: LossCrossEntropy
   loss = make_shared<LossCrossEntropy>(len_out, 1);
   //--------------------------------------------------------------

}
//---------------------------- End InitLPModel3 ---------------------------------------

void InitTestSpec(bool restore)
{
   model_name = "LP3\\TS";
   ConvoLayerList.clear();
   LayerList.clear();
   cout << "Initializing model: " << model_name << " , restore = " << restore << endl;

   // Top Layer -----------------------------------------
   // Type: poolColSpec
   clSize size_in(8, 1);
   clSize size_out(4, 1);
   clSize size_kern(INPUT_ROWS, INPUT_COLS);
   int kern_per_chn = 1;
   int chn_in = 1;
   int chn_out = kern_per_chn * chn_in;
   int l = 1; // Layer counter

   assert(!(size_in.rows % size_out.rows));
   assert(!(size_in.cols % size_out.cols));
   ConvoLayerList.push_back(make_shared<poolColSpec>(size_in, chn_in));

   dynamic_pointer_cast<poolColSpec>(ConvoLayerList.back())->SetJacobianCallBack(MCB);

   l++;
   //---------------------------------------------------------------
   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in = size_out;
   chn_in = chn_out;
   int len_out = size_in.rows * size_in.cols * chn_in;
   chn_out = 1;
   ConvoLayerList.push_back(make_shared<Flatten2D>(size_in, chn_in));
   l++;
   //---------------------------------------------------------------      
   // 
   //--------- setup the fully connected network -------------------------------------------------------------------------
   //   
   // Fully Connected Layer ---------------------------------------
   // Type: SoftMAX
   int len_in = len_out;
   len_out = 10;
   LayerList.push_back(make_shared<Layer>(len_in, len_out, make_unique<actSoftMax>(),
      restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
      dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, 1))));
   l++;
   //---------------------------------------------------------------      

   // Loss Layer - Not part of network, must be called seperatly.
   // Type: LossCrossEntropy
   loss = make_shared<LossCrossEntropy>(len_out, 1);
   //--------------------------------------------------------------

}
//---------------------------- End InitTestSpec ---------------------------------------

typedef void (*InitModelFunction)(bool);

//InitModelFunction InitModel = InitBigKernelModel;
InitModelFunction InitModel = InitLPModel2;

void SaveModelWeights()
{
   int l = 1;
   for (auto lli : ConvoLayerList) {
      lli->Save(make_shared<OWeightsCSVFile>(path, model_name + "." + to_string(l) ));
      lli->Save(make_shared<OMultiWeightsBMP>(path, model_name + "." + to_string(l) ));
      lli->Save(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l++) ));
   }
   for (const auto& lit : LayerList) {
      lit->Save(make_shared<OWeightsCSVFile>(path, model_name + "." + to_string(l) ));
      lit->Save(make_shared<OMultiWeightsBMP>(path, model_name + "." + to_string(l) ));
      lit->Save(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l++) ));
   }
}

#define COMPUTE_LOSS {\
   m[0] = data;\
   for (auto lli : ConvoLayerList) {\
               m = lli->Eval(m);\
            }\
   cv = m[0].col(0);\
   for (auto lli : LayerList) {\
      cv = lli->Eval(cv);\
   }\
   e = loss->Eval(cv, Y);\
}

void MakeBiasErrorFunction( string outroot, string dataroot )
{
   ofstream owf(outroot + ".0.csv", ios::trunc);
   ofstream odwf(outroot + ".0.dB.csv", ios::trunc);

   assert(owf.is_open());
   assert(odwf.is_open());

   ColVector w0(1000);
   w0.setLinSpaced( -0.2 , 1.2 );

   ColVector f(1000);
   ColVector df(1000);
   MNISTReader reader(  dataroot + "\\train\\train-images-idx3-ubyte",
                        dataroot + "\\train\\train-labels-idx1-ubyte");

   InitModel(false);

   vector_of_matrix m(1);
   Matrix data;

   double e;

   MNISTReader::MNIST_list dl = reader.read_batch(10);

   ColVector cv;
   int n = 0;
   const int kn = 0;

   data.resize(28, 28);
   ColVector Y;
   Y = dl[n].y;
   TrasformMNISTtoMatrix(data, dl[n].x);
   ScaleToOne(data.data(), (int)(data.rows() * data.cols()));

   // NOTE: This is a blind downcast to FilterLayer2D.  We only do this in testing.
   //       The result of the downcast could be tested for null.
   auto ipcl = dynamic_pointer_cast<FilterLayer2D>(ConvoLayerList[0]);

   double f1;
   int pos = 0;
   for (int i = 0; i < 1000; ++i) {
      ipcl->B[kn] = w0[i];
      COMPUTE_LOSS
      f(i) = e;

      ipcl->Count = 0;
      vector_of_matrix vm_backprop(1);  // kpc kernels * 1 channel
      RowVector g = loss->LossGradient();
      for (int i = (int)LayerList.size() - 1; i >= 0; --i) {
         if (i==0) {
            vm_backprop[0] = LayerList[i]->BackProp(g);
         }
         else {
            g = LayerList[i]->BackProp(g);
         }
      }

      for (int i = (int)ConvoLayerList.size() - 1; i >= 0; --i) {
         if (i==0) {
            ConvoLayerList[i]->BackProp(vm_backprop,false);
         }
         else {
            vm_backprop = ConvoLayerList[i]->BackProp(vm_backprop);
         }
      }   
      df(i) = ipcl->dB[kn];
   }

   // octave file format
   const static Eigen::IOFormat OctaveFmt(6, 0, ", ", "\n", "", "", "", "");
   owf << f.format(OctaveFmt);
   owf.close();
   odwf << df.format(OctaveFmt);
   odwf.close();
}

void TestGradComp(string dataroot)
{
   //MNISTReader reader(  dataroot + "\\train\\train-images-idx3-ubyte",
   //                     dataroot + "\\train\\train-labels-idx1-ubyte");

   InitModel(false);

   int n = 1;
   vector_of_matrix m(1);
   Matrix data;
   double e;
   //MNISTReader::MNIST_list dl = reader.read_batch(10);
   ColVector Y(10);
   Y.setZero();
   Y[5] = 1.0;
   data.resize(INPUT_ROWS,INPUT_COLS);
   //TrasformMNISTtoMatrix(data, dl[n].x);
   //ScaleToOne(data.data(), (int)(data.rows() * data.cols()));
   data.setRandom();

   // NOTE: This is a blind downcast to FilterLayer2D.  Normally is will resolve to a FilterLayer2D object because
   //       we are working with the top layer.  The assert will make sure the downcast is valid.
   shared_ptr<FilterLayer2D> ipcl = dynamic_pointer_cast<FilterLayer2D>(ConvoLayerList[0]);
   assert(ipcl);
   int kpc = (int)ipcl->W.size();
   int ksr = ipcl->KernelSize.rows;
   int ksc = ipcl->KernelSize.cols;

   //for (int kn = 0; kn < kpc; kn++) {
   //   cout << ipcl->W[kn] << endl << endl;
   //}

   Matrix dif(ksr, ksc);

   for (int kn = 0; kn < kpc; kn++) {
      dif.setZero();
      ColVector cv;
      for (int r = 0; r < ksr; r++) {
         cout << ".";
         for (int c = 0; c < ksc; c++) {
            double f1, f2;
            double eta = 1.0e-5;

            double w1 = ipcl->W[kn](r, c);
            //----- Eval ------
            ipcl->W[kn](r, c) = w1 - eta;
            COMPUTE_LOSS
               f1 = e;

            ipcl->W[kn](r, c) = w1 + eta;
            COMPUTE_LOSS
               f2 = e;

            ipcl->W[kn](r, c) = w1;
            COMPUTE_LOSS
            vector_of_matrix vm_backprop(1);
            RowVector g = loss->LossGradient();
            for (int i = (int)LayerList.size() - 1; i >= 0; --i) {
               if (i == 0) {
                  vm_backprop[0] = LayerList[i]->BackProp(g);
               }
               else {
                  g = LayerList[i]->BackProp(g);
               }
            }

            for (int i = (int)ConvoLayerList.size() - 1; i >= 0; --i) {
               if (i == 0) {
                  vm_backprop = ConvoLayerList[i]->BackProp(vm_backprop, false);
               }
               else {
                  vm_backprop = ConvoLayerList[i]->BackProp(vm_backprop);
               }
            }

            double grad1 = ipcl->dW[kn](r, c);
            //-------------------------

            double grad = (f2 - f1) / (2.0 * eta);

            //cout << f1 << ", " << grad1 << ", " << grad << ", " << abs(grad - grad1) << endl;

            dif(r, c) = abs(grad - grad1);
         }
      }

      OMultiWeightsBMP lpbmp(path, "et");
      OWeightsCSVFile lpcsv(path, "et");
      lpbmp.Write(dif, 0);
      lpcsv.Write(dif, 0);

      cout << "\ndW[" << kn << "] Max error: " << dif.maxCoeff() << endl;// << dif << endl;


      if (!ipcl->NoBias) {
         // Test the bias value.
         double f1, f2;
         double eta = 1.0e-5;

         double b = ipcl->B[kn];
         //----- Eval ------
         ipcl->B[kn] = b - eta;
         COMPUTE_LOSS
            f1 = e;

         ipcl->B[kn] = b + eta;
         COMPUTE_LOSS
            f2 = e;

         ipcl->B[kn] = b;
         COMPUTE_LOSS
         vector_of_matrix vm_backprop(1);
         RowVector g = loss->LossGradient();

         for (int i = (int)LayerList.size() - 1; i >= 0; --i) {
            if (i == 0) {
               vm_backprop[0] = LayerList[i]->BackProp(g);
            }
            else {
               g = LayerList[i]->BackProp(g);
            }
         }

         for (int i = (int)ConvoLayerList.size() - 1; i >= 0; --i) {
            if (i == 0) {
               ConvoLayerList[i]->BackProp(vm_backprop, false);
            }
            else {
               vm_backprop = ConvoLayerList[i]->BackProp(vm_backprop);
            }
         }

         double grad1 = ipcl->dB[kn];
         //-------------------------

         double grad = (f2 - f1) / (2.0 * eta);
         double b_dif = abs(grad - grad1);
         cout << "dB[" << kn << "] = " << grad1 << " Estimate: " << grad << " Error: " << b_dif << endl;
      }
   }

   cout << "enter a key and press Enter" << endl;
   char c;
   cin >> c;
}

void FilterTop(double sig)
{
   InitModel(true);

   // NOTE: This is a blind downcast to FilterLayer2D.  Normally is will resolve to a FilterLayer2D object because
   //       we are working with the top layer.  The assert will make sure the downcast is valid.
   shared_ptr<FilterLayer2D> ipcl = dynamic_pointer_cast<FilterLayer2D>(ConvoLayerList[0]);
   assert(ipcl);
   int kpc = (int)ipcl->W.size();
   int ksr = ipcl->KernelSize.rows;
   int ksc = ipcl->KernelSize.cols;

   double r_c = ((double)ksr) / 2.0;
   double c_c = ((double)ksr) / 2.0;

   double norm = 1.0 / (2.0 * M_PI * sig);

   Matrix h(ksr, ksc);
   for (int r = 0; r < ksr; r++) {
      for (int c = 0; c < ksc; c++) {
         double rr = r - r_c;
         double cc = c - c_c;
         //double e = std::pow(rr * rr + cc * cc, 2.0) / (2.0 * sig);
         double e = (rr * rr + cc * cc) / (2.0 * sig);
         h(r, c) = norm * std::exp(-e);
      }
   }

   for (int kn = 0; kn < kpc; kn++) {
      Matrix fw(ksr, ksc);
      LinearCorrelate3(ipcl->W[kn], h, fw);
      ipcl->W[kn] = fw;
    }

   int l = 1;
   ipcl->Save(make_shared<OWeightsCSVFile>(path, model_name + "." + to_string(l) ));
   ipcl->Save(make_shared<OMultiWeightsBMP>(path, model_name + "." + to_string(l) ));
   ipcl->Save(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l) ));

   int l0rows = LayerList[0]->W.rows();
   int l0cols = LayerList[0]->W.cols();

   Matrix l0(l0rows, l0cols-1);
   Matrix fl0(l0rows, l0cols-1);
   l0 = LayerList[0]->W.block(0, 0, l0rows, l0cols - 1);
   LinearCorrelate3(l0, h, fl0);
   LayerList[0]->W.block(0, 0, l0rows, l0cols - 1) = fl0;

   l = 4;
   LayerList[0]->Save(make_shared<OWeightsCSVFile>(path, model_name + "." + to_string(l) ));
   LayerList[0]->Save(make_shared<OMultiWeightsBMP>(path, model_name + "." + to_string(l) ));
   LayerList[0]->Save(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l) ));

   cout << "enter a key and press Enter" << endl;
   char c;
   cin >> c;
}

// Filter the FC matrix.
// REVIEW: not complete.
// The dims are pow of 2 but the bias column makes it non pow 2
// so need to do a bit of work to seperate.
void FilterFC(double sig)
{
   InitModel(true);

   // NOTE: This is a blind downcast to FilterLayer2D.  Normally is will resolve to a FilterLayer2D object because
   //       we are working with the top layer.  The assert will make sure the downcast is valid.
   shared_ptr<FilterLayer2D> ipcl = dynamic_pointer_cast<FilterLayer2D>(ConvoLayerList[0]);
   assert(ipcl);
   int kpc = (int)ipcl->W.size();
   int ksr = ipcl->KernelSize.rows;
   int ksc = ipcl->KernelSize.cols;

   double r_c = ((double)ksr) / 2.0;
   double c_c = ((double)ksr) / 2.0;

   double norm = 1.0 / (2.0 * M_PI * sig);

   Matrix h(ksr, ksc);
   for (int r = 0; r < ksr; r++) {
      for (int c = 0; c < ksc; c++) {
         double rr = r - r_c;
         double cc = c - c_c;
         //double e = std::pow(rr * rr + cc * cc, 2.0) / (2.0 * sig);
         double e = (rr * rr + cc * cc) / (2.0 * sig);
         h(r, c) = norm * std::exp(-e);
      }
   }

   for (int kn = 0; kn < kpc; kn++) {
      Matrix fw(ksr, ksc);
      LinearCorrelate3(ipcl->W[kn], h, fw);
      ipcl->W[kn] = fw;
    }

   int l = 1;
   ipcl->Save(make_shared<OWeightsCSVFile>(path, model_name + "." + to_string(l) ));
   ipcl->Save(make_shared<OMultiWeightsBMP>(path, model_name + "." + to_string(l) ));
   ipcl->Save(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l) ));

   cout << "enter a key and press Enter" << endl;
   char c;
   cin >> c;
}

void NetGrowthExp()
{
   clSize size_in(INPUT_ROWS, INPUT_COLS);
   clSize size_out(INPUT_ROWS, 4);
   clSize size_kern(INPUT_ROWS, INPUT_COLS);

   // NOTE: !!! this is the thing that changes. !!!!!
   int kern_per_chn = 4;
   
   int chn_in = 1;
   int chn_out = kern_per_chn * chn_in;

   string local_model_name = "LP1C\\LP1C";
   FilterLayer2D src(size_in, chn_in, size_out, size_kern, kern_per_chn,
      make_unique<actReLU>(),
      dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, local_model_name + "." + to_string(1))),
      true);

   int len_in = (INPUT_ROWS>>1) * kern_per_chn;
   int src_len_in = len_in;
   int len_out = 32;
   Layer lsrc(len_in, len_out, make_unique<actLeakyReLU>(0.01), 
                           dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, local_model_name + "." + to_string(4)) ) );

   // NOTE: !!! this is the thing that changes. !!!!!
   kern_per_chn = 5;
   
   chn_in = 1;
   chn_out = kern_per_chn * chn_in;

   local_model_name = "LP1C1\\LP1C1";
   FilterLayer2D tar(size_in, chn_in, size_out, size_kern, kern_per_chn,
      make_unique<actReLU>(),
      dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, local_model_name + "." + to_string(1))),
      true);

   len_in = (INPUT_ROWS>>1) * kern_per_chn;
   len_out = 32;
   Layer ltar(len_in, len_out, make_unique<actLeakyReLU>(0.01), 
                           dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, local_model_name + "." + to_string(4)) ) );

   tar.W[0] = src.W[0];
   tar.W[1] = src.W[1];
   tar.W[2] = src.W[2];
   tar.W[3] = src.W[3];

   ltar.W.block(0, 0, len_out, src_len_in) = lsrc.W.block(0, 0, len_out, src_len_in);
   int half_col = INPUT_ROWS >> 2;
   ltar.W.block(0, src_len_in,len_out,half_col) = -ltar.W.block(0, src_len_in+half_col,len_out,half_col);

   // This copies the bias vector from the source to the target.
   ltar.W.col(len_in) = lsrc.W.col(src_len_in);

   tar.Save(make_shared<OWeightsCSVFile>(path, local_model_name + "." + to_string(1) ));
   tar.Save(make_shared<OMultiWeightsBMP>(path, local_model_name + "." + to_string(1) ));
   tar.Save(make_shared<IOWeightsBinaryFile>(path, local_model_name + "." + to_string(1) ));

   ltar.Save(make_shared<OWeightsCSVFile>(path, local_model_name + "." + to_string(4) ));
   ltar.Save(make_shared<OMultiWeightsBMP>(path, local_model_name + "." + to_string(4) ));
   ltar.Save(make_shared<IOWeightsBinaryFile>(path, local_model_name + "." + to_string(4) ));


}

void TestSave()
{
   // Initialize the model to random values.
   InitModel(false);
   convo_layer_list clist1 = ConvoLayerList;
   layer_list list1 = LayerList;

   //----------- save ----------------------------------------
   SaveModelWeights();
   //---------------------------------------------------------

   InitModel(true);

   // The test is designed for a specific network.
   // Currently designed for LeNet-5

   int clay[] = { 0, 2 };

   for (int i = 0; i < 2; i++) {
      cout << "Convo Layer: " << clay[i] << endl;

      auto ipc0 = dynamic_pointer_cast<FilterLayer2D>(ConvoLayerList[clay[i]]);
      auto ipc1 = dynamic_pointer_cast<FilterLayer2D>(clist1[clay[i]]);

      int kpc = ipc1->KernelPerChannel;
      int chn = ipc1->Channels;
      int ks = ipc1->KernelSize.rows; // Assume square
      int kn = chn * kpc;

      Matrix kdif(ks, ks);
      for (int i = 0; i < kn; i++) {
         kdif = ipc0->W[i] - ipc1->W[i]; kdif.cwiseAbs();
         cout << "W[" << i << "] diff: " << kdif.maxCoeff() << endl;
      }
   }
   for (int i = 0; i < 2; i++) {
      cout << "Layer: " << i+1 << endl;
      int osz = LayerList[i]->OutputSize;
      int isz = LayerList[i]->InputSize;
      Matrix l1dif(osz, isz+1);
      l1dif = LayerList[i]->W - list1[i]->W;  l1dif.cwiseAbs();
      cout << "W diff: " << l1dif.maxCoeff() << endl;
   }

}

void CreateCylinder(Matrix& m, double ro2, double co2, double radius)
{
   m.setZero();
   double stop_rad2 = radius * radius;
   double pass_rad2 = 3 * 3;
   for (int r = 0; r < m.rows(); r++) {
      double rc = r - ro2 + 1.0;;
      double r2 = rc * rc;
      for (int c = 0; c < m.cols(); c++) {
         double cc = c - co2 + 1.0;
         double c2 = cc * cc;
         double rad2 = (r2 + c2);
         if ( rad2 < stop_rad2 && rad2 > pass_rad2) {

            m(r, c) = 1.0;
         }
      }
   }
}

Matrix MakeDoughnut(float scale, int qtr)
{
	if (qtr < 1 || qtr > 4) {
		qtr = 0;
	}

	Matrix img(28,28);

	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			int x = i - 14;
			int y = j - 14;
			float ang = atan2f((float)y, (float)x);
			if (ang < 0.0f) { ang += (2.0f * M_PI); }

         bool in = false;
			if (qtr == 1 && (ang >= 0.0f && ang < M_PI_2) ) {
				in = true;
			}
			else if (qtr == 2 && ( ang >= M_PI_2 && ang < M_PI) ) {
				in = true;
			}
			else if (qtr == 3 && (ang >= M_PI && ang < (3.0 * M_PI / 2.0)) ) {
				in = true;
			}
			else if (qtr == 4 && (ang >= (3.0 * M_PI / 2.0) && ang < (2.0 * M_PI )) ) {
				in = true;
			}
         else if (qtr == 0) {
            in = true;
         }

         double p = sqrt(x * x + y * y);
			if( in  && (p>5.0 && p<=9.0) ){
				img(j, i) = 1.0;
			}
			else {
				img(j, i) = 0;
			}
		}
	}

   return img;
}

Matrix ReadImage(string name)
{
   cv::Mat image = imread(path + "\\" + name, cv::IMREAD_GRAYSCALE );
   Matrix out(image.rows, image.cols);
   //imshow("resize", image);
   //char c = cv::waitKey(0);
   for (int r = 0; r < image.rows; r++) {
      for (int c = 0; c < image.cols; c++) {
         out(r, c) = image.at<unsigned char>(image.rows - r - 1, c);
      }
   }
   return out;
}

void MakeCorrelations(string dataroot, int label)
{
   /*
   const double da = 2.0 * M_PI / (double)28;
   for (int i = 0; i < 28; i++) {
      double a = (double)i * da;
      double x = cos(a);
      double y = sin(a);
      double aa = atan2(y, x);
      if (aa < 0) {
         aa = 2.0 * M_PI + aa;
      }
      cout << x << " , " << y << " | " << a << " , " << aa << endl;
   }
   exit(0);
   */

   //InitModel(true);

   MNISTReader reader(dataroot + "\\train\\train-images-idx3-ubyte",
                     dataroot + "\\train\\train-labels-idx1-ubyte");
   MNISTReader::MNIST_list dl = reader.read_batch(1000);

   Matrix origional(28, 28);
   Matrix m(INPUT_ROWS, INPUT_COLS);
   LogPolarSupportMatrixCenter lpsmc;
   LogPolarSupportMatrix lpsm = PrecomputeLogPolarSupportMatrix(origional.rows(), origional.cols(), m.rows(), m.cols(), &lpsmc );
   MNISTReader::MNIST_list::iterator idl = dl.begin();

   //int k = 0;
   for (int k = 0; k < 10; k++) {

      for (; idl != dl.end(); idl++) {
         MNISTReader::MNIST_Pair& mp = *idl;
         if (GetLabel(mp.y) == label) {
            TrasformMNISTtoMatrix(origional, mp.x);
            ScaleToOne(origional.data(), (int)(origional.rows() * origional.cols()));
#ifdef LOGPOLAR
            ConvertToLogPolar(origional, m, lpsm);
#endif
            idl++;
            break;
         }
      }


      //origional = MakeDoughnut(0.0, label);
      //ConvertToLogPolar(origional, m, lpsm);

      OMultiWeightsBMP lpbmp(path, to_string(label) + "_LP");
      OWeightsCSVFile lpcsv(path, to_string(label) + "_LP");
      lpbmp.Write(m, k);
      lpcsv.Write(m, k);


      /*
      Matrix o(28,28);
      LinearCorrelate3(m, m, o);
      OWeightsCSVFile of(path, "autocor_" + to_string(label));
      of.Write(o, 0);
      */

      Matrix o(28, 28);
      ConvertLogPolarToCart(m, o, lpsmc);
      OMultiWeightsBMP of(path, "LP_recover_resize" + to_string(label));
      of.Write(o, k);
      OMultiWeightsBMP oo(path, to_string(label) +  "_orig");
      oo.Write(origional, k);
   }
   /*
   // Shift columns of m to the left.
   m.block(0, 0, m.rows(), m.cols()-3) = m.block(0, 3, m.rows(), m.cols() - 3);
   m.col(m.cols() - 1).setZero();
   m.col(m.cols() - 2).setZero();
   m.col(m.cols() - 3).setZero();
   OMultiWeightsBMP lpos(path, "LPS_" + to_string(label));
   lpos.Write(m, 0);
   ConvertLogPolarToCart(m, o, 14, 14, 13);
   OMultiWeightsBMP ofs(path, "LP_recover_shift" + to_string(label));
   ofs.Write(o, 0);
   */
   
   /*
   shared_ptr<FilterLayer2D> ipcl = dynamic_pointer_cast<FilterLayer2D>(ConvoLayerList[0]);
   assert(ipcl);
   int kpc = (int)ipcl->W.size();
   int ks = ipcl->KernelSize.rows;
   assert( ks == ipcl->KernelSize.cols);

   Matrix dif(ks, ks);

   //Matrix o(ipcl->OutputSize.rows, ipcl->OutputSize.cols);
   Matrix o(28,28);
   
   OWeightsCSVFile of(path, "cor_" + to_string(label));
   
   for (int kn = 0; kn < kpc; kn++) {
      LinearCorrelate3(m, ipcl->W[kn], o);
      of.Write(o, kn);
   }
   */

}

class MatrixManipulator
{
   // Set N=0 and M=1 to produce just the origional.
   // 
   // The number of shifts plus the origional.  This number
   // can only be changed in conjunction with work on the shift method.
   const int N = 1;
   //const int N = 5;  
   // The number of rotations plus zero rotation.
   // This number should be odd.
   const int M = 1;
   //const int M = 3;
   // How much to shift the image left or right.
   const int SHIFT = 1;
   const int LpRows;
   const int LpCols;
   int S;
   int A;
   int C;
   Matrix Base;
   Matrix ShiftState;
   Matrix LPState;
   LogPolarSupportMatrix lpsm;
public:
   MatrixManipulator(Matrix m, int lprows, int lpcols) : 
         Base(m), 
         LPState(lprows,lpcols), 
         LpRows(lprows), 
         LpCols(lpcols), 
         S(0), 
         A(0),
         C(0)
   {
      ShiftState.resize(Base.rows(), Base.cols());
      lpsm = PrecomputeLogPolarSupportMatrix(28, 28, lprows, lpcols);
      begin();
   }

   bool isDone()
   {
      //return (S == N && A == M);
      //return (S == N );
      return C == 1;
   }

   void begin()
   {
      A = 0;
      S = 0;
      shift();
      computeLP();
      prerotate(M);
   }

   void next()
   {
      C++;
      if (S==0 && A < M) {
      //if (A < M) {
         rotate();
         A++;
      }
      else {
         A = 0;
         shift();
         computeLP();
         // Uncomment if you want to rotate every shift.
         //prerotate(M);
         S++;
      }
   }

   const Matrix& get() {
      return LPState;
   }

private:
   void shift()
   {
      const int rows = Base.rows();
      const int cols = Base.cols();
      const int r2 = rows - SHIFT;
      const int c2 = cols - SHIFT;
      ShiftState.setZero();
      switch (S) {
         case 0:
            ShiftState = Base;
            break;
         case 1:
            ShiftState.block(0, 0, r2, c2) = Base.block(SHIFT, SHIFT, r2, c2);
            break;
         case 2:
            ShiftState.block(SHIFT, 0, r2, c2) = Base.block(0, SHIFT, r2, c2);
            break;
         case 3:
            ShiftState.block(0, SHIFT, r2, c2) = Base.block(SHIFT, 0, r2, c2);
            break;
         case 4:
            ShiftState.block(SHIFT, SHIFT, r2, c2) = Base.block(0, 0, r2, c2);
            break;
         default:
            cout << "Something wrong." << endl;
            exit(0);
      }

      //MakeMatrixImage(path + "\\shift_trace.bmp", ShiftState);
   }

   void computeLP()
   {
      ConvertToLogPolar(ShiftState, LPState, lpsm);
   }

   // m should be odd.
   void prerotate(int m)
   {
      // bottom to top
      int mrows = m >> 1;
      if (mrows > 0) {
         Matrix temp = LPState.block(0, 0, mrows, LpCols);
         // Shift the matrix down.
         LPState.block(0, 0, LpRows - mrows, LpCols) = LPState.block(mrows, 0, LpRows - mrows, LpCols);
         LPState.block(LpRows - mrows, 0, mrows, LpCols) = temp;
      }
   }

   void rotate()
   {
      // Rotate top to bottom.
      RowVector row(LpCols);
      // Save the top
      row = LPState.row(LpRows - 1);
      // Shift the matrix up.
      Matrix temp;
      temp = LPState.block(0, 0, LpRows - 1, LpCols);
      LPState.block(1, 0, LpRows - 1, LpCols) = temp;
      // Put the top at the bottom
      LPState.row(0) = row;
   }
};

void Train(int nloop, string dataroot, double eta, int load)
{
   cout << dataroot << endl;
#ifdef RETRY
   cout << "NOTE: There is auto-repeate code running!  retry = " << RETRY << endl;
#endif

#ifdef MOMENTUM
   cout << "Momentum is on.  a = " << MOMENTUM << endl;
#else
   cout << "Momentum is off." << endl;
#endif
#ifdef LOGPOLAR
   cout << "Running Log-Polar samples." << endl;
#endif
#ifdef SGD
   cout << "Stocastic decent is on." << endl;
#else
   cout << "Batch grad descent is in use." << endl;
#endif
#ifdef FFT
   cout << "Using FFT convolution." << endl;
   #ifdef CYCLIC
      cout << "Using Cyclic convolution in the row direction." << endl;
   #endif
#else
   cout << "Using linear convolution." << endl;
#endif // FFT



   {
      char s;
      cout << "hit a key to continue";
      cin >> s;
   }

   MNISTReader reader(dataroot + "\\train\\train-images-idx3-ubyte",
                      dataroot + "\\train\\train-labels-idx1-ubyte");

   InitModel(load > 0 ? true : false);

   for ( auto cli : ConvoLayerList ) {
      auto ist = dynamic_pointer_cast<iStash>(cli);
      if (ist) { ist->StashWeights(); }
   }

   for (auto lli : LayerList) {
      auto ist = dynamic_pointer_cast<iStash>(lli);
      if (ist) { ist->StashWeights(); }
   }

//#define STATS
#ifdef STATS
   vector<GradOutput> lveo(LayerList.size());
   for (int i = 0; i < lveo.size(); i++) {
      lveo[i].Init("leval" + to_string(i));
   }
   vector<GradOutput> clveo(ConvoLayerList.size());
   for (int i = 0; i < clveo.size(); i++) {
      clveo[i].Init("cleval" + to_string(i));
   }

   vector<GradOutput> lvgo(LayerList.size()+1);
   for (int i = 0; i < lvgo.size(); i++) {
      lvgo[i].Init("lgrad" + to_string(i));
   }
   vector<GradOutput> clvgo(ConvoLayerList.size());
   for (int i = 0; i < clvgo.size(); i++) {
      clvgo[i].Init("clgrad" + to_string(i));
   }

   #define clveo_write(I,M) clveo[I].Write(M)
   #define lveo_write(I,M) lveo[I].Write(M)
   #define clvgo_write(I,M) clvgo[I].Write(M)
   #define lvgo_write(I,M) lvgo[I].Write(M)
#else
   #define clveo_write(I,M) ((void)0)
   #define lveo_write(I,M) ((void)0)
   #define clvgo_write(I,M) ((void)0)
   #define lvgo_write(I,M) ((void)0)
#endif
   ErrorOutput err_out(path, model_name);
   ClassifierStats stat_class;

   const int reader_batch = 1000;  // Should divide into 60K
   const int batch = 100; // Should divide evenly into reader_batch
   const int batch_loop = 11;

   std::random_device rd;     // only used once to initialise (seed) engine
   std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
   std::uniform_int_distribution<int> uni(0,reader_batch-1); // guaranteed unbiased

   //for (int i = 1; i <= 5; i++) {
   //   ColVector d, l;
   //   Matrix m(28, 28);
   //   reader.read_next();
   //   d = reader.data();
   //   l = reader.label();
   //   TrasformMNISTtoMatrix(m, d);
   //   ScaleToOne(m.data(), (int)(m.rows() * m.cols()));
   //   //CreateCylinder(m, 14, 14, 5);
   //   Matrix temp(28, 28);
   //   ConvertToLogPolar(m, temp);
   //   string file = path + "\\lp" + "_" + to_string(GetLabel(l)) + ".bmp";
   //   //string file = path + "\\lp" + "_cylinder.bmp";
   //   MakeMatrixImage(file, temp);
   //   file = path + "\\cart" + "_" + to_string(GetLabel(l)) + ".bmp";
   //   MakeMatrixImage(file, m
   //   );
   //}
   //exit(0);
   
   LogPolarSupportMatrix lpsm = PrecomputeLogPolarSupportMatrix(28, 28, INPUT_ROWS, INPUT_COLS);

   double e = 0;
   int avg_n;
   for (int loop = 0; loop < nloop; loop++) {
      MNISTReader::MNIST_list dl = reader.read_batch(reader_batch);
      for (int bl = 0; bl < batch_loop; bl++) {
         //--------------------------------------------------------
         // Stash once per batch.  This is to try
         // to be many ittereations away from where
         // the solution may blow up.
         // 
         // REVIEW: There are some faults in this plan.  I suppose
         //         if the solution blows up early in the batch then there
         //         is not a good recording of a stable state.
         //         One good thing is that the value of eta is adjusted outside of
         //         this loop.
         //
         for (auto cli : ConvoLayerList) {
            auto ist = dynamic_pointer_cast<iStash>(cli);
            if (ist) { ist->StashWeights(); }
         }

         for (auto lli : LayerList) {
            auto ist = dynamic_pointer_cast<iStash>(lli);
            if (ist) { ist->StashWeights(); }
         }
         //----------------------------------------------------------
         e = 0;
         avg_n = 1;
         int retry = 0;
         int n = 0;
         int b = 0;
         while (b < batch) {
            if (retry == 0) {
               n = uni(rng); // Select a random entry out of the batch.
               b++;
            }
            vector_of_matrix m(1);
            Matrix temp(28, 28);
            myAveFlattenCallBack::label = GetLabel(dl[n].y);

            m[0].resize(INPUT_ROWS, INPUT_COLS);

            TrasformMNISTtoMatrix(temp, dl[n].x);
            ScaleToOne(temp.data(), (int)(temp.rows() * temp.cols()));

            //#ifdef LOGPOLAR
            //            ConvertToLogPolar(temp, m[0], lpsm);
            //#else
            //            m[0] = temp;
            //#endif
            for (MatrixManipulator mm(temp, INPUT_ROWS, INPUT_COLS); !mm.isDone(); mm.next())
            {
               m[0] = mm.get();

               for (int i = 0; i < ConvoLayerList.size(); i++) {
                  m = ConvoLayerList[i]->Eval(m);
                  clveo_write(i, m);
               }

               ColVector cv;
               cv = LayerList[0]->Eval(m[0].col(0));
               lveo_write(0, cv);
               for (int i = 1; i < LayerList.size(); i++) {
                  cv = LayerList[i]->Eval(cv);
                  lveo_write(i, cv);
               }

               if (retry == 0) {
                  if (stat_class.Eval(cv, dl[n].y) == false) {
#ifdef RETRY
                     //if (     loop > 200) { retry = 4; }
                     //else if (loop > 100) { retry = 2; }
                     //else if (loop > 50) {  retry = 1; }
                     //cout << "retry" << endl;
                     retry = RETRY;
#endif
                  }
                  //else {
                  //   continue;
                  //}
               }
               else {
                  retry--;
               }

               double le = loss->Eval(cv, dl[n].y);
               if (isnan(le)) {
                  for (auto cli : ConvoLayerList) {
                     auto ist = dynamic_pointer_cast<iStash>(cli);
                     if (ist) { ist->ApplyStash(); }
                  }

                  for (auto lli : LayerList) {
                     auto ist = dynamic_pointer_cast<iStash>(lli);
                     if (ist) { ist->ApplyStash(); }
                  }

                  eta /= 10.0;
                  if (eta <= numeric_limits<double>::epsilon()) {
                     cout << "eta limit reached.  Jumping out of training loop." << endl;
                     goto TESTJMP;
                  }
                  cout << "eta reset: " << eta << endl;
                  continue;
               }
               // NOTE: The reset will just pick up in the middile  of the batch.
               //err_out.Write(le);

               //if (le > e) { e = le; }
               double a = 1.0 / (double)(avg_n);
               avg_n++;
               double d = 1.0 - a;
               e = a * le + d * e;

               vector_of_matrix vm_backprop(1);
               RowVector g = loss->LossGradient();
               lvgo_write(LayerList.size(), g);
               for (int i = (int)LayerList.size() - 1; i >= 0; --i) {
                  if (i == 0) {
                     vm_backprop[0] = LayerList[i]->BackProp(g);
                     clvgo_write(i, vm_backprop); // Debug
                  }
                  else {
                     g = LayerList[i]->BackProp(g);
                     lvgo_write(i, g); // Debug
                  }
               }

               for (int i = (int)ConvoLayerList.size() - 1; i >= 0; --i) {
                  if (i == 0) {
                     ConvoLayerList[i]->BackProp(vm_backprop, false);
                     clvgo_write(i, dynamic_pointer_cast<FilterLayer2D>(ConvoLayerList[i])->dW);  // Debug
                  }
                  else {
                     vm_backprop = ConvoLayerList[i]->BackProp(vm_backprop);
                     clvgo_write(i, vm_backprop); // Debug
                  }
               }
            }
#ifdef SGD
            // This is stoastic descent.  It is inside the batch loop.
            for (auto lli : ConvoLayerList) {
               lli->Update(eta);
            }
            for (auto lit : LayerList) {
               lit->Update(eta);
            }
#endif
         }

// if not defined
#ifndef SGD
         //eta = (1.0 / (1.0 + 0.001 * loop)) * eta;
         for (auto lli : ConvoLayerList) {
            lli->Update(eta);
         }
         for (auto lit : LayerList) {
            lit->Update(eta);
         }
#endif
         err_out.Write(stat_class.Correct);
         cout << "count: " << loop << "\terror:" << left << setw(9) << std::setprecision(4) << e << "\tcorrect: " << stat_class.Correct << "\tincorrect: " << stat_class.Incorrect << endl;
         stat_class.Reset();
      }
      /*if (!(loop % 1000)) {
         MNISTReader reader1(dataroot + "\\test\\t10k-images-idx3-ubyte",
            dataroot + "\\test\\t10k-labels-idx1-ubyte");

         stat_class.Reset();

         ColVector X;
         ColVector Y;

         double avg_e = 0.0;
         int count = 0;

         while (reader1.read_next()) {
            X = reader1.data();
            Y = reader1.label();
            vector_of_matrix m(1);
            m[0].resize(28, 28);
            TrasformMNISTtoMatrix(m[0], X);
            ScaleToOne(m[0].data(), (int)(m[0].rows() * m[0].cols()));

            //Matrix temp(28, 28);
            //ConvertToLogPolar(m[0], temp);
            //m[0] = temp;

            for (auto lli : ConvoLayerList) {
               m = lli->Eval(m);
            }

            ColVector cv;
            cv = LayerList[0]->Eval(m[0].col(0));
            for (int i = 1; i < LayerList.size(); i++) {
               cv = LayerList[i]->Eval(cv);
            }

            stat_class.Eval(cv, Y);

         }

         std::cout << " correct/incorrect " << stat_class.Correct << " , " << stat_class.Incorrect << endl;

         SaveModelWeights();
         stat_class.Reset();
      }*/
      // This is %10 down per epoc.
      //const double dec_per_loop = 1.0 - (0.1 / 60.0);
      //eta *= dec_per_loop;
      // The idea is to keep pushing the step larger by %10 per epoc.
      // When the step inevidebly gets too large it will be knocked back
      // an order of magnitude.
      //const double inc_per_loop = 1.0 + (0.1 / 60.0);
      //eta *= inc_per_loop;
      cout << "eta: " << eta << endl;
   }

   TESTJMP:

   MNISTReader reader1( dataroot + "\\test\\t10k-images-idx3-ubyte",
                        dataroot + "\\test\\t10k-labels-idx1-ubyte");

   stat_class.Reset();

   ColVector X;
   ColVector Y;

   double avg_e = 0.0;
   int count = 0;

   while (reader1.read_next()) {
      X = reader1.data();
      Y = reader1.label();
      vector_of_matrix m(1);
      Matrix temp(28, 28);
      m[0].resize(INPUT_ROWS, INPUT_COLS);
      TrasformMNISTtoMatrix(temp, X);
      ScaleToOne(temp.data(), (int)(temp.rows() * temp.cols()));
#ifdef LOGPOLAR
            ConvertToLogPolar(temp, m[0], lpsm);
#else
            m[0] = temp;
#endif
      for (auto lli : ConvoLayerList) {
         m = lli->Eval(m);
      }

      ColVector cv;
      cv = LayerList[0]->Eval(m[0].col(0));
      for (int i = 1; i < LayerList.size(); i++) {
         cv = LayerList[i]->Eval(cv);
      }
      
      stat_class.Eval(cv, Y);
      
   }

   std::cout << " correct/incorrect " << stat_class.Correct << " , " << stat_class.Incorrect << endl;
   std::cout << "Save? y/n:  ";
   char c;
   std::cin >> c;
   if (c == 'y') {
      SaveModelWeights();
   }
}


void OutputIncorrects(string dataroot, MNISTReader::MNIST_list& ml)
{
   //Model should already be initialized.
   //InitModel(true);

   cout << "Getting data from: " << dataroot << endl;
   MNISTReader reader(dataroot + "\\train\\train-images-idx3-ubyte",
                      dataroot + "\\train\\train-labels-idx1-ubyte");

   ClassifierStats stat_class;

   ColVector X;
   ColVector Y;

   double avg_e = 0.0;
   int count = 0;
   //int bad = 0;
   while (reader.read_next()) {
      X = reader.data();
      Y = reader.label();
      vector_of_matrix m(1);

      m[0].resize(28, 28);
      TrasformMNISTtoMatrix(m[0], X);
      ScaleToOne(m[0].data(), (int)(m[0].rows() * m[0].cols()));
      Matrix current_data(m[0]);

      for (auto lli : ConvoLayerList) {
         m = lli->Eval(m);
      }

      ColVector cv;
      cv = LayerList[0]->Eval(m[0].col(0));
      for (int i = 1; i < LayerList.size(); i++) {
         cv = LayerList[i]->Eval(cv);
      }

      if (!stat_class.Eval(cv, Y)) {
         //string file = path + "\\bad" + "_" + to_string(GetLabel(Y)) + "." + to_string(bad) + ".bmp";
         //MakeMatrixImage(file, current_data);
         ml.emplace_back(MNISTReader::MNIST_Pair(X, Y));
         //bad++;
      }
      /*
      if (++count == 10) {
         count = 0;
         std::cout << " correct/incorrect " << stat_class.Correct << " , " << stat_class.Incorrect << endl;
      }
      */
   }
   std::cout << " correct/incorrect " << stat_class.Correct << " , " << stat_class.Incorrect << endl;
}

void OutputIncorrects1(string dataroot)
{
   //Model should already be initialized.
   InitModel(true);
   int label = 2;
   Matrix correct(10, 10);
   Matrix incorrect(10, 10);

   cout << "Getting data from: " << dataroot << endl;
   MNISTReader reader(dataroot + "\\train\\train-images-idx3-ubyte",
                      dataroot + "\\train\\train-labels-idx1-ubyte");

   ClassifierStats stat_class;

   ColVector X;
   ColVector Y;

   int cc = 0;
   int ic = 0;

   double avg_e = 0.0;
   int count = 0;
   //int bad = 0;
   while (reader.read_next()) {
      X = reader.data();
      Y = reader.label();
      if (GetLabel(Y) != label) {
         continue;
      }
      vector_of_matrix m(1);

      m[0].resize(28, 28);
      TrasformMNISTtoMatrix(m[0], X);
      ScaleToOne(m[0].data(), (int)(m[0].rows() * m[0].cols()));
      Matrix current_data(m[0]);

      for (auto lli : ConvoLayerList) {
         m = lli->Eval(m);
      }

      ColVector cv;
      cv = LayerList[0]->Eval(m[0].col(0));
      for (int i = 1; i < LayerList.size(); i++) {
         cv = LayerList[i]->Eval(cv);
      }

      if (stat_class.Eval(cv, Y)) {
         //string file = path + "\\bad" + "_" + to_string(GetLabel(Y)) + "." + to_string(bad) + ".bmp";
         //MakeMatrixImage(file, current_data);
         if (cc < 10) {
            correct.col(cc) = cv;
            cc++;
         }
         //bad++;
      }
      else {
         if (ic < 10) {
            incorrect.col(ic) = cv;
            ic++;
         }
      }
      
      if (cc==10 && ic==10) {
         OWeightsCSVFile occf(path, "correct_" + to_string(label));
         OWeightsCSVFile oicf(path, "incorrect_" + to_string(label));
         occf.Write(correct, 0);
         oicf.Write(incorrect, 0);
         return;
      }
   }
}

/*
void Train1(int nloop, string dataroot, double eta, int load)
{
   cout << dataroot << endl;
   cout << "Train1 running." << endl;
   cout << "NOTE: There is auto-repeate code running!" << endl;
   cout << "NOTE: Only operating on incorrect samples!" << endl;
   {
      char s;
      cout << "hit a key to continue";
      cin >> s;
   }

   InitModel(load > 0 ? true : false);

   for ( auto cli : ConvoLayerList ) {
      auto ist = dynamic_pointer_cast<iStash>(cli);
      if (ist) { ist->StashWeights(); }
   }

   for (auto lli : LayerList) {
      auto ist = dynamic_pointer_cast<iStash>(lli);
      if (ist) { ist->StashWeights(); }
   }

//#define STATS
#ifdef STATS
   vector<GradOutput> lveo(LayerList.size());
   for (int i = 0; i < lveo.size(); i++) {
      lveo[i].Init("leval" + to_string(i));
   }
   vector<GradOutput> clveo(ConvoLayerList.size());
   for (int i = 0; i < clveo.size(); i++) {
      clveo[i].Init("cleval" + to_string(i));
   }

   vector<GradOutput> lvgo(LayerList.size()+1);
   for (int i = 0; i < lvgo.size(); i++) {
      lvgo[i].Init("lgrad" + to_string(i));
   }
   vector<GradOutput> clvgo(ConvoLayerList.size());
   for (int i = 0; i < clvgo.size(); i++) {
      clvgo[i].Init("clgrad" + to_string(i));
   }

   #define clveo_write(I,M) clveo[I].Write(M)
   #define lveo_write(I,M) lveo[I].Write(M)
   #define clvgo_write(I,M) clvgo[I].Write(M)
   #define lvgo_write(I,M) lvgo[I].Write(M)
#else
   #define clveo_write(I,M) ((void)0)
   #define lveo_write(I,M) ((void)0)
   #define clvgo_write(I,M) ((void)0)
   #define lvgo_write(I,M) ((void)0)
#endif
   ErrorOutput err_out(path, model_name);
   ClassifierStats stat_class;
   
   double e = 0;
   int avg_n;
   for (int loop = 0; loop < nloop; loop++) {
      MNISTReader::MNIST_list ml;
      OutputIncorrects(dataroot, ml);
      cout << "Incorrect: " << ml.size() << endl;
      for (int t = 1; t <= 10; t++) {
         e = 0;
         avg_n = 1;
         int nn = 0;
         for (MNISTReader::MNIST_Pair& dl : ml) {

            vector_of_matrix m(1);
            m[0].resize(28, 28);
            TrasformMNISTtoMatrix(m[0], dl.x);
            ScaleToOne(m[0].data(), (int)(m[0].rows() * m[0].cols()));

            //Matrix temp(28, 28);
            //ConvertToLogPolar(m[0], temp);
            //m[0] = temp;

            for (int i = 0; i < ConvoLayerList.size(); i++) {
               m = ConvoLayerList[i]->Eval(m);
               clveo_write(i, m);
            }

            ColVector cv;
            cv = LayerList[0]->Eval(m[0].col(0));
            lveo_write(0, cv);
            for (int i = 1; i < LayerList.size(); i++) {
               cv = LayerList[i]->Eval(cv);
               lveo_write(i, cv);
            }

            stat_class.Eval(cv, dl.y);
            double le = loss->Eval(cv, dl.y);

            if (isnan(le)) {
               for (auto cli : ConvoLayerList) {
                  auto ist = dynamic_pointer_cast<iStash>(cli);
                  if (ist) { ist->ApplyStash(); }
               }

               for (auto lli : LayerList) {
                  auto ist = dynamic_pointer_cast<iStash>(lli);
                  if (ist) { ist->ApplyStash(); }
               }

               eta /= 10.0;
               if (eta <= numeric_limits<double>::epsilon()) {
                  cout << "eta limit reached.  Jumping out of training loop." << endl;
                  goto TESTJMP;
               }
               cout << "eta reset: " << eta << endl;
               continue;
            }
            // NOTE: The reset will just pick up in the middile  of the batch.
            //err_out.Write(le);

            //if (le > e) { e = le; }
            double a = 1.0 / (double)(avg_n);
            double d = 1.0 - a;
            e = a * le + d * e;

            vector_of_matrix vm_backprop(1);
            RowVector g = loss->LossGradient();
            lvgo_write(LayerList.size(), g);
            for (int i = (int)LayerList.size() - 1; i >= 0; --i) {
               if (i == 0) {
                  vm_backprop[0] = LayerList[i]->BackProp(g);
                  clvgo_write(i, vm_backprop); // Debug
               }
               else {
                  // REVIEW: Add a visitor interface to BackProp that can be used
                  //         to produce metric's such as scale of dW.
                  g = LayerList[i]->BackProp(g);
                  lvgo_write(i, g); // Debug
               }
            }

            for (int i = (int)ConvoLayerList.size() - 1; i >= 0; --i) {
               if (i == 0) {
                  ConvoLayerList[i]->BackProp(vm_backprop, false);
                  clvgo_write(i, dynamic_pointer_cast<FilterLayer2D>(ConvoLayerList[i])->dW);  // Debug
               }
               else {
                  vm_backprop = ConvoLayerList[i]->BackProp(vm_backprop);
                  clvgo_write(i, vm_backprop); // Debug
               }
            }

            // This is stoastic descent here.  It is inside the batch loop.
            //eta = (1.0 / (1.0 + 0.001 * loop)) * eta;
            for (auto lli : ConvoLayerList) {
               lli->Update(eta);
            }
            for (auto lit : LayerList) {
               lit->Update(eta);
            }

            // This is to try
            // to be many ittereations away from where
            // the solution blew up.
            if (!(nn++ % 20)) {
               //cout << "Cache at " << nn << endl;
               for (auto cli : ConvoLayerList) {
                  auto ist = dynamic_pointer_cast<iStash>(cli);
                  if (ist) { ist->StashWeights(); }
               }

               for (auto lli : LayerList) {
                  auto ist = dynamic_pointer_cast<iStash>(lli);
                  if (ist) { ist->StashWeights(); }
               }
            }
         }
         err_out.Write(e);
         cout << "count: " << loop << "\terror:" << left << setw(9) << std::setprecision(4) << e << "\tcorrect: " << stat_class.Correct << "\tincorrect: " << stat_class.Incorrect << endl;
         stat_class.Reset();
      }
   }

   TESTJMP:

   MNISTReader reader1( dataroot + "\\test\\t10k-images-idx3-ubyte",
                        dataroot + "\\test\\t10k-labels-idx1-ubyte");

   stat_class.Reset();

   ColVector X;
   ColVector Y;

   double avg_e = 0.0;
   int count = 0;

   while (reader1.read_next()) {
      X = reader1.data();
      Y = reader1.label();
      vector_of_matrix m(1);
      m[0].resize(28, 28);
      TrasformMNISTtoMatrix(m[0], X);
      ScaleToOne(m[0].data(), (int)(m[0].rows() * m[0].cols()));

            //Matrix temp(28, 28);
            //ConvertToLogPolar(m[0], temp);
            //m[0] = temp;

      for (auto lli : ConvoLayerList) {
         m = lli->Eval(m);
      }

      ColVector cv;
      cv = LayerList[0]->Eval(m[0].col(0));
      for (int i = 1; i < LayerList.size(); i++) {
         cv = LayerList[i]->Eval(cv);
      }
      
      stat_class.Eval(cv, Y);
      
   }

   std::cout << " correct/incorrect " << stat_class.Correct << " , " << stat_class.Incorrect << endl;
   std::cout << "Save? y/n:  ";
   char c;
   std::cin >> c;
   if (c == 'y') {
      SaveModelWeights();
   }
}
*/

void Test(string dataroot)
{
   InitModel(true);

   MNISTReader reader(dataroot + "\\test\\t10k-images-idx3-ubyte",
                      dataroot + "\\test\\t10k-labels-idx1-ubyte");

   ClassifierStats stat_class;

   ColVector X;
   ColVector Y;

   double avg_e = 0.0;
   int count = 0;

   LogPolarSupportMatrix lpsm = PrecomputeLogPolarSupportMatrix(28, 28, 28, 28);

   while (reader.read_next()) {
      X = reader.data();
      Y = reader.label();
      vector_of_matrix m(1);

      Matrix temp(28, 28);
      m[0].resize(INPUT_ROWS, INPUT_COLS);
      TrasformMNISTtoMatrix(m[0], X);
      ScaleToOne(m[0].data(), (int)(temp.rows() * temp.cols()));
#ifdef LOGPOLAR
            ConvertToLogPolar(temp, m[0], lpsm);
#else
            m[0] = temp;
#endif
      for (auto lli : ConvoLayerList) {
         m = lli->Eval(m);
      }

      ColVector cv;
      cv = LayerList[0]->Eval(m[0].col(0));
      for (int i = 1; i < LayerList.size(); i++) {
         cv = LayerList[i]->Eval(cv);
      }

      double e = loss->Eval(cv, Y);
      stat_class.Eval(cv, Y);
      /*
      if (++count == 10) {
         count = 0;
         std::cout << " correct/incorrect " << stat_class.Correct << " , " << stat_class.Incorrect << endl;
      }
      */
   }
   std::cout << " correct/incorrect " << stat_class.Correct << " , " << stat_class.Incorrect << endl;
}

void MakeLogPolar(string name1, string name_out)
{
   Matrix m = ReadImage( name1 );
   ScaleToOne(m.data(), (int)(m.rows() * m.cols()));

   Matrix lp1(INPUT_ROWS, INPUT_COLS);

   LogPolarSupportMatrix lpsm = PrecomputeLogPolarSupportMatrix(m.rows(), m.cols(), lp1.rows(), lp1.cols());

   ConvertToLogPolar(m, lp1, lpsm);

   OMultiWeightsBMP oo(path, name_out);
   oo.Write(lp1, 0);
}

void CompareLogPolar(string name1, string name2, string name_out)
{
   Matrix m = ReadImage( name1 );
   ScaleToOne(m.data(), (int)(m.rows() * m.cols()));

   Matrix lp1(128, 128);

   LogPolarSupportMatrix lpsm = PrecomputeLogPolarSupportMatrix(m.rows(), m.cols(), lp1.rows(), lp1.cols());

   ConvertToLogPolar(m, lp1, lpsm);

   m.setZero(); // Sanity check.

   m = ReadImage( name2 );
   ScaleToOne(m.data(), (int)(m.rows() * m.cols()));

   Matrix lp2(128, 128);
   ConvertToLogPolar(m, lp2, lpsm);

   Matrix cr(128, 128);
   LinearCorrelate3(lp1, lp2, cr);

   OMultiWeightsBMP oo(path, name_out);
   oo.Write(cr, 0);
   OWeightsCSVFile ocsv(path, name_out);
   ocsv.Write(cr, 0);

}

void Correlate(string name1, string name2, string name_out)
{
   Matrix m1 = ReadImage( name1 );
   ScaleToOne(m1.data(), (int)(m1.rows() * m1.cols()));

   Matrix m2 = ReadImage( name2 );
   ScaleToOne(m2.data(), (int)(m2.rows() * m2.cols()));

   Matrix cr(m1.rows(), m1.cols());

   //LinearCorrelate3(m1, m2, cr);
   
   int cols = m1.cols();
   int rows = m1.rows();
   runtime_assert(rows == m2.rows() && cols == m2.cols());

   Matrix m1p(rows, 2 * cols);
   Matrix m2p(rows, 2 * cols);
   m1p.setZero();
   m2p.setZero();

   m1p.block(0, 0, rows, cols) = m1;
   m2p.block(0, 0, rows, cols) = m2;

   fft2convolve(m1p, m2p, cr, -1);
   
   OMultiWeightsBMP oo(path, name_out);
   oo.Write(cr, 0);
   OWeightsCSVFile ocsv(path, name_out);
   ocsv.Write(cr, 0);
}

void MakeSpectrum(ColVector& x, ColVector& s)
{
   const int N = x.size();
   const int N2 = N >> 1;

   rfftsine(x.data(), N, 1);
   x.array() /= (double)(N2);

   // Real valued 1st and last complex pair.
   s(0) = fabs(x(0));
   // Ignore Nyquist frequency.
   for (int c = 1; c < N2; c++) {
      int p = c << 1;
      double r = x(p);
      double i = x(p + 1);
      s(c) = sqrt(r * r + i * i);
   }
}

void MakeGrad(ColVector& x, ColVector& s)
{
   ColVector d(s.size());

   MakeSpectrum(x, s);
/*
   // Gradient
   d(0) = x(0) / s(0);
   X(1) = 0.0;
   // Ignore Nyquist frequency.
   for (int n = 1; n < (N >> 1); n++) {
      int p = n << 1;
      double r = F(p);
      double i = F(p + 1);
      X(p) = r / S(n);
      X(p + 1) = -i / S(n);
   } */
}

void TestFourierLayer()
{
   // Test computation of dS/dX
   // 1. Generate a signal X.
   // 2. Perturb each of the elements of X.
   // 3. Compute a spectrum S from each of these perturbations.  
   //    This will be used to compute the estimate of the derivitive dS/dX.
   //    Save the Fourier coefficients.
   // 4. Compute the estimate for dX/dS.
   // 5. Compute the gradient based on S, F, and the inverse Fourier transform.

   const int N = 32;
   const int N2 = N >> 1;
   ColVector X(N);
   Matrix J(N2, N);

   //X.setRandom(); // Sets in the range of -1 to 1
   
   double q = 2.0 * M_PI / N;
   for (int i = 0; i < N; i++) {
      X(i) = sin(2 * q * i);
      //X(i) = i / 15;
   }

   ColVector S(N2);
   ColVector S1(N2);
   ColVector F(N);
   double eta = 5.0E-5;
   for (int d = 0; d < N; d++) {
      double v = X[d];
      X[d] = v + eta;
      F = X;
      MakeSpectrum(F, S);

      X[d] = v - eta;
      F = X;
      MakeSpectrum(F, S1);

      S -= S1;
      S /= (2 * eta);

      J.col(d) = S;
   }



   rfftsine(X.data(), X.size(), -1);

   ofstream os(path + "\\spec.csv");
   for (int i = 0; i < S.size(); i++) {
      os << S[i];
      if (i < S.size() - 1) {
         os << "," << endl;
      }
   }
   ofstream o(path + "\\dsdx.csv");
   for (int i = 0; i < X.size(); i++) {
      o << X[i];
      if (i < X.size() - 1) {
         o << "," << endl;
      }
   }

   ofstream os1(path + "\\dsdx.csv");
   os1 << J;
}

//NOTE:
//      Momentum currently implemented.  When momentum is implemented it means that the current
//      dW has to be stashed as well (to be accurate) or dW at least has to be zeroed when
//      the Stash is applied because the Count parameter is not used.  Momentum uses a low pass filter
//      instead of averaging, which itself is a waste of cycles when Stoccastic Descent is used.
//      !!! To turn on momentum define MOMENTUM . !!
//      !!!   MOMENTUM currently defined for this project.  !!!
//      Would be interesting to do a comparison of error curve with and without momentum.

int main(int argc, char* argv[])
{
   try {
      std::cout << "Starting Convolution MNIST\n";
      string dataroot = "C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data";
      //TestSave(); ;
      //TestGradComp(dataroot);
      //TestFourierLayer();
      //exit(0);
      /*
      vector_of_matrix vmx(1);
      vector_of_matrix vmy(1);
      vmx[0].resize(16, 1);

      double q = 2.0 * M_PI / 16;
      for (int i = 0; i < 16; i++) {
         //vmx[0](i, 0) = sin(2 * q * i);
         vmx[0](i, 0) = i / 15;
      }

      //vmx[0].setConstant(1.0);

      cout << "x:" << endl << vmx[0] << endl;

      //poolColSpec pcs(clSize(16, 1), 1);
      //pcs.SetJacobianCallBack(MCB);

      //vmy = pcs.Eval(vmx);

      //cout << "s:" << endl << vmy[0] << endl;
      ColVector f(16);
      f = vmx[0].col(0);
      rfftsine(f.data(), 16, 1);

      ColVector dfy(16);
      dfy.setZero();

      /////////////////////////////////////////////////////////
      //   The discrete foruier transform
      // 
      // The factor is q = 2 * M_PI / N;
      // 
      // n = 0
      // Here the sin term is always sin(0), which is 0.
      for (int k = 0; k < 16; k++) {
         dfy[0] += vmx[0](k, 0);
      }
      // n = 8
      // Here the sin term is always sin(PI * k), which is multiples of PI, which is always 0.
      for (int k = 0; k < 16; k++) {
         dfy[1] += vmx[0](k, 0) * ( (k%2) ? 1.0 : -1.0 );
      }
      for (int n = 1; n < 8; n++) {
         for (int k = 0; k < 16; k++) {
            dfy[2 * n] += vmx[0](k, 0) * cos(q * k * n);
            dfy[2 * n + 1] += vmx[0](k, 0) * sin(q * k * n);
         }
      }

      for (int i = 0; i < 16; i++) {
         cout << dfy[i] << "," << f[i] << endl;
      }
      */

      /*
      vmy[0].setConstant(1.0);
      vmy[0](0, 0) = 0.0;

      cout << "g:" << endl << vmy[0] << endl;

      vmx = pcs.BackProp(vmy);

      cout << "g out:" << endl << vmx[0] << endl;
      exit(0);
      */
      

      //MakeBiasErrorFunction("C:\\projects\\neuralnet\\simplenet\\SNCVMNIST\\bias_error");
      /*
         MNISTReader reader(dataroot + "\\train\\train-images-idx3-ubyte",
                      dataroot + "\\train\\train-labels-idx1-ubyte");
         MNISTReader::MNIST_list dl = reader.read_batch(100);

         MNISTReader::MNIST_Pair d;
         for (auto p : dl) {
            if (GetLabel(p.y) == 2) {
               d = p;
               break;
            }
         }
         
         LogPolarSupportMatrixCenter lpsmc;
         LogPolarSupportMatrix lpsm = PrecomputeLogPolarSupportMatrix(28, 28, INPUT_ROWS, INPUT_COLS, &lpsmc);
         Matrix temp(28, 28);

         TrasformMNISTtoMatrix(temp, d.x);
         ScaleToOne(temp.data(), (int)(temp.rows() * temp.cols()));
         OMultiWeightsBMP ow(path, "manip.rc");
         int cnt = 0;
         for (MatrixManipulator mm(temp, INPUT_ROWS, INPUT_COLS); !mm.isDone(); mm.next())
         {
            ConvertLogPolarToCart(mm.get(), temp, lpsmc);
            ow.Write(temp, cnt);
            cnt++;
         }
      */
      
      if (argc > 1 && string(argv[1]) == "train") {
         if (argc < 3) {
            cout << "Not enough parameters.  Parameters: train | epochs | eta | read stored coefs (0|1) [optional] | dataroot [optional] | path [optional]" << endl;
            return 0;
         }
         double eta = atof(argv[3]);
         int load = 0;
         if (argc > 4) { load = atoi(argv[4]); }
         if (argc > 5) { dataroot = argv[5]; }
         if (argc > 6) { path = argv[6]; }

         Train(atoi(argv[2]), dataroot, eta, load);
         //Train1(atoi(argv[2]), dataroot, eta, load);
      }
      else if(argc > 1 && string(argv[1]) == "findbad") {

         if (argc < 1) {
            cout << "Not enough parameters.  Parameters: findbad | dataroot [optional] | path [optional]" << endl;
            return 0;
         }

         if (argc > 2) { dataroot = argv[2]; }
         if (argc > 3) { path = argv[3]; }

         OutputIncorrects1(dataroot);
      }
      else if(argc > 1 && string(argv[1]) == "smooth") {

         if (argc < 1) {
            cout << "Not enough parameters.  Parameters: smooth | sigma [optional] | path [optional]" << endl;
            return 0;
         }
         
         double sigma = 2.0;
         if (argc > 2) { sigma = atof(argv[2]); }
         if (argc > 3) { path = argv[3]; }

         FilterTop(sigma);
      }
      else if(argc > 1 && string(argv[1]) == "blp") {

         if (argc < 3) {
            cout << "Not enough parameters.  Parameters: blp | label | dataroot [optional] | path [optional]" << endl;
            return 0;
         }
         int label = atoi(argv[2]);
         if (argc > 3) { dataroot = argv[3]; }
         if (argc > 4) { path = argv[4]; }

         MakeCorrelations(dataroot,label);
      }
      else if(argc > 1 && string(argv[1]) == "lp") {

         if (argc < 4) {
            cout << "Not enough parameters.  Parameters: lp |  name 1 | name 2 | path [optional]" << endl;
            return 0;
         }
         string name1, name2;
         name1 = argv[2];
         name2 = argv[3];
         if (argc > 4) { path = argv[4]; }

         MakeLogPolar(name1,name2);
      }
      else if(argc > 1 && string(argv[1]) == "corlp") {

         if (argc < 5) {
            cout << "Not enough parameters.  Parameters: corlp |  name 1 | name 2  |  name 3 | path [optional]" << endl;
            return 0;
         }
         string name1, name2, name3;
         name1 = argv[2];
         name2 = argv[3];
         name3 = argv[4];
         if (argc > 5) { path = argv[5]; }

         CompareLogPolar(name1,name2,name3);
      }
      else if(argc > 1 && string(argv[1]) == "cor") {

         if (argc < 5) {
            cout << "Not enough parameters.  Parameters: cor |  name 1 | name 2 |  name 3 | path [optional]" << endl;
            return 0;
         }
         string name1, name2, name3;
         name1 = argv[2];
         name2 = argv[3];
         name3 = argv[4];
         if (argc > 5) { path = argv[5]; }

         Correlate(name1,name2,name3);
      }
      else if(argc > 1 && string(argv[1]) == "test") {

         if (argc < 1) {
            cout << "Not enough parameters.  Parameters: test | dataroot [optional] | path [optional]" << endl;
            return 0;
         }

         if (argc > 2) { dataroot = argv[2]; }
         if (argc > 3) { path = argv[3]; }

         Test(dataroot);
      }
      else if(argc > 1 && string(argv[1]) == "exp") {

         NetGrowthExp();
      }
      else {
         cout << "Enter a command." << endl;
      }
   }
   catch (std::exception ex) {
      cout << "Error:\n" << ex.what() << endl;
   }
}
