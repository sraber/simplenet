#pragma once

#include <Eigen>
#include <iostream>
#include <fstream>
#include <list>
#include <stdexcept>

// Dev Notes:
// - The Save functions are not working for convo layers.
// - Save to CSV:  
//   Not implemented for new 2 parameter saving method.

//#define TRACE
#ifdef TRACE
   #define DebugOut( a ) std::cout << a;
#else
   #define DebugOut( a )
#endif

using namespace std;

typedef double Number;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::MatrixXi iMatrix;
typedef Eigen::RowVectorXd RowVector;
typedef Eigen::VectorXd ColVector;
typedef Eigen::VectorXi iColVector;

typedef  std::vector<Matrix> vector_of_matrix;
typedef  std::vector<RowVector> vector_of_rowvector;
typedef  std::vector<ColVector> vector_of_colvector;

typedef std::vector<iMatrix> vector_of_matrix_i;
typedef std::vector<iColVector> vector_of_colvector_i;
typedef std::vector<double> vector_of_number;


// -------- The Activation function interface -----------
// Each kind of Activation function must implement this interface.
//
// The interface is meant to be implemented as a fly-weight pattern.
// The vector passed into the Eval method is used as an input/output 
// parameter.  It carries the so-called Z vector (W*X+B) in and may be
// unaltered or trasfomed to what is needed by the Jacobian method.
// The caller is responsible for storing the vector until the 
// Jacobian is needed.
//
class iActive{
public:
   virtual ~iActive() = 0 {};
   virtual ColVector Eval( ColVector& x) = 0;
   virtual ColVector Eval(Eigen::Map<ColVector>& x) = 0;
   virtual Matrix Jacobian(const ColVector& z) = 0;
   virtual int Length() = 0;
};
// -------------------------------------------------------

class iInitWeights {
public:
   virtual ~iInitWeights() = 0 {};
   virtual void Init(Matrix& w) = 0;
};

class iOutputWeights {
public:
   virtual ~iOutputWeights() = 0 {};
   virtual bool Write(Matrix& w) = 0;
};

class iConvoLayer {
public:
   struct Size {
      int cols;
      int rows;
      Size() {}
      Size(int r, int c) : cols(c), rows(r) {}
   };
   virtual vector_of_matrix Eval(const vector_of_matrix& _x) = 0;
   virtual vector_of_matrix BackProp(vector_of_matrix& child_grad, bool want_backprop_grad = true) = 0;
   virtual void Update(double eta) = 0;
   virtual void Save(shared_ptr<iOutputWeights> _pOut) = 0;
};

class InitWeightsToConstants : public iInitWeights {
public:
   double Weight;
   double Bias;
   InitWeightsToConstants(double weight, double bias) : Weight(weight), Bias(bias) {}
   void Init(Matrix& w) {
      w.setConstant(Weight);
      w.rightCols(1).setConstant(Bias);
   }
};

// REVIEW: This doesn't work well for convo layer.  The bias vector is a seperate vector
//         in a convo layer.  We don't have a way to handel that now.
class InitWeightsToRandom : public iInitWeights {
public:
   double Scale;
   double Bias;
   bool BiasConstant;
   bool IsConvoLayer;
   int Count = 0;
   InitWeightsToRandom(double scale) : Scale(scale), Bias(0.0), BiasConstant(false), IsConvoLayer(false), Count(1) {}
   InitWeightsToRandom(double scale, double bias) : Scale(scale), Bias(bias), BiasConstant(true), IsConvoLayer(false), Count(1) {}
   InitWeightsToRandom(double scale, double bias, int kernels) : Scale(scale), Bias(bias), BiasConstant(true), IsConvoLayer(true), Count(kernels) {}
   void Init(Matrix& w) {
      if (Count < 0) {
         throw("Error.  InitWeightsToRandom not called correctly.");
      }
      // Count only reachs zero for Convo Layers. 
      // Count == 0 means this is a call on the bias matrix.
      if (Count == 0) {
         if (BiasConstant) {
            w.setConstant(Bias);
         }
         else {
            w.setRandom();
            w *= Scale;
         }
      }
      else{
         w.setRandom();
         w *= Scale;
         if (!IsConvoLayer && BiasConstant) {
            w.rightCols(1).setConstant(Bias);
         }
      }
      if (IsConvoLayer) { Count--; }
   }
};

class WriteWeightsCSV : public iOutputWeights {
   string Filename;
public:
   WriteWeightsCSV(string file_name) : Filename(file_name) {}
   bool Write(Matrix& m) {
      ofstream file(Filename, ios::trunc);
      if (!file.is_open()) {
         return false;
      }
      int rows = m.rows();
      int cols = m.cols();
      for (int r = 0; r < rows; r++) {
         for (int c = 0; c < cols; c++) {
            file << m(r, c);
            if (c != (cols - 1)) { file << ","; }
         }
         file << endl;
      }
      file.close();
      return true;
   }
};

class IOWeightsBinary : public iOutputWeights , public iInitWeights{
   struct MatHeader {
      int rows;
      int cols;
      int step;
   };
   string Filename;
   bool Append;
public:
   IOWeightsBinary(string file_name) : Filename(file_name), Append(false) {}
   IOWeightsBinary(string file_name, bool _append) : Filename(file_name), Append(_append) {}
   bool Write(Matrix& m) {
      ofstream file(Filename, (Append ? ios::app : ios::trunc) | ios::binary | ios::out );
      if (!file.is_open()) {
         char c[255];
         strerror_s(c, 255, errno);
         throw(c);
         return false;
      }
      MatHeader header;
      header.rows = m.rows();
      header.cols = m.cols();
      header.step = sizeof(Matrix::Scalar);

      file.write(reinterpret_cast<char*>(&header), sizeof(MatHeader));
      //file.write(reinterpret_cast<char*>(m.data()), header.step*header.cols*header.rows);
      for (int r = 0; r < header.rows; r++) {
         for (int c = 0; c < header.cols; c++) {
            double v = m(r, c);
            file.write((char*)&v, sizeof(double));
         }
      }
      file.close();
      return true;
   }

   void Init(Matrix& m) {
      ifstream file(Filename, ios::in | ios::binary );
      assert(file.is_open());
      MatHeader header;
      file.read(reinterpret_cast<char*>((&header)), sizeof(MatHeader));
      assert(header.step == sizeof(typename Matrix::Scalar));
      assert(header.rows == m.rows());
      assert(header.cols == m.cols());

      for (int r = 0; r < header.rows; r++) {
         for (int c = 0; c < header.cols; c++) {
            double v;
            file.read((char*)&v, sizeof(double));
            m(r, c) = v;
         }
      }
      //file.read(reinterpret_cast<char*>(m.data()), header.step * header.cols * header.rows);

      file.close();
   }
};

class IOMultiWeightsBinary : public iOutputWeights , public iInitWeights{
   string Path;
   string RootName;
   int ReadCount;
   int WriteCount;
public:
   IOMultiWeightsBinary(string path, string root_name) : RootName(root_name), Path(path) {
      ReadCount = 1;
      WriteCount = 1;
   }
   bool Write(Matrix& m) {
      string str_count;
      str_count = to_string(WriteCount);
      WriteCount++;
      string pathname = Path + "\\" + RootName + "." + str_count + ".wts";
      IOWeightsBinary iow(pathname);
      iow.Write(m);
      //cout << "Save " << pathname << endl;
      //cout << m << endl;
      return true;
   }

   void Init(Matrix& m) {
      string str_count;
      str_count = to_string(ReadCount);
      ReadCount++;
      string pathname = Path + "\\" + RootName + "." + str_count + ".wts";

      IOWeightsBinary iow(pathname);
      iow.Init(m);
      //cout << "Read " << pathname << endl;
      //cout << m << endl;
   }
};

class OMultiWeightsCSV : public iOutputWeights{
   string Path;
   string RootName;
   int WriteCount;
public:
   OMultiWeightsCSV(string path, string root_name) : RootName(root_name), Path(path) {
      WriteCount = 1;
   }
   bool Write(Matrix& m) {
      string str_count;
      str_count = to_string(WriteCount);
      WriteCount++;
      string pathname = Path + "\\" + RootName + "." + str_count + ".csv";
      WriteWeightsCSV iow(pathname);
      iow.Write(m);
      //cout << "Save " << pathname << endl;
      //cout << m << endl;
      return true;
   }
};

class ReLu : public iActive {
   int Size;
   ColVector Temp;
public:
   ReLu(int size) : Size(size),Temp(size) {}
   // Z = W x + b.  Z is what is passed into the activation function.
   // It can be modified for use by the Jacobian later.
   ColVector Eval(ColVector& z) {
      for (int i = 0; i < Size; i++) {
         Temp(i) = z(i) >= 0.0 ? z(i) : 0.0;
      }
      return Temp;
   }
   ColVector Eval(Eigen::Map<ColVector>& z) {
      for (int i = 0; i < Size; i++) {
         Temp(i) = z(i) >= 0.0 ? z(i) : 0.0;
      }
      return Temp;
   }
   Matrix Jacobian(const ColVector& z) {
      for (int i = 0; i < Size; i++) {
         Temp(i) = z(i) > 0.0 ? 1.0 : 0.0;
      }
      return Temp.asDiagonal();
   }
   int Length() { return Size; }
};

class actSigmoid : public iActive {
   int Size;
   ColVector SM;
   ColVector J;
   double Sigmoid(double s) {
      return 1.0 / (1.0 + exp(-s));
   }
public:
   actSigmoid(int size) : Size(size), SM(size), J(size) {}
   // Z = W x + b.  Z is what is passed into the activation function.
   ColVector Eval(ColVector& q) {
      // Vector q carries the z vector on the way in, then is trasformed
      // to the values of the Sigmoid on the way out.  It will later
      // be passed into the Jacobian method and used to compute the jacobian.
      for (int i = 0; i < Size; i++) {
         SM(i) = Sigmoid(q(i));
         // REVIEW: Next is to get rid of SM and return q.
         q(i) = SM(i);
      }
      return SM;
   }
   ColVector Eval(Eigen::Map<ColVector>& q) {
      for (int i = 0; i < Size; i++) {
         SM(i) = Sigmoid(q(i));
         // REVIEW: Next is to get rid of SM and return q.
         q(i) = SM(i);
      }
      return SM;
   }
   Matrix Jacobian(const ColVector& q) {
      for (int i = 0; i < Size; i++) {
         J(i) = q(i)*(1.0 - q(i));
      }
      return J.asDiagonal();
   }

   int Length() { return Size; }
};

class SoftMax : public iActive {
   int Size;
   Matrix J;
public:
   SoftMax(int size) : Size(size), J(size, size) {}
   ColVector Eval(ColVector& q) {
      double sum = 0.0;
      for (int i = 0; i < Size; i++) { q(i) = exp( q(i) ); }
      sum = q.sum();
      q /= sum;
      return q;
   }
   ColVector Eval(Eigen::Map<ColVector>& q) {
      double sum = 0.0;
      for (int i = 0; i < Size; i++) { q(i) = exp( q(i) ); }
      sum = q.sum();
      q /= sum;
      return q;
   }

   Matrix Jacobian(const ColVector& q) {
      for (int r = 0; r < Size; r++) {
         for (int c = 0; c < Size; c++) {
            if (r == c) {
               J(r, c) = q[r] * (1.0 - q[r]);
            }
            else {
               J(r, c) = -q[r] * q[c];
            }
         }
      }
   return J;
   }

   int Length() { return Size; }
};

class Linear : public iActive {
   int Size;
   ColVector J;
public:
   Linear(int size) : Size(size), J(size) {
      J.setOnes();
   }
   ColVector Eval(ColVector& x) {
      return x;
   }
   ColVector Eval(Eigen::Map<ColVector>& x) {
      return x;
   }
   Matrix Jacobian(const ColVector& x) {
      return J.asDiagonal();
   }

   int Length() { return Size; }
};

class Square : public iActive {
   int Size;
   ColVector J;
public:
   Square(int size) : Size(size), J(size) {}
   ColVector Eval(ColVector& x) {
      return x*x;
   }
   ColVector Eval(Eigen::Map<ColVector>& x) {
      return x*x;
   }
   Matrix Jacobian(const ColVector& z) {
      J = 2.0 * z;
      DebugOut( "Jacobian" << endl << J << endl )
      return J.asDiagonal();
   }

   int Length() { return Size; }
};


class Layer {
public:
   int Count;
   int InputSize;
   int OutputSize;
   ColVector X;
   ColVector Z;
   Matrix W;
   Matrix grad_W;
   iActive* pActive;
   std::ofstream fout;
   bool bout;
   Layer(int input_size, int output_size, iActive* _pActive, shared_ptr<iInitWeights> _pInit, string filename = "") :
      // Add an extra row to align with the bias weight.
      // This row should always be set to 1.
      X(input_size+1), 
      // Add an extra column for the bias weight.
      W(output_size, input_size+1),
      grad_W(input_size+1, output_size ),
      pActive(_pActive),
      InputSize(input_size),
      OutputSize(output_size),
      Z(output_size)
   {
      if (output_size != _pActive->Length()) {
         throw runtime_error("The activation size and the Layer output size should match.");
      }

      _pInit->Init(W);

      grad_W.setZero();
      Count = 0;

      bout = false;
      if (filename.length() > 0) {
         fout.open(filename, ios::trunc);
         assert(fout.is_open());
         bout = true;
      }
   }
   ~Layer() {
      delete pActive;
      fout.close();
   }

   ColVector Eval(const ColVector& _x) {
      X.topRows(InputSize) = _x;
      X(InputSize) = 1;  // This accounts for the bias weight.
      DebugOut("W: " << W << endl << " X: " << X << endl << " W x: " << W * X << endl)
      Z = W * X;
      return pActive->Eval( Z );
   }

   RowVector BackProp(const RowVector& child_grad, bool want_layer_grad = true ) {
      Count++;
      RowVector delta_grad = child_grad * pActive->Jacobian(Z);
      DebugOut( "W: " << W << endl )
      DebugOut( "child_grad: " << child_grad << " J: " << pActive->Jacobian(Z) << "l grad: " << delta_grad << endl )
      Matrix iter_w_grad = X * delta_grad;

      DebugOut("X: " << X << "iter_w_grad: " << iter_w_grad << endl )
      double a = 1.0 / (double)Count;
      double b = 1.0 - a;
      grad_W = a * iter_w_grad + b * grad_W;
      if (want_layer_grad) {
         // REVIEW: Performance issue??  The block method seems like less multiplications,
         //         unless it has to do a copy.
         //return (layer_grad * W).leftCols(InputSize);
         return (delta_grad * W.block(0,0,OutputSize, InputSize));
      }
      else {
         RowVector dummy;
         return dummy;
      }
   }

   void Update(double eta) {
      Count = 0;
      W = W - eta * grad_W.transpose();
      DebugOut( "-------- UPDATE --------" << endl <<
         "grad T: " << grad_W.transpose() << endl <<
         "W: " << W << endl )
      if (bout) {
         fout << W(0, 0) << "," << W(0, 1) << "," << grad_W(0, 0) << "," << grad_W(1, 0) << endl;
      }

      grad_W.setZero();
   }

   void Save(shared_ptr<iOutputWeights> _pOut) {
      if (_pOut->Write(W) ){
         cout << "Weights saved" << endl;
      }
      else {
         cout << "error: weights not saved" << endl;
      }
   }
};

//------------------------- Convolution Layer -----------

class FilterLayer2D : public iConvoLayer {
public:
   int Count;
   Size InputSize;
   Size OutputSize;
   Size KernelSize;
   int KernelPerChannel;
   int Channels;
   int Padding;
   // Vector of input matrix.  One per channel.
   vector_of_matrix X;
   // Vector of kernel matrix.  There are KernelPerChannel * input channels.
   vector_of_matrix W;
   // Vector of kernel matrix gradients.  There are KernelPerChannel * input channels.
   vector_of_matrix dW;
   // The bias vector.  One bias for each kernel.
   vector_of_number B;
   // The bias gradient vector.
   vector_of_number dB;
   // Vector of activation complementary matrix.  There are KernelPerChannel * input channels.
   // The values may be the convolution prior to activation or something else.  The activation
   // objects use the fly-weight pattern and this is the storage for that.
   vector_of_matrix Z;
   iActive* pActive;
   bool NoBias;

   FilterLayer2D(Size input_size, int input_padding, int input_channels, Size output_size, Size kernel_size, int kernel_number, iActive* _pActive, shared_ptr<iInitWeights> _pInit, bool no_bias = false ) :
      X(input_channels), 
      W(input_channels*kernel_number),
      B(input_channels*kernel_number),
      Z(input_channels*kernel_number),
      dW(input_channels*kernel_number),
      dB(input_channels*kernel_number),
      pActive(_pActive),
      InputSize(input_size),
      OutputSize(output_size),
      KernelSize(kernel_size),
      Padding(input_padding),
      KernelPerChannel(kernel_number),
      Channels(input_channels),
      NoBias(no_bias)
   {
      if (output_size.rows * output_size.cols != _pActive->Length()) {
         throw runtime_error("The activation size and the Layer output size should match.");
      }

      for (Matrix& m : X) { 
         m.resize(input_size.rows + input_padding, input_size.cols + input_padding); 
         m.setZero();
      }
      for (Matrix& m : W) { 
         m.resize(kernel_size.rows,kernel_size.cols); 
         _pInit->Init(m);
      }

      if (NoBias) {
         for (double& b : B) {
            b = 0.0;
         }
      }
      else {
         Matrix temp(B.size(), 1);
         _pInit->Init(temp);
         for (int i = 0; i < B.size(); i++) {
            B[i] = temp(i, 0);
         }
      }

      for (double& db : dB) {
         db = 0.0;
      }
      for (Matrix& m : dW) { 
         m.resize(kernel_size.rows,kernel_size.cols); 
         m.setZero();
      }
      for (Matrix& m : Z) { m.resize(output_size.rows,output_size.cols); }

      Count = 0;
   }
   ~FilterLayer2D() {
      delete pActive;
   }

   void LinearCorrelate( Matrix g, Matrix h, Matrix& out, double bias = 0.0 )
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
            out(r, c) = sum + bias;
         }
      }
   }

   void vecLinearCorrelate()
   {
      int chn = 0;
      int count = 0;
      vector_of_matrix::iterator iw = W.begin();
      vector_of_matrix::iterator iz = Z.begin();
      vector_of_number::iterator ib = B.begin();
      for (; iw != W.end(); iw++, iz++, ib++) {
         if (count == KernelPerChannel) {
            count = 0;
            chn++;
         }
         LinearCorrelate(X[chn], *iw, *iz, *ib);
         count++;
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
   void vecSetXValue(vector_of_matrix& t, const vector_of_matrix& s)
   {
      vector_of_matrix::iterator it = t.begin();
      vector_of_matrix::const_iterator is = s.begin();
      for (; it != t.end(); ++it, ++is) {
         it->block(Padding,Padding,InputSize.rows,InputSize.cols) = *is;
      }
   }

   static void vecDebugOut(string label, vector_of_matrix vom)
   {
      return;  // !!!!!! NOTE !!!!!!!!
      cout << label << endl;
      int count = 0;
      for (Matrix& m : vom) {
         count++;
         cout << "mat max" << count << ": " << m.maxCoeff() << endl;
      }
   }

   vector_of_matrix Eval(const vector_of_matrix& _x) 
   {
      vecSetXValue(X, _x);
      vecLinearCorrelate();
      vecDebugOut("X", X);
      vecDebugOut("W", W);
      vecDebugOut("Z", Z);
      vector_of_matrix vecOut(Z.size());
      for (Matrix& mm : vecOut) { mm.resize(OutputSize.rows, OutputSize.cols); }
      vector_of_matrix::iterator iz = Z.begin();
      vector_of_matrix::iterator io = vecOut.begin();
      for (; iz != Z.end(); ++iz, ++io) {
         // REVIEW: Checkout this link on Eigen sequences.
         //         https://eigen.tuxfamily.org/dox-devel/group__TutorialSlicingIndexing.html
         Eigen::Map<ColVector> z(iz->data(), iz->size());
         Eigen::Map<ColVector> v(io->data(), io->size());
         v = pActive->Eval(z);
         //v = pActive->Eval(iz->data(), iz->size() );
      }
      vecDebugOut("vecOut", vecOut);
      return vecOut;
   }

   // Figure out how many output gradiens there are.  There will be the same or less out going 
   // than in-comming.  Accumulate in-comming gradients into outgoing.
   // There is input_channels*kernel_number in-comming.
   // There are input_channels out-going.
   vector_of_matrix BackProp(vector_of_matrix& child_grad, bool want_backprop_grad = true ) 
   {
      Count++;
      // REVIEW: Notes need editing...
      // child_grad will be a vector of input_channels*kernel_number.
      // Each of those needs to be multiplied by Jacobian, 
      // then linear correlate X mat with layer_grad
      // then avg sum into grad_W vec mat.
      // propagate upstream 
      // layer_grad corr=> rotated kernel to size of InputSize
      // sum results according to kernels per channel.
      const int in_coming_channels =  KernelPerChannel * Channels;
      assert(child_grad.size() == in_coming_channels);

      // child_grad * Jacobian is stored in m_delta_grad.  The computation is made on
      // a row vector map onto m_delta_grad.
      Matrix m_delta_grad(OutputSize.rows, OutputSize.cols);
      Eigen::Map<RowVector> rv_delta_grad(m_delta_grad.data(), m_delta_grad.size());

      // This layer gradient is stored in iter_w_grad.
      // It is computed by Correlation between the input matrix X
      // and the delta gradient (m_delta_grad).  Recall that this is 
      // because the derivitive of each of the kernel elements results in
      // this simplification.
      Matrix iter_dW(KernelSize.rows, KernelSize.cols);

      // The pass through stage (propagating the delta gradient up stream) uses
      // a 180 degree rotation of the Kernel matrix.  It is stored here
      // since we don't want to lose the Kernel matrix.
      Matrix rot_w(KernelSize.rows, KernelSize.cols);

      // Allocate the vector of matrix for the return but only allocate
      // the matricies if the caller wants them, else we'll return an empty vector.
      // We have to return something!
      vector_of_matrix vm_backprop_grad(Channels);
      Matrix pad_delta_grad, temp1;
      // NOTE: A kernel size with rows not equal columns has not been tested.
      //
      // Delta matrix padding is a function of image (input matrix) padding
      // and kernel size.
      int dpr = KernelSize.rows - Padding - 1;
      int dpc = KernelSize.cols - Padding - 1;
      // Delta matrix size with padding.
      int ddr = OutputSize.rows + 2 * dpr;
      int ddc = OutputSize.cols + 2 * dpc;
      if (want_backprop_grad) {
         // The caller wants this information so allocate the matricies.
         for (Matrix& mm : vm_backprop_grad) { 
            mm.resize(InputSize.rows, InputSize.cols); 
            mm.setZero(); // The matrix is an accumulator.
         }
         pad_delta_grad.resize(ddr, ddc);
         pad_delta_grad.setZero();
         temp1.resize(InputSize.rows, InputSize.cols);
      }

      int chn = 0;
      int i = 0;
      for (Matrix& mm : child_grad ) {
         Eigen::Map<RowVector> rv_child_grad(mm.data(), mm.size());
         Eigen::Map<ColVector> cv_z(Z[i].data(), Z[i].size());
         rv_delta_grad = rv_child_grad * pActive->Jacobian(cv_z);

         // The bias gradient is the sum of the delta matrix.
         double iter_dB = 0.0;
         if (!NoBias) { iter_dB = m_delta_grad.sum(); }

         // Recall that rv_delta_grad is a vector map onto m_delta_grad.
         LinearCorrelate(X[chn], m_delta_grad, iter_dW);

         // Average the result (iter_w_grad) into the Kernel gradient (dW).
         double a = 1.0 / (double)Count;
         double b = 1.0 - a;
         dW[i] = a * iter_dW + b * dW[i];
         if (!NoBias) { dB[i] = a * iter_dB + b * dB[i]; }

         if (want_backprop_grad) {
            rot_w = W[i];
            Rotate180(rot_w);
            pad_delta_grad.block(dpr, dpc, OutputSize.rows, OutputSize.cols) = m_delta_grad;
            LinearCorrelate(pad_delta_grad, rot_w, temp1);
            vm_backprop_grad[chn] += temp1;
         }

         if (  !( (i+1) % KernelPerChannel ) ){  chn++; }
         i++;
      }
      return vm_backprop_grad;
   }

   void Update(double eta) {
      Count = 0;
      int maps = Channels * KernelPerChannel;
      for (int i = 0; i < maps;i++) {
         W[i] = W[i] - eta * dW[i];
         dW[i].setZero();

         if (!NoBias) {
            B[i] = B[i] - eta * dB[i];
            dB[i] = 0.0;
         }
      }
   }

   void Save(shared_ptr<iOutputWeights> _pOut) {
      for (Matrix& ww : W) {
         if (!_pOut->Write(ww)) {
            throw("error: weights not saved");
         }
      }
      Matrix temp(B.size(), 1);
      for (int i = 0; i < B.size(); i++){
         temp(i, 0) = B[i];
      }
      if (!_pOut->Write(temp)) {
            throw("error: bias not saved");
      }
      cout << "Weights saved" << endl;
   }


};

class MaxPool2D : public iConvoLayer {
public:
   Size InputSize;
   Size OutputSize;
   Size KernelSize;
   int Channels;
   // Vector of input matrix.  One per channel.
   vector_of_matrix X;

   // Used for output.
   vector_of_matrix Z;

   // Used for backprop.
   vector_of_matrix_i Zr;
   vector_of_matrix_i Zc;

   MaxPool2D(Size input_size, int input_channels, Size output_size) :
      X(input_channels),
      Z(input_channels),
      Zr(input_channels),
      Zc(input_channels),
      InputSize(input_size),
      OutputSize(output_size),
      Channels(input_channels)
   {
      for (Matrix& m : X) {
         m.resize(input_size.rows, input_size.cols);
      }

      for (Matrix& m : Z) { m.resize(output_size.rows, output_size.cols); }
      for (iMatrix& m : Zr) { m.resize(output_size.rows, output_size.cols); }
      for (iMatrix& m : Zc) { m.resize(output_size.rows, output_size.cols); }
   }
   ~MaxPool2D() {
   }
   void MaxPool( Matrix g, Matrix& out, iMatrix& maxr, iMatrix& maxc )
   {
      int rstep = (int)floor((float)g.rows() / (float)out.rows());
      int cstep = (int)floor((float)g.cols() / (float)out.cols());
      int rend = rstep * out.rows();
      int cend = cstep * out.cols();
      for (int r = 0; r < out.rows(); r++) {
         for (int c = 0; c < out.cols(); c++) {
            int gr = r * rstep;
            int gc = c * cstep;
            double max = 0.0;
            int mr = gr;
            int mc = gc;
            for (int rr = 0; rr < rstep; rr++) {
               for (int cc = 0; cc < cstep; cc++) {
                  int grr = gr + rr;
                  int gcc = gc + cc;
                  if ( g(grr, gcc) > max ){
                     max = g(grr, gcc);
                     mr = grr;
                     mc = gcc;
                  }
               }
            }
            out(r, c) = max;
            maxr(r, c) = (Eigen::Index)mr;
            maxc(r, c) = (Eigen::Index)mc;
         }
      }
   }

   void BackPool(Matrix& out, Matrix dw, iMatrix mr, iMatrix mc)
   {
      out.setZero();
      for (int r = 0; r < dw.rows(); r++) {
         for (int c = 0; c < dw.cols(); c++) {
            out(mr(r, c), mc(r, c)) = dw(r, c);
         }
      }
   }

   vector_of_matrix Eval(const vector_of_matrix& _x)
   {
      assert(_x.size() == Channels);
      for (int i = 0; i < Channels; i++)
      {
         MaxPool(_x[i], Z[i], Zr[i], Zc[i]);
      }
      return Z;
   }
   vector_of_matrix BackProp(vector_of_matrix& child_grad, bool want_backprop_grad = true)
   {
      for (int i = 0; i < Channels; i++)
      {
         BackPool(X[i], child_grad[i], Zr[i], Zc[i]);
      }
      return X;
   }
   void Update(double eta) 
   {
   }
   void Save(shared_ptr<iOutputWeights> _pOut) 
   {
   }
};

class MaxPool3D : public iConvoLayer {
public:
   Size InputSize;
   Size OutputSize;
   Size KernelSize;
   int InputChannels;
   int OutputChannels;
   vector_of_colvector_i OutputMap;
   // Vector of input matrix.  One per channel.  Used for backprop.
   vector_of_matrix X;

   // Used for output.
   vector_of_matrix Z;

   // Used for backprop.
   vector_of_matrix_i Zr;
   vector_of_matrix_i Zc;
   vector_of_matrix_i Zm;

   MaxPool3D(Size input_size, int input_channels, Size output_size, int output_channels, vector_of_colvector_i& output_map) :
      X(input_channels),
      Z(output_channels),
      Zr(output_channels),
      Zc(output_channels),
      Zm(output_channels),
      InputSize(input_size),
      OutputSize(output_size),
      InputChannels(input_channels),
      OutputChannels(output_channels),
      OutputMap(output_map)
   {
      for (Matrix& m : X) {
         m.resize(input_size.rows, input_size.cols);
      }

      for (Matrix& m : Z) { m.resize(output_size.rows, output_size.cols); }
      for (iMatrix& m : Zr) { m.resize(output_size.rows, output_size.cols); }
      for (iMatrix& m : Zc) { m.resize(output_size.rows, output_size.cols); }
      for (iMatrix& m : Zm) { m.resize(output_size.rows, output_size.cols); }
   }
   ~MaxPool3D() {
   }
   void MaxPool( const vector_of_matrix& g, iColVector& m, Matrix& out, iMatrix& maxr, iMatrix& maxc, iMatrix& maxm )
   {
      int rstep = (int)floor((float)g[0].rows() / (float)out.rows());
      int cstep = (int)floor((float)g[0].cols() / (float)out.cols());
      int rend = rstep * out.rows();
      int cend = cstep * out.cols();
      for (int r = 0; r < out.rows(); r++) {
         for (int c = 0; c < out.cols(); c++) {
            int gr = r * rstep;
            int gc = c * cstep;
            double max = 0.0;
            int mr = gr;
            int mc = gc;
            int mm = m(0);
            // Here we are looping over one or more maps.
            // The vector g is the complete list of convolution maps.
            // The M vector tells us which of these maps is pooled together.
            for (int k = 0; k < m.rows(); k++) {
               for (int rr = 0; rr < rstep; rr++) {
                  for (int cc = 0; cc < cstep; cc++) {
                     int grr = gr + rr;
                     int gcc = gc + cc;
                     if (g[ m(k) ](grr, gcc) > max) {
                        max = g[ m(k) ](grr, gcc);
                        mr = grr;
                        mc = gcc;
                        mm = k;
                     }
                  }
               }
            }
            out(r, c) = max;
            maxr(r, c) = mr;
            maxc(r, c) = mc;
            maxm(r, c) = mm;

         }
      }
   }

   void BackPool(vector_of_matrix& out, Matrix dw, iMatrix maxr, iMatrix maxc, iMatrix maxm)
   {

      for (int r = 0; r < dw.rows(); r++) {
         for (int c = 0; c < dw.cols(); c++) {
            // It needs to sum (+=) because the map from the layer above may have
            // been pooled into more than one output map.
            out[maxm(r,c)](maxr(r, c), maxc(r, c)) += dw(r, c);
         }
      }
   }

   vector_of_matrix Eval(const vector_of_matrix& _x)
   {
      assert(_x.size() == InputChannels);
      for (int i = 0; i < OutputChannels; i++)
      {
         MaxPool(_x, OutputMap[i], Z[i], Zr[i], Zc[i], Zm[i] );
      }
      return Z;
   }
   vector_of_matrix BackProp(vector_of_matrix& child_grad, bool want_backprop_grad = true)
   {
      for (auto m : X) {
         m.setZero();
      }
      for (int i = 0; i < InputChannels; i++)
      {
         BackPool(X, child_grad[i], Zr[i], Zc[i], Zm[i] );
      }
      return X;
   }
   void Update(double eta) 
   {
   }
   void Save(shared_ptr<iOutputWeights> _pOut) 
   {
   }
};

class Flatten2D : public iConvoLayer {
public:
   Size InputSize;

   int Channels;

   // Used for output.
   vector_of_matrix Z;
   vector_of_matrix D;

   Flatten2D(Size input_size, int input_channels) :
      Z(1),
      D(input_channels),
      InputSize(input_size),
      Channels(input_channels)
   {
      Z[0].resize(input_size.rows * input_size.cols * input_channels,1);
      for (Matrix& m : D) { m.resize(input_size.rows, input_size.cols); }

   }
   ~Flatten2D() {
   }

   // REVIEW:
   // Flatten the convo result.
   // Can't do it this way now.  But...
   // If output matricies were allocated in one stacked matrix, then use
   // vector_of_matrix to keep track of each block (so would be vector_of_block),
   // that stacked matrix could then be flattened with this kind of code
   // and a copy would be avoided.
   // Eigen::Map<ColVector> v(convo_out.data(), convo_out.size());
   // ColVector cv = v;
   vector_of_matrix Eval(const vector_of_matrix& x)
   {
      assert(x.size() == Channels);
      double* pcv = Z[0].data();
      int step = InputSize.rows * InputSize.cols;
      for (int i = 0; i < Channels; i++){
         double* pmm = (double*)x[i].data();
         double* pcve = pcv + step;
         for (; pcv < pcve; pcv++, pmm++) {
            *pcv = *pmm;
         }
      }
      return Z;
   }
   vector_of_matrix BackProp(vector_of_matrix& child_grad, bool want_backprop_grad = true)
   {
      const int step = InputSize.rows * InputSize.cols;
      Matrix& g = child_grad[0];
      assert(g.size() == step * D.size());
      int pos = 0;
      for (Matrix& mm : D) {
         Eigen::Map<RowVector> rv(mm.data(), mm.size());
         rv = g.block(0, pos, 1, step);
         pos += step;
      }

      return D;
   }
   void Update(double eta) 
   {
   }
   void Save(shared_ptr<iOutputWeights> _pOut) 
   {
   }
};

//--------------------------------------------

// ------------ Loss Layer
class LossL2 {
public:
   int Size;
   ColVector Z;

   LossL2(int input_size, int output_size) :
      Z(input_size),
      Size(input_size)
   {}


   Number Eval(const ColVector& x, const ColVector& y) {
      Z = x - y;
      DebugOut( "yh " << x << " y " << y << " Loss Z " << Z.dot(Z) << endl )
      return Z.dot(Z);
   }

   // REVIEW: How is the Weight Decay term added to this formulation?
   //         Weight decay is sum of all layer weights squared.  A scalar.
   //         The derivitive seems to be a matrix which is incompatiable 
   //         with the RowVector.
   //         http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/

   RowVector LossGradiant(void) {
      DebugOut("Loss gradiant : " << 2.0 * Z.transpose() << endl)
      return 2.0 * Z.transpose();
   }
};

class LossCrossEntropy {
public:
   int Size;
   ColVector X;
   ColVector Y;
   RowVector G;

   int Correct;
   int Incorrect;
   LossCrossEntropy() {}
   LossCrossEntropy(int input_size, int output_size) :
      X(input_size),
      Y(input_size),
      G(input_size),
      Size(input_size)
   {
      Correct = 0;
      Incorrect = 0;
   }

   void Init(int input_size, int output_size) {
      X.resize(input_size);
      Y.resize(input_size),
      G.resize(input_size),
      Size = input_size;
      Correct = 0;
      Incorrect = 0;
   }

   Number Eval(const ColVector& x, const ColVector& y) {
      assert(Size > 0);
      X = x;
      Y = y;
      int y_max_index = 0;
      int x_max_index = 0;
      double max = 0.0;
      double loss = 0.0;
      for (int i = 0; i < Size; i++) {
         if (x[i] > max) { max = x[i]; x_max_index = i; }
         if (y[i] != 0.0) {
            y_max_index = i; 
            // No reason to evaulate this expression if y[i]==0.0 .
            //                        Prevent undefined results when taking the log of 0
            loss -= y[i] * std::log( std::max(x[i], std::numeric_limits<Number>::epsilon()));
         }

      }
      if (x_max_index == y_max_index) {
         Correct++;
      } else {
         Incorrect++;
      }
      return loss;
   }

   RowVector LossGradiant(void) {
      //RowVector g(Size);
      for (int i = 0; i < Size; i++) {
         if (X[i] == 0.0 ) {
            G[i] = Y[i] == 0.0 ? 0.0 : -10.0;
         }
         else {
            G[i] = -Y[i] / X[i];
         }
      }
      #ifdef TRACE
      cout << "Loss gradiant: " << G << endl;
      #endif
      return G;
   }

   void ResetCounters()
   {
      Correct = 0;
      Incorrect = 0;
   }
};