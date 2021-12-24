#pragma once

#include <Eigen>
#include <iostream>
#include <fstream>
#include <list>
#include <stdexcept>

//#define TRACE
#ifdef TRACE
   #define DebugOut( a ) std::cout << a;
#else
   #define DebugOut( a )
#endif

using namespace std;

typedef double Number;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::RowVectorXd RowVector;
typedef Eigen::VectorXd ColVector;

typedef  std::vector<Matrix> vector_of_matrix;
typedef  std::vector<RowVector> vector_of_rowvector;
typedef  std::vector<ColVector> vector_of_colvector;

// -------- The Activation function interface -----------
// Each kind of Activation function must implement this interface.
//
class iActive{
public:
   virtual ~iActive() = 0 {};
   virtual ColVector Eval(const ColVector& x) = 0;
   virtual Matrix Jacobian(void) = 0;
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

class InitWeightsToConstants : public iInitWeights {
public:
   double Weight;
   double Bias;
   InitWeightsToConstants(double weight, double bias) : Weight(weight), Bias(bias) {}
   void Init(Matrix& w) {
      Eigen::Index rows = w.rows();
      Eigen::Index cols = w.cols();
      w.setConstant(Weight);
      w.rightCols(1).setConstant(Bias);
   }
};

class InitWeightsToRandom : public iInitWeights {
public:
   double Scale;
   double Bias;
   bool BiasConstant;
   InitWeightsToRandom(double scale) : Scale(scale), Bias(0.0), BiasConstant(false) {}
   InitWeightsToRandom(double scale, double bias) : Scale(scale), Bias(bias), BiasConstant(true) {}
   void Init(Matrix& w) {
      Eigen::Index rows = w.rows();
      Eigen::Index cols = w.cols();
      w.setRandom();
      w *= Scale;
      if (BiasConstant) {
         w.rightCols(1).setConstant(Bias);
      }
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
         cout << c << endl;
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

class WriteWeightsBinaryBlock : public iOutputWeights {
   ofstream& file;
public:
   WriteWeightsBinaryBlock(ofstream& _file) : file(_file) {}
   bool Write(Matrix& m) {
      if (!file.is_open()) {
         return false;
      }
      assert(0);
      return true;
   }
};

class ReadWeightsBinaryBlock : public iInitWeights {
   ifstream& file;
public:
   ReadWeightsBinaryBlock(ifstream& _file) : file(_file) {}
   void Init(Matrix& m) {
      assert(file.is_open());
      assert(0);
   }
};

class ReLu : public iActive {
   int Size;
   ColVector Z;
   ColVector Temp;
public:
   ReLu(int size) : Size(size),Z(size),Temp(size) {}
   // Z = W x + b.  Z is what is passed into the activation function.
   ColVector Eval(const ColVector& z) {
      // Note that we don't really need to store Z.
      // We only need to store it's derivitive which is 0 or 1.
      // This statement only applies to ReLU.
      Z = z;
      for (int i = 0; i < Size; i++) {
         Temp(i) = z(i) >= 0.0 ? z(i) : 0.0;
      }
      return Temp;
   }
   Matrix Jacobian(void) {
      ColVector J(Size);
      for (int i = 0; i < Size; i++) {
         J(i) = Z(i) > 0.0 ? 1.0 : 0.0;
      }
      return J.asDiagonal();
   }
   int Length() { return Size; }
};

class actSigmoid : public iActive {
   int Size;
   ColVector Z;
   ColVector F;
   double Sigmoid(double s) {
      return 1.0 / (1.0 + exp(-s));
   }
public:
   actSigmoid(int size) : Size(size), Z(size), F(size) {}
   // Z = W x + b.  Z is what is passed into the activation function.
   ColVector Eval(const ColVector& z) {
      // Note that we don't really need to store Z.
      // We only need to store it's derivitive which is 0 or 1.
      // This statement only applies to ReLU.
      Z = z;
      for (int i = 0; i < Size; i++) {
         F(i) = Sigmoid(Z(i));
      }
      return F;
   }
   Matrix Jacobian(void) {
      ColVector J(Size);
      for (int i = 0; i < Size; i++) {
         J(i) = F(i)*(1.0 - F(i));
      }
      return J.asDiagonal();
   }

   int Length() { return Size; }
};

class SoftMax : public iActive {
   int Size;
   ColVector SM;
public:
   SoftMax(int size) : Size(size), SM(size) {}
   ColVector Eval(const ColVector& x) {
      double sum = 0.0;
      for (int i = 0; i < Size; i++) { SM[i] = exp( x[i] ); }
      sum = SM.sum();
      SM /= sum;
      return SM;
   }
   Matrix Jacobian(void) {
      Matrix J(Size, Size);
      for (int r = 0; r < Size; r++) {
         for (int c = 0; c < Size; c++) {
            if (r == c) {
               J(r, c) = SM[r] * (1.0 - SM[r]);
            }
            else {
               J(r, c) = -SM[r] * SM[c];
            }
         }
      }
   return J;
   }

   int Length() { return Size; }
};

class Linear : public iActive {
   int Size;
public:
   Linear(int size) : Size(size) {}
   ColVector Eval(const ColVector& x) {
      return x;
   }
   Matrix Jacobian(void) {
      ColVector J(Size);
      J.setOnes();
      return J.asDiagonal();
   }

   int Length() { return Size; }
};

class Square : public iActive {
   int Size;
   ColVector Z;
public:
   Square(int size) : Size(size) {}
   ColVector Eval(const ColVector& x) {
      Z = x;
      return x*x;
   }
   Matrix Jacobian(void) {
      ColVector J(Size);
      J = 2.0 * Z;
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
   ColVector Temp;
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
      Temp(output_size)
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
      Temp = W * X;
      return pActive->Eval( Temp );
   }

   RowVector BackProp(const RowVector& child_grad, bool want_layer_grad = false ) {
      Count++;
      RowVector layer_grad = child_grad * pActive->Jacobian();
      DebugOut( "W: " << W << endl )
      DebugOut( "child_grad: " << child_grad << " J: " << pActive->Jacobian() << "l grad: " << layer_grad << endl )
      Matrix iter_w_grad = X * layer_grad;

      if (want_layer_grad) {
         cout << iter_w_grad.transpose() << endl;
      }

      DebugOut("X: " << X << "iter_w_grad: " << iter_w_grad << endl )
      double a = 1.0 / (double)Count;
      double b = 1.0 - a;
      grad_W = a * iter_w_grad + b * grad_W;
      // REVIEW: Performance issue??
      return (layer_grad * W).leftCols(InputSize);
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
class FilterLayer2D {
public:
   struct Size {
      int cols;
      int rows;
      Size() {}
      Size(int c, int r) : cols(c), rows(r) {}
   };
   int Count;
   Size InputSize;
   Size OutputSize;
   Size KernelSize;
   int Padding;
   Matrix X;
   Matrix W;
   Matrix grad_W;
   Matrix Temp;
   iActive* pActive;
   std::ofstream fout;
   bool bout;
   FilterLayer2D(Size input_size, int input_padding, int input_channels, Size output_size, Size kernel_size, int kernel_number, iActive* _pActive, shared_ptr<iInitWeights> _pInit, string filename = "") :
      X(input_size.rows + input_padding,input_size.cols + input_padding), 
      W(kernel_size.rows,kernel_size.cols),
      Temp(output_size.rows,output_size.cols),
      grad_W(kernel_size.rows,kernel_size.cols),
      pActive(_pActive),
      InputSize(input_size),
      OutputSize(output_size),
      KernelSize(kernel_size),
      Padding(input_padding)
   {
      if (output_size.rows * output_size.cols != _pActive->Length()) {
         throw runtime_error("The activation size and the Layer output size should match.");
      }

      _pInit->Init(W);
      X.setZero();
      grad_W.setZero();
      Count = 0;

      bout = false;
      if (filename.length() > 0) {
         fout.open(filename, ios::trunc);
         assert(fout.is_open());
         bout = true;
      }
   }
   ~FilterLayer2D() {
      delete pActive;
      fout.close();
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

   Matrix Eval(const Matrix& _x) {
      X.block(Padding,Padding,InputSize.rows,InputSize.cols) = _x;
      LinearCorrelate(X, W, Temp);
      DebugOut("W: " << W << endl << " X: " << X << endl << " W x X: " << Temp << endl)
      Eigen::Map<ColVector> v(Temp.data(), Temp.size());
      v = pActive->Eval(v);
      return Temp;
   }

   Matrix BackProp(Matrix& child_grad, bool want_backprop_grad = true ) {
      Count++;
      Eigen::Map<RowVector> rv_child_grad(child_grad.data(), child_grad.size());
      Matrix m_layer_grad(InputSize.rows, InputSize.cols);
      Eigen::Map<RowVector> layer_grad(m_layer_grad.data(), m_layer_grad.size());
      layer_grad = rv_child_grad * pActive->Jacobian();
      DebugOut("W: " << W << endl)
      DebugOut("child_grad: " << child_grad << " J: " << pActive->Jacobian() << "l grad: " << layer_grad << endl)

      Matrix iter_w_grad(KernelSize.rows, KernelSize.cols);
      LinearCorrelate(X, m_layer_grad, iter_w_grad);

      DebugOut("X: " << X << "iter_w_grad: " << iter_w_grad << endl )
      double a = 1.0 / (double)Count;
      double b = 1.0 - a;
      grad_W = a * iter_w_grad + b * grad_W;
      if (want_backprop_grad) {
         
      }
      return layer_grad; // Not usable upstream.  We just have to return something.
   }

   void Update(double eta) {
      Count = 0;
      W = W - eta * grad_W;
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
   int Correct;
   int Incorrect;
   LossCrossEntropy() {}
   LossCrossEntropy(int input_size, int output_size) :
      X(input_size),
      Y(input_size),
      Size(input_size)
   {
      Correct = 0;
      Incorrect = 0;
   }

   void Init(int input_size, int output_size) {
      X = ColVector(input_size);
      Y = ColVector(input_size),
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
         if (y[i] != 0.0) {y_max_index = i; }

         //                        Prevent undefined results when taking the log of 0
         loss -= y[i] * std::log( std::max(x[i], std::numeric_limits<Number>::epsilon()));
      }
      if (x_max_index == y_max_index) {
         Correct++;
      } else {
         Incorrect++;
      }
      return loss;
   }

   RowVector LossGradiant(void) {
      RowVector g(Size);
      for (int i = 0; i < Size; i++) {
         g[i] = -Y[i] / X[i];
      }
      #ifdef TRACE
      cout << "Loss gradiant: " << g << endl;
      #endif
      return g;
   }

   void ResetCounters()
   {
      Correct = 0;
      Incorrect = 0;
   }
};