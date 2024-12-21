#ifndef _LAYER_H
#define _LAYER_H

#include <Eigen>
#include <iostream>
#include <fstream>
#include <sstream>
#include <list>
#include <stdexcept>
#include <random>
#include <functional>
#include <chrono>

using namespace std;

typedef double Number;
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> Matrix;
typedef Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> iMatrix;
typedef Eigen::Matrix<double,1,Eigen::Dynamic> RowVector;
typedef Eigen::Matrix<double,Eigen::Dynamic, 1> ColVector;
typedef Eigen::Matrix<int,Eigen::Dynamic, 1> iColVector;

typedef  std::vector<Matrix> vector_of_matrix;
typedef  std::vector<RowVector> vector_of_rowvector;
typedef  std::vector<ColVector> vector_of_colvector;

typedef std::vector<iMatrix> vector_of_matrix_i;
typedef std::vector<iColVector> vector_of_colvector_i;
typedef std::vector<double> vector_of_number;

// Note: Circular reference.  Layer uses fft2convo and fft2convo
//       uses Matrix defined here.
#include <fft2convo.h>

//--------- Error Handeling -------------------------------
#define runtime_assert(expression) \
   if(!(expression)){ \
      stringstream s;\
      s << "runtime assert failed: " << #expression \
        << "file: " << __FILE__ << endl \
        << "line: " << __LINE__ << endl; \
      throw runtime_error(s.str());\
   }

// -------- The Activation function interface -----------
// Each kind of Activation function must implement this interface.
//
// The interface is meant to be implemented as a fly-weight pattern.
// The vector passed into the Eval method is used as an input/output 
// parameter.  It carries the so-called Z vector (W*X+B) in, and may be
// unaltered or trasfomed to what is needed by the Jacobian method.
// The caller is responsible for storing the vector until the 
// Jacobian is needed.
// The Resize method is called by the Layer or other parent object to
// initialize the Activation object to the correct size.
//
class iActive{
public:
   virtual ~iActive() = 0 {};
   virtual void Resize(int size) = 0;
   virtual ColVector Eval( ColVector& x) = 0;
   virtual ColVector Eval(Eigen::Map<ColVector>& x) = 0;
   virtual Matrix Jacobian(const ColVector& z) = 0;
   virtual int Length() = 0;
};
// -------------------------------------------------------
// ----------- Optomizer interface -----------------------
class iOptomizer {
public:
   virtual ~iOptomizer() = 0 {};
   virtual void Resize(int rows, int cols, int channels = 1) = 0;
   virtual void Backprop(Matrix& idW, int chn = 0) = 0;
   virtual void BackpropBias(double idB, int chn = 0) = 0;
   virtual Matrix& WeightGrad(int chn = 0) = 0;
   virtual double BiasGrad(int chn = 0) = 0;
   virtual void UpdateComplete() = 0;
   virtual int BeginBackprop() = 0;
   virtual void ResetCount() = 0;
   virtual void Reinit() = 0;
};
// -------------------------------------------------------
// ----------- Weight Matrix IO interfaces ---------------
// iGetWeights:
// Implement the iGetWeights interface by objects that are 
// used to initialize a network layer.  The interface pattern 
// is somewhat analogous to the visitor pattern.  ReadFC is 
// called by the (Fully Connected) Layer class and the ReadConvoXXX 
// functions are called by the (Convolutional) FilterLayer2D class.
// iPutWeights:
// Implement the iPutWeights interface by objects that
// are used by a layer Save method to persist the layer weights.
//
class iGetWeights {
public:
   virtual ~iGetWeights () = 0 {};
   virtual void ReadConvoWeight(Matrix& w, int k) = 0;
   virtual void ReadConvoBias(Matrix& w, int k) = 0;
   // ReadFC allows the implementation to account for the 
   // augmented weight matrix and initialize the bias properly.
   // It is only called by the  Layer class.
   virtual void ReadFC(Matrix& w) = 0;
};

class iPutWeights {
public:
   virtual ~iPutWeights () = 0 {};
   virtual void Write(const Matrix& w, int k) = 0;
};
//---------------------------------------------------------
// ------- The Loss Layer interface ------------------------
class iLossLayer {
public:
   virtual ~iLossLayer() = 0 {};
   virtual Number Eval(const ColVector& x, const ColVector& y) = 0;
   virtual RowVector LossGradient(void) = 0;
};
// ---------------------------------------------------------

//----------- The Network layer interface ----

class iLayer {
public:
   virtual ~iLayer() = 0 {};
   virtual ColVector Eval(const ColVector& _x) = 0;
   virtual RowVector BackProp(const RowVector& child_grad, bool want_backprop_grad = true) = 0;
   virtual void Update(double eta) = 0;
   virtual void Save(shared_ptr<iPutWeights> _pOut) = 0;
};

//-----------------------------------------------------------

//----------- The Convolutional Network layer interface ----
// The convolutional network requires an interface to abstract
// the various network components.  Unlike a fully connected 
// network, a convolutional network is not a homogenious list of 
// objects.  There are convolution, pooling layers, and flattening
// layers, to name a few.  The iConvoLayer interface allows us
// to place the various objects into a single list.
class iConvoLayer {
public:

   virtual ~iConvoLayer () = 0 {};
   virtual vector_of_matrix Eval(const vector_of_matrix& _x) = 0;
   virtual vector_of_matrix BackProp(vector_of_matrix& child_grad, bool want_backprop_grad = true) = 0;
   virtual void Update(double eta) = 0;
   virtual void Save(shared_ptr<iPutWeights> _pOut) = 0;
};

// The convolutional layer as well as many other types of layers need a representation for
// a matrix size.  This defination might be better named Size2D or Rect, but for now, due to
// legacy, it is simply named Size.
// REVIEW: All of these definations should be placed in a namespace.
struct Size {
   int cols;
   int rows;
   Size() : cols(0), rows(0) {}
   Size(int r, int c) : cols(c), rows(r) {}
   Size(const Size& s) {
      rows = s.rows; 
      cols = s.cols;
   }
   inline void Resize(int r, int c) { rows = r; cols = c; }
   inline bool operator==(const Matrix& m) {
      return (m.rows() == rows && m.cols() == cols);
   }
};
typedef Size clSize;
//-----------------------------------------------------------

//---------- The Stash interface ----------------------------
// Object that have learnable parameters and want to offer ability
// to stash and apply stash to those parameters can implement this
// interface.
class iStash {
public:
   virtual ~iStash () = 0 {};
   virtual void StashWeights() = 0;
   virtual void ApplyStash() = 0;
};
//-----------------------------------------------------------

//---------- The Layer Call Back Interface ------------------
// An implementation of this interface can be passed to layer objects
// At certain places in the object code the Properties method is
// called which passes a reference to working variables back to
// the implementor.  These values can be used for debugging and 
// observation.
//
class iCallBackSink
{
public:
   struct CallBackObj {
      union {
         std::reference_wrapper<const int> i;
         std::reference_wrapper<const long> l;
         std::reference_wrapper<const double> d;
         std::reference_wrapper<const Matrix> m;
         std::reference_wrapper<const ColVector> cv;
         std::reference_wrapper<const RowVector> rv;
         std::reference_wrapper<const vector_of_matrix> vm;
         std::reference_wrapper<const vector_of_number> vn;
      };
      enum class Type{
         Int, Long, Doub, Mat, CVec, RVec, VMat , VNum
      };
      Type t;
      CallBackObj() {}
      CallBackObj(const int& x) { i = x; t = Type::Int; }
      CallBackObj(const long& x) { l = x; t = Type::Long; }
      CallBackObj(const double& x) { d = x; t = Type::Doub; }
      CallBackObj(const Matrix& x) { m = x; t = Type::Mat; }
      CallBackObj(const ColVector& x) { cv = x; t = Type::CVec; }
      CallBackObj(const RowVector& x) { rv = x; t = Type::RVec; }
      CallBackObj(const vector_of_matrix& x) { vm = x; t = Type::VMat; }
      CallBackObj(const vector_of_number& x) { vn = x; t = Type::VNum; }
   };
   virtual ~iCallBackSink() = 0 {};
   virtual void Properties( std::map<string, CallBackObj>& props) = 0 {};
};

typedef iCallBackSink::CallBackObj CBObj;

//-----------------------------------------------------------

//---------- Optomizer implementations ----------------------

// Average Optomizer.
// The gradient is the average gradient over the batch.
class optoAverage : public iOptomizer {
   vector_of_matrix dW;
   vector_of_number dB;
   int Count;
public:
   optoAverage() : Count(0){}
   ~optoAverage() {}
   void Resize(int rows, int cols, int channels) {
      // dB is not used by FC Layer.
      dB.resize(channels);
      dW.resize(channels);
      for (Matrix& m : dW) { m.resize(rows, cols); }
   }
   void Backprop(Matrix& idW, int chn) {
      if (Count > 1) {
         const double a = 1.0 / Count;
         const double b = 1.0 - a;
         dW[chn] = a * idW + b * dW[chn];
      }
      else {
         dW[chn] = idW;
      }
   }
   void BackpropBias(double idB, int chn) {
      if (Count > 1) {
         const double a = 1.0 / Count;
         const double b = 1.0 - a;
         dB[chn] = a * idB + b * dB[chn];
      }
      else {
         dB[chn] = idB;
      }
   }
   Matrix& WeightGrad(int chn) { return dW[chn]; }
   double BiasGrad(int chn) { return dB[chn]; }
   int BeginBackprop() { return ++Count; }
   void ResetCount() { Count = 0; }
   void UpdateComplete() { Count = 0; }
   void Reinit() {}
};

// Low Pass Optomizer. (Like Momentum.)
// The gradient is the average gradient over the batch.
class optoLowPass : public iOptomizer {
   vector_of_matrix dW;
   vector_of_number dB;
public:
   static double Momentum;
   // Momentum should before instantiation.  This is just a 
   // way to make sure that we don't forget to set it.
   optoLowPass() { runtime_assert( Momentum > 0.0) }
   ~optoLowPass() {}
   void Resize(int rows, int cols, int channels) {
      // dB is not used by FC Layer.
      dB.resize(channels);
      for (double& b : dB) { b = 0.0; }
      dW.resize(channels);
      for (Matrix& m : dW) {
         m.resize(rows, cols);
         m.setZero();
      }
   }
   void Backprop(Matrix& idW, int chn) {
      double a = Momentum;  // Valid range 0 - 1
      double b = 1.0 - a;
      // NOTE: The b and a are swapped compared to the average equation above.
      //       This can be confusing.
      //       This is an implementation of a simple IIR low pass filter.
      //dW = b * iter_w_grad + a * dW;
      // Faster implementation (I think).
      dW[chn] += b * (idW - dW[chn]);
   }
   void BackpropBias(double idB, int chn) {
      const double a = 1.0 / Momentum;
      const double b = 1.0 - a;
      dB[chn] += b * (idB - dB[chn]);
   }
   Matrix& WeightGrad(int chn) { return dW[chn]; }
   double BiasGrad(int chn) { return dB[chn]; }
   int BeginBackprop() { return 0; }
   void ResetCount() {}
   void UpdateComplete() {}
   void Reinit() {
      for (double& b : dB) { b = 0.0; }
      for (Matrix& m : dW) { m.setZero(); }
   }
};

// ADAM Optomizer.
// The gradient is the average gradient over the batch.
class optoADAM : public iOptomizer {
   vector_of_matrix T1;
   vector_of_matrix T2;
   vector_of_number N1;
   vector_of_number N2;
   Matrix H;
   int Count;
   int Rows;
   int Cols;
public:
   static double B1;
   static double B2;
   // Momentum should before instantiation.  This is just a 
   // way to make sure that we don't forget to set it.
   optoADAM() : Count(0) { 
      runtime_assert(B1 > 0.0)
      runtime_assert(B2 > 0.0)
   }
   ~optoADAM() {}
   void Resize(int rows, int cols, int channels) {
      Rows = rows;
      Cols = cols;

      H.resize(rows, cols);

      T1.resize(channels);
      for (Matrix& m : T1) { m.resize(rows, cols); m.setZero(); }

      T2.resize(channels);
      for (Matrix& m : T2) { m.resize(rows, cols); m.setZero(); }

      N1.resize(channels);
      for (double& m : N1) { m = 0.0; }

      N2.resize(channels);
      for (double& m : N2) { m = 0.0; }
   }
   void Backprop(Matrix& idW, int chn) {
      T1[chn] = B1 * T1[chn] + (1.0 - B1) * idW;
      T2[chn].array() = B2 * T2[chn].array() + (1.0 - B2) * idW.array().square();
   }
   void BackpropBias(double idB, int chn) {
      N1[chn] = B1 * N1[chn] + (1.0 - B1) * idB;
      N2[chn] = B2 * N2[chn] + (1.0 - B2) * idB * idB;
   }
   Matrix& WeightGrad(int chn) { 
      // REVIEW: Count could be removed by keeping two varaibles that accumulate the 
      //         power.  
      //         B1P = B1P * B1;  Initialize B1P to 1.
      double b1 = 1.0 - std::pow(B1, Count);
      double b2 = 1.0 - std::pow(B2, Count);

      for (int r = 0; r < Rows; r++) {
         for (int c = 0; c < Cols; c++) {
            double h1 = T1[chn](r, c) / b1;
            double h2 = T2[chn](r, c) / b2;
            H(r, c) = h1 / (std::sqrt(h2) + 1.0e-8);
         }
      }

      return H;
   }
   double BiasGrad(int chn) {
      double b1 = 1.0 - std::pow(B1, Count);
      double b2 = 1.0 - std::pow(B2, Count);
      double h1 = N1[chn] / b1;
      double h2 = N2[chn] / b2;
      double h = h1 / (std::sqrt(h2) + 1.0e-8);
      return h;
   }
   int BeginBackprop() { return ++Count; }
   void ResetCount() { Count = 0; }
   void UpdateComplete() {}
   void Reinit() {
      Count = 0;
      for (Matrix& m : T1) { m.setZero(); }
      for (Matrix& m : T2) { m.setZero(); }
      for (double& m : N1) { m = 0.0; }
      for (double& m : N2) { m = 0.0; }
   }
};

//-----------------------------------------------------------

//---------- Weight initializer and output implementations -----

class IWeightsToConstants : public iGetWeights {
   double Weight;
   double Bias;
public:
   IWeightsToConstants(double weight, double bias) : Weight(weight), Bias(bias) {}
   void ReadConvoWeight(Matrix& w, int k) {
      w.setConstant(Weight);
   }
   void ReadConvoBias(Matrix& b, int k){
      b.setConstant(Bias);
   }
   void ReadFC(Matrix& w){
      w.setConstant(Weight);
      w.rightCols(1).setConstant(Bias);
   }
};

class IWeightsToRandom : public iGetWeights {

public:
   double Scale;
   double Bias;
   bool BiasConstant;

   IWeightsToRandom(double scale) : Scale(scale), Bias(0.0), BiasConstant(false){}
   IWeightsToRandom(double scale, double bias) : Scale(scale), Bias(bias), BiasConstant(true){}
   void ReadConvoWeight(Matrix& w, int k) {
      w.setRandom();
      w *= Scale;
   }
   void ReadConvoBias(Matrix& w, int k) {
      if (BiasConstant) {
         w.setConstant(Bias);
      }
      else {
         w.setRandom();
         w *= Scale;
      }
   }
   void ReadFC(Matrix& w){
      // No way to set the random seed in the Eigan implimintation so we get the same random values each run.
      //w.setRandom();
      std::random_device rd;
      std::default_random_engine generator(rd());      std::normal_distribution<double> distribution(0.0, Scale);
      for (int r = 0; r < w.rows(); r++) {
         for (int c = 0; c < w.cols(); c++) {
            w(r, c) = distribution(generator);
         }
      }
      if (BiasConstant) {
         w.rightCols(1).setConstant(Bias);
      }
   }
};

class IWeightsToNormDist : public iGetWeights {
   double StdDev;
   int Channels;
   double Bias;
   /*
   Delving Deep into Rectifiers:
Surpassing Human-Level Performance on ImageNet Classification
   * */
   std::function<double(double, double, int)> stdv_Kaiming = [](double r, double c, int k) {
      return sqrt( 2.0 / (r * c * (double)k) );
   };
   std::function<double(double, double, int)> stdv_Xavier = [](double r, double c, int k) {
      return sqrt( 2.0 / (r + c + k) );
   };

   std::function<double(double, double, int)> fstdv;
public:
   enum InitType{
      Xavier,
      Kaiming
   };
   IWeightsToNormDist(double std_dev, double bias) : StdDev(std_dev), Channels(-1), Bias(bias) {
      // Not used in this mode but we should init fstdv to something.
      fstdv = stdv_Xavier;
   }
   // Xavier recommended for Tanh activation function.
   // Kaiming recommended for ReLu and Sigmoid.
   IWeightsToNormDist(InitType init_type, int channels) : StdDev(-1.0), Channels(channels), Bias(0.0) {
      switch(init_type) {
         case Xavier:      fstdv = stdv_Xavier; break;
         case Kaiming:     fstdv = stdv_Kaiming; break;
         default:          fstdv = stdv_Xavier;
      }
   }
   void ReadConvoWeight(Matrix& w, int k) {
      //double stdv = StdDev < 0.0 ? sqrt(2.0 / (double)(w.rows() * w.cols())) :  StdDev;
      double stdv = StdDev < 0.0 ? fstdv((double)w.rows(),(double)w.cols(),Channels) :  StdDev;

      std::random_device rd;
      std::default_random_engine generator(rd());
      std::normal_distribution<double> distribution(0.0, stdv);
      for (int r = 0; r < w.rows(); r++) {
         for (int c = 0; c < w.cols(); c++) {
            w(r, c) = distribution(generator);
         }
      }
   }
   void ReadConvoBias(Matrix& w, int k) {
      w.setConstant(Bias);
   }
   void ReadFC(Matrix& w){
      double stdv = StdDev < 0.0 ? sqrt(2.0 / (double)w.cols()) :  StdDev;
      std::random_device rd;
      std::default_random_engine generator(rd());
      std::normal_distribution<double> distribution(0.0, stdv);
      for (int r = 0; r < w.rows(); r++) {
         for (int c = 0; c < w.cols(); c++) {
            w(r, c) = distribution(generator);
         }
      }
      w.rightCols(1).setConstant(Bias);
   }
};

class IOWeightsBinaryFile : public iGetWeights, public iPutWeights{
   struct MatHeader {
      int rows;
      int cols;
      int step;
   };
   string Path;
   string RootName;

   void Put(const Matrix& m, string filename) {
      ofstream file(filename,  ios::trunc | ios::binary | ios::out );
      runtime_assert(file.is_open());
      MatHeader header;
      header.rows = (int)m.rows();
      header.cols = (int)m.cols();
      header.step = sizeof(Matrix::Scalar);

      file.write(reinterpret_cast<char*>(&header), sizeof(MatHeader));
      // REVIEW: One line storage I got from the internet.  It didn't work.  Don't know why.
      //file.write(reinterpret_cast<char*>(m.data()), header.step*header.cols*header.rows);
      for (int r = 0; r < header.rows; r++) {
         for (int c = 0; c < header.cols; c++) {
            double v = m(r, c);
            file.write((char*)&v, sizeof(double));
         }
      }
      file.close();
   }

   void Get(Matrix& m, string filename) {
      ifstream file(filename, ios::in | ios::binary );
      runtime_assert(file.is_open());

      MatHeader header;
      file.read(reinterpret_cast<char*>((&header)), sizeof(MatHeader));
      runtime_assert(header.step == sizeof(typename Matrix::Scalar));
      runtime_assert(header.rows == m.rows());
      runtime_assert(header.cols == m.cols());

      for (int r = 0; r < header.rows; r++) {
         for (int c = 0; c < header.cols; c++) {
            double v;
            file.read((char*)&v, sizeof(double));
            m(r, c) = v;
         }
      }
      // REVIEW: One line storage I got from the internet.  It didn't work.  Don't know why.
      //file.read(reinterpret_cast<char*>(m.data()), header.step * header.cols * header.rows);

      file.close();
   }

public:
   IOWeightsBinaryFile(string path, string root_name) : RootName(root_name), Path(path) {}
   void Write(const Matrix& m, int k) {
      string str_count;
      str_count = to_string(k);
      string pathname = Path + "\\" + RootName + "." + str_count + ".wts";
      Put(m,pathname);
   }
   void ReadConvoWeight(Matrix& w, int k) {
      string str_count;
      str_count = to_string(k);
      string pathname = Path + "\\" + RootName + "." + str_count + ".wts";
      Get(w, pathname);
   }
   void ReadConvoBias(Matrix& w, int k) {
      ReadConvoWeight(w, k);
   }
   void ReadFC(Matrix& w) {
      ReadConvoWeight(w, 1);
   }
};

class OWeightsCSVFile : public iPutWeights{
   string Path;
   string RootName;

   void Put(const Matrix& m, string filename) {
      ofstream file(filename,  ios::trunc | ios::out );
      runtime_assert(file.is_open());
      int rows = (int)m.rows();
      int cols = (int)m.cols();
      for (int r = 0; r < rows; r++) {
         for (int c = 0; c < cols; c++) {
            file << m(r, c);
            if (c != (cols - 1)) { file << ","; }
         }
         file << endl;
      }
      file.close();
   }

public:
   OWeightsCSVFile(string path, string root_name) : RootName(root_name), Path(path) {}
   void Write(const Matrix& m, int k) {
      string str_count;
      str_count = to_string(k);
      string pathname = Path + "\\" + RootName + "." + str_count + ".csv";
      Put(m,pathname);
   }
   void Write(string root_name, const Matrix& m, int k) {
      RootName = root_name;
      Write(m, k);
   }
};
//----------------------------------------------------------------

//--------------------- Activation Functions ---------------------

class actReLU : public iActive {
   int Size;
   ColVector Temp;
public:
   actReLU(int size) : Size(size),Temp(size) {}
   actReLU() : Size(0) {}
   void Resize(int size) {
      Size = size;
      Temp.resize(size);
   }
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

class actLeakyReLU : public iActive {
   int Size;
   const double Alpha;
   ColVector Temp;
public:
   actLeakyReLU(int size, double alp) : Size(size),Temp(size),Alpha(alp) {}
   actLeakyReLU(double alp) : Size(0), Alpha(alp) {}
   void Resize(int size) {
      Size = size;
      Temp.resize(size);
   }
   // Z = W x + b.  Z is what is passed into the activation function.
   // It can be modified for use by the Jacobian later.
   ColVector Eval(ColVector& z) {
      for (int i = 0; i < Size; i++) {
         Temp(i) = z(i) >= 0.0 ? z(i) : Alpha * z(i);
      }
      return Temp;
   }
   ColVector Eval(Eigen::Map<ColVector>& z) {
      for (int i = 0; i < Size; i++) {
         Temp(i) = z(i) >= 0.0 ? z(i) : Alpha * z(i);
      }
      return Temp;
   }
   Matrix Jacobian(const ColVector& z) {
      for (int i = 0; i < Size; i++) {
         Temp(i) = z(i) > 0.0 ? 1.0 : Alpha;
      }
      return Temp.asDiagonal();
   }
   int Length() { return Size; }
};

class actSigmoid : public iActive {
   int Size;
   ColVector J;
   double Sigmoid(double s) {
      return 1.0 / (1.0 + exp(-s));
   }
public:
   actSigmoid(int size) : Size(size), J(size) {}
   actSigmoid() : Size(0) {}
   void Resize(int size) {
      Size = size;
      J.resize(size);
   }
   // Z = W x + b.  Z is what is passed into the activation function.
   ColVector Eval(ColVector& q) {
      // Vector q carries the z vector on the way in, then is trasformed
      // to the values of the Sigmoid on the way out.  It will later
      // be passed into the Jacobian method and used to compute the Jacobian.
      for (int i = 0; i < Size; i++) {
         q(i) = Sigmoid(q(i));
      }
      return q;
   }
   ColVector Eval(Eigen::Map<ColVector>& q) {
      for (int i = 0; i < Size; i++) {
         q(i) = Sigmoid(q(i));
      }
      return q;
   }
   Matrix Jacobian(const ColVector& q) {
      for (int i = 0; i < Size; i++) {
         J(i) = q(i)*(1.0 - q(i));
      }
      return J.asDiagonal();
   }

   int Length() { return Size; }
};

class actTanh : public iActive {
   int Size;
   ColVector J;
   double Tanh(double s) {
      double tp = exp(s);
      double tn = exp(-s);
      //tanh = (exp(s) - exp(-s)) / (exp(s) + exp(-s))
      return (tp - tn) / (tp + tn);
   }
public:
   actTanh(int size) : Size(size), J(size) {}
   actTanh() : Size(0) {}
   void Resize(int size) {
      Size = size;
      J.resize(size);
   }
   // Z = W x + b.  Z is what is passed into the activation function.
   ColVector Eval(ColVector& q) {
      // Vector q carries the z vector on the way in, then is trasformed
      // to the values of the Tanh on the way out.  It will later
      // be passed into the Jacobian method and used to compute the jacobian.
      for (int i = 0; i < Size; i++) {
         q(i) = Tanh(q(i));
      }
      return q;
   }
   ColVector Eval(Eigen::Map<ColVector>& q) {
      for (int i = 0; i < Size; i++) {
         q(i) = Tanh(q(i));
      }
      return q;
   }
   Matrix Jacobian(const ColVector& q) {
      for (int i = 0; i < Size; i++) {
         J(i) = 1.0 - q(i)*q(i);
      }
      return J.asDiagonal();
   }

   int Length() { return Size; }
};

class actSoftMax : public iActive {
   int Size;
   Matrix J;
public:
   actSoftMax(int size) : Size(size), J(size, size) {}
   actSoftMax() : Size(0) {}
   void Resize(int size) {
      Size = size;
      J.resize(size, size);
   }
   ColVector Eval(ColVector& q) {
      double sum = 0.0;
      for (int i = 0; i < Size; i++) { q(i) = exp( q(i) ); }
      sum = q.sum();
      // NOTE: Use L'Hopital's rule.
      //       S = x0 + x1 .. + xi + .. xN
      //       xi = S - x0 - x1 - ... - xN
      //       Lim as S -> 0 of xi / S = (d xi / dS) / (d S / dS) = 1 / 1
      //
      if (sum <= std::numeric_limits<double>::epsilon()) {
         q.setOnes();
      }
      else {
         q /= sum;
      }

      return q;
   }
   ColVector Eval(Eigen::Map<ColVector>& q) {
      double sum = 0.0;
      for (int i = 0; i < Size; i++) { q(i) = exp( q(i) ); }
      sum = q.sum();
      //if (sum <= std::numeric_limits<double>::epsilon()) {
      // q has to sum to one.  Only one element can be set to 1.
      //   q.setOnes();
      //}
      //else {
         q /= sum;
      //}
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

class actLinear : public iActive {
   int Size;
   ColVector J;
public:
   actLinear(int size) : Size(size), J(size) {
      J.setOnes();
   }
   actLinear() : Size(0) {}
   void Resize(int size) {
      Size = size;
      J.resize(size);
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

class actSquare : public iActive {
   int Size;
   ColVector J;
public:
   actSquare(int size) : Size(size), J(size) {}
   actSquare() : Size(0) {}
   void Resize(int size) {
      Size = size;
      J.resize(size);
   }
   ColVector Eval(ColVector& x) {
      return x*x;
   }
   ColVector Eval(Eigen::Map<ColVector>& x) {
      return x*x;
   }
   Matrix Jacobian(const ColVector& z) {
      J = 2.0 * z;
      return J.asDiagonal();
   }

   int Length() { return Size; }
};

// Fully Connected Layer ----------------------------------------
class Layer : 
   public iLayer,
   public iStash {
public:
   int InputSize;
   int OutputSize;
   ColVector X;
   ColVector Z;
   Matrix W;
   Matrix stash_W;
   shared_ptr<iOptomizer> pOptomizer;
   unique_ptr<iActive> pActive;
   enum CallBackID { EvalPreActivation, EvalPostActivation, Backprop };
private:
   shared_ptr<iCallBackSink> EvalPreActivationCallBack;
   shared_ptr<iCallBackSink> EvalPostActivationCallBack;
   shared_ptr<iCallBackSink> BackpropCallBack;
public:
   Layer(int input_size, int output_size, unique_ptr<iActive> _pActive, shared_ptr<iGetWeights> _pInit, shared_ptr<iOptomizer> _pOptomizer = make_shared<optoAverage>()) :
      // Add an extra row to align with the bias weight.
      // This row should always be set to 1.
      X(input_size+1), 
      // Add an extra column for the bias weight.
      W(output_size, input_size+1),
      pActive(move(_pActive)),
      InputSize(input_size),
      OutputSize(output_size),
      Z(output_size),
      pOptomizer(_pOptomizer)
   {
      pActive->Resize(output_size); 
      pOptomizer->Resize(input_size + 1, output_size);
      _pInit->ReadFC(W);
   }

   ~Layer() {}

   void StashWeights() {
      stash_W.resize(W.rows(), W.cols());
      stash_W = W;
   }

   void ApplyStash() {
      W = stash_W;
      pOptomizer->Reinit();
   }
   
   ColVector Eval(const ColVector& _x) {
      X.topRows(InputSize) = _x;   // Will throw an exception if _x.size != InputSize
      X(InputSize) = 1;            // This accounts for the bias weight.
      Z = W * X;

      if (EvalPreActivationCallBack != nullptr) {
         map<string, CBObj> props;   
         int id = EvalPreActivation;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "X", CBObj(X) });
         props.insert({ "W", CBObj(W) });
         props.insert({ "dW", CBObj(pOptomizer->WeightGrad()) });
         props.insert({ "Z", CBObj(Z) });
         EvalPreActivationCallBack->Properties( props );
      }

      // NOTE: The Active classes use the flyweight pattern.
      //       Z is an in/out variable, its contents may be modified by Eval
      //       but its dimension will not be changed.
      if (EvalPostActivationCallBack != nullptr) {
         ColVector out = pActive->Eval( Z );
         map<string, CBObj> props;
         int id = EvalPostActivation;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "X", CBObj(X) });
         props.insert({ "W", CBObj(W) });
         props.insert({ "dW", CBObj(pOptomizer->WeightGrad()) });
         props.insert({ "Z", CBObj(out) });
         EvalPostActivationCallBack->Properties( props );
         return out;
      }
      return pActive->Eval( Z );
   }

   RowVector BackProp(const RowVector& child_grad, bool want_layer_grad = true ) {
      pOptomizer->BeginBackprop();
      RowVector delta_grad = child_grad * pActive->Jacobian(Z);
      Matrix iter_w_grad = X * delta_grad;

      pOptomizer->Backprop(iter_w_grad);

      if (BackpropCallBack != nullptr) {
         map<string, CBObj> props;
         int id = Backprop;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "X", CBObj(X) });
         props.insert({ "W", CBObj(W) });
         props.insert({ "dC", CBObj(child_grad) });
         props.insert({ "dG", CBObj(delta_grad) });
         props.insert({ "dW", CBObj(pOptomizer->WeightGrad()) });
         props.insert({ "idW", CBObj(iter_w_grad) });
         props.insert({ "Z", CBObj(Z) });
         BackpropCallBack->Properties( props );
      }

      if (want_layer_grad) {
         return (delta_grad * W.block(0,0,OutputSize, InputSize));
      }
      else {
         RowVector dummy;
         return dummy;
      }
   }

   void Update(double eta) {
      W = W - eta * pOptomizer->WeightGrad().transpose();
      pOptomizer->UpdateComplete();
   }

   void Save(shared_ptr<iPutWeights> _pOut) {
      _pOut->Write(W, 1);
      cout << "Weights saved" << endl;
   }

   void SetEvalPreActivationCallBack(shared_ptr<iCallBackSink> icb) {  EvalPreActivationCallBack = icb; }
   void SetEvalPostActivationCallBack(shared_ptr<iCallBackSink> icb) {  EvalPostActivationCallBack = icb; }
   void SetBackpropCallBack(shared_ptr<iCallBackSink> icb) {  BackpropCallBack = icb; }
};
//---------------------------------------------------------------

/*
   std::function<void(Matrix&, Matrix&, Matrix&)> FFT_LinearConvolutionAccumulate = [](Matrix& m, Matrix& h, Matrix& out) {
      return sqrt( 2.0 / (r * c * (double)k) );
   };
   std::function<void(Matrix&, Matrix&, Matrix&)> CART_LinearConvolutionAccumulate = [](Matrix& m, Matrix& h, Matrix& out) {
      return sqrt( 2.0 / (r + c + k) );
   };

   std::function<void(Matrix&, Matrix&, Matrix&)> LinearConvolutionAccumulate;
public:

   Initialize() {
     
      LinearConvolutionAccumulate = FFT_LinearConvolutionAccumulate;
   }

   IWeightsToNormDist(InitType init_type, int channels) : StdDev(-1.0), Channels(channels), Bias(0.0) {
      switch(init_type) {
         case Xavier:      fstdv = stdv_Xavier; break;
         case Kaiming:     fstdv = stdv_Kaiming; break;
         default:          fstdv = stdv_Xavier;
      }
*/

// Convolution Layer classes ------------------------------------

class FilterLayer2D : 
   public iConvoLayer, 
   public iStash 
{
public:
   bool Cyclic_By_Row;
   bool Cyclic_By_Col;
   enum ConvolutionType { CART, FFT };
   ConvolutionType ConvoType = FFT;
   void SetConvolutionParameters(ConvolutionType ct, bool cyclic_row, bool cyclic_col) {
      if ((cyclic_row || cyclic_col) && ct == CART) {
         runtime_assert(0)
      }

      ConvoType = ct;

      Cyclic_By_Row = cyclic_row;
      Cyclic_By_Col = cyclic_col;

      if (ct == FFT) {
         LinearConvolutionAccumulate = FFT_LinearConvolutionAccumulate;
         LinearCorrelate = FFT_LinearCorrelate;
      }
      else {
         LinearConvolutionAccumulate = CART_LinearConvolutionAccumulate;
         LinearCorrelate = CART_LinearCorrelate;
      }
   }
   ConvolutionType GetConvoDomain() { return ConvoType; }
   bool IsCyclicByRow() { return Cyclic_By_Row; }
   bool IsCyclicByCol() { return Cyclic_By_Col; }

private:
   std::function<void(Matrix&, Matrix&, Matrix&)> LinearConvolutionAccumulate;

   std::function<void(Matrix&, Matrix&, Matrix&)> FFT_LinearConvolutionAccumulate = [&](Matrix& m, Matrix& h, Matrix& out)
   {
      // Parameter 3: -1 = correlate | 1 = convolution
      // Parameter 4 and 5 are defined as "force padding".  Force padding is used to insure a linear
      // convolution, thus if you want a cyclical convolution pass in false.  That is the reason for
      // the not operator below.
      fft2convolve(m, h, out, 1, !Cyclic_By_Row, !Cyclic_By_Col, true);
   };


   std::function<void(Matrix&, Matrix&, Matrix&)> CART_LinearConvolutionAccumulate = [&](Matrix& m, Matrix& h, Matrix& out)
   {
      // Cartesion cyclic operation is not implemented.
      assert(!Cyclic_By_Row && !Cyclic_By_Col);

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
      
      // Nested Lambda function. 
      auto MultiplyReverseBlock = [](Matrix& m, int mr, int mc, Matrix& h, int hr, int hc, int size_r, int size_c)
      {
         double sum = 0.0;
         for (int r = 0; r < size_r; r++) {
            for (int c = 0; c < size_c; c++) {
               sum += m(mr - r, mc - c) * h(hr + r, hc + c);
            }
         }
         return sum;
      };

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
               //out(r, c) = 0.0;
               ;
            }
            else {
               //cout << m1r << "," << m1c << "," << h1r << "," << h1c << "," << shr << "," << shc << "," << endl;
               out(r, c) += MultiplyReverseBlock(m, m2r, m2c, h, h1r, h1c, shr, shc);
            }
         }
      }
   };

   std::function<void(Matrix&, Matrix&, Matrix&, double )> LinearCorrelate;

   std::function<void(Matrix&, Matrix&, Matrix&, double)> FFT_LinearCorrelate = [&](Matrix& m, Matrix& h, Matrix& out, double bias = 0.0)
   {
      // Parameter 3: -1 = correlate | 1 = convolution
      // Parameter 4 and 5 are defined as "force padding".  Force padding is used to insure a linear
      // convolution, thus if you want a cyclical convolution pass in false.  That is the reason for
      // the not operator below.
      fft2convolve(m, h, out, -1, !Cyclic_By_Row, !Cyclic_By_Col, false);
      if (bias != 0.0) {
         out.array() += bias;
      }
   };

   std::function<void(Matrix&, Matrix&, Matrix&, double)> CART_LinearCorrelate = [&](Matrix& m, Matrix& h, Matrix& out, double bias = 0.0)
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

      //****************************************
      // Nested Lambda function. 
      auto MultiplyBlock = [](Matrix& m, int mr, int mc, Matrix& h, int hr, int hc, int size_r, int size_c)
      {
         double sum = 0.0;
         for (int r = 0; r < size_r; r++) {
            for (int c = 0; c < size_c; c++) {
               sum += m(mr + r, mc + c) * h(hr + r, hc + c);
            }
         }
         return sum;
      };

      auto MultiplyBlockWithEigen = [](Matrix& m, int mr, int mc, Matrix& h, int hr, int hc, int size_r, int size_c)
      {
         double sum = (m.array().block(mr, mc, size_r, size_c) * h.array().block(hr, hc, size_r, size_c)).sum();
         return sum;
      };
      //****************************************

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
   };

public:
   int FilterChannels;
   Size InputSize;
   Size OutputSize;
   Size KernelSize;
   int OutputChannels;
   int Channels;
   // Vector of input matrix.  One per channel.
   vector_of_matrix X;
   // Vector of kernel matrix.  There are KernelPerChannel * input channels.
   vector_of_matrix W;
   // Vector of kernel matrix stash.
   vector_of_matrix stash_W;
   // The optomizer produces the final gradient to be applied to the decent method.
   shared_ptr<iOptomizer> pOptomizer;
   // The bias vector.  One bias for each kernel.
   vector_of_number B;
   // The bias vector stash.
   vector_of_number stash_B;
   // The bias gradient vector.
   vector_of_number dB;
   // Vector of activation complementary matrix.  It is length of output channels and each
   // Matrix is size output_size.
   // The values may be the convolution prior to activation or something else.  The activation
   // objects use the fly-weight pattern and this is the storage for that.
   vector_of_matrix Z;
   unique_ptr<iActive> pActive;
   bool NoBias;
   enum CallBackID { EvalPreActivation, EvalPostActivation, Backprop1, Backprop2 };
private:
   shared_ptr<iCallBackSink> EvalPreActivationCallBack;
   shared_ptr<iCallBackSink> EvalPostActivationCallBack;
   shared_ptr<iCallBackSink> BackpropCallBack;

public:

   FilterLayer2D(Size input_size, int input_channels, Size output_size, Size kernel_size, int output_channels, unique_ptr<iActive> _pActive, 
                  shared_ptr<iGetWeights> _pInit, shared_ptr<iOptomizer> _pOptomizer = make_shared<optoAverage>(), bool no_bias = false ) :
      X(input_channels), 
      FilterChannels(input_channels * output_channels),
      W(FilterChannels),
      // Aux is helper data for backprop.  It tells how to divide up
      // the incomming error from the lower layer.
      //Aux(FilterChannels),
      stash_W(FilterChannels),
      // There is one bias value for each (volumn) of filters.
      B(output_channels),
      stash_B(output_channels),
      Z(output_channels),
      pActive(move(_pActive)),
      InputSize(input_size),
      OutputSize(output_size),
      KernelSize(kernel_size),
      OutputChannels(output_channels),
      // REVIEW: Rename to InputChannels.
      Channels(input_channels),
      Cyclic_By_Row(false),
      Cyclic_By_Col(false),
      NoBias(no_bias),
      pOptomizer(_pOptomizer)
   {
      LinearConvolutionAccumulate = FFT_LinearConvolutionAccumulate;
      LinearCorrelate = FFT_LinearCorrelate;

      pActive->Resize(output_size.rows * output_size.cols); 
      pOptomizer->Resize(kernel_size.rows, kernel_size.cols, FilterChannels);

      for (Matrix& m : X) { 
         m.resize(input_size.rows, input_size.cols); 
         m.setZero();
      }
      for (int i = 0; i < W.size();i++) {
         W[i].resize(kernel_size.rows,kernel_size.cols); 
         _pInit->ReadConvoWeight(W[i],i);
      }

      if (NoBias) {
         for (double& b : B) { b = 0.0; }
      }
      else {
         Matrix temp(B.size(), 1);
         // NOTE: The bias matrix "index" as it pertains to the storage system
         //       is just a continuation of the of the weight index, which ends
         //       at W.size()-1 .
         _pInit->ReadConvoBias(temp, (int)W.size() );
         for (int i = 0; i < B.size(); i++) {
            B[i] = temp(i, 0);
         }
      }

      for (Matrix& m : Z) { m.resize(output_size.rows, output_size.cols); }
   }

   ~FilterLayer2D() {}

   void StashWeights() {
      vector_of_matrix::iterator istsh = stash_W.begin();
      vector_of_matrix::iterator iw = W.begin();
      for (; iw != W.end();++iw, ++istsh) {
         istsh->resize(KernelSize.rows, KernelSize.cols);
         *istsh = *iw;
         // Example Lambda call using only captures.  All pass by value.
         //[this,iw,istsh]() {
         //   istsh->resize(KernelSize.rows, KernelSize.cols);
         //   *istsh = *iw;
         //}(); //<-- the last () is to call it immeadiatly.  Remove to pass to another function.
         // 
         // It would be very simple to make an interface that accepts functors as 
         // arguments with specification that they not require parameters.
         // The interface would be to a thread pool object.  The functors would be
         // placed in a list if no thread is free and when the list is
         // empty the class would unblock.
         // The interface supports a blocking method to provide a way
         // to wait for the work to complete.
      }

      stash_B = B;
   }

   void ApplyStash() {
      vector_of_matrix::iterator istsh = stash_W.begin();
      vector_of_matrix::iterator iw = W.begin();
      for (; iw != W.end();++iw, ++istsh) {
         *iw = *istsh;
      }

      B = stash_B;

      pOptomizer->Reinit();
   }

   vector_of_matrix Eval(const vector_of_matrix& _x) 
   {
      vector_of_matrix::const_iterator is = _x.begin();
      vector_of_matrix::iterator it = X.begin();
      for ( ; it != X.end(); ++it, ++is) {
         // Copy from matrix of source (_x) to matrix of target (X).
         *it = *is;
      }

      //**********************************
      int chn = 0;
      vector_of_matrix::iterator iw = W.begin();
      vector_of_matrix::iterator iz = Z.begin();
      //vector_of_matrix::iterator ia = Aux.begin();
      vector_of_number::iterator ib = B.begin();
      Matrix temp(OutputSize.rows, OutputSize.cols);

      iz->setZero();

      for (; iw != W.end(); iw++ ) {
         if (chn == Channels) {
            chn = 0;
            iz++;
            iz->setZero();
         }
         LinearCorrelate(X[chn], *iw, temp, 0.0);
         *iz += temp;
         chn++;
      }

      /*
      iz = Z.begin();
      chn = 0;
      for (ia = Aux.begin(); ia != Aux.end(); ia++) {
         if (chn == Channels) {
            chn = 0;
            iz++;
         }
         // Normalize each filter output channel by its sum z.
         for (Eigen::Index r = 0; r < iz->rows(); r++) {
            for (Eigen::Index c = 0; c < iz->cols(); c++) {
               if ((*iz)(r, c) > 0.0) {
                  (*ia)(r, c) /= (*iz)(r, c);
               }
            }
         }
         chn++;
      }
      */
      if (!NoBias) {
         ib = B.begin();
         for (iz = Z.begin(); iz != Z.end(); iz++, ib++) {
            if (*ib != 0.0) {
               iz->array() += *ib;
            }
         }
      }

      //************************************


      vector_of_matrix vecOut(Z.size());

      if (EvalPreActivationCallBack != nullptr) {
         map<string, CBObj> props;   
         int id = EvalPreActivation;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "X", CBObj(X) });
         props.insert({ "W", CBObj(W) });
         props.insert({ "dW", CBObj(pOptomizer->WeightGrad()) });
         props.insert({ "Z", CBObj(Z) });
         EvalPreActivationCallBack->Properties( props );
      }

      for (Matrix& mm : vecOut) { mm.resize(OutputSize.rows, OutputSize.cols); }
      iz = Z.begin();
      vector_of_matrix::iterator io = vecOut.begin();
      for (; iz != Z.end(); ++iz, ++io) {
         // REVIEW: Checkout this link on Eigen sequences.
         //         https://eigen.tuxfamily.org/dox-devel/group__TutorialSlicingIndexing.html
         Eigen::Map<ColVector> z(iz->data(), iz->size());
         Eigen::Map<ColVector> v(io->data(), io->size());
         v = pActive->Eval(z);
      }

      if (EvalPostActivationCallBack != nullptr) {
         map<string, CBObj> props;
         int id = EvalPostActivation;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "X", CBObj(X) });
         props.insert({ "W", CBObj(W) });
         props.insert({ "dW", CBObj(pOptomizer->WeightGrad()) });
         props.insert({ "Z", CBObj(vecOut) });
         EvalPostActivationCallBack->Properties( props );
      }

      return vecOut;
   }

   //
   // BackProp
   // NOTES:
   //    This method is passed the error gradients from the layer below this one.  It
   //    will use gradents to compute update values (dW) for each of its filter (kernel)
   //    matricies.  The optomizer determines how to apply the update values.  Next it
   //    must compute the gradients that will be propagated to the layer above this one.
   // Input:
   //    child_grad  : An array of Matrix the size of OutputChannels with each Matrix having
   //                  the dimention of OutputSize.
   //    want_backprop_grad : It takes computing power to propagate the error matricies through
   //                  this layer to the next.  If this is the top layer then the return
   //                  value will go unused, so why compute it.  This flag lets us avoid that
   //                  computation.
   // Return:
   //    A vector_of_matrix containing the backpropagation gradients of length InputChannels.
   // 
   // Error:
   //    Proper input vector and matrix dimensions are checked for Debug compiles.
   // 
   vector_of_matrix BackProp(vector_of_matrix& child_grad, bool want_backprop_grad = true ) 
   {
      // Initialize the optomizer.  The details are implementation dependant.
      pOptomizer->BeginBackprop();
      // REVIEW: Notes need editing...
      // child_grad will be a vector of OutputChannels.
      // Each of those needs to be multiplied by Jacobian, 
      // then linear correlate X mat with layer_grad
      // then avg sum into dW vec mat.
      // propagate upstream 
      // layer_grad corr=> rotated kernel to size of InputSize
      // sum results according to kernels per channel.
      const int incoming_channels =  OutputChannels;
      assert(child_grad.size() == incoming_channels);

      // Pass the child_grad matricies through the Activation function.  This is done
      // by multipling each Matrix by the Jacobian of the Activation function.  This is
      // not matrix multiplication on the gradient matrix.  The Activation is applied
      // to each element individually and so backprop is also applied to indvidual elements.
      // This is done by 'vectorizing' the Matrix.  In this code the Eigen::Matrix is
      // setup to be stored as row major so just taking the head of the data and
      // calling it a RowVector will turn it into a RowMatirx.
      // 
      // Here we allocate the delta_grad and map a RowVector on it.
      //
      Matrix m_delta_grad(OutputSize.rows, OutputSize.cols);
      
      Matrix m_chan_delta_grad(OutputSize.rows, OutputSize.cols);

      // 
      // child_grad * Jacobian is stored in m_delta_grad.  The computation is made on
      // the RowVector map.
      Eigen::Map<RowVector> rv_delta_grad(m_delta_grad.data(), m_delta_grad.size());

      // This layer's update gradient is stored in iter_dw.
      // It is computed by Correlation between the input matrix X
      // and the delta gradient (m_delta_grad).  Recall that this is 
      // because the derivitive of each of the kernel elements results in
      // this simplification.
      Matrix iter_dW(KernelSize.rows, KernelSize.cols);

      // Allocate the vector_of_matrix for the return but only allocate
      // the matricies if the caller wants them, else we'll return an empty vector.
      // We have to return something!
      vector_of_matrix vm_backprop_grad(Channels);

      if (want_backprop_grad) {
         // The caller wants this information so allocate the matricies.
         for (Matrix& mm : vm_backprop_grad) { 
            mm.resize(InputSize.rows, InputSize.cols); 
            mm.setZero(); // The matrix is an accumulator.
         }
      }

      // Each child_grad is passed through the Activation funciton
      // Jacobian and is then applied to each channel of the filter
      // volumn.  Each filter volumn is of depth InputChannels.


      vector_of_matrix::iterator im = child_grad.begin();
      vector_of_matrix::iterator iz = Z.begin();
      int weight_matrix_count = 0;
      int filter_count = 0;
      for (; im != child_grad.end(); im++, iz++, filter_count++) {
         Eigen::Map<RowVector> rv_child_grad(im->data(), im->size());
         Eigen::Map<ColVector> cv_z(iz->data(), iz->size());
         rv_delta_grad = rv_child_grad * pActive->Jacobian(cv_z);

         // The bias gradient is the sum of the delta matrix.
         double iter_dB = 0.0;
         if (!NoBias) { 
            iter_dB = m_delta_grad.sum(); 
            pOptomizer->BackpropBias(iter_dB, filter_count);
         }

         double div = 1.0 / (double)Channels;
         for (int chn = 0; chn < Channels; chn++) {
            // The convolution volume has a depth equal to the number of input channels
            // and is completed by summing the channel convolutions into one output channel.
            // Backpropagation must now figure out how to propagate a single error gradient
            // back through each of the source channels.
            // Here we simply divide the gradent equally into each channel matrix.
            m_chan_delta_grad = div * m_delta_grad;


            //cout << X[chn] << endl << endl << m_chan_delta_grad << endl<< endl;

            // Recall that rv_delta_grad is a vector map onto m_delta_grad.  Using
            // this channel delta_grad compute the update gradients for the current 
            // filter matrix iter_dW.
            LinearCorrelate(X[chn], m_chan_delta_grad, iter_dW, 0.0);

            //cout << iter_dW << endl << endl;

            if (want_backprop_grad) {
               LinearConvolutionAccumulate(m_chan_delta_grad, W[weight_matrix_count], vm_backprop_grad[chn]);
            }

            pOptomizer->Backprop(iter_dW, weight_matrix_count);

            if (BackpropCallBack != nullptr) {
               map<string, CBObj> props;
               int id = Backprop1;
               props.insert({ "ID", CBObj(id) });
               props.insert({ "OC", CBObj(chn) });  // Output Channel
               props.insert({ "K", CBObj(weight_matrix_count) }); // Kernel
               props.insert({ "cdG", CBObj(*im) }); // Kernel
               props.insert({ "dG", CBObj(m_delta_grad) });
               props.insert({ "idW", CBObj(iter_dW) });
               props.insert({ "dW", CBObj(pOptomizer->WeightGrad(chn)) });
               BackpropCallBack->Properties( props );
            }

            weight_matrix_count++;
         }
      }

      if (BackpropCallBack != nullptr) {
         map<string, CBObj> props;
         int id = Backprop2;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "X", CBObj(X) });
         props.insert({ "W", CBObj(W) });
         props.insert({ "odW", CBObj(vm_backprop_grad) });
         props.insert({ "Z", CBObj(Z) });
         BackpropCallBack->Properties( props );
      }

      return vm_backprop_grad;
   }

   void Update(double eta) {
      for (int i = 0; i < FilterChannels;i++) {
         W[i] = W[i] - eta * pOptomizer->WeightGrad(i);
      }
      if (!NoBias) {
         for (int i = 0; i < OutputChannels; i++) {
            B[i] = B[i] - eta * pOptomizer->BiasGrad(i);
         }
      }
      pOptomizer->UpdateComplete();
   }

   void Save(shared_ptr<iPutWeights> _pOut) {
      for (int i = 0; i < W.size(); i++) {
         _pOut->Write(W[i], i);
      }
      Matrix temp(B.size(), 1);
      for (int i = 0; i < B.size(); i++) {
         temp(i, 0) = B[i];
      }
      _pOut->Write(temp, (int)W.size());

      cout << "Weights saved" << endl;
   }

   void SetEvalPreActivationCallBack(shared_ptr<iCallBackSink> icb) {  EvalPreActivationCallBack = icb; }
   void SetEvalPostActivationCallBack(shared_ptr<iCallBackSink> icb) {  EvalPostActivationCallBack = icb; }
   void SetBackpropCallBack(shared_ptr<iCallBackSink> icb) {  BackpropCallBack = icb; }
};

class poolMax2D : public iConvoLayer {
   
   shared_ptr<iCallBackSink> EvalPostActivationCallBack;
   shared_ptr<iCallBackSink> BackpropCallBack;

public:
   Size InputSize;
   Size OutputSize;
   int Channels;
   // Vector of input matrix.  One per channel.
   vector_of_matrix X;

   // Used for output.
   vector_of_matrix Z;

   // Used for backprop.
   vector_of_matrix_i Zr;
   vector_of_matrix_i Zc;

   poolMax2D(Size input_size, int input_channels, Size output_size) :
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
   ~poolMax2D() {
   }
   void MaxPool( const Matrix& g, Matrix& out, iMatrix& maxr, iMatrix& maxc )
   {
      int rstep = (int)floor((float)g.rows() / (float)out.rows());
      int cstep = (int)floor((float)g.cols() / (float)out.cols());
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

   void BackPool(Matrix& out, const Matrix& dw, const iMatrix& mr, const iMatrix& mc)
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
      if (EvalPostActivationCallBack != nullptr) {
         map<string, CBObj> props;   
         int id = 1;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "X", CBObj(_x) });
         props.insert({ "Z", CBObj(Z) });
         EvalPostActivationCallBack->Properties( props );
      }
      return Z;
   }
   vector_of_matrix BackProp(vector_of_matrix& child_grad, bool want_backprop_grad = true)
   {
      for (int i = 0; i < Channels; i++)
      {
         BackPool(X[i], child_grad[i], Zr[i], Zc[i]);
      }
      if (BackpropCallBack != nullptr) {
         map<string, CBObj> props;  
         int id = 3;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "G", CBObj(child_grad) });
         props.insert({ "X", CBObj(X) });
         BackpropCallBack->Properties( props );
      }
      return X;
   }
   void Update(double eta) 
   {
   }
   void Save(shared_ptr<iPutWeights> _pOut) 
   {
   }

   void SetEvalPostActivationCallBack(shared_ptr<iCallBackSink> icb) {  EvalPostActivationCallBack = icb; }
   void SetBackpropCallBack(shared_ptr<iCallBackSink> icb) {  BackpropCallBack = icb; }
};

class poolAvg2D : public iConvoLayer {
   double Denominator;
   int rstep;
   int cstep;

   shared_ptr<iCallBackSink> EvalPostActivationCallBack;
   shared_ptr<iCallBackSink> BackpropCallBack;

public:
   Size InputSize;
   Size OutputSize;
   int Channels;
   // Vector of input matrix.  One per channel.
   vector_of_matrix X;

   // Used for output.
   vector_of_matrix Z;

   poolAvg2D(Size input_size, int input_channels, Size output_size) :
      X(input_channels),
      Z(input_channels),
      InputSize(input_size),
      OutputSize(output_size),
      Channels(input_channels)
   {
      rstep = (int)floor((float)InputSize.rows / (float)OutputSize.rows);
      cstep = (int)floor((float)InputSize.cols / (float)OutputSize.cols);
      Denominator = rstep * cstep;
      assert(rstep > 0 && cstep > 0);

      for (Matrix& m : X) {
         m.resize(input_size.rows, input_size.cols);
      }

      for (Matrix& m : Z) { m.resize(output_size.rows, output_size.cols); }
   }
   ~poolAvg2D() {
   }
   void AveragePool( const Matrix& g, Matrix& out )
   {
      assert(out.rows() == OutputSize.rows && out.cols() == OutputSize.cols);
      for (int r = 0; r < OutputSize.rows; r++) {
         for (int c = 0; c < OutputSize.cols; c++) {
            int gr = r * rstep;
            int gc = c * cstep;
            double avg = 0.0;
            for (int rr = 0; rr < rstep; rr++) {
               for (int cc = 0; cc < cstep; cc++) {
                  int grr = gr + rr;
                  int gcc = gc + cc;
                  avg += g(grr, gcc) / Denominator;
               }
            }
            out(r, c) = avg;
         }
      }
   }

   void BackPool(Matrix& out, const Matrix& dw)
   {
      // REVIEW: It is possible that a combination of parameters could lead
      //         to a down-sample that does not reach some rows and columns of
      //         the input matrix.  Here that would lead to untouched rows and columns.
      //         Those elements need to be zeroed.  The math could be done so that
      //         the entire matrix does not need to be zeroed.
      out.setZero();
      assert(dw.rows() == OutputSize.rows && dw.cols() == OutputSize.cols);
      for (int r = 0; r < OutputSize.rows; r++) {
         for (int c = 0; c < OutputSize.cols; c++) {
            int gr = r * rstep;
            int gc = c * cstep;
            double ddw = dw(r,c) / Denominator;
            for (int rr = 0; rr < rstep; rr++) {
               for (int cc = 0; cc < cstep; cc++) {
                  int grr = gr + rr;
                  int gcc = gc + cc;
                  out(grr, gcc) = ddw;
               }
            }
         }
      }
   }

   vector_of_matrix Eval(const vector_of_matrix& _x)
   {
      assert(_x.size() == Channels);
      for (int i = 0; i < Channels; i++)
      {
         AveragePool(_x[i], Z[i] );
      }
      if (EvalPostActivationCallBack != nullptr) {
         map<string, CBObj> props;   
         int id = 1;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "X", CBObj(_x) });
         props.insert({ "Z", CBObj(Z) });
         EvalPostActivationCallBack->Properties( props );
      }
      return Z;
   }
   vector_of_matrix BackProp(vector_of_matrix& child_grad, bool want_backprop_grad = true)
   {
      for (int i = 0; i < Channels; i++)
      {
         BackPool(X[i], child_grad[i] );
      }
      if (BackpropCallBack != nullptr) {
         map<string, CBObj> props;  
         int id = 3;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "G", CBObj(child_grad) });
         props.insert({ "X", CBObj(X) });
         BackpropCallBack->Properties( props );
      }
      return X;
   }
   void Update(double eta) 
   {
   }
   void Save(shared_ptr<iPutWeights> _pOut) 
   {
   }
   void SetEvalPostActivationCallBack(shared_ptr<iCallBackSink> icb) {  EvalPostActivationCallBack = icb; }
   void SetBackpropCallBack(shared_ptr<iCallBackSink> icb) {  BackpropCallBack = icb; }
};

class poolMax3D : public iConvoLayer {
   
   shared_ptr<iCallBackSink> EvalPostActivationCallBack;
   shared_ptr<iCallBackSink> BackpropCallBack;

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

   poolMax3D(Size input_size, int input_channels, Size output_size, int output_channels, vector_of_colvector_i& output_map) :
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
   ~poolMax3D() {
   }
   void MaxPool( const vector_of_matrix& g, iColVector& m, Matrix& out, iMatrix& maxr, iMatrix& maxc, iMatrix& maxm )
   {
      int rstep = (int)floor((float)g[0].rows() / (float)out.rows());
      int cstep = (int)floor((float)g[0].cols() / (float)out.cols());
      int rend = (int)(rstep * out.rows());
      int cend = (int)(cstep * out.cols());
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

   void BackPool(vector_of_matrix& out, const Matrix& dw, const iMatrix& maxr, const iMatrix& maxc, const iMatrix& maxm)
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
      if (EvalPostActivationCallBack != nullptr) {
         map<string, CBObj> props;   
         int id = 1;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "X", CBObj(_x) });
         props.insert({ "Z", CBObj(Z) });
         EvalPostActivationCallBack->Properties( props );
      }
      return Z;
   }
   vector_of_matrix BackProp(vector_of_matrix& child_grad, bool want_backprop_grad = true)
   {
      for (Matrix& m : X) {
         m.setZero();
      }
      for (int i = 0; i < OutputChannels; i++)
      {
         BackPool(X, child_grad[i], Zr[i], Zc[i], Zm[i] );
      }
      if (BackpropCallBack != nullptr) {
         map<string, CBObj> props;  
         int id = 3;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "G", CBObj(child_grad) });
         props.insert({ "X", CBObj(X) });
         BackpropCallBack->Properties( props );
      }
      return X;
   }
   void Update(double eta) 
   {
   }
   void Save(shared_ptr<iPutWeights> _pOut) 
   {
   }
   void SetEvalPostActivationCallBack(shared_ptr<iCallBackSink> icb) {  EvalPostActivationCallBack = icb; }
   void SetBackpropCallBack(shared_ptr<iCallBackSink> icb) {  BackpropCallBack = icb; }
};

class poolAvg3D : public iConvoLayer {
   shared_ptr<iCallBackSink> EvalPostActivationCallBack;
   shared_ptr<iCallBackSink> BackpropCallBack;
   inline void callback(shared_ptr<iCallBackSink> icb, int id)
   {
      if (icb != nullptr) {
         map<string, CBObj> props;   
         props.insert({ "ID", CBObj(id) });
         props.insert({ "X", CBObj(X) });
         props.insert({ "Z", CBObj(Z) });
         icb->Properties( props );
      }
   }
public:
   Size InputSize;
   Size OutputSize;
   Size KernelSize;
   int InputChannels;
   int OutputChannels;
   int rstep;
   int cstep;
   vector_of_colvector_i OutputMap;
   // Vector of input matrix.  One per channel.  Used for backprop.
   vector_of_matrix X;

   // Used for output.
   vector_of_matrix Z;

   poolAvg3D(Size input_size, int input_channels, Size output_size, int output_channels, vector_of_colvector_i& output_map) :
      X(input_channels),
      Z(output_channels),
      InputSize(input_size),
      OutputSize(output_size),
      InputChannels(input_channels),
      OutputChannels(output_channels),
      OutputMap(output_map)
   {
      rstep = (int)floor((float)InputSize.rows / (float)OutputSize.rows);
      cstep = (int)floor((float)InputSize.cols / (float)OutputSize.cols);

      for (Matrix& m : X) {
         m.resize(input_size.rows, input_size.cols);
      }

      for (Matrix& m : Z) { m.resize(output_size.rows, output_size.cols); }
   }
   ~poolAvg3D() {
   }
   void AvgPool( const vector_of_matrix& g, const iColVector& m, Matrix& out )
   {
      double denominator = (double)(rstep * cstep * m.size());

      for (int r = 0; r < out.rows(); r++) {
         for (int c = 0; c < out.cols(); c++) {
            int gr = r * rstep;
            int gc = c * cstep;
            double avg = 0.0;
            // Here we are looping over one or more maps.
            // The vector g is the complete list of convolution maps.
            // The M vector tells us which of these maps is pooled together.
            for (int k = 0; k < m.rows(); k++) {
               for (int rr = 0; rr < rstep; rr++) {
                  for (int cc = 0; cc < cstep; cc++) {
                     int grr = gr + rr;
                     int gcc = gc + cc;
                     avg += g[ m(k) ](grr, gcc) / denominator;
                  }
               }
            }
            out(r, c) = avg;
         }
      }
   }

   void BackPool( vector_of_matrix& g, const iColVector& m, const Matrix& dw )
   {
      double denominator = (double)(rstep * cstep * m.size());

      for (int r = 0; r < dw.rows(); r++) {
         for (int c = 0; c < dw.cols(); c++) {
            int gr = r * rstep;
            int gc = c * cstep;
            double ddw = dw(r,c) / denominator;
            for (int k = 0; k < m.rows(); k++) {
               for (int rr = 0; rr < rstep; rr++) {
                  for (int cc = 0; cc < cstep; cc++) {
                     int grr = gr + rr;
                     int gcc = gc + cc;
                     g[ m(k) ](grr, gcc) = ddw;
                  }
               }
            }
         }
      }
   }

   vector_of_matrix Eval(const vector_of_matrix& _x)
   {
      assert(_x.size() == InputChannels);
      for (int i = 0; i < OutputChannels; i++)
      {
         AvgPool(_x, OutputMap[i], Z[i] );
      }
      if (EvalPostActivationCallBack != nullptr) {
         map<string, CBObj> props;   
         int id = 1;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "X", CBObj(_x) });
         props.insert({ "Z", CBObj(Z) });
         EvalPostActivationCallBack->Properties( props );
      }

      return Z;
   }

   vector_of_matrix BackProp(vector_of_matrix& child_grad, bool want_backprop_grad = true)
   {
      for (Matrix& m : X) {
         m.setZero();
      }
      for (int i = 0; i < OutputChannels; i++)
      {
         BackPool(X, OutputMap[i], child_grad[i] );
      }
      if (BackpropCallBack != nullptr) {
         map<string, CBObj> props;  
         int id = 3;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "G", CBObj(child_grad) });
         props.insert({ "X", CBObj(X) });
         BackpropCallBack->Properties( props );
      }
      return X;
   }
   void Update(double eta) 
   {
   }
   void Save(shared_ptr<iPutWeights> _pOut) 
   {
   }

   void SetEvalPostActivationCallBack(shared_ptr<iCallBackSink> icb) {  EvalPostActivationCallBack = icb; }
   void SetBackpropCallBack(shared_ptr<iCallBackSink> icb) {  BackpropCallBack = icb; }
};

class Flatten2D : public iConvoLayer {
   shared_ptr<iCallBackSink> EvalPostActivationCallBack;
   shared_ptr<iCallBackSink> BackpropCallBack;
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

      if (EvalPostActivationCallBack != nullptr) {
         map<string, CBObj> props;   
         int id = 1;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "X", CBObj(D) });
         props.insert({ "Z", CBObj(Z) });
         EvalPostActivationCallBack->Properties( props );
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
      
      if (BackpropCallBack != nullptr) {
         map<string, CBObj> props;  
         int id = 3;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "G", CBObj(g) });
         props.insert({ "Z", CBObj(D) });
         BackpropCallBack->Properties( props );
      }

      return D;
   }
   void Update(double eta) 
   {
   }
   void Save(shared_ptr<iPutWeights> _pOut) 
   {
   }

   void SetEvalPostActivationCallBack(shared_ptr<iCallBackSink> icb) {  EvalPostActivationCallBack = icb; }
   void SetBackpropCallBack(shared_ptr<iCallBackSink> icb) {  BackpropCallBack = icb; }
};

//--------------------------------------------------------------

// ------------ Loss Layer -------------------------------------
class LossL2 : public iLossLayer{
public:
   int Size;
   ColVector Z;

   LossL2() : Size(0) {}
   LossL2(int input_size, int output_size) :
      Z(input_size),
      Size(input_size){}

   void Init(int input_size, int output_size) {
      Z.resize(input_size);
      Size = input_size;
   }
   Number Eval(const ColVector& x, const ColVector& y) {
      Z = x - y;
      return Z.dot(Z);
   }

   // REVIEW: How is the Weight Decay term added to this formulation?
   //         Weight decay is sum of all layer weights squared.  A scalar.
   //         The derivitive seems to be a matrix which is incompatiable 
   //         with the RowVector.
   //         I got it.  If you want to regularize, just add lamda*W to delta W at
   //         each layer before update.
   //         http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/

   RowVector LossGradient(void) {
      return 2.0 * Z.transpose();
   }
};

class LossL4 : public iLossLayer{
public:
   int Size;
   ColVector Z;

   LossL4() : Size(0) {}
   LossL4(int input_size, int output_size) :
      Z(input_size),
      Size(input_size){}

   void Init(int input_size, int output_size) {
      Z.resize(input_size);
      Size = input_size;
   }
   Number Eval(const ColVector& x, const ColVector& y) {
      Z = x - y;
      double* iz = Z.data();
      double* ize = iz + Z.size();
      double sum = 0.0;
      for (; iz < ize; iz++) {
         double v = *iz * *iz;
         sum += v * v;
         *iz = v * *iz;
      }
      return sum;
   }

   // REVIEW: How is the Weight Decay term added to this formulation?
   //         Weight decay is sum of all layer weights squared.  A scalar.
   //         The derivitive seems to be a matrix which is incompatiable 
   //         with the RowVector.
   //         http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/

   RowVector LossGradient(void) {
      return 4.0 * Z.transpose();
   }
};

class LossCrossEntropy : public iLossLayer{
public:
   int Length;
   ColVector X;
   ColVector Y;
   RowVector G;

   LossCrossEntropy() 
   {
      Length = 0;
   }

   LossCrossEntropy(int input_size, int output_size) :
      X(input_size),
      Y(input_size),
      G(input_size),
      Length(input_size)
   {
   }

   void Init(int input_size, int output_size) {
      X.resize(input_size);
      Y.resize(input_size),
      G.resize(input_size),
      Length = input_size;
   }

   Number Eval(const ColVector& x, const ColVector& y) {
      assert(Length > 0);
      X = x;
      Y = y;
      double loss = 0.0;
      for (int i = 0; i < Length; i++) {
         // No reason to evaulate this expression if y[i]==0.0 .
         if (y[i] != 0.0) {
            //                        Prevent undefined results when taking the log of 0
            //double v = x[i] > std::numeric_limits<Number>::epsilon() ? x[i] : std::numeric_limits<Number>::epsilon();
            loss -= y[i] * std::log( x[i] > std::numeric_limits<Number>::epsilon() ? x[i] : std::numeric_limits<Number>::epsilon() );
         }
      }
      return loss;
   }

   RowVector LossGradient(void) {
      for (int i = 0; i < Length; i++) {
         if (X[i] <= std::numeric_limits<double>::epsilon() ) {
            if (Y[i] <= std::numeric_limits<double>::epsilon() ) {
               G[i] = 0.0;
            }
            else {
               //G[i] = -10.0;
               // This may be the wrong value, but it is safe.  The worst it does is that
               // it does not progress the solution.
               G[i] = 0.0;
               cout << "Loss Gradient encountered div by zero" << endl; // Debug
            }
         }
         else {
            G[i] = -Y[i] / X[i];
         }
      }
      return G;
   }
};

class ClassifierStats
{
public:
   int Correct;
   int Incorrect;
   ClassifierStats() : Correct(0), Incorrect(0) {}
   bool Eval(const ColVector& x, const ColVector& y) {
      assert(x.size() == y.size());
      int y_max_index = 0;
      int x_max_index = 0;
      double xmax = 0.0;
      double ymax = 0.0;
      double loss = 0.0;
      for (int i = 0; i < x.size(); i++) {
         if (x[i] > xmax) { xmax = x[i]; x_max_index = i; }
         // This method will handle a y array that is a proper distribution not
         // just one-hot encoded.
         if (y[i] > ymax) { ymax = y[i]; y_max_index = i; }

      }
      if (x_max_index == y_max_index) {
         Correct++;
         return true;
      }

      Incorrect++;
      return false;
   }
   void Reset()
   {
      Correct = 0;
      Incorrect = 0;
   }
};

   class ErrorOutput
   {
      ofstream owf;
   public:
      ErrorOutput(string path, string name, bool append = false) : owf(path + "\\" + name + ".error.csv", append ? ios::app : ios::trunc)
      {
         // Not usually nice to throw an error out of a constructor.
         runtime_assert(owf.is_open());
      }
      void Write(double e)
      {
         owf << e << endl;
      }
   };

   #endif // !_LAYER_H