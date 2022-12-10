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

//----------- The Convolutional Network layer interface ----
// The convolutional network requires an interface to abstract
// the various network components.  Unlike a fully connected 
// network, a convolutional network is not a homogenious list of 
// objects.  There is convolution, pooling layers, and flattening
// layers, to name a few.  The iConvoLayer interface allows us
// to place the various objects into a single list.
class iConvoLayer {
public:
   struct Size {
      int cols;
      int rows;
      Size() : cols(0), rows(0) {}
      Size(int r, int c) : cols(c), rows(r) {}
      inline void Resize(int r, int c) { rows = r; cols = c; }
   };
   virtual ~iConvoLayer () = 0 {};
   virtual vector_of_matrix Eval(const vector_of_matrix& _x) = 0;
   virtual vector_of_matrix BackProp(vector_of_matrix& child_grad, bool want_backprop_grad = true) = 0;
   virtual void Update(double eta) = 0;
   virtual void Save(shared_ptr<iPutWeights> _pOut) = 0;
};

typedef iConvoLayer::Size clSize;
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
      w.setRandom();
      w *= Scale;
      if (BiasConstant) {
         w.rightCols(1).setConstant(Bias);
      }
   }
};

class IWeightsToNormDist : public iGetWeights {
   double StdDev;
   int Channels;
   /*
   Delving Deep into Rectifiers:
Surpassing Human-Level Performance on ImageNet Classification
   * */
   std::function<double(double, double, int)> stdv_Kaiming = [](double r, double c, int k) {
      return sqrt( 2.0 / (r * c * (double)k) );
   };
   std::function<double(double, double, int)> stdv_Xavier = [](double r, double c, int k) {
      return sqrt(6.0 / (r + c));
   };

   std::function<double(double, double, int)> fstdv;
public:
   enum InitType{
      Xavier,
      Kanning
   };
   IWeightsToNormDist(double std_dev) : StdDev(std_dev), Channels(-1) {
      // Not used in this mode but we should init fstdv to something.
      fstdv = stdv_Xavier;
   }
   IWeightsToNormDist(InitType init_type, int channels) : StdDev(-1.0), Channels(channels) {
      switch(init_type) {
         case Xavier:      fstdv = stdv_Xavier; break;
         case Kanning:     fstdv = stdv_Kaiming; break;
         default:          fstdv = stdv_Xavier;
      }
   }
   void ReadConvoWeight(Matrix& w, int k) {
      //double stdv = StdDev < 0.0 ? sqrt(2.0 / (double)(w.rows() * w.cols())) :  StdDev;
      double stdv = StdDev < 0.0 ? fstdv((double)w.rows(),(double)w.cols(),Channels) :  StdDev;
      std::default_random_engine generator;
      std::normal_distribution<double> distribution(0.0, stdv);
      for (int r = 0; r < w.rows(); r++) {
         for (int c = 0; c < w.cols(); c++) {
            w(r, c) = distribution(generator);
         }
      }
   }
   void ReadConvoBias(Matrix& w, int k) {
      w.setConstant(0.0);
   }
   void ReadFC(Matrix& w){
      double stdv = StdDev < 0.0 ? sqrt(2.0 / (double)w.cols()) :  StdDev;
      std::default_random_engine generator;
      std::normal_distribution<double> distribution(0.0, stdv);
      for (int r = 0; r < w.rows(); r++) {
         for (int c = 0; c < w.cols(); c++) {
            w(r, c) = distribution(generator);
         }
      }
      // The bias is always set to zero.
      w.rightCols(1).setConstant(0.0);
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

class actLinear : public iActive {
   int Size;
   ColVector J;
public:
   actLinear(int size) : Size(size), J(size) {
      J.setOnes();
   }
   actLinear() : Size(0) {}
   void Resize(int size) {
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
class Layer : public iStash {
public:
   int Count;
   int InputSize;
   int OutputSize;
   ColVector X;
   ColVector Z;
   Matrix W;
   Matrix stash_W;
   Matrix dW;
   unique_ptr<iActive> pActive;
   enum CallBackID { EvalPreActivation, EvalPostActivation, Backprop };
private:
   shared_ptr<iCallBackSink> EvalPreActivationCallBack;
   shared_ptr<iCallBackSink> EvalPostActivationCallBack;
   shared_ptr<iCallBackSink> BackpropCallBack;
public:
   Layer(int input_size, int output_size, unique_ptr<iActive> _pActive, shared_ptr<iGetWeights> _pInit ) :
      // Add an extra row to align with the bias weight.
      // This row should always be set to 1.
      X(input_size+1), 
      // Add an extra column for the bias weight.
      W(output_size, input_size+1),
      dW(input_size+1, output_size ),
      pActive(move(_pActive)),
      InputSize(input_size),
      OutputSize(output_size),
      Z(output_size)
   {
      pActive->Resize(output_size); 

      _pInit->ReadFC(W);

      dW.setZero();
      Count = 0;
   }

   ~Layer() {}

   void StashWeights() {
      stash_W.resize(W.rows(), W.cols());
      stash_W = W;
   }

   void ApplyStash() {
      W = stash_W;
      dW.setZero(); // Setting Count to zero effectivly zeros dW.  But needed if using momentum.
      Count = 0;
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
         props.insert({ "dW", CBObj(dW) });
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
         props.insert({ "dW", CBObj(dW) });
         props.insert({ "Z", CBObj(out) });
         EvalPostActivationCallBack->Properties( props );
         return out;
      }
      return pActive->Eval( Z );
   }

   RowVector BackProp(const RowVector& child_grad, bool want_layer_grad = true ) {
      Count++;
      RowVector delta_grad = child_grad * pActive->Jacobian(Z);
      Matrix iter_w_grad = X * delta_grad;

#ifdef MOMENTUM
      // REVIEW: Momentum 
      // iter_W_grad --> x (input)
      // dW --> y(n-1)
      double a = MOMENTUM;  // Valid range 0 - 1
      double b = 1.0 - a;
      // NOTE: The b and a are swapped from the equation above.
      //       This can be confusing.
      //       This is implementation of simple IIR low pass filter.
      //dW = b * iter_w_grad + a * dW;
      // Faster implementation (I think).
      dW += b * (iter_w_grad - dW);
#else
      double a = 1.0 / (double)Count;
      double b = 1.0 - a;
      dW = a * iter_w_grad + b * dW;
#endif
      if (BackpropCallBack != nullptr) {
         map<string, CBObj> props;
         int id = Backprop;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "X", CBObj(X) });
         props.insert({ "W", CBObj(W) });
         props.insert({ "dG", CBObj(delta_grad) });
         props.insert({ "dW", CBObj(dW) });
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
      Count = 0;
      W = W - eta * dW.transpose();
      //dW.setZero();  // Not strictly needed.
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

// Convolution Layer classes ------------------------------------

class FilterLayer2D : 
   public iConvoLayer, 
   public iStash {
public:
   int Count;
   Size InputSize;
   Size OutputSize;
   Size KernelSize;
   int KernelPerChannel;
   int Channels;
   // Vector of input matrix.  One per channel.
   vector_of_matrix X;
   // Vector of kernel matrix.  There are KernelPerChannel * input channels.
   vector_of_matrix W;
   // Vector of kernel matrix stash.
   vector_of_matrix stash_W;
   // Vector of kernel matrix gradients.  There are KernelPerChannel * input channels.
   vector_of_matrix dW;
   // The bias vector.  One bias for each kernel.
   vector_of_number B;
   // The bias vector stash.
   vector_of_number stash_B;
   // The bias gradient vector.
   vector_of_number dB;
   // Vector of activation complementary matrix.  There are KernelPerChannel * input channels.
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

   FilterLayer2D(Size input_size, int input_channels, Size output_size, Size kernel_size, int kernel_number, unique_ptr<iActive> _pActive, shared_ptr<iGetWeights> _pInit, bool no_bias = false ) :
      X(input_channels), 
      W(input_channels*kernel_number),
      stash_W(input_channels*kernel_number),
      B(input_channels*kernel_number),
      stash_B(input_channels*kernel_number),
      Z(input_channels*kernel_number),
      dW(input_channels*kernel_number),
      dB(input_channels*kernel_number),
      pActive(move(_pActive)),
      InputSize(input_size),
      OutputSize(output_size),
      KernelSize(kernel_size),
      KernelPerChannel(kernel_number),
      Channels(input_channels),
      NoBias(no_bias)
   {
      pActive->Resize(output_size.rows * output_size.cols); 

      for (Matrix& m : X) { 
         m.resize(input_size.rows, input_size.cols ); 
         m.setZero();
      }
      for (int i = 0; i < W.size();i++) {
         W[i].resize(kernel_size.rows,kernel_size.cols); 
         _pInit->ReadConvoWeight(W[i],i);
      }

      if (NoBias) {
         for (double& b : B) {
            b = 0.0;
         }
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
      //Setting Count to zero effectivly zeros dW.
      Count = 0;

      // Necessary when using momentum.
 #ifdef MOMENTUM
      for (Matrix& m : dW) {
         m.setZero();
      }
#endif
   }

   double MultiplyReverseBlock( Matrix& m, int mr, int mc, Matrix& h, int hr, int hc, int size_r, int size_c )
   {
      double sum = 0.0;
      for (int r = 0; r < size_r; r++) {
         for (int c = 0; c < size_c; c++) {
            sum += m(mr-r, mc-c) * h(hr+r, hc+c);
         }
      }
      return sum;
   }

   void LinearConvolutionAccumulate( Matrix& m, Matrix& h, Matrix& out )
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
               //out(r, c) = 0.0;
               ;
            }
            else {
               //cout << m1r << "," << m1c << "," << h1r << "," << h1c << "," << shr << "," << shc << "," << endl;
               out(r, c) += MultiplyReverseBlock(m, m2r, m2c, h, h1r, h1c, shr, shc);
            }
         }
      }
   }

   double MultiplyBlock( Matrix& m, int mr, int mc, Matrix& h, int hr, int hc, int size_r, int size_c )
   {
      double sum = 0.0;
      for (int r = 0; r < size_r; r++) {
         for (int c = 0; c < size_c; c++) {
            sum += m(mr+r, mc+c) * h(hr+r, hc+c);
         }
      }
      return sum;
   }

   double MultiplyBlockWithEigen( Matrix& m, int mr, int mc, Matrix& h, int hr, int hc, int size_r, int size_c )
   {
      double sum = (m.array().block(mr, mc, size_r, size_c) * h.array().block(hr, hc, size_r, size_c)).sum();

      return sum;
   }

   void LinearCorrelate( Matrix& m, Matrix& h, Matrix& out, double bias = 0.0 )
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

   inline void vecLinearCorrelate()
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

   vector_of_matrix Eval(const vector_of_matrix& _x) 
   {
      vector_of_matrix::const_iterator is = _x.begin();
      vector_of_matrix::iterator it = X.begin();
      for ( ; it != X.end(); ++it, ++is) {
         // Copy from matrix of source (_x) to matrix of target (X).
         *it = *is;
      }

      vecLinearCorrelate();
      vector_of_matrix vecOut(Z.size());

      if (EvalPreActivationCallBack != nullptr) {
         map<string, CBObj> props;   
         int id = EvalPreActivation;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "X", CBObj(X) });
         props.insert({ "W", CBObj(W) });
         props.insert({ "dW", CBObj(dW) });
         props.insert({ "Z", CBObj(Z) });
         EvalPreActivationCallBack->Properties( props );
      }

      for (Matrix& mm : vecOut) { mm.resize(OutputSize.rows, OutputSize.cols); }
      vector_of_matrix::iterator iz = Z.begin();
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
         props.insert({ "dW", CBObj(dW) });
         props.insert({ "Z", CBObj(vecOut) });
         EvalPostActivationCallBack->Properties( props );
      }

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
      // then avg sum into dW vec mat.
      // propagate upstream 
      // layer_grad corr=> rotated kernel to size of InputSize
      // sum results according to kernels per channel.
      const int incoming_channels =  KernelPerChannel * Channels;
      assert(child_grad.size() == incoming_channels);

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

      // Allocate the vector of matrix for the return but only allocate
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

         if (BackpropCallBack != nullptr) {
            map<string, CBObj> props;
            int id = Backprop1;
            props.insert({ "ID", CBObj(id) });
            props.insert({ "OC", CBObj(i) });  // Output Channel
            props.insert({ "K", CBObj(chn) }); // Kernel
            props.insert({ "dG", CBObj(m_delta_grad) });
            props.insert({ "idW", CBObj(iter_dW) });
            BackpropCallBack->Properties( props );
         }

#ifdef MOMENTUM
         // REVIEW: Momentum 
         // iter_W_grad --> x (input)
         // dW --> y(n-1)
         double a = MOMENTUM;  // Valid range 0 - 1
         double b = 1.0 - a;
         // NOTE: The b and a are swapped from the equation above.
         //       This can be confusing.
         //       This is implementation of simple IIR low pass filter.
         //dW = b * iter_w_grad + a * dW;
         // Faster implementation (I think).
         dW[i] += b * (iter_dW - dW[i]);
         if (!NoBias) { dB[i] += b * (iter_dB - dB[i]); }
#else
         // Average the result (iter_dW) into the Kernel gradient (dW).
         double a = 1.0 / (double)Count;
         double b = 1.0 - a;
         dW[i] = a * iter_dW + b * dW[i];
         if (!NoBias) { dB[i] = a * iter_dB + b * dB[i]; }
#endif

         if (want_backprop_grad) {
            LinearConvolutionAccumulate(m_delta_grad, W[i], vm_backprop_grad[chn]);
         }

         if (  !( (i+1) % KernelPerChannel ) ){  chn++; }
         i++;
      }

      if (BackpropCallBack != nullptr) {
         map<string, CBObj> props;
         int id = Backprop2;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "X", CBObj(X) });
         props.insert({ "W", CBObj(W) });
         props.insert({ "dW", CBObj(dW) });
         props.insert({ "Z", CBObj(Z) });
         BackpropCallBack->Properties( props );
      }

      return vm_backprop_grad;
   }

   void Update(double eta) {
      Count = 0;
      int maps = Channels * KernelPerChannel;
      for (int i = 0; i < maps;i++) {
         // REVIEW:  WARNING Will Robenson.. The "if" is for a special experiment!!!!!
         //if (i > 3) {
            W[i] = W[i] - eta * dW[i];
         //}

         if (!NoBias) {
            B[i] = B[i] - eta * dB[i];
            dB[i] = 0.0;
         }
      }
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
   int Size;
   ColVector X;
   ColVector Y;
   RowVector G;

   LossCrossEntropy() 
   {
      Size = 0;
   }

   LossCrossEntropy(int input_size, int output_size) :
      X(input_size),
      Y(input_size),
      G(input_size),
      Size(input_size)
   {
   }

   void Init(int input_size, int output_size) {
      X.resize(input_size);
      Y.resize(input_size),
      G.resize(input_size),
      Size = input_size;
   }

   Number Eval(const ColVector& x, const ColVector& y) {
      assert(Size > 0);
      X = x;
      Y = y;
      double loss = 0.0;
      for (int i = 0; i < Size; i++) {
         // No reason to evaulate this expression if y[i]==0.0 .
         if (y[i] != 0.0) {
            //                        Prevent undefined results when taking the log of 0
            loss -= y[i] * std::log( std::max(x[i], std::numeric_limits<Number>::epsilon()));
         }
      }
      return loss;
   }

   RowVector LossGradient(void) {
      //RowVector g(Size);
      for (int i = 0; i < Size; i++) {
         if (X[i] == 0.0 ) {
            if (Y[i] == 0.0) {
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
      ErrorOutput(string path, string name) : owf(path + "\\" + name + ".error.csv", ios::trunc)
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