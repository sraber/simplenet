#pragma once
#include <Layer.h>

class actCube : public iActive {
   int Size;
   ColVector J;

public:
   actCube(int size) : Size(size), J(size) {}
   actCube() : Size(1), J(1) {}
   void Resize(int size) {
      Size = size;
      J.resize(size);
   }
   // Z = W x + b.  Z is what is passed into the activation function.
   ColVector Eval(ColVector& q) {
      // Vector q carries the z vector on the way in, then is trasformed
      // to the values of the special function for the way out.  It will later
      // be passed into the Jacobian method and used to compute the Jacobian.
      // In this case it should simple retain the value of z.

      J = q.array().pow(3);

      return J;
   }
   ColVector Eval(Eigen::Map<ColVector>& q) {
      J = q.array().pow(3);
      return J;
   }
   Matrix Jacobian(const ColVector& q) {
      J = 3.0 * q.array().square();

      return J.asDiagonal();
   }

   int Length() { return Size; }
};

class actTanhEx : public iActive {
   int Size;
   ColVector J;
   double Tanh(double s) {
      double tp = exp(s);
      double tn = exp(-s);
      //tanh = (exp(s) - exp(-s)) / (exp(s) + exp(-s))
      return (tp - tn) / (tp + tn);
   }
public:
   actTanhEx(int size) : Size(size), J(size) {}
   actTanhEx() : Size(0) {}
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
      return 16.0 * q;
   }
   ColVector Eval(Eigen::Map<ColVector>& q) {
      for (int i = 0; i < Size; i++) {
         q(i) = Tanh(q(i));
      }
      return 16.0 * q;
   }
   Matrix Jacobian(const ColVector& q) {
      for (int i = 0; i < Size; i++) {
         J(i) = 1.0 - q(i) * q(i);
      }

      J *= 16.0;

      return J.asDiagonal();
   }

   int Length() { return Size; }
};

class actLinearEx : public iActive {
   int Size;
   ColVector J;
   ColVector J2;
   bool Clipped;
public:
   actLinearEx(int size) : Size(size), J(size), J2(size), Clipped(false) {
      J.setOnes();
      J2.setConstant(2.0);
   }
   actLinearEx() : Size(0), Clipped(false) {}
   void Resize(int size) {
      Size = size;
      J.resize(size);
      J.setOnes();
      J2.resize(size);
      J2.setConstant(2.0);
   }
   ColVector Eval(ColVector& x) {
      Clipped = false;
      for (int i = 0; i < Size; i++) {
         if (x[i] > 6.0) {
            x[i] = 6.0;
            Clipped = true;
         }
         else if (x[i] < -6.0) {
            x[i] = -6.0;
            Clipped = true;
         }
      }
      return x;
   }
   ColVector Eval(Eigen::Map<ColVector>& x) {
      Clipped = false;
      for (int i = 0; i < Size; i++) {
         if (x[i] > 8.0) {
            x[i] = 8.0;
            Clipped = true;
         }
         else if (x[i] < -8.0) {
            x[i] = -8.0;
            Clipped = true;
         }
      }
      return x;
   }
   Matrix Jacobian(const ColVector& x) {
      // REVIEW: This clipping idea would need more work to apply to 
      //         x vector greater than 1 element.  It'll work here.
      if (Clipped) {
         return J2.asDiagonal();
      }
      return J.asDiagonal();
   }

   int Length() { return Size; }
};

// ------------ Loss Layer -------------------------------------
class LossL2X : public iLossLayer {
public:
   int Size;
   ColVector Z;
   double L2;

   LossL2X() : Size(0), L2(0.0) {}
   LossL2X(int input_size, int output_size) :
      Z(input_size),
      Size(input_size) {}

   void Init(int input_size, int output_size) {
      Z.resize(input_size);
      Size = input_size;
   }
   Number Eval(const ColVector& x, const ColVector& y) {
      Z = x - y;
      L2 = std::sqrt(Z.dot(Z));
      return L2;
   }

   // REVIEW: How is the Weight Decay term added to this formulation?
   //         Weight decay is sum of all layer weights squared.  A scalar.
   //         The derivitive seems to be a matrix which is incompatiable 
   //         with the RowVector.
   //         I got it.  If you want to regularize, just add lamda*W to delta W at
   //         each layer before update.
   //         http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/

   RowVector LossGradient(void) {
      return (Z/L2).transpose();
   }
};

class CyclicVectorTransformer
{
private:
   typedef std::vector< std::tuple<int, int, double>> SampleVector;

   int InputSize;
   int OutputSize;
   ColVector GridR;

   SampleVector SM;

   // Augmented Input Vector.
   ColVector U;

   shared_ptr<iCallBackSink> EvalPostActivationCallBack;
   shared_ptr<iCallBackSink> BackpropCallBack;

   enum CallBackID { EvalPreActivation, EvalPostActivation, Backprop };
public:

   CyclicVectorTransformer(const int input_size, const int output_size) :
      InputSize(input_size),
      OutputSize(output_size),
      U(input_size + 1)
   {
      GridR = ColVector::LinSpaced(OutputSize, 0.0, InputSize - 1);
      SM.resize(OutputSize);
   }

   void Eval(const ColVector& UU, ColVector& V, const double T)
   {
      if (T < -8.0 || T > 8.0) {
         cout << "T bonkers.  T:" << T << endl;
      }
      runtime_assert(UU.size() == InputSize);
      U.block(0, 0, InputSize, 1) = UU;
      // The augmentation.  This back of the array to the front
      // of the array so that when interpolation is done between
      // the (InputSize-1) element and the InputSize element it will
      // be interpolation between the last element and the first element.
      U(InputSize) = UU[0];  

      V.resize(OutputSize);


      GenerateGrid(T);
      // Sample the input image
      LinearSampler(U, V);

      if (EvalPostActivationCallBack != nullptr) {
         map<string, CBObj> props;
         int id = EvalPreActivation;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "U", CBObj(U) });
         props.insert({ "V", CBObj(V) });
         props.insert({ "T", CBObj(T) });
         EvalPostActivationCallBack->Properties(props);
      }
   }

   RowVector BackpropGrid(const RowVector& dV)
   {
      RowVector dLdR(OutputSize);
      ComputeGridGradient(dV, dLdR, SM);
      RowVector T(1);
      T(0) = dLdR.sum();

      if (BackpropCallBack != nullptr) {
         map<string, CBObj> props;
         int id = 1;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "dV", CBObj(dV) });
         props.insert({ "dLdR", CBObj(dLdR) });
         props.insert({ "dG", CBObj(T) });
         BackpropCallBack->Properties(props);
      }
      return T;
   }

   RowVector BackpropSampler(const RowVector& dV)
   {
      RowVector dU(InputSize);
      dU.setZero();

      for (int r = 0; r < OutputSize; r++) {
         const tuple<int, int, double>& t = SM[r];
         const int rl = std::get<0>(t);
         const int rh = std::get<1>(t);
         const double rr = std::get<2>(t);
         const double dr = rh - rr;

         ColVector corners(2);
         corners(0) = rl;
         corners(1) = rh < InputSize ? rh : 0;  // Wrap around.

         // Compute the weights for the linear interpolation
         ColVector weights(2);
         weights(0) = dr;
         weights(1) = (1.0 - dr);

         // Compute the gradient for the input patch
         for (int i = 0; i < 2; ++i) {
            dU(corners(i)) += weights(i) * dV(r);
         }
      }

      if (BackpropCallBack != nullptr) {
         map<string, CBObj> props;
         int id = 2;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "dV", CBObj(dV) });
         props.insert({ "dU", CBObj(dU) });
         BackpropCallBack->Properties(props);
      }

      return dU;
   }

   void SetEvalPostActivationCallBack(shared_ptr<iCallBackSink> icb) { EvalPostActivationCallBack = icb; }
   void SetBackpropCallBack(shared_ptr<iCallBackSink> icb) { BackpropCallBack = icb; }

private:
   // The Sample Matrix generator for Affine transform.
   void GenerateGrid(double T)
   {
      //REVIEW: Do some range checking.
      SampleVector& sm = SM;

      // Compute the grid coordinates
 // grid_r element values are in the range of 0 to (InputSize - 1) but there are 
 // OutputSize of them.
 // This feature alone could be used for upsampling or downsampling.

      for (int r = 0; r < OutputSize; ++r)
      {
         const double xs = static_cast<double>(GridR(r)) + T;
         const double input_size = static_cast<double>(InputSize);
         double rr = xs - input_size * floor(xs / input_size);

         int rl = static_cast<int>(floor(rr));
         int rh = static_cast<int>(ceil(rr));

         sm[r] = std::tuple<int, int, double>(rl, rh, rr);
      }
   }

   void LinearSampler(const ColVector& m, ColVector& out)
   {
      SampleVector& sm = SM;

      int rows = out.rows();
      int srows = m.rows();

      for (int r = 0; r < rows; r++) {
         const tuple<int, int, double>& t = sm[r];

         const int rl = std::get<0>(t);
         const int rh = std::get<1>(t);
         const double rr = std::get<2>(t);

         const double dr = static_cast<double>(rh) - rr;

         out(r) = m(rl) * dr + m(rh) * (1.0 - dr);
      }
   }

   void ComputeGridGradient(const RowVector& dLdV, RowVector& dLdR, SampleVector& sm)
   {
      // REVIEW: Do some bounds checking.

      for (int r = 0; r < OutputSize; r++) {
         // The range of the element values of grid_xand grid_y is the range of the 
         // input matrix but are not necessarily whole numbers as there can be a 
         // dissimilar output matrix range that is larger or smaller.

         const tuple<int, int, double>& t = sm[r];

         // These values are all in the range of the input matrix U.
         const int rl = std::get<0>(t);
         const int rh = std::get<1>(t);
         const double rr = std::get<2>(t);

         const double dr = static_cast<double>(rh) - rr;

         double dvdr = U(rh) - U(rl);

         dLdR(r) = dvdr * dLdV(r);
      }
   }
};