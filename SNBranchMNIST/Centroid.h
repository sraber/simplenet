#pragma once
#include <Layer.h>
#define _USE_MATH_DEFINES
#include <math.h>

class Centroid : public iConvoLayer {
   shared_ptr<iCallBackSink> EvalPostActivationCallBack;
   shared_ptr<iCallBackSink> BackpropCallBack;

   Size InputSize;
   double F;
   double G;
   double H;

   // Used for output.
   vector_of_matrix Z;
   vector_of_matrix D;
public:


   // NOTE: Input Matrix should always be one channel.
   //       Backprop expects two channels.
   Centroid(Size input_size) :
      InputSize(input_size),
      Z(1),
      D(1)
   {
      Z[0].resize(2, 1);
      D[0].resize(InputSize.rows, InputSize.cols);
      F = 0.0;
      G = 0.0;
      H = 0.0;
   }
   ~Centroid() {
   }

   vector_of_matrix Eval(const vector_of_matrix& x)
   {
      assert(x.size() == 1);
      const Matrix& m = x[0];
      assert(InputSize == m);

      F = 0.0;
      G = 0.0;
      H = 0.0;

      for (int r = 0; r < InputSize.rows; r++) {
         for(int c = 0; c < InputSize.cols; c++ ){
            F += static_cast<double>(r) * m(r, c);
            H += static_cast<double>(c) * m(r, c);
            G += m(r, c);
         }
      }

      if (G == 0.0) {
         cout << "Centroid input is blank" << endl;
         Z[0](0, 0) = InputSize.rows >> 1;
         Z[0](1, 0) = InputSize.cols >> 1;
      }
      else {
         Z[0](0, 0) = F / G;
         Z[0](1, 0) = H / G;
      }

      if (EvalPostActivationCallBack != nullptr) {
         map<string, CBObj> props;
         int id = 1;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "F", CBObj(F) });
         props.insert({ "G", CBObj(G) });
         props.insert({ "H", CBObj(H) });
         props.insert({ "X", CBObj(x) });
         props.insert({ "Z", CBObj(Z) });
         EvalPostActivationCallBack->Properties(props);
      }

      return Z;
   }
   vector_of_matrix BackProp(vector_of_matrix& child_grad, bool want_backprop_grad = true)
   {
      assert(child_grad.size() == 1);
      const Matrix& m = child_grad[0];
      assert(Size(1,2) == m);

      Matrix& o = D[0];

      double dLdCr = m(0, 0);
      double dLdCc = m(0, 1);

      double dCrdu = 0.0;
      double dCcdu = 0.0;

      double gg = G * G;

      for (int r = 0; r < InputSize.rows; r++) {
         for (int c = 0; c < InputSize.cols; c++) {
            dCrdu = (static_cast<double>(r) * G - F) / gg;
            dCcdu = (static_cast<double>(c) * G - H) / gg;

            o(r, c) = dLdCr * dCrdu + dLdCc * dCcdu;
         }
      }

      if (BackpropCallBack != nullptr) {
         map<string, CBObj> props;
         int id = 3;
         props.insert({ "ID", CBObj(id) });
         props.insert({ "D", CBObj(D) });
         props.insert({ "G", CBObj(child_grad) });
         BackpropCallBack->Properties(props);
      }

      return D;
   }
   void Update(double eta)
   {
   }
   void Save(shared_ptr<iPutWeights> _pOut)
   {
   }

   void SetEvalPostActivationCallBack(shared_ptr<iCallBackSink> icb) { EvalPostActivationCallBack = icb; }
   void SetBackpropCallBack(shared_ptr<iCallBackSink> icb) { BackpropCallBack = icb; }
};

//--------------------------------------------------------------
