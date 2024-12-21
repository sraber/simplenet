// SNBranchMNIST.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

//#define LOGPOLAR
#ifdef LOGPOLAR
const int INPUT_ROWS = 32;
//const int INPUT_COLS = 32;
const int LP_WASTE = 6;
//const int LP_WASTE = 12;
const int INPUT_COLS = 32 - LP_WASTE;
#else
const int LP_WASTE = 6;
//const int INPUT_ROWS = 28;
//const int INPUT_COLS = 28;
const int INPUT_ROWS = 32;
const int INPUT_COLS = 32 - LP_WASTE;
#endif

//#define RETRY 1
#define SGD
//#define USE_FILTER_FOR_POOL
// Use MOMENTUM 0.0 for grad comp testing.
#define MOMENTUM 0.6

//#define M_ANG (1.5*M_PI_4)
//#define M_ANG M_PI_4
// 20 degrees
#define M_ANG (0.1111111111 * M_PI)
const double branch_pass_threshold = 0.6;

#include <Eigen>
#include <iostream>
#include <iomanip>
#include <MNISTReader.h>
#include <Layer.h>
#include "FilterLayer.h"
#include <SpacialTransformer.h>
#include <CyclicVectorTransformer.h>
#include "Centroid.h"
#include <bmp.h>
#include <chrono>
#include <deque>
#include <functional>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define _USE_MATH_DEFINES
#include <math.h>
#include <fft2convo.h>
#include <fft1.h>

#include <map>

#include <utility.h>
#include <conio.h>

#include <LogPolar.h>

   // Define static optomizer variables.
double optoADAM::B1 = 0.0;
double optoADAM::B2 = 0.0;
double optoLowPass::Momentum = 0.0;

int gModelBranches = 0;
int gAutoSave = 0;
int gTrainingBranch = 4;
int gWarmupOn = 0;

string path = "C:\\projects\\neuralnet\\simplenet\\SNBranchMNIST\\weights";
string model_name = "layer";

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


// Fully Connected Layer ----------------------------------------
class SpectrumOutputLayer :
   public iLayer
{
public:
   int InputSize;
   int InputPaddedSize;
   int OutputSize;
   ColVector X;
   ColVector Z;
   enum CallBackID { EvalPreActivation };
private:
   shared_ptr<iCallBackSink> EvalPreActivationCallBack;
public:
   SpectrumOutputLayer(int input_size,int output_size) :
      InputSize(input_size)
   {
      InputPaddedSize = nearest_power_ceil(input_size);
      OutputSize = InputPaddedSize >> 1;
      runtime_assert( output_size <= OutputSize )
      OutputSize = output_size;
      X.resize(InputPaddedSize);
      Z.resize(OutputSize);
   }

   ~SpectrumOutputLayer() {}

   ColVector Eval(const ColVector& _x) {
      X.block(0, 0, InputSize, 1) = _x;
      if (InputPaddedSize > InputSize) {
         X.block(InputSize, 0, InputPaddedSize - InputSize, 1).setZero();
      }
      rfftsine(X.data(), InputPaddedSize, 1);

      // NOTE: 2X multiplier to make spectrum magnitudes nominally more equivilant
      //       to signal amplitudes.
      //       The NR book says that the result of rfft should be divided by N
      //       but N/2 seems to give a more reasonable result.
      X.array() /= (double)(InputPaddedSize>>1);

      // Real valued 1st and last complex pair.
      Z(0) = fabs(X(0));
      //Z(OutputSize-1) = fabs(X(1));  No room for Nyquist frequency in half spectrum.
      // Ignore Nyquist frequency.
      for (int c = 1; c < OutputSize; c++) {
         int p = c << 1;
         double r = X(p);
         double i = X(p + 1);
         Z(c) = sqrt(r * r + i * i);
      }
      return Z;
   }

   RowVector BackProp(const RowVector& child_grad, bool want_layer_grad = true) {
      //throw runtime_error("No backpropagation is allowed through the Spectrum Transform layer");
      return RowVector(0);
   }

   void Update(double eta) {}
   void Save(shared_ptr<iPutWeights> _pOut) {}
   void SetEvalPreActivationCallBack(shared_ptr<iCallBackSink> icb) { EvalPreActivationCallBack = icb; }
};
//---------------------------------------------------------------

//---------- Statistical Support --------------------------------
class StatRunningAverage
{
   unsigned int BSize = 1000;
   std::deque<double> buffer;
   const double den;
   double avg;
   ofstream owfAvg;
   ofstream owfSdv;
public:
   StatRunningAverage(string path, string model, string param, unsigned int window_size, bool truncate = true) :
      owfAvg(path + "\\" + model + "." + param + ".avg.csv", truncate ? ios::trunc : std::ios::app ),
      owfSdv(path + "\\" + model + "." + param + ".sdv.csv", truncate ? ios::trunc : std::ios::app),
      BSize(window_size),
      avg(0.0),
      buffer(0),
      den(1.0 / static_cast<double>(BSize))
   {
      for (int i = 0; i < BSize; i++) {
         buffer.push_back(0.0);
      }
   };
   void Push(double value) {
      // Running Average
      avg += (value * den);
      avg -= (buffer.front() * den);
      buffer.pop_front();
      buffer.push_back(value);
   }
   void write() {
      owfAvg << GetRunningAverage() << endl;
      owfSdv << GetRunningStdv() << endl;
   }
   double GetRunningAverage() {
      return avg;
   }
   double GetRunningStdv() {
      double sum = 0.0;
      for (const auto& e : buffer) {
         double v = e - avg;
         sum += (v * v);
      }
      return std::sqrt(sum * den);
   }
};

//---------------------------------------------------------------

//---------- DAG Support -----------------------

class NetContext
{
public:
   ColVector v;
   RowVector g;
   NetContext() {}
};

class CovNetContext
{
public:
   // m is used for forward pass and backprop pass.
   vector_of_matrix m;
   CovNetContext() : m(1) {}
   void Reset() {
      m.resize(1);
   }
};

class DAGContext
{
public:
   ColVector v;
   RowVector g;
   vector_of_matrix m;
   DAGContext() : m(1) {}
   void Reset() {
      m.resize(1);
      v.resize(0);
      g.resize(0);
   }
};

class ErrorContext
{
   const int BSize = 1000;
   unsigned int count;
   std::deque<double> buffer;

   const double den;
public:
   bool touched;
   unsigned int id;
   double avg;
   // REVIEW: Make this a pointer to a ColVector.
   //         Set up the gErrorContexts the same way but
   //         ColVecotr label is a global.
   //         This way other ErrorContexts can point at other
   //         labels.
   static ColVector label;
   double error;
   double average_error;
   bool correct;
   unsigned int total_correct;
   double class_max;
   ErrorContext(unsigned int _id = 0) :
      id(_id),
      error(0.0),
      average_error(0.0),
      correct(false),
      class_max(0.0),
      total_correct(0),
      count(0),
      buffer(0),
      avg(0.0),
      touched(false),
      den(1.0 / static_cast<double>(BSize))
   {
      for (int i = 0; i < BSize; i++) {
         buffer.push_back(0.0);
      }
   };
   void SetStatus(bool status, double _error) {
      count++;
      touched = true;
      error = _error;
      correct = status;
      if (status) {
         total_correct++;
      }

      double a = 1.0 / (double)(count);
      double d = 1.0 - a;
      average_error = a * error + d * average_error;

      // Running Average
      avg += (error * den);
      avg -= (buffer.front() * den);
      buffer.pop_front();
      buffer.push_back(error);

      //if (id == 2 && (error < -10 || error > 10)) {
      //   cout << "Error bonkers.  error:" << error << endl;
      //}
   }
   void Reset() {
      touched = false;
      error = 0.0;
      correct = false;
      average_error = 0.0;
      count = 0;
      total_correct = 0;
      class_max = 0.0;
   }
   inline double PctCorrect() {
      return count > 0 ? 100.0 * static_cast<double>(total_correct) / static_cast<double>(count) : 100.0;
   }
   double GetRunningAverage() {
      return avg;
   }
   double GetRunningStdv() {
      double sum = 0.0;
      for (const auto& e : buffer) {
         double v = e - avg;
         sum += (v * v);
      }
      return std::sqrt(sum * den);
   }

};
ColVector ErrorContext::label;

// Use only one of these per model.
class ExitContext
{
public:
   bool stop;
   ExitContext() : stop(false){}
};

class iDAGObj {
public:
   virtual ~iDAGObj() = 0 {};
   virtual void Eval() = 0;
   virtual void BackProp() = 0;
   virtual void Update(double eta) = 0;
   virtual void Save(shared_ptr<iPutWeights> _pOut) = 0;
   virtual void StashWeights() = 0;
   virtual void ApplyStash() = 0;
   virtual  int GetID() = 0;
};

class DAGConvoLayerObj : public iDAGObj
{
public:
   unsigned int ID;
   shared_ptr<iConvoLayer> pLayer;
   //shared_ptr<CovNetContext> pContext;
   DAGContext& Context;

   //DAGConvoLayerObj(unsigned int id, shared_ptr<iConvoLayer> _pLayer, shared_ptr<CovNetContext> _pContext) :
   DAGConvoLayerObj(unsigned int id, shared_ptr<iConvoLayer> _pLayer, DAGContext& _pContext) :
      ID(id),
      pLayer(std::move(_pLayer)),
      Context(_pContext)
   {}

   // iDAGObj implementation
   void Eval() {
      Context.m = pLayer->Eval(Context.m);
   }
   void BackProp() {
      Context.m = pLayer->BackProp(Context.m);
   }
   void Update(double eta) { pLayer->Update(eta); }
   void Save(shared_ptr<iPutWeights> _pOut) { pLayer->Save(_pOut); }
   void StashWeights() {
      auto ist = dynamic_pointer_cast<iStash>(pLayer);
      if (ist) { ist->StashWeights(); }
   }
   void ApplyStash() {
      auto ist = dynamic_pointer_cast<iStash>(pLayer);
      if (ist) { ist->ApplyStash(); }
   }
   int GetID() { return ID; }
   //--------------------------------------
};

class DAGFlattenObj : public iDAGObj
{
   shared_ptr<iConvoLayer> pLayer;
   DAGContext& Context;
   //shared_ptr<NetContext> pLayerContext; 
public:
   unsigned int ID;
   //DAGFlattenObj(unsigned int id, shared_ptr<iConvoLayer> _pLayer, shared_ptr<CovNetContext> _pContext1, shared_ptr<NetContext> _pContext2) :
   DAGFlattenObj(unsigned int id, shared_ptr<iConvoLayer> _pLayer, DAGContext& _pContext1 ) :
      ID(id),
      pLayer(std::move(_pLayer)),
      Context(_pContext1)
      //pLayerContext(_pContext2)
   {}
   void Eval() {
      Context.v = pLayer->Eval(Context.m)[0].col(0);
   }
   void BackProp() {
      vector_of_matrix t(1);
      t[0] = Context.g;
      // REVIEW: Note here that I am trying to use the VOM of the forward
      //         pass for the backward pass as well.
      Context.m = pLayer->BackProp(t);
   }
   void Update(double eta) { pLayer->Update(eta); }
   void Save(shared_ptr<iPutWeights> _pOut) { pLayer->Save(_pOut); }
   void StashWeights() {}
   void ApplyStash() {}
   int GetID() { return ID; }
};

class DAGLayerObj : public iDAGObj
{
   shared_ptr<iLayer> pLayer;
   DAGContext& Context;
public:
   unsigned int ID;
   DAGLayerObj(unsigned int id, shared_ptr<iLayer> _pLayer, DAGContext& _pContext) :
      ID(id),
      pLayer( std::move(_pLayer) ),
      Context( _pContext)
   {}
   void Eval() {
      Context.v = pLayer->Eval(Context.v);
   }
   void BackProp() {
      Context.g = pLayer->BackProp(Context.g);
   }
   void Update(double eta) { pLayer->Update(eta); }
   void Save(shared_ptr<iPutWeights> _pOut) { pLayer->Save(_pOut); }
   void StashWeights() {
      auto ist = dynamic_pointer_cast<iStash>(pLayer);
      if (ist) { ist->StashWeights(); }
   }
   void ApplyStash() {
      auto ist = dynamic_pointer_cast<iStash>(pLayer);
      if (ist) { ist->ApplyStash(); }
   }
   int GetID() { return ID; }
};

// REVIEW: Maybe make something called a stash object or context.  It holds a reference to the
//         input data.  Then make a CopyStash object.  This DAG object can be place right after
//         an exit test so the copy to context only happens if needed.

class DAGConvoContextCopyObj : public iDAGObj
{
   DAGContext& Context1;
   DAGContext& Context2;
public:
   unsigned int ID;
   DAGConvoContextCopyObj(unsigned int id, DAGContext& _pContext1, DAGContext& _pContext2) :
      ID(id),
      Context1(_pContext1),
      Context2(_pContext2)
   {}
   void Eval() {
      Context2.m = Context1.m;
   }
   void BackProp() {}
   void Update(double eta) { }
   void Save(shared_ptr<iPutWeights> _pOut) {}
   void StashWeights() {}
   void ApplyStash() {}
   int GetID() { return ID; }
};

class DAGContextCopyObj : public iDAGObj
{
   DAGContext& Context1;
   DAGContext& Context2;
public:
   unsigned int ID;
   DAGContextCopyObj(unsigned int id, DAGContext& _pContext1, DAGContext& _pContext2) :
      ID(id),
      Context1(_pContext1),
      Context2(_pContext2)
   {}
   void Eval() {
      Context2.v = Context1.v;
   }
   void BackProp() {}
   void Update(double eta) { }
   void Save(shared_ptr<iPutWeights> _pOut) {}
   void StashWeights() {}
   void ApplyStash() {}
   int GetID() { return ID; }
};

class DAGBranchObj : public iDAGObj
{
   friend class DAGExitTest;
   DAGContext& Context1;
   DAGContext& Context2;
   bool bBackprop;
   double BranchWeight;
public:
   unsigned int ID;
   DAGBranchObj(unsigned int id, DAGContext& _pContext1, DAGContext& _pContext2, bool backprop_branch = true, double branch_weight = 0.3 ) :
      ID(id),
      Context1(_pContext1),
      Context2(_pContext2),
      bBackprop(backprop_branch),
      BranchWeight(branch_weight)
   {}
   void Eval() {
      Context2.v = Context1.v;
   }
   void BackProp() {
      // REVIEW: bBackprop is not needed.  Just use BranchWeight as flag.
      if (bBackprop && BranchWeight > 0.0) {
         Context1.g += (BranchWeight * Context2.g);
      }
   }
   void Update(double eta) { }
   void Save(shared_ptr<iPutWeights> _pOut) {}
   void StashWeights() {}
   void ApplyStash() {}
   int GetID() { return ID; }
};

class DAGJoinObj : public iDAGObj
{
   DAGContext& Context1;
   DAGContext& Context2;
   int s1 = 0;
   int s2 = 0;
public:
   unsigned int ID;
   // Note: The contexts are assigned at creation time.  They can't be accessed until
   //       Eval is called.
   DAGJoinObj(unsigned int id, DAGContext& _pContext1, DAGContext& _pContext2) :
      ID(id),
      Context1(_pContext1),
      Context2(_pContext2)
   {}
   void Eval() {
      s1 = static_cast<int>(Context1.v.size());
      s2 = static_cast<int>(Context2.v.size());
      ColVector t = Context2.v;
      Context2.v.resize(s1 + s2);
      Context2.v.block(0, 0, s1, 1) = Context1.v;
      Context2.v.block(s1, 0, s2, 1) = t;
   }
   void BackProp() {
      RowVector t = Context2.g.block(0, s1, 1, s2);
      Context1.g = Context2.g.block(0, 0, 1, s1);
      Context2.g.resize(s2);
      Context2.g = t;
   }
   void Update(double eta) { }
   void Save(shared_ptr<iPutWeights> _pOut) {}
   void StashWeights() {}
   void ApplyStash() {}
   int GetID() { return ID; }
};

class DAGAddObj : public iDAGObj
{
   DAGContext& Context1;
   DAGContext& Context2;
public:
   unsigned int ID;
   // Note: The contexts are assigned at creation time.  They can't be accessed until
   //       Eval is called.
   DAGAddObj(unsigned int id, DAGContext& _pContext1, DAGContext& _pContext2) :
      ID(id),
      Context1(_pContext1),
      Context2(_pContext2)
   {}
   void Eval() {
      assert( static_cast<int>(Context1.v.size()) == static_cast<int>(Context2.v.size()) );
      Context2.v += Context1.v;
   }
   void BackProp() {
      assert(static_cast<int>(Context1.g.size()) == static_cast<int>(Context2.g.size()));
      Context2.g /= 2.0;
      Context1.g = Context2.g;
   }
   void Update(double eta) { }
   void Save(shared_ptr<iPutWeights> _pOut) {}
   void StashWeights() {}
   void ApplyStash() {}
   int GetID() { return ID; }
};

class DAGCompAvg : public iDAGObj
{
   DAGContext& Context1;
   DAGContext& ContextAvg;
   static unsigned int N;
public:
   unsigned int ID;
   // Note: The contexts are assigned at creation time.  They can't be accessed until
   //       Eval is called.
   DAGCompAvg(unsigned int id, DAGContext& _pContext1, DAGContext& _pContextAvg) :
      ID(id),
      Context1(_pContext1),
      ContextAvg(_pContextAvg)
   {}
   void Eval() {
      N++;
      int s = static_cast<int>(Context1.v.size());
      runtime_assert( s == static_cast<int>(ContextAvg.v.size()));
      double a = 1.0 / (double)(N);
      double b = 1.0 - a;
      //for (int i = 0; i < s; i++) {
      //   pContextAvg->v[i] = a * pContext1->v[i] + b * pContextAvg->v[i];
      //}
         
      ContextAvg.v = a * Context1.v + b * ContextAvg.v;
   }
   void BackProp() {
      N = 0;
      Context1.g = ContextAvg.g;
   }
   void Update(double eta) { }
   void Save(shared_ptr<iPutWeights> _pOut) {}
   void StashWeights() {}
   void ApplyStash() {}
   int GetID() { return ID; }
};
unsigned int DAGCompAvg::N;

class DAGAvgFilterGrad : public iDAGObj
{
   DAGContext& Context1;
   DAGContext& ContextAvg;
   shared_ptr<unsigned int> pN;
public:
   unsigned int ID;
   DAGAvgFilterGrad(unsigned int id, DAGContext& _pContext1, DAGContext& _pContextAvg, shared_ptr<unsigned int> _pCounter) :
      ID(id),
      Context1(_pContext1),
      ContextAvg(_pContextAvg),
      pN(_pCounter)
   {}
   void Eval() {
      (*pN) = 0;
   }
   void BackProp() {
      (*pN)++;
      assert(Context1.m.size()==ContextAvg.m.size());
      assert(Context1.m[0].rows() == ContextAvg.m[0].rows()
          && Context1.m[0].cols() == ContextAvg.m[0].cols());

      vector_of_matrix::iterator i1 = Context1.m.begin();
      vector_of_matrix::iterator iavg = ContextAvg.m.begin();
      int rows = i1->rows();
      int cols = i1->cols();

      if (*pN == 1) {
         for (; i1 < Context1.m.end(); ++i1, ++iavg) {
            (*iavg) = (*i1);
         }
      }
      else {
         double a = 1.0 / (double)((*pN));
         double b = 1.0 - a;

         for (; i1 < Context1.m.end(); ++i1, ++iavg) {
            // REVIEW: Which is faster, line below 
            //(*iavg) = a * (*i1) + b * (*iavg);
            (*iavg) *= b;
            (*iavg) += a * (*i1);
            //   or the loops written out?
            //for (int r = 0; r < rows; ++r) {
            //   for (int c = 0; c < cols; ++c) {
            //      iavg->operator()(r, c) = a * i1->operator()(r, c) + b * iavg->operator()(r, c);
            //   }
            //}
         }
      }

      //cout << (*pN) << endl
      //   << ContextAvg.m[0] << endl << endl
      //   << Context1.m[0] << endl;
   }
   void Update(double eta) { }
   void Save(shared_ptr<iPutWeights> _pOut) {}
   void StashWeights() {}
   void ApplyStash() {}
   int GetID() { return ID; }
};

class DAGLambdaLayer : public iDAGObj
{
   std::function<void()> EvalLambda;
   std::function<void()> BackPropLambda;
public:
   unsigned int ID;
   DAGLambdaLayer(unsigned int id, std::function<void()> eval_lambda, std::function<void()> bd_lambda ) :
      ID(id),
      EvalLambda(eval_lambda),
      BackPropLambda(bd_lambda)
   {}

   void Eval() {
      EvalLambda();
   }
   void BackProp() {
      BackPropLambda();
   }
   void Update(double eta) { }
   void Save(shared_ptr<iPutWeights> _pOut) {}
   void StashWeights() {}
   void ApplyStash() {}
   int GetID() { return ID; }
};

typedef std::function<std::pair<double, bool>(const ColVector& x, const ColVector& label)> DAGErrorLayerEvalLambda;

DAGErrorLayerEvalLambda DAGErrorLayerClassifierLambda = [](const ColVector& x, const ColVector& y) {
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

   return std::pair<double, bool>(xmax, (x_max_index == y_max_index));
};


class DAGErrorLayer : public iDAGObj
{
   shared_ptr<iLossLayer> pLossLayer;
   ErrorContext& errContext;
   DAGContext& Context;
   DAGErrorLayerEvalLambda EvalLambda;

   void StatEval(const ColVector& x, const ColVector& y){
      assert(x.size() == y.size());
      /*

      */


      //errContext.SetStatus(  (x_max_index == y_max_index),
      //                           pLossLayer->Eval(Context.v, errContext.label) );
      //errContext.class_max = xmax;
   }
public:
   unsigned int ID;
   DAGErrorLayer(unsigned int id, shared_ptr<iLossLayer> _pLayer, DAGContext& _pContext, ErrorContext& _pEContext, DAGErrorLayerEvalLambda _lambda = DAGErrorLayerClassifierLambda) :
      ID(id),
      pLossLayer(std::move(_pLayer)),
      errContext(_pEContext),
      EvalLambda(_lambda),
      Context( _pContext )
   {}
   void Eval() {
      //StatEval(Context.v, errContext.label);
      assert(Context.v.size() == errContext.label.size());
      std::pair<double,bool> res = EvalLambda(Context.v, errContext.label);
      errContext.SetStatus(res.second, pLossLayer->Eval(Context.v, errContext.label) );
      errContext.class_max = res.first;
   }
   void BackProp() {
      Context.g = pLossLayer->LossGradient();
   }
   void Update(double eta) { }
   void Save(shared_ptr<iPutWeights> _pOut) {}
   void StashWeights() {}
   void ApplyStash() {}
   int GetID() { return ID; }
};

// REVIEW: This class only works with classifiers.
class DAGExitTest : public iDAGObj
{
   ErrorContext& Error;
   shared_ptr<ExitContext> pExit;
   shared_ptr<DAGBranchObj> pBranch;
   unsigned long WarmUpCount;
   unsigned long Count;
   double BackpropLimitUpper;
   double BackpropLimitLower;
   double EvalBranchThreshold;
public:
   unsigned int ID;
   DAGExitTest(unsigned int id, shared_ptr<ExitContext> _pContext, shared_ptr<DAGBranchObj> _pBranch, ErrorContext& _pError,
               unsigned long warm_up_count = 0.0, double backprop_limit_lower = -1.0, double backprop_limit_upper = -1.0, double eval_branch_threshold = 0.9) :
      ID(id),
      Count(0),
      WarmUpCount(warm_up_count),
      BackpropLimitLower(backprop_limit_lower),
      BackpropLimitUpper(backprop_limit_upper),
      EvalBranchThreshold(eval_branch_threshold),
      pExit(_pContext),
      pBranch(_pBranch),
      Error(_pError)
   {
      runtime_assert(EvalBranchThreshold >= 0 && EvalBranchThreshold <= 1.0);
   }
   void Eval() {
      Count++;
      if( WarmUpCount > 0 && Count < WarmUpCount ){
         pBranch->bBackprop = false;
         pExit->stop = true;
      }
      else if( Error.correct && Error.class_max >= EvalBranchThreshold){
         pBranch->bBackprop = false;
         pExit->stop = true;
         }
      else {
         pBranch->bBackprop = true;
         pExit->stop = false;
      }
   }
   void BackProp() {
      // REVIEW: This seemed to work.  This should be it's own count threshold, like BackpropLimitCount.
      if (Count > 6 * WarmUpCount) {
         if (BackpropLimitUpper > 0.0 && Error.class_max >= BackpropLimitUpper) {
            //cout << "Limit backprop on high end. ID" << ID << " error:" << pError->class_max << endl;
            pBranch->bBackprop = false;
            pExit->stop = true;
         }
         else if (BackpropLimitLower > 0.0 && Error.class_max < BackpropLimitLower) {
            //cout << "Limit backprop on low end. ID" << ID << " error:" << pError->class_max << endl;
            pBranch->bBackprop = false;
            pExit->stop = true;
         }
      }
   }
   void Update(double eta) { }
   void Save(shared_ptr<iPutWeights> _pOut) {}
   void StashWeights() {}
   void ApplyStash() {}
   int GetID() { return ID; }
};

class DAGSpacialTransformGridLayer : public iDAGObj
{
   shared_ptr<SpacialTransformer> pTransformer;
   shared_ptr<CovNetContext> pCovNetContext;
   shared_ptr<NetContext> pNetContext;
public:
   unsigned int ID;
   DAGSpacialTransformGridLayer(unsigned int id, shared_ptr<SpacialTransformer> _pTransformer, shared_ptr<CovNetContext> _pContext1, shared_ptr<NetContext> _pContext2) :
      ID(id),
      pCovNetContext(_pContext1),
      pNetContext(_pContext2),
      pTransformer(_pTransformer)
   {}
   void Eval() {}
   void BackProp() {
      // REVIEW: How is multiple channels handled??
      int rows = pCovNetContext->m[0].rows();
      int cols = pCovNetContext->m[0].cols();
      Matrix s(rows, cols);
      s.setZero();
      for (Matrix& mm : pCovNetContext->m) {
         s += mm;
      }

      //for (int r = 0; r < rows; ++r) {
      //   for (int c = 0; c < cols; ++c) {
      //      if (std::isnan(s(r, c))) {
      //         cout << "s has nan" << endl;
      //      }
      //   }
      //}

      // NOTE: Origioinal paper's author suggested they decreaseed the size of the 
      //       grid derivitive by some amount due to entire image gradient suming 
      //       into grad of trasform coordinates.
      pNetContext->g = 0.5 * pTransformer->BackpropGrid(s);
      // 
      //cout << pNetContext->g << endl;

      // Use the following lines to turn off grid transformation.
      // They set the grid gradient to zero so the localization network
      // won't change the nominal transform.  Note that the trasform will
      // still be perturbed about the Identity transform.
      //pNetContext->g.resize(6);
      //pNetContext->g.setZero();
   }
   void Update(double eta) { }
   void Save(shared_ptr<iPutWeights> _pOut) {}
   void StashWeights() {}
   void ApplyStash() {}
   int GetID() { return ID; }
};

class DAGSpacialTransformSampleLayer : public iDAGObj
{
   shared_ptr<SpacialTransformer> pTransformer;
   shared_ptr<CovNetContext> pCovNetContext;
   shared_ptr<NetContext> pNetContext;
   bool WantBackprop;
public:
   unsigned int ID;
   DAGSpacialTransformSampleLayer(unsigned int id, shared_ptr<SpacialTransformer> _pTransformer, shared_ptr<CovNetContext> _pContext1, shared_ptr<NetContext> _pContext2, bool want_backprop = true) :
      ID(id),
      pCovNetContext(_pContext1),
      pNetContext(_pContext2),
      pTransformer(_pTransformer),
      WantBackprop(want_backprop)
   {}
   void Eval(){
      //OWeightsCSVFile fcsv(path, "xform");
      for (Matrix& mm : pCovNetContext->m) {
         //cout << "v: " << pNetContext->v.transpose() << endl;

         //fcsv.Write(mm, 0);
         pTransformer->Eval(mm, mm, pNetContext->v);
         //fcsv.Write(mm, 1);
      }
   }
   void BackProp() {
      if (WantBackprop) {
         for (Matrix& mm : pCovNetContext->m) {
            mm = pTransformer->BackpropSampler(mm);
         }
      }
   }
   void Update(double eta) { }
   void Save(shared_ptr<iPutWeights> _pOut) {}
   void StashWeights() {}
   void ApplyStash() {}
   int GetID() { return ID; }
};

class DAGCyclicTransformLayer : public iDAGObj
{
   shared_ptr<CyclicVectorTransformer> pTransformer;
   DAGContext& InOutNC;
   DAGContext& LocNC;
public:
   unsigned int ID;
   DAGCyclicTransformLayer(unsigned int id, shared_ptr<CyclicVectorTransformer> _pTransformer, DAGContext& _pContext1, DAGContext& _pContext2 ) :
      ID(id),
      InOutNC(_pContext1),
      LocNC(_pContext2),
      pTransformer(_pTransformer)
   {}
   void Eval() {
      pTransformer->Eval(InOutNC.v, InOutNC.v, LocNC.v[0]);
   }
   void BackProp() {
      LocNC.g = pTransformer->BackpropGrid(InOutNC.g);
      InOutNC.g = pTransformer->BackpropSampler(InOutNC.g);
   }
   void Update(double eta) { }
   void Save(shared_ptr<iPutWeights> _pOut) {}
   void StashWeights() {}
   void ApplyStash() {}
   int GetID() { return ID; }
};

class DAGSpacialTransformLayer : public iDAGObj
{
   shared_ptr<iSampler> pSampler;
   shared_ptr<iGrid> pGrid;
   DAGContext& InOutNC;
   DAGContext& LocNC;
   SampleMatrix SM;
   bool WantBackprop;
public:
   unsigned int ID;
   DAGSpacialTransformLayer(unsigned int id, shared_ptr<iSampler> _pSampler, shared_ptr<iGrid> _pGrid, DAGContext& _pContext1, DAGContext& _pContext2,
                            bool want_backprop = true) :
      ID(id),
      InOutNC(_pContext1),
      LocNC(_pContext2),
      pSampler(_pSampler),
      pGrid(_pGrid),
      WantBackprop(want_backprop)
   {}
   void Eval() {
      pGrid->Eval(SM, LocNC.v );
      pSampler->Eval(InOutNC.m, InOutNC.m, SM);
   }
   void BackProp() {
      Matrix dLdR;
      Matrix dLdC;

      pSampler->ComputeGridGradients(dLdR, dLdC, InOutNC.m, SM);
      LocNC.g = pGrid->Backprop(dLdR,dLdC, SM);

      if (WantBackprop) {
         InOutNC.m = pSampler->Backprop(InOutNC.m, SM);
      }
   }
   void Update(double eta) { }
   void Save(shared_ptr<iPutWeights> _pOut) {}
   void StashWeights() {}
   void ApplyStash() {}
   int GetID() { return ID; }
};

typedef vector< shared_ptr<iDAGObj> > layer_list;
layer_list LayerList;

std::map<int, DAGContext> CMap;
std::map<int, ErrorContext> EMap;

shared_ptr<ExitContext> gpExit = make_shared<ExitContext>();

//----------------------------------------------
class myMCallBack : public iCallBackSink
{
   //ofstream file;
public:
   myMCallBack() {
      //file.open(path + "//offset.csv");
   }
   ~myMCallBack() {
      //file.close();
   }
   void Properties(std::map<string, CallBackObj>& props) override
   {
      //file << props["T"].d.get() << endl;

      if (props["ID"].i.get() == 1) {
         EMap[3].SetStatus(false, props["dG"].rv.get()[0] );
      }
   }
};

shared_ptr<iCallBackSink> MCB;// = make_shared<myMCallBack>();
//----------------------------------------------
// 
//----------------------------------------------
class myMCallBack1 : public iCallBackSink
{
   ofstream file;
public:
   myMCallBack1() {
      file.open(path + "//DLDV.csv");
   }
   ~myMCallBack1() {
      file.close();
   }
   void Properties(std::map<string, CallBackObj>& props) override
   {
      file << props["DLDV"].vm.get()[0].norm() << endl;

      //EMap[3].SetStatus(false, props["dC"].rv.get().norm() );

   }
};

shared_ptr<iCallBackSink> MCB1;// = make_shared<myMCallBack1>();
//----------------------------------------------
//----------------------------------------------
class myMCallBack2 : public iCallBackSink
{
   ofstream file;
   ofstream file1;
public:
   myMCallBack2() {
      file.open(path + "//join.0.csv");
      //file.open(path + "//v.3.csv");
      //file1.open(path + "//u.3.csv");
   }
   ~myMCallBack2() {
      file.close();
      //file1.close();
   }
   void Properties(std::map<string, CallBackObj>& props) override
   {
      file << props["X"].cv.get() << endl;
      //file << props["V"].cv.get() << endl;
      //file1 << props["U"].cv.get() << endl;
   }
};

shared_ptr<iCallBackSink> MCB2;// = make_shared<myMCallBack2>();
//----------------------------------------------

class myVMCallBack : public iCallBackSink
{
   vector_of_matrix X;
   vector_of_matrix Z;
   int count = 0;
public:
   myVMCallBack() { }
   // Custom method.
   void Save(int lbl) {
      OWeightsCSVFile osi(path, "filter.in." + to_string(lbl));
      OWeightsCSVFile oso(path, "filter.out." + to_string(lbl));
      osi.Write(X[0], count);
      oso.Write(Z[0], count);
      count++;
   }
   // Inherited via iCallBackSink
   void Properties(std::map<string, CallBackObj>& props) override
   {
      X = props["X"].vm.get();
      Z = props["Z"].vm.get();
   }
};
shared_ptr<myVMCallBack> MVCB;// = make_shared<myVMCallBack>();
//----------------------------------------------

class LossCrossEntropyEx : public iLossLayer {
public:
   int Size;
   ColVector X;
   ColVector Y;
   RowVector G;

   LossCrossEntropyEx()
   {
      Size = 0;
   }

   LossCrossEntropyEx(int input_size, int output_size) :
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
         if (std::isnan(X[i])) {
            cout << "X NaN found. Terminating." << endl;
            throw runtime_error("Nan");
         }
         if (std::isnan(Y[i])) {
            cout << "Y NaN found. Terminating." << endl;
            throw runtime_error("Nan");
         }
         // No reason to evaulate this expression if y[i]==0.0 .
         if (y[i] != 0.0) {
            //                        Prevent undefined results when taking the log of 0
            //double v = x[i] > std::numeric_limits<Number>::epsilon() ? x[i] : std::numeric_limits<Number>::epsilon();
            loss -= y[i] * std::log(x[i] > std::numeric_limits<Number>::epsilon() ? x[i] : std::numeric_limits<Number>::epsilon());
         }
      }
      return loss;
   }

   RowVector LossGradient(void) {
      for (int i = 0; i < Size; i++) {
         if (X[i] <= std::numeric_limits<double>::epsilon()) {
            if (Y[i] <= std::numeric_limits<double>::epsilon()) {
               G[i] = 0.0;
            }
            else {
               // NOTE: Usually, if Y (the label) is not zero then it will be one.  The result of the division
               //       below will be a very large number.
               G[i] = -Y[i] / std::numeric_limits<double>::epsilon();
               //cout << "Loss Gradient encountered div by zero.  G =" << G[i] << endl; // Debug
            }
         }
         else {
            G[i] = -Y[i] / X[i];
         }
         // Gradient clipping.
         if (std::abs(G[i]) > 1.0E6) {
            G[i] = std::signbit(G[i]) == true ? -1.0E6 : 1.0E6;
         }
      }

      return G;
   }
};

class actSoftMaxEx : public iActive {
   int Size;
   Matrix J;
public:
   actSoftMaxEx(int size) : Size(size), J(size, size) {}
   actSoftMaxEx() : Size(0) {}
   void Resize(int size) {
      Size = size;
      J.resize(size, size);
   }
   ColVector Eval(ColVector& q) {
      double sum = 0.0;
      double avg = 0.0;
      const double mult = 1.0 / (double)Size;

      for (int i = 0; i < Size; i++) { avg += mult * q(i); }

      double mx = 0.0;
      if (avg < -100.0) {
         //if (avg < -700.0) {
         //   cout << "Average q < -100\n" << q.transpose() << endl << "hit a key" << endl;
         //   char c;
         //   cin >> c;
         //}
         mx = q.maxCoeff();
         if (mx < 0.0) {
            q.array() -= mx;
         }
      }

      ColVector t(q);
      for (int i = 0; i < Size; i++) { q(i) = exp(q(i)); }

      sum = q.sum();

      //if (sum <= std::numeric_limits<double>::epsilon()) {
      if (sum == 0.0) {
            //cout << "sum==0 mx = :" << mx << endl << "q before:" << endl
            //   << t << "\nq:\n" << q.transpose() << endl << "hit a key" << endl;
            //char c;
            //cin >> c;
         q.setRandom();  // Set to -1,1 for floating types.
         q.array() += 1.0;
         q /= q.sum();
      }
      else {
         q /= sum;
      }

      //for (int i = 0; i < Size; i++) {
      //   if (std::isnan(q[i])) {
      //      cout << "SoftMax NaN found. Terminating. sum = " << sum << endl << "q:" << endl << t << endl;
      //      throw runtime_error("Nan.");
      //   }
      //}


      return q;
   }
   ColVector Eval(Eigen::Map<ColVector>& q) {
      double sum = 0.0;
      for (int i = 0; i < Size; i++) { q(i) = exp(q(i)); }
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

#define INITIALIZE( NAME, OP ) \
   typedef OP OPTO;\
   model_name = NAME;\
   cout << "Initializing model: " << NAME << " , restore = " << restore << endl;\
   cout << "Optomizer: " #OP << endl;\
   if(!gAutoSave){\
   char s;\
   cout << "hit a key to continue";\
   cin >> s;\
   }

void InitLeNet5AModel(bool restore)
{
   INITIALIZE("LeNet5\\LeNet5A", optoADAM)
   gModelBranches = 1;

   LayerList.clear();

   optoADAM::B1 = 0.9;
   optoADAM::B2 = 0.999;
   optoLowPass::Momentum = 0.8;

   int l = 1; // Layer ID

   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   clSize size_in(INPUT_ROWS, INPUT_COLS);
   clSize size_out(INPUT_ROWS, INPUT_COLS);
   clSize size_kern(5, 5);
   int chn_in = 1;
   int chn_out = 6;
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
         make_unique<actTanh>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[1]));
   }

   //---------------------------------------------------------------
   //
   // Pooling Layer ----------------------------------------------
   // Type: poolAvg2D
   size_in = size_out;
   size_out.Resize(14, 14);
   chn_in = chn_out;

   assert(!(size_in.rows % size_out.rows));
   assert(!(size_in.cols % size_out.cols));
   {
      l = 2;
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
      //pl->SetEvalPostActivationCallBack(MCB);
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[1]));
   }
   //---------------------------------------------------------------

   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   size_in = size_out;
   size_out.Resize(10,10);
   // size_kern; // same.
   chn_in = chn_out;
   chn_out = 16;
   {
      l = 3;
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
         make_unique<actTanh>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[1]));
   }
   //---------------------------------------------------------------
   //
   // Pooling Layer ----------------------------------------------
   // Type: poolAvg2D
   size_in = size_out;
   size_out.Resize(5,5);
   chn_in = chn_out;

   assert(!(size_in.rows % size_out.rows));
   assert(!(size_in.cols % size_out.cols));
   {
      l = 4;
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>( make_shared<poolAvg2D>(size_in, chn_in, size_out) ), CMap[1]));
   }
   //---------------------------------------------------------------      

   // Convolution Layer 5 -----------------------------------------
   // Type: FilterLayer2D
   size_in = size_out;
   size_out.Resize(1,1);
   //size_kern; // same
   chn_in = chn_out;
   chn_out = 120;
   {
      l = 5;
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
         make_unique<actTanh>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[1]));
   }
   //---------------------------------------------------------------


   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in = size_out;
   chn_in = chn_out;
   int len_out = size_in.rows * size_in.cols * chn_in;
   chn_out = 1;
   {
      l = 6;
      shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[1] ));
   }
   //---------------------------------------------------------------      

   //--------- setup the fully connected network -------------------------------------------------------------------------

   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   int len_in = len_out;
   len_out = 84;
   {
      l = 7;
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
         //make_unique<actLeakyReLU>(0.01),
         make_unique<actReLU>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[1]));
   }

   //---------------------------------------------------------------      
  
   // Fully Connected Layer 7 ---------------------------------------
   // Type: SoftMAX
   len_in = len_out;
   len_out = 10;
   {
      l = 8;
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
         //make_unique<actLeakyReLU>(0.01),
         make_unique<actSoftMax>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[1]));
   }

   //---------------------------------------------------------------      

   // Error Layer ---------------------------------------
   // Type: LossCrossEntropy
   {
      l = 9;
      shared_ptr<LossCrossEntropy> pl = make_shared<LossCrossEntropy>(len_out, 1);
      // DAGErrorLayer(shared_ptr<iLossLayer> _pLayer, shared_ptr<NetContext> _pContext, shared_ptr<ErrorContext> _pEContext) 
      EMap.emplace(0, 0);
      LayerList.push_back(make_shared<DAGErrorLayer>(l, static_pointer_cast<iLossLayer>(pl), CMap[1], EMap[0]));
   }
   l++;

}
//---------------------------- End LeNet 5A ---------------------------------------

/*
//----------------------------------------------------------------------------------

void InitLeNet5BModel(bool restore)
{
   INITIALIZE("LeNet5\\LeNet5B", optoADAM)
   gModelBranches = 1;
   //bool strestore = restore;
   bool strestore = true;

   LayerList.clear();

   optoADAM::B1 = 0.9;
   optoADAM::B2 = 0.999;
   optoLowPass::Momentum = 0.8;

   int l = 1; // Layer counter

   clSize size_in(INPUT_ROWS, INPUT_COLS);
   clSize size_out(INPUT_ROWS, INPUT_COLS);
   clSize size_kern(28, 28);
   int chn_in = 1;
   int chn_out = 1;
   int len_out = 1;
   int len_in = 1;

   //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
// Copy Start Context -----------------------------------------
//
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, pContextStart, pContext1));
   l++;
   //
   //-----------------------------------------------------------------------
   // Copy Input Layer -----------------------------------------
   //
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, pContextStart, pContext4));
   l++;
   //
   //-----------------------------------------------------------------------
   //***********   BEGIN TRANSFORMER  *******************************
   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   size_in.Resize(INPUT_ROWS, INPUT_COLS);
   size_out.Resize(INPUT_ROWS, INPUT_COLS);
   size_kern.Resize(5, 5);
   chn_in = 1;
   chn_out = 2;
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
         make_unique<actTanh>(),
         strestore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), pContext4));
   }
   l++;
   //---------------------------------------------------------------
   //
   // Pooling Layer ----------------------------------------------
   // Type: poolAvg2D
   size_in = size_out;
   size_out.Resize(14, 14);
   chn_in = chn_out;

   assert(!(size_in.rows % size_out.rows));
   assert(!(size_in.cols % size_out.cols));
   {
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
      //pl->SetEvalPostActivationCallBack(MCB);
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), pContext4));
   }
   l++;
   //---------------------------------------------------------------

   // Convolution Layer 3 -----------------------------------------
   // Type: FilterLayer2D
   size_in = size_out;
   size_out.Resize(10, 10);
   // size_kern; // same.
   chn_in = chn_out;
   chn_out = 4;
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
         make_unique<actTanh>(),
         strestore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), pContext4));
   }
   l++;
   //---------------------------------------------------------------
   //
   // Pooling Layer 4 ----------------------------------------------
   // Type: poolAvg2D
   size_in = size_out;
   size_out.Resize(5, 5);
   chn_in = chn_out;

   assert(!(size_in.rows % size_out.rows));
   assert(!(size_in.cols % size_out.cols));
   {
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(make_shared<poolAvg2D>(size_in, chn_in, size_out)), pContext4));
   }
   l++;
   //---------------------------------------------------------------      

   // Convolution Layer 5 -----------------------------------------
   // Type: FilterLayer2D
   size_in = size_out;
   size_out.Resize(1, 1);
   //size_kern; // same
   chn_in = chn_out;
   chn_out = 32;
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
         make_unique<actTanh>(),
         strestore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), pContext4));
   }
   l++;
   //---------------------------------------------------------------

   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in = size_out;
   chn_in = chn_out;
   len_out = size_in.rows * size_in.cols * chn_in;
   chn_out = 1;
   {
      shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, static_pointer_cast<iConvoLayer>(pl), pContext4, pContext3));
   }
   l++;
   //---------------------------------------------------------------    
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   len_in = len_out;
   len_out = 16;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
         make_unique<actReLU>(),
         strestore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(0.01, 0.0)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), pContext3));
   }
   l++;
   //---------------------------------------------------------------  
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   //len_in = len_out;
   //len_out = 36;
   //{
   //   shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
   //      make_unique<actReLU>(),
   //      strestore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
   //      dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(0.1, 0.0)),
   //      make_shared<OPTO>());
   //   LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), pContext3));
   //}
   //l++;
   //---------------------------------------------------------------  
   // Fully Connected Layer ---------------------------------------
   // Type: Linear
   len_in = len_out;
   len_out = 6;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
         make_unique<actLinearForStAffine>(),
         strestore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(0.01, 0.0)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), pContext3));
   }
   l++;


   //-----------------------------------------------------------------
   // Create the Transformer
   //
   shared_ptr<SpacialTransformer> pST = make_shared<SpacialTransformer>(SpacialTransformer::Size(INPUT_ROWS, INPUT_COLS), 
                                                                        SpacialTransformer::Size(INPUT_ROWS, INPUT_COLS));

   // REVIEW: The two DAG can be combined.
   //-----------------------------------------------------------------
   LayerList.push_back(make_shared<DAGSpacialTransformSampleLayer>(l, pST, pContext1, pContext3,false));
   l++;
   //-----------------------------------------------------------------
   //-----------------------------------------------------------------
   LayerList.push_back(make_shared<DAGSpacialTransformGridLayer>(l, pST, pContext1, pContext3));
   l++;
   //-----------------------------------------------------------------
   //---------------------------------------------------------------  
   //**********    END TRANSFORMER  ***********************************
   // 
   // 
   //-----------------------------------------------------------------------
   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   size_in.Resize(INPUT_ROWS, INPUT_COLS);
   size_out.Resize(INPUT_ROWS, INPUT_COLS);
   size_kern.Resize(5, 5);
   chn_in = 1;
   chn_out = 6;
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
         make_unique<actTanh>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), pContext1));
   }
   l++;
   //---------------------------------------------------------------
   //
   // Pooling Layer ----------------------------------------------
   // Type: poolAvg2D
   size_in = size_out;
   size_out.Resize(14, 14);
   chn_in = chn_out;

   assert(!(size_in.rows % size_out.rows));
   assert(!(size_in.cols % size_out.cols));
   {
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
      //pl->SetEvalPostActivationCallBack(MCB);
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), pContext1));
   }
   l++;
   //---------------------------------------------------------------

   // Convolution Layer 3 -----------------------------------------
   // Type: FilterLayer2D
   size_in = size_out;
   size_out.Resize(10, 10);
   // size_kern; // same.
   chn_in = chn_out;
   chn_out = 16;
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
         make_unique<actTanh>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), pContext1));
   }
   l++;
   //---------------------------------------------------------------
   //
   // Pooling Layer 4 ----------------------------------------------
   // Type: poolAvg2D
   size_in = size_out;
   size_out.Resize(5, 5);
   chn_in = chn_out;

   assert(!(size_in.rows % size_out.rows));
   assert(!(size_in.cols % size_out.cols));
   {
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(make_shared<poolAvg2D>(size_in, chn_in, size_out)), pContext1));
   }
   l++;
   //---------------------------------------------------------------      

   // Convolution Layer 5 -----------------------------------------
   // Type: FilterLayer2D
   size_in = size_out;
   size_out.Resize(1, 1);
   //size_kern; // same
   chn_in = chn_out;
   chn_out = 120;
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
         make_unique<actTanh>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), pContext1));
   }
   l++;
   //---------------------------------------------------------------


   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in = size_out;
   chn_in = chn_out;
   len_out = size_in.rows * size_in.cols * chn_in;
   chn_out = 1;
   {
      shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, static_pointer_cast<iConvoLayer>(pl), pContext1, pContext2));
   }
   l++;
   //---------------------------------------------------------------      

   //--------- setup the fully connected network -------------------------------------------------------------------------

   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   len_in = len_out;
   len_out = 84;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
         //make_unique<actLeakyReLU>(0.01),
         make_unique<actReLU>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), pContext2));
   }
   l++;
   //---------------------------------------------------------------      

   // Fully Connected Layer 7 ---------------------------------------
   // Type: SoftMAX
   len_in = len_out;
   len_out = 10;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
         //make_unique<actLeakyReLU>(0.01),
         make_unique<actSoftMax>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), pContext2));
   }
   l++;
   //---------------------------------------------------------------      

   // Error Layer ---------------------------------------
   // Type: LossCrossEntropy
   {
      shared_ptr<LossCrossEntropy> pl = make_shared<LossCrossEntropy>(len_out, 1);
      // DAGErrorLayer(shared_ptr<iLossLayer> _pLayer, shared_ptr<NetContext> _pContext, shared_ptr<ErrorContext> _pEContext) 
      LayerList.push_back(make_shared<DAGErrorLayer>(l, static_pointer_cast<iLossLayer>(pl), pContext2, gpError1));
   }
   l++;

}

void LocalizerInitLeNet5B(bool restore)
{
   INITIALIZE("LeNet5\\LeNet5B", optoADAM);
   gModelBranches = 1;

   LayerList.clear();

   optoADAM::B1 = 0.9;
   optoADAM::B2 = 0.999;
   optoLowPass::Momentum = 0.8;

   int l = 1; // Layer counter

   clSize size_in(INPUT_ROWS, INPUT_COLS);
   clSize size_out(INPUT_ROWS, INPUT_COLS);
   clSize size_kern(28, 28);
   int chn_in = 1;
   int chn_out = 1;
   int len_out = 1;
   int len_in = 1;

   // Copy Input Layer -----------------------------------------
   //
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, pContextStart, pContext4));
   l++;
   //
   //-----------------------------------------------------------------------
   //***********   BEGIN TRANSFORMER  *******************************
   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   size_in.Resize(INPUT_ROWS, INPUT_COLS);
   size_out.Resize(INPUT_ROWS, INPUT_COLS);
   size_kern.Resize(5, 5);
   chn_in = 1;
   chn_out = 2;
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
         make_unique<actTanh>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), pContext4));
   }
   l++;
   //---------------------------------------------------------------
   //
   // Pooling Layer ----------------------------------------------
   // Type: poolAvg2D
   size_in = size_out;
   size_out.Resize(14, 14);
   chn_in = chn_out;

   assert(!(size_in.rows % size_out.rows));
   assert(!(size_in.cols % size_out.cols));
   {
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
      //pl->SetEvalPostActivationCallBack(MCB);
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), pContext4));
   }
   l++;
   //---------------------------------------------------------------

   // Convolution Layer 3 -----------------------------------------
   // Type: FilterLayer2D
   size_in = size_out;
   size_out.Resize(10, 10);
   // size_kern; // same.
   chn_in = chn_out;
   chn_out = 4;
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
         make_unique<actTanh>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), pContext4));
   }
   l++;
   //---------------------------------------------------------------
   //
   // Pooling Layer 4 ----------------------------------------------
   // Type: poolAvg2D
   size_in = size_out;
   size_out.Resize(5, 5);
   chn_in = chn_out;

   assert(!(size_in.rows % size_out.rows));
   assert(!(size_in.cols % size_out.cols));
   {
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(make_shared<poolAvg2D>(size_in, chn_in, size_out)), pContext4));
   }
   l++;
   //---------------------------------------------------------------      

   // Convolution Layer 5 -----------------------------------------
   // Type: FilterLayer2D
   size_in = size_out;
   size_out.Resize(1, 1);
   //size_kern; // same
   chn_in = chn_out;
   chn_out = 32;
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
         make_unique<actTanh>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), pContext4));
   }
   l++;
   //---------------------------------------------------------------

   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in = size_out;
   chn_in = chn_out;
   len_out = size_in.rows * size_in.cols * chn_in;
   chn_out = 1;
   {
      shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, static_pointer_cast<iConvoLayer>(pl), pContext4, pContext3));
   }
   l++;
   //---------------------------------------------------------------    
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   len_in = len_out;
   len_out = 16;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
         make_unique<actReLU>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(0.01, 0.0)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), pContext3));
   }
   l++;
   //---------------------------------------------------------------  
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   //len_in = len_out;
   //len_out = 36;
   //{
   //   shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
   //      make_unique<actReLU>(),
   //      restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
   //      dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(0.1, 0.0)),
   //      make_shared<OPTO>());
   //   LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), pContext3));
   //}
   //l++;
   //---------------------------------------------------------------  
   // Fully Connected Layer ---------------------------------------
   // Type: Linear
   len_in = len_out;
   len_out = 6;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
         make_unique<actLinearForStAffine>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(0.01, 0.0)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), pContext3));
   }
   l++;


   // Transformer Error Layer ---------------------------------------
   // Type: L2
   LayerList.push_back(make_shared<DAGErrorLayer>(l, make_shared<LossL2>(len_out, 1), pContext3, gpError1));
   //-----------------------------------------------------------------


//---------------------------------------------------------------  
//**********    END TRANSFORMER  ***********************************
}

// Well this didn't work!!
void LocalizerInitLeNet5B1(bool restore)
{
   INITIALIZE("LeNet5\\LeNet5B1", optoADAM);
   gModelBranches = 1;

   LayerList.clear();

   optoADAM::B1 = 0.9;
   optoADAM::B2 = 0.999;
   optoLowPass::Momentum = 0.8;

   int l = 1; // Layer counter

   clSize size_in(INPUT_ROWS, INPUT_COLS);
   clSize size_out(INPUT_ROWS, INPUT_COLS);
   clSize size_kern(28, 28);
   int chn_in = 1;
   int chn_out = 1;
   int len_out = 1;
   int len_in = 1;

   // Copy Input Layer -----------------------------------------
   //
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, pContextStart, pContext4));
   l++;
   //
   //-----------------------------------------------------------------------
   //***********   BEGIN TRANSFORMER  *******************************
   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   size_in.Resize(INPUT_ROWS, INPUT_COLS);
   size_out.Resize(20, 20);
   size_kern.Resize(7, 7);
   chn_in = 1;
   chn_out = 2;
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
         make_unique<actTanh>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), pContext4));
   }
   l++;
   //---------------------------------------------------------------
   //

   // Convolution Layer 3 -----------------------------------------
   // Type: FilterLayer2D
   size_in = size_out;
   size_out.Resize(14, 14);
   // size_kern; // same.
   chn_in = chn_out;
   chn_out = 2;
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
         make_unique<actTanh>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), pContext4));
   }
   l++;
   //---------------------------------------------------------------
   //  

   // Convolution Layer 5 -----------------------------------------
   // Type: FilterLayer2D
   size_in = size_out;
   size_out.Resize(7, 7);
   //size_kern; // same
   chn_in = chn_out;
   chn_out = 1;
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
         make_unique<actTanh>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), pContext4));
   }
   l++;
   //---------------------------------------------------------------

   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in = size_out;
   chn_in = chn_out;
   len_out = size_in.rows * size_in.cols * chn_in;
   chn_out = 1;
   {
      shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, static_pointer_cast<iConvoLayer>(pl), pContext4, pContext3));
   }
   l++;
   //---------------------------------------------------------------    
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   len_in = len_out;
   len_out = 16;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
         make_unique<actReLU>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(0.01, 0.0)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), pContext3));
   }
   l++;
   //---------------------------------------------------------------  
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   //len_in = len_out;
   //len_out = 36;
   //{
   //   shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
   //      make_unique<actReLU>(),
   //      restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
   //      dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(0.1, 0.0)),
   //      make_shared<OPTO>());
   //   LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), pContext3));
   //}
   //l++;
   //---------------------------------------------------------------  
   // Fully Connected Layer ---------------------------------------
   // Type: Linear
   len_in = len_out;
   len_out = 6;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
         make_unique<actLinearForStAffine>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(0.01, 0.0)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), pContext3));
   }
   l++;


   // Transformer Error Layer ---------------------------------------
   // Type: L2
   LayerList.push_back(make_shared<DAGErrorLayer>(l, make_shared<LossL2>(len_out, 1), pContext3, gpError1));
   //-----------------------------------------------------------------


//---------------------------------------------------------------  
//**********    END TRANSFORMER  ***********************************
}

//---------------------------- End LeNet 5B ---------------------------------------

*/

void InitLPBranchModel1(bool restore)
{
   INITIALIZE("LPB1\\LPB1", optoADAM)
   gModelBranches = 1;

   optoADAM::B1 = 0.9;
   optoADAM::B2 = 0.999;
   optoLowPass::Momentum = 0.8;

   LayerList.clear();

   // NOTE: Used by Filter Pool.
   clSize size_kernel(2, 4);


   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   clSize size_in(INPUT_ROWS, INPUT_COLS);
   clSize size_out(INPUT_ROWS, 4);
   clSize size_kern(INPUT_ROWS, INPUT_COLS);
   int chn_in = 1;
   int chn_out = 1;
   int l = 1; // Layer counter
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
         make_unique<actReLU>(),
         //make_unique<actLinear>(), 
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      pl->SetConvolutionParameters(FilterLayer2D::FFT, true, false);

      //pl->SetBackpropCallBack(MCB);

      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[1] ));
   }
   l++;
   //---------------------------------------------------------------
   // Pooling Layer ----------------------------------------------
   // Type: poolAvg2D
   size_in = size_out;
   size_out.Resize(INPUT_ROWS, 1);
   chn_in = chn_out;

   assert(!(size_in.rows % size_out.rows));
   assert(!(size_in.cols % size_out.cols));
   {
#ifdef USE_FILTER_FOR_POOL
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      //pl->SetEvalPostActivationCallBack(MVCB);
#else
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
#endif
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, CMap[1] ));
   }
   l++;
   //---------------------------------------------------------------
   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in = size_out;
   chn_in = chn_out;
   int len_out = size_in.rows * size_in.cols * chn_in;
   chn_out = 1;
   {
      shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[1] ));
   }
   l++;
   //---------------------------------------------------------------      
   // 
   //--------- setup the fully connected network -------------------------------------------------------------------------
   // 
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   int len_in = len_out;
   len_out = 16;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, 
         make_unique<actReLU>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[1] ));
   }
   l++;
   //---------------------------------------------------------------  

   // Fully Connected Layer ---------------------------------------
   // Type: SoftMAX
   len_in = len_out;
   len_out = 10;
   {
      // REVIEW:  USING SoftMaxEx   !!!!!!!!!!!!!!!!
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actSoftMaxEx>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      //pl->SetBackpropCallBack(MCB1);
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[1] ));
   }
   l++;
   //---------------------------------------------------------------      
   // Error Layer ---------------------------------------
   // Type: LossCrossEntropy
   {
      //shared_ptr<LossCrossEntropy> pl = make_shared<LossCrossEntropy>(len_out, 1);
      shared_ptr<LossCrossEntropyEx> pl = make_shared<LossCrossEntropyEx>(len_out, 1);
      EMap.emplace(1, 1);
      LayerList.push_back(make_shared<DAGErrorLayer>(l, static_pointer_cast<iLossLayer>(pl), CMap[1], EMap[1]));
   }
   l++;
}

void InitLPBranchModel1S(bool restore)
{
   INITIALIZE("LPB1\\LPB1S", optoADAM)
   gModelBranches = 1;

   optoADAM::B1 = 0.9;
   optoADAM::B2 = 0.999;
   optoLowPass::Momentum = 0.8;

   LayerList.clear();

   // NOTE: Used by Filter Pool.
   clSize size_kernel(2, 4);

   CMap[0].v.resize(1);
   CMap[0].v[0] = 0;

   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   clSize size_in(INPUT_ROWS, INPUT_COLS);
   clSize size_out(INPUT_ROWS, 4);
   clSize size_kern(INPUT_ROWS, INPUT_COLS);
   int chn_in = 1;
   int chn_out = 1;
   int l = 1; // Layer counter
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
         make_unique<actReLU>(),
         //make_unique<actLinear>(), 
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      pl->SetConvolutionParameters(FilterLayer2D::FFT, true, false);

      //pl->SetBackpropCallBack(MCB);

      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[1]));
   }
   //---------------------------------------------------------------
   // Pooling Layer ----------------------------------------------
   // Type: poolAvg2D
   size_in = size_out;
   size_out.Resize(INPUT_ROWS, 1);
   chn_in = chn_out;
   l = 2;
   assert(!(size_in.rows % size_out.rows));
   assert(!(size_in.cols % size_out.cols));
   {
#ifdef USE_FILTER_FOR_POOL
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      //pl->SetEvalPostActivationCallBack(MVCB);
#else
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
#endif
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, CMap[1]));
   }
   //---------------------------------------------------------------
   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in = size_out;
   chn_in = chn_out;
   int len_out = size_in.rows * size_in.cols * chn_in;
   chn_out = 1;
   l = 3;
   {
      shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[1]));
   }
   //---------------------------------------------------------------    
      // Copy Context2 to Spectrum Context -----------------------------------------
      //
   l = 4;
   LayerList.push_back(make_shared<DAGContextCopyObj>(l, CMap[1], CMap[2]));
   //l++;
   //
   //-----------------------------------------------------------------------

   // Spectrum Layer ---------------------------------------
   //
   int len_in_branch_t = len_out;
   int len_out_branch_t = len_out / 2;
   {
      l = 5;
      shared_ptr<SpectrumOutputLayer> pl = make_shared<SpectrumOutputLayer>(len_in_branch_t, len_out_branch_t);
      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, CMap[2]));
   }
   //---------------------------------------------------------------    
   // Join Fully Connected Layer -----------------------------------------
   //
   l = 6;
   len_out_branch_t += len_out;  // out len of fully connected layer prior to Transformer layer.
   LayerList.push_back(make_shared<DAGJoinObj>(l, CMap[2], CMap[1]));
   //
   //-----------------------------------------------------------------------    
   // 
   //--------- setup the fully connected network -------------------------------------------------------------------------
   // 
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   int len_in = len_out_branch_t;
   len_out = 32;
   l = 7;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
         make_unique<actReLU>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[1]));
   }
   //---------------------------------------------------------------  

   // Fully Connected Layer ---------------------------------------
   // Type: SoftMAX
   len_in = len_out;
   len_out = 10;
   l = 8;
   {
      // REVIEW:  USING SoftMaxEx   !!!!!!!!!!!!!!!!
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actSoftMaxEx>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      //pl->SetBackpropCallBack(MCB1);
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[1]));
   }
   //---------------------------------------------------------------      
   // Error Layer ---------------------------------------
   // Type: LossCrossEntropy
   l = 9;
   {
      //shared_ptr<LossCrossEntropy> pl = make_shared<LossCrossEntropy>(len_out, 1);
      shared_ptr<LossCrossEntropyEx> pl = make_shared<LossCrossEntropyEx>(len_out, 1);
      EMap.emplace(1, 1);
      LayerList.push_back(make_shared<DAGErrorLayer>(l, static_pointer_cast<iLossLayer>(pl), CMap[1], EMap[1]));
   }
}

#define TRAIN_TRANSFORMER
#define TRAIN_NET
#define TRAIN_END_TO_END
//#define THREE_LOC

void InitLPBranchModel1T(bool restore)
{
   INITIALIZE("LPB1\\LPB1T", optoLowPass)
   gModelBranches = 1;


#ifdef TRAIN_END_TO_END
   bool lrestore = true;
   lrestore = restore;
#else
   #ifdef TRAIN_TRANSFORMER
      // NOTE: Use restore option to turn on or off localize net restore.
      //       Training the transformer is based on a trained primary network so that
      //       should always be restored.
      bool lrestore = restore;
      restore = true;
   #endif
#endif

   optoADAM::B1 = 0.9;
   optoADAM::B2 = 0.999;
   optoLowPass::Momentum = 0.8;

   LayerList.clear();

   // NOTE: Used by Filter Pool.
   // REIVEW: !!!!!!!!!!!!!!!!  LOOK LOOK !!!!!!!!!!!!!!!!!!!!!!!
   //          Set to 4,4 to test downsampling.  It has been 2,4 .
   clSize size_kernel(2, 4);

   int l = 0; // Layer counter

   // copy to 200
   // copy to 2
   // Filter layer
   // Centroid layer
   // LP Spacial Transformer c2, c200


   // Copy Start Context -----------------------------------------
   //
   l = 1;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[2]));
   //
   //-----------------------------------------------------------------------
   // 
   //***************** LP Transformer *******************************
   // 
   // Copy Start Context -----------------------------------------
   //
   l = 20;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[200]));
   //
   //-----------------------------------------------------------------------

   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   clSize size_ct_in(28, 28);
   clSize size_ct_out(28, 28);
   clSize size_ct_kern(28, 28);
   {
      l = 21;
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_ct_in, 1, size_ct_out, size_ct_kern, 1,
         make_unique<actReLU>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>(),
         true); // No bias. true/false
      pl->SetConvolutionParameters(FilterLayer2D::FFT, false, false);

      //pl->SetBackpropCallBack(MCB);

      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[200]));
   }
   //---------------------------------------------------------------

   // Centroid Layer --------------------------------------------
   // Type: Centroid
   // Note: Uses DAGFlattenObj.
   size_ct_in = size_ct_out;
   {
      l = 22;
      shared_ptr<Centroid> pl = make_shared<Centroid>(size_ct_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, CMap[200]));
   }
   //-----------------------------------------------------------------

   //---------------------------------------------------------------  
   // LP Spacial Transformer
   //
   {
      l = 23;
      shared_ptr<samplerBiLinear> psmp = make_shared<samplerBiLinear>(Size(28, 28), Size(INPUT_ROWS, INPUT_COLS));
      shared_ptr<gridLogPolar> pgrd = make_shared<gridLogPolar>(Size(28, 28), Size(INPUT_ROWS, INPUT_COLS), LP_WASTE);
      // REVIEW: Transform sanity check test.
      //pCVT->SetEvalPostActivationCallBack(MCB2);
      LayerList.push_back(make_shared<DAGSpacialTransformLayer>(l, psmp, pgrd, CMap[2], CMap[200], false)); // C200 holds center point value.
   }
   //-----------------------------------------------------------------

   //********* End LP Transformer *********************************************

   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   clSize size_in(INPUT_ROWS, INPUT_COLS);
   clSize size_out(INPUT_ROWS, 4);
   clSize size_kern(INPUT_ROWS, INPUT_COLS);
   int chn_in = 1;
   int chn_out = 1;
   {
      l = 2;
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
         make_unique<actReLU>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false
      pl->SetConvolutionParameters(FilterLayer2D::FFT, true, false);

      //pl->SetBackpropCallBack(MCB);

      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[2]));
   }
   //---------------------------------------------------------------
   // Pooling Layer ----------------------------------------------
   // Type: poolAvg2D
   size_in = size_out;
   size_out.Resize(INPUT_ROWS, 1);
   chn_in = chn_out;

   assert(!(size_in.rows % size_out.rows));
   assert(!(size_in.cols % size_out.cols));
   {
      l = 3;
#ifdef USE_FILTER_FOR_POOL
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      //pl->SetEvalPostActivationCallBack(MVCB);
#else
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
#endif
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, CMap[2]));
   }
   //---------------------------------------------------------------
   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in = size_out;
   chn_in = chn_out;

   int len_in = 0;
   int len_out = size_in.rows* size_in.cols* chn_in;
   chn_out = 1;
   {
      l = 4;
      shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, CMap[2]));
   }
   //-----------------------------------------------------------------
   //           Transformer Layers
   //---------------------------------------------------------------      
   // 
#ifdef TRAIN_TRANSFORMER
      // Copy Context2 to Spectrum Context -----------------------------------------
      //
      l = 9;
      LayerList.push_back(make_shared<DAGContextCopyObj>(l, CMap[2], CMap[3]));
      //
      //-----------------------------------------------------------------------

      // Spectrum Layer ---------------------------------------
      //
      int len_in_branch_t = len_out;
      //int len_out_branch_t = 3 * len_out / 8;
      int len_out_branch_t = len_out / 2;
      {
         l = 10;
         shared_ptr<SpectrumOutputLayer> pl = make_shared<SpectrumOutputLayer>(len_in_branch_t, len_out_branch_t );
         LayerList.push_back(make_shared<DAGLayerObj>(l, pl, CMap[3]));
      }
      //---------------------------------------------------------------    
      // Branch to Transform Layer -----------------------------------------
      //
      l = 11;
      shared_ptr<DAGBranchObj> p_branch_t = make_shared<DAGBranchObj>(l, CMap[2], CMap[4], false); // false -> no backprop.
      LayerList.push_back(p_branch_t);
      //
      //-----------------------------------------------------------------------

      // Join Fully Connected Layer -----------------------------------------
      //
      l = 12;
      len_out_branch_t += len_out;  // out len of fully connected layer prior to Transformer layer.
      LayerList.push_back(make_shared<DAGJoinObj>(l, CMap[4], CMap[3]));
      //
      //-----------------------------------------------------------------------  
#ifndef TRAIN_END_TO_END
      // Force Stop Layer -----------------------------------------------------
      // Stop backprop.
      l = 16;
      LayerList.push_back(make_shared<DAGLambdaLayer>(l, []() {}, []() { gpExit->stop = true;  }) );
#endif

#ifdef THREE_LOC
      //-------------------------------------------------------------------------
      // Fully Connected Layer ---------------------------------------
      // Type: ReLU
      len_in_branch_t = len_out_branch_t;
      len_out_branch_t = 120;
      {
         l = 13;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in_branch_t, len_out_branch_t,
            make_unique<actReLU>(),
            lrestore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         pl->SetEvalPreActivationCallBack(MCB2);
         LayerList.push_back(make_shared<DAGLayerObj>(l, pl, CMap[3]));
      }
      //l++;
      //---------------------------------------------------------------  
      // Fully Connected Layer ---------------------------------------
      // Type: ReLU
      len_in_branch_t = len_out_branch_t;
      len_out_branch_t = 52;
      {
         l = 14;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in_branch_t, len_out_branch_t,
            make_unique<actReLU>(),
            lrestore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, pl, CMap[3]));
      }
      //l++;
      //---------------------------------------------------------------  
      // Fully Connected Layer ---------------------------------------
      // Type: ReLU
      len_in_branch_t = len_out_branch_t;
      len_out_branch_t = 24;
      {
         l = 18;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in_branch_t, len_out_branch_t,
            make_unique<actReLU>(),
            lrestore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, pl, CMap[3]));
      }
      //l++;
      //---------------------------------------------------------------  
#else

            // Fully Connected Layer ---------------------------------------
      // Type: ReLU
      len_in_branch_t = len_out_branch_t;
      len_out_branch_t = 60;
      {
         l = 13;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in_branch_t, len_out_branch_t,
            make_unique<actReLU>(),
            lrestore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         pl->SetEvalPreActivationCallBack(MCB2);
         LayerList.push_back(make_shared<DAGLayerObj>(l, pl, CMap[3]));
      }
      //---------------------------------------------------------------  
      // Fully Connected Layer ---------------------------------------
      // Type: ReLU
      len_in_branch_t = len_out_branch_t;
      len_out_branch_t = 30;
      {
         l = 14;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in_branch_t, len_out_branch_t,
            make_unique<actReLU>(),
            lrestore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, pl, CMap[3]));
      }
      //---------------------------------------------------------------  

#endif

      // Fully Connected Layer ---------------------------------------
      // Type: Linear
      len_in_branch_t = len_out_branch_t;
      len_out_branch_t = 1;
      {
         l = 15;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in_branch_t, len_out_branch_t,
            //make_unique<actLinear>(), 
            make_unique<actLinearEx>(),
            lrestore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());

         //pl->SetBackpropCallBack(MCB1);

         LayerList.push_back(make_shared<DAGLayerObj>(l, pl, CMap[3]));
      }
      //l++;
      // -----------------------------------------------------------------
      // Locializer Loss Layer -------------------------------------------
      //
      // NOTE: This must be removed for end-to-end run.
#ifndef TRAIN_END_TO_END
      {
         l = 17;
         shared_ptr<LossL2X> pl = make_shared<LossL2X>(len_out, 1);
         EMap.emplace(1, 1);
         LayerList.push_back(make_shared<DAGErrorLayer>(l, pl, CMap[3], EMap[1]));
      }
#endif
      //---------------------------------------------------------------------
#endif
#ifdef TRAIN_NET
      //---------------------------------------------------------------  
      // Cyclic Transformer
      //
      len_in = len_out;
      {
         l = 5;
         shared_ptr<CyclicVectorTransformer> pCVT = make_shared<CyclicVectorTransformer>(len_in, len_out);
         // REVIEW: Transform sanity check test.
         //pCVT->SetEvalPostActivationCallBack(MCB2);
         LayerList.push_back(make_shared<DAGCyclicTransformLayer>(l, pCVT, CMap[2], CMap[3])); // C3 holds rotation value.
      }
      //-----------------------------------------------------------------
      // Copy Context 3 to Context 0 -----------------------------------------
      //
      // NOTE: This is just to put this model in line with others that use an average rotation which
      //       is stored in C0 by convention.
      l = 0;
      LayerList.push_back(make_shared<DAGContextCopyObj>(l, CMap[3], CMap[0]));
      //
      //-----------------------------------------------------------------------

   //****************** End Transformer Layer ***********************************************     
   //--------- setup the fully connected network -------------------------------------------------------------------------
   // 
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   len_in = len_out;
   len_out = 16;
   {
      l = 6;
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
         make_unique<actReLU>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[2]));
   }
   //l++;
   //---------------------------------------------------------------  

   // Fully Connected Layer ---------------------------------------
   // Type: SoftMAX
   len_in = len_out;
   len_out = 10;
   {
      l = 7;
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actSoftMaxEx>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[2]));
   }
   //l++;
   //---------------------------------------------------------------      
   // Error Layer ---------------------------------------
   // Type: LossCrossEntropy
   {
      l = 8;
      shared_ptr<LossCrossEntropyEx> pl = make_shared<LossCrossEntropyEx>(len_out, 1);
      EMap.emplace(1, 1);
      LayerList.push_back(make_shared<DAGErrorLayer>(l, static_pointer_cast<iLossLayer>(pl), CMap[2], EMap[1]));
   }
   //l++;
#endif
}

void InitLPBranchModel2T(bool restore)
{
   INITIALIZE("LPB2\\LPB2T", optoLowPass)
   gModelBranches = 2;


#ifdef TRAIN_END_TO_END
   bool lrestore = true;
   lrestore = restore;
#else
#ifdef TRAIN_TRANSFORMER
   // NOTE: Use restore option to turn on or off localize net restore.
   //       Training the transformer is based on a trained primary network so that
   //       should always be restored.
   bool lrestore = restore;
   restore = true;
#endif
#endif

   //optoADAM::B1 = 0.9;
   optoADAM::B1 = 0.7;
   optoADAM::B2 = 0.999;
   optoLowPass::Momentum = 0.8;

   LayerList.clear();

   // NOTE: Used by Filter Pool.
   // REIVEW: !!!!!!!!!!!!!!!!  LOOK LOOK !!!!!!!!!!!!!!!!!!!!!!!
   //          Set to 4,4 to test downsampling.  It has been 2,4 .
   clSize size_kernel(2, 4);

   int l = 1; // Layer counter

   // Copy Start Context -----------------------------------------
   //
   l = 1;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[2]));
   //
   //-----------------------------------------------------------------------

   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   clSize size_in(INPUT_ROWS, INPUT_COLS);
   clSize size_out(INPUT_ROWS, 4);
   clSize size_kern(INPUT_ROWS, INPUT_COLS);
   int chn_in = 1;
   int chn_out = 1;
   {
      l = 2;
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
         make_unique<actReLU>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false

      //pl->SetBackpropCallBack(MCB);

      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[2]));
   }
   //---------------------------------------------------------------
   // Pooling Layer ----------------------------------------------
   // Type: poolAvg2D
   size_in = size_out;
   size_out.Resize(INPUT_ROWS, 1);
   chn_in = chn_out;

   assert(!(size_in.rows % size_out.rows));
   assert(!(size_in.cols % size_out.cols));
   {
      l = 3;
#ifdef USE_FILTER_FOR_POOL
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      //pl->SetEvalPostActivationCallBack(MVCB);
#else
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
#endif
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, CMap[2]));
   }
   //---------------------------------------------------------------
   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in = size_out;
   chn_in = chn_out;

   int len_in = 0;
   int len_out = size_in.rows * size_in.cols * chn_in;
   chn_out = 1;
   {
      l = 4;
      shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, CMap[2]));
   }
   //-----------------------------------------------------------------
   //           Transformer Layers
   //---------------------------------------------------------------      
   // 
#ifdef TRAIN_TRANSFORMER
      // Copy Context2 to Spectrum Context -----------------------------------------
      //
   l = 9;
   LayerList.push_back(make_shared<DAGContextCopyObj>(l, CMap[2], CMap[3]));
   //
   //-----------------------------------------------------------------------

   // Spectrum Layer ---------------------------------------
   //
   int len_in_branch_t = len_out;
   //int len_out_branch_t = 3 * len_out / 8;
   int len_out_branch_t = len_out / 2;
   {
      l = 10;
      shared_ptr<SpectrumOutputLayer> pl = make_shared<SpectrumOutputLayer>(len_in_branch_t, len_out_branch_t);
      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, CMap[3]));
   }
   //---------------------------------------------------------------    
   // Branch to Transform Layer -----------------------------------------
   //
   {
      l = 11;
      shared_ptr<DAGBranchObj> p_branch_t = make_shared<DAGBranchObj>(l, CMap[2], CMap[4], false); // false -> no backprop.
      LayerList.push_back(p_branch_t);
   }
   //
   //-----------------------------------------------------------------------

   // Join Fully Connected Layer -----------------------------------------
   //
   l = 12;
   len_out_branch_t += len_out;  // out len of fully connected layer prior to Transformer layer.
   LayerList.push_back(make_shared<DAGJoinObj>(l, CMap[4], CMap[3]));
   //
   //-----------------------------------------------------------------------  
#ifndef TRAIN_END_TO_END
      // Force Stop Layer -----------------------------------------------------
      // Stop backprop.
   l = 16;
   LayerList.push_back(make_shared<DAGLambdaLayer>(l, []() {}, []() { gpExit->stop = true;  }));
#endif

   // Fully Connected Layer ---------------------------------------
// Type: ReLU
   len_in_branch_t = len_out_branch_t;
   len_out_branch_t = 60;
   {
      l = 13;
      shared_ptr<Layer> pl = make_shared<Layer>(len_in_branch_t, len_out_branch_t,
         make_unique<actReLU>(),
         lrestore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      pl->SetEvalPreActivationCallBack(MCB2);
      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, CMap[3]));
   }
   //---------------------------------------------------------------  
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   len_in_branch_t = len_out_branch_t;
   len_out_branch_t = 30;
   {
      l = 14;
      shared_ptr<Layer> pl = make_shared<Layer>(len_in_branch_t, len_out_branch_t,
         make_unique<actReLU>(),
         lrestore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, CMap[3]));
   }
   //---------------------------------------------------------------  


      // Fully Connected Layer ---------------------------------------
      // Type: Linear
   len_in_branch_t = len_out_branch_t;
   len_out_branch_t = 1;
   {
      l = 15;
      shared_ptr<Layer> pl = make_shared<Layer>(len_in_branch_t, len_out_branch_t,
         //make_unique<actLinear>(), 
         make_unique<actLinearEx>(),
         lrestore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());

      //pl->SetBackpropCallBack(MCB1);

      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, CMap[3]));
   }
   // -----------------------------------------------------------------
#ifndef TRAIN_END_TO_END
   // Locializer Loss Layer -------------------------------------------
   //
   {
      l = 17;
      shared_ptr<LossL2X> pl = make_shared<LossL2X>(len_out, 1);
      // DAGErrorLayer(shared_ptr<iLossLayer> _pLayer, shared_ptr<NetContext> _pContext, shared_ptr<ErrorContext> _pEContext) 
      LayerList.push_back(make_shared<DAGErrorLayer>(l, pl, CMap[3], EMap[0]));
   }
   //---------------------------------------------------------------------
#endif
   // ------ Vector Composite -----------------------------------------
   //
   // NOTE: C0 carries average transform parameter.  Initialize it here.
   CMap[0].v.resize(1);
   l = 21;
   LayerList.push_back( make_shared<DAGCompAvg>(l, CMap[3], CMap[0]) );
   //
   //-----------------------------------------------------------------------
#endif
#ifdef TRAIN_NET
      //---------------------------------------------------------------  
      // Cyclic Transformer
      //
   len_in = len_out;
   {
      l = 5;
      shared_ptr<CyclicVectorTransformer> pCVT = make_shared<CyclicVectorTransformer>(len_in, len_out);
      // REVIEW: Transform sanity check test.
      //pCVT->SetEvalPostActivationCallBack(MCB2);
      LayerList.push_back(make_shared<DAGCyclicTransformLayer>(l, pCVT, CMap[2], CMap[0])); // C5 holds rotation value.
   }
   //-----------------------------------------------------------------

//****************** End Transformer Layer ***********************************************     
//--------- setup the fully connected network -------------------------------------------------------------------------
// 
   // Out to Branch 2 Fully Connected Layer -----------------------------------------
   //
   int len_out_branch_1 = len_out;
   // This branch will be paired with Test1.
   l = 19;
   shared_ptr<DAGBranchObj> p_branch1 = make_shared<DAGBranchObj>(l, CMap[2], CMap[6], false);
   LayerList.push_back(p_branch1);
   //
   //-----------------------------------------------------------------------
// Fully Connected Layer ---------------------------------------
// Type: ReLU
   len_in = len_out;
   len_out = 16;
   {
      l = 6;
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
         make_unique<actReLU>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[2]));
   }
   //l++;
   //---------------------------------------------------------------  

   // Fully Connected Layer ---------------------------------------
   // Type: SoftMAX
   len_in = len_out;
   len_out = 10;
   {
      l = 7;
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actSoftMaxEx>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[2]));
   }
   //---------------------------------------------------------------      
   // Error Layer ---------------------------------------
   // Type: LossCrossEntropy
   {
      l = 8;
      shared_ptr<LossCrossEntropyEx> pl = make_shared<LossCrossEntropyEx>(len_out, 1);
      // DAGErrorLayer(shared_ptr<iLossLayer> _pLayer, shared_ptr<NetContext> _pContext, shared_ptr<ErrorContext> _pEContext) 
      LayerList.push_back(make_shared<DAGErrorLayer>(l, static_pointer_cast<iLossLayer>(pl), CMap[2], EMap[0]));
   }

   //---------------------------------------------------------------      
   // Branch Test Layer (Test 1) -----------------------------------------
   // upper limit: read --> if you're that accurate don't bother backpropagating, not much to learn.
   // lower limit: read --> if you're that screwed up I don't want to try to learn what you are.
   //                                                                              warm up             backprop lower limit   backprop upper limit
   l = 20;
   //LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, EMap[1], restore ? 0 : 6 * 1100, 0.7, 0.98));
   LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, EMap[0], 20 * 1100, 0.0, 0.0));
   //
   //-----------------------------------------------------------------------
   //************************************************************************
   //                       Branch 2
   //************************************************************************
   // Copy Start Context -----------------------------------------
   //
   // NOTE: Start branch 2 at ID = 30 to give room to add to branch 1.
   l = 30;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[7]));
   //
   //-----------------------------------------------------------------------

   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   size_in.Resize(INPUT_ROWS, INPUT_COLS);
   size_out.Resize(INPUT_ROWS, 4);
   chn_in = 1;
   chn_out = 1;
   {
      l = 31;
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
         make_unique<actReLU>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false

      //pl->SetBackpropCallBack(MCB);

      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[7]));
   }
   //---------------------------------------------------------------
   // Pooling Layer ----------------------------------------------
   // Type: poolAvg2D
   size_in = size_out;
   size_out.Resize(INPUT_ROWS, 1);
   chn_in = chn_out;

   assert(!(size_in.rows% size_out.rows));
   assert(!(size_in.cols% size_out.cols));
   {
      l = 32;
#ifdef USE_FILTER_FOR_POOL
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      //pl->SetEvalPostActivationCallBack(MVCB);
#else
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
#endif
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, CMap[7]));
   }
   //---------------------------------------------------------------
   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in = size_out;
   chn_in = chn_out;

   len_in = 0;
   len_out = size_in.rows * size_in.cols * chn_in;
   chn_out = 1;
   {
      l = 33;
      shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, CMap[7]));
   }
   //-----------------------------------------------------------------
   //           Transformer Layers
   //---------------------------------------------------------------      
   // 
   // Copy to Spectrum Context -----------------------------------------
   //
   l = 38;
   LayerList.push_back(make_shared<DAGContextCopyObj>(l, CMap[7], CMap[8]));
   //
   //-----------------------------------------------------------------------

   // Spectrum Layer ---------------------------------------
   //
   len_in_branch_t = len_out;
   //int len_out_branch_t = 3 * len_out / 8;
   len_out_branch_t = len_out / 2;
   {
      l = 39;
      shared_ptr<SpectrumOutputLayer> pl = make_shared<SpectrumOutputLayer>(len_in_branch_t, len_out_branch_t);
      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, CMap[8]));
   }
   //---------------------------------------------------------------    
   // Branch to Transform Layer -----------------------------------------
   //
   {
      l = 40;
      shared_ptr<DAGBranchObj> p_branch_t = make_shared<DAGBranchObj>(l, CMap[7], CMap[9], false); // false -> no backprop.
      LayerList.push_back(p_branch_t);
   }
   //
   //-----------------------------------------------------------------------

   // Join Fully Connected Layer -----------------------------------------
   //
   l = 41;
   len_out_branch_t += len_out;  // out len of fully connected layer prior to Transformer layer.
   LayerList.push_back(make_shared<DAGJoinObj>(l, CMap[9], CMap[8]));
   //
   //-----------------------------------------------------------------------  

   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   len_in_branch_t = len_out_branch_t;
   len_out_branch_t = 60;
   {
      l = 42;
      shared_ptr<Layer> pl = make_shared<Layer>(len_in_branch_t, len_out_branch_t,
         make_unique<actReLU>(),
         lrestore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      pl->SetEvalPreActivationCallBack(MCB2);
      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, CMap[8]));
   }
   //---------------------------------------------------------------  
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   len_in_branch_t = len_out_branch_t;
   len_out_branch_t = 30;
   {
      l = 43;
      shared_ptr<Layer> pl = make_shared<Layer>(len_in_branch_t, len_out_branch_t,
         make_unique<actReLU>(),
         lrestore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, CMap[8]));
   }
   //---------------------------------------------------------------  


   // Fully Connected Layer ---------------------------------------
   // Type: Linear
   len_in_branch_t = len_out_branch_t;
   len_out_branch_t = 1;
   {
      l = 45;
      shared_ptr<Layer> pl = make_shared<Layer>(len_in_branch_t, len_out_branch_t,
         //make_unique<actLinear>(), 
         make_unique<actLinearEx>(),
         lrestore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());

      //pl->SetBackpropCallBack(MCB1);

      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, CMap[8]));
   }
   // -----------------------------------------------------------------
   // ------ Vector Composite -----------------------------------------
   //
   // NOTE: pContext4 carries average transform parameter.
   l = 34;
   LayerList.push_back(make_shared<DAGCompAvg>(l, CMap[8], CMap[0]));
   //
   //-----------------------------------------------------------------------
#endif
   //---------------------------------------------------------------  
   // Cyclic Transformer
   //
   // NOTE: Here there is oportunity to up-sample or down-sample.
   len_in = len_out;
   {
      l = 35;
      shared_ptr<CyclicVectorTransformer> pCVT = make_shared<CyclicVectorTransformer>(len_in, len_out);
      // REVIEW: Transform sanity check test.
      //pCVT->SetEvalPostActivationCallBack(MCB2);
      LayerList.push_back(make_shared<DAGCyclicTransformLayer>(l, pCVT, CMap[7], CMap[0])); // C5 holds rotation value.
   }
   //-----------------------------------------------------------------

//****************** End Transformer Layer ***********************************************     

// Join Fully Connected Layer -----------------------------------------
//
   l = 46;
   len_out += len_out_branch_1;
   LayerList.push_back(make_shared<DAGJoinObj>(l, CMap[6], CMap[7]));
//
//----------------------------------------------------------------------- 

//--------- setup the fully connected network -------------------------------------------------------------------------
// 
// Fully Connected Layer ---------------------------------------
// Type: ReLU
   len_in = len_out;
   len_out = 32;
   {
      l = 36;
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
         make_unique<actReLU>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[7]));
   }
   //l++;
   //---------------------------------------------------------------  
   // Branch 2 Fully Connected Layer -----------------------------------------
   //
   //int len_out_branch_2 = len_out;
   // This branch will be paired with Test2.
   //l = 39;
   //shared_ptr<DAGBranchObj> p_branch2 = make_shared<DAGBranchObj>(l, pContext8, pContext_y);
   //LayerList.push_back(p_branch2);
   //
   //-----------------------------------------------------------------------
   // Fully Connected Layer ---------------------------------------
   // Type: SoftMAX
   len_in = len_out;
   len_out = 10;
   {
      l = 37;
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actSoftMaxEx>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[7]));
   }
   //---------------------------------------------------------------      
   // Error Layer ---------------------------------------
   // Type: LossCrossEntropy
   {
      l = 38;
      shared_ptr<LossCrossEntropyEx> pl = make_shared<LossCrossEntropyEx>(len_out, 1);
      // DAGErrorLayer(shared_ptr<iLossLayer> _pLayer, shared_ptr<NetContext> _pContext, shared_ptr<ErrorContext> _pEContext) 
      LayerList.push_back(make_shared<DAGErrorLayer>(l, static_pointer_cast<iLossLayer>(pl), CMap[7], EMap[1]));
   }

}

void InitLPBranchModel3T(bool restore)
{
   INITIALIZE("LPB3\\LPB3T", optoADAM)
   gModelBranches = 3;


   bool lrestore = true;
   lrestore = restore;

   optoADAM::B1 = 0.9;
   //optoADAM::B1 = 0.7;
   optoADAM::B2 = 0.999;
   optoLowPass::Momentum = 0.8;

   LayerList.clear();

   int P1 = INPUT_ROWS;

   // NOTE: Establish the context that will store the composite pose estimate.
   //       It is fixed and does not change accross layers.
   CMap[0].v.resize(1);

   // *********************************************************************
   //            Branch Lambda function
   //**********************************************************************

   auto fbranch = [&restore, &lrestore, &P1](int n, bool last_branch = false, bool copy_start_map = true) {
      // NOTE: Used by Filter Pool.
      // REIVEW: !!!!!!!!!!!!!!!!  LOOK LOOK !!!!!!!!!!!!!!!!!!!!!!!
      //          Set to 4,4 to test downsampling.  It has been 2,4 .
      clSize size_kernel(2, 4);

      int l = 1; // Layer label
      int lo = n * 30;  // Layer offset
      int b = 2 + n*4;

      if (copy_start_map) {
         // Copy Start Context -----------------------------------------
         //
         l = 1 + lo;
         LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[b]));
         //
         //-----------------------------------------------------------------------
      }

      // Convolution Layer -----------------------------------------
      // Type: FilterLayer2D
      clSize size_in(INPUT_ROWS, INPUT_COLS);
      clSize size_out(INPUT_ROWS, 4);
      clSize size_kern(INPUT_ROWS, INPUT_COLS);
      int chn_in = 1;
      int chn_out = 1;
      {
         l = 2 + lo;
         shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
            make_unique<actReLU>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, chn_in)),
            make_shared<OPTO>(),
            true); // No bias. true/false
         pl->SetConvolutionParameters(FilterLayer2D::FFT, true, false);
         //pl->SetBackpropCallBack(MCB);

         LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[b]));
      }
      //---------------------------------------------------------------
      // Pooling Layer ----------------------------------------------
      // Type: poolAvg2D
      size_in = size_out;
      size_out.Resize(P1, 1);
      chn_in = chn_out;

      assert(!(size_in.rows % size_out.rows));
      assert(!(size_in.cols % size_out.cols));
      {
         l = 3 + lo;
#ifdef USE_FILTER_FOR_POOL
         shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
         //pl->SetEvalPostActivationCallBack(MVCB);
#else
         shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
#endif
         LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, CMap[b]));
      }
      //---------------------------------------------------------------
      // Flattening Layer --------------------------------------------
      // Type: Flatten2D
      size_in = size_out;
      chn_in = chn_out;

      int len_in = 0;
      int len_out = size_in.rows * size_in.cols * chn_in;
      chn_out = 1;
      {
         l = 4 + lo;
         shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
         LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, CMap[b]));
      }
      //-----------------------------------------------------------------
      //           Transformer Layers
      //---------------------------------------------------------------      
      // 

      // Copy Context2 to Spectrum Context -----------------------------------------
      //
      l = 9 + lo;
      LayerList.push_back(make_shared<DAGContextCopyObj>(l, CMap[b], CMap[b + 1]));
      //
      //-----------------------------------------------------------------------

      // Spectrum Layer ---------------------------------------
      //
      int len_in_branch_t = len_out;
      //int len_out_branch_t = 3 * len_out / 8;
      int len_out_branch_t = len_out / 2;
      {
         l = 10 + lo;
         shared_ptr<SpectrumOutputLayer> pl = make_shared<SpectrumOutputLayer>(len_in_branch_t, len_out_branch_t);
         LayerList.push_back(make_shared<DAGLayerObj>(l, pl, CMap[b + 1]));
      }
      //---------------------------------------------------------------    
      // Branch to Transform Layer -----------------------------------------
      //
      {
         l = 11 + lo;
         shared_ptr<DAGBranchObj> p_branch_t = make_shared<DAGBranchObj>(l, CMap[b], CMap[b + 2], false); // false -> no backprop.
         LayerList.push_back(p_branch_t);
      }
      //
      //-----------------------------------------------------------------------

      // Join Fully Connected Layer -----------------------------------------
      //
      l = 12 + lo;
      len_out_branch_t += len_out;  // out len of fully connected layer prior to Transformer layer.
      LayerList.push_back(make_shared<DAGJoinObj>(l, CMap[b + 2], CMap[b + 1]));
      //
      //-----------------------------------------------------------------------  

      // Fully Connected Layer ---------------------------------------
   // Type: ReLU
      len_in_branch_t = len_out_branch_t;
      len_out_branch_t = 60;
      {
         l = 13 + lo;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in_branch_t, len_out_branch_t,
            make_unique<actReLU>(),
            lrestore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         //pl->SetEvalPreActivationCallBack(MCB2);
         LayerList.push_back(make_shared<DAGLayerObj>(l, pl, CMap[b + 1]));
      }
      //---------------------------------------------------------------  
      // Fully Connected Layer ---------------------------------------
      // Type: ReLU
      len_in_branch_t = len_out_branch_t;
      len_out_branch_t = 30;
      {
         l = 14 + lo;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in_branch_t, len_out_branch_t,
            make_unique<actReLU>(),
            lrestore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, pl, CMap[b + 1]));
      }
      //---------------------------------------------------------------  


      // Fully Connected Layer ---------------------------------------
      // Type: Linear
      len_in_branch_t = len_out_branch_t;
      len_out_branch_t = 1;
      {
         l = 15 + lo;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in_branch_t, len_out_branch_t,
            //make_unique<actLinear>(), 
            make_unique<actLinearEx>(),
            lrestore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());

         //pl->SetBackpropCallBack(MCB1);

         LayerList.push_back(make_shared<DAGLayerObj>(l, pl, CMap[b + 1]));
      }
      // -----------------------------------------------------------------

      // ------ Vector Composite -----------------------------------------
      //
      // NOTE: C0 carries average transform parameter.
      // Numbers are skipped to get here.
      l = 21 + lo;
      LayerList.push_back(make_shared<DAGCompAvg>(l, CMap[b + 1], CMap[0]));
      //
      //-----------------------------------------------------------------------

      //---------------------------------------------------------------  
      // Cyclic Transformer
      //
      // REVIEW: Note that the rotation value has been updated by this layer's estimate of it.
      //         Be aware that each layer is potentially rotated differently.
      len_in = len_out;
      {
         l = 5 + lo;
         shared_ptr<CyclicVectorTransformer> pCVT = make_shared<CyclicVectorTransformer>(len_in, len_out);
         // REVIEW: Transform sanity check test.
         //pCVT->SetEvalPostActivationCallBack(MCB2);
         LayerList.push_back(make_shared<DAGCyclicTransformLayer>(l, pCVT, CMap[b], CMap[0])); // C0 holds rotation value.
      }
      //-----------------------------------------------------------------

      //****************** End Transformer Layer ***********************************************     
      // Join Fully Connected Layer -----------------------------------------
      //
      if (n > 0) {
         l = 22 + lo;
         LayerList.push_back(make_shared<DAGJoinObj>(l, CMap[b - 1], CMap[b]));
      }
      //
      //----------------------------------------------------------------------- 
      //--------- setup the fully connected network -------------------------------------------------------------------------
      // 
      shared_ptr<DAGBranchObj> p_branch1;
      if (!last_branch) {
         // Out to Branch 2 Fully Connected Layer -----------------------------------------
         //
         // This branch will be paired with Test1.
         l = 23 + lo;
         // REVIEW: Currently no backprop from other branches .........   v 
         //p_branch1 = make_shared<DAGBranchObj>(l, CMap[b], CMap[b + 3], false);
         p_branch1 = make_shared<DAGBranchObj>(l, CMap[b], CMap[b + 3], true);
         LayerList.push_back(p_branch1);
         //
         //-----------------------------------------------------------------------
      }
      // Fully Connected Layer ---------------------------------------
      // Type: ReLU
      len_in = (n + 1) * P1;
      len_out = len_in >> 1;
      {
         l = 6 + lo;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
            make_unique<actReLU>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[b]));
      }
      //---------------------------------------------------------------  

      // Fully Connected Layer ---------------------------------------
      // Type: SoftMAX
      len_in = len_out;
      len_out = 10;
      {
         l = 7 + lo;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actSoftMaxEx>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[b]));
      }
      //---------------------------------------------------------------      
      // Error Layer ---------------------------------------
      // Type: LossCrossEntropy
      {
         l = 8 + lo;
         shared_ptr<LossCrossEntropyEx> pl = make_shared<LossCrossEntropyEx>(len_out, 1);
         // DAGErrorLayer(shared_ptr<iLossLayer> _pLayer, shared_ptr<NetContext> _pContext, shared_ptr<ErrorContext> _pEContext) 
         LayerList.push_back(make_shared<DAGErrorLayer>(l, static_pointer_cast<iLossLayer>(pl), CMap[b], EMap[n]));
      }

      //---------------------------------------------------------------      
      if (!last_branch) {
         // Branch Test Layer (Test 1) -----------------------------------------
         // upper limit: read --> if your that accurate don't bother backpropagating, not much to learn.
         // lower limit: read --> if your that screwed up I don't want to try to learn what you are.
         //                                                                              warm up             backprop lower limit   backprop upper limit
         l = 24 + lo;
         //LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, EMap[e], restore ? 0 : 6 * 1100, 0.7, 0.98));
         LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, EMap[n], restore ? 0 : 20 * 1100, 0.0, 0.0));
         //LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, EMap[n], 20 * 1100, 0.0, 0.0));
         //
         //-----------------------------------------------------------------------
      }
   };


   //***************** LP Transformer *******************************
   // 
   // Copy Start Context -----------------------------------------
   //
   int l = 200;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[200]));
   //
   //-----------------------------------------------------------------------
   // Copy Start Context -----------------------------------------
   //
   l = 201;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[2]));
   //
   //-----------------------------------------------------------------------

   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   clSize size_ct_in(28, 28);
   clSize size_ct_out(28, 28);
   clSize size_ct_kern(28, 28);
   {
      l = 202;
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_ct_in, 1, size_ct_out, size_ct_kern, 1,
         make_unique<actReLU>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>(),
         true); // No bias. true/false
      pl->SetConvolutionParameters(FilterLayer2D::FFT, false, false);

      //pl->SetBackpropCallBack(MCB);

      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[200]));
   }
   //---------------------------------------------------------------

   // Centroid Layer --------------------------------------------
   // Type: Centroid
   // Note: Uses DAGFlattenObj.
   size_ct_in = size_ct_out;
   {
      l = 203;
      shared_ptr<Centroid> pl = make_shared<Centroid>(size_ct_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, CMap[200]));
   }
   //-----------------------------------------------------------------

   //---------------------------------------------------------------  
   // LP Spacial Transformer
   //
   {
      l = 204;
      shared_ptr<samplerBiLinear> psmp = make_shared<samplerBiLinear>(Size(28, 28), Size(INPUT_ROWS, INPUT_COLS));
      psmp->SetGridBackpropCallBack(MCB1);

      shared_ptr<gridLogPolar> pgrd = make_shared<gridLogPolar>(Size(28, 28), Size(INPUT_ROWS, INPUT_COLS), LP_WASTE);
      // REVIEW: Transform sanity check test.
      //pCVT->SetEvalPostActivationCallBack(MCB2);
      LayerList.push_back(make_shared<DAGSpacialTransformLayer>(l, psmp, pgrd, CMap[2], CMap[200], false)); // C200 holds center point value.
   }
   //-----------------------------------------------------------------

   //-----------------------------------------------------------------------
   // Update the Start Context to Log Polar.  This is what the branches start with. -----------------------------------------
   //
   l = 205;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[2], CMap[1]));
   //
   //-----------------------------------------------------------------------

   //********* End LP Transformer *********************************************

   //fbranch(0,true,false);
   fbranch(0,false,false);
   fbranch(1);
   //fbranch(2, true); // last branch = true
   fbranch(2); // last branch = true
   //fbranch(3, true); // last branch = true
   fbranch(3); // last branch = true
   fbranch(4, true); // last branch = true
}

void InitLPBranchModel4(bool restore)
{
   INITIALIZE("LPB4\\LPB4", optoADAM)
   gModelBranches = 5;

   // lrestore controls when warm-up is on.
   bool lrestore = restore;
   restore = true;


   optoADAM::B1 = 0.9;
   //optoADAM::B1 = 0.7;
   optoADAM::B2 = 0.999;
   optoLowPass::Momentum = 0.8;

   LayerList.clear();

   int P1 = INPUT_ROWS;

   // NOTE: Establish the context that will store the composite pose estimate.
   //       It is fixed and does not change accross layers.
   CMap[0].v.resize(1);

   // *********************************************************************
   //            Branch Lambda function
   //**********************************************************************

   auto fbranch = [&restore, &lrestore, &P1](int n, bool last_branch = false, bool copy_start_map = true) {
      // NOTE: Used by Filter Pool.
      // REIVEW: !!!!!!!!!!!!!!!!  LOOK LOOK !!!!!!!!!!!!!!!!!!!!!!!
      //          Set to 4,4 to test downsampling.  It has been 2,4 .
      clSize size_kernel(2, 4);

      int l = 1; // Layer label
      int lo = n * 30;  // Layer offset
      int b = 2 + n * 4;

      if (copy_start_map) {
         // Copy Start Context -----------------------------------------
         //
         l = 1 + lo;
         LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[b]));
         //
         //-----------------------------------------------------------------------
      }

      // Convolution Layer -----------------------------------------
      // Type: FilterLayer2D
      clSize size_in(INPUT_ROWS, INPUT_COLS);
      clSize size_out(INPUT_ROWS, 4);
      clSize size_kern(INPUT_ROWS, INPUT_COLS);
      int chn_in = 1;
      int chn_out = 1;
      {
         l = 2 + lo;
         shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
            make_unique<actReLU>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, chn_in)),
            make_shared<OPTO>(),
            true); // No bias. true/false
         pl->SetConvolutionParameters(FilterLayer2D::FFT, true, false);
         //pl->SetBackpropCallBack(MCB);

         LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[b]));
      }
      //---------------------------------------------------------------
      // Pooling Layer ----------------------------------------------
      // Type: poolAvg2D
      size_in = size_out;
      size_out.Resize(P1, 1);
      chn_in = chn_out;

      assert(!(size_in.rows % size_out.rows));
      assert(!(size_in.cols % size_out.cols));
      {
         l = 3 + lo;
#ifdef USE_FILTER_FOR_POOL
         shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
         //pl->SetEvalPostActivationCallBack(MVCB);
#else
         shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
#endif
         LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, CMap[b]));
      }
      //---------------------------------------------------------------
      // Flattening Layer --------------------------------------------
      // Type: Flatten2D
      size_in = size_out;
      chn_in = chn_out;

      int len_in = 0;
      int len_out = size_in.rows * size_in.cols * chn_in;
      chn_out = 1;
      {
         l = 4 + lo;
         shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
         LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, CMap[b]));
      }
 
      // Join Fully Connected Layer -----------------------------------------
      //
      if (n > 0) {
         l = 22 + lo;
         LayerList.push_back(make_shared<DAGJoinObj>(l, CMap[b - 1], CMap[b]));
      }
      //
      //----------------------------------------------------------------------- 
      //--------- setup the fully connected network -------------------------------------------------------------------------
      // 
      shared_ptr<DAGBranchObj> p_branch1;
      if (!last_branch) {
         // Out to Branch 2 Fully Connected Layer -----------------------------------------
         //
         // This branch will be paired with Test1.
         l = 23 + lo;
         // REVIEW: Currently no backprop from other branches .........   v 
         //p_branch1 = make_shared<DAGBranchObj>(l, CMap[b], CMap[b + 3], false);
         p_branch1 = make_shared<DAGBranchObj>(l, CMap[b], CMap[b + 3], true);
         LayerList.push_back(p_branch1);
         //
         //-----------------------------------------------------------------------
      }
      // Fully Connected Layer ---------------------------------------
      // Type: ReLU
      len_in = (n + 1) * P1;
      len_out = len_in >> 1;
      {
         l = 6 + lo;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
            make_unique<actReLU>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[b]));
      }
      //---------------------------------------------------------------  

      // Fully Connected Layer ---------------------------------------
      // Type: SoftMAX
      len_in = len_out;
      len_out = 10;
      {
         l = 7 + lo;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actSoftMaxEx>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[b]));
      }
      //---------------------------------------------------------------      
      // Error Layer ---------------------------------------
      // Type: LossCrossEntropy
      {
         l = 8 + lo;
         shared_ptr<LossCrossEntropyEx> pl = make_shared<LossCrossEntropyEx>(len_out, 1);
         // DAGErrorLayer(shared_ptr<iLossLayer> _pLayer, shared_ptr<NetContext> _pContext, shared_ptr<ErrorContext> _pEContext) 
         LayerList.push_back(make_shared<DAGErrorLayer>(l, static_pointer_cast<iLossLayer>(pl), CMap[b], EMap[n]));
      }

      //---------------------------------------------------------------      
      if (!last_branch) {
         // Branch Test Layer (Test 1) -----------------------------------------
         // upper limit: read --> if your that accurate don't bother backpropagating, not much to learn.
         // lower limit: read --> if your that screwed up I don't want to try to learn what you are.
         //                                                                              warm up             backprop lower limit   backprop upper limit
         l = 24 + lo;
         //LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, EMap[n], lrestore ? 0 : 20 * 1100, 0.7 - ((double)(n - 1)/10.0), 0.98));
         //cout << "0.7 -(n-1)/10 , 0.98 Exit test" << endl;
         //LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, EMap[n], lrestore ? 0 : 20 * 1100, 0.7, 0.98));
         //cout << "0.7 , 0.98 Exit test" << endl;
         LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, EMap[n], lrestore ? 0 : 20 * 1100, 0.0, 0.0));
         cout << "0, 0 Exit Test" << endl;
         //
         //-----------------------------------------------------------------------
      }
   };


   //***************** LP Transformer *******************************
   // 
   // Copy Start Context -----------------------------------------
   //
   int l = 200;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[200]));
   //
   //-----------------------------------------------------------------------
   // Copy Start Context -----------------------------------------
   //
   l = 201;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[2]));
   //
   //-----------------------------------------------------------------------

   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   clSize size_ct_in(28, 28);
   clSize size_ct_out(28, 28);
   clSize size_ct_kern(28, 28);
   {
      l = 202;
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_ct_in, 1, size_ct_out, size_ct_kern, 1,
         make_unique<actReLU>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>(),
         true); // No bias. true/false
      pl->SetConvolutionParameters(FilterLayer2D::FFT, false, false);

      //pl->SetBackpropCallBack(MCB);

      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[200]));
   }
   //---------------------------------------------------------------

   // Centroid Layer --------------------------------------------
   // Type: Centroid
   // Note: Uses DAGFlattenObj.
   size_ct_in = size_ct_out;
   {
      l = 203;
      shared_ptr<Centroid> pl = make_shared<Centroid>(size_ct_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, CMap[200]));
   }
   //-----------------------------------------------------------------

   //---------------------------------------------------------------  
   // LP Spacial Transformer
   //
   {
      l = 204;
      shared_ptr<samplerBiLinear> psmp = make_shared<samplerBiLinear>(Size(28, 28), Size(INPUT_ROWS, INPUT_COLS));
      psmp->SetGridBackpropCallBack(MCB1);

      shared_ptr<gridLogPolar> pgrd = make_shared<gridLogPolar>(Size(28, 28), Size(INPUT_ROWS, INPUT_COLS), LP_WASTE);
      // REVIEW: Transform sanity check test.
      //pCVT->SetEvalPostActivationCallBack(MCB2);
      LayerList.push_back(make_shared<DAGSpacialTransformLayer>(l, psmp, pgrd, CMap[2], CMap[200], false)); // C200 holds center point value.
   }
   //-----------------------------------------------------------------

   //-----------------------------------------------------------------------
   // Update the Start Context to Log Polar.  This is what the branches start with. -----------------------------------------
   //
   l = 205;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[2], CMap[1]));
   //
   //-----------------------------------------------------------------------

   //********* End LP Transformer *********************************************

   //fbranch(0,true,false);
   fbranch(0, false, false);
   fbranch(1);
   //fbranch(2, true); // last branch = true
   fbranch(2); // last branch = true
   //fbranch(3, true); // last branch = true
   fbranch(3); // last branch = true
   fbranch(4, true); // last branch = true
}

void InitLPBranchModel5(bool restore)
{
   INITIALIZE("LPB5\\LPB5", optoADAM)
      gModelBranches = 5;


   bool lrestore = restore;
   restore = true;

   optoADAM::B1 = 0.9;
   //optoADAM::B1 = 0.7;
   optoADAM::B2 = 0.999;
   optoLowPass::Momentum = 0.8;

   LayerList.clear();

   int P1 = INPUT_ROWS;

   // NOTE: Establish the context that will store the composite pose estimate.
   //       It is fixed and does not change accross layers.
   CMap[0].v.resize(1);

   // *********************************************************************
   //            Branch Lambda function
   //**********************************************************************

   auto fbranch = [&restore, &lrestore, &P1](int n, bool last_branch = false, bool copy_start_map = true) {
      // NOTE: Used by Filter Pool.
      // REIVEW: !!!!!!!!!!!!!!!!  LOOK LOOK !!!!!!!!!!!!!!!!!!!!!!!
      //          Set to 4,4 to test downsampling.  It has been 2,4 .
      clSize size_kernel(2, 4);

      int l = 1; // Layer label
      int lo = n * 30;  // Layer offset
      int b = 2 + n * 4;

      if (copy_start_map) {
         // Copy Start Context -----------------------------------------
         //
         l = 1 + lo;
         LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[b]));
         //
         //-----------------------------------------------------------------------
      }

      // Convolution Layer -----------------------------------------
      // Type: FilterLayer2D
      clSize size_in(INPUT_ROWS, INPUT_COLS);
      clSize size_out(INPUT_ROWS, 4);
      clSize size_kern(INPUT_ROWS, INPUT_COLS);
      int chn_in = 1;
      int chn_out = 1;
      {
         l = 2 + lo;
         shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
            make_unique<actReLU>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, chn_in)),
            make_shared<OPTO>(),
            true); // No bias. true/false
         pl->SetConvolutionParameters(FilterLayer2D::FFT, true, false);
         //pl->SetBackpropCallBack(MCB);

         LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[b]));
      }
      //---------------------------------------------------------------
      // Pooling Layer ----------------------------------------------
      // Type: poolAvg2D
      size_in = size_out;
      size_out.Resize(P1, 1);
      chn_in = chn_out;

      assert(!(size_in.rows % size_out.rows));
      assert(!(size_in.cols % size_out.cols));
      {
         l = 3 + lo;
#ifdef USE_FILTER_FOR_POOL
         shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
         //pl->SetEvalPostActivationCallBack(MVCB);
#else
         shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
#endif
         LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, CMap[b]));
      }
      //---------------------------------------------------------------
      // Flattening Layer --------------------------------------------
      // Type: Flatten2D
      size_in = size_out;
      chn_in = chn_out;

      int len_in = 0;
      int len_out = size_in.rows * size_in.cols * chn_in;
      chn_out = 1;
      {
         l = 4 + lo;
         shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
         LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, CMap[b]));
      }

      // Sum Fully Connected Layer -----------------------------------------
      //
      if (n > 0) {
         l = 22 + lo;
         LayerList.push_back(make_shared<DAGAddObj>(l, CMap[b - 1], CMap[b]));
      }
      //
      //----------------------------------------------------------------------- 
      //--------- setup the fully connected network -------------------------------------------------------------------------
      // 
      shared_ptr<DAGBranchObj> p_branch1;
      if (!last_branch) {
         // Out to Branch 2 Fully Connected Layer -----------------------------------------
         //
         // This branch will be paired with Test1.
         l = 23 + lo;
         // REVIEW: Currently no backprop from other branches .........   v 
         //p_branch1 = make_shared<DAGBranchObj>(l, CMap[b], CMap[b + 3], false);
         p_branch1 = make_shared<DAGBranchObj>(l, CMap[b], CMap[b + 3], true);
         LayerList.push_back(p_branch1);
         //
         //-----------------------------------------------------------------------
      }
      // Fully Connected Layer ---------------------------------------
      // Type: ReLU
      len_in = P1;
      if (n == 0) {
         len_out = len_in >> 1;
      }
      else {
         len_out = len_in;
      }

      {
         l = 6 + lo;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
            make_unique<actReLU>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[b]));
      }
      //---------------------------------------------------------------  

      // Fully Connected Layer ---------------------------------------
      // Type: SoftMAX
      len_in = len_out;
      len_out = 10;
      {
         l = 7 + lo;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actSoftMaxEx>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[b]));
      }
      //---------------------------------------------------------------      
      // Error Layer ---------------------------------------
      // Type: LossCrossEntropy
      {
         l = 8 + lo;
         shared_ptr<LossCrossEntropyEx> pl = make_shared<LossCrossEntropyEx>(len_out, 1);
         // DAGErrorLayer(shared_ptr<iLossLayer> _pLayer, shared_ptr<NetContext> _pContext, shared_ptr<ErrorContext> _pEContext) 
         LayerList.push_back(make_shared<DAGErrorLayer>(l, static_pointer_cast<iLossLayer>(pl), CMap[b], EMap[n]));
      }

      //---------------------------------------------------------------      
      if (!last_branch) {
         // Branch Test Layer (Test 1) -----------------------------------------
         // upper limit: read --> if your that accurate don't bother backpropagating, not much to learn.
         // lower limit: read --> if your that screwed up I don't want to try to learn what you are.
         //                                                                              warm up             backprop lower limit   backprop upper limit
         l = 24 + lo;
         //LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, EMap[n], lrestore ? 0 : 20 * 1100, 0.7 - ((double)(n - 1)/10.0), 0.98));
         //LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, EMap[n], lrestore ? 0 : 20 * 1100, 0.7, 0.98));
         //cout << "0.7, 0.98 Exit test" << endl;
         //LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, EMap[n], lrestore ? 0 : 20 * 1100, 0.0, 0.0));
         LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, EMap[n], 20 * 1100, 0.0, 0.0));
         cout << "0, 0 Exit test" << endl;
         //
         //-----------------------------------------------------------------------
      }
   };


   //***************** LP Transformer *******************************
   // 
   // Copy Start Context -----------------------------------------
   //
   int l = 200;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[200]));
   //
   //-----------------------------------------------------------------------
   // Copy Start Context -----------------------------------------
   //
   l = 201;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[2]));
   //
   //-----------------------------------------------------------------------

   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   clSize size_ct_in(28, 28);
   clSize size_ct_out(28, 28);
   clSize size_ct_kern(28, 28);
   {
      l = 202;
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_ct_in, 1, size_ct_out, size_ct_kern, 1,
         make_unique<actReLU>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>(),
         true); // No bias. true/false
      pl->SetConvolutionParameters(FilterLayer2D::FFT, false, false);

      //pl->SetBackpropCallBack(MCB);

      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[200]));
   }
   //---------------------------------------------------------------

   // Centroid Layer --------------------------------------------
   // Type: Centroid
   // Note: Uses DAGFlattenObj.
   size_ct_in = size_ct_out;
   {
      l = 203;
      shared_ptr<Centroid> pl = make_shared<Centroid>(size_ct_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, CMap[200]));
   }
   //-----------------------------------------------------------------

   //---------------------------------------------------------------  
   // LP Spacial Transformer
   //
   {
      l = 204;
      shared_ptr<samplerBiLinear> psmp = make_shared<samplerBiLinear>(Size(28, 28), Size(INPUT_ROWS, INPUT_COLS));
      psmp->SetGridBackpropCallBack(MCB1);

      shared_ptr<gridLogPolar> pgrd = make_shared<gridLogPolar>(Size(28, 28), Size(INPUT_ROWS, INPUT_COLS), LP_WASTE);
      // REVIEW: Transform sanity check test.
      //pCVT->SetEvalPostActivationCallBack(MCB2);
      LayerList.push_back(make_shared<DAGSpacialTransformLayer>(l, psmp, pgrd, CMap[2], CMap[200], false)); // C200 holds center point value.
   }
   //-----------------------------------------------------------------

   //-----------------------------------------------------------------------
   // Update the Start Context to Log Polar.  This is what the branches start with. -----------------------------------------
   //
   l = 205;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[2], CMap[1]));
   //
   //-----------------------------------------------------------------------

   //********* End LP Transformer *********************************************

   //fbranch(0,true,false);
   fbranch(0, false, false);
   fbranch(1);
   //fbranch(2, true); // last branch = true
   fbranch(2);
   fbranch(3);
   fbranch(4, true); // last branch = true
}

void InitLPBranchModel6(bool restore)
{
   INITIALIZE("LPB6\\LPB6", optoADAM)
   gModelBranches = 5;

   // lrestore controls when warm-up is on.
   bool lrestore = restore;
   restore = true;


   optoADAM::B1 = 0.9;
   //optoADAM::B1 = 0.7;
   optoADAM::B2 = 0.999;
   optoLowPass::Momentum = 0.8;

   LayerList.clear();

   int P1 = INPUT_ROWS;

   // NOTE: Establish the context that will store the composite pose estimate.
   //       It is fixed and does not change accross layers.
   CMap[0].v.resize(1);

   shared_ptr<unsigned int> pAvgGradCounter = make_shared<unsigned int>(0);

   // *********************************************************************
   //            Branch Lambda function
   //**********************************************************************

   auto fbranch = [&restore, &lrestore, &P1, &pAvgGradCounter](int n, bool last_branch = false) {
      // NOTE: Used by Filter Pool.
      // REIVEW: !!!!!!!!!!!!!!!!  LOOK LOOK !!!!!!!!!!!!!!!!!!!!!!!
      //          Set to 4,4 to test downsampling.  It has been 2,4 .
      clSize size_kernel(2, 4);

      int l = 1; // Layer label
      int lo = n * 30;  // Layer offset
      int b = 2 + n * 4;

      {
         // Copy Start Context -----------------------------------------
         //
         l = 1 + lo;
         LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[b]));
         //
         //-----------------------------------------------------------------------
      }

      {
         // Compute average filter gradient  -----------------------------------------
         // NOTE: This does not work with backprop early stop.
         //
         l = 5 + lo;
         LayerList.push_back(make_shared<DAGAvgFilterGrad>(l, CMap[b], CMap[201], pAvgGradCounter ));
         //
         //-----------------------------------------------------------------------
      }

      
      // Convolution Layer -----------------------------------------
      // Type: FilterLayer2D
      clSize size_in(INPUT_ROWS, INPUT_COLS);
      clSize size_out(INPUT_ROWS, 4);
      clSize size_kern(INPUT_ROWS, INPUT_COLS);
      int chn_in = 1;
      int chn_out = 1;
      {
         l = 2 + lo;
         shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
            make_unique<actReLU>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, chn_in)),
            make_shared<OPTO>(),
            true); // No bias. true/false
         pl->SetConvolutionParameters(FilterLayer2D::FFT, true, false);
         //pl->SetBackpropCallBack(MCB);

         LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[b]));
      }
      //---------------------------------------------------------------
      // Pooling Layer ----------------------------------------------
      // Type: poolAvg2D
      size_in = size_out;
      size_out.Resize(P1, 1);
      chn_in = chn_out;

      assert(!(size_in.rows % size_out.rows));
      assert(!(size_in.cols % size_out.cols));
      {
         l = 3 + lo;
#ifdef USE_FILTER_FOR_POOL
         shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
         //pl->SetEvalPostActivationCallBack(MVCB);
#else
         shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
#endif
         LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, CMap[b]));
      }
      //---------------------------------------------------------------
      // Flattening Layer --------------------------------------------
      // Type: Flatten2D
      size_in = size_out;
      chn_in = chn_out;

      int len_in = 0;
      int len_out = size_in.rows * size_in.cols * chn_in;
      chn_out = 1;
      {
         l = 4 + lo;
         shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
         LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, CMap[b]));
      }

      // Join Fully Connected Layer -----------------------------------------
      //
      if (n > 0) {
         l = 22 + lo;
         LayerList.push_back(make_shared<DAGJoinObj>(l, CMap[b - 1], CMap[b]));
      }
      //
      //----------------------------------------------------------------------- 
      //--------- setup the fully connected network -------------------------------------------------------------------------
      // 
      shared_ptr<DAGBranchObj> p_branch1;
      if (!last_branch) {
         // Out to Branch 2 Fully Connected Layer -----------------------------------------
         //
         // This branch will be paired with Test1.
         l = 23 + lo;
         // REVIEW: Currently no backprop from other branches .........   v 
         //p_branch1 = make_shared<DAGBranchObj>(l, CMap[b], CMap[b + 3], false);
         // Branch backprop percent ------------------------------------------ v
         p_branch1 = make_shared<DAGBranchObj>(l, CMap[b], CMap[b + 3], true, 0.1);
         LayerList.push_back(p_branch1);
         //
         //-----------------------------------------------------------------------
      }
      // Fully Connected Layer ---------------------------------------
      // Type: ReLU
      len_in = (n + 1) * P1;
      len_out = len_in >> 1;
      {
         l = 6 + lo;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
            make_unique<actReLU>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[b]));
      }
      //---------------------------------------------------------------  
      /*
      if (n > 0) {
         len_in = len_out;
         len_out = len_out >> 1;

         l = 7 + lo;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
            make_unique<actReLU>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[b]));
      }

      if (n > 1) {
         len_in = len_out;
         len_out = len_out >> 1;

         l = 8 + lo;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
            make_unique<actReLU>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[b]));
      }

      if (n > 2) {
         len_in = len_out;
         len_out = len_out >> 1;

         l = 9 + lo;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
            make_unique<actReLU>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[b]));
      }

      if (n > 3) {
         len_in = len_out;
         len_out = len_out >> 1;

         l = 10 + lo;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
            make_unique<actReLU>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[b]));
      }
      */
      // Fully Connected Layer ---------------------------------------
      // Type: SoftMAX
      len_in = len_out;
      len_out = 10;
      {
         l = 15 + lo;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actSoftMaxEx>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[b]));
      }
      //---------------------------------------------------------------      
      // Error Layer ---------------------------------------
      // Type: LossCrossEntropy
      {
         l = 16 + lo;
         shared_ptr<LossCrossEntropyEx> pl = make_shared<LossCrossEntropyEx>(len_out, 1);
         // DAGErrorLayer(shared_ptr<iLossLayer> _pLayer, shared_ptr<NetContext> _pContext, shared_ptr<ErrorContext> _pEContext) 
         LayerList.push_back(make_shared<DAGErrorLayer>(l, static_pointer_cast<iLossLayer>(pl), CMap[b], EMap[n]));
      }

      //---------------------------------------------------------------      
      if (!last_branch) {
         
         // Branch Test Layer (Test 1) -----------------------------------------
         // upper limit: read --> if your that accurate don't bother backpropagating, not much to learn.
         // lower limit: read --> if your that screwed up I don't want to try to learn what you are.
         //                                                                              warm up             backprop lower limit   backprop upper limit
         l = 24 + lo;
         //LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, EMap[n], lrestore ? 0 : 20 * 1100, 0.7 - ((double)(n - 1)/10.0), 0.98));
         //cout << "0.7 -(n-1)/10 , 0.98 Exit test" << endl;
         //LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, EMap[n], lrestore ? 0 : 20 * 1100, 0.7, 0.98));
         //cout << "0.7 , 0.98 Exit test" << endl;
         //                                                                     Sets branch threshold -------------------- V
         LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, EMap[n], lrestore ? 0 : 20 * 1100, 0.0, 0.0, branch_pass_threshold));
         cout << "0.0, 0, Exit Test" << endl;
         //
         //-----------------------------------------------------------------------

      }

      if( n < gTrainingBranch) {
         cout << "Stop learning at layer: " << n << endl;
         l = 26 + lo;
         LayerList.push_back(make_shared<DAGLambdaLayer>(l, []() {}, []() { gpExit->stop = true;  }));
      }

   };


   //***************** LP Transformer *******************************
   // 
   // Copy Start Context -----------------------------------------
   //
   int l = 200;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[200]));
   //
   //-----------------------------------------------------------------------
   // Copy Start Context -----------------------------------------
   //
   l = 201;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[201]));
   //
   //-----------------------------------------------------------------------

   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   clSize size_ct_in(28, 28);
   clSize size_ct_out(28, 28);
   clSize size_ct_kern(28, 28);
   {
      l = 202;
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_ct_in, 1, size_ct_out, size_ct_kern, 1,
         make_unique<actReLU>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>(),
         true); // No bias. true/false
      pl->SetConvolutionParameters(FilterLayer2D::FFT, false, false);

      //pl->SetBackpropCallBack(MCB);

      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[200]));
   }
   //---------------------------------------------------------------

   // Centroid Layer --------------------------------------------
   // Type: Centroid
   // Note: Uses DAGFlattenObj.
   size_ct_in = size_ct_out;
   {
      l = 203;
      shared_ptr<Centroid> pl = make_shared<Centroid>(size_ct_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, CMap[200]));
   }
   //-----------------------------------------------------------------

   //---------------------------------------------------------------  
   // LP Spacial Transformer
   //
   {
      l = 204;
      shared_ptr<samplerBiLinear> psmp = make_shared<samplerBiLinear>(Size(28, 28), Size(INPUT_ROWS, INPUT_COLS));
      psmp->SetGridBackpropCallBack(MCB1);

      shared_ptr<gridLogPolar> pgrd = make_shared<gridLogPolar>(Size(28, 28), Size(INPUT_ROWS, INPUT_COLS), LP_WASTE);
      // REVIEW: Transform sanity check test.
      //pCVT->SetEvalPostActivationCallBack(MCB2);
      LayerList.push_back(make_shared<DAGSpacialTransformLayer>(l, psmp, pgrd, CMap[201], CMap[200], false)); // C200 holds center point value.
   }
   //-----------------------------------------------------------------

   //-----------------------------------------------------------------------
   // Update the Start Context to Log Polar.  This is what the branches start with. -----------------------------------------
   //
   l = 205;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[201], CMap[1]));
   //
   //-----------------------------------------------------------------------

   // !!!!!!!!!!!!----->  LOOK <------ !!!!!!!!!!!!
   cout << "Stop learning at LP Transform layer. " << endl;
   l = 206;
   LayerList.push_back(make_shared<DAGLambdaLayer>(l, []() {}, []() { gpExit->stop = true;  }));

   //********* End LP Transformer *********************************************

   for (int n = 0; n <= gTrainingBranch; n++) {
      fbranch(n, gTrainingBranch == n ? true : false);
   }
}

// This model should be used with Train4.
// This model is trying to train one branch at a time.  If the structure of the network
// isn't changed the results should be compatible with Model 6.
// The code in this model utilizes global parameters at the top of the file.
//
void InitLPBranchModel6A(bool restore)
{
   INITIALIZE("LPB6\\LPB6", optoADAM)
   gModelBranches = 5;

   // It only makes sence to initialize the parameters when gTrainingBranch is set to 4.
   //runtime_assert( restore || (!restore && gTrainingBranch==4) )
   //runtime_assert(gTrainingBranch>=0 && gTrainingBranch<=4)

   // lrestore controls when warm-up is on.
   bool lrestore = gWarmupOn > 0 ? false : true;

   optoADAM::B1 = 0.9;
   optoADAM::B2 = 0.999;
   //optoADAM::B1 = 0.8;
   //optoADAM::B2 = 0.9;
   optoLowPass::Momentum = 0.8;

   LayerList.clear();

   int P1 = INPUT_ROWS;

   // NOTE: Establish the context that will store the composite pose estimate.
   //       It is fixed and does not change accross layers.
   CMap[0].v.resize(1);

   shared_ptr<unsigned int> pAvgGradCounter = make_shared<unsigned int>(0);

   // *********************************************************************
   //            Branch Lambda function
   //**********************************************************************

   auto fbranch = [&P1, &pAvgGradCounter](int n, bool restore, bool lrestore, bool last_branch = false ) {
      // NOTE: Used by Filter Pool.
      // REIVEW: !!!!!!!!!!!!!!!!  LOOK LOOK !!!!!!!!!!!!!!!!!!!!!!!
      //          Set to 4,4 to test downsampling.  It has been 2,4 .
      clSize size_kernel(2, 4);

      int l = 1; // Layer label
      int lo = n * 30;  // Layer offset
      int b = 2 + n * 4;

      // Copy Start Context -----------------------------------------
      //
      l = 1 + lo;
      LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[b]));
      //
      //-----------------------------------------------------------------------

      // Compute average filter gradient  -----------------------------------------
      // NOTE: This does not work with backprop early stop.
      //
      l = 5 + lo;
      LayerList.push_back(make_shared<DAGAvgFilterGrad>(l, CMap[b], CMap[201], pAvgGradCounter));
      //
      //-----------------------------------------------------------------------

      // Convolution Layer -----------------------------------------
      // Type: FilterLayer2D
      clSize size_in(INPUT_ROWS, INPUT_COLS);
      clSize size_out(INPUT_ROWS, 4);
      clSize size_kern(INPUT_ROWS, INPUT_COLS);
      int chn_in = 1;
      int chn_out = 1;
      {
         l = 2 + lo;
         shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
            make_unique<actReLU>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, chn_in)),
            make_shared<OPTO>(),
            true); // No bias. true/false
         pl->SetConvolutionParameters(FilterLayer2D::FFT, true, false);
         //pl->SetBackpropCallBack(MCB);

         LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[b]));
      }
      //---------------------------------------------------------------
      // Pooling Layer ----------------------------------------------
      // Type: poolAvg2D
      size_in = size_out;
      size_out.Resize(P1, 1);
      chn_in = chn_out;

      assert(!(size_in.rows % size_out.rows));
      assert(!(size_in.cols % size_out.cols));
      {
         l = 3 + lo;
#ifdef USE_FILTER_FOR_POOL
         shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
         //pl->SetEvalPostActivationCallBack(MVCB);
#else
         shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
#endif
         LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, CMap[b]));
      }
      //---------------------------------------------------------------
      // Flattening Layer --------------------------------------------
      // Type: Flatten2D
      size_in = size_out;
      chn_in = chn_out;

      int len_in = 0;
      int len_out = size_in.rows * size_in.cols * chn_in;
      chn_out = 1;
      {
         l = 4 + lo;
         shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
         LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, CMap[b]));
      }

      // Join Fully Connected Layer -----------------------------------------
      //
      if (n > 0) {
         l = 22 + lo;
         LayerList.push_back(make_shared<DAGJoinObj>(l, CMap[b - 1], CMap[b]));
      }
      //
      //----------------------------------------------------------------------- 
      //--------- setup the fully connected network -------------------------------------------------------------------------
      // 
      shared_ptr<DAGBranchObj> p_branch1;
      if (!last_branch) {
         // Out to Branch 2 Fully Connected Layer -----------------------------------------
         //
         // This branch will be paired with Test1.
         l = 23 + lo;
         // REVIEW: Currently no backprop from other branches .........   v 
         //p_branch1 = make_shared<DAGBranchObj>(l, CMap[b], CMap[b + 3], false);
         // Branch backprop percent ------------------------------------------ v
         p_branch1 = make_shared<DAGBranchObj>(l, CMap[b], CMap[b + 3], true, 0.1);
         LayerList.push_back(p_branch1);
         //
         //-----------------------------------------------------------------------
      }
      // Fully Connected Layer ---------------------------------------
      // Type: ReLU
      len_in = (n + 1) * P1;
      len_out = len_in >> 1;
      {
         l = 6 + lo;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
            make_unique<actReLU>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[b]));
      }
      //---------------------------------------------------------------  
      /*
      if (n > 2 || n > 3) {
         len_in = len_out;
         len_out = len_out >> 1;

         l = 7 + lo;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
            make_unique<actReLU>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[b]));
      }
      */
      /*
      if (n > 1) {
         len_in = len_out;
         len_out = len_out >> 1;

         l = 8 + lo;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
            make_unique<actReLU>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[b]));
      }

      if (n > 2) {
         len_in = len_out;
         len_out = len_out >> 1;

         l = 9 + lo;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
            make_unique<actReLU>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[b]));
      }

      if (n > 3) {
         len_in = len_out;
         len_out = len_out >> 1;

         l = 10 + lo;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
            make_unique<actReLU>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[b]));
      }
      */
      // Fully Connected Layer ---------------------------------------
      // Type: SoftMAX
      len_in = len_out;
      len_out = 10;
      {
         l = 15 + lo;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actSoftMaxEx>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[b]));
      }
      //---------------------------------------------------------------      
      // Error Layer ---------------------------------------
      // Type: LossCrossEntropy
      {
         l = 16 + lo;
         shared_ptr<LossCrossEntropyEx> pl = make_shared<LossCrossEntropyEx>(len_out, 1);
         // DAGErrorLayer(shared_ptr<iLossLayer> _pLayer, shared_ptr<NetContext> _pContext, shared_ptr<ErrorContext> _pEContext) 
         LayerList.push_back(make_shared<DAGErrorLayer>(l, static_pointer_cast<iLossLayer>(pl), CMap[b], EMap[n]));
      }

      //---------------------------------------------------------------      
      if (!last_branch) {
         //if (n < gTrainingBranch) {
            cout << "Stop learning at layer: " << n << endl;
            l = 25 + lo;
            LayerList.push_back(make_shared<DAGLambdaLayer>(l, []() {}, []() { gpExit->stop = true;  }));
         //}
         // Branch Test Layer (Test 1) -----------------------------------------
         // upper limit: read --> if your that accurate don't bother backpropagating, not much to learn.
         // lower limit: read --> if your that screwed up I don't want to try to learn what you are.
         //                                                                              warm up             backprop lower limit   backprop upper limit
         l = 24 + lo;
         //LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, EMap[n], lrestore ? 0 : 20 * 1100, 0.7 - ((double)(n - 1)/10.0), 0.98));
         //cout << "0.7 -(n-1)/10 , 0.98 Exit test" << endl;
         //LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, EMap[n], lrestore ? 0 : 20 * 1100, 0.7, 0.98));
         //cout << "0.7 , 0.98 Exit test" << endl;
         //                                                                     Sets branch threshold -------------------- V
         LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, EMap[n], lrestore ? 0 : 20 * 1100, 0.0, 0.0, branch_pass_threshold));
         cout << "0.0, 0, Exit Test" << endl;
         //
         //-----------------------------------------------------------------------

      }
   };


   //***************** LP Transformer *******************************
   // 
   // Copy Start Context -----------------------------------------
   //
   int l = 200;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[200]));
   //
   //-----------------------------------------------------------------------
   // Copy Start Context -----------------------------------------
   //
   l = 201;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[201]));
   //
   //-----------------------------------------------------------------------

   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   clSize size_ct_in(28, 28);
   clSize size_ct_out(28, 28);
   clSize size_ct_kern(28, 28);
   {
      l = 202;
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_ct_in, 1, size_ct_out, size_ct_kern, 1,
         make_unique<actReLU>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>(),
         true); // No bias. true/false
      pl->SetConvolutionParameters(FilterLayer2D::FFT, false, false);

      //pl->SetBackpropCallBack(MCB);

      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[200]));
   }
   //---------------------------------------------------------------

   // Centroid Layer --------------------------------------------
   // Type: Centroid
   // Note: Uses DAGFlattenObj.
   size_ct_in = size_ct_out;
   {
      l = 203;
      shared_ptr<Centroid> pl = make_shared<Centroid>(size_ct_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, CMap[200]));
   }
   //-----------------------------------------------------------------

   //---------------------------------------------------------------  
   // LP Spacial Transformer
   //
   {
      l = 204;
      shared_ptr<samplerBiLinear> psmp = make_shared<samplerBiLinear>(Size(28, 28), Size(INPUT_ROWS, INPUT_COLS));
      psmp->SetGridBackpropCallBack(MCB1);

      shared_ptr<gridLogPolar> pgrd = make_shared<gridLogPolar>(Size(28, 28), Size(INPUT_ROWS, INPUT_COLS), LP_WASTE);
      // REVIEW: Transform sanity check test.
      //pCVT->SetEvalPostActivationCallBack(MCB2);
      LayerList.push_back(make_shared<DAGSpacialTransformLayer>(l, psmp, pgrd, CMap[201], CMap[200], false)); // C200 holds center point value.
   }
   //-----------------------------------------------------------------

   //-----------------------------------------------------------------------
   // Update the Start Context to Log Polar.  This is what the branches start with. -----------------------------------------
   //
   l = 205;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[201], CMap[1]));
   //
   //-----------------------------------------------------------------------

   // !!!!!!!!!!!!----->  LOOK <------ !!!!!!!!!!!!
   cout << "Stop learning at LP Transform layer. " << endl;
   l = 206;
   LayerList.push_back(make_shared<DAGLambdaLayer>(l, []() {}, []() { gpExit->stop = true;  }));

   //********* End LP Transformer *********************************************

   for (int n = 0; n <= gTrainingBranch; n++) {
      fbranch(n, restore, lrestore, gTrainingBranch == n ? true : false);
   }
}

// This model should be used with Train4.
// This model is trying to train one branch at a time.  If the structure of the network
// isn't changed the results should be compatible with Model 6.
// The code in this model utilizes global parameters at the top of the file.
//
void InitLPBranchModel7A(bool restore)
{
   INITIALIZE("LPB7\\LPB7", optoADAM)
   gModelBranches = 5;

   // It only makes sence to initialize the parameters when gTrainingBranch is set to 4.
   //runtime_assert( restore || (!restore && gTrainingBranch==4) )
   //runtime_assert(gTrainingBranch>=0 && gTrainingBranch<=4)

   // lrestore controls when warm-up is on.
   bool lrestore = gWarmupOn > 0 ? false : true;

   optoADAM::B1 = 0.9;
   optoADAM::B2 = 0.999;
   //optoADAM::B1 = 0.8;
   //optoADAM::B2 = 0.9;
   optoLowPass::Momentum = 0.8;

   LayerList.clear();

   int P1 = INPUT_ROWS;

   // NOTE: Establish the context that will store the composite pose estimate.
   //       It is fixed and does not change accross layers.
   CMap[0].v.resize(1);

   shared_ptr<unsigned int> pAvgGradCounter = make_shared<unsigned int>(0);

   // *********************************************************************
   //            Branch Lambda function
   //**********************************************************************

   auto fbranch = [&P1, &pAvgGradCounter](int n, bool restore, bool lrestore, bool last_branch = false) {
      // NOTE: Used by Filter Pool.
      // REIVEW: !!!!!!!!!!!!!!!!  LOOK LOOK !!!!!!!!!!!!!!!!!!!!!!!
      //          Set to 4,4 to test downsampling.  It has been 2,4 .
      clSize size_kernel(2, 4);

      int l = 1; // Layer label
      int lo = n * 30;  // Layer offset
      int b = 2 + n * 4;

      // Copy Start Context -----------------------------------------
      //
      l = 1 + lo;
      LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[b]));
      //
      //-----------------------------------------------------------------------

      // Compute average filter gradient  -----------------------------------------
      // NOTE: This does not work with backprop early stop.
      //
      l = 5 + lo;
      LayerList.push_back(make_shared<DAGAvgFilterGrad>(l, CMap[b], CMap[201], pAvgGradCounter));
      //
      //-----------------------------------------------------------------------

      // Convolution Layer -----------------------------------------
      // Type: FilterLayer2D
      clSize size_in(INPUT_ROWS, INPUT_COLS);
      clSize size_out(INPUT_ROWS, 4);
      clSize size_kern(INPUT_ROWS, INPUT_COLS);
      int chn_in = 1;
      int chn_out = 1;
      {
         l = 2 + lo;
         shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
            make_unique<actReLU>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, chn_in)),
            make_shared<OPTO>(),
            true); // No bias. true/false
         pl->SetConvolutionParameters(FilterLayer2D::FFT, true, false);
         //pl->SetBackpropCallBack(MCB);

         LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[b]));
      }
      //---------------------------------------------------------------
      // Pooling Layer ----------------------------------------------
      // Type: poolAvg2D
      size_in = size_out;
      size_out.Resize(P1, 1);
      chn_in = chn_out;

      assert(!(size_in.rows % size_out.rows));
      assert(!(size_in.cols % size_out.cols));
      {
         l = 3 + lo;
#ifdef USE_FILTER_FOR_POOL
         shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
         //pl->SetEvalPostActivationCallBack(MVCB);
#else
         shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
#endif
         LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, CMap[b]));
      }
      //---------------------------------------------------------------
      // Flattening Layer --------------------------------------------
      // Type: Flatten2D
      size_in = size_out;
      chn_in = chn_out;

      int len_in = 0;
      int len_out = size_in.rows * size_in.cols * chn_in;
      chn_out = 1;
      {
         l = 4 + lo;
         shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
         LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, CMap[b]));
      }

      // Join Fully Connected Layer -----------------------------------------
      //
      if (n > 0) {
         l = 22 + lo;
         LayerList.push_back(make_shared<DAGJoinObj>(l, CMap[b - 1], CMap[b]));
      }
      //
      //----------------------------------------------------------------------- 

      // Fully Connected Layer ---------------------------------------
      // Type: ReLU
      len_in = P1;
      if (n > 0) {
         len_in += 10;
      }
      len_out = len_in >> 1;
      {
         l = 6 + lo;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
            make_unique<actReLU>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[b]));
      }

      // Fully Connected Layer ---------------------------------------
      // Type: SoftMAX
      len_in = len_out;
      len_out = 10;
      {
         l = 15 + lo;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actSoftMaxEx>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[b]));
      }
      //---------------------------------------------------------------      
      //--------- Branch Layer -------------------------------------------------------------------------
      // 
      shared_ptr<DAGBranchObj> p_branch1;
      if (!last_branch) {
         // Out to Branch 2 Fully Connected Layer -----------------------------------------
         //
         // This branch will be paired with Test1.
         l = 23 + lo;
         // REVIEW: Currently no backprop from other branches .........   v 
         //p_branch1 = make_shared<DAGBranchObj>(l, CMap[b], CMap[b + 3], false);
         // Branch backprop percent ------------------------------------------ v
         p_branch1 = make_shared<DAGBranchObj>(l, CMap[b], CMap[b + 3], true, 0.1);
         LayerList.push_back(p_branch1);
         //
         //-----------------------------------------------------------------------
      }
      // Error Layer ---------------------------------------
      // Type: LossCrossEntropy
      {
         l = 16 + lo;
         shared_ptr<LossCrossEntropyEx> pl = make_shared<LossCrossEntropyEx>(len_out, 1);
         // DAGErrorLayer(shared_ptr<iLossLayer> _pLayer, shared_ptr<NetContext> _pContext, shared_ptr<ErrorContext> _pEContext) 
         LayerList.push_back(make_shared<DAGErrorLayer>(l, static_pointer_cast<iLossLayer>(pl), CMap[b], EMap[n]));
      }

      //---------------------------------------------------------------      
      if (!last_branch) {
         //if (n < gTrainingBranch) {
         cout << "Stop learning at layer: " << n << endl;
         l = 25 + lo;
         LayerList.push_back(make_shared<DAGLambdaLayer>(l, []() {}, []() { gpExit->stop = true;  }));
         //}
         // Branch Test Layer (Test 1) -----------------------------------------
         // upper limit: read --> if your that accurate don't bother backpropagating, not much to learn.
         // lower limit: read --> if your that screwed up I don't want to try to learn what you are.
         //                                                                              warm up             backprop lower limit   backprop upper limit
         l = 24 + lo;
         //LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, EMap[n], lrestore ? 0 : 20 * 1100, 0.7 - ((double)(n - 1)/10.0), 0.98));
         //cout << "0.7 -(n-1)/10 , 0.98 Exit test" << endl;
         //LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, EMap[n], lrestore ? 0 : 20 * 1100, 0.7, 0.98));
         //cout << "0.7 , 0.98 Exit test" << endl;
         //                                                                     Sets branch threshold -------------------- V
         LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, EMap[n], lrestore ? 0 : 20 * 1100, 0.0, 0.0, branch_pass_threshold));
         cout << "0.0, 0, Exit Test" << endl;
         //
         //-----------------------------------------------------------------------

      }
   };


   //***************** LP Transformer *******************************
   // 
   // Copy Start Context -----------------------------------------
   //
   int l = 200;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[200]));
   //
   //-----------------------------------------------------------------------
   // Copy Start Context -----------------------------------------
   //
   l = 201;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[201]));
   //
   //-----------------------------------------------------------------------

   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   clSize size_ct_in(28, 28);
   clSize size_ct_out(28, 28);
   clSize size_ct_kern(28, 28);
   {
      l = 202;
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_ct_in, 1, size_ct_out, size_ct_kern, 1,
         make_unique<actReLU>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>(),
         true); // No bias. true/false
      pl->SetConvolutionParameters(FilterLayer2D::FFT, false, false);

      //pl->SetBackpropCallBack(MCB);

      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[200]));
   }
   //---------------------------------------------------------------

   // Centroid Layer --------------------------------------------
   // Type: Centroid
   // Note: Uses DAGFlattenObj.
   size_ct_in = size_ct_out;
   {
      l = 203;
      shared_ptr<Centroid> pl = make_shared<Centroid>(size_ct_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, CMap[200]));
   }
   //-----------------------------------------------------------------

   //---------------------------------------------------------------  
   // LP Spacial Transformer
   //
   {
      l = 204;
      shared_ptr<samplerBiLinear> psmp = make_shared<samplerBiLinear>(Size(28, 28), Size(INPUT_ROWS, INPUT_COLS));
      psmp->SetGridBackpropCallBack(MCB1);

      shared_ptr<gridLogPolar> pgrd = make_shared<gridLogPolar>(Size(28, 28), Size(INPUT_ROWS, INPUT_COLS), LP_WASTE);
      // REVIEW: Transform sanity check test.
      //pCVT->SetEvalPostActivationCallBack(MCB2);
      LayerList.push_back(make_shared<DAGSpacialTransformLayer>(l, psmp, pgrd, CMap[201], CMap[200], false)); // C200 holds center point value.
   }
   //-----------------------------------------------------------------

   //-----------------------------------------------------------------------
   // Update the Start Context to Log Polar.  This is what the branches start with. -----------------------------------------
   //
   l = 205;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[201], CMap[1]));
   //
   //-----------------------------------------------------------------------

   // !!!!!!!!!!!!----->  LOOK <------ !!!!!!!!!!!!
   cout << "Stop learning at LP Transform layer. " << endl;
   l = 206;
   LayerList.push_back(make_shared<DAGLambdaLayer>(l, []() {}, []() { gpExit->stop = true;  }));

   //********* End LP Transformer *********************************************

   for (int n = 0; n <= gTrainingBranch; n++) {
      fbranch(n, restore, lrestore, gTrainingBranch == n ? true : false);
   }
}

// This model should be used with Train4.
// This model is trying to train one branch at a time.  If the structure of the network
// isn't changed the results should be compatible with Model 6.
// The code in this model utilizes global parameters at the top of the file.
//
void InitLPBranchModel8(bool restore)
{
   INITIALIZE("LPB8\\LPB8", optoADAM)
      gModelBranches = 5;

   // It only makes sence to initialize the parameters when gTrainingBranch is set to 4.
   //runtime_assert( restore || (!restore && gTrainingBranch==4) )
   //runtime_assert(gTrainingBranch>=0 && gTrainingBranch<=4)

   // lrestore controls when warm-up is on.
   bool lrestore = gWarmupOn > 0 ? false : true;

   optoADAM::B1 = 0.9;
   optoADAM::B2 = 0.999;
   //optoADAM::B1 = 0.8;
   //optoADAM::B2 = 0.9;
   optoLowPass::Momentum = 0.8;

   LayerList.clear();

   int P1 = INPUT_ROWS;

   // NOTE: Establish the context that will store the composite pose estimate.
   //       It is fixed and does not change accross layers.
   CMap[0].v.resize(1);

   shared_ptr<unsigned int> pAvgGradCounter = make_shared<unsigned int>(0);

   // *********************************************************************
   //            Branch Lambda function
   //**********************************************************************

   auto fbranch = [&P1, &pAvgGradCounter](int n, bool restore, bool lrestore, bool last_branch = false) {
      // NOTE: Used by Filter Pool.
      // REIVEW: !!!!!!!!!!!!!!!!  LOOK LOOK !!!!!!!!!!!!!!!!!!!!!!!
      //          Set to 4,4 to test downsampling.  It has been 2,4 .
      clSize size_kernel(2, 4);

      int l = 1; // Layer label
      int lo = n * 30;  // Layer offset
      int b = 2 + n * 4;

      // Copy Start Context -----------------------------------------
      //
      l = 1 + lo;
      LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[b]));
      //
      //-----------------------------------------------------------------------

      // Compute average filter gradient  -----------------------------------------
      // NOTE: This does not work with backprop early stop.
      //
      l = 5 + lo;
      LayerList.push_back(make_shared<DAGAvgFilterGrad>(l, CMap[b], CMap[201], pAvgGradCounter));
      //
      //-----------------------------------------------------------------------

      // Convolution Layer -----------------------------------------
      // Type: FilterLayer2D
      clSize size_in(INPUT_ROWS, INPUT_COLS);
      clSize size_out(INPUT_ROWS, 4);
      clSize size_kern(INPUT_ROWS, INPUT_COLS);
      int chn_in = 1;
      int chn_out = 1;
      {
         l = 2 + lo;
         shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
            make_unique<actReLU>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, chn_in)),
            make_shared<OPTO>(),
            true); // No bias. true/false
         pl->SetConvolutionParameters(FilterLayer2D::FFT, true, false);
         //pl->SetBackpropCallBack(MCB);

         LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[b]));
      }
      //---------------------------------------------------------------
      // Pooling Layer ----------------------------------------------
      // Type: poolAvg2D
      size_in = size_out;
      size_out.Resize(P1, 1);
      chn_in = chn_out;

      assert(!(size_in.rows % size_out.rows));
      assert(!(size_in.cols % size_out.cols));
      {
         l = 3 + lo;
#ifdef USE_FILTER_FOR_POOL
         shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
         //pl->SetEvalPostActivationCallBack(MVCB);
#else
         shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
#endif
         LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, CMap[b]));
      }
      //---------------------------------------------------------------
      // Flattening Layer --------------------------------------------
      // Type: Flatten2D
      size_in = size_out;
      chn_in = chn_out;

      int len_in = 0;
      int len_out = size_in.rows * size_in.cols * chn_in;
      chn_out = 1;
      {
         l = 4 + lo;
         shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
         LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, CMap[b]));
      }

      //--------- setup the fully connected network -------------------------------------------------------------------------
      // 
      shared_ptr<DAGBranchObj> p_branch1;
      if (!last_branch) {
         // Out to Branch 2 Fully Connected Layer -----------------------------------------
         //
         // This branch will be paired with Test1.
         l = 23 + lo;
         // Branch backprop percent ------------------------------------------- |
         // Currently no backprop from other branches ................   v      v
         p_branch1 = make_shared<DAGBranchObj>(l, CMap[b], CMap[b + 3], false, 0.1);
         LayerList.push_back(p_branch1);
         //
         //-----------------------------------------------------------------------
      }

      // Join Fully Connected Layer -----------------------------------------
      //
      if (n > 0) {
         l = 22 + lo;
         LayerList.push_back(make_shared<DAGJoinObj>(l, CMap[b - 1], CMap[b]));
      }
      //
      //----------------------------------------------------------------------- 

      // Fully Connected Layer ---------------------------------------
      // Type: ReLU
      len_in = (n == 0) ? P1 : 2 * P1;
      len_out = len_in >> 1;
      {
         l = 6 + lo;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
            make_unique<actReLU>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[b]));
      }
      //---------------------------------------------------------------  

      // Fully Connected Layer ---------------------------------------
      // Type: SoftMAX
      len_in = len_out;
      len_out = 10;
      {
         l = 15 + lo;
         shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actSoftMaxEx>(),
            restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
            dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
            make_shared<OPTO>());
         LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[b]));
      }
      //---------------------------------------------------------------      
      // Error Layer ---------------------------------------
      // Type: LossCrossEntropy
      {
         l = 16 + lo;
         shared_ptr<LossCrossEntropyEx> pl = make_shared<LossCrossEntropyEx>(len_out, 1);
         // DAGErrorLayer(shared_ptr<iLossLayer> _pLayer, shared_ptr<NetContext> _pContext, shared_ptr<ErrorContext> _pEContext) 
         LayerList.push_back(make_shared<DAGErrorLayer>(l, static_pointer_cast<iLossLayer>(pl), CMap[b], EMap[n]));
      }

      //---------------------------------------------------------------      
      if (!last_branch) {
         //if (n < gTrainingBranch) {
         cout << "Stop learning at layer: " << n << endl;
         l = 25 + lo;
         LayerList.push_back(make_shared<DAGLambdaLayer>(l, []() {}, []() { gpExit->stop = true;  }));
         //}
         // Branch Test Layer (Test 1) -----------------------------------------
         // upper limit: read --> if your that accurate don't bother backpropagating, not much to learn.
         // lower limit: read --> if your that screwed up I don't want to try to learn what you are.
         //                                                                              warm up             backprop lower limit   backprop upper limit
         l = 24 + lo;
         //LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, EMap[n], lrestore ? 0 : 20 * 1100, 0.7 - ((double)(n - 1)/10.0), 0.98));
         //cout << "0.7 -(n-1)/10 , 0.98 Exit test" << endl;
         //LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, EMap[n], lrestore ? 0 : 20 * 1100, 0.7, 0.98));
         //cout << "0.7 , 0.98 Exit test" << endl;
         //                                                                     Sets branch threshold -------------------- V
         LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, EMap[n], lrestore ? 0 : 20 * 1100, 0.0, 0.0, branch_pass_threshold));
         cout << "0.0, 0, Exit Test" << endl;
         //
         //-----------------------------------------------------------------------

      }
   };


   //***************** LP Transformer *******************************
   // 
   // Copy Start Context -----------------------------------------
   //
   int l = 200;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[200]));
   //
   //-----------------------------------------------------------------------
   // Copy Start Context -----------------------------------------
   //
   l = 201;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[201]));
   //
   //-----------------------------------------------------------------------

   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   clSize size_ct_in(28, 28);
   clSize size_ct_out(28, 28);
   clSize size_ct_kern(28, 28);
   {
      l = 202;
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_ct_in, 1, size_ct_out, size_ct_kern, 1,
         make_unique<actReLU>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>(),
         true); // No bias. true/false
      pl->SetConvolutionParameters(FilterLayer2D::FFT, false, false);

      //pl->SetBackpropCallBack(MCB);

      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[200]));
   }
   //---------------------------------------------------------------

   // Centroid Layer --------------------------------------------
   // Type: Centroid
   // Note: Uses DAGFlattenObj.
   size_ct_in = size_ct_out;
   {
      l = 203;
      shared_ptr<Centroid> pl = make_shared<Centroid>(size_ct_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, CMap[200]));
   }
   //-----------------------------------------------------------------

   //---------------------------------------------------------------  
   // LP Spacial Transformer
   //
   {
      l = 204;
      shared_ptr<samplerBiLinear> psmp = make_shared<samplerBiLinear>(Size(28, 28), Size(INPUT_ROWS, INPUT_COLS));
      psmp->SetGridBackpropCallBack(MCB1);

      shared_ptr<gridLogPolar> pgrd = make_shared<gridLogPolar>(Size(28, 28), Size(INPUT_ROWS, INPUT_COLS), LP_WASTE);
      // REVIEW: Transform sanity check test.
      //pCVT->SetEvalPostActivationCallBack(MCB2);
      LayerList.push_back(make_shared<DAGSpacialTransformLayer>(l, psmp, pgrd, CMap[201], CMap[200], false)); // C200 holds center point value.
   }
   //-----------------------------------------------------------------

   //-----------------------------------------------------------------------
   // Update the Start Context to Log Polar.  This is what the branches start with. -----------------------------------------
   //
   l = 205;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[201], CMap[1]));
   //
   //-----------------------------------------------------------------------

   // !!!!!!!!!!!!----->  LOOK <------ !!!!!!!!!!!!
   cout << "Stop learning at LP Transform layer. " << endl;
   l = 206;
   LayerList.push_back(make_shared<DAGLambdaLayer>(l, []() {}, []() { gpExit->stop = true;  }));

   //********* End LP Transformer *********************************************

   for (int n = 0; n <= gTrainingBranch; n++) {
      fbranch(n, restore, lrestore, gTrainingBranch == n ? true : false);
   }
}

void InitCharacterEmbedding(bool restore, int number)
{
   // ToDo: Make a single branch network with LP spacial trasnformer that
   //       results in a single number that ideally is 0 or 1.
   //       It trains to identify one number, the target number, which should
   //       return a 1, any other number should result in 0.
   //       Use L2 loss to train it.
   //       There should be no training on the spacial transformer.


   string model = "LPCE\\LPCE." + to_string(number);
   INITIALIZE(model, optoADAM)
   gModelBranches = 1;

   // It only makes sence to initialize the parameters when gTrainingBranch is set to 4.
   //runtime_assert( restore || (!restore && gTrainingBranch==4) )
   //runtime_assert(gTrainingBranch>=0 && gTrainingBranch<=4)

   // lrestore controls when warm-up is on.
   //bool lrestore = gWarmupOn > 0 ? false : true;
   bool lrestore = true;
   bool st_restore = true;

   optoADAM::B1 = 0.9;
   optoADAM::B2 = 0.999;
   //optoADAM::B1 = 0.8;
   //optoADAM::B2 = 0.9;
   optoLowPass::Momentum = 0.8;

   LayerList.clear();

   int P1 = INPUT_ROWS;

   // NOTE: Establish the context that will store the composite pose estimate.
   //       It is fixed and does not change accross layers.
   CMap[0].v.resize(1);

   shared_ptr<unsigned int> pAvgGradCounter = make_shared<unsigned int>(0);


   //***************** LP Transformer *******************************
   // 
   // Copy Start Context -----------------------------------------
   //
   int l = 200;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[200]));
   //
   //-----------------------------------------------------------------------
   // Copy Start Context -----------------------------------------
   //
   l = 201;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[201]));
   //
   //-----------------------------------------------------------------------

   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   clSize size_ct_in(28, 28);
   clSize size_ct_out(28, 28);
   clSize size_ct_kern(28, 28);
   {
      l = 202;
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_ct_in, 1, size_ct_out, size_ct_kern, 1,
         make_unique<actReLU>(),
         st_restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, "LPCE\\LPB6." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>(),
         true); // No bias. true/false
      pl->SetConvolutionParameters(FilterLayer2D::FFT, false, false);

      //pl->SetBackpropCallBack(MCB);

      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[200]));
   }
   //---------------------------------------------------------------

   // Centroid Layer --------------------------------------------
   // Type: Centroid
   // Note: Uses DAGFlattenObj.
   size_ct_in = size_ct_out;
   {
      l = 203;
      shared_ptr<Centroid> pl = make_shared<Centroid>(size_ct_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, CMap[200]));
   }
   //-----------------------------------------------------------------

   //---------------------------------------------------------------  
   // LP Spacial Transformer
   //
   {
      l = 204;
      shared_ptr<samplerBiLinear> psmp = make_shared<samplerBiLinear>(Size(28, 28), Size(INPUT_ROWS, INPUT_COLS));
      psmp->SetGridBackpropCallBack(MCB1);

      shared_ptr<gridLogPolar> pgrd = make_shared<gridLogPolar>(Size(28, 28), Size(INPUT_ROWS, INPUT_COLS), LP_WASTE);
      // REVIEW: Transform sanity check test.
      //pCVT->SetEvalPostActivationCallBack(MCB2);
      LayerList.push_back(make_shared<DAGSpacialTransformLayer>(l, psmp, pgrd, CMap[201], CMap[200], false)); // C200 holds center point value.
   }
   //-----------------------------------------------------------------

   //-----------------------------------------------------------------------
   // Update the Start Context to Log Polar.  This is what the branches start with. -----------------------------------------
   //
   l = 205;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[201], CMap[1]));
   //
   //-----------------------------------------------------------------------

   // !!!!!!!!!!!!----->  LOOK <------ !!!!!!!!!!!!
   cout << "Stop learning at LP Transform layer. " << endl;
   l = 206;
   LayerList.push_back(make_shared<DAGLambdaLayer>(l, []() {}, []() { gpExit->stop = true;  }));

   //********* End LP Transformer *********************************************

         // NOTE: Used by Filter Pool.
      // REIVEW: !!!!!!!!!!!!!!!!  LOOK LOOK !!!!!!!!!!!!!!!!!!!!!!!
      //          Set to 4,4 to test downsampling.  It has been 2,4 .
   clSize size_kernel(2, 4);

   l = 1; // Layer label
   int lo = 0;  // Layer offset
   int b = 2;

   // Copy Start Context -----------------------------------------
   //
   l = 1 + lo;
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, CMap[1], CMap[b]));
   //
   //-----------------------------------------------------------------------

   // Compute average filter gradient  -----------------------------------------
   // NOTE: This does not work with backprop early stop.
   //
   //l = 5 + lo;
   //LayerList.push_back(make_shared<DAGAvgFilterGrad>(l, CMap[b], CMap[201], pAvgGradCounter));
   //
   //-----------------------------------------------------------------------

   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   clSize size_in(INPUT_ROWS, INPUT_COLS);
   clSize size_out(INPUT_ROWS, 4);
   clSize size_kern(INPUT_ROWS, INPUT_COLS);
   int chn_in = 1;
   int chn_out = 1;
   {
      l = 2 + lo;
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
         make_unique<actReLU>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false
      pl->SetConvolutionParameters(FilterLayer2D::FFT, true, false);
      //pl->SetBackpropCallBack(MCB);

      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), CMap[b]));
   }
   //---------------------------------------------------------------
   // Pooling Layer ----------------------------------------------
   // Type: poolAvg2D
   size_in = size_out;
   size_out.Resize(INPUT_ROWS, 1);
   chn_in = chn_out;

   assert(!(size_in.rows% size_out.rows));
   assert(!(size_in.cols% size_out.cols));
   {
      l = 3 + lo;
#ifdef USE_FILTER_FOR_POOL
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      //pl->SetEvalPostActivationCallBack(MVCB);
#else
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
#endif
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, CMap[b]));
   }
   //---------------------------------------------------------------
   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in = size_out;
   chn_in = chn_out;

   int len_in = 0;
   int len_out = size_in.rows * size_in.cols * chn_in;
   chn_out = 1;
   {
      l = 4 + lo;
      shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, CMap[b]));
   }

   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   len_in = len_out;

   len_out = len_in >> 1;
   {
      l = 6 + lo;
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out,
         make_unique<actLinear>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[b]));
   }

   // Fully Connected Layer ---------------------------------------
   // Type:
   len_in = len_out;
   len_out = 1;
   {
      l = 15 + lo;
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actTanh>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), CMap[b]));
   }
   //---------------------------------------------------------------      

   // Error Layer ---------------------------------------
   {
      // Type:
      typedef LossL2 LT;
      l = 16 + lo;
      shared_ptr<LT> pl = make_shared<LT>(len_out, 1);
      LayerList.push_back(make_shared<DAGErrorLayer>(l, static_pointer_cast<iLossLayer>(pl), CMap[b], EMap[0], 
         [](const ColVector& x, const ColVector& y) {
            runtime_assert(x.size() == 1);
            bool correct = (x[0] * y[0]) > 0.0 ? true : false;
            return std::pair<double, bool>(x[0], correct);
         }
         ));
   }
}

/*

// NOTES:
//   6/11 - Made the lower limit on the branch test successivly lower, meaning
//          as the branches get deeper each lower branch trains on more of the
//          samples that it sees.  The ideas is that the top branches are particular,
//          they are only interested in the good stuff, the well behaved stuff, and the
//          lower branches have to cast the net wider was they only get misfits.
//          This scheme lead to 99.64 on the test data.
//
//    6/9 - Implemented a data augmentation scheme that randomly injects offsets
//          and rotation into the sample.  Rotation is +- pi/8 rad.
void InitLPBranchModel9(bool restore)
{
   INITIALIZE("LPB9\\LPB9", optoADAM)
   gModelBranches = 5;

   optoADAM::B1 = 0.9;
   optoADAM::B2 = 0.999;
   optoLowPass::Momentum = 0.75;

   clSize size_kernel(4, 4);

   const int l1warmup = 10 * 1000;
   const int l2warmup = 10 * 1000;
   const int l3warmup = 10 * 1000;
   const int l4warmup = 10 * 1000;

   int l = 1;
   LayerList.clear();

   // Copy Start Context -----------------------------------------
   //
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, pContextStart, pContext1));
   l++;
   //
   //-----------------------------------------------------------------------

   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   clSize size_in(INPUT_ROWS, INPUT_COLS);
   clSize size_out(INPUT_ROWS, 4);
   // Use this to have a look at the entire correlation plane.
   //clSize size_out(INPUT_ROWS, INPUT_COLS);
   clSize size_kern(INPUT_ROWS, INPUT_COLS);
   int kern_per_chn = 1;
   int chn_in = 1;
   int chn_out = kern_per_chn * chn_in;
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, kern_per_chn,
         make_unique<actReLU>(),
         //make_unique<actLinear>(), 
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.

      //pl->SetEvalPreActivationCallBack(MCB);
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, pContext1));
   }
   l++;
   //---------------------------------------------------------------
   // Average Layer ----------------------------------------------
   size_in = size_out;
   size_out.Resize(INPUT_ROWS, 1);
   chn_in = chn_out;

   {
#ifdef USE_FILTER_FOR_POOL
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      pl->SetEvalPostActivationCallBack(MVCB);
#else
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
#endif
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, pContext1));
   }
   l++;
   //---------------------------------------------------------------

   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in = size_out;
   chn_in = chn_out;
   int len_out = size_in.rows * size_in.cols * chn_in;
   chn_out = 1;
   {
      shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, static_pointer_cast<iConvoLayer>(pl), pContext1, pContext2));
   }
   l++;
   //---------------------------------------------------------------      
   // 
   //--------- setup the fully connected network -------------------------------------------------------------------------
   // 
   // Branch 1 Fully Connected Layer -----------------------------------------
   //
   int len_out_branch_1 = len_out;
   // This branch will be paired with Test1.
   shared_ptr<DAGBranchObj> p_branch1 = make_shared<DAGBranchObj>(l, pContext2, pContext3, false);
   LayerList.push_back(p_branch1);
   l++;
   //
   //-----------------------------------------------------------------------

   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   int len_in = len_out;
   len_out = 24;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actLeakyReLU>(0.01),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), pContext2));
   }
   l++;
   //---------------------------------------------------------------  

   // Fully Connected Layer ---------------------------------------
   // Type: SoftMAX
   len_in = len_out;
   len_out = 10;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actSoftMax>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), pContext2));
   }
   l++;
   //---------------------------------------------------------------      
   // Error Layer ---------------------------------------
   // Type: LossCrossEntropy
   LayerList.push_back(make_shared<DAGErrorLayer>(l, make_shared<LossCrossEntropy>(len_out, 1), pContext2, gpError1));
   l++;
   //---------------------------------------------------------------      
   // Branch Test Layer (Test 1) -----------------------------------------
   // upper limit: read --> if your that accurate don't bother backpropagating, not much to learn.
   // lower limit: read --> if your that screwed up I don't want to try to learn what you are.
   //                                                                                  warm up   backprop lower limit   backprop upper limit
   LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, gpError1, restore ? 0 : l1warmup, 0.7, 0.98));
   l++;
   //
   //-----------------------------------------------------------------------

   //return;

   //************************************************************************
   //              Branch 2
   // Copy Start Context -----------------------------------------
   //
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, pContextStart, pContext4));
   l++;
   //
   //-----------------------------------------------------------------------
   // Branch 2 Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   size_in.Resize(INPUT_ROWS, INPUT_COLS);
   size_out.Resize(INPUT_ROWS, 4);
   size_kern.Resize(INPUT_ROWS, INPUT_COLS);
   kern_per_chn = 1;
   chn_in = 1;
   chn_out = kern_per_chn * chn_in;
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, kern_per_chn,
         make_unique<actReLU>(),
         //make_unique<actLinear>(), 
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, pContext4));
   }
   l++;
   //---------------------------------------------------------------
      // Average Layer ----------------------------------------------
   size_in = size_out;
   size_out.Resize(INPUT_ROWS, 1);
   chn_in = chn_out;

   {
#ifdef USE_FILTER_FOR_POOL
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      pl->SetEvalPostActivationCallBack(MVCB);
#else
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
#endif
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, pContext4));
   }
   l++;
   //---------------------------------------------------------------

   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in = size_out;
   chn_in = chn_out;
   len_out = size_in.rows * size_in.cols * chn_in;
   chn_out = 1;
   {
      shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, pContext4, pContext5));
   }
   l++;
   //---------------------------------------------------------------  

   // Join Fully Connected Layer -----------------------------------------
   //
   len_out += len_out_branch_1;
   LayerList.push_back(make_shared<DAGJoinObj>(l, pContext3, pContext5));
   l++;
   //
   //----------------------------------------------------------------------- 
   // Branch 2 Fully Connected Layer -----------------------------------------
   //
   int len_out_branch_2 = len_out;
   // This branch will be paired with Test2.
   shared_ptr<DAGBranchObj> p_branch2 = make_shared<DAGBranchObj>(l, pContext5, pContext6, false);
   LayerList.push_back(p_branch2);
   l++;
   //
   //-----------------------------------------------------------------------    
   //--------- setup the fully connected network -------------------------------------------------------------------------
   // 
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   len_in = len_out;
   len_out = 48; //64;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actLeakyReLU>(0.01),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, pContext5));
   }
   l++;
   //---------------------------------------------------------------  

   // Fully Connected Layer ---------------------------------------
   // Type: SoftMAX
   len_in = len_out;
   len_out = 10;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actSoftMax>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, pContext5));
   }
   l++;
   //---------------------------------------------------------------      
   // Error Layer ---------------------------------------
   // Type: LossCrossEntropy
   LayerList.push_back(make_shared<DAGErrorLayer>(l, make_shared<LossCrossEntropy>(len_out, 1), pContext5, gpError2));
   l++;
   // Branch Test Layer (Test 2) -----------------------------------------
   //
   LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch2, gpError2, restore ? 0 : l2warmup, 0.6, 0.98));
   l++;
   //
   //-----------------------------------------------------------------------

   //************************************************************************
   //              Branch 3
   // Copy Start Context -----------------------------------------
   //
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, pContextStart, pContext7));
   l++;
   //
   //-----------------------------------------------------------------------
   // Branch 2 Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   size_in.Resize(INPUT_ROWS, INPUT_COLS);
   size_out.Resize(INPUT_ROWS, 4);
   size_kern.Resize(INPUT_ROWS, INPUT_COLS);
   kern_per_chn = 1;
   chn_in = 1;
   chn_out = kern_per_chn * chn_in;
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, kern_per_chn,
         make_unique<actReLU>(),
         //make_unique<actLinear>(), 
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, pContext7));
   }
   l++;
   //---------------------------------------------------------------
   // Average Layer ----------------------------------------------
   size_in = size_out;
   size_out.Resize(INPUT_ROWS, 1);
   chn_in = chn_out;

   {
#ifdef USE_FILTER_FOR_POOL
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      pl->SetEvalPostActivationCallBack(MVCB);
#else
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
#endif
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, pContext7));
   }
   l++;
   //---------------------------------------------------------------
   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in = size_out;
   chn_in = chn_out;
   len_out = size_in.rows * size_in.cols * chn_in;
   chn_out = 1;
   {
      shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, pContext7, pContext8));
   }
   l++;
   //---------------------------------------------------------------  

   // Join Fully Connected Layer -----------------------------------------
   //
   len_out += len_out_branch_2;
   LayerList.push_back(make_shared<DAGJoinObj>(l, pContext6, pContext8));
   l++;
   //
   //----------------------------------------------------------------------- 
   // Branch 3 Fully Connected Layer -----------------------------------------
   //
   int len_out_branch_3 = len_out;
   // This branch will be paired with Test2.
   shared_ptr<DAGBranchObj> p_branch3 = make_shared<DAGBranchObj>(l, pContext8, pContext9, false);
   LayerList.push_back(p_branch3);
   l++;
   //
   //-----------------------------------------------------------------------    
   //--------- setup the fully connected network -------------------------------------------------------------------------
   // 
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   len_in = len_out;
   len_out = 72; // 96;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actLeakyReLU>(0.01),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, pContext8));
   }
   l++;
   //---------------------------------------------------------------  

   // Fully Connected Layer ---------------------------------------
   // Type: SoftMAX
   len_in = len_out;
   len_out = 10;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actSoftMax>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, pContext8));
   }
   l++;
   //---------------------------------------------------------------      
   // Error Layer ---------------------------------------
   // Type: LossCrossEntropy
   LayerList.push_back(make_shared<DAGErrorLayer>(l, make_shared<LossCrossEntropy>(len_out, 1), pContext8, gpError3));
   l++;
   // Branch Test Layer (Test 3) -----------------------------------------
   //
   LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch3, gpError3, restore ? 0 : l3warmup, 0.5, 0.98));
   l++;
   //
   //-----------------------------------------------------------------------

   //************************************************************************
   //              Branch 4
   // Copy Start Context -----------------------------------------
   //
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, pContextStart, pContext10));
   l++;
   //
   //-----------------------------------------------------------------------
   // Branch 2 Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   size_in.Resize(INPUT_ROWS, INPUT_COLS);
   size_out.Resize(INPUT_ROWS, 4);
   size_kern.Resize(INPUT_ROWS, INPUT_COLS);
   kern_per_chn = 1;
   chn_in = 1;
   chn_out = kern_per_chn * chn_in;
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, kern_per_chn,
         make_unique<actReLU>(),
         //make_unique<actLinear>(), 
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, pContext10));
   }
   l++;
   //---------------------------------------------------------------

   // Average Layer ----------------------------------------------
   size_in = size_out;
   size_out.Resize(INPUT_ROWS, 1);
   chn_in = chn_out;

   {
#ifdef USE_FILTER_FOR_POOL
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      pl->SetEvalPostActivationCallBack(MVCB);
#else
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
#endif
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, pContext10));
   }
   l++;
   //---------------------------------------------------------------
   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in = size_out;
   chn_in = chn_out;
   len_out = size_in.rows * size_in.cols * chn_in;
   chn_out = 1;
   {
      shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, pContext10, pContext11));
   }
   l++;
   //---------------------------------------------------------------  

   // Join Fully Connected Layer -----------------------------------------
   //
   len_out += len_out_branch_3;
   LayerList.push_back(make_shared<DAGJoinObj>(l, pContext9, pContext11));
   l++;
   //
   //----------------------------------------------------------------------- 
   // Branch 4 Fully Connected Layer -----------------------------------------
   //
   int len_out_branch_4 = len_out;
   // This branch will be paired with Test4.
   shared_ptr<DAGBranchObj> p_branch4 = make_shared<DAGBranchObj>(l, pContext11, pContext12 , false);
   LayerList.push_back(p_branch4);
   l++;
   //
   //-----------------------------------------------------------------------    
   //--------- setup the fully connected network -------------------------------------------------------------------------
   // 
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   len_in = len_out;
   len_out = 96; // 128;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actLeakyReLU>(0.01),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, pContext11));
   }
   l++;
   //---------------------------------------------------------------  

   // Fully Connected Layer ---------------------------------------
   // Type: SoftMAX
   len_in = len_out;
   len_out = 10;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actSoftMax>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, pContext11));
   }
   l++;
   //---------------------------------------------------------------      
   // Error Layer ---------------------------------------
   // Type: LossCrossEntropy
   LayerList.push_back(make_shared<DAGErrorLayer>(l, make_shared<LossCrossEntropy>(len_out, 1), pContext11, gpError4));
   l++;
   // Branch Test Layer (Test 4) -----------------------------------------
   //
   LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch4, gpError4, restore ? 0 : l4warmup, 0.4, 0.98));
   l++;
   //
   //-----------------------------------------------------------------------
   //************************************************************************
   //              Branch 5
   // Copy Start Context -----------------------------------------
   //
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, pContextStart, pContext13));
   l++;
   //
   //-----------------------------------------------------------------------
   // Branch Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   size_in.Resize(INPUT_ROWS, INPUT_COLS);
   size_out.Resize(INPUT_ROWS, 4);
   size_kern.Resize(INPUT_ROWS, INPUT_COLS);
   kern_per_chn = 1;
   chn_in = 1;
   chn_out = kern_per_chn * chn_in;
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, kern_per_chn,
         make_unique<actReLU>(),
         //make_unique<actLinear>(), 
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, pContext13));
   }
   l++;
   //---------------------------------------------------------------

   // Average Layer ----------------------------------------------
   size_in = size_out;
   size_out.Resize(INPUT_ROWS, 1);
   chn_in = chn_out;

   {
#ifdef USE_FILTER_FOR_POOL
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      pl->SetEvalPostActivationCallBack(MVCB);
#else
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
#endif
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, pContext13));
   }
   l++;
   //---------------------------------------------------------------
   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in = size_out;
   chn_in = chn_out;
   len_out = size_in.rows * size_in.cols * chn_in;
   chn_out = 1;
   {
      shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, pContext13, pContext14));
   }
   l++;
   //---------------------------------------------------------------  

   // Join Fully Connected Layer -----------------------------------------
   //
   len_out += len_out_branch_4;
   LayerList.push_back(make_shared<DAGJoinObj>(l, pContext12, pContext14));
   l++;
   //
   //----------------------------------------------------------------------- 
   // Branch 5 Fully Connected Layer -----------------------------------------
   //
   //int len_out_branch_5 = len_out;
   // This branch will be paired with Test4.
   //shared_ptr<DAGBranchObj> p_branch5 = make_shared<DAGBranchObj>(l, pContext14, pContext, false);
   //LayerList.push_back(p_branch5);
   //l++;
   //
   //-----------------------------------------------------------------------    
   //--------- setup the fully connected network -------------------------------------------------------------------------
   // 
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   len_in = len_out;
   len_out = 120; // 160;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actLeakyReLU>(0.01),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, pContext14));
   }
   l++;
   //---------------------------------------------------------------  

   // Fully Connected Layer ---------------------------------------
   // Type: SoftMAX
   len_in = len_out;
   len_out = 10;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actSoftMax>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, pContext14));
   }
   l++;
   //---------------------------------------------------------------      
   // Error Layer ---------------------------------------
   // Type: LossCrossEntropy
   LayerList.push_back(make_shared<DAGErrorLayer>(l, make_shared<LossCrossEntropy>(len_out, 1), pContext14, gpError5));
   l++;
   // Branch Test Layer (Test 5) -----------------------------------------
   //
   //LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch5, gpError5, restore ? 0 : 6 * 1000, 0.6, 0.98));
   //l++;
   //
   //-----------------------------------------------------------------------

}
void InitLPBranchModel9A(bool restore)
{
   INITIALIZE("LPB9\\LPB9A", optoADAM)
   gModelBranches = 5;

   optoADAM::B1 = 0.9;
   optoADAM::B2 = 0.999;
   optoLowPass::Momentum = 0.75;

   clSize size_kernel(4, 4);

   const int l1warmup = 10 * 1000;
   const int l2warmup = 10 * 1000;
   const int l3warmup = 10 * 1000;
   const int l4warmup = 10 * 1000;

   int l = 1;
   LayerList.clear();

   //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
   // REVIEW: Could redistribute the other copies.  To lazy right now.
   // Copy Start Context -----------------------------------------
//
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, pContextStart, pContext1));
   l++;
   //
   //-----------------------------------------------------------------------


   // Copy Input Layer -----------------------------------------
   //
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, pContext1, pContext4));
   l++;
   //
   //-----------------------------------------------------------------------
   // Copy Input Layer -----------------------------------------
   //
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, pContext1, pContext7));
   l++;
   //
   //-----------------------------------------------------------------------
   // Copy Input Layer -----------------------------------------
   //
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, pContext1, pContext10));
   l++;
   //
   //-----------------------------------------------------------------------
   // Copy Input Layer -----------------------------------------
   //
   LayerList.push_back(make_shared<DAGConvoContextCopyObj>(l, pContext1, pContext13));
   l++;
   //
   //-----------------------------------------------------------------------
   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   clSize size_in(INPUT_ROWS, INPUT_COLS);
   clSize size_out(INPUT_ROWS, 4);
   // Use this to have a look at the entire correlation plane.
   //clSize size_out(INPUT_ROWS, INPUT_COLS);
   clSize size_kern(INPUT_ROWS, INPUT_COLS);
   int kern_per_chn = 1;
   int chn_in = 1;
   int chn_out = kern_per_chn * chn_in;
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, kern_per_chn,
         make_unique<actReLU>(),
         //make_unique<actLinear>(), 
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.

      pl->SetEvalPreActivationCallBack(MVCB);
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, pContext1));
   }
   l++;
   //---------------------------------------------------------------
   // Average Layer ----------------------------------------------
   size_in = size_out;
   size_out.Resize(INPUT_ROWS, 1);
   chn_in = chn_out;

   {
#ifdef USE_FILTER_FOR_POOL
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      pl->SetEvalPostActivationCallBack(MVCB);
#else
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
#endif
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, pContext1));
   }
   l++;
   //---------------------------------------------------------------

   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in = size_out;
   chn_in = chn_out;
   int len_out = size_in.rows * size_in.cols * chn_in;
   chn_out = 1;
   {
      shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, static_pointer_cast<iConvoLayer>(pl), pContext1, pContext2));
   }
   l++;
   //---------------------------------------------------------------      
   // 
   //--------- setup the fully connected network -------------------------------------------------------------------------
   // 
   // Branch 1 Fully Connected Layer -----------------------------------------
   //
   int len_out_branch_1 = len_out;
   // This branch will be paired with Test1.
   shared_ptr<DAGBranchObj> p_branch1 = make_shared<DAGBranchObj>(l, pContext2, pContext3, false);
   LayerList.push_back(p_branch1);
   l++;
   //
   //-----------------------------------------------------------------------

   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   int len_in = len_out;
   len_out = 24;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actLeakyReLU>(0.01),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), pContext2));
   }
   l++;
   //---------------------------------------------------------------  

   // Fully Connected Layer ---------------------------------------
   // Type: SoftMAX
   len_in = len_out;
   len_out = 10;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actSoftMax>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, static_pointer_cast<iLayer>(pl), pContext2));
   }
   l++;
   //---------------------------------------------------------------      
   // Error Layer ---------------------------------------
   // Type: LossCrossEntropy
   LayerList.push_back(make_shared<DAGErrorLayer>(l, make_shared<LossCrossEntropy>(len_out, 1), pContext2, gpError1));
   l++;
   //---------------------------------------------------------------      
   // Branch Test Layer (Test 1) -----------------------------------------
   // upper limit: read --> if your that accurate don't bother backpropagating, not much to learn.
   // lower limit: read --> if your that screwed up I don't want to try to learn what you are.
   //                                                                              warm up             backprop lower limit   backprop upper limit
   LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, gpError1, restore ? 0 : l1warmup, 0.6, 0.98));
   l++;
   //
   //-----------------------------------------------------------------------

   //return;

   //************************************************************************
   //              Branch 2
   //-----------------------------------------------------------------------
   // Branch 2 Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   size_in.Resize(INPUT_ROWS, INPUT_COLS);
   size_out.Resize(INPUT_ROWS, 4);
   size_kern.Resize(INPUT_ROWS, INPUT_COLS);
   kern_per_chn = 1;
   chn_in = 1;
   chn_out = kern_per_chn * chn_in;
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, kern_per_chn,
         make_unique<actReLU>(),
         //make_unique<actLinear>(), 
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, pContext4));
   }
   l++;
   //---------------------------------------------------------------
      // Average Layer ----------------------------------------------
   size_in = size_out;
   size_out.Resize(INPUT_ROWS, 1);
   chn_in = chn_out;

   {
#ifdef USE_FILTER_FOR_POOL
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      pl->SetEvalPostActivationCallBack(MVCB);
#else
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
#endif
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, pContext4));
   }
   l++;
   //---------------------------------------------------------------

   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in = size_out;
   chn_in = chn_out;
   len_out = size_in.rows * size_in.cols * chn_in;
   chn_out = 1;
   {
      shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, pContext4, pContext5));
   }
   l++;
   //---------------------------------------------------------------  

   // Join Fully Connected Layer -----------------------------------------
   //
   len_out += len_out_branch_1;
   LayerList.push_back(make_shared<DAGJoinObj>(l, pContext3, pContext5));
   l++;
   //
   //----------------------------------------------------------------------- 
   // Branch 2 Fully Connected Layer -----------------------------------------
   //
   int len_out_branch_2 = len_out;
   // This branch will be paired with Test2.
   shared_ptr<DAGBranchObj> p_branch2 = make_shared<DAGBranchObj>(l, pContext5, pContext6, false);
   LayerList.push_back(p_branch2);
   l++;
   //
   //-----------------------------------------------------------------------    
   //--------- setup the fully connected network -------------------------------------------------------------------------
   // 
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   len_in = len_out;
   len_out = 48; //64;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actLeakyReLU>(0.01),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, pContext5));
   }
   l++;
   //---------------------------------------------------------------  

   // Fully Connected Layer ---------------------------------------
   // Type: SoftMAX
   len_in = len_out;
   len_out = 10;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actSoftMax>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, pContext5));
   }
   l++;
   //---------------------------------------------------------------      
   // Error Layer ---------------------------------------
   // Type: LossCrossEntropy
   LayerList.push_back(make_shared<DAGErrorLayer>(l, make_shared<LossCrossEntropy>(len_out, 1), pContext5, gpError2));
   l++;
   // Branch Test Layer (Test 2) -----------------------------------------
   //
   LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch2, gpError2, restore ? 0 : l2warmup, 0.6, 0.98));
   l++;
   //
   //-----------------------------------------------------------------------

   //************************************************************************
   //              Branch 3
   //-----------------------------------------------------------------------
   // Branch 2 Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   size_in.Resize(INPUT_ROWS, INPUT_COLS);
   size_out.Resize(INPUT_ROWS, 4);
   size_kern.Resize(INPUT_ROWS, INPUT_COLS);
   kern_per_chn = 1;
   chn_in = 1;
   chn_out = kern_per_chn * chn_in;
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, kern_per_chn,
         make_unique<actReLU>(),
         //make_unique<actLinear>(), 
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, pContext7));
   }
   l++;
   //---------------------------------------------------------------
   // Average Layer ----------------------------------------------
   size_in = size_out;
   size_out.Resize(INPUT_ROWS, 1);
   chn_in = chn_out;

   {
#ifdef USE_FILTER_FOR_POOL
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      pl->SetEvalPostActivationCallBack(MVCB);
#else
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
#endif
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, pContext7));
   }
   l++;
   //---------------------------------------------------------------
   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in = size_out;
   chn_in = chn_out;
   len_out = size_in.rows * size_in.cols * chn_in;
   chn_out = 1;
   {
      shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, pContext7, pContext8));
   }
   l++;
   //---------------------------------------------------------------  

   // Join Fully Connected Layer -----------------------------------------
   //
   len_out += len_out_branch_2;
   LayerList.push_back(make_shared<DAGJoinObj>(l, pContext6, pContext8));
   l++;
   //
   //----------------------------------------------------------------------- 
   // Branch 3 Fully Connected Layer -----------------------------------------
   //
   int len_out_branch_3 = len_out;
   // This branch will be paired with Test2.
   shared_ptr<DAGBranchObj> p_branch3 = make_shared<DAGBranchObj>(l, pContext8, pContext9, false);
   LayerList.push_back(p_branch3);
   l++;
   //
   //-----------------------------------------------------------------------    
   //--------- setup the fully connected network -------------------------------------------------------------------------
   // 
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   len_in = len_out;
   len_out = 72; // 96;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actLeakyReLU>(0.01),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, pContext8));
   }
   l++;
   //---------------------------------------------------------------  

   // Fully Connected Layer ---------------------------------------
   // Type: SoftMAX
   len_in = len_out;
   len_out = 10;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actSoftMax>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, pContext8));
   }
   l++;
   //---------------------------------------------------------------      
   // Error Layer ---------------------------------------
   // Type: LossCrossEntropy
   LayerList.push_back(make_shared<DAGErrorLayer>(l, make_shared<LossCrossEntropy>(len_out, 1), pContext8, gpError3));
   l++;
   // Branch Test Layer (Test 3) -----------------------------------------
   //
   LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch3, gpError3, restore ? 0 : l3warmup, 0.6, 0.98));
   l++;
   //
   //-----------------------------------------------------------------------

   //************************************************************************
   //              Branch 4
   //-----------------------------------------------------------------------
   // Branch 2 Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   size_in.Resize(INPUT_ROWS, INPUT_COLS);
   size_out.Resize(INPUT_ROWS, 4);
   size_kern.Resize(INPUT_ROWS, INPUT_COLS);
   kern_per_chn = 1;
   chn_in = 1;
   chn_out = kern_per_chn * chn_in;
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, kern_per_chn,
         make_unique<actReLU>(),
         //make_unique<actLinear>(), 
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, pContext10));
   }
   l++;
   //---------------------------------------------------------------

   // Average Layer ----------------------------------------------
   size_in = size_out;
   size_out.Resize(INPUT_ROWS, 1);
   chn_in = chn_out;

   {
#ifdef USE_FILTER_FOR_POOL
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      pl->SetEvalPostActivationCallBack(MVCB);
#else
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
#endif
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, pContext10));
   }
   l++;
   //---------------------------------------------------------------
   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in = size_out;
   chn_in = chn_out;
   len_out = size_in.rows * size_in.cols * chn_in;
   chn_out = 1;
   {
      shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, pContext10, pContext11));
   }
   l++;
   //---------------------------------------------------------------  

   // Join Fully Connected Layer -----------------------------------------
   //
   len_out += len_out_branch_3;
   LayerList.push_back(make_shared<DAGJoinObj>(l, pContext9, pContext11));
   l++;
   //
   //----------------------------------------------------------------------- 
   // Branch 4 Fully Connected Layer -----------------------------------------
   //
   int len_out_branch_4 = len_out;
   // This branch will be paired with Test4.
   shared_ptr<DAGBranchObj> p_branch4 = make_shared<DAGBranchObj>(l, pContext11, pContext12, false);
   LayerList.push_back(p_branch4);
   l++;
   //
   //-----------------------------------------------------------------------    
   //--------- setup the fully connected network -------------------------------------------------------------------------
   // 
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   len_in = len_out;
   len_out = 96; // 128;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actLeakyReLU>(0.01),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, pContext11));
   }
   l++;
   //---------------------------------------------------------------  
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   len_in = len_out;
   len_out = 48;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actLeakyReLU>(0.01),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, pContext11));
   }
   l++;
   //---------------------------------------------------------------  
   // Fully Connected Layer ---------------------------------------
   // Type: SoftMAX
   len_in = len_out;
   len_out = 10;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actSoftMax>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, pContext11));
   }
   l++;
   //---------------------------------------------------------------      
   // Error Layer ---------------------------------------
   // Type: LossCrossEntropy
   LayerList.push_back(make_shared<DAGErrorLayer>(l, make_shared<LossCrossEntropy>(len_out, 1), pContext11, gpError4));
   l++;
   // Branch Test Layer (Test 4) -----------------------------------------
   //
   LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch4, gpError4, restore ? 0 : l4warmup, 0.6, 0.98));
   l++;
   //
   //-----------------------------------------------------------------------
   //************************************************************************
   //              Branch 5
   //-----------------------------------------------------------------------
   // Branch 2 Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   size_in.Resize(INPUT_ROWS, INPUT_COLS);
   size_out.Resize(INPUT_ROWS, 4);
   size_kern.Resize(INPUT_ROWS, INPUT_COLS);
   kern_per_chn = 1;
   chn_in = 1;
   chn_out = kern_per_chn * chn_in;
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, kern_per_chn,
         make_unique<actReLU>(),
         //make_unique<actLinear>(), 
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, pContext13));
   }
   l++;
   //---------------------------------------------------------------

   // Average Layer ----------------------------------------------
   size_in = size_out;
   size_out.Resize(INPUT_ROWS, 1);
   chn_in = chn_out;

   {
#ifdef USE_FILTER_FOR_POOL
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      pl->SetEvalPostActivationCallBack(MVCB);
#else
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
#endif
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, pContext13));
   }
   l++;
   //---------------------------------------------------------------
   // Flattening Layer --------------------------------------------
   // Type: Flatten2D
   size_in = size_out;
   chn_in = chn_out;
   len_out = size_in.rows * size_in.cols * chn_in;
   chn_out = 1;
   {
      shared_ptr<Flatten2D> pl = make_shared<Flatten2D>(size_in, chn_in);
      LayerList.push_back(make_shared<DAGFlattenObj>(l, pl, pContext13, pContext14));
   }
   l++;
   //---------------------------------------------------------------  

   // Join Fully Connected Layer -----------------------------------------
   //
   len_out += len_out_branch_4;
   LayerList.push_back(make_shared<DAGJoinObj>(l, pContext12, pContext14));
   l++;
   //
   //----------------------------------------------------------------------- 
   // Branch 5 Fully Connected Layer -----------------------------------------
   //
   //int len_out_branch_5 = len_out;
   // This branch will be paired with Test4.
   //shared_ptr<DAGBranchObj> p_branch5 = make_shared<DAGBranchObj>(l, pContext14, pContext, false);
   //LayerList.push_back(p_branch5);
   //l++;
   //
   //-----------------------------------------------------------------------    
   //--------- setup the fully connected network -------------------------------------------------------------------------
   // 
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   len_in = len_out;
   len_out = 120; // 160;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actLeakyReLU>(0.01),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, pContext14));
   }
   l++;
   //---------------------------------------------------------------  
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   len_in = len_out;
   len_out = 64; // 160;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actLeakyReLU>(0.01),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, pContext14));
   }
   l++;
   //---------------------------------------------------------------  
   // Fully Connected Layer ---------------------------------------
   // Type: SoftMAX
   len_in = len_out;
   len_out = 10;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actSoftMax>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
         make_shared<OPTO>());
      LayerList.push_back(make_shared<DAGLayerObj>(l, pl, pContext14));
   }
   l++;
   //---------------------------------------------------------------      
   // Error Layer ---------------------------------------
   // Type: LossCrossEntropy
   LayerList.push_back(make_shared<DAGErrorLayer>(l, make_shared<LossCrossEntropy>(len_out, 1), pContext14, gpError5));
   l++;
   // Branch Test Layer (Test 5) -----------------------------------------
   //
   //LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch5, gpError5, restore ? 0 : 6 * 1000, 0.6, 0.98));
   //l++;
   //
   //-----------------------------------------------------------------------

}
*/
typedef void (*InitModelFunction)(bool);


// NOTE: Model Selector
InitModelFunction InitModel = InitLPBranchModel8;
//InitModelFunction InitLocalizerModel = InitLPBranchModel1T;

void SaveModelWeights()
{
   for (const auto& lit : LayerList) {
      int l = lit->GetID();
      lit->Save(make_shared<OWeightsCSVFile>(path, model_name + "." + to_string(l)));
      lit->Save(make_shared<OMultiWeightsBMP>(path, model_name + "." + to_string(l)));
      lit->Save(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l)));
   }
}

void ComputeCentroid(double& x, double& y, Matrix& m) 
{
      int rows = m.rows();
      int cols = m.cols();
      double total_x = 0;
      double total_y = 0;
      double weight = 0;

      // Iterate through the matrix to calculate the centroid
      for (int i = 0; i < rows; i++) {
         for (int j = 0; j < cols; j++) {
            double value = m(i,j);
            if (value > 0.0) {
               total_x += j * value;
               total_y += i * value;
               weight += value;
            }
         }
      }

      // Calculate the centroid
      if (weight > 0.0) {
         x = total_x / weight;
         y = total_y / weight;
      }
}

int getch_noblock() {
   if (_kbhit())
      return _getch();
   else
      return -1;
}

void TestModel(string dataroot, bool test_on_train = false)
{
   MNISTReader reader1;
   if (test_on_train) {
      reader1.open(dataroot + "\\train\\train-images-idx3-ubyte",
         dataroot + "\\train\\train-labels-idx1-ubyte");
   }
   else {
      reader1.open(dataroot + "\\test\\t10k-images-idx3-ubyte",
         dataroot + "\\test\\t10k-labels-idx1-ubyte");
   }


   bool bsave = false;
   bool bsingle = false;
   if (!gAutoSave){
      char c;
      cout << "Do you want to save incorrect images to disk? y/n ";
      cin >> c;
      cout << endl;
      if (c == 'y' || c == 'Y') {
         bsave = true;
      }
      cout << "Do you want to do single test? y/n ";
      cin >> c;
      cout << endl;
      if (c == 'y' || c == 'Y') {
         bsingle = true;
      }
   }


#ifdef LOGPOLAR
   LogPolarSupportMatrix lpsm = PrecomputeLogPolarSupportMatrix1(28, 28, INPUT_ROWS, INPUT_COLS+LP_WASTE);
#endif



#ifdef LOGPOLAR
   SpacialTransformer st(Size(INPUT_ROWS, INPUT_COLS), Size(INPUT_ROWS, INPUT_COLS));
#else
   SpacialTransformer st(Size(28, 28), Size(28, 28));
#endif

   ColVector T(6);

   ColVector X;

   int EN = EMap.size();
   double avg_e = 0.0;
   int count = 0;
   long correct = 0;
   vector<int> hist(EN);
   std::fill(hist.begin(), hist.end(), 0);

   vector<int> lhist(EN);
   std::fill(lhist.begin(), lhist.end(), 0);

   while (reader1.read_next()) {
      count++;
      X = reader1.data();
      ErrorContext::label = reader1.label();
      Matrix temp(28, 28);
      TrasformMNISTtoMatrix(temp, X);
      ScaleToOne(temp.data(), (int)(temp.rows() * temp.cols()));
#ifdef LOGPOLAR
      Matrix mlp(INPUT_ROWS, INPUT_COLS + LP_WASTE);
      if (bsingle) {
         double ang = 45.0 * M_PI / 180.0;
         T(0) = cos(ang); T(1) = -sin(ang); T(2) = sin(ang); T(3) = cos(ang);
         T(4) = 0.0; T(5) = 0.0;
         st.Eval(temp, temp, T);
      }
      ConvertToLogPolar(temp, mlp, lpsm);
      CMap[1].m[0] = mlp.block(0, LP_WASTE, INPUT_ROWS, INPUT_COLS);
#else
      if (bsingle) {
         double dang = 0.0;
         double xo = 0.0;
         double yo = 0.0;
         cout << "Enter angle (-90 to 90): ";  cin >> dang;  cout << endl;
         cout << "Enter col offfset: "; cin >> xo; cout << endl;
         cout << "Enter row offfset: "; cin >> yo; cout << endl;
         double ang = dang * M_PI / 180.0;
         T(0) = cos(ang);
         T(1) = -sin(ang);
         T(2) = sin(ang);
         T(3) = cos(ang);
         T(4) = -xo;
         T(5) = yo;
         CMap[1].Reset();
         CMap[1].m[0].resize(28, 28);
         st.Eval(temp, CMap[1].m[0], T);
         ScaleToOne(CMap[1].m[0].data(), (int)(CMap[1].m[0].rows() * CMap[1].m[0].cols()));
         temp = CMap[1].m[0];
      }
      else {
         CMap[1].m[0] = temp;
      }
#endif

      //*******************************************************
      //            Forward Pass
      size_t fwd_count = 0;
      do {
         LayerList[fwd_count]->Eval();
         fwd_count++; // The loop exits at last count + 1.  This is by design.
      } while ((!gpExit->stop) && (fwd_count < LayerList.size()));
      gpExit->stop = false;
      //*******************************************************

      bool ecorrect = false;
      int stage = -1;
      int last_stage = 0;
      double class_max = 0.0;
      for (pair<const int, ErrorContext>& ep : EMap) {
         stage++;
         if (ep.second.touched && ep.second.correct) {
            last_stage = stage;
            ecorrect = true;
            if (class_max < ep.second.class_max) {
               class_max = ep.second.class_max;
            }
            hist[stage] += 1;
            //break;
         }
         ep.second.touched = false;
      }

      if (ecorrect) {
         correct++;
         lhist[last_stage] += 1;
         
         if (bsingle) {
            //if (EMap[1].correct && EMap[1].class_max >= 0.9) {
               int nl = GetLabel(ErrorContext::label);
               cout << "number " << nl << " correct on stage: " << last_stage
                     << "\nclass max: " << class_max 
                     << "\nrow center: " << CMap[200].v[0] << " col center: " << CMap[200].v[1]
                     << "\nEstimated correction angle: " << -CMap[0].v[0] * 180.0 / 16.0
                     << "\nDo you want to save it?  Enter Y - yes | N - no | E - exit.";
               char c;
               cin >> c;
               cout << endl;
               if (c == 'Y' || c == 'y') {
                  double astp = M_PI / 16.0;
                  double ang = -CMap[0].v[0] * astp;  // angle in radians
                  T(0) = cos(ang); T(1) = -sin(ang); T(2) = sin(ang); T(3) = cos(ang);
                  T(4) = CMap[200].v[1] - 13.5; 
                  T(5) = CMap[200].v[0] - 13.5;
                  MakeMatrixImage(path + "\\org." + to_string(count) + ".bmp", temp);
                  st.Eval(temp, temp, T);
                  MakeMatrixImage(path + "\\fix." + to_string(count) + ".bmp", temp);
               }
               else if (c == 'E' || c == 'e') {
                  exit(0);
               }
            //}
         }
      }
      else if(bsave) {
         int nl = GetLabel(ErrorContext::label);
         MakeMatrixImage(path + "\\wrong.L" + to_string(nl) + "." + to_string(count) + ".bmp", temp);
      }
      
      //if (bsingle) {
      //   cout << (gpError1->correct ? "correct" : "incorrect") << endl << "T: " << pContext3->v.transpose() << endl;
      //}

   }
   std::cout << "total correct: " << correct << endl
      << "Histogram (all correct branch 1 through 5)" << endl;
   for (auto v : hist) { cout << v << " | "; }
   cout << "\nHistogram (last correct branch 1 through 5)" << endl;
   for (auto v : lhist) { cout << v << " | "; }   
   cout << "\nHistogram (percent correct branch 1 through 5)" << endl;
   for (int i = 0; i < EN; i++) { cout << EMap[i].PctCorrect() << " | "; }
   cout << endl;


}

Matrix MakeFatFilter;
bool MakeFatInit = false;

void MakeFat(Matrix& g)
{
   int rows = g.rows();
   int cols = g.cols();
   Matrix o(rows, cols);

   if (!MakeFatInit) {
      MakeFatInit = true;
      MakeFatFilter.resize(rows, cols);

      double r_c = (double)rows / 2.0;
      double c_c = (double)cols / 2.0;
      double sig = 2.0;
      double norm = 1.0 / (2.0 * M_PI * sig);

      for (int r = 0; r < rows; r++) {
         for (int c = 0; c < cols; c++) {
            double rr = r - r_c;
            double cc = c - c_c;
            double e = (rr * rr + cc * cc) / (2.0 * sig);
            MakeFatFilter(r, c) = norm * std::exp(-e);
         }
      }
   }

   fft2convolve(g, MakeFatFilter, o, true, true);

   double t = 0.15;
   for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
         if (o(r, c) >= t) {
            g(r, c) = 1.0;
         }
         else {
            g(r, c) = 0.0;
         }
      }
   }
}


// Use this function to make a list of index numbers to correctly
// identified images.
void MakeListOfCorrect(string dataroot)
{
   MNISTReader reader1(dataroot + "\\train\\train-images-idx3-ubyte",
      dataroot + "\\train\\train-labels-idx1-ubyte");

   InitModel(true);

#ifdef LOGPOLAR
   LogPolarSupportMatrix lpsm = PrecomputeLogPolarSupportMatrix1(28, 28, INPUT_ROWS, INPUT_COLS + LP_WASTE);
#endif

   ColVector X;
   const int nsave = 30000;
   Eigen::VectorXi indexes(nsave);
   int count = 0;
   long index = 0;

   while (reader1.read_next()) {
      X = reader1.data();
      ErrorContext::label = reader1.label();
      Matrix temp(28, 28);
      TrasformMNISTtoMatrix(temp, X);
      ScaleToOne(temp.data(), (int)(temp.rows() * temp.cols()));
#ifdef LOGPOLAR
      Matrix mlp(INPUT_ROWS, INPUT_COLS + LP_WASTE);
      ConvertToLogPolar(temp, mlp, lpsm);
      CMap[1].m[0] = mlp.block(0, LP_WASTE - 1, INPUT_ROWS, INPUT_COLS);
#else
      CMap[1].m[0] = temp;
#endif

      //*******************************************************
      //            Forward Pass
      size_t fwd_count = 0;
      do {
         LayerList[fwd_count]->Eval();
         fwd_count++; // The loop exits at last count + 1.  This is by design.
      } while ((!gpExit->stop) && (fwd_count < LayerList.size()));
      gpExit->stop = false;
      //*******************************************************

      if (EMap[1].correct && EMap[1].class_max >= 0.95) {
      //if (gpError1->correct || gpError2->correct || gpError3->correct || gpError4->correct || gpError5->correct) {
         indexes(index) = count;
         if (index == (nsave - 1)) {
            break;
         }
         index++;
      }

      count++;
   }
   std::cout << "Complete. " << index << " items saved." << endl;

   PutMatrix(indexes, path + "\\localizer_training.dat");

   //std::cout << "test storage" << endl;

   //GetMatrix(indexes, path + "\\localizer_training.dat");
   //cout << indexes;
}

void Train(int nloop, string dataroot, double eta, int load, double lower_limit = -1.0, double upper_limit = -1.0)
{
cout << dataroot << endl;
#ifdef RETRY
   cout << "NOTE: There is auto-repeate code running!  retry = " << RETRY << endl;
#endif

#ifdef LOGPOLAR
   cout << "Running Log-Polar samples." << endl;
#endif
#ifdef SGD
   cout << "Stocastic decent is on." << endl;
#else
   cout << "Batch grad descent is in use." << endl;
#endif
#ifdef USE_FFT
   cout << "Using FFT convolution." << endl;
#else
   cout << "Using linear convolution." << endl;
#endif // USE_FFT

   MNISTReader reader(dataroot + "\\train\\train-images-idx3-ubyte",
      dataroot + "\\train\\train-labels-idx1-ubyte");

   InitModel(load > 0 ? true : false);

   ErrorOutput err_out(path, model_name);
   ErrorOutput err_out1(path, model_name + "_1");
   ErrorOutput err_out2(path, model_name + "_2");
   ErrorOutput err_out3(path, model_name + "_3");
   ErrorOutput err_out4(path, model_name + "_4");
   //ClassifierStats stat_class;
   //ClassifierStats stat_angle_class;


   const int reader_batch = 1000;  // Should divide into 60K
   const int batch = 100; // Should divide evenly into reader_batch
   const int batch_loop = 11;

   std::random_device rd;     // only used once to initialise (seed) engine
   std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
   std::uniform_int_distribution<int> uni(0, reader_batch - 1); // guaranteed unbiased

   // One of the LP papers shows a better way to compute resampling corrdinates.
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

         for (auto lli : LayerList) { lli->StashWeights(); }
         //----------------------------------------------------------
         e = 0;
         avg_n = 1;
         int retry = 0;
         int n = 0;
         int b = 0;
         int t = 0;
         unsigned long correct1 = 0;
         unsigned long correct2 = 0;
         unsigned long correct3 = 0;
         unsigned long correct4 = 0;
         unsigned long early_stop = 0;
         while (b < batch) {
            if (retry == 0) {
               n = uni(rng); // Select a random entry out of the batch.
               b++;
            }
            
            // Total try counter.
            t++;

            Matrix temp(28, 28);

            ErrorContext::label = dl[n].y;
            TrasformMNISTtoMatrix(temp, dl[n].x);
            ScaleToOne(temp.data(), (int)(temp.rows() * temp.cols()));

            #ifdef LOGPOLAR
               CMap[1].Reset();
               CMap[1].m[0].resize(INPUT_ROWS, INPUT_COLS);
               ConvertToLogPolar(temp, CMap[1].m[0], lpsm);
            #else
               CMap[1].m[0] = temp;
            #endif
            // REVIEW: There is something wrong with this Matrix manipulator!
            //for (MatrixManipulator mm(temp, INPUT_ROWS, INPUT_COLS, true); !mm.isDone(); mm.next())
            //{
               //ColVector ay = mm.AngleLabel();
               //pContext1->Reset();
               //pContext1->m[0] = mm.get();
               
               //*******************************************************
               //            Forward Pass
               size_t fwd_count = 0;
               do{
                  LayerList[fwd_count]->Eval();
                  fwd_count++; // The loop exits at last count + 1.  This is by design.
               } while ( (!gpExit->stop) && (fwd_count < LayerList.size()) );
               gpExit->stop = false;
               //*******************************************************


               //double le = loss->Eval(pContext2->v, dl[n].y);
               double le = EMap[2].error;
               //double lb = loss_branch->Eval(pContext4->v, ay);
               if (isnan(le)) {
                  for (auto lli : LayerList) { lli->ApplyStash(); }

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

               for (pair<const int, ErrorContext>& ep : EMap) {
                  if (ep.second.correct) {
                     switch (ep.first) {
                     case 1: correct1++; break;
                     case 2: correct2++; break;
                     case 3: correct3++; break;
                     case 4: correct4++; break;
                     }
                  }
               }

               //*******************************************************
               //            Backward Pass
               //
               // NOTE: To stop training on an early branch (like branch 1) the 
               // "stop" mechanism can be used.  The DAGError object could evaulate
               // conditions and set the "stop" condition just like it does during the
               // forward pass.  In this way we can cleanly stop training on early
               // branchs if that is desired.
               //
               while( (!gpExit->stop) && (fwd_count > 0)){
                  fwd_count--;
                  LayerList[fwd_count]->BackProp();
                  // Only SGD for now.
                  LayerList[fwd_count]->Update(eta);
               }
               if (fwd_count > 0) {
                  early_stop++;
               }
               gpExit->stop = false;
               //*******************************************************
            //}
//#ifdef SGD
//            // This is stoastic descent.  It is inside the batch loop.
//            for (auto lit : LayerList) {
//               lit->Update(eta);
//            }
//#endif
         }

         // if not defined
//#ifndef SGD
//         //eta = (1.0 / (1.0 + 0.001 * loop)) * eta;
//         for (auto lit : LayerList) {
//            lit->Update(eta);
//         }
//#endif

         double ac1 = 100.0 * (double)correct1 / t;
         double ac2 = (t - correct1) > 0 ? 100.0 * (double)correct2 / (t - correct1) : 100.0;
         double ac3 = (t - correct1 - correct2) > 0 ? 100.0 * (double)correct3 / (t - correct1 - correct2) : 100.0;
         double ac4 = (t - correct1 - correct2 - correct3) > 0 ? 100.0 * (double)correct4 / (t - correct1 - correct2 - correct3) : 100.0;

         err_out.Write(correct1 + correct2 + correct3);
         err_out1.Write(ac1);
         err_out2.Write(ac2);
         err_out3.Write(ac3);
         err_out4.Write(ac4);
         cout << "count: " << loop << "\tearly stops: " << early_stop
            << "\t\terror:" << left << setw(9) << std::setprecision(4) << e 
            << "\t %1: " << left << setw(5) << std::setprecision(2) << std::fixed << ac1
            << "\t %2: " << left << setw(5) << std::setprecision(2) << std::fixed << ac2
            << "\t %3: " << left << setw(5) << std::setprecision(2) << std::fixed << ac3
            << "\t %4: " << left << setw(5) << std::setprecision(2) << std::fixed << ac4
            << "\ttotal correct: " << correct1 + correct2 + correct3 + correct4 << endl;
      }

      cout << "eta: " << setw(9) << std::setprecision(5) << std::fixed << eta << endl;
   }

TESTJMP:

   TestModel(dataroot);

   std::cout << "Save? y/n:  ";
   char c;
   std::cin >> c;
   if (c == 'y') {
      SaveModelWeights();
   }
}

int g_augment_type = 4;
bool g_transformer = false;

void Train2(int nloop, string dataroot, double eta, int load, int eta2 )
{
   cout << "Train2" << endl;
   cout << dataroot << endl;
#ifdef RETRY
   cout << "NOTE: There is auto-repeate code running!  retry = " << RETRY << endl;
#endif

#ifdef LOGPOLAR
   cout << "Running Log-Polar samples." << endl;
#endif
#ifdef SGD
   cout << "Stocastic decent is on." << endl;
#else
   cout << "Batch grad descent is in use." << endl;
#endif
#ifdef USE_FFT
   cout << "Using FFT convolution." << endl;
#else
   cout << "Using linear convolution." << endl;
#endif // USE_FFT
#ifdef USE_FILTER_FOR_POOL
   cout << "Using convolution filter in place of Average Pool." << endl;
#else
   cout << "Using Average or Max Pool." << endl;
#endif

   MNISTReader reader(dataroot + "\\train\\train-images-idx3-ubyte",
      dataroot + "\\train\\train-labels-idx1-ubyte");

   InitModel(load > 0 ? true : false);

   ErrorOutput err_out(path, model_name);
   map<int, ErrorOutput> err_out_map;
   for (int i = 0; i < EMap.size(); i++) {
      err_out_map.emplace(i, ErrorOutput(path, model_name + "_" + std::to_string(i + 1)));
   }

   StatRunningAverage pose_error(path, model_name, "terr", 1000);
   StatRunningAverage center_estimate_r(path, model_name, "center_r", 1000);
   StatRunningAverage center_estimate_c(path, model_name, "center_c", 1000);

   const int reader_batch = 1000;  // Should divide into 60K
   const int batchs_per_epoch = 60;
   const double astp = 2.0 * M_PI / 32.0;

   double e = 0;

   struct Sample_Pair {
      Matrix m;
      ColVector y;
      Sample_Pair(){}
      Sample_Pair(Matrix _m, ColVector _y) : m(_m), y(_y) {}
   };

   vector< Sample_Pair> samples;
   Matrix temp(28, 28);

   // For debug.
   //OWeightsCSVFile fcsv(path, "lpout");
   
   cout << "Loading the training data." << endl;
   for (int k = 0; k < batchs_per_epoch; k++) {
      cout << "*";
      MNISTReader::MNIST_list dl = reader.read_batch(reader_batch);
      for (MNISTReader::MNIST_Pair& mp : dl) {
         TrasformMNISTtoMatrix(temp, mp.x);
         ScaleToOne(temp.data(), (int)(temp.rows() * temp.cols()));
         samples.push_back(Sample_Pair(temp, mp.y));
      }
   }
   cout << endl;
   
   /*
   cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!! Loading the localizer training indexes. !!!!!!!!!!!!!!!" << endl;
   Eigen::VectorXi indexes;
   GetMatrix(indexes, path + "\\localizer_training.dat");

   cout << "Loading the training data." << endl;
   int minst_counter = 0;
   int index_counter = 0;
   bool got_one = false;
   for (int k = 0; k < batchs_per_epoch; k++) {
      cout << "*";
      MNISTReader::MNIST_list dl = reader.read_batch(reader_batch);
      for (MNISTReader::MNIST_Pair& mp : dl) {
         if (minst_counter == indexes[index_counter]) {
            TrasformMNISTtoMatrix(temp, mp.x);
            ScaleToOne(temp.data(), (int)(temp.rows() * temp.cols()));
            samples.push_back(Sample_Pair(temp, mp.y));

            index_counter++;
            if (index_counter == indexes.size()) {
               cout << "Full" << endl;
               goto FULL;
            }
         }
         minst_counter++;
      }
   }

FULL:
*/
   std::random_device rd;     // only used once to initialise (seed) engine
   std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
   std::uniform_int_distribution<int> uni(0, samples.size() - 1);
   std::uniform_int_distribution<int> typ(0, 3);
   std::uniform_int_distribution<int> shift(-3, 3);
   std::uniform_int_distribution<int> shift1(-3, 3);
   std::uniform_real_distribution<double> rotate(-M_ANG, M_ANG);
   std::uniform_real_distribution<double> scalex(0.8, 1.2);
   std::uniform_real_distribution<double> scaley(0.8, 1.2);
   std::uniform_int_distribution<int> fat(0, 100);


   //SpacialTransformer st(Size(28, 28), Size(28, 28));
   SampleMatrix sm;
   samplerBiLinear smp(Size(28, 28), Size(28, 28));
   gridAffine grd(Size(28, 28), Size(28, 28));
   ColVector T(6);

#ifdef LOGPOLAR
   //LogPolarSupportMatrix lpsm = PrecomputeLogPolarSupportMatrix(28, 28, INPUT_ROWS, INPUT_COLS + LP_WASTE);
   //cout << "Using origional LP Support Matrix builder" << endl;
   LogPolarSupportMatrix lpsm = PrecomputeLogPolarSupportMatrix1(28, 28, INPUT_ROWS, INPUT_COLS + LP_WASTE);
   Matrix mlp(INPUT_ROWS, INPUT_COLS + LP_WASTE);
   //cout << "Using LP Support Matrix builder from the Polar Transformaer paper with Alpha = 0.5." << endl;

   //auto logxform = [&temp, &mlp, &st, &lpsm](Matrix& s, Matrix& t, ColVector& T) {
   auto logxform = [&temp, &mlp, &sm, &smp, &grd, &lpsm](Matrix& s, Matrix& t, ColVector& T) {
         //st.Eval(s, temp, T);
      grd.Eval(sm, T);
      smp.Eval(s, temp, sm);
      ConvertToLogPolar(temp, mlp, lpsm);
      t = mlp.block(0, LP_WASTE, INPUT_ROWS, INPUT_COLS);
   };
#else
   CMap[1].Reset();
   CMap[1].m[0].resize(28, 28);
#endif

   int n = 0;
   int b = 0;
   int t = 0;
   unsigned long total_correct = 0;

   unsigned long early_stop = 0;

   unsigned int total_wrong = 0;

   for (int loop = 0; loop < nloop; loop++) {
      if (eta2 > 0 && loop > 0  && !(loop % eta2)) {
         eta /= sqrt( (double)(loop) / (double)eta2 );
      }

      e = 0;
      b = 0;

      while (b < samples.size()) {
         b++;
         n = uni(rng); // Select a random entry out of the batch.

         // Total try counter.
         t++;

         //*******************************************************
         //            Forward Pass

         double ang = 0.0;
         double sx = 1.0;
         double sy = 1.0;
         sx = scalex(rng);
         sy = scaley(rng);
         ang = rotate(rng);
         T(0) = sx * cos(ang);
         T(1) = sy * (-sin(ang));
         T(2) = sx * sin(ang);
         T(3) = sy * cos(ang);
         T(4) = shift(rng);
         T(5) = shift1(rng);

#ifdef LOGPOLAR
         logxform(samples[n].m, CMap[1].m[0], T);
         MakeFat(CMap[1].m[0]);
#else
         {
            vector_of_matrix vom;
            vom.push_back(samples[n].m);
            if (fat(rng) < 20.0) {
               MakeFat(vom[0]);
            }
            grd.Eval(sm, T);
            smp.Eval(vom, CMap[1].m, sm);
         }
#endif

         ScaleToOne(CMap[1].m[0].data(), (int)(CMap[1].m[0].rows()* CMap[1].m[0].cols()));
         Matrix theInput = CMap[1].m[0];

         ErrorContext::label = samples[n].y;
         size_t fwd_count = 0;
         do {
            LayerList[fwd_count]->Eval();
            fwd_count++; // The loop exits at last count + 1.  This is by design.
         } while ((!gpExit->stop) && (fwd_count < LayerList.size()));
         gpExit->stop = false;
         //*******************************************************

         // NOTE: Can not remove the counting here because sometimes two braches correctly 
         //       identify the same image.  If the image is identified correctly but
         //       the probability is less than the threshold set in the DAGExit the
         //       image will be forwarded to the next branch.
         bool correct = false;
         for (pair<const int, ErrorContext>& ep : EMap) {
            if (ep.second.correct) {
               correct = true;

               break;
            }
         }
         if (correct) { total_correct++; }
         else {
            //total_wrong++;
            //MakeMatrixImage(path + "\\wrong." + to_string(total_wrong) + ".bmp", theInput);
            //if (total_wrong > 29) {
            //   exit(0);
            //}
         }


         //*******************************************************
         //            Backward Pass
         //
         while ((!gpExit->stop) && (fwd_count > 0)) {
            fwd_count--;
            LayerList[fwd_count]->BackProp();
            // Only SGD for now.
            LayerList[fwd_count]->Update(eta);
         }
         if (fwd_count > 0) {
            early_stop++;
         }
         gpExit->stop = false;
         //*******************************************************

         // NOTE: Put back for Transformer test !!!!!!!!!!!!!!!!!!!!!!!
         //if (g_transformer) {
         //   gpError2->SetStatus(true, std::abs((ang / astp) - pContext3->v[0]));
         //}

         // REVIEW: Using the average pose value. 
         //pose_error.Push( std::abs((ang / astp) - CMap[0].v[0]) );
         // REVIEW: This currently only works for InitModel1T - use CMap[5]
         //                                       InitModel3T - use CMap[200]
         center_estimate_r.Push( CMap[200].v[0] );
         center_estimate_c.Push( CMap[200].v[1] );

         if (t >= 1000) {
            err_out.Write(total_correct);
            int i = 0;
            for (auto& p : err_out_map) {
               p.second.Write(EMap[i].PctCorrect());
               i++;
            }

            //pose_error.write();

            center_estimate_r.write();
            center_estimate_c.write();

            cout << "ep,loop:" << left << setw(3) << loop << ", " << left << setw(5) << b << "\tearly stops: " << left << setw(4) << early_stop;
            for (pair<const int, ErrorContext>& ep : EMap) {
               cout << "\t %" << ep.first << ": " << left << setw(5) << std::setprecision(2) << std::fixed << ep.second.PctCorrect();
            }
            cout << "\ttotal correct: " << total_correct << endl;
            t = 0;

            total_correct = 0;

            for (pair<const int, ErrorContext>& ep : EMap) {
               ep.second.Reset();
            }

            early_stop = 0;

            char c = getch_noblock();
            if (c == 'x' || c == 'X') {
               goto JMP;
            }
         }
      }

   }

   JMP:


   // Ask if Transformer should be trained.
   // If so take a forward pass and record index of correct id,
   // use these ids to train the transformer.
   // Pass the Sapmples vector and Correct List to the transformer training algorithm or pass a new vector with references
   // to the training list.
   // The tranformer model must be able to run the underlaying model without the transformer in order
   // to generate a base of good training samples.

   TestModel(dataroot);

   if (gAutoSave) {
      SaveModelWeights();
   }
   else {
      std::cout << "Save? y/n:  ";
      char c;
      std::cin >> c;
      if (c == 'y') {
         SaveModelWeights();
      }
   }
}

void Train3(int nloop, string dataroot, double eta, int load)
{
   cout << dataroot << endl;
#ifdef RETRY
   cout << "NOTE: There is auto-repeate code running!  retry = " << RETRY << endl;
#endif

#ifdef LOGPOLAR
   cout << "Running Log-Polar samples." << endl;
#endif
#ifdef SGD
   cout << "Stocastic decent is on." << endl;
#else
   cout << "Batch grad descent is in use." << endl;
#endif
#ifdef USE_FFT
   cout << "Using FFT convolution." << endl;
#else
   cout << "Using linear convolution." << endl;
#endif // USE_FFT


   MNISTReader reader(dataroot + "\\train\\train-images-idx3-ubyte",
      dataroot + "\\train\\train-labels-idx1-ubyte");

   InitModel(load > 0 ? true : false);

   cout << "Model Branches: " << gModelBranches << endl;
   if (gModelBranches < 3) {
      cout << "Train3 does not work with less than 3 branches." << endl;
      return;
   }
   ErrorOutput err_out(path, model_name);
   ErrorOutput err_out1(path, model_name + "_1");
   ErrorOutput err_out2(path, model_name + "_2");
   ErrorOutput err_out3(path, model_name + "_3");
   ErrorOutput err_out4(path, model_name + "_4");
   //ClassifierStats stat_class;
   //ClassifierStats stat_angle_class;

   size_t fwd_count = 0;

   const int reader_batch = 1000;  // Should divide into 60K
   const int batchs_per_epoch = 60;

   // One of the LP papers shows a better way to compute resampling corrdinates.
   LogPolarSupportMatrix lpsm = PrecomputeLogPolarSupportMatrix(28, 28, INPUT_ROWS, INPUT_COLS);

   double e = 0;
   int avg_n;

   struct Sample_Pair {
      Matrix m;
      ColVector y;
      Sample_Pair() {}
      Sample_Pair(Matrix _m, ColVector _y) : m(_m), y(_y) {}
   };

   vector<Sample_Pair> samples1, samples2, samples3, samples4;
   Matrix temp(28, 28);

   cout << "Loading the training data." << endl;
   for (int k = 0; k < batchs_per_epoch; k++) {
      cout << "*";
      MNISTReader::MNIST_list dl = reader.read_batch(reader_batch);
      for (MNISTReader::MNIST_Pair& mp : dl) {
         TrasformMNISTtoMatrix(temp, mp.x);
         ScaleToOne(temp.data(), (int)(temp.rows() * temp.cols()));
         Matrix mlp(INPUT_ROWS, INPUT_COLS);
#ifdef LOGPOLAR
         ConvertToLogPolar(temp, mlp, lpsm);
#else
         mlp = temp;
#endif

         //*******************************************************
         //            Forward Pass
         CMap[1].m[0] = mlp;
         ErrorContext::label = mp.y;
         fwd_count = 0;
         do {
            LayerList[fwd_count]->Eval();
            fwd_count++; // The loop exits at last count + 1.  This is by design.
         } while ((!gpExit->stop) && (fwd_count < LayerList.size()));
         gpExit->stop = false;
         //*******************************************************

         if (EMap[1].correct) {
            samples1.emplace_back(mlp, mp.y);
         }
         else if (EMap[2].correct) {
            samples2.emplace_back(mlp, mp.y);
         }
         else if (EMap[3].correct) {
            samples2.emplace_back(mlp, mp.y);
            //samples3.emplace_back(mlp, mp.y);
         }
         else{
            if (gModelBranches == 3) {
               samples2.emplace_back(mlp, mp.y);
               //samples3.emplace_back(mlp, mp.y);
            }
            else {
               samples4.emplace_back(mlp, mp.y);
            }
         }
      }
   }
   cout << "\n" << "Sample sets. 1: " << samples1.size() << " 2: " << samples2.size() << " 3: " << samples3.size() << " 4: " << samples4.size() <<  endl;
   //***************** End Data Load ***************************************

   std::random_device rd;     // only used once to initialise (seed) engine
   std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)

   std::uniform_int_distribution<int> uni1(0, samples1.size() - 1); // guaranteed unbiased
   std::uniform_int_distribution<int> uni2(0, samples2.size() - 1); // guaranteed unbiased
   std::uniform_int_distribution<int> uni3(0, samples3.size() - 1); // guaranteed unbiased

   std::uniform_int_distribution<int> uni4(0, (gModelBranches == 3 ? samples3.size() : samples4.size()) - 1); // guaranteed unbiased

   std::random_device rd_mixer;
   std::mt19937 rng_mixer(rd_mixer());
   std::uniform_int_distribution<int> uni_mixer(1, 100);

   int b = 0;
   int t = 0;
   unsigned long correct1 = 0;
   unsigned long correct2 = 0;
   unsigned long correct3 = 0;
   unsigned long correct4 = 0;
   unsigned long early_stop = 0;

   int set_size = 0;

   // REIVEW: 2 or 3 or 4 !!!!!!
   set_size = samples2.size();

   for (int loop = 0; loop < nloop; loop++) {

      e = 0;
      avg_n = 1;
      b = 0;

      while (b < set_size) {
         bool b_mixed = false;
         int mixer = uni_mixer(rng_mixer);
         if (mixer <= 5) {
            b_mixed = true;
            int n = uni1(rng);
            CMap[1].m[0] = samples1[n].m;
            ErrorContext::label = samples1[n].y;
         }
         else {
            // REIVEW: 2 or 3 or 4 !!!!!!
            int n = uni2(rng);
            b++;

            // Total try counter.
            t++;
            // REIVEW: 2 or 3 or 4 !!!!!!
            CMap[1].m[0] = samples2[n].m;
            ErrorContext::label = samples2[n].y;
         }
         //*******************************************************
         //            Forward Pass
         fwd_count = 0;
         if (b_mixed) {
            // If a upper branch sample is mixed in we want to push it
            // through the network.
            do {
               LayerList[fwd_count]->Eval();
               fwd_count++;
            } while ( fwd_count < LayerList.size() );
         }
         else {
            do {
               LayerList[fwd_count]->Eval();
               fwd_count++; // The loop exits at last count + 1.  This is by design.
            } while ((!gpExit->stop) && (fwd_count < LayerList.size()));

            double le = EMap[2].error;

            double a = 1.0 / (double)(avg_n);
            avg_n++;
            double d = 1.0 - a;
            e = a * le + d * e;

            for (pair<const int, ErrorContext>& ep : EMap) {
               if (ep.second.correct) {
                  switch (ep.first) {
                     case 1: correct1++; break;
                     case 2: correct2++; break;
                     case 3: correct3++; break;
                     case 4: correct4++; break;
                  }
               }
            }
         }
         gpExit->stop = false;
         //*******************************************************


         //*******************************************************
         //            Backward Pass
         //
         while ((!gpExit->stop) && (fwd_count > 0)) {
         //while ((!gpExit->stop) && (fwd_count > 10)) {  // !!!!!!!!!!!!!   TESTING !!!!!!!!!!!!!!!!!!!!!!
            fwd_count--;
            LayerList[fwd_count]->BackProp();
            // Only SGD for now.
            LayerList[fwd_count]->Update(eta);
         }
         if (!b_mixed && fwd_count > 0) {
            early_stop++;
         }
         gpExit->stop = false;
         //*******************************************************

         if (t >= 1000) {
            double ac1 = 100.0 * (double)correct1 / t;
            double ac2 = (t - correct1) > 0 ? 100.0 * (double)correct2 / (t - correct1) : 100.0;
            double ac3 = (t - correct1 - correct2) > 0 ? 100.0 * (double)correct3 / (t - correct1 - correct2) : 100.0;
            double ac4 = (t - correct1 - correct2 - correct3) > 0 ? 100.0 * (double)correct4 / (t - correct1 - correct2 - correct3) : 100.0;

            err_out.Write(correct1 + correct2 + correct3 + correct4);
            err_out1.Write(ac1);
            err_out2.Write(ac2);
            err_out3.Write(ac3);
            err_out4.Write(ac4);

            cout << "ep,loop:" << left << setw(3) << loop << ", " << left << setw(5) << b << "\tearly stops: " << left << setw(4) << early_stop
               //<< "\terror:" << left << setw(9) << std::setprecision(4) << e
               << "\t %1: " << left << setw(5) << std::setprecision(2) << std::fixed << ac1
               << "\t %2: " << left << setw(5) << std::setprecision(2) << std::fixed << ac2
               << "\t %3: " << left << setw(5) << std::setprecision(2) << std::fixed << ac3
               << "\t %4: " << left << setw(5) << std::setprecision(2) << std::fixed << ac4
               << "\ttotal correct: " << correct1 + correct2 + correct3 + correct4 << endl;
            t = 0;
            correct1 = 0;
            correct2 = 0;
            correct3 = 0;
            correct4 = 0;
            early_stop = 0;
         }
      }

   }

   TestModel(dataroot);

   std::cout << "Save? y/n:  ";
   char c;
   std::cin >> c;
   if (c == 'y') {
      SaveModelWeights();
   }
}

void TrainLocalizer(int nloop, string dataroot, double eta, int load)
{
   cout << dataroot << endl;
#ifdef RETRY
   cout << "NOTE: There is auto-repeate code running!  retry = " << RETRY << endl;
#endif

#ifdef LOGPOLAR
   cout << "Running Log-Polar samples." << endl;
#endif
#ifdef SGD
   cout << "Stocastic decent is on." << endl;
#else
   cout << "Batch grad descent is in use." << endl;
#endif
#ifdef USE_FFT
   cout << "Using FFT convolution." << endl;
#else
   cout << "Using linear convolution." << endl;
#endif // USE_FFT
#ifdef USE_FILTER_FOR_POOL
   cout << "Using convolution filter in place of Average Pool." << endl;
#else
   cout << "Using Average or Max Pool." << endl;
#endif

   MNISTReader reader(dataroot + "\\train\\train-images-idx3-ubyte",
      dataroot + "\\train\\train-labels-idx1-ubyte");

   InitModel(load > 0 ? true : false);

   ErrorOutput err_out(path, model_name);
   ErrorOutput err_out1(path, model_name + "_1");
   ErrorOutput err_out2(path, model_name + "_2");
   ErrorOutput err_out3(path, model_name + "_3");
   ErrorOutput err_out4(path, model_name + "_4");
   ErrorOutput err_out5(path, model_name + "_5");

   const int reader_batch = 1000;  // Should divide into 60K
   const int batchs_per_epoch = 10; // 60;

   double e = 0;

   struct Sample_Pair {
      Matrix m;
      ColVector y;
      Sample_Pair() {}
      Sample_Pair(Matrix _m, ColVector _y) : m(_m), y(_y) {}
   };

   vector< Sample_Pair> samples;
   Matrix temp(28, 28);

   cout << "Loading the localizer training indexes." << endl;
   Eigen::VectorXi indexes;
   GetMatrix(indexes, path + "\\localizer_training.dat");

   cout << "Loading the training data." << endl;
   int minst_counter = 0;
   int index_counter = 0;
   for (int k = 0; k < batchs_per_epoch; k++) {
      cout << "*";
      MNISTReader::MNIST_list dl = reader.read_batch(reader_batch);
      for (MNISTReader::MNIST_Pair& mp : dl) {
         if (minst_counter == indexes[index_counter]) {
            TrasformMNISTtoMatrix(temp, mp.x);
            ScaleToOne(temp.data(), (int)(temp.rows() * temp.cols()));
            samples.push_back(Sample_Pair(temp, mp.y));
            index_counter++;
            if (index_counter == indexes.size()) {
               goto FULL;
            }
         }
         minst_counter++;
      }
   }

   FULL:

   cout << endl;

   std::random_device rd;     // only used once to initialise (seed) engine
   std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
   std::uniform_int_distribution<int> uni(0, samples.size() - 1);
   std::uniform_int_distribution<int> typ(0, 3);
   std::uniform_int_distribution<int> shift(-2, 2);
   //std::uniform_real_distribution<double> rotate(-M_PI_4, M_PI_4);
   std::uniform_real_distribution<double> rotate(-M_ANG, M_ANG);

   SpacialTransformer st(Size(28, 28), Size(28, 28));
   ColVector T(6);

#ifdef LOGPOLAR
   //LogPolarSupportMatrix lpsm = PrecomputeLogPolarSupportMatrix(28, 28, INPUT_ROWS, INPUT_COLS + LP_WASTE);
   //cout << "Using origional LP Support Matrix builder" << endl;
   LogPolarSupportMatrix lpsm = PrecomputeLogPolarSupportMatrix1(28, 28, INPUT_ROWS, INPUT_COLS + LP_WASTE);
   cout << "Using LP Support Matrix builder from the Polar Transformaer paper with Alpha = 0.5." << endl;   Matrix mlp(INPUT_ROWS, INPUT_COLS + LP_WASTE);

   auto logxform = [&temp, &mlp, &st, &lpsm](Matrix& s, Matrix& t, ColVector& T) {
      st.Eval(s, temp, T);
      ConvertToLogPolar(temp, mlp, lpsm);
      t = mlp.block(0, LP_WASTE, INPUT_ROWS, INPUT_COLS);
   };
#endif

   int n = 0;
   int b = 0;
   int t = 0;
   unsigned long correct1 = 0;
   unsigned long correct2 = 0;
   unsigned long correct3 = 0;
   unsigned long correct4 = 0;
   unsigned long correct5 = 0;
   unsigned long early_stop = 0;

#ifdef TRAIN_TRANSFORMER
   ErrorContext::label.resize(1);
   const double astp = 2.0 * M_PI / 32.0;
#endif
#ifdef TRAIN_NET
   CMap[3].v.resize(1);
   CMap[3].v[0] = 0.0;
#endif


   for (int loop = 0; loop < nloop; loop++) {

      e = 0;
      b = 0;

      while (b < samples.size()) {
         b++;
         n = uni(rng); // Select a random entry out of the batch.

         //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         // REVIEW: Transform sanity check test.
         //n = 2;
         //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

         // Total try counter.
         t++;

         //*******************************************************
         //            Forward Pass

         double ang = 0.0;
#ifdef TRAIN_NET
         switch (4 /*typ(rng)*/) {
#else
         switch ( 1 ) {
#endif
         case 1: // Rotate

            // Row angle bin size is (360 / 32).  That's about 11 deg.

            ang = rotate(rng);
            //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            // REVIEW: Transform sanity check test.
            //         Once at zero once at M_PI_4
            //ang = 0.0;
            //ang = M_PI_4 * 0.333333333;
            //ang = M_PI_4;
            //pContext3->v[0] = ang * 16.0 / M_PI;
            //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            T(0) = cos(ang);
            T(1) = -sin(ang);
            T(2) = sin(ang);
            T(3) = cos(ang);
            T(4) = 0.0;
            T(5) = 0.0;

#ifdef TRAIN_TRANSFORMER
            // Initialize Label for localizer training
            ErrorContext::label[0] = ang / astp;
#endif


#ifdef LOGPOLAR
            logxform(samples[n].m, CMap[1].m[0], T);
#else
            st.Eval(samples[n].m, CMap[1].m[0], T);
#endif
            break;
         case 2: // Shift up / down
            T(0) = 1.0;
            T(1) = 0.0;
            T(2) = 0.0;
            T(3) = 1.0;
            T(4) = shift(rng);
            T(5) = 0.0;
#ifdef LOGPOLAR
            logxform(samples[n].m, CMap[1].m[0], T);
#else
            st.Eval(samples[n].m, CMap[1].m[0], T);
#endif
            break;
         case 3: // Shift left / right
            T(0) = 1.0;
            T(1) = 0.0;
            T(2) = 0.0;
            T(3) = 1.0;
            T(4) = 0.0;
            T(5) = shift(rng);
#ifdef LOGPOLAR
            logxform(samples[n].m, CMap[1].m[0], T);
#else
            st.Eval(samples[n].m, CMap[1].m[0], T);
#endif
            break;
         default:
#ifdef LOGPOLAR
            T(0) = 1.0;
            T(1) = 0.0;
            T(2) = 0.0;
            T(3) = 1.0;
            T(4) = 0.0;
            T(5) = 0.0;
            logxform(samples[n].m, CMap[1].m[0], T);
#else
            CMap[1].m[0] = samples[n].m;
#endif            
         };

         // Initialize Label
#ifdef TRAIN_NET
         ErrorContext::label = samples[n].y;
#endif
         
         size_t fwd_count = 0;
         do {
            LayerList[fwd_count]->Eval();
            fwd_count++; // The loop exits at last count + 1.  This is by design.
         } while ((!gpExit->stop) && (fwd_count < LayerList.size()));
         gpExit->stop = false;
         //*******************************************************

         // NOTE: Can not remove the counting here because sometimes two braches correctly 
         //       identify the same image.  If the image is identified correctly but
         //       the probability is less than the threshold set in the DAGExit the
         //       image will be forwarded to the next branch.

         for (pair<const int, ErrorContext>& ep : EMap) {
            if (ep.second.correct) {
               switch (ep.first) {
               case 1: correct1++; break;
               case 2: correct2++; break;
               case 3: correct3++; break;
               case 4: correct4++; break;
               case 5: correct5++; break;
               }
            }
         }

         //*******************************************************
         //            Backward Pass
         //
         while ((!gpExit->stop) && (fwd_count > 0)) {
            fwd_count--;
            LayerList[fwd_count]->BackProp();
            // Only SGD for now.
            LayerList[fwd_count]->Update(eta);
         }
         if (fwd_count > 0) {
            early_stop++;
         }
         gpExit->stop = false;
         //*******************************************************

         if (t >= 1000) {
            double ac1 = EMap[1].PctCorrect();
#ifdef TRAIN_TRANSFORMER
            double ac2 = EMap[1].GetRunningAverage();
            double ac3 = EMap[1].GetRunningStdv();
#else
            double ac2 = 0;
            double ac3 = 0;
#endif
            //double ac2 = gpError2->PctCorrect();
            //double ac3 = gpError3->PctCorrect();
            double ac4 = EMap[4].PctCorrect();
            double ac5 = EMap[5].PctCorrect();

            int total_correct = correct1 + correct2 + correct3 + correct4 + correct5;

            err_out.Write(total_correct);
            err_out1.Write(ac1);
            err_out2.Write(ac2);
            err_out3.Write(ac3);
            err_out4.Write(ac4);
            err_out5.Write(ac5);

            cout << "ep,loop:" << left << setw(3) << loop << ", " << left << setw(5) << b << "\tearly stops: " << left << setw(4) << early_stop
               //<< "\terror:" << left << setw(9) << std::setprecision(4) << e
               << "\t %1: " << left << setw(5) << std::setprecision(2) << std::fixed << ac1
               << "\t %2: " << left << setw(5) << std::setprecision(2) << std::fixed << ac2
               << "\t %3: " << left << setw(5) << std::setprecision(2) << std::fixed << ac3
               << "\t %4: " << left << setw(5) << std::setprecision(2) << std::fixed << ac4
               << "\t %5: " << left << setw(5) << std::setprecision(2) << std::fixed << ac5
               << "\ttotal correct: " << total_correct << endl;
            //<< "T: " << pContext3->v.transpose() << endl;
            t = 0;

            correct1 = 0;
            correct2 = 0;
            correct3 = 0;
            correct4 = 0;
            correct5 = 0;

            for (pair<const int, ErrorContext>& ep : EMap) {
               ep.second.Reset();
            }
            early_stop = 0;

            char c = getch_noblock();
            if (c == 'x' || c == 'X') {
               goto JMP;
            }
         }
      }
   }


JMP:
   SaveModelWeights();

}

void Train4(int nloop, string dataroot, double eta, int load, int batch_size, int eta2)
{
   cout << "Train4" << endl;
 
   if (gTrainingBranch == 0) {
      Train2(nloop, dataroot, eta, load, eta2);
      return;
   }

   cout << dataroot << endl;
#ifdef RETRY
   cout << "NOTE: There is auto-repeate code running!  retry = " << RETRY << endl;
#endif

#ifdef LOGPOLAR
   cout << "Running Log-Polar samples." << endl;
#endif
#ifdef SGD
   cout << "Stocastic decent is on." << endl;
#else
   cout << "Batch grad descent is in use." << endl;
#endif
#ifdef USE_FFT
   cout << "Using FFT convolution." << endl;
#else
   cout << "Using linear convolution." << endl;
#endif // USE_FFT
#ifdef USE_FILTER_FOR_POOL
   cout << "Using convolution filter in place of Average Pool." << endl;
#else
   cout << "Using Average or Max Pool." << endl;
#endif
   cout << "eta2 = " << eta2 << endl;

   MNISTReader reader(dataroot + "\\train\\train-images-idx3-ubyte",
      dataroot + "\\train\\train-labels-idx1-ubyte");

   InitModel(load > 0 ? true : false);

   ErrorOutput err_out(path, model_name);
   map<int, ErrorOutput> err_out_map;
   for (int i = 0; i < EMap.size(); i++) {
      err_out_map.emplace(i, ErrorOutput(path, model_name + "_" + std::to_string(i+1)) );
   }

   StatRunningAverage pose_error(path, model_name, "terr", 1000);
   StatRunningAverage center_estimate_r(path, model_name, "center_r", 1000);
   StatRunningAverage center_estimate_c(path, model_name, "center_c", 1000);

   const int reader_batch = 1000;  // Should divide into 60K
   const int batchs_per_epoch = 60;
   const double astp = 2.0 * M_PI / 32.0;

   double e = 0;

   struct Sample_Pair {
      int iLabel;
      Matrix m;
      Sample_Pair() : iLabel(0) {}
      Sample_Pair(Matrix _m, ColVector _y) : m(_m){
         iLabel = GetLabel(_y);
      }
      void Put(Matrix _m, int label) {
         m = _m;
         iLabel = label;
      }
      ColVector Label() {
         ColVector y(10);
         y.setZero();
         y(iLabel) = 1.0;
         return y;
      }
      void operator=(Sample_Pair& sp) {
         m = sp.m;
         iLabel = sp.iLabel;
      }
   };

   vector<Sample_Pair> samples;
   Matrix temp(28, 28);

   // For debug.
   //OWeightsCSVFile fcsv(path, "lpout");

   cout << "Loading the training data." << endl;
   for (int k = 0; k < batchs_per_epoch; k++) {
      cout << "*";
      MNISTReader::MNIST_list dl = reader.read_batch(reader_batch);
      for (MNISTReader::MNIST_Pair& mp : dl) {
         TrasformMNISTtoMatrix(temp, mp.x);
         ScaleToOne(temp.data(), (int)(temp.rows() * temp.cols()));
         samples.push_back(Sample_Pair(temp, mp.y));
      }
   }
   cout << endl;

   /*
   cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!! Loading the localizer training indexes. !!!!!!!!!!!!!!!" << endl;
   Eigen::VectorXi indexes;
   GetMatrix(indexes, path + "\\localizer_training.dat");

   cout << "Loading the training data." << endl;
   int minst_counter = 0;
   int index_counter = 0;
   bool got_one = false;
   for (int k = 0; k < batchs_per_epoch; k++) {
      cout << "*";
      MNISTReader::MNIST_list dl = reader.read_batch(reader_batch);
      for (MNISTReader::MNIST_Pair& mp : dl) {
         if (minst_counter == indexes[index_counter]) {
            TrasformMNISTtoMatrix(temp, mp.x);
            ScaleToOne(temp.data(), (int)(temp.rows() * temp.cols()));
            samples.push_back(Sample_Pair(temp, mp.y));

            index_counter++;
            if (index_counter == indexes.size()) {
               cout << "Full" << endl;
               goto FULL;
            }
         }
         minst_counter++;
      }
   }

FULL:

   */

   std::random_device rd;     // only used once to initialise (seed) engine
   std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
   std::uniform_int_distribution<int> uni(0, samples.size() - 1);
   std::uniform_int_distribution<int> bni(0, batch_size - 1);
   std::uniform_int_distribution<int> shift(-3, 3);
   std::uniform_int_distribution<int> shift1(-3, 3);
   std::uniform_real_distribution<double> rotate(-M_ANG, M_ANG);
   std::uniform_real_distribution<double> scalex(0.6, 1.4);
   std::uniform_real_distribution<double> scaley(0.6, 1.4);
   std::uniform_int_distribution<int> fat(0, 100);
   std::uniform_int_distribution<int> maskx(0, 3);
   std::uniform_int_distribution<int> masky(0, 3);
   Matrix msk(7, 7);
   msk.setZero();

   //SpacialTransformer st(Size(28, 28), Size(28, 28));
   SampleMatrix sm;
   samplerBiLinear smp(Size(28, 28), Size(28, 28));
   gridAffine grd(Size(28, 28), Size(28, 28));
   ColVector T(6);

#ifdef LOGPOLAR
   //LogPolarSupportMatrix lpsm = PrecomputeLogPolarSupportMatrix(28, 28, INPUT_ROWS, INPUT_COLS + LP_WASTE);
   //cout << "Using origional LP Support Matrix builder" << endl;
   LogPolarSupportMatrix lpsm = PrecomputeLogPolarSupportMatrix1(28, 28, INPUT_ROWS, INPUT_COLS + LP_WASTE);
   Matrix mlp(INPUT_ROWS, INPUT_COLS + LP_WASTE);
   //cout << "Using LP Support Matrix builder from the Polar Transformaer paper with Alpha = 0.5." << endl;

   //auto logxform = [&temp, &mlp, &st, &lpsm](Matrix& s, Matrix& t, ColVector& T) {
   auto logxform = [&temp, &mlp, &sm, &smp, &grd, &lpsm](Matrix& s, Matrix& t, ColVector& T) {
      //st.Eval(s, temp, T);
      grd.Eval(sm, T);
      smp.Eval(s, temp, sm);
      ConvertToLogPolar(temp, mlp, lpsm);
      t = mlp.block(0, LP_WASTE, INPUT_ROWS, INPUT_COLS);
   };
#else
   CMap[1].Reset();
   CMap[1].m[0].resize(28, 28);
#endif

   int n = 0;
   int b = 0;
   int t = 0;
   unsigned long total_correct = 0;
   unsigned long early_stop = 0;

   unsigned int total_wrong = 0;

   vector<Sample_Pair> training_samples(batch_size);

   for (int loop = 0; loop < nloop; loop++) {
      if (eta2 > 0 && loop > 0 && !(loop % eta2)) {
         eta /= sqrt((double)loop/(double)eta2);
      }

      e = 0;
      b = 0;

      // Run through the samples until a traing set is reached.
      cout << "Acquiring training batch for branch. eta = " << std::setprecision(5) << eta << endl;
      do {
         n = uni(rng); // Select a random entry out of the batch.

         //*******************************************************
         //            Forward Pass

         double ang = 0.0;
         double sx = 1.0;
         double sy = 1.0;
         sx = scalex(rng);
         sy = scaley(rng);
         ang = rotate(rng);
         T(0) = sx * cos(ang);
         T(1) = sy * (-sin(ang));
         T(2) = sx * sin(ang);
         T(3) = sy * cos(ang);
         T(4) = shift(rng);
         T(5) = shift1(rng);

#ifdef LOGPOLAR
         logxform(samples[n].m, CMap[1].m[0], T);
         MakeFat(CMap[1].m[0]);
#else
         {
            vector_of_matrix vom;
            vom.push_back(samples[n].m);
            int ft = fat(rng);
            if( ft < 20) {
               MakeFat(vom[0]);
            }
            //if (ft > 10 && ft < 30) {
            //   int x = 7 * maskx(rng);
            //   int y = 7 * masky(rng);
            //   vom[0].block(x, y, 7, 7) = msk;
            //}
            grd.Eval(sm, T);
            smp.Eval(vom, CMap[1].m, sm);
         }
#endif

         ScaleToOne(CMap[1].m[0].data(), (int)(CMap[1].m[0].rows()* CMap[1].m[0].cols()));
         Matrix theInput = CMap[1].m[0];

         ErrorContext::label = samples[n].Label();
         size_t fwd_count = 0;
         do {
            LayerList[fwd_count]->Eval();
            fwd_count++; // The loop exits at last count + 1.  This is by design.
         } while ((!gpExit->stop) && (fwd_count < LayerList.size()));
         gpExit->stop = false;
         //*******************************************************

         //NOTE: gTrainingBranch > 0 should be garunteed above.
         assert(gTrainingBranch > 0);

         ErrorContext& ec = EMap.at(gTrainingBranch - 1);
         if (ec.touched && (ec.class_max < branch_pass_threshold || ec.correct==false)) {  // This currently takes correctly and incorrectly classified samples.
            training_samples[b].Put(theInput, samples[n].iLabel);
            b++;
         }

         for (pair<const int, ErrorContext>& ep : EMap) {
            ep.second.Reset();
         }

      } while (b < training_samples.size());

      b = 0;
      for (int repeat_learn_samples = 0; repeat_learn_samples < 6 * batch_size; repeat_learn_samples++) {

         // Total try counter.
         t++;

         //*******************************************************
         //            Forward Pass
         Sample_Pair& the_pair = training_samples[bni(rng)];
         CMap[1].m[0] = the_pair.m;

         ErrorContext::label = the_pair.Label();
         size_t fwd_count = 0;
         do {
            LayerList[fwd_count]->Eval();
            fwd_count++; // The loop exits at last count + 1.  This is by design.
         } while ((!gpExit->stop) && (fwd_count < LayerList.size()));
         gpExit->stop = false;
         //*******************************************************

         // NOTE: Can not remove the counting here because sometimes two braches correctly 
         //       identify the same image.  If the image is identified correctly but
         //       the probability is less than the threshold set in the DAGExit the
         //       image will be forwarded to the next branch.
         bool correct = false;
         for (pair<const int, ErrorContext>& ep : EMap) {
            if (ep.second.correct) {
               correct = true;
               break;
            }
         }
         if (correct) { total_correct++; }

         //*******************************************************
         //            Backward Pass
         //
         while ((!gpExit->stop) && (fwd_count > 0)) {
            fwd_count--;
            LayerList[fwd_count]->BackProp();
            // Only SGD for now.
            LayerList[fwd_count]->Update(eta);
         }
         if (fwd_count > 0) {
            early_stop++;
         }
         gpExit->stop = false;
         //*******************************************************

         // NOTE: Put back for Transformer test !!!!!!!!!!!!!!!!!!!!!!!
         //if (g_transformer) {
         //   gpError2->SetStatus(true, std::abs((ang / astp) - pContext3->v[0]));
         //}

         // REVIEW: Using the average pose value. 
         //pose_error.Push( std::abs((ang / astp) - CMap[0].v[0]) );
         // REVIEW: This currently only works for InitModel1T - use CMap[5]
         //                                       InitModel3T - use CMap[200]
         center_estimate_r.Push(CMap[200].v[0]);
         center_estimate_c.Push(CMap[200].v[1]);

         if (t >= 1000) {
            b++;

            err_out.Write(total_correct);

            int i = 0;
            for (auto& p : err_out_map) {
               p.second.Write(EMap[i].PctCorrect());
               i++;
            }

            //pose_error.write();

            center_estimate_r.write();
            center_estimate_c.write();

            cout << "ep,loop:" << left << setw(3) << loop << ", " << left << setw(5) << b << "\tearly stops: " << left << setw(4) << early_stop;
            //<< "\terror:" << left << setw(9) << std::setprecision(4) << e
            for (pair<const int, ErrorContext>& ep : EMap) {
               cout << "\t %" << ep.first << ": " << left << setw(5) << std::setprecision(2) << std::fixed << ep.second.PctCorrect();
            }
            cout << "\ttotal correct: " << total_correct << endl;

            t = 0;

            total_correct = 0;

            for (pair<const int, ErrorContext>& ep : EMap) {
               ep.second.Reset();
            }

            early_stop = 0;

            char c = getch_noblock();
            if (c == 'x' || c == 'X') {
               goto JMP;
            }
         }
      }
   }

JMP:


   // Ask if Transformer should be trained.
   // If so take a forward pass and record index of correct id,
   // use these ids to train the transformer.
   // Pass the Sapmples vector and Correct List to the transformer training algorithm or pass a new vector with references
   // to the training list.
   // The tranformer model must be able to run the underlaying model without the transformer in order
   // to generate a base of good training samples.

   TestModel(dataroot);

   if (gAutoSave) {
      SaveModelWeights();
   }
   else {
      std::cout << "Save? y/n:  ";
      char c;
      std::cin >> c;
      if (c == 'y') {
         SaveModelWeights();
      }
   }
}

class Loss2DGaussL2 {
private:
   struct grad3 {
      double dgdrc;
      double dgdcc;
      double dgdlam;
      grad3(double drc, double dcc, double dlam) : dgdrc(drc), dgdcc(dcc), dgdlam(dlam) {}
      grad3() : dgdrc(0.0), dgdcc(0.0), dgdlam(0.0) {}
   };

   double gauss(double rc, double cc, double lam, double r, double c) {
      double dr = r - rc;
      double dc = c - cc;
      double sig = 2.0 * lam;
      return (exp(-(dr * dr + dc * dc) / sig));
   }

   grad3 grad_g(double rc, double cc, double lam, double r, double c) {
      double dr = r - rc;
      double dc = c - cc;
      double dgdrc = gauss(rc, cc, lam, r, c) * (r - rc) / lam;
      double dgdcc = gauss(rc, cc, lam, r, c) * (c - cc) / lam;
      double dgdl = 0.5 * gauss(rc, cc, lam, r, c) * (dr * dr + dc * dc) / (lam * lam);

      return grad3(dgdrc, dgdcc, dgdl);
   }

   void ComputeCentroid(double& x, double& y, const Matrix& m)
   {
      int rows = m.rows();
      int cols = m.cols();
      double total_x = 0;
      double total_y = 0;
      double weight = 0;

      // Iterate through the matrix to calculate the centroid
      for (int i = 0; i < rows; i++) {
         for (int j = 0; j < cols; j++) {
            double value = m(i, j);
            if (value > 0.0) {
               total_x += j * value;
               total_y += i * value;
               weight += value;
            }
         }
      }

      // Calculate the centroid
      if (weight > 0.0) {
         x = total_x / weight;
         y = total_y / weight;
      }
   }

public:
   double step = 0.05;
   double mc = 16.0;
   double nc = 14.0;
   double lam = 6.0;
   double fit_err = 0.0;
private:
   void FitGauss(const Matrix& f)
   {
      int rows = f.rows();
      int cols = f.cols();

      ComputeCentroid(mc, nc, f);

      for (int n = 0; n < 100; n++) {
         // Compute E and the grad of E
         grad3 grad_E;
         fit_err = 0;

         for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
               double dr = r - mc;
               double dc = c - nc;
               double gg = gauss(mc, nc, lam, r, c);
               double zz = f(r, c) - gg;
               grad3 g3 = grad_g(mc, nc, lam, r, c);


               // Accumulate the error term
               fit_err += (zz * zz);

               // Compute the grad of E
               grad_E.dgdrc += (-2.0 * zz * g3.dgdrc);
               grad_E.dgdcc += (-2.0 * zz * g3.dgdcc);
               grad_E.dgdlam += (-2.0 * zz * g3.dgdlam);
            }
         }
         //if (!(n % 100)) {
         //   step *= 0.99;
         //   cout << "step: " << step << " err: " << err << " mc: " << mc << " nc: " << nc << " lam: " << lam << endl;
         //}

         mc = mc - step * grad_E.dgdrc;
         nc = nc - step * grad_E.dgdcc;
         lam = lam - step * grad_E.dgdlam;
         if (lam < 1.0) {
            lam = 1.0;
         }
      }
   }

   void ComputeDiff(const Matrix& f)
   {
      int rows = f.rows();
      int cols = f.cols();
      for (int r = 0; r < rows; r++) {
         for (int c = 0; c < cols; c++) {
            double gg = gauss(mc, nc, lam, r, c);
            Z(r, c) = f(r, c) - gg;
         }
      }
   }

public:
   Size InputSize;
   Matrix Z;

   Loss2DGaussL2() : InputSize(0.0, 0.0) {}
   Loss2DGaussL2(Size input_size) :
      Z(input_size.rows,input_size.cols),
      InputSize(input_size) {}

   void Init(Size input_size) {
      Z.resize(input_size.rows, input_size.cols);
      InputSize = input_size;
   }



   // Matrix representing correlation plane and Vector of
   // Gauss parameters goes in.  Compute difference, sum and pass put.
   Number Eval(const Matrix& x) {
      FitGauss(x);
      ComputeDiff(x);

      return Z.array().square().sum();
   }

   Matrix LossGradient(void) {
      return 2.0 * Z;
   }
};

void TestCharacterEmbedding(int number, string dataroot, bool test_on_train )
{
   // ToDo: Refactor this test for the new encoding method.

   MNISTReader reader1;
   if (test_on_train) {
      reader1.open(dataroot + "\\train\\train-images-idx3-ubyte",
         dataroot + "\\train\\train-labels-idx1-ubyte");
   }
   else {
      reader1.open(dataroot + "\\test\\t10k-images-idx3-ubyte",
         dataroot + "\\test\\t10k-labels-idx1-ubyte");
   }

   InitCharacterEmbedding(true, number);

   ColVector X;
   int count = 0;

   while (reader1.read_next()) {
      count++;
      X = reader1.data();
      ColVector label = reader1.label();
      int nl = GetLabel( label );
      Matrix temp(28, 28);
      TrasformMNISTtoMatrix(temp, X);
      ScaleToOne(temp.data(), (int)(temp.rows() * temp.cols()));

      CMap[1].m[0] = temp;
      ColVector lbl(1);
      lbl(0) = nl == number ? 1.0 : -1.0;
      ErrorContext::label = lbl;

      //*******************************************************
      //            Forward Pass
      size_t fwd_count = 0;
      do {
         LayerList[fwd_count]->Eval();
         fwd_count++; // The loop exits at last count + 1.  This is by design.
      } while ((!gpExit->stop) && (fwd_count < LayerList.size()));
      gpExit->stop = false;
      //*******************************************************

      /*
      cout << "pattern: " << number << " Test number: " << nl << " output: " << CMap[2].v[0] << " Error: " << EMap[0].error << "\n"
         << "\nDo you want to save it?  Enter Y - yes | N - no | E - exit." << endl;
      char c;
      cin >> c;
      cout << endl;
      if (c == 'Y' || c == 'y') {
         MakeMatrixImage(path + "\\org." + to_string(count) + ".bmp", temp);
      }
      else if (c == 'E' || c == 'e') {
         exit(0);
      }
      */
   }

   cout << "pattern: " << number << " Test number: " << EMap[0].PctCorrect() << endl;

}

void TrainCharacterEmbedding(int number, int nloop, double eta, string dataroot)
{ 
   cout << dataroot << endl;

   #ifdef SGD
      cout << "Stocastic decent is on." << endl;
   #else
      cout << "Batch grad descent is in use." << endl;
   #endif
   #ifdef USE_FFT
      cout << "Using FFT convolution." << endl;
   #else
      cout << "Using linear convolution." << endl;
   #endif // USE_FFT

   model_name = "CE." + to_string(number);

   MNISTReader reader(dataroot + "\\train\\train-images-idx3-ubyte",
      dataroot + "\\train\\train-labels-idx1-ubyte");

   int load = 0;
   InitCharacterEmbedding(load > 0 ? true : false, number);

   map<int, ErrorOutput> err_out_map;
   for (int i = 0; i < EMap.size(); i++) {
      err_out_map.emplace(i, ErrorOutput(path, model_name + "_" + std::to_string(i + 1)));
   }

   const int reader_batch = 1000;  // Should divide into 60K
   const int batchs_per_epoch = 60;

   double e = 0;

   struct Sample_Pair {
      Matrix m;
      ColVector y;
      Sample_Pair() {}
      Sample_Pair(Matrix _m, ColVector _y) : m(_m), y(_y) {}
   };

   vector< Sample_Pair> samples;
   Matrix temp(28, 28);

   cout << "Loading the localizer training indexes." << endl;
   Eigen::VectorXi indexes;
   GetMatrix(indexes, path + "\\localizer_training.dat");

   cout << "Loading the training data." << endl;
   int minst_counter = 0;
   int index_counter = 0;
   bool got_one = false;
   for (int k = 0; k < batchs_per_epoch; k++) {
      cout << "*";
      MNISTReader::MNIST_list dl = reader.read_batch(reader_batch);
      for (MNISTReader::MNIST_Pair& mp : dl) {
         if (minst_counter == indexes[index_counter]) {
            TrasformMNISTtoMatrix(temp, mp.x);
            ScaleToOne(temp.data(), (int)(temp.rows() * temp.cols()));
            if (GetLabel(mp.y) == number) {
               got_one = true;
               ColVector lab(1); lab(0) = 1.0;
               samples.push_back(Sample_Pair(temp, lab));
            }
            else if( got_one ) {
               got_one = false;
               ColVector lab(1); lab(0) = -1.0;
               samples.push_back(Sample_Pair(temp, lab));
            }
            index_counter++;
            if (index_counter == indexes.size()) {
               cout << "Full" << endl;
               goto FULL;
            }
         }
         minst_counter++;
      }
   }

FULL:

   cout << endl;
   SpacialTransformer st(Size(28, 28), Size(28, 28));
   ColVector T(6);

   std::random_device rd;     // only used once to initialise (seed) engine
   std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
   std::uniform_int_distribution<int> uni(0, samples.size() - 1);
   std::uniform_int_distribution<int> shift(-3, 3);
   std::uniform_int_distribution<int> shift1(-3, 3);
   std::uniform_real_distribution<double> rotate(-M_ANG, M_ANG);

   //LogPolarSupportMatrix lpsm = PrecomputeLogPolarSupportMatrix(28, 28, INPUT_ROWS, INPUT_COLS + LP_WASTE);
   //cout << "Using origional LP Support Matrix builder" << endl;
   LogPolarSupportMatrix lpsm = PrecomputeLogPolarSupportMatrix1(28, 28, INPUT_ROWS, INPUT_COLS + LP_WASTE);
   cout << "Using LP Support Matrix builder from the Polar Transformaer paper with Alpha = 0.5." << endl;
   Matrix mlp(INPUT_ROWS, INPUT_COLS + LP_WASTE);

   auto logxform = [&temp, &mlp, &st, &lpsm](Matrix& s, Matrix& t) {
      ConvertToLogPolar(temp, mlp, lpsm);
      t = mlp.block(0, LP_WASTE, INPUT_ROWS, INPUT_COLS);
   };

   auto logxform1 = [&temp, &mlp, &st, &lpsm](Matrix& s, Matrix& t, ColVector& T) {
      st.Eval(s, temp, T);
      ConvertToLogPolar(temp, mlp, lpsm);
      t = mlp.block(0, LP_WASTE, INPUT_ROWS, INPUT_COLS);
   };


   optoADAM::B1 = 0.9;
   optoADAM::B2 = 0.999;
   optoLowPass::Momentum = 0.8;

   /*
   bool restore = false;
   clSize size_in(INPUT_ROWS, INPUT_COLS);
   clSize size_out(INPUT_ROWS, INPUT_COLS);
   clSize size_kern(INPUT_ROWS, INPUT_COLS);
   int kern_per_chn = 1;
   int chn_in = 1;
   int chn_out = kern_per_chn * chn_in;
   int l = 1; // Layer counter
   FilterLayer2D f2d(size_in, chn_in, size_out, size_kern, kern_per_chn,
      make_unique<actLinear>(), 
      restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
      dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, chn_in)),
      make_shared<optoADAM>(),
      true);
   f2d.SetConvolutionParameters(FilterLayer2D::FFT, true, false);

   Loss2DGaussL2 loss(size_in);

   //--------------------
   Matrix a;
   logxform(samples[0].m, a);
   f2d.W[0] = a / 700.0;
   //--------------------
*/

   int n = 0;
   int b = 0;
   int t = 0;

   for (int loop = 0; loop < nloop; loop++) {
      if (loop > 0 && !(loop % 2)) {
         eta /= sqrt((double)loop / 2.0);
      }
      e = 0;
      b = 0;
      while (b < samples.size()) {
         t++;
         b++;
         n = uni(rng); // Select a random entry out of the batch.

         //*******************************************************
         //            Forward Pass
         CMap[1].m[0] = samples[n].m;
         Matrix theInput = CMap[1].m[0];

         ErrorContext::label = samples[n].y;
         size_t fwd_count = 0;
         do {
            LayerList[fwd_count]->Eval();
            fwd_count++; // The loop exits at last count + 1.  This is by design.
         } while ((!gpExit->stop) && (fwd_count < LayerList.size()));
         gpExit->stop = false;
         //*******************************************************

         //*******************************************************
         //            Backward Pass
         //
         while ((!gpExit->stop) && (fwd_count > 0)) {
            fwd_count--;
            LayerList[fwd_count]->BackProp();
            // Only SGD for now.
            LayerList[fwd_count]->Update(eta);
         }
         gpExit->stop = false;
         //*******************************************************

         if (t >= 100) {

            int i = 0;
            for (auto& p : err_out_map) {
               p.second.Write(EMap[i].GetRunningAverage());
               i++;
            }

            cout << "ep,loop:" << left << setw(3) << loop << ", " << left << setw(5) << b;
            //<< "\terror:" << left << setw(9) << std::setprecision(4) << e
            for (pair<const int, ErrorContext>& ep : EMap) {
               cout << "\t %" << ep.first << ": " << left << setw(5) << std::setprecision(2) << std::fixed << ep.second.GetRunningAverage()
                  << " : " << left << setw(5) << std::setprecision(2) << std::fixed << ep.second.GetRunningStdv();
            }
            cout << endl;

            t = 0;

            //total_correct = 0;

            for (pair<const int, ErrorContext>& ep : EMap) {
               ep.second.Reset();
            }

            char c = getch_noblock();
            if (c == 'x' || c == 'X') {
               goto JMP;
            }
         }
      }
   }

   JMP:
      if (gAutoSave) {
         SaveModelWeights();
      }
      else {
         std::cout << "Save? y/n:  ";
         char c;
         std::cin >> c;
         if (c == 'y') {
            SaveModelWeights();
         }
      }

}

void Test(string dataroot, bool test_on_train)
{
   InitModel(true);
   TestModel(dataroot,test_on_train);

   std::cout << "Hit a key and press Enter to continue.";
   char c;
   std::cin >> c;
}

void FilterTop(unsigned int li, double sig)
{
   InitModel(true);

   // NOTE: This is a blind downcast to FilterLayer2D.  Normally is will resolve to a FilterLayer2D object because
   //       we are working with the top layer.  The assert will make sure the downcast is valid.
   shared_ptr<DAGConvoLayerObj> idcl = dynamic_pointer_cast<DAGConvoLayerObj>(LayerList[li-1]);
   assert(idcl);
   shared_ptr<FilterLayer2D> ipcl = dynamic_pointer_cast<FilterLayer2D>(idcl->pLayer);
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
         double e = (rr * rr + cc * cc) / (2.0 * sig);
         h(r, c) = norm * std::exp(-e);
      }
   }

   for (int kn = 0; kn < kpc; kn++) {
      //Matrix fw(ksr, ksc);
      fft2convolve(ipcl->W[kn], h, ipcl->W[kn], 1, true, true);
      //ipcl->W[kn] = fw;
   }

   int l = li;
   idcl->Save(make_shared<OWeightsCSVFile>(path, model_name + "." + to_string(l)));
   idcl->Save(make_shared<OMultiWeightsBMP>(path, model_name + "." + to_string(l)));
   idcl->Save(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l)));
}

void GetMatrix(Matrix& m, string filename) {
   ifstream file(filename, ios::in | ios::binary);
   runtime_assert(file.is_open());

   struct Header {
      int rows;
      int cols;
      int step;
   };

   Header header;
   file.read(reinterpret_cast<char*>((&header)), sizeof(Header));
   runtime_assert(header.step == sizeof(typename Matrix::Scalar));

   m.resize(header.rows, header.cols);

   for (int r = 0; r < header.rows; r++) {
      for (int c = 0; c < header.cols; c++) {
         double v;
         file.read((char*)&v, sizeof(double));
         m(r, c) = v;
      }
   }

   file.close();
}

void FilterMatrix(string name, int number, string dataroot, bool pad_rows, bool pad_cols, double sig) 
{

   string str_count;
   str_count = to_string(number);

   string file = dataroot + "\\" + name + "." + str_count + ".wts";
   cout << "Smooth: " << file << endl;
   Matrix m;
   GetMatrix(m, file);

   int ksr = m.rows();
   int ksc = m.cols();
   double r_c = ((double)ksr) / 2.0;
   double c_c = ((double)ksr) / 2.0;

   double norm = 1.0 / (2.0 * M_PI * sig);

   Matrix h(ksr, ksc);
   /*
   for (int r = 0; r < ksr; r++) {
      for (int c = 0; c < ksc; c++) {
         double rr = r - r_c;
         double cc = c - c_c;
         double e = (rr * rr + cc * cc) / (2.0 * sig);
         h(r, c) = norm * std::exp(-e);
      }
   }
   */
   h.setZero();
   h(15, 15) = 0.25;
   h(15, 16) = 0.25;
   h(16, 15) = 0.25;
   h(16, 16) = 0.25;

   fft2convolve(m, h, m, 1, pad_rows, pad_cols);

   OWeightsCSVFile ocsv(path, name);
   ocsv.Write(m, number);
   OMultiWeightsBMP obmp(path, name);
   obmp.Write(m, number);
   IOWeightsBinaryFile owts(path, name);
   owts.Write(m, number);
}

int main(int argc, char* argv[])
{
   //int loop = 0;
   //int eta2 = 5;
   //if (!(loop % eta2)) {
   //   cout << "inside" << endl;
   //}
   //exit(0);

   try {
      std::cout << "Starting Branch MNIST.  Build date: " << __DATE__ << endl;
      string dataroot = "C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data";

      //std::map<int, DAGContext> cm;
      //DAGContext& c1 = cm[1];
      //c1.v.resize(10);
      //DAGContext& c2 = cm[1];

      //cout << c2.v.size();
      //InitModel = InitLPBranchModel1;
      //MakeListOfCorrect(dataroot);


      //std::map<unsigned int, ErrorContext> errorMap;
      //errorMap.emplace(1, 1);

      //ErrorContext& e1 = errorMap[1];

      //return 0;

      if (argc > 1 && string(argv[1]) == "train") {
         if (argc < 3) {
            cout << "Not enough parameters.  Parameters: train | batches | eta | read stored coefs (0|1) [optional] | dataroot [optional] | path [optional]" << endl;
            return 0;
         }
         double eta = atof(argv[3]);
         int load = 0;
         if (argc > 4) { load = atoi(argv[4]); }
         if (argc > 5) { dataroot = argv[5]; }
         if (argc > 6) { path = argv[6]; }

         Train(atoi(argv[2]), dataroot, eta, load);
      }
      else if (argc > 1 && string(argv[1]) == "ltrain") {
         if (argc < 3) {
            cout << "Not enough parameters.  Parameters: ltrain | epochs | eta | read stored coefs (0|1) [optional] | dataroot [optional] | path [optional]" << endl;
            return 0;
         }
         double eta = atof(argv[3]);
         int load = 0;
         if (argc > 4) { load = atoi(argv[4]); }
         if (argc > 5) { dataroot = argv[5]; }
         if (argc > 6) { path = argv[6]; }

         TrainLocalizer(atoi(argv[2]), dataroot, eta, load);
      }
      else if (argc > 1 && string(argv[1]) == "train2") {
         if (argc < 6) {
            cout << "Not enough parameters.  Parameters: train2 | epochs | eta | eta2 | read stored coefs (0|1) | warmup (0|1) | training branch (0|1|2|3|4) [optional] | batch size [not used] | auto save (0|1) [optional] | dataroot [optional] | path [optional]" << endl;
            return 0;
         }
         double eta = atof(argv[3]);
         int load = 0;
         int eta2 = 0;
         gWarmupOn = 0;
         gAutoSave = 0;
         gTrainingBranch = 0;
         if (argc > 4) { eta2 = atoi(argv[4]); }
         if (argc > 5) { load = atoi(argv[5]); }
         if (argc > 6) { gWarmupOn = atoi(argv[6]); }
         if (argc > 7) { gTrainingBranch = atoi(argv[7]); }
         //if (argc > 8) { batch_size = atoi(argv[8]); }
         if (argc > 9) { gAutoSave = atoi(argv[9]); }
         if (argc > 10) { dataroot = argv[10]; }
         if (argc > 11) { path = argv[11]; }

         Train2(atoi(argv[2]), dataroot, eta, load, eta2);
      }
      else if (argc > 1 && string(argv[1]) == "train4") {
         if (argc < 7) {
            cout << "Not enough parameters.  Parameters: train4 | epochs | eta | eta2 | read stored coefs (0|1) | warmup (0|1) | training branch (0|1|2|3|4) | batch size [optional] | auto save (0|1) [optional] | dataroot [optional] | path [optional]" << endl;
            return 0;
         }
         double eta = atof(argv[3]);
         int load = 0;
         int batch_size = 1000;
         gWarmupOn = 0;
         gAutoSave = 0;
         int eta2 = 0;
         if (argc > 4) { eta2 = atoi(argv[4]); }
         if (argc > 5) { load = atoi(argv[5]); }
         if (argc > 6) { gWarmupOn = atoi(argv[6]); }
         if (argc > 7) { gTrainingBranch = atoi(argv[7]); }
         if (argc > 8) { batch_size = atoi(argv[8]); }
         if (argc > 9) { gAutoSave = atoi(argv[9]); }
         if (argc > 10) { dataroot = argv[10]; }
         if (argc > 11) { path = argv[11]; }

         Train4(atoi(argv[2]), dataroot, eta, load, batch_size, eta2);
      }
      else if (argc > 1 && string(argv[1]) == "train5") {
            if (argc < 3) {
               cout << "Not enough parameters.  Parameters: train5 | epochs | eta | number | dataroot [optional] | path [optional]" << endl;
               return 0;
            }

         if (argc > 5) { dataroot = argv[5]; }
         if (argc > 6) { path = argv[6]; }
         double eta = atof(argv[3]);

         TrainCharacterEmbedding(atoi(argv[4]), atoi(argv[2]), eta, dataroot);
      }
      else if (argc > 1 && string(argv[1]) == "testce") {
         if (argc < 3) {
            cout << "Not enough parameters.  Parameters: testce | number | dataroot [optional] | path [optional]" << endl;
            return 0;
         }

         if (argc > 3) { dataroot = argv[3]; }

         TestCharacterEmbedding(atoi(argv[2]), dataroot, true);
      }
      else if (argc > 1 && string(argv[1]) == "test") {

         if (argc < 1) {
            cout << "Not enough parameters.  Parameters: test | layers [optional] | dataroot [optional] | path [optional] | test on training data (0|1) [optional]" << endl;
            return 0;
         }

         bool test_on_train = false;
         if (argc > 2) { gTrainingBranch = atoi(argv[2]); }
         if (argc > 3) { dataroot = argv[3]; }
         if (argc > 4) { path = argv[4]; }
         if (argc > 5) { test_on_train = atoi(argv[5]) > 0 ? true : false; }

         Test(dataroot, test_on_train);
      }
      else if (argc > 1 && string(argv[1]) == "smooth") {

         if (argc < 3) {
            cout << "Not enough parameters.  Parameters: smooth | name | pad rows | pad cols | number in set [optional default = 0] | sigma [optional] | path [optional]" << endl;
            return 0;
         }

         bool pad_rows = true;
         bool pad_cols = true;
         if (argc > 3) { pad_rows = atoi(argv[3]) == 1 ? true : false; }
         if (argc > 4) { pad_cols = atoi(argv[4]) == 1 ? true : false; }

         int k = 0;
         if (argc > 5) { k = atoi(argv[5]); }

         double sigma = 2.0;
         if (argc > 6) { sigma = atof(argv[6]); }
         if (argc > 7) { path = argv[7]; }

         FilterMatrix(argv[2],k,path,pad_rows,pad_cols,sigma);
      }
      //else if (argc > 1 && string(argv[1]) == "exp") {
      //   NetGrowthExp();
      //}
      else {
         cout << "Enter a command." << endl;
      }
   }
   catch (std::exception ex) {
      cout << "Error:\n" << ex.what() << endl;
   }
}

