// SNBranchMNIST.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#define LOGPOLAR
#ifdef LOGPOLAR
const int INPUT_ROWS = 32;
//const int INPUT_COLS = 32;
const int LP_WASTE = 12;
const int INPUT_COLS = 32 - LP_WASTE;
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
#include "FilterLayer.h"
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
#include <conio.h>

   // Define static optomizer variables.
double optoADAM::B1 = 0.0;
double optoADAM::B2 = 0.0;
double optoLowPass::Momentum = 0.0;

int gModelBranches = 0;

string path = "C:\\projects\\neuralnet\\simplenet\\SNBranchMNIST\\weights";
string model_name = "layer";

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
         if (rr < 0.0) { rr = 0.0; }
         if (cc < 0.0) { cc = 0.0; }

         if (rr > (in_rows - 1)) { rr = in_rows - 1; }
         if (cc > (in_cols - 1)) { cc = in_cols - 1; }

         //runtime_assert(rr >= 0.0 && cc >= 0.0);

         int rl = static_cast<int>(floor(rr));
         int rh = static_cast<int>(ceil(rr));
         int cl = static_cast<int>(floor(cc));
         int ch = static_cast<int>(ceil(cc));

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

void ConvertToLogPolar(Matrix& m, Matrix& out, LogPolarSupportMatrix& lpsm)
{
   int rows = out.rows();
   int cols = out.cols();

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

class ErrorContext
{
public:
   static ColVector label;
   double error;
   bool correct;
   double class_max;
   ErrorContext() :
      error(0.0),
      correct(false),
      class_max(0.0)
   {};
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
   // REVIEW: What about want_backprop optional parameter.
   virtual void BackProp() = 0;
   virtual void Update(double eta) = 0;
   virtual void Save(shared_ptr<iPutWeights> _pOut) = 0;
   virtual void StashWeights() = 0;
   virtual void ApplyStash() = 0;
};

class DAGConvoLayerObj : public iDAGObj
{
public:
   unsigned int ID;
   shared_ptr<iConvoLayer> pLayer;
   shared_ptr<CovNetContext> pContext;

   DAGConvoLayerObj(unsigned int id, shared_ptr<iConvoLayer> _pLayer, shared_ptr<CovNetContext> _pContext ) :
      ID(id),
      pLayer(std::move(_pLayer)),
      pContext(_pContext)
   {}

   // iDAGObj implementation
   void Eval() {
      pContext->m = pLayer->Eval(pContext->m);
   }
   void BackProp() {
      pContext->m = pLayer->BackProp(pContext->m);
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
   //--------------------------------------
};

class DAGFlattenObj : public iDAGObj
{
   shared_ptr<iConvoLayer> pLayer;
   shared_ptr<CovNetContext> pCovContext;
   shared_ptr<NetContext> pLayerContext; 
public:
   unsigned int ID;
   DAGFlattenObj(unsigned int id, shared_ptr<iConvoLayer> _pLayer, shared_ptr<CovNetContext> _pContext1, shared_ptr<NetContext> _pContext2) :
      ID(id),
      pLayer(std::move(_pLayer)),
      pCovContext(_pContext1),
      pLayerContext(_pContext2)
   {}
   void Eval() {
      pLayerContext->v = pLayer->Eval(pCovContext->m)[0].col(0);
   }
   void BackProp() {
      vector_of_matrix t(1);
      t[0] = pLayerContext->g;
      // REVIEW: Note here that I am trying to use the VOM of the forward
      //         pass for the backward pass as well.
      pCovContext->m = pLayer->BackProp(t);
   }
   void Update(double eta) { pLayer->Update(eta); }
   void Save(shared_ptr<iPutWeights> _pOut) { pLayer->Save(_pOut); }
   void StashWeights() {}
   void ApplyStash() {}
};

class DAGLayerObj : public iDAGObj
{
   shared_ptr<iLayer> pLayer;
   shared_ptr<NetContext> pContext;
public:
   unsigned int ID;
   DAGLayerObj(unsigned int id, shared_ptr<iLayer> _pLayer, shared_ptr<NetContext> _pContext) :
      ID(id),
      pLayer( std::move(_pLayer) ),
      pContext( _pContext)
   {}
   void Eval() {
      pContext->v = pLayer->Eval(pContext->v);
   }
   void BackProp() {
      pContext->g = pLayer->BackProp(pContext->g);
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
};

class DAGConvoContextCopyObj : public iDAGObj
{
   shared_ptr<CovNetContext> pContext1;
   shared_ptr<CovNetContext> pContext2;
public:
   unsigned int ID;
   DAGConvoContextCopyObj(unsigned int id, shared_ptr<CovNetContext> _pContext1, shared_ptr<CovNetContext> _pContext2) :
      ID(id),
      pContext1(_pContext1),
      pContext2(_pContext2)
   {}
   void Eval() {
      pContext2->m = pContext1->m;
   }
   void BackProp() {}
   void Update(double eta) { }
   void Save(shared_ptr<iPutWeights> _pOut) {}
   void StashWeights() {}
   void ApplyStash() {}
};

class DAGBranchObj : public iDAGObj
{
   friend class DAGExitTest;
   shared_ptr<NetContext> pContext1;
   shared_ptr<NetContext> pContext2;
   bool bBackprop;
   double BranchWeight;
public:
   unsigned int ID;
   DAGBranchObj(unsigned int id, shared_ptr<NetContext> _pContext1, shared_ptr<NetContext> _pContext2, bool backprop_branch = true, double branch_weight = 0.3 ) :
      ID(id),
      pContext1(_pContext1),
      pContext2(_pContext2),
      bBackprop(backprop_branch),
      BranchWeight(branch_weight)
   {}
   void Eval() {
      pContext2->v = pContext1->v;
   }
   void BackProp() {
      if (bBackprop && BranchWeight > 0.0) {
         pContext1->g += (BranchWeight * pContext2->g);
      }
   }
   void Update(double eta) { }
   void Save(shared_ptr<iPutWeights> _pOut) {}
   void StashWeights() {}
   void ApplyStash() {}
};

class DAGJoinObj : public iDAGObj
{
   shared_ptr<NetContext> pContext1;
   shared_ptr<NetContext> pContext2;
   int s1 = 0;
   int s2 = 0;
public:
   unsigned int ID;
   // Note: The contexts are assigned at creation time.  They can't be accessed until
   //       Eval is called.
   DAGJoinObj(unsigned int id, shared_ptr<NetContext> _pContext1, shared_ptr<NetContext> _pContext2) :
      ID(id),
      pContext1(_pContext1),
      pContext2(_pContext2)
   {}
   void Eval() {
      s1 = static_cast<int>(pContext1->v.size());
      s2 = static_cast<int>(pContext2->v.size());
      ColVector t = pContext2->v;
      pContext2->v.resize(s1 + s2);
      pContext2->v.block(0, 0, s1, 1) = pContext1->v;
      pContext2->v.block(s1, 0, s2, 1) = t;
   }
   void BackProp() {
      RowVector t = pContext2->g.block(0, s1, 1, s2);
      pContext1->g = pContext2->g.block(0, 0, 1, s1);
      pContext2->g.resize(s2);
      pContext2->g = t;
   }
   void Update(double eta) { }
   void Save(shared_ptr<iPutWeights> _pOut) {}
   void StashWeights() {}
   void ApplyStash() {}
};

class DAGErrorLayer : public iDAGObj
{
   shared_ptr<iLossLayer> pLossLayer;
   shared_ptr<ErrorContext> pErrorContext;
   shared_ptr<NetContext> pNetContext;

   void StatEval(const ColVector& x, const ColVector& y) {
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

      pErrorContext->correct = (x_max_index == y_max_index);
      pErrorContext->class_max = xmax;
   }
public:
   unsigned int ID;
   DAGErrorLayer(unsigned int id, shared_ptr<iLossLayer> _pLayer, shared_ptr<NetContext> _pContext, shared_ptr<ErrorContext> _pEContext) :
      ID(id),
      pLossLayer(std::move(_pLayer)),
      pErrorContext(_pEContext),
      pNetContext( _pContext )
   {}
   void Eval() {
      pErrorContext->error = pLossLayer->Eval(pNetContext->v, pErrorContext->label);
      StatEval(pNetContext->v, pErrorContext->label);
   }
   void BackProp() {
      pNetContext->g = pLossLayer->LossGradient();
   }
   void Update(double eta) { }
   void Save(shared_ptr<iPutWeights> _pOut) {}
   void StashWeights() {}
   void ApplyStash() {}
};

class DAGExitTest : public iDAGObj
{
   shared_ptr<ErrorContext> pError;
   shared_ptr<ExitContext> pExit;
   shared_ptr<DAGBranchObj> pBranch;
   unsigned long WarmUpCount;
   unsigned long Count;
   double BackpropLimitUpper;
   double BackpropLimitLower;
public:
   unsigned int ID;
   DAGExitTest(unsigned int id, shared_ptr<ExitContext> _pContext, shared_ptr<DAGBranchObj> _pBranch, shared_ptr<ErrorContext> _pError,
               unsigned long warm_up_count = -1.0, double backprop_limit_lower = -1.0, double backprop_limit_upper = -1.0) :
      ID(id),
      Count(0),
      WarmUpCount(warm_up_count),
      BackpropLimitLower(backprop_limit_lower),
      BackpropLimitUpper(backprop_limit_upper),
      pExit(_pContext),
      pBranch(_pBranch),
      pError(_pError)
   {
   }
   void Eval() {
      Count++;
      if( WarmUpCount > 0 && Count < WarmUpCount ){
         pBranch->bBackprop = false;
         pExit->stop = true;
      }
      else if( pError->correct && pError->class_max >= 0.9){
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
         if (BackpropLimitUpper > 0.0 && pError->class_max >= BackpropLimitUpper) {
            //cout << "Limit backprop on high end. ID" << ID << " error:" << pError->class_max << endl;
            pBranch->bBackprop = false;
            pExit->stop = true;
         }
         else if (BackpropLimitLower > 0.0 && pError->class_max < BackpropLimitLower) {
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
};

typedef vector< shared_ptr<iDAGObj> > layer_list;
layer_list LayerList;

//----------------------------------------------
class myMCallBack : public iCallBackSink
{
public:
   myMCallBack() {}
   void Properties(std::map<string, CallBackObj>& props) override
   {
      const RowVector& dg = props["dG"].rv.get();
      cout << dg.norm()  << endl;

      // For examining the Jacobian of the Spectral Pool.
      //cout << "Norm J:" << endl << props["J"].m.get() << endl;
   }
};

shared_ptr<iCallBackSink> MCB;// = make_shared<myMCallBack>();
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
      OWeightsCSVFile osi(path, "filter.in." + to_string(lbl) );
      OWeightsCSVFile oso(path, "filter.out." + to_string(lbl) );
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


// This is the main input context.
   // Convo Layer
shared_ptr<CovNetContext> pContext1 = make_shared<CovNetContext>();
shared_ptr<CovNetContext> pContext4 = make_shared<CovNetContext>();
shared_ptr<CovNetContext> pContext7 = make_shared<CovNetContext>();
shared_ptr<CovNetContext> pContext10 = make_shared<CovNetContext>();
shared_ptr<CovNetContext> pContext13 = make_shared<CovNetContext>();
// FC Layer main branch
shared_ptr<NetContext> pContext2 = make_shared<NetContext>();

shared_ptr<NetContext> pContext3 = make_shared<NetContext>();
shared_ptr<NetContext> pContext5 = make_shared<NetContext>();
shared_ptr<NetContext> pContext6 = make_shared<NetContext>();
shared_ptr<NetContext> pContext8 = make_shared<NetContext>();
shared_ptr<NetContext> pContext9 = make_shared<NetContext>();
shared_ptr<NetContext> pContext11 = make_shared<NetContext>();
shared_ptr<NetContext> pContext12 = make_shared<NetContext>();
shared_ptr<NetContext> pContext14 = make_shared<NetContext>();


shared_ptr<ErrorContext> gpError1 = make_shared<ErrorContext>();
shared_ptr<ErrorContext> gpError2 = make_shared<ErrorContext>();
shared_ptr<ErrorContext> gpError3 = make_shared<ErrorContext>();
shared_ptr<ErrorContext> gpError4 = make_shared<ErrorContext>();
shared_ptr<ErrorContext> gpError5 = make_shared<ErrorContext>();

shared_ptr<ExitContext> gpExit = make_shared<ExitContext>();

#define INITIALIZE( NAME, OP ) \
   typedef OP OPTO;\
   model_name = NAME;\
   cout << "Initializing model: " << NAME << " , restore = " << restore << endl;\
   cout << "Optomizer: " #OP << endl;\
   {\
   char s;\
   cout << "hit a key to continue";\
   cin >> s;\
   }

void InitLPBranchModel1(bool restore)
{
   INITIALIZE("LPB1\\LPB1", optoADAM)
   gModelBranches = 1;

   optoADAM::B1 = 0.9;
   optoADAM::B2 = 0.999;
   optoLowPass::Momentum = 0.8;

   LayerList.clear();

   // Convolution Layer -----------------------------------------
   // Type: FilterLayer2D
   clSize size_in(INPUT_ROWS, INPUT_COLS);
   clSize size_out(INPUT_ROWS, 4);
   clSize size_kern(INPUT_ROWS, INPUT_COLS);
   int chn_in = 1;
   int chn_out = 2;
   int l = 1; // Layer counter
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, chn_out,
         make_unique<actReLU>(),
         //make_unique<actLinear>(), 
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, chn_in)),
         make_shared<OPTO>(),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, static_pointer_cast<iConvoLayer>(pl), pContext1));
   }
   l++;
   //---------------------------------------------------------------
   // Pooling Layer ----------------------------------------------
   // Type: poolAvg2D
   size_in = size_out;
   size_out.Resize(INPUT_ROWS >> 1, 1);
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
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   int len_in = len_out;
   len_out = 16;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, 
         //make_unique<actLeakyReLU>(0.01),
         make_unique<actSigmoid>(),
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
   {
      shared_ptr<LossCrossEntropy> pl = make_shared<LossCrossEntropy>(len_out, 1);
      // DAGErrorLayer(shared_ptr<iLossLayer> _pLayer, shared_ptr<NetContext> _pContext, shared_ptr<ErrorContext> _pEContext) 
      LayerList.push_back(make_shared<DAGErrorLayer>(l, static_pointer_cast<iLossLayer>(pl), pContext2, gpError1));
   }
   l++;
}

void InitLPBranchModel5(bool restore)
{
   INITIALIZE("LPB5\\LPB5", optoADAM)
   gModelBranches = 3;

   optoADAM::B1 = 0.9;
   optoADAM::B2 = 0.999;
   optoLowPass::Momentum = 0.8;

   int l = 1;
   LayerList.clear();

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
   // Pooling Layer ----------------------------------------------
   //size_in = size_out;
   //size_out.Resize(INPUT_ROWS, 1);
   //chn_in = chn_out;

   //assert(!(size_in.rows % size_out.rows));
   //assert(!(size_in.cols % size_out.cols));
   //{
   //   shared_ptr<poolMax2D> pl = make_shared<poolMax2D>(size_in, chn_in, size_out);
   //   //pl->SetEvalPostActivationCallBack(MCB);

   //   LayerList.push_back(make_shared<DAGConvoLayerObj>(static_pointer_cast<iConvoLayer>(pl), pContext1));
   //}
   //l++;
   //---------------------------------------------------------------
   // Pooling Layer ----------------------------------------------
   size_in = size_out;
   //size_out.Resize(INPUT_ROWS >> 1, 1);
   size_out.Resize(INPUT_ROWS, 1);
   chn_in = chn_out;

   assert(!(size_in.rows % size_out.rows));
   assert(!(size_in.cols % size_out.cols));
   {
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
      //pl->SetEvalPostActivationCallBack(MCB);

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
   len_out = 16;
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
   LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, gpError1, restore ? 0 : 6 * 1100, 0.7, 0.98));
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
   // Pooling Layer ----------------------------------------------
   //size_in = size_out;
   //size_out.Resize(INPUT_ROWS, 1);
   //chn_in = chn_out;

   //assert(!(size_in.rows% size_out.rows));
   //assert(!(size_in.cols% size_out.cols));
   //{
   //   shared_ptr<poolMax2D> pl = make_shared<poolMax2D>(size_in, chn_in, size_out);
   //   //pl->SetEvalPostActivationCallBack(MCB);

   //   LayerList.push_back(make_shared<DAGConvoLayerObj>(pl, pContext4));
   //}
   //l++;
   //---------------------------------------------------------------
   // Pooling Layer ----------------------------------------------
   size_in = size_out;
   //size_out.Resize(INPUT_ROWS >> 1, 1);
   size_out.Resize(INPUT_ROWS, 1);
   chn_in = chn_out;

   assert(!(size_in.rows % size_out.rows));
   assert(!(size_in.cols % size_out.cols));
   {
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
      //pl->SetEvalPostActivationCallBack(MCB);

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
   len_out = 32; //64;
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
   // Type: ReLU
   //len_in = len_out;
   //len_out = 48;
   //{
   //   shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actLeakyReLU>(0.01),
   //      restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
   //      dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
   //      make_shared<OPTO>());
   //   LayerList.push_back(make_shared<DAGLayerObj>(l, pl, pContext5));
   //}
   //l++;
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
   LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch2, gpError2, restore ? 0 : 6 * 1100 , 0.6, 0.98));
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
   // Pooling Layer ----------------------------------------------
   //size_in = size_out;
   //size_out.Resize(INPUT_ROWS, 1);
   //chn_in = chn_out;

   //assert(!(size_in.rows% size_out.rows));
   //assert(!(size_in.cols% size_out.cols));
   //{
   //   shared_ptr<poolMax2D> pl = make_shared<poolMax2D>(size_in, chn_in, size_out);
   //   //pl->SetEvalPostActivationCallBack(MCB);

   //   LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, pContext4));
   //}
   //l++;
   //---------------------------------------------------------------
   // Pooling Layer ----------------------------------------------
   size_in = size_out;
   //size_out.Resize(INPUT_ROWS >> 1, 1);
   size_out.Resize(INPUT_ROWS, 1);
   chn_in = chn_out;

   assert(!(size_in.rows% size_out.rows));
   assert(!(size_in.cols% size_out.cols));
   {
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
      //pl->SetEvalPostActivationCallBack(MCB);

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
   // Branch 4 Fully Connected Layer -----------------------------------------
   //
   //int len_out_branch_4 = len_out;
   // This branch will be paired with Test2.
   //shared_ptr<DAGBranchObj> p_branch3 = make_shared<DAGBranchObj>(pContext8, pContext9, false);
   //LayerList.push_back(l, p_branch3);
   //l++;
   //
   //-----------------------------------------------------------------------    
   //--------- setup the fully connected network -------------------------------------------------------------------------
   // 
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   len_in = len_out;
   len_out = 64; // 128;
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
   // Type: ReLU
   //len_in = len_out;
   //len_out = 48;
   //{
   //   shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actLeakyReLU>(0.01),
   //      restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
   //      dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
   //      make_shared<OPTO>());
   //   LayerList.push_back(make_shared<DAGLayerObj>(l, pl, pContext8));
   //}
   //l++;
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
   //LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch2, gpError2));
   //l++;
   //
   //-----------------------------------------------------------------------

}

void InitLPBranchModel6(bool restore)
{
   INITIALIZE("LPB6\\LPB6", optoLowPass)
   gModelBranches = 4;

   optoADAM::B1 = 0.9;
   optoADAM::B2 = 0.999;
   optoLowPass::Momentum = 0.75;

   int l = 1;
   LayerList.clear();

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

   // Pooling Layer ----------------------------------------------
   size_in = size_out;
   //size_out.Resize(INPUT_ROWS >> 1, 1);
   size_out.Resize(INPUT_ROWS, 1);
   chn_in = chn_out;

   assert(!(size_in.rows % size_out.rows));
   assert(!(size_in.cols % size_out.cols));
   {
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
      //pl->SetEvalPostActivationCallBack(MCB);

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
   len_out = 16;
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
   LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, gpError1, restore ? 0 : 6 * 1100, 0.7, 0.98));
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

   // Pooling Layer ----------------------------------------------
   size_in = size_out;
   //size_out.Resize(INPUT_ROWS >> 1, 1);
   size_out.Resize(INPUT_ROWS, 1);
   chn_in = chn_out;

   assert(!(size_in.rows % size_out.rows));
   assert(!(size_in.cols % size_out.cols));
   {
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
      //pl->SetEvalPostActivationCallBack(MCB);

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
   len_out = 32; //64;
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
   LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch2, gpError2, restore ? 0 : 6 * 1100, 0.6, 0.98));
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

   // Pooling Layer ----------------------------------------------
   size_in = size_out;
   //size_out.Resize(INPUT_ROWS >> 1, 1);
   size_out.Resize(INPUT_ROWS, 1);
   chn_in = chn_out;

   assert(!(size_in.rows % size_out.rows));
   assert(!(size_in.cols % size_out.cols));
   {
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
      //pl->SetEvalPostActivationCallBack(MCB);

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
   len_out = 64; // 128;
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
   LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch3, gpError3, restore ? 0 : 6 * 1100, 0.6, 0.98));
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

   // Pooling Layer ----------------------------------------------
   size_in = size_out;
   //size_out.Resize(INPUT_ROWS >> 1, 1);
   size_out.Resize(INPUT_ROWS, 1);
   chn_in = chn_out;

   assert(!(size_in.rows% size_out.rows));
   assert(!(size_in.cols% size_out.cols));
   {
      shared_ptr<poolAvg2D> pl = make_shared<poolAvg2D>(size_in, chn_in, size_out);
      //pl->SetEvalPostActivationCallBack(MCB);

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
   //int len_out_branch_4 = len_out;
   // This branch will be paired with Test2.
   //shared_ptr<DAGBranchObj> p_branch3 = make_shared<DAGBranchObj>(pContext11, pContext , false);
   //LayerList.push_back(l, p_branch3);
   //l++;
   //
   //-----------------------------------------------------------------------    
   //--------- setup the fully connected network -------------------------------------------------------------------------
   // 
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   len_in = len_out;
   len_out = 64; // 128;
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
   // Branch Test Layer (Test 3) -----------------------------------------
   //
   //LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch3, gpError3, restore ? 0 : 6 * 1100, 0.6, 0.98));
   //l++;
   //
   //-----------------------------------------------------------------------


}

void InitLPBranchModel7(bool restore)
{
   INITIALIZE("LPB7\\LPB7", optoADAM)
   gModelBranches = 3;

   optoADAM::B1 = 0.9;
   optoADAM::B2 = 0.999;
   optoLowPass::Momentum = 0.8;

   clSize size_kernel(4, 4);

   int l = 1;
   LayerList.clear();

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
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      pl->SetEvalPostActivationCallBack(MVCB);
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, pContext1));
   }
   l++;
   //---------------------------------------------------------------
   // Pooling Layer ----------------------------------------------
   //size_in = size_out;
   //size_out.Resize(INPUT_ROWS >> 1, 1);
   //chn_in = chn_out;

   //assert(!(size_in.rows % size_out.rows));
   //assert(!(size_in.cols % size_out.cols));
   //{
   //   shared_ptr<poolMax2D> pl = make_shared<poolMax2D>(size_in, chn_in, size_out);
   //   //pl->SetEvalPostActivationCallBack(MCB);

   //   LayerList.push_back(make_shared<DAGConvoLayerObj>(static_pointer_cast<iConvoLayer>(pl), pContext1));
   //}
   //l++;
   //---------------------------------------------------------------

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
   len_out = 16;
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
   LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, gpError1, restore ? 0 : 6 * 1100, 0.7, 0.98));
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
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      //pl->SetEvalPostActivationCallBack(MCB);
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, pContext4));
   }
   l++;
   //---------------------------------------------------------------
   // Pooling Layer ----------------------------------------------
   //size_in = size_out;
   //size_out.Resize(INPUT_ROWS >> 1, 1);
   //chn_in = chn_out;

   //assert(!(size_in.rows% size_out.rows));
   //assert(!(size_in.cols% size_out.cols));
   //{
   //   shared_ptr<poolMax2D> pl = make_shared<poolMax2D>(size_in, chn_in, size_out);
   //   //pl->SetEvalPostActivationCallBack(MCB);

   //   LayerList.push_back(make_shared<DAGConvoLayerObj>(pl, pContext4));
   //}
   //l++;
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
   len_out = 32; //64;
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
   // Type: ReLU
   //len_in = len_out;
   //len_out = 48;
   //{
   //   shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actLeakyReLU>(0.01),
   //      restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
   //      dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
   //      make_shared<OPTO>());
   //   LayerList.push_back(make_shared<DAGLayerObj>(l, pl, pContext5));
   //}
   //l++;
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
   LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch2, gpError2, restore ? 0 : 6 * 1100, 0.6, 0.98));
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
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      //pl->SetEvalPostActivationCallBack(MCB);
      LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, pContext7));
   }
   l++;
   //---------------------------------------------------------------
   // Pooling Layer ----------------------------------------------
   //size_in = size_out;
   //size_out.Resize(INPUT_ROWS >> 1, 1);
   //chn_in = chn_out;

   //assert(!(size_in.rows% size_out.rows));
   //assert(!(size_in.cols% size_out.cols));
   //{
   //   shared_ptr<poolMax2D> pl = make_shared<poolMax2D>(size_in, chn_in, size_out);
   //   //pl->SetEvalPostActivationCallBack(MCB);

   //   LayerList.push_back(make_shared<DAGConvoLayerObj>(l, pl, pContext4));
   //}
   //l++;
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
   // Branch 4 Fully Connected Layer -----------------------------------------
   //
   //int len_out_branch_4 = len_out;
   // This branch will be paired with Test2.
   //shared_ptr<DAGBranchObj> p_branch3 = make_shared<DAGBranchObj>(pContext8, pContext9, false);
   //LayerList.push_back(l, p_branch3);
   //l++;
   //
   //-----------------------------------------------------------------------    
   //--------- setup the fully connected network -------------------------------------------------------------------------
   // 
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   len_in = len_out;
   len_out = 64; // 128;
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
   // Type: ReLU
   //len_in = len_out;
   //len_out = 48;
   //{
   //   shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actLeakyReLU>(0.01),
   //      restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
   //      dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kaiming, 1)),
   //      make_shared<OPTO>());
   //   LayerList.push_back(make_shared<DAGLayerObj>(l, pl, pContext8));
   //}
   //l++;
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
   //LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch2, gpError2));
   //l++;
   //
   //-----------------------------------------------------------------------

}
void InitLPBranchModel8(bool restore)
{
   INITIALIZE("LPB8\\LPB8", optoADAM)
   gModelBranches = 4;

   optoADAM::B1 = 0.9;
   optoADAM::B2 = 0.999;
   optoLowPass::Momentum = 0.75;

   clSize size_kernel(4, 4);

   int l = 1;
   LayerList.clear();

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
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      pl->SetEvalPostActivationCallBack(MVCB);
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
   len_out = 16;
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
   LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, gpError1, restore ? 0 : 6 * 1100, 0.7, 0.98));
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
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      pl->SetEvalPostActivationCallBack(MVCB);
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
   len_out = 32; //64;
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
   LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch2, gpError2, restore ? 0 : 6 * 1100, 0.6, 0.98));
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
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      pl->SetEvalPostActivationCallBack(MVCB);
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
   len_out = 64; // 128;
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
   LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch3, gpError3, restore ? 0 : 6 * 1100, 0.6, 0.98));
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
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      pl->SetEvalPostActivationCallBack(MVCB);
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
   //int len_out_branch_4 = len_out;
   // This branch will be paired with Test2.
   //shared_ptr<DAGBranchObj> p_branch3 = make_shared<DAGBranchObj>(pContext11, pContext , false);
   //LayerList.push_back(l, p_branch3);
   //l++;
   //
   //-----------------------------------------------------------------------    
   //--------- setup the fully connected network -------------------------------------------------------------------------
   // 
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   len_in = len_out;
   len_out = 64; // 128;
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
   // Branch Test Layer (Test 3) -----------------------------------------
   //
   //LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch3, gpError3, restore ? 0 : 6 * 1100, 0.6, 0.98));
   //l++;
   //
   //-----------------------------------------------------------------------


}
void InitLPBranchModel9(bool restore)
{
   INITIALIZE("LPB9\\LPB9", optoADAM)
   gModelBranches = 5;

   optoADAM::B1 = 0.9;
   optoADAM::B2 = 0.999;
   optoLowPass::Momentum = 0.75;

   clSize size_kernel(4, 4);

   int l = 1;
   LayerList.clear();

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
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      pl->SetEvalPostActivationCallBack(MVCB);
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
   len_out = 16;
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
   LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch1, gpError1, restore ? 0 : 10 * 1000, 0.6, 0.98));
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
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      pl->SetEvalPostActivationCallBack(MVCB);
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
   len_out = 32; //64;
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
   LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch2, gpError2, restore ? 0 : 10 * 1000, 0.6, 0.98));
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
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      pl->SetEvalPostActivationCallBack(MVCB);
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
   len_out = 64; // 128;
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
   LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch3, gpError3, restore ? 0 : 10 * 1000, 0.6, 0.98));
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
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      pl->SetEvalPostActivationCallBack(MVCB);
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
   len_out = 64; // 128;
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
   LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch4, gpError4, restore ? 0 : 10 * 1000, 0.6, 0.98));
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
      shared_ptr<Filter> pl = make_shared<Filter>(size_in, chn_in, size_out, size_kernel);
      pl->SetEvalPostActivationCallBack(MVCB);
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
   len_out = 80; // 160;
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
   //LayerList.push_back(make_shared<DAGExitTest>(l, gpExit, p_branch5, gpError5, restore ? 0 : 6 * 1100, 0.6, 0.98));
   //l++;
   //
   //-----------------------------------------------------------------------

}

typedef void (*InitModelFunction)(bool);

InitModelFunction InitModel = InitLPBranchModel7;

void SaveModelWeights()
{
   int l = 1;
   for (const auto& lit : LayerList) {
      lit->Save(make_shared<OWeightsCSVFile>(path, model_name + "." + to_string(l)));
      lit->Save(make_shared<OMultiWeightsBMP>(path, model_name + "." + to_string(l)));
      lit->Save(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l++)));
   }
}

class MatrixManipulator
{
   // Set N=0 and M=1 to produce just the origional.
   // 
   // The number of shifts plus the origional.  This number
   // can only be changed in conjunction with work on the shift method.
   //const int N = 1;
   const int N = 5;  
   // The number of rotations plus zero rotation.
   // This number should be odd.
   //const int M = 1;
   const int M = 5;
   // How much to shift the image left or right.
   const int SHIFT = 1;
   const int LpRows;
   const int LpCols;
   int S;
   int A;
   int C;
   bool NoManip;
   Matrix Base;
   Matrix ShiftState;
   Matrix LPState;
   ColVector angle_label;
   int angle;
   LogPolarSupportMatrix lpsm;
public:
   MatrixManipulator(Matrix m, int lprows, int lpcols, bool no_manipulation = false) :
      NoManip(no_manipulation),
      Base(m),
      LPState(lprows, lpcols),
      LpRows(lprows),
      LpCols(lpcols),
      S(0),
      A(0),
      C(0),
      angle_label(M)
   {
      angle = M >> 1;
      ShiftState.resize(Base.rows(), Base.cols());
      lpsm = PrecomputeLogPolarSupportMatrix(28, 28, lprows, lpcols);
      begin();
   }

   bool isDone()
   {

      // Use for no shift or rotate.
      if (NoManip) {
         return C == 1;
      }
      //return (S == N && A == M);
      return (S == N );      
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
      if (S == 0 && A < M) {
         //if (A < M) {
         rotate();
         angle = A;
         A++;
      }
      else {
         A = 0;
         angle = M >> 1;
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

   ColVector& AngleLabel() {
      angle_label.setZero();
      angle_label(angle) = 1.0;
      return angle_label;
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

void ComputeCentroid(unsigned int& x, unsigned int& y, Matrix& m) 
{
      int rows = m.rows();
      int cols = m.cols();
      int total_x = 0;
      int total_y = 0;
      int count = 0;

      // Iterate through the matrix to calculate the centroid
      for (int i = 0; i < rows; i++) {
         for (int j = 0; j < cols; j++) {
            int value = m(i,j);
            if (value > 0) {
               total_x += j * value;
               total_y += i * value;
               count += value;
            }
         }
      }

      // Calculate the centroid
      if (count > 0) {
         x = total_x / count;
         y = total_y / count;
      }
}

int getch_noblock() {
   if (_kbhit())
      return _getch();
   else
      return -1;
}

void TestModel(string dataroot)
{
   MNISTReader reader1(dataroot + "\\test\\t10k-images-idx3-ubyte",
      dataroot + "\\test\\t10k-labels-idx1-ubyte");

   bool bsave = false;
   char c;
   cout << "Do you want to save incorrect images to disk? y/n ";
   cin >> c;
   cout << endl;
   if (c == 'y' || c == 'Y') {
      bsave = true;
   }

   LogPolarSupportMatrix lpsm = PrecomputeLogPolarSupportMatrix(28, 28, INPUT_ROWS, INPUT_COLS+LP_WASTE);

   ColVector X;

   double avg_e = 0.0;
   int count = 0;
   long correct = 0;

   while (reader1.read_next()) {
      count++;
      X = reader1.data();
      ErrorContext::label = reader1.label();
      Matrix temp(28, 28);
      TrasformMNISTtoMatrix(temp, X);
      ScaleToOne(temp.data(), (int)(temp.rows() * temp.cols()));
#ifdef LOGPOLAR
      pContext1->Reset();
      pContext1->m[0].resize(INPUT_ROWS, INPUT_COLS);
      Matrix mlp(INPUT_ROWS, INPUT_COLS + LP_WASTE);
      ConvertToLogPolar(temp, mlp, lpsm);
      pContext1->m[0] = mlp.block(0, LP_WASTE - 1, INPUT_ROWS, INPUT_COLS);

      //ConvertToLogPolar(temp, pContext1->m[0], lpsm);

#else
      pContext1->m[0] = temp;
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

      if (gpError1->correct || gpError2->correct || gpError3->correct || gpError4->correct || gpError5->correct) {
         correct++;
         /*
         // NOTE: Enable the MVCB above if these lines are uncommented.
         if (gpError1->correct) {
            int nl = GetLabel(ErrorContext::label);
            cout << "number " << nl << " correct. Do you want to save it?  Enter Y for yes.";
            char c;
            cin >> c;
            cout << endl;
            if (c == 'Y' || c == 'y') {
               MVCB->Save(nl);
            }
         }
         */
      }
      else if(bsave) {
         MakeMatrixImage(path + "\\wrong." + to_string(count) + ".bmp", temp);
      }
   }
   std::cout << " correct: " << correct << endl;
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
#ifdef FFT
   cout << "Using FFT convolution." << endl;
#ifdef CYCLIC
   cout << "Using Cyclic convolution in the row direction." << endl;
#endif
#else
   cout << "Using linear convolution." << endl;
#endif // FFT

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
               pContext1->Reset();
               pContext1->m[0].resize(INPUT_ROWS, INPUT_COLS);
               ConvertToLogPolar(temp, pContext1->m[0], lpsm);
            #else
               pContext1->m[0] = temp;
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
               double le = gpError2->error;
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

               if (gpError1->correct) {
                  correct1++;
               }
               else if (gpError2->correct) {
                  correct2++;
               }
               else if (gpError3->correct) {
                  correct3++;
               }
               else if (gpError4->correct) {
                  correct4++;
               }

               //*******************************************************
               //            Backward Pass
               //
               // REVIEW: To stop training on an early branch (like branch 1) the 
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

void Train2(int nloop, string dataroot, double eta, int load )
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
#ifdef FFT
   cout << "Using FFT convolution." << endl;
#ifdef CYCLIC
   cout << "Using Cyclic convolution in the row direction." << endl;
#endif
#else
   cout << "Using linear convolution." << endl;
#endif // FFT
   cout << "Using " << Eigen::nbThreads() << " threads." << endl;

   MNISTReader reader(dataroot + "\\train\\train-images-idx3-ubyte",
      dataroot + "\\train\\train-labels-idx1-ubyte");

   InitModel(load > 0 ? true : false);

   ErrorOutput err_out(path, model_name);
   ErrorOutput err_out1(path, model_name + "_1");
   ErrorOutput err_out2(path, model_name + "_2");
   ErrorOutput err_out3(path, model_name + "_3");
   ErrorOutput err_out4(path, model_name + "_4");
   ErrorOutput err_out5(path, model_name + "_5");
   //ClassifierStats stat_class;
   //ClassifierStats stat_angle_class;


   const int reader_batch = 1000;  // Should divide into 60K
   const int batchs_per_epoch = 60;

   // One of the LP papers shows a better way to compute resampling corrdinates.
   LogPolarSupportMatrix lpsm = PrecomputeLogPolarSupportMatrix(28, 28, INPUT_ROWS, INPUT_COLS+LP_WASTE);

   double e = 0;
   int avg_n;

   struct Sample_Pair {
      Matrix m;
      ColVector y;
      Sample_Pair(){}
      Sample_Pair(Matrix _m, ColVector _y) : m(_m), y(_y) {}
   };

   vector< Sample_Pair> samples;
   Matrix temp(28, 28);

   cout << "Loading the training data." << endl;
   for (int k = 0; k < batchs_per_epoch; k++) {
      cout << "*";
      MNISTReader::MNIST_list dl = reader.read_batch(reader_batch);
      for (MNISTReader::MNIST_Pair& mp : dl) {
         TrasformMNISTtoMatrix(temp, mp.x);
         ScaleToOne(temp.data(), (int)(temp.rows() * temp.cols()));
         #ifdef LOGPOLAR
            Matrix mlp(INPUT_ROWS, INPUT_COLS + LP_WASTE);
            ConvertToLogPolar(temp, mlp, lpsm);
            samples.emplace_back(mlp.block(0, LP_WASTE-1, INPUT_ROWS, INPUT_COLS), mp.y);
         #else
            samples.push_back(Sample_Pair(temp, mp.y));
         #endif
      }
   }
   cout << endl;

   std::random_device rd;     // only used once to initialise (seed) engine
   std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
   std::uniform_int_distribution<int> uni(0, samples.size() - 1); // guaranteed unbiased

   int n = 0;
   int b = 0;
   int t = 0;
   unsigned long correct1 = 0;
   unsigned long correct2 = 0;
   unsigned long correct3 = 0;
   unsigned long correct4 = 0;
   unsigned long correct5 = 0;
   unsigned long early_stop = 0;

   for (int loop = 0; loop < nloop; loop++) {

      e = 0;
      avg_n = 1;
      b = 0;

      while (b < 60000) {
         b++;
         n = uni(rng); // Select a random entry out of the batch.

         // Total try counter.
         t++;

         //*******************************************************
         //            Forward Pass
         pContext1->m[0] = samples[n].m;
         ErrorContext::label = samples[n].y;
         size_t fwd_count = 0;
         do {
            LayerList[fwd_count]->Eval();
            fwd_count++; // The loop exits at last count + 1.  This is by design.
         } while ((!gpExit->stop) && (fwd_count < LayerList.size()));
         gpExit->stop = false;
         //*******************************************************

         double le = gpError2->error;

         double a = 1.0 / (double)(avg_n);
         avg_n++;
         double d = 1.0 - a;
         e = a * le + d * e;

         if (gpError1->correct) {
            correct1++;
         }
         else if (gpError2->correct) {
            correct2++;
         }
         else if (gpError3->correct) {
            correct3++;
         }
         else if (gpError4->correct) {
            correct4++;
         }
         else if (gpError5->correct) {
            correct5++;
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
            double ac1 = 100.0 * (double)correct1 / t;
            double ac2 = (t - correct1) > 0 ? 100.0 * (double)correct2 / (t - correct1) : 100.0;
            double ac3 = (t - correct1 - correct2) > 0 ? 100.0 * (double)correct3 / (t - correct1 - correct2) : 100.0;
            double ac4 = (t - correct1 - correct2 - correct3) > 0 ? 100.0 * (double)correct4 / (t - correct1 - correct2 - correct3) : 100.0;
            double ac5 = (t - correct1 - correct2 - correct3 - correct4) > 0 ? 100.0 * (double)correct5 / (t - correct1 - correct2 - correct3 - correct4) : 100.0;

            err_out.Write(correct1 + correct2 + correct3 + correct4);
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
               << "\ttotal correct: " << correct1 + correct2 + correct3 + correct4 + correct5 << endl;
            t = 0;
            correct1 = 0;
            correct2 = 0;
            correct3 = 0;
            correct4 = 0;
            correct5 = 0;
            early_stop = 0;

            char c = getch_noblock();
            if (c == 'x' || c == 'X') {
               goto JMP;
            }
         }
      }

   }

   JMP:

   TestModel(dataroot);

   std::cout << "Save? y/n:  ";
   char c;
   std::cin >> c;
   if (c == 'y') {
      SaveModelWeights();
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
#ifdef FFT
   cout << "Using FFT convolution." << endl;
#ifdef CYCLIC
   cout << "Using Cyclic convolution in the row direction." << endl;
#endif
#else
   cout << "Using linear convolution." << endl;
#endif // FFT


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
#ifdef LOGPOLAR
         Matrix mlp(INPUT_ROWS, INPUT_COLS);
         ConvertToLogPolar(temp, mlp, lpsm);
#else
         mlp = temp;
#endif

         //*******************************************************
         //            Forward Pass
         pContext1->m[0] = mlp;
         ErrorContext::label = mp.y;
         fwd_count = 0;
         do {
            LayerList[fwd_count]->Eval();
            fwd_count++; // The loop exits at last count + 1.  This is by design.
         } while ((!gpExit->stop) && (fwd_count < LayerList.size()));
         gpExit->stop = false;
         //*******************************************************

         if (gpError1->correct) {
            samples1.emplace_back(mlp, mp.y);
         }
         else if (gpError2->correct) {
            samples2.emplace_back(mlp, mp.y);
         }
         else if (gpError3->correct) {
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
            pContext1->m[0] = samples1[n].m;
            ErrorContext::label = samples1[n].y;
         }
         else {
            // REIVEW: 2 or 3 or 4 !!!!!!
            int n = uni2(rng);
            b++;

            // Total try counter.
            t++;
            // REIVEW: 2 or 3 or 4 !!!!!!
            pContext1->m[0] = samples2[n].m;
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

            double le = gpError2->error;

            double a = 1.0 / (double)(avg_n);
            avg_n++;
            double d = 1.0 - a;
            e = a * le + d * e;

            if (gpError1->correct) {
               correct1++;
            }
            else if (gpError2->correct) {
               correct2++;
            }
            else if (gpError3->correct) {
               correct3++;
            }
            else if (gpError4->correct) {
               correct4++;
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

void Test(string dataroot)
{
   InitModel(true);
   TestModel(dataroot);

   std::cout << "Hit a key and press Enter to continue.";
   char c;
   std::cin >> c;}

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

int main(int argc, char* argv[])
{
   try {
      std::cout << "Starting Convolution MNIST\n";
      string dataroot = "C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data";

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
      if (argc > 1 && string(argv[1]) == "train2") {
         if (argc < 3) {
            cout << "Not enough parameters.  Parameters: train2 | epochs | eta | read stored coefs (0|1) [optional] | dataroot [optional] | path [optional]" << endl;
            return 0;
         }
         double eta = atof(argv[3]);
         int load = 0;
         if (argc > 4) { load = atoi(argv[4]); }
         if (argc > 5) { dataroot = argv[5]; }
         if (argc > 6) { path = argv[6]; }

         Train2(atoi(argv[2]), dataroot, eta, load);
      }
      if (argc > 1 && string(argv[1]) == "train3") {
         if (argc < 3) {
            cout << "Not enough parameters.  Parameters: train3 | epochs | eta | read stored coefs (0|1) [optional] | dataroot [optional] | path [optional]" << endl;
            return 0;
         }
         double eta = atof(argv[3]);
         int load = 0;
         if (argc > 4) { load = atoi(argv[4]); }
         if (argc > 5) { dataroot = argv[5]; }
         if (argc > 6) { path = argv[6]; }

         Train3(atoi(argv[2]), dataroot, eta, load);
      }
      else if (argc > 1 && string(argv[1]) == "test") {

         if (argc < 1) {
            cout << "Not enough parameters.  Parameters: test | dataroot [optional] | path [optional]" << endl;
            return 0;
         }

         if (argc > 2) { dataroot = argv[2]; }
         if (argc > 3) { path = argv[3]; }

         Test(dataroot);
      }
      else if (argc > 2 && string(argv[1]) == "smooth") {

         if (argc < 2) {
            cout << "Not enough parameters.  Parameters: smooth | layer # | sigma [optional] | path [optional]" << endl;
            return 0;
         }

         double sigma = 2.0;
         unsigned int ln = 3;
         if (argc > 2) { ln = atoi(argv[2]); }
         if (argc > 3) { sigma = atof(argv[3]); }
         if (argc > 4) { path = argv[4]; }

         FilterTop(ln,sigma);
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

