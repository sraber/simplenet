// SNBranchMNIST.cpp : This file contains the 'main' function. Program execution begins and ends there.
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

//typedef vector< shared_ptr<Layer> > layer_list;
//typedef vector< shared_ptr<iConvoLayer> > convo_layer_list;
//convo_layer_list ConvoLayerList;
//layer_list LayerList;

shared_ptr<iLossLayer> loss;
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

   // For now restrict to square matrix.
   runtime_assert(out_rows == out_cols)
      runtime_assert(in_rows == in_cols)
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
   // Not right.
   // double rad = cols > rows ? (double)rows : (double)cols;

   // For now restrict to square matrix.
   runtime_assert(rows == cols)
   runtime_assert(m.rows() == m.cols())

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
   //void Reset(const int rows) {
   //   v.resize(rows);
   //}
};

class CovNetContext
{
public:
   vector_of_matrix m;
   //vector_of_matrix g;
   CovNetContext() : m(1) {}
   //void Reset(const int rows, const int cols) {
   //   m[0].resize(rows,cols);
   //}
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
   shared_ptr<iConvoLayer> pLayer;
   shared_ptr<CovNetContext> pContext;
public:
   DAGConvoLayerObj(shared_ptr<iConvoLayer> _pLayer, shared_ptr<CovNetContext> _pContext ) :
      pLayer(std::move(_pLayer)),
      pContext(_pContext)
   {}
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
};

class DAGFlattenObj : public iDAGObj
{
   shared_ptr<iConvoLayer> pLayer;
   shared_ptr<CovNetContext> pCovContext;
   shared_ptr<NetContext> pLayerContext; 
public:
   DAGFlattenObj(shared_ptr<iConvoLayer> _pLayer, shared_ptr<CovNetContext> _pContext1, shared_ptr<NetContext> _pContext2) :
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
   DAGLayerObj(shared_ptr<iLayer> _pLayer, shared_ptr<NetContext> _pContext) :
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

class DAGBranchObj : public iDAGObj
{
   shared_ptr<NetContext> pContext1;
   shared_ptr<NetContext> pContext2;
   bool bBackprop;
public:
   DAGBranchObj(shared_ptr<NetContext> _pContext1, shared_ptr<NetContext> _pContext2, bool backprop_branch = true) :
      pContext1(_pContext1),
      pContext2(_pContext2),
      bBackprop(backprop_branch)
   {}
   void Eval() {
      pContext2->v = pContext1->v;
   }
   void BackProp() {
      if (bBackprop) {
         pContext1->g += pContext2->g;
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
   // Note: The contexts are assigned at creation time.  They can't be accessed until
   //       Eval is called.
   DAGJoinObj(shared_ptr<NetContext> _pContext1, shared_ptr<NetContext> _pContext2) :
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

// This is the main input context.
   // Convo Layer
shared_ptr<CovNetContext> pContext1 = make_shared<CovNetContext>();
// FC Layer main branch
shared_ptr<NetContext> pContext2 = make_shared<NetContext>();
// FC Layer angle branch
shared_ptr<NetContext> pContext3 = make_shared<NetContext>();
shared_ptr<NetContext> pContext4 = make_shared<NetContext>();

shared_ptr<iLossLayer> loss_branch;


void InitLPBranchModel1(bool restore)
{
   model_name = "LPB1\\LPB1";
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
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, kern_per_chn,
         make_unique<actReLU>(),
         //make_unique<actLinear>(), 
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, chn_in)),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(static_pointer_cast<iConvoLayer>(pl), pContext1));
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

      LayerList.push_back(make_shared<DAGConvoLayerObj>(static_pointer_cast<iConvoLayer>(pl), pContext1));
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
      LayerList.push_back(make_shared<DAGFlattenObj>(static_pointer_cast<iConvoLayer>(pl), pContext1, pContext2));
   }
   l++;
   //---------------------------------------------------------------      
   // 
   //--------- setup the fully connected network -------------------------------------------------------------------------
   // 
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   int len_in = len_out;
   len_out = 32;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actLeakyReLU>(0.01),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, 1)));
      LayerList.push_back(make_shared<DAGLayerObj>(static_pointer_cast<iLayer>(pl), pContext2));
   }
   l++;
   //---------------------------------------------------------------  
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   //len_in = len_out;
   //len_out = 16;
   // {
   //shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actLeakyReLU>(0.01),
   //   restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
   //   dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, 1)));
   //LayerList.push_back(make_shared<DAGLayerObj>(static_pointer_cast<iLayer>(pl), pContext2));
   // }
   //l++;
   //---------------------------------------------------------------  
   // Fully Connected Layer ---------------------------------------
   // Type: SoftMAX
   len_in = len_out;
   len_out = 10;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actSoftMax>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, 1)));
      LayerList.push_back(make_shared<DAGLayerObj>(static_pointer_cast<iLayer>(pl), pContext2));
   }
   l++;
   //---------------------------------------------------------------      

   // Loss Layer - Not part of network, must be called seperatly.
   // Type: LossCrossEntropy
   loss = make_shared<LossCrossEntropy>(len_out, 1);
   //--------------------------------------------------------------

}
void InitLPBranchModel2(bool restore)
{
   model_name = "LPB2\\LPB2";
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
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(size_in, chn_in, size_out, size_kern, kern_per_chn,
         make_unique<actReLU>(),
         //make_unique<actLinear>(), 
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, chn_in)),
         true); // No bias. true/false  - REVIEW: Should flip the meaning of this switch.
      LayerList.push_back(make_shared<DAGConvoLayerObj>(static_pointer_cast<iConvoLayer>(pl), pContext1));
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

      LayerList.push_back(make_shared<DAGConvoLayerObj>(static_pointer_cast<iConvoLayer>(pl), pContext1));
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
      LayerList.push_back(make_shared<DAGFlattenObj>(static_pointer_cast<iConvoLayer>(pl), pContext1, pContext2));
   }
   l++;
   //---------------------------------------------------------------      
   // 
   // Branch Fully Connected Layer -----------------------------------------
   //
   int len_out_branch = len_out;
   LayerList.push_back(make_shared<DAGBranchObj>(pContext2, pContext3));
   l++;
   //
   //-----------------------------------------------------------------------

   //--------- setup the fully connected network -------------------------------------------------------------------------
   // 
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   int len_in = len_out;
   len_out = 32;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actLeakyReLU>(0.01),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, 1)));
      LayerList.push_back(make_shared<DAGLayerObj>(static_pointer_cast<iLayer>(pl), pContext2));
   }
   l++;
   //---------------------------------------------------------------  
   // Fully Connected Layer ---------------------------------------
   // Type: ReLU
   //len_in = len_out;
   //len_out = 16;
   // {
   //shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actLeakyReLU>(0.01),
   //   restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
   //   dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, 1)));
   //LayerList.push_back(make_shared<DAGLayerObj>(static_pointer_cast<iLayer>(pl), pContext2));
   // }
   //l++;
   //---------------------------------------------------------------  
   // Fully Connected Layer ---------------------------------------
   // Type: SoftMAX
   len_in = len_out;
   len_out = 10;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in, len_out, make_unique<actSoftMax>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, 1)));
      LayerList.push_back(make_shared<DAGLayerObj>(static_pointer_cast<iLayer>(pl), pContext2));
   }
   l++;
   //---------------------------------------------------------------      

   // Branch Fully Connected Layer -----------------------------------------
   //
   //                                                                  v - don't backprop branch
   LayerList.push_back(make_shared<DAGBranchObj>(pContext2, pContext4, false));
   //LayerList.push_back(make_shared<DAGBranchObj>(pContext2, pContext4, true));
   l++;
   //
   //-----------------------------------------------------------------------

   // Join Fully Connected Layer -----------------------------------------
   //
   len_out_branch += len_out;
   LayerList.push_back(make_shared<DAGJoinObj>(pContext3, pContext4));
   l++;
   //
   //-----------------------------------------------------------------------

   //--------- setup the fully connected branch network -------------------------------------------------------------------------
   // 
   // Fully Branch Connected Layer ---------------------------------------
   // Type: ReLU
   int len_in_branch = len_out_branch;
   len_out_branch = 37;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in_branch, len_out_branch, make_unique<actLeakyReLU>(0.01),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, 1)));
      LayerList.push_back(make_shared<DAGLayerObj>(static_pointer_cast<iLayer>(pl), pContext4));
   }
   l++;
   // Fully Connected Layer ---------------------------------------
   // Type: SoftMAX
   len_in_branch = len_out_branch;

   // REVIEW: This has to stay consistant with the number of rotations.
   len_out_branch = 5;
   {
      shared_ptr<Layer> pl = make_shared<Layer>(len_in_branch, len_out_branch, make_unique<actSoftMax>(),
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>(make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, 1)));

      pl->SetBackpropCallBack(MCB);

      LayerList.push_back(make_shared<DAGLayerObj>(static_pointer_cast<iLayer>(pl), pContext4));
   }
   l++;

   // Loss Layer - Not part of network, must be called seperatly.
   // Type: LossCrossEntropy
   loss = make_shared<LossCrossEntropy>(len_out, 1);
   loss_branch = make_shared<LossCrossEntropy>(len_out_branch, 1);
   //--------------------------------------------------------------

}

typedef void (*InitModelFunction)(bool);

InitModelFunction InitModel = InitLPBranchModel2;

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

   ErrorOutput err_out(path, model_name);
   ClassifierStats stat_class;
   ClassifierStats stat_angle_class;

   const int reader_batch = 1000;  // Should divide into 60K
   const int batch = 100; // Should divide evenly into reader_batch
   const int batch_loop = 11;

   std::random_device rd;     // only used once to initialise (seed) engine
   std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
   std::uniform_int_distribution<int> uni(0, reader_batch - 1); // guaranteed unbiased

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
         while (b < batch) {
            if (retry == 0) {
               n = uni(rng); // Select a random entry out of the batch.
               b++;
            }
            Matrix temp(28, 28);

            TrasformMNISTtoMatrix(temp, dl[n].x);
            ScaleToOne(temp.data(), (int)(temp.rows() * temp.cols()));

            //#ifdef LOGPOLAR
            //            ConvertToLogPolar(temp, m[0], lpsm);
            //#else
            //            m[0] = temp;
            //#endif
            for (MatrixManipulator mm(temp, INPUT_ROWS, INPUT_COLS); !mm.isDone(); mm.next())
            {
               ColVector ay = mm.AngleLabel();
               pContext1->m[0] = mm.get();
               for (auto lli : LayerList) { lli->Eval(); }

               stat_angle_class.Eval(pContext4->v, ay);

               if (retry == 0) {
                  if (stat_class.Eval(pContext2->v, dl[n].y) == false) {
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

               double le = loss->Eval(pContext2->v, dl[n].y);
               double lb = loss_branch->Eval(pContext4->v, ay);
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

               pContext2->g = loss->LossGradient();
               pContext4->g = loss_branch->LossGradient();
               for (layer_list::reverse_iterator idag = LayerList.rbegin(); idag != LayerList.rend(); ++idag) {
                  (*idag)->BackProp();
               }
            }
#ifdef SGD
            // This is stoastic descent.  It is inside the batch loop.
            for (auto lit : LayerList) {
               lit->Update(eta);
            }
#endif
         }

         // if not defined
#ifndef SGD
         //eta = (1.0 / (1.0 + 0.001 * loop)) * eta;
         for (auto lit : LayerList) {
            lit->Update(eta);
         }
#endif
         err_out.Write(stat_class.Correct);
         cout << "count: " << loop << "\terror:" << left << setw(9) << std::setprecision(4) << e << "\tcorrect: " << stat_class.Correct << "\tincorrect: " << stat_class.Incorrect << "\tlabels correct: " << stat_angle_class.Correct << endl;
         stat_angle_class.Reset();
         stat_class.Reset();
      }

      cout << "eta: " << eta << endl;
   }

TESTJMP:

   MNISTReader reader1(dataroot + "\\test\\t10k-images-idx3-ubyte",
      dataroot + "\\test\\t10k-labels-idx1-ubyte");

   stat_angle_class.Reset();
   stat_class.Reset();

   ColVector X;
   ColVector Y;

   double avg_e = 0.0;
   int count = 0;

   while (reader1.read_next()) {
      X = reader1.data();
      Y = reader1.label();
      Matrix temp(28, 28);
      TrasformMNISTtoMatrix(temp, X);
      ScaleToOne(temp.data(), (int)(temp.rows() * temp.cols()));
#ifdef LOGPOLAR
      pContext1->m.resize(1);
      pContext1->m[0].resize(INPUT_ROWS, INPUT_COLS);
      ConvertToLogPolar(temp, pContext1->m[0], lpsm);
#else
      pContext1->m[0] = temp;
#endif

      for (auto lli : LayerList) { lli->Eval(); }

      stat_class.Eval(pContext2->v, Y);
   }

   std::cout << " correct/incorrect " << stat_class.Correct << " , " << stat_class.Incorrect << endl;
   std::cout << "Save? y/n:  ";
   char c;
   std::cin >> c;
   if (c == 'y') {
      SaveModelWeights();
   }
}

int main(int argc, char* argv[])
{
   try {
      std::cout << "Starting Convolution MNIST\n";
      string dataroot = "C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data";
 
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
      else if (argc > 1 && string(argv[1]) == "test") {

         if (argc < 1) {
            cout << "Not enough parameters.  Parameters: test | dataroot [optional] | path [optional]" << endl;
            return 0;
         }

         if (argc > 2) { dataroot = argv[2]; }
         if (argc > 3) { path = argv[3]; }

         //Test(dataroot);
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

