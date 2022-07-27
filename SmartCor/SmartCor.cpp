// SmartCor.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <Eigen>
#include <iostream>
#include <iomanip>
#include <Layer.h>
#include <bmp.h>
#include <map>

typedef vector< shared_ptr<Layer> > layer_list;
typedef vector< shared_ptr<iConvoLayer> > convo_layer_list;
convo_layer_list ConvoLayerList;
layer_list LayerList;

shared_ptr<iLossLayer> loss;
string path = "C:\\projects\\neuralnet\\simplenet\\SmartCor\\weights";
string model_name = "layer";

void Normalize(Matrix& m)
{
   m.normalize();

   //double u = m.array().mean();
   //m.array() = m.array() - u;

   //double std = 0;
   //for (int r = 0; r < m.rows(); r++) {
   //   for (int c = 0; c < m.cols(); c++) {
   //      std += m(r, c) * m(r, c);
   //   }
   //}
   //std = sqrt( std / (double)m.size()-1);

   //m.array() /= std;
}

void CreateCone(Matrix& m, double ro2, double co2, double radius)
{
   for (int r = 0; r < m.rows(); r++) {
      double rc = r - ro2 + 1.0; // -1 accounts for zero offset matrix index.
      double r2 = rc * rc;
      for (int c = 0; c < m.cols(); c++) {
         double cc = c - co2 + 1.0;
         double c2 = cc * cc;
         double cur_radius = sqrt((double)(r2 + c2));
         if (cur_radius < radius) {
            // Amplitude of 0 at radius rising to 1 at center.  It's a cone.
            m(r, c) = 1.0 - cur_radius / radius;
         }
      }
   }
}

void CreateCylinder(Matrix& m, double ro2, double co2, double radius)
{
   m.setZero();
   double pass_rad2 = radius * radius;
   for (int r = 0; r < m.rows(); r++) {
      double rc = r - ro2 + 1.0;;
      double r2 = rc * rc;
      for (int c = 0; c < m.cols(); c++) {
         double cc = c - co2 + 1.0;
         double c2 = cc * cc;
         if ((r2 + c2) < pass_rad2) {

            m(r, c) = 1.0;
         }
      }
   }
}

void CreateNormalDistribution(Matrix& m, double ro2, double co2, double std)
{
   const double den = 0.39894;  // 1/sqrt(2 pi)  need more sig digites
   double a = den / std;
   m.setZero();
   for (int r = 0; r < m.rows(); r++) {
      double rc = r - ro2 + 1.0;
      double r2 = rc * rc;
      for (int c = 0; c < m.cols(); c++) {
         double cc = c - co2 + 1.0;
         double c2 = cc * cc;
         double cur_radius = sqrt((double)(r2 + c2));
         double b = cur_radius / std;
         m(r, c) = 10.0 * a * exp(-0.5 * b * b);
      }
   }
}

class StatsOutput
{
   ofstream owf;
public:
   StatsOutput(string name) : owf(path + "\\"  + name + ".csv", ios::trunc)
   {
   }
   void Write(shared_ptr<FilterLayer2D> fl)
   {
      runtime_assert(owf.is_open());
      runtime_assert(fl);

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
   bool output_norm;
   string Name;
public:
   GradOutput(string name, bool want_norm = true) : 
      owf(path + "\\"  + name + ".csv", ios::trunc),
      output_norm(want_norm),
      Name(name){}
   GradOutput() : output_norm(true) {}
   void Init(string name, bool want_norm = true) {
      //owf.open(path + "\\" + name + ".csv", ios::trunc);
      Name = name;
      output_norm = want_norm;
   }
   void Write(RowVector& g)
   {
      //assert(owf.is_open());
      //owf << g.blueNorm() << endl;
   }
   void Write(ColVector& g)
   {
      //assert(owf.is_open());
      //owf << g.blueNorm() << endl;
   }
   void Write(vector_of_matrix& vg)
   {
      /*
      assert(owf.is_open());
      for (int i = 0; i < vg.size();i++) {
         owf << (output_norm ? vg[i].blueNorm() : vg[i].maxCoeff());
         if (i < vg.size() - 1) {
            owf << ",";
         }
      }
      owf << endl;
      */
      cout << Name << " output: " << endl;
      if (vg[0].rows() == 32) {
         cout << vg[0].block(12, 12, 10, 10) << endl;
      }
      else {
         //cout << vg[0] << endl;
      }

   }
};

void MakeMatrixImage(string file, Matrix m)
{
   pixel_data pixel;
   int rows = (int)m.rows();
   int cols = (int)m.cols();

   unsigned char* pbytes = new unsigned char[rows * cols * sizeof(pixel_data)]; // 24 bit BMP
   unsigned char* pbs = pbytes;
   for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
         double v = m(r, c) * 254;
         pixel.r = static_cast<unsigned char>(v);
         pixel.g = static_cast<unsigned char>(v);
         pixel.b = static_cast<unsigned char>(v);

         std::memcpy(pbs, &pixel, sizeof(pixel_data));
         pbs += sizeof(pixel_data);
      }
   }

   generateBitmapImage(pbytes, rows, cols, cols * sizeof(pixel_data), file);
}

void ScaleToOne(double* pdata, int size)
{
   double max = 0.0;
   double min = 0.0;
   double* pd = pdata;
   double* pde = pd + size;
   for (; pd < pde; pd++) {
      if (max < *pd) { max = *pd; }
      if (min > * pd) { min = *pd; }
   }
   for (pd = pdata; pd < pde; pd++) {
      *pd = (*pd - min) / (max - min);
   }
}

class OMultiWeightsBMP : public iPutWeights{
   string Path;
   string RootName;
public:
   OMultiWeightsBMP(string path, string root_name) : RootName(root_name), Path(path) {}
   void Write(Matrix& m, int k) {
      string str_count;
      str_count = to_string(k);
      string pathname = Path + "\\" + RootName + "." + str_count + ".bmp";
      Matrix temp;
      temp = m;
      ScaleToOne(temp.data(), (int)temp.size());
      MakeMatrixImage(pathname, temp);
   }
};

class SmartCorStats
{
   double Center;
   double Threshold;
   const double acceptable_error = 4.0; // This is pixels.
public:
   int Correct;
   int Incorrect;
   SmartCorStats(double _center, double _thresh) : Center(_center), Threshold(_thresh), Correct(0), Incorrect(0) {}
   void Eval(const Matrix& m, bool is_pattern) {
      int rm = 0;
      int cm = 0;
      bool exceeded_threshold = false;
      double max = Threshold;
      for (int r = 0; r < m.rows(); r++) {
         for (int c = 0; c < m.cols(); c++) {
            if (m(r, c) > max) {
               rm = r;
               cm = c;
               max = m(r, c);
               exceeded_threshold = true;
            }
         }
      }
      double dr = (double)rm - Center;
      double dc = (double)cm - Center;
      double err = sqrt(dr * dr + dc * dc);
      if (exceeded_threshold==is_pattern &&
          err<=acceptable_error ){
         Correct++;
      } else {
         Incorrect++;
      }
      return;
   }
   void Reset()
   {
      Correct = 0;
      Incorrect = 0;
   }
};

const int pattern_size = 32;
const double pattern_center = (double)pattern_size / 2.0;

class InitSmartCorConvoLayer : public iGetWeights{
   double Size;
   double Center;
   int What;
   double Radius;
   int Layer;

   void CreateWave(Matrix& m, double ro2, double co2, double radius)
   {
      m.setZero();
      for (int r = 0; r < m.rows(); r++) {
         double rc = r - ro2 + 1.0;;
         double r2 = rc * rc;
         for (int c = 0; c < m.cols(); c++) {
            double cc = c - co2 + 1.0;
            double c2 = cc * cc;
            double rad = sqrt(r2 + c2);
            m(r, c) = 0.0001 * sin(6.283185307 * 4.0 * rad / 32.0);
         }
      }
   }

public:
   InitSmartCorConvoLayer(double _size, double _center, double _rad, int _what, int layer = 1) : 
      Size(_size), Center(_center), Radius(_rad) ,What(_what), Layer(layer){}
   void ReadConvoWeight(Matrix& m, int k) {
      if (Layer == 1) {
         switch (What) {
         case 1:
            CreateCone(m, Center, Center, Radius);
            break;
         case 2:
            CreateCylinder(m, Center, Center, Radius);
            break;
         case 3:
            CreateWave(m, Center, Center, Radius);
            break;
         default:
            throw runtime_error("incorrect object number.");
         }
         Normalize(m);
      }
      else {
         m.setZero();
         if ((int)floor(Center) == (int)ceil(Center) ){
            Eigen::Index f = (floor(Center) - 1.0);
            m.block(f, f, 2, 2).setConstant(0.25);
         }
         else {
            Eigen::Index cen = (Eigen::Index)(Center - 1.0);
            m(cen, cen ) = 1.0;
         }
      }
   }
   void ReadConvoBias(Matrix& w, int k) {
      w.setZero();
   }
   void ReadFC(Matrix& m) {
      throw runtime_error("InitBigKernelConvoLayer::ReadFC not implemented");
   }
};

class FL2Debug : public FilterLayer2D::iCallBack
{
   string Name;
   void HitKey() {
      cout << "Hit Enter to continue." << endl;
      char c;
      cin >> c;
   }
public:
   FL2Debug(string name) : Name(name) {}
   // Inherited via iCallBack
   //void Propeties(const bool& no_bias, const vector_of_matrix& x, const vector_of_matrix& w, const vector_of_matrix& dw, const vector_of_number& b, const vector_of_number& db, const vector_of_matrix& z) override
   //{
   //   cout << "W[0] norm: " << w[0].blueNorm() << endl << "16 x 16 center: " << endl << w[0].block(8, 8, 16, 16) << endl;
   //   HitKey();
   //}
   void Propeties(const bool& no_bias, const vector_of_matrix& x, const vector_of_matrix& w, const vector_of_matrix& dw, const vector_of_number& b, const vector_of_number& db, const vector_of_matrix& z) override
   {
      cout << Name << endl << "Norms:" << endl
         << "X: " << x[0].blueNorm()
         << " W: " << w[0].blueNorm()
         << " dW: " << dw[0].blueNorm()
         << " Z: " << z[0].blueNorm() << endl;

      cout << x[0].block(12, 12, 10, 10) << endl;

      //HitKey();
   }

};

//--------------------------------------------------------------------
void InitSmartCorA(bool restore)
{
   model_name = "SCA";
   ConvoLayerList.clear();
   LayerList.clear();

   // Convolution Layer 1 -----------------------------------------
   // Type: FilterLayer2D
   int size_in  = pattern_size;
   int size_out = pattern_size;
   int kern = pattern_size;
   int pad = pattern_size/2;
   int kern_per_chn = 1;
   int chn_in = 1;
   int chn_out = kern_per_chn * chn_in;
   int l = 1; // Layer counter
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(clSize(size_in, size_in), chn_in, clSize(size_out, size_out), clSize(kern, kern), kern_per_chn,
         //make_unique<actReLU>(),
         make_unique<actLinear>(), 
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToRandom>(0.1)),
         //dynamic_pointer_cast<iGetWeights>(make_shared<InitSmartCorConvoLayer>(pattern_size, pattern_center, pattern_size / 4, 3)),
         true);

      //pl->SetEvalPreActivationCallBack(make_shared<FL2Debug>("Pre"));
      //pl->SetEvalPostActivationCallBack(make_shared<FL2Debug>("Post"));

      ConvoLayerList.push_back(pl);

   }
   l++;
   //---------------------------------------------------------------
 

   // Flattening Layer 2 --------------------------------------------
   // Type: Flatten2D
   size_in  = size_out;
   chn_in = chn_out;
   size_out = size_in * size_in * chn_in;
   chn_out = 1;
   ConvoLayerList.push_back( make_shared<Flatten2D>(clSize(size_in, size_in), chn_in) );
   l++;
   //---------------------------------------------------------------      
   

   // Loss Layer - Not part of network, must be called seperatly.
   loss = make_shared<LossL4>(size_out, 1);   
   //--------------------------------------------------------------

}
//---------------------------- End InitSmartCorA ---------------------------------------
//--------------------------------------------------------------------
void InitSmartCorB(bool restore)
{
   model_name = "SCB";
   ConvoLayerList.clear();
   LayerList.clear();

   // Convolution Layer 1 -----------------------------------------
   // Type: FilterLayer2D
   int size_in  = pattern_size;
   int size_out = pattern_size;
   int kern = pattern_size;
   int pad = pattern_size/2;
   int kern_per_chn = 1;
   int chn_in = 1;
   int chn_out = kern_per_chn * chn_in;
   int l = 1; // Layer counter
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(clSize(size_in, size_in), chn_in, clSize(size_out, size_out), clSize(kern, kern), kern_per_chn,
         //make_unique<actReLU>(),
         make_unique<actLinear>(), 
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         //dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToRandom>(0.1)),
         dynamic_pointer_cast<iGetWeights>(make_shared<InitSmartCorConvoLayer>(pattern_size, pattern_center, pattern_size / 4, 2)),
         true);

      //pl->SetEvalPreActivationCallBack(make_shared<FL2Debug>("Pre"));
      //pl->SetEvalPostActivationCallBack(make_shared<FL2Debug>("Post"));

      ConvoLayerList.push_back(pl);

   }
   l++;
   //---------------------------------------------------------------
   
   // Convolution Layer 2 -----------------------------------------
   // Type: FilterLayer2D
   size_in  = pattern_size;
   size_out = pattern_size;
   kern = pattern_size;
   pad = pattern_size/2;
   kern_per_chn = 1;
   chn_in = 1;
   chn_out = kern_per_chn * chn_in;
   {
      shared_ptr<FilterLayer2D> pl = make_shared<FilterLayer2D>(clSize(size_in, size_in), chn_in, clSize(size_out, size_out), clSize(kern, kern), kern_per_chn,
         //make_unique<actReLU>(),
         make_unique<actLinear>(), 
         restore ? dynamic_pointer_cast<iGetWeights>(make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) :
         //dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToRandom>(0.01)),
         dynamic_pointer_cast<iGetWeights>(make_shared<InitSmartCorConvoLayer>( pattern_size, pattern_center, pattern_size / 4, 2, 2)),
         true);

      //pl->SetEvalPreActivationCallBack(make_shared<FL2Debug>("Pre"));
      //pl->SetBackpropCallBack(make_shared<FL2Debug>("Backprop"));

      ConvoLayerList.push_back(pl);

   }
   l++;
   //---------------------------------------------------------------
 

   // Flattening Layer 2 --------------------------------------------
   // Type: Flatten2D
   size_in  = size_out;
   chn_in = chn_out;
   size_out = size_in * size_in * chn_in;
   chn_out = 1;
   ConvoLayerList.push_back( make_shared<Flatten2D>(clSize(size_in, size_in), chn_in) );
   l++;
   //---------------------------------------------------------------      
   

   // Loss Layer - Not part of network, must be called seperatly.
   // Type: LossCrossEntropy
   loss = make_shared<LossL2>(size_out, 1);   
   //--------------------------------------------------------------

}
//---------------------------- End InitSmartCorB ---------------------------------------

void InitSmartCor(bool restore)
{
   model_name = "SC";
   ConvoLayerList.clear();
   LayerList.clear();

   // Convolution Layer 1 -----------------------------------------
   // Type: FilterLayer2D
   int size_in  = pattern_size;
   int size_out = pattern_size;
   int kern = pattern_size;
   int pad = pattern_size/2;
   int kern_per_chn = 1;
   int chn_in = 1;
   int chn_out = kern_per_chn * chn_in;
   int l = 1; // Layer counter
   ConvoLayerList.push_back( make_shared<FilterLayer2D>(clSize(size_in, size_in), chn_in, clSize(size_out, size_out), clSize(kern, kern), kern_per_chn, 
                           make_unique<actReLU>(), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     //dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, chn_in)),
                                     dynamic_pointer_cast<iGetWeights>( make_shared<InitSmartCorConvoLayer>(pattern_size, pattern_center, pattern_size/4, 2)),
                           true )
                           );
   l++;
   //---------------------------------------------------------------
 

   // Flattening Layer 2 --------------------------------------------
   // Type: Flatten2D
   size_in  = size_out;
   chn_in = chn_out;
   size_out = size_in * size_in * chn_in;
   chn_out = 1;
   ConvoLayerList.push_back( make_shared<Flatten2D>(clSize(size_in, size_in), chn_in) );
   l++;
   //---------------------------------------------------------------      

   //--------- setup the fully connected network -------------------------------------------------------------------------

   // Fully Connected Layer 3 ---------------------------------------
   // Type: ReLU
   //size_in = size_out;
   //size_out = size_in;
   //LayerList.push_back(make_shared<Layer>(size_in, size_out, new actReLU(size_out), 
   //                        restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
   //                                  dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, 1))) );   l++;
   //---------------------------------------------------------------      

   // Fully Connected Layer 4 ---------------------------------------
   // Type: SoftMAX
   size_in = size_out;
   size_out = size_in;
   LayerList.push_back(make_shared<Layer>(size_in, size_out, make_unique<actSoftMax>(), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, 1))) );   l++;
   //---------------------------------------------------------------      

   // Loss Layer - Not part of network, must be called seperatly.
   // Type: LossCrossEntropy
   loss = make_shared<LossL2>(size_out, 1);   
   //--------------------------------------------------------------

}
//---------------------------- End InitSmartCor ---------------------------------------

typedef void (*InitModelFunction)(bool);

InitModelFunction InitModel = InitSmartCorA;
//InitModelFunction InitModel = InitSmartCorB;

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

//std::random_device rd;
//std::mt19937 rng(rd());
//std::uniform_real_distribution<double> und_low(-1.0,0.0);
//std::uniform_real_distribution<double> und_high(std::numeric_limits<double>::epsilon(),1.0);

//void CreateNoise(Matrix& m)
//{
//   double low = und_low(rng);
//   double high = und_high(rng);
//
//   std::uniform_real_distribution<double> value(low, high);
//   for (int r = 0; r < m.rows(); r++) {
//      for (int c = 0; c < m.cols(); c++) {
//         m(r, c) = value(rng);
//      }
//   }
//}

//vector_of_matrix CreateBatch(int size)
//{
//   const double rad_small = (double)pattern_size / 8.0;
//   const double rad_big = (double)pattern_size / 2.0;
//
//   Matrix sum(pattern_size, pattern_size);
//   Matrix std(pattern_size, pattern_size);
//   Matrix tmp(pattern_size, pattern_size);
//   sum.setZero();
//   std.setZero();
//
//   vector_of_matrix vom(size);
//   for (Matrix& m : vom) { m.resize(pattern_size, pattern_size); }
//
//   double g = (rad_big - rad_small) / (double)(size - 1);
//   for (int i = 0; i < size; i++) {
//      double r = (double)i * g + rad_small;
//      CreateCone(vom[i], pattern_center, pattern_center, r);
//      sum += vom[i];
//   }
//
//   sum /= (double)size;
//   //cout << sum << endl;
//
//   for (Matrix& m : vom) {
//      tmp = sum - m;
//      tmp.array() *= tmp.array();
//      std += tmp;
//   }
//   std /= (double)size;
//   std.array().cwiseSqrt();
//
//   //cout << std << endl;
//
//   for (Matrix& m : vom) {
//      m -= sum;
//      //cwiseDiv(m.data(), std.data(), m.size());
//
//      cout << m << endl;
//   }
//   
//   return vom;
//}

vector_of_matrix CreateBatch(int size, int what )
{
   const double rad_small = (double)pattern_size / 8.0;
   const double rad_big = (double)pattern_size / 2.0;

   vector_of_matrix vom(size);
   for (Matrix& m : vom) { m.resize(pattern_size, pattern_size); }

   double g = (rad_big - rad_small) / (double)(size - 1);
   for (int i = 0; i < size; i++) {
      double r = (double)i * g + rad_small;
      switch (what) {
      case 1:
         CreateCone(vom[i], pattern_center, pattern_center, r);
         break;
      case 2:
         CreateCylinder(vom[i], pattern_center, pattern_center, r);
         break;
      default:
         throw runtime_error("incorrect object number.");
      }

      Normalize(vom[i]);
   }
   
   return vom;
}
//#define ALLPASS
void Train(int nloop, double eta, int load)
{
#ifdef ALLPASS
   cout << "Running all pass batch" << endl;
#else
   cout << "Running half reject batch" << endl;
#endif

   InitModel(load > 0 ? true : false);

   Matrix y_pattern(pattern_size, pattern_size);
   Matrix y_zero(pattern_size, pattern_size);

   CreateNormalDistribution(y_pattern, pattern_center, pattern_center, 2.0);
   y_zero.setConstant( 0.0);

   Matrix* y = &y_pattern;

//#define STATS
#ifdef STATS
   map<int,GradOutput> lveo;
   map<int,GradOutput> clveo;
   map<int,GradOutput> lvgo;
   map<int,GradOutput> clvgo;

   #define clveo_write(I,M) { if(clveo.find(I)==clveo.end()){ clveo[I].Init("cleval." + to_string(I)); } clveo[I].Write(M); } 
   #define lveo_write(I,M) { if(lveo.find(I)==lveo.end()){ lveo[I].Init("leval." + to_string(I)); } lveo[I].Write(M); } 
   #define clvgo_write(I,M) { if(clvgo.find(I)==clvgo.end()){ clvgo[I].Init("clgrad." + to_string(I)); } clvgo[I].Write(M); } 
   #define lvgo_write(I,M) { if(lvgo.find(I)==lvgo.end()){ lvgo[I].Init("lgrad." + to_string(I)); } lvgo[I].Write(M); } 
#else
   #define clveo_write(I,M) ((void)0)
   #define lveo_write(I,M) ((void)0)
   #define clvgo_write(I,M) ((void)0)
   #define lvgo_write(I,M) ((void)0)
#endif
   ErrorOutput err_out(path, model_name);

   const int batch_size = 20;

#ifdef ALLPASS
   vector_of_matrix batch = CreateBatch(batch_size, 2);
#else
   //******** Pattern Pass and Reject Setup **********
   vector_of_matrix batch = CreateBatch(batch_size/2, 2);
   vector_of_matrix batch_reject = CreateBatch(batch_size/2, 1);
   for (Matrix& m : batch_reject) { batch.push_back(m); }
   batch_reject.clear();  
   //*************************************************
#endif

   for (int loop = 0; loop < nloop; loop++) {

      double e = 0;
      int n = 0;
      for (int b = 0; b < batch_size; b++, n++) {
         vector_of_matrix m(1);
         m[0].resize(pattern_size, pattern_size);

         m[0] = batch[b];

#ifndef ALLPASS
         //************ Pattern Reject Switch **************
         y = (b < batch_size / 2) ? &y_pattern : &y_zero;
         // 
         //*************************************************
#endif

         for (int i = 0; i < ConvoLayerList.size(); i++) {
            m = ConvoLayerList[i]->Eval(m);
            clveo_write(i, m);
         }

         ColVector cv;
         cv = m[0].col(0);
         for (int i = 0; i < LayerList.size(); i++) {
            cv = LayerList[i]->Eval(cv);
            lveo_write(i, cv);
         }
         //cout << "y: " << y.block(8,8,16,16) << endl;
         Eigen::Map<ColVector> yv(y->data(), y->size());
         double le = loss->Eval(cv, yv);
         //if (le > e) { e = le; }
         double a = 1.0 / (double)(n + 1);
         double d = 1.0 - a;
         e = a * le + d * e;

         vector_of_matrix vm_backprop(1);
         RowVector g = loss->LossGradient();
         if (LayerList.size() == 0) {
            vm_backprop[0] = g;
         }
         lvgo_write(LayerList.size(), g);
         for (int i = (int)LayerList.size() - 1; i >= 0; --i) {
            if (i == 0) {
               vm_backprop[0] = LayerList[i]->BackProp(g);
               lvgo_write(i, vm_backprop); // Debug
            }
            else {
               g = LayerList[i - 1]->BackProp(g);
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
      if (loop < (nloop - 1)) {
         for (auto lli : ConvoLayerList) {
            lli->Update(eta);
         }
         for (auto lit : LayerList) {
            lit->Update(eta);
         }
      }

      err_out.Write(e);
      cout << "------------------- count: " << loop << " error:" << e << " -----------------" << endl;
   }

   SmartCorStats stat_class(pattern_center, 0.1);

   for (int b = 0; b < batch_size;b++) {
      vector_of_matrix m(1);
      m[0].resize(pattern_size, pattern_size);

      bool is_pattern = true;
#ifndef ALLPASS
      is_pattern = b < batch_size/2 ? true : false;
#endif

      m[0] = batch[b];

      for (auto lli : ConvoLayerList) {
         m = lli->Eval(m);
      }

      ColVector cv;
      cv = m[0].col(0);
      for (int i = 0; i < LayerList.size(); i++) {
         cv = LayerList[i]->Eval(cv);
      }
      
      runtime_assert(cv.size() == (pattern_size * pattern_size));
      Eigen::Map<Matrix> mat_distribution(cv.data(), pattern_size, pattern_size);

      // REVIEW:
      // Eval needs to be modified to return the result (true/false).  This
      // could be used to form the weighted average vector.
      stat_class.Eval(mat_distribution, is_pattern);
      
   }

   std::cout << " correct/incorrect " << stat_class.Correct << " , " << stat_class.Incorrect << endl;
   
   std::cout << "Save? y/n:  ";
   char c;
   std::cin >> c;
   if (c == 'y') {
      SaveModelWeights();
   }
}

void Test(int what)
{
   cout << "Starting test" << endl << "path: " << path << endl << "test obj: " << what << endl;
   InitModel(true);

//#define STATS
#ifdef STATS
   map<int,GradOutput> lveo;
   map<int,GradOutput> clveo;
   map<int,GradOutput> lvgo;
   map<int,GradOutput> clvgo;

   #define clveo_write(I,M) { if(clveo.find(I)==clveo.end()){ clveo[I].Init("cleval." + to_string(I)); } clveo[I].Write(M); } 
   #define lveo_write(I,M) { if(lveo.find(I)==lveo.end()){ lveo[I].Init("leval." + to_string(I)); } lveo[I].Write(M); } 
   #define clvgo_write(I,M) { if(clvgo.find(I)==clvgo.end()){ clvgo[I].Init("clgrad." + to_string(I)); } clvgo[I].Write(M); } 
   #define lvgo_write(I,M) { if(lvgo.find(I)==lvgo.end()){ lvgo[I].Init("lgrad." + to_string(I)); } lvgo[I].Write(M); } 
#else
   #define clveo_write(I,M) ((void)0)
   #define lveo_write(I,M) ((void)0)
   #define clvgo_write(I,M) ((void)0)
   #define lvgo_write(I,M) ((void)0)
#endif
   ErrorOutput err_out(path, model_name);

   const int batch_size = 100;
#ifdef ALLPASS
   vector_of_matrix batch = CreateBatch(batch_size, 2);
#else
   //******** Pattern Pass and Reject Setup **********
   vector_of_matrix batch = CreateBatch(batch_size/2, 2);
   vector_of_matrix batch_reject = CreateBatch(batch_size/2, 1);
   for (Matrix& m : batch_reject) { batch.push_back(m); }
   batch_reject.clear();  
   //*************************************************
#endif
   SmartCorStats stat_class(pattern_center, 0.000001);

   for (int b = 0; b < batch_size;b++) {
      vector_of_matrix m(1);
      m[0].resize(pattern_size, pattern_size);
      bool is_pattern = true;
#ifndef ALLPASS
      is_pattern = b < batch_size/2 ? true : false;
#endif
      m[0] = batch[b];

      for (auto lli : ConvoLayerList) {
         m = lli->Eval(m);
         clveo_write(b, m);
      }

      ColVector cv;
      cv = m[0].col(0);
      for (int i = 0; i < LayerList.size(); i++) {
         cv = LayerList[i]->Eval(cv);
         lveo_write(i, cv);
      }
      
      runtime_assert(cv.size() == (pattern_size * pattern_size));
      Eigen::Map<Matrix> mat_distribution(cv.data(), pattern_size, pattern_size);
      stat_class.Eval(mat_distribution, is_pattern);
      
   }

   std::cout << " correct/incorrect " << stat_class.Correct << " , " << stat_class.Incorrect << endl;
   std::cout << "Hit a key to continue.  ";
   char c;
   std::cin >> c;
}

void Info(double rad, int what)
{
   InitModel(true);

   //#define STATS
#ifdef STATS
   map<int, GradOutput> lveo;
   map<int, GradOutput> clveo;
   map<int, GradOutput> lvgo;
   map<int, GradOutput> clvgo;

#define clveo_write(I,M) { if(clveo.find(I)==clveo.end()){ clveo[I].Init("cleval." + to_string(I)); } clveo[I].Write(M); } 
#define lveo_write(I,M) { if(lveo.find(I)==lveo.end()){ lveo[I].Init("leval." + to_string(I)); } lveo[I].Write(M); } 
#define clvgo_write(I,M) { if(clvgo.find(I)==clvgo.end()){ clvgo[I].Init("clgrad." + to_string(I)); } clvgo[I].Write(M); } 
#define lvgo_write(I,M) { if(lvgo.find(I)==lvgo.end()){ lvgo[I].Init("lgrad." + to_string(I)); } lvgo[I].Write(M); } 
#else
#define clveo_write(I,M) ((void)0)
#define lveo_write(I,M) ((void)0)
#define clvgo_write(I,M) ((void)0)
#define lvgo_write(I,M) ((void)0)
#endif

   SmartCorStats stat_class(pattern_center, 0.1);

   vector_of_matrix m(1);
   m[0].resize(pattern_size, pattern_size);
   bool is_pattern = true;

   switch (what) {
   case 1:
      CreateCone(m[0], pattern_center, pattern_center, rad);
      break;
   case 2:
      CreateCylinder(m[0], pattern_center, pattern_center, rad);
      break;
   default:
      throw runtime_error("Unknown pattern");
   };

   Normalize(m[0]);

   OWeightsCSVFile ocsv(path, "info");
   for (int i = 0; i < ConvoLayerList.size(); i++) {
      m = ConvoLayerList[i]->Eval(m);
      clveo_write(i, m);
      ocsv.Write(m[0], i);
   }

   ColVector cv;
   cv = m[0].col(0);
   for (int i = 0; i < LayerList.size(); i++) {
      cv = LayerList[i]->Eval(cv);
      lveo_write(i, cv);
   }

   runtime_assert(cv.size() == (pattern_size * pattern_size));
   Eigen::Map<Matrix> mat_distribution(cv.data(), pattern_size, pattern_size);
   stat_class.Eval(mat_distribution, is_pattern);

   std::cout << " correct/incorrect " << stat_class.Correct << " , " << stat_class.Incorrect << endl;
   std::cout << "Hit a key to exit.";
   char c;
   std::cin >> c;

}

// There is a 1-off issue when the out matrix dimension is odd.
   void LinearCorrelate( const Matrix g, const Matrix h, Matrix& out, double bias = 0.0 )
   {
      for (int r = 0; r < out.rows(); r++) {
         for (int c = 0; c < out.cols(); c++) {
            double sum = 0.0;
            for (int rr = 0; rr < h.rows(); rr++) {
               for (int cc = 0; cc < h.cols(); cc++) {
                  int gr = r + rr +1;
                  int gc = c + cc +1;
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

int main(int argc, char* argv[])
{
   std::cout << "Starting Smart Correlation\n";

      //OWeightsCSVFile ocsv(path, "normal");
      //
      //Matrix cor(pattern_size, pattern_size);
      //Matrix h(pattern_size, pattern_size);
      //Matrix x(2 * pattern_size, 2 * pattern_size);

      //CreateNormalDistribution(h, pattern_center, pattern_center, 2.0);
      //x.setZero();
      //x.block(pattern_center, pattern_center, pattern_size, pattern_size) = h;

      //LinearCorrelate(x, h, cor);
      //CreateCylinder(nd, pattern_center, pattern_center, 12.0);
      //Normalize(nd);

      //ocsv.Write("norm",h, 1);
      //ocsv.Write("cor", cor, 1);
      //exit(0);

   try {
      if (argc > 1 && string(argv[1]) == "train") {
         if (argc < 3) {
            cout << "Not enough parameters.  Parameters: train | loops | eta | read stored coefs (0|1) [optional] | path [optional]" << endl;
            return 0;
         }
         double eta = atof(argv[3]);
         int load = 0;
         if (argc > 4) { load = atoi(argv[4]); }
         if (argc > 5) { path = argv[5]; }

         Train(atoi(argv[2]), eta, load);
      }
      else if (argc > 1 && string(argv[1]) == "info") {
         runtime_assert(argc > 3);
         if (argc > 4) { path = argv[4]; }
         Info( atof(argv[2]), atoi(argv[3]) );
      }
      else if (argc > 1 && string(argv[1]) == "test") {
         if (argc < 3) {
            cout << "Not enough parameters.  Parameters: obj | path [optional]" << endl;
            return 0;
         }
         if (argc > 3) { path = argv[3]; }
         Test(atoi(argv[2]));
      }
      else {
         cout << "Not a command.\n  "
            "train:  loops | eta | read stored coefs (0|1) [optional] | path [optional]" << endl
            << "info: radius | object (1|2) | path [optional]" << endl << "test: obj (1|2) | path [optional]" << endl;
      }
   }
   catch (std::exception ex) {
      cout << "Error:\n" << ex.what() << endl;
   }
}

