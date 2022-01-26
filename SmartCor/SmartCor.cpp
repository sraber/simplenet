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

void CreatePattern(Matrix& m, double ro2, double co2, double radius);

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
public:
   GradOutput(string name, bool want_norm = true) : 
      owf(path + "\\"  + name + ".csv", ios::trunc),
      output_norm(want_norm){}
   GradOutput() : output_norm(true) {}
   void Init(string name, bool want_norm = true) {
      owf.open(path + "\\" + name + ".csv", ios::trunc);
      output_norm = want_norm;
   }
   void Write(RowVector& g)
   {
      assert(owf.is_open());
      owf << g.blueNorm() << endl;
   }
   void Write(ColVector& g)
   {
      assert(owf.is_open());
      owf << g.blueNorm() << endl;
   }
   void Write(vector_of_matrix& vg)
   {
      assert(owf.is_open());

      for (int i = 0; i < vg.size();i++) {
         owf << (output_norm ? vg[i].blueNorm() : vg[i].maxCoeff());
         if (i < vg.size() - 1) {
            owf << ",";
         }
      }
      owf << endl;
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
   int Center;
   double Threshold;
   const double acceptable_error_sqd = 2.0 * 2.0; // This is pixels.
public:
   int Correct;
   int Incorrect;
   SmartCorStats(int _center, double _thresh) : Center(_center), Threshold(_thresh), Correct(0), Incorrect(0) {}
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
      if (exceeded_threshold==is_pattern &&
          (rm*rm+cm*cm)<=acceptable_error_sqd ){
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
const double pattern_center = (double)(pattern_size + 1) / 2.0;

class InitSmartCorConvoLayer : public iGetWeights{
   int Size;
   int Center;
public:
   InitSmartCorConvoLayer(int _size, int _center) : Size(_size), Center(_center){}
   void ReadConvoWeight(Matrix& m, int k) {
      CreatePattern(m, Center, Center, Size / 4);
      m *= 0.01;
   }
   void ReadConvoBias(Matrix& w, int k) {
      w.setZero();
   }
   void ReadFC(Matrix& m) {
      throw runtime_error("InitBigKernelConvoLayer::ReadFC not implemented");
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
   ConvoLayerList.push_back( make_shared<FilterLayer2D>(iConvoLayer::Size(size_in, size_in), pad, chn_in, iConvoLayer::Size(size_out, size_out), iConvoLayer::Size(kern, kern), kern_per_chn, 
                           new actTanh(size_out * size_out), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToRandom>(0.1)),
                                     //dynamic_pointer_cast<iGetWeights>( make_shared<InitSmartCorConvoLayer>(pattern_size, (double)(pattern_size + 1) / 2.0)),
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
   ConvoLayerList.push_back( make_shared<Flatten2D>(iConvoLayer::Size(size_in, size_in), chn_in) );
   l++;
   //---------------------------------------------------------------      
   

   // Loss Layer - Not part of network, must be called seperatly.
   // Type: LossCrossEntropy
   loss = make_shared<LossL2>(size_out, 1);   
   //--------------------------------------------------------------

}
//---------------------------- End InitSmartCorA ---------------------------------------

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
   ConvoLayerList.push_back( make_shared<FilterLayer2D>(iConvoLayer::Size(size_in, size_in), pad, chn_in, iConvoLayer::Size(size_out, size_out), iConvoLayer::Size(kern, kern), kern_per_chn, 
                           new actReLU(size_out * size_out), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     //dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, chn_in)),
                                     dynamic_pointer_cast<iGetWeights>( make_shared<InitSmartCorConvoLayer>(pattern_size, (double)(pattern_size + 1) / 2.0)),
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
   ConvoLayerList.push_back( make_shared<Flatten2D>(iConvoLayer::Size(size_in, size_in), chn_in) );
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
   LayerList.push_back(make_shared<Layer>(size_in, size_out, new actSoftMax(size_out), 
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

//InitModelFunction InitModel = InitSmartCor;
InitModelFunction InitModel = InitSmartCorA;

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

void CreatePattern(Matrix& m, double ro2, double co2, double radius)
{
   m.setZero();
   int pass_rad2 = radius * radius;
   for (int r = 0; r < m.rows(); r++) {
      int rc = r - ro2;
      int r2 = rc * rc;
      for (int c = 0; c < m.cols(); c++) {
         int cc = c - co2;
         int c2 = cc * cc;
         if ((r2 + c2) < pass_rad2) {
            double cur_radius = sqrt((double)(r2 + c2));
            // Amplitude of 0 at radius rising to 1 at center.  It's a cone.
            m(r, c) = 1.0 - cur_radius / radius;
         }
      }
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

   void cwiseDiv(double* t, double* s, int size)
   {
      double* te = t + size;
      for (; t < te; t++, s++) {
         if (*s > 0.0) {
            *t /= *s;
         }
      }
   }

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
//      CreatePattern(vom[i], pattern_center, pattern_center, r);
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

vector_of_matrix CreateBatch(int size)
{
   const double rad_small = (double)pattern_size / 8.0;
   const double rad_big = (double)pattern_size / 2.0;

   double sum = 0.0;

   vector_of_matrix vom(size);
   for (Matrix& m : vom) { m.resize(pattern_size, pattern_size); }

   double g = (rad_big - rad_small) / (double)(size - 1);
   for (int i = 0; i < size; i++) {
      double r = (double)i * g + rad_small;
      CreatePattern(vom[i], pattern_center, pattern_center, r);
      sum += vom[i].mean();
   }

   sum /= (double)size;

   //cout << std << endl;

   for (Matrix& m : vom) {
      m.array() -= sum; // sum is now average.
      //cwiseDiv(m.data(), std.data(), m.size());

      //cout << m << endl;
   }
   
   return vom;
}

void Train(int nloop, double eta, int load)
{
   InitModel(load > 0 ? true : false);

   double center = (double)(pattern_size + 1) / 2.0;
   Matrix y_pattern(pattern_size, pattern_size);
   Matrix y_noise(pattern_size, pattern_size);

   y_pattern.setZero();
   y_pattern((int)(center + 0.5), (int)(center + 0.5)) = 1.0;

   y_noise.setConstant( 1.0 / (double)(pattern_size * pattern_size));

   Matrix& y = y_pattern;

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

   //std::random_device rd;     // only used once to initialise (seed) engine
   //std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
   //std::uniform_real_distribution<double> unr(pattern_size/8,pattern_size/2); // guaranteed unbiased
   //std::uniform_int_distribution<int> uni(1, 10);

   const int batch_size = 100;
   vector_of_matrix batch = CreateBatch(batch_size);

   for (int loop = 0; loop < nloop; loop++) {
      //if (uni(rng) == 10) {
      //   CreateNoise(m[0]);
      //   y = y_noise;
      //}
      //else {
      //   CreatePattern(m[0], center, center, unr(rng));
      //   //CreatePattern(m[0], center, center, pattern_size/4);
      //   y = y_pattern;
      //}
      double e = 0;
      int n = 0;
      for (int b = 0; b < batch_size; b++, n++) {
         vector_of_matrix m(1);
         m[0].resize(pattern_size, pattern_size);

         m[0] = batch[b];

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

         Eigen::Map<ColVector> yv(y.data(), y.size());
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
               // REVIEW: Add a visitor interface to BackProp that can be used
               //         to produce metric's such as scale of dW.
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

      for (auto lli : ConvoLayerList) {
         lli->Update(eta);
      }
      for (auto lit : LayerList) {
         lit->Update(eta);
      }

      err_out.Write(e);
      cout << "count: " << loop << " error:" << e << endl;


   }

   SmartCorStats stat_class(pattern_center, 0.000001);

   for (int b = 0; b < batch_size;b++) {
      vector_of_matrix m(1);
      m[0].resize(pattern_size, pattern_size);
      bool is_pattern = true;
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

void Test()
{
   InitModel(true);

   double center = (double)(pattern_size + 1) / 2.0;
   Matrix y_pattern(pattern_size, pattern_size);
   Matrix y_noise(pattern_size, pattern_size);

   y_pattern.setZero();
   y_pattern((int)(center + 0.5), (int)(center + 0.5)) = 1.0;

   y_noise.setConstant( 1.0 / (double)(pattern_size * pattern_size));

   Matrix& y = y_pattern;

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
   vector_of_matrix batch = CreateBatch(batch_size);



   SmartCorStats stat_class(pattern_center, 0.000001);


   for (int b = 0; b < batch_size;b++) {
      vector_of_matrix m(1);
      m[0].resize(pattern_size, pattern_size);
      bool is_pattern = true;
      m[0] = batch[b];

      for (auto lli : ConvoLayerList) {
         m = lli->Eval(m);
         clveo_write(i, m);
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
   std::cout << "Save? y/n:  ";
   char c;
   std::cin >> c;
   if (c == 'y') {
      SaveModelWeights();
   }
}

void Info(double rad)
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
   const int batch_size = 100;
   vector_of_matrix batch = CreateBatch(batch_size);

   SmartCorStats stat_class(pattern_center, 0.000001);

   vector_of_matrix m(1);
   m[0].resize(pattern_size, pattern_size);
   bool is_pattern = true;

   CreatePattern(m[0], pattern_center, pattern_center, rad);

   OWeightsCSVFile ocsv(path, "info");
   for (int i = 0; i < ConvoLayerList.size(); i++) {
      m = ConvoLayerList[i]->Eval(m);
      clveo_write(i, m);
      ocsv.Write(m[i], i);
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


int main(int argc, char* argv[])
{
   std::cout << "Starting Smart Correlation\n";

   try {
      if (argc > 1 && string(argv[1]) == "train") {
         if (argc < 3) {
            cout << "Not enough parameters.  Parameters: train | batches | eta | read stored coefs (0|1) [optional] | path [optional]" << endl;
            return 0;
         }
         double eta = atof(argv[3]);
         int load = 0;
         if (argc > 4) { load = atoi(argv[4]); }
         if (argc > 5) { path = argv[5]; }

         Train(atoi(argv[2]), eta, load);
      }
      else if (argc > 1 && string(argv[1]) == "info") {
         Interesting stuff.  The correlator is working.  Peak seems to be off a bit.
         runtime_assert(argc > 2);
         Info( atof(argv[2]) );
      }
      else {
         if (argc > 1) {
            path = argv[1];
         }

         //Test(dataroot);
      }
   }
   catch (std::exception ex) {
      cout << "Error:\n" << ex.what() << endl;
   }
}

