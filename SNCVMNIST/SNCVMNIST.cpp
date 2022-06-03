// SNCVMNIST.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <Eigen>
#include <iostream>
#include <iomanip>
#include <MNISTReader.h>
#include <Layer.h>
#include <bmp.h>
#include <chrono>

typedef vector< shared_ptr<Layer> > layer_list;
typedef vector< shared_ptr<iConvoLayer> > convo_layer_list;
convo_layer_list ConvoLayerList;
layer_list LayerList;

shared_ptr<iLossLayer> loss;
string path = "C:\\projects\\neuralnet\\simplenet\\SNCVMNIST\\weights";
string model_name = "layer";

class StatsOutput
{
   ofstream owf;
public:
   StatsOutput(string name) : owf(path + "\\"  + name + ".csv", ios::trunc)
   {
   }
   void Write(shared_ptr<FilterLayer2D> fl)
   {
      assert(owf.is_open());
      assert(fl);

      //Matrix& dw = fl->dW[0];
      //Eigen::Map<RowVector> rvw(dw.data(), dw.size());
      //const static Eigen::IOFormat OctaveFmt(6, 0, ", ", "\n", "", "", "", "");
      //owf << rvw.format(OctaveFmt) << endl;

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
public:
   GradOutput(string name) : owf(path + "\\"  + name + ".csv", ios::trunc){}
   GradOutput() {}
   void Init(string name) {
      owf.open(path + "\\" + name + ".csv", ios::trunc);
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
         owf << vg[i].blueNorm();
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

void ScalePerLeNet98(double* pdata, int size)
{
   // According to LeCun's 1998 paper, map the extrema of the image
   // to the range -0.1 to 1.175.
   const double y1 = -0.1;
   const double y2 = 1.175;
   double x2 = 0.0;
   double x1 = 0.0;
   double* pd = pdata;
   double* pde = pd + size;
   for (; pd < pde; pd++) {
      if (x2 < *pd) { x2 = *pd; }
      if (x1 > * pd) { x1 = *pd; }
   }

   double m = (y2 - y1) / (x2 - x1);
   double b = y1 - m * x1;

   for (pd = pdata; pd < pde; pd++) {
      *pd = *pd * m + b;
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

//---------------------------------------------------------------------------------
//                         Big Kernel Method
//

class InitBigKernelConvoLayer : public iGetWeights{
   MNISTReader::MNIST_list dl;
   int itodl[10];
public:
   InitBigKernelConvoLayer(){
      MNISTReader reader("C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data\\train\\train-images-idx3-ubyte",
         "C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data\\train\\train-labels-idx1-ubyte");

      dl = reader.read_batch(20);
      itodl[0] = 1;
      itodl[1] = 14;
      itodl[2] = 5;
      itodl[3] = 12;
      itodl[4] = 2;
      itodl[5] = 0;
      itodl[6] = 13;
      itodl[7] = 15;
      itodl[8] = 17;
      itodl[9] = 4;
   }
   void ReadConvoWeight(Matrix& m, int k) {
      if (k > 9) {
         m.setZero();
         return;
      }
      assert(m.rows() == 28 && m.cols() == 28);
      TrasformMNISTtoMatrix(m, dl[ itodl[k] ].x );
      ScaleToOne(m.data(), (int)(m.rows() * m.cols()));

      //cout << "Read " << pathname << endl;
      //cout << m << endl;
   }
   void ReadConvoBias(Matrix& w, int k) {
      w.setZero();
   }
   void ReadFC(Matrix& m) {
      throw runtime_error("InitBigKernelConvoLayer::ReadFC not implemented");
   }
};

class InitBigKernelFCLayer : public iGetWeights{

public:
   InitBigKernelFCLayer(){
   }

   void ReadConvoWeight(Matrix& w, int k) {
      throw runtime_error("InitBigKernelFCLayer::ReadConvoWeight not implemented");
   }
   void ReadConvoBias(Matrix& w, int k) {
      throw runtime_error("InitBigKernelFCLayer::ReadConvoBias not implemented");
   }
   void ReadFC(Matrix& m) {
      assert(m.rows() == 10);
      int step = (int)m.cols() / 10;
      assert(step == 14 * 14);
      Matrix pass_field(14, 14);
      pass_field.setZero();
      int pass_rad2 = 4 * 4;
      for (int r = 0; r < 14; r++) {
         int r2 = r * r;
         for (int c = 0; c < 14; c++) {
            int c2 = c * c;
            if ((r2 + c2) <= pass_rad2) {
               pass_field(r, c) = 0.01;
            }
         }
      }
      Eigen::Map<RowVector> rv_pass_field(pass_field.data(), pass_field.size());
      for (int i = 0; i < 10; i++) {
         int pos = i * step;
         m.row(i).setZero();
         m.block(i, pos, 1, step) = rv_pass_field;
      }
      
      // Make sure the bias row is zero.
      m.rightCols(1).setConstant(0.0);
   }
};

//---------------------------------------------------------------------------------
void InitBigKernelModel(bool restore)
{
   model_name = "BKM5";
   ConvoLayerList.clear();
   LayerList.clear();

   // Convolution Layer 1 -----------------------------------------
   // Type: FilterLayer2D
   int size_in  = 28;
   int size_out = 14;
   int kern = 28;
   int pad = 7;
   int kern_per_chn = 5;
   int chn_in = 1;
   int chn_out = kern_per_chn * chn_in;

   int l = 1; // Layer counter
   //                 Size input_size, int input_padding, int input_channels, Size output_size, Size kernel_size, int kernel_number, iActive* _pActive, shared_ptr<iInitWeights> _pInit 
   ConvoLayerList.push_back( make_shared<FilterLayer2D>(iConvoLayer::Size(size_in, size_in), pad, chn_in, iConvoLayer::Size(size_out, size_out), iConvoLayer::Size(kern, kern), kern_per_chn, 
                           new actReLU(size_out * size_out), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l++))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, chn_in) ),
                           true) ); // No bias
                      //               dynamic_pointer_cast<iGetWeights>( make_shared<InitBigKernelConvoLayer>()),

   // Flattening Layer 2 --------------------------------------------
   // Type: Flatten2D
   size_in  = size_out;
   chn_in = chn_out;
   size_out = size_in * size_in * chn_in;
   chn_out = 1;
   ConvoLayerList.push_back( make_shared<Flatten2D>(iConvoLayer::Size(size_in, size_in), chn_in) );
   l++;
   //--------- setup the fully connected network -----------------

   // Fully Connected Layer 1
   // Type: SoftMAX
   size_in = size_out;
   size_out = 10;
   LayerList.push_back(make_shared<Layer>(size_in, size_out, new actSoftMax(size_out), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l++))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, chn_in))) );
                                //     dynamic_pointer_cast<iGetWeights>( make_shared<InitBigKernelFCLayer>())) );
   l++;

   // Loss Layer - Not part of network, must be called seperatly.
   // Type: LossCrossEntropy
   loss = make_shared<LossL2>(size_out, 1);   
   //--------------------------------------------------------------

}
//---------------------------- End Big Kernel Method -------------------------------------
//---------------------------- LeNet-5 -------------------------------------------
// LeNet 5 1 chn --> 6 chn -- pool 2 --> (14,14) --> convo 16 (16 * 6 in)[I think] --> pool 2 --> 7
// 
void InitLeNet5Model( bool restore )
{
   model_name = "LeNet5";
   ConvoLayerList.clear();
   LayerList.clear();

   // Convolution Layer 1
   // Type: FilterLayer2D

   int size_in  = 28;
   int size_out = 28;
   int kern_per_chn = 6;
   int chn_in = 1;
   int chn_out = kern_per_chn * chn_in;
   int kern = 5;
   int pad = 2;

   int l = 1; // Layer counter
   //                 Size input_size, int input_padding, int input_channels, Size output_size, Size kernel_size, int kernel_number, iActive* _pActive, shared_ptr<iInitWeights> _pInit 
   ConvoLayerList.push_back( make_shared<FilterLayer2D>( iConvoLayer::Size(size_in, size_in), pad, chn_in, iConvoLayer::Size(size_out, size_out), iConvoLayer::Size(kern, kern), kern_per_chn, 
//                           new actLeakyReLU(size_out * size_out, 0.01), 
                           new actTanh(size_out * size_out), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l++))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, chn_in))) );

   // Pooling Layer 2
   // Type: poolAvg3D
   size_in  = size_out;
   size_out = 14;
   chn_in = chn_out;
   chn_out = 16;
   assert(!(size_in % size_out));
   vector_of_colvector_i maps(chn_out);
   int k;
   k = 0;  maps[k].resize(3); maps[k](0) = 0; maps[k](1) = 1; maps[k](2) = 2;
   k = 1;  maps[k].resize(3); maps[k](0) = 1; maps[k](1) = 2; maps[k](2) = 3;
   k = 2;  maps[k].resize(3); maps[k](0) = 2; maps[k](1) = 3; maps[k](2) = 4;
   k = 3;  maps[k].resize(3); maps[k](0) = 3; maps[k](1) = 4; maps[k](2) = 5;
   k = 4;  maps[k].resize(3); maps[k](0) = 0; maps[k](1) = 4; maps[k](2) = 5;
   k = 5;  maps[k].resize(3); maps[k](0) = 0; maps[k](1) = 1; maps[k](2) = 5;
   k = 6;  maps[k].resize(4); maps[k](0) = 0; maps[k](1) = 1; maps[k](2) = 2;  maps[k](3) = 3;
   k = 7;  maps[k].resize(4); maps[k](0) = 1; maps[k](1) = 2; maps[k](2) = 3;  maps[k](3) = 4;
   k = 8;  maps[k].resize(4); maps[k](0) = 2; maps[k](1) = 3; maps[k](2) = 4;  maps[k](3) = 5;
   k = 9;  maps[k].resize(4); maps[k](0) = 0; maps[k](1) = 3; maps[k](2) = 4;  maps[k](3) = 5;
   k = 10;  maps[k].resize(4); maps[k](0) = 0; maps[k](1) = 1; maps[k](2) = 4;  maps[k](3) = 5;
   k = 11;  maps[k].resize(4); maps[k](0) = 0; maps[k](1) = 1; maps[k](2) = 2;  maps[k](3) = 5;
   k = 12;  maps[k].resize(4); maps[k](0) = 0; maps[k](1) = 1; maps[k](2) = 3;  maps[k](3) = 4;
   k = 13;  maps[k].resize(4); maps[k](0) = 1; maps[k](1) = 2; maps[k](2) = 4;  maps[k](3) = 5;
   k = 14;  maps[k].resize(4); maps[k](0) = 0; maps[k](1) = 2; maps[k](2) = 3;  maps[k](3) = 5;
   k = 15;  maps[k].resize(6); maps[k](0) = 0; maps[k](1) = 1; maps[k](2) = 2;  maps[k](3) = 3;  maps[k](4) = 4;  maps[k](5) = 5;

   //                                    poolMax3D(Size input_size, int input_channels, Size output_size, int output_channels, vector_of_colvector_i& output_map) 
   ConvoLayerList.push_back(make_shared<poolAvg3D>(iConvoLayer::Size(size_in, size_in), chn_in, iConvoLayer::Size(size_out, size_out), chn_out, maps));
   l++;  //Need to account for each layer when restoring.

   // Convolution Layer 3
   // Type: FilterLayer2D
   size_in  = size_out;
   size_out = 10; // due to zero padding.
   kern_per_chn = 1;
   chn_in = chn_out;
   chn_out = kern_per_chn * chn_in;
   kern = 5;
   pad = 0;
   //                 Size input_size, int input_padding, int input_channels, Size output_size, Size kernel_size, int kernel_number, iActive* _pActive, shared_ptr<iInitWeights> _pInit 
   ConvoLayerList.push_back( make_shared<FilterLayer2D>(iConvoLayer::Size(size_in, size_in), pad, chn_in, iConvoLayer::Size(size_out, size_out), iConvoLayer::Size(kern, kern), kern_per_chn, 
//                           new actLeakyReLU(size_out * size_out, 0.01), 
                           new actTanh(size_out * size_out), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l++))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, chn_in))) );

   // Pooling Layer 4
   // Type: poolAvg3D
    chn_in = chn_out;
    chn_out = 1;  // Pool all layers into one layer.
   vector_of_colvector_i maps4(1);
   maps4[0].resize(chn_in); 
   for (int i = 0; i < chn_in; i++) { maps4[0](i) = i; }
   size_in  = size_out;
   size_out = 5;

   assert(!(size_in % size_out));
   ConvoLayerList.push_back(make_shared<poolAvg3D>(iConvoLayer::Size(size_in, size_in), chn_in, iConvoLayer::Size(size_out, size_out), chn_out, maps4));
   l++;  //Need to account for each layer when restoring.

   // Convolution Layer 5
   // Type: FilterLayer2D
   size_in  = size_out;
   size_out = 1;
   kern_per_chn = 120;
   chn_in = chn_out;
   chn_out = kern_per_chn * chn_in;
   kern = 5;
   pad = 0;
   //                 Size input_size, int input_padding, int input_channels, Size output_size, Size kernel_size, int kernel_number, iActive* _pActive, shared_ptr<iInitWeights> _pInit 
   ConvoLayerList.push_back( make_shared<FilterLayer2D>(iConvoLayer::Size(size_in, size_in), pad, chn_in, iConvoLayer::Size(size_out, size_out), iConvoLayer::Size(kern, kern), kern_per_chn, 
//                           new actLeakyReLU(size_out * size_out, 0.01), 
                           new actTanh(size_out * size_out), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l++))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, chn_in))) );
   // Flattening Layer 6
   // Type: Flatten2D
   size_in  = size_out;
   chn_in = chn_out;
   size_out = size_in * size_in * chn_in;
   chn_out = 1;
   ConvoLayerList.push_back( make_shared<Flatten2D>(iConvoLayer::Size(size_in, size_in), chn_in) );
   l++;  //Need to account for each layer when restoring.

   //--------- setup the fully connected network -----------------

   // Fully Connected Layer 1
   // Type: ReLU
   size_in = size_out;
   size_out = 84;
   LayerList.push_back(make_shared<Layer>(size_in, size_out, new actTanh(size_out), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l++))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, 1))) );

   // Fully Connected Layer 2
   // Type: SoftMAX
   size_in = size_out;
   size_out = 10;
   LayerList.push_back(make_shared<Layer>(size_in, size_out, new actSoftMax(size_out), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l++))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, 1))) );

   // Loss Layer - Not part of network, must be called seperatly.
   loss = make_shared<LossL2>(size_out, 1);   
   //--------------------------------------------------------------

}
//------------------------------- End LeNet-5 -------------------------------------
//---------------------------------------------------------------------------------
void InitLeNet5AModel(bool restore)
{
   model_name = "LeNet5A";
   ConvoLayerList.clear();
   LayerList.clear();

   // Convolution Layer 1 -----------------------------------------
   // Type: FilterLayer2D
   int size_in  = 28;
   int size_out = 28;
   int kern = 5;
   int pad = 2;
   int kern_per_chn = 32;
   int chn_in = 1;
   int chn_out = kern_per_chn * chn_in;
   int l = 1; // Layer counter
   ConvoLayerList.push_back( make_shared<FilterLayer2D>(iConvoLayer::Size(size_in, size_in), pad, chn_in, iConvoLayer::Size(size_out, size_out), iConvoLayer::Size(kern, kern), kern_per_chn, 
                           new actTanh(size_out * size_out), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToRandom>(0.01,0.0)),
                           false) ); // No bias - false -> use bias
   l++;
   //---------------------------------------------------------------
 
   // Pooling Layer 2 ----------------------------------------------
   // Type: poolMax2D
   size_in  = size_out;
   size_out = 14;
   chn_in = chn_out;

   assert(!(size_in % size_out));
   ConvoLayerList.push_back(make_shared<poolAvg2D>(iConvoLayer::Size(size_in, size_in), chn_in, iConvoLayer::Size(size_out, size_out)));
   l++;  //Need to account for each layer when restoring.
   //---------------------------------------------------------------

   // Convolution Layer 3 -----------------------------------------
   // Type: FilterLayer2D
   size_in  = size_out;
   size_out = 7;
   kern = 5;
   pad = 2;
   kern_per_chn = 2;
   chn_in = chn_out;
   chn_out = kern_per_chn * chn_in;
   ConvoLayerList.push_back( make_shared<FilterLayer2D>(iConvoLayer::Size(size_in, size_in), pad, chn_in, iConvoLayer::Size(size_out, size_out), iConvoLayer::Size(kern, kern), kern_per_chn, 
                           new actTanh(size_out * size_out), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToRandom>(0.01,0.0)) ));
   l++;
   //---------------------------------------------------------------

   // Pooling Layer 4 ----------------------------------------------
   // Type: poolMax2D
   size_in  = size_out;
   size_out = 7;
   chn_in = chn_out;

   assert(!(size_in % size_out));
   ConvoLayerList.push_back(make_shared<poolAvg2D>(iConvoLayer::Size(size_in, size_in), chn_in, iConvoLayer::Size(size_out, size_out)));
   l++;  //Need to account for each layer when restoring.
   //---------------------------------------------------------------      

   // Flattening Layer 5 --------------------------------------------
   // Type: Flatten2D
   size_in  = size_out;
   chn_in = chn_out;
   size_out = size_in * size_in * chn_in;
   chn_out = 1;
   ConvoLayerList.push_back( make_shared<Flatten2D>(iConvoLayer::Size(size_in, size_in), chn_in) );
   l++;
   //---------------------------------------------------------------      

   //--------- setup the fully connected network -------------------------------------------------------------------------

   // Fully Connected Layer 6 ---------------------------------------
   // Type: ReLU
   size_in = size_out;
   size_out = 512;
   LayerList.push_back(make_shared<Layer>(size_in, size_out, new actReLU(size_out), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToRandom>(0.01,0.0)) ));
   l++;
   //---------------------------------------------------------------      

   // Fully Connected Layer 7 ---------------------------------------
   // Type: SoftMAX
   size_in = size_out;
   size_out = 10;
   LayerList.push_back(make_shared<Layer>(size_in, size_out, new actSoftMax(size_out), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToRandom>(0.01,0.0)) ));
   l++;
   //---------------------------------------------------------------      

   // Loss Layer - Not part of network, must be called seperatly.
   // Type: LossCrossEntropy
   loss = make_shared<LossL2>(size_out, 1);   
   //--------------------------------------------------------------

}
//---------------------------- End LeNet 5A ---------------------------------------
//---------------------------------------------------------------------------------
void InitLeNet5BModel(bool restore)
{
   model_name = "LeNet5B";
   ConvoLayerList.clear();
   LayerList.clear();

   // Convolution Layer 1 -----------------------------------------
   // Type: FilterLayer2D
   int size_in  = 28;
   int size_out = 28;
   int kern = 5;
   int pad = 2;
   int kern_per_chn = 32;
   int chn_in = 1;
   int chn_out = kern_per_chn * chn_in;
   int l = 1; // Layer counter
   ConvoLayerList.push_back( make_shared<FilterLayer2D>(iConvoLayer::Size(size_in, size_in), pad, chn_in, iConvoLayer::Size(size_out, size_out), iConvoLayer::Size(kern, kern), kern_per_chn, 
                           new actLeakyReLU(size_out * size_out,0.01), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, chn_in))) );
   l++;
   //---------------------------------------------------------------
 
   // Pooling Layer 2 ----------------------------------------------
   // Type: PoolAvg2D
   size_in  = size_out;
   size_out = 14;
   chn_in = chn_out;

   assert(!(size_in % size_out));
   ConvoLayerList.push_back(make_shared<poolAvg2D>(iConvoLayer::Size(size_in, size_in), chn_in, iConvoLayer::Size(size_out, size_out)));
   l++;  //Need to account for each layer when restoring.
   //---------------------------------------------------------------

   // Convolution Layer 3 -----------------------------------------
   // Type: FilterLayer2D
   size_in  = size_out;
   size_out = 7;
   kern = 5;
   pad = 2;
   kern_per_chn = 2;
   chn_in = chn_out;
   chn_out = kern_per_chn * chn_in;
   ConvoLayerList.push_back( make_shared<FilterLayer2D>(iConvoLayer::Size(size_in, size_in), pad, chn_in, iConvoLayer::Size(size_out, size_out), iConvoLayer::Size(kern, kern), kern_per_chn, 
                           new actLeakyReLU(size_out * size_out,0.01), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, chn_in))) );   l++;
   //---------------------------------------------------------------

   // Pooling Layer 4 ----------------------------------------------
   // Type: poolAvg2D
   size_in  = size_out;
   size_out = 7;
   chn_in = chn_out;

   assert(!(size_in % size_out));
   ConvoLayerList.push_back(make_shared<poolAvg2D>(iConvoLayer::Size(size_in, size_in), chn_in, iConvoLayer::Size(size_out, size_out)));
   l++;  //Need to account for each layer when restoring.
   //---------------------------------------------------------------      

   // Flattening Layer 5 --------------------------------------------
   // Type: Flatten2D
   size_in  = size_out;
   chn_in = chn_out;
   size_out = size_in * size_in * chn_in;
   chn_out = 1;
   ConvoLayerList.push_back( make_shared<Flatten2D>(iConvoLayer::Size(size_in, size_in), chn_in) );
   l++;
   //---------------------------------------------------------------      

   //--------- setup the fully connected network -------------------------------------------------------------------------

   // Fully Connected Layer 6 ---------------------------------------
   // Type: ReLU
   size_in = size_out;
   size_out = 512;
   LayerList.push_back(make_shared<Layer>(size_in, size_out, new actLeakyReLU(size_out,0.01), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, 1))) );   l++;
   //---------------------------------------------------------------      

   // Fully Connected Layer 7 ---------------------------------------
   // Type: SoftMAX
   size_in = size_out;
   size_out = 10;
   LayerList.push_back(make_shared<Layer>(size_in, size_out, new actSoftMax(size_out), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, 1))) );   l++;
   //---------------------------------------------------------------      

   // Loss Layer - Not part of network, must be called seperatly.
   loss = make_shared<LossL2>(size_out, 1);   
   //--------------------------------------------------------------

}
//---------------------------- End LeNet 5B ---------------------------------------
//---------------------------------------------------------------------------------
void InitAdHockModel(bool restore)
{
   model_name = "AH1";
   ConvoLayerList.clear();
   LayerList.clear();

   // Convolution Layer 1 -----------------------------------------
   // Type: FilterLayer2D
   int size_in  = 28;
   int size_out = 14;
   int kern = 28;
   int pad = 7;
   int kern_per_chn = 10;
   int chn_in = 1;
   int chn_out = kern_per_chn * chn_in;
   int l = 1; // Layer counter
   ConvoLayerList.push_back( make_shared<FilterLayer2D>(iConvoLayer::Size(size_in, size_in), pad, chn_in, iConvoLayer::Size(size_out, size_out), iConvoLayer::Size(kern, kern), kern_per_chn, 
                           new actReLU(size_out * size_out), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, chn_in))) );   l++;
   //---------------------------------------------------------------
 
   // Pooling Layer 2 ----------------------------------------------
   // Type: poolMax2D
   /*
   size_in  = size_out;
   size_out = 7;
   chn_in = chn_out;

   assert(!(size_in % size_out));
   ConvoLayerList.push_back(make_shared<poolMax2D>(iConvoLayer::Size(size_in, size_in), chn_in, iConvoLayer::Size(size_out, size_out)));
   l++;  //Need to account for each layer when restoring.
   */
   //---------------------------------------------------------------
   /*
   // Convolution Layer 3 -----------------------------------------
   // Type: FilterLayer2D
   size_in  = size_out;
   size_out = 7;
   kern = 5;
   pad = 2;
   kern_per_chn = 2;
   chn_in = chn_out;
   chn_out = kern_per_chn * chn_in;
   ConvoLayerList.push_back( make_shared<FilterLayer2D>(iConvoLayer::Size(size_in, size_in), pad, chn_in, iConvoLayer::Size(size_out, size_out), iConvoLayer::Size(kern, kern), kern_per_chn, 
                           new actLeakyReLU(size_out * size_out,0.01), 
                           restore ? dynamic_pointer_cast<iInitWeights>( make_shared<IOMultiWeightsBinary>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iInitWeights>( make_shared<InitWeightsToRandomXavier>(sqrt(2.0/kern/kern),0.0,chn_out))) );
   l++;
   //---------------------------------------------------------------

   // Pooling Layer 4 ----------------------------------------------
   // Type: poolAvg2D
   size_in  = size_out;
   size_out = 7;
   chn_in = chn_out;

   assert(!(size_in % size_out));
   ConvoLayerList.push_back(make_shared<poolAvg2D>(iConvoLayer::Size(size_in, size_in), chn_in, iConvoLayer::Size(size_out, size_out)));
   l++;  //Need to account for each layer when restoring.
   //---------------------------------------------------------------      
   */

   // Flattening Layer 5 --------------------------------------------
   // Type: Flatten2D
   size_in  = size_out;
   chn_in = chn_out;
   size_out = size_in * size_in * chn_in;
   chn_out = 1;
   ConvoLayerList.push_back( make_shared<Flatten2D>(iConvoLayer::Size(size_in, size_in), chn_in) );
   l++;
   //---------------------------------------------------------------      

   //--------- setup the fully connected network -------------------------------------------------------------------------

   // Fully Connected Layer 6 ---------------------------------------
   // Type: ReLU
   size_in = size_out;
   size_out = 84;
   LayerList.push_back(make_shared<Layer>(size_in, size_out, new actReLU(size_out), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Kanning, 1))) );   l++;
   //---------------------------------------------------------------      

   // Fully Connected Layer 7 ---------------------------------------
   // Type: SoftMAX
   size_in = size_out;
   size_out = 10;
   LayerList.push_back(make_shared<Layer>(size_in, size_out, new actSoftMax(size_out), 
                           restore ? dynamic_pointer_cast<iGetWeights>( make_shared<IOWeightsBinaryFile>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iGetWeights>( make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier, 1))) );   l++;
   //---------------------------------------------------------------      

   // Loss Layer - Not part of network, must be called seperatly.
   // Type: LossCrossEntropy
   loss = make_shared<LossCrossEntropy>(size_out, 1);   
   //--------------------------------------------------------------

}
//---------------------------- End InitAdHockModel ---------------------------------------

typedef void (*InitModelFunction)(bool);

//InitModelFunction InitModel = InitLeNet5BModel;
//InitModelFunction InitModel = InitLeNet5AModel;
//InitModelFunction InitModel = InitLeNet5Model;
//InitModelFunction InitModel = InitBigKernelModel;
InitModelFunction InitModel = InitAdHockModel;

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

#define COMPUTE_LOSS {\
   m[0].resize(28, 28);\
   m[0] = data;\
   for (auto lli : ConvoLayerList) {\
               m = lli->Eval(m);\
            }\
   cv = m[0].col(0);\
   for (auto lli : LayerList) {\
      cv = lli->Eval(cv);\
   }\
   e = loss->Eval(cv, dl[n].y);\
}

void MakeBiasErrorFunction( string outroot, string dataroot )
{
   ofstream owf(outroot + ".0.csv", ios::trunc);
   ofstream odwf(outroot + ".0.dB.csv", ios::trunc);

   assert(owf.is_open());
   assert(odwf.is_open());

   ColVector w0(1000);
   w0.setLinSpaced( -0.2 , 1.2 );

   ColVector f(1000);
   ColVector df(1000);
   MNISTReader reader(  dataroot + "\\train\\train-images-idx3-ubyte",
                        dataroot + "\\train\\train-labels-idx1-ubyte");

   InitModel(false);

   vector_of_matrix m(1);
   Matrix data;

   double e;

   MNISTReader::MNIST_list dl = reader.read_batch(10);

   ColVector cv;
   int n = 0;
   const int kn = 0;

   data.resize(28, 28);
   TrasformMNISTtoMatrix(data, dl[n].x);
   ScaleToOne(data.data(), (int)(data.rows() * data.cols()));

   // NOTE: This is a blind downcast to FilterLayer2D.  We only do this in testing.
   //       The result of the downcast could be tested for null.
   auto ipcl = dynamic_pointer_cast<FilterLayer2D>(ConvoLayerList[0]);

   double f1;
   int pos = 0;
   for (int i = 0; i < 1000; ++i) {
      ipcl->B[kn] = w0[i];
      COMPUTE_LOSS
      f(i) = e;

      ipcl->Count = 0;
      vector_of_matrix vm_backprop(1);  // kpc kernels * 1 channel
      RowVector g = loss->LossGradient();
      for (int i = (int)LayerList.size() - 1; i >= 0; --i) {
         if (i==0) {
            vm_backprop[0] = LayerList[i]->BackProp(g);
         }
         else {
            g = LayerList[i]->BackProp(g);
         }
      }

      for (int i = (int)ConvoLayerList.size() - 1; i >= 0; --i) {
         if (i==0) {
            ConvoLayerList[i]->BackProp(vm_backprop,false);
         }
         else {
            vm_backprop = ConvoLayerList[i]->BackProp(vm_backprop);
         }
      }   
      df(i) = ipcl->dB[kn];
   }

   // octave file format
   const static Eigen::IOFormat OctaveFmt(6, 0, ", ", "\n", "", "", "", "");
   owf << f.format(OctaveFmt);
   owf.close();
   odwf << df.format(OctaveFmt);
   odwf.close();
}

void TestGradComp(string dataroot)
{
   MNISTReader reader(  dataroot + "\\train\\train-images-idx3-ubyte",
                        dataroot + "\\train\\train-labels-idx1-ubyte");

   InitModel(false);

   int n = 1;
   vector_of_matrix m(1);
   Matrix data;
   double e;
   MNISTReader::MNIST_list dl = reader.read_batch(10);
   data.resize(28, 28);
   //TrasformMNISTtoMatrix(data, dl[n].x);
   //ScaleToOne(data.data(), (int)(data.rows() * data.cols()));
   data.setRandom();

   // NOTE: This is a blind downcast to FilterLayer2D.  Normally is will resolve to a FilterLayer2D object because
   //       we are working with the top layer.  The assert will make sure the downcast is valid.
   shared_ptr<FilterLayer2D> ipcl = dynamic_pointer_cast<FilterLayer2D>(ConvoLayerList[0]);
   assert(ipcl);
   int kpc = (int)ipcl->W.size();
   int ks = ipcl->KernelSize.rows;
   assert( ks == ipcl->KernelSize.cols);

   for (int kn = 0; kn < kpc; kn++) {
      cout << ipcl->W[kn] << endl << endl;
   }

   Matrix dif(ks, ks);

   for (int kn = 0; kn < kpc; kn++) {
      ColVector cv;

      for (int r = 0; r < ks; r++) {
         for (int c = 0; c < ks; c++) {
            double f1, f2;
            double eta = 1.0e-12;

            double w1 = ipcl->W[kn](r, c);
            //----- Eval ------
            ipcl->W[kn](r, c) = w1 - eta;
            COMPUTE_LOSS
            f1 = e;

            ipcl->W[kn](r, c) = w1 + eta;
            COMPUTE_LOSS
            f2 = e;

            ipcl->W[kn](r, c) = w1;
            COMPUTE_LOSS
            vector_of_matrix vm_backprop(1); 
            RowVector g = loss->LossGradient();
            for (int i = (int)LayerList.size() - 1; i >= 0; --i) {
               if (i==0) {
                  vm_backprop[0] = LayerList[i]->BackProp(g);
               }
               else {
                  g = LayerList[i]->BackProp(g);
               }
            }

            for (int i = (int)ConvoLayerList.size() - 1; i >= 0; --i) {
               if (i==0) {
                  vm_backprop = ConvoLayerList[i]->BackProp(vm_backprop,false);
               }
               else {
                  vm_backprop = ConvoLayerList[i]->BackProp(vm_backprop);
               }
            }    

            double grad1 = ipcl->dW[kn](r, c);
            //-------------------------

            double grad = (f2 - f1) / (2.0*eta);

            //cout << f1 << ", " << grad1 << ", " << grad << ", " << abs(grad - grad1) << endl;

            dif(r, c) = abs(grad - grad1);
         }
      }

      cout << "dW[" << kn << "] Max error: " << dif.maxCoeff() << endl;
   
      // Test the bias value.
      double f1, f2;
      double eta = 1.0e-12;

      double b = ipcl->B[kn];
      //----- Eval ------
      ipcl->B[kn] = b - eta;
      COMPUTE_LOSS
      f1 = e;

      ipcl->B[kn] = b + eta;
      COMPUTE_LOSS
      f2 = e;

      ipcl->B[kn] = b;
      COMPUTE_LOSS
      vector_of_matrix vm_backprop(1);
      RowVector g = loss->LossGradient();

      for (int i = (int)LayerList.size() - 1; i >= 0; --i) {
         if (i==0) {
            vm_backprop[0] = LayerList[i]->BackProp(g);
         }
         else {
            g = LayerList[i]->BackProp(g);
         }
      }

      for (int i = (int)ConvoLayerList.size() - 1; i >= 0; --i) {
         if (i==0) {
            ConvoLayerList[i]->BackProp(vm_backprop,false);
         }
         else {
            vm_backprop = ConvoLayerList[i]->BackProp(vm_backprop);
         }
      }

      double grad1 = ipcl->dB[kn];
      //-------------------------

      double grad = (f2 - f1) / (2.0 * eta);
      double b_dif = abs(grad - grad1);
      cout << "dB[" << kn << "] = " << grad1 << " Estimate: " << grad << " Error: " << b_dif << endl;
   }

   cout << "enter a key and press Enter" << endl;
   char c;
   cin >> c;
}

void TestSave()
{
   // Initialize the model to random values.
   InitModel(false);
   convo_layer_list clist1 = ConvoLayerList;
   layer_list list1 = LayerList;

   //----------- save ----------------------------------------
   SaveModelWeights();
   //---------------------------------------------------------

   InitModel(true);

   // The test is designed for a specific network.
   // Currently designed for LeNet-5

   int clay[] = { 0, 2 };

   for (int i = 0; i < 2; i++) {
      cout << "Convo Layer: " << clay[i] << endl;

      auto ipc0 = dynamic_pointer_cast<FilterLayer2D>(ConvoLayerList[clay[i]]);
      auto ipc1 = dynamic_pointer_cast<FilterLayer2D>(clist1[clay[i]]);

      int kpc = ipc1->KernelPerChannel;
      int chn = ipc1->Channels;
      int ks = ipc1->KernelSize.rows; // Assume square
      int kn = chn * kpc;

      Matrix kdif(ks, ks);
      for (int i = 0; i < kn; i++) {
         kdif = ipc0->W[i] - ipc1->W[i]; kdif.cwiseAbs();
         cout << "W[" << i << "] diff: " << kdif.maxCoeff() << endl;
      }
   }
   for (int i = 0; i < 2; i++) {
      cout << "Layer: " << i+1 << endl;
      int osz = LayerList[i]->OutputSize;
      int isz = LayerList[i]->InputSize;
      Matrix l1dif(osz, isz+1);
      l1dif = LayerList[i]->W - list1[i]->W;  l1dif.cwiseAbs();
      cout << "W diff: " << l1dif.maxCoeff() << endl;
   }

}

void Train(int nloop, string dataroot, double eta, int load)
{
   MNISTReader reader(dataroot + "\\train\\train-images-idx3-ubyte",
                      dataroot + "\\train\\train-labels-idx1-ubyte");

   InitModel(load > 0 ? true : false);

//#define STATS
#ifdef STATS
   vector<GradOutput> lveo(LayerList.size());
   for (int i = 0; i < lveo.size(); i++) {
      lveo[i].Init("leval" + to_string(i));
   }
   vector<GradOutput> clveo(ConvoLayerList.size());
   for (int i = 0; i < clveo.size(); i++) {
      clveo[i].Init("cleval" + to_string(i));
   }

   vector<GradOutput> lvgo(LayerList.size()+1);
   for (int i = 0; i < lvgo.size(); i++) {
      lvgo[i].Init("lgrad" + to_string(i));
   }
   vector<GradOutput> clvgo(ConvoLayerList.size());
   for (int i = 0; i < clvgo.size(); i++) {
      clvgo[i].Init("clgrad" + to_string(i));
   }

   #define clveo_write(I,M) clveo[I].Write(M)
   #define lveo_write(I,M) lveo[I].Write(M)
   #define clvgo_write(I,M) clvgo[I].Write(M)
   #define lvgo_write(I,M) lvgo[I].Write(M)
#else
   #define clveo_write(I,M) ((void)0)
   #define lveo_write(I,M) ((void)0)
   #define clvgo_write(I,M) ((void)0)
   #define lvgo_write(I,M) ((void)0)
#endif
   ErrorOutput err_out(path, model_name);

   const int reader_batch = 1000;  // Should divide into 60K
   const int batch = 100; // Should divide evenly into reader_batch
   const int batch_loop = 11;

   std::random_device rd;     // only used once to initialise (seed) engine
   std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
   std::uniform_int_distribution<int> uni(0,reader_batch-1); // guaranteed unbiased

   double e = 0;
   for (int loop = 0; loop < nloop; loop++) {
      MNISTReader::MNIST_list dl = reader.read_batch(reader_batch);
      for (int bl = 0; bl < batch_loop; bl++) {
         e = 0;
         for (int b = 0; b < batch; b++) {
            int n = uni(rng); // Select a random entry out of the batch.
            vector_of_matrix m(1);
            m[0].resize(28, 28);
            TrasformMNISTtoMatrix(m[0], dl[n].x);
            ScalePerLeNet98(m[0].data(), (int)(m[0].rows() * m[0].cols()));

            for (int i = 0; i < ConvoLayerList.size(); i++) {
               m = ConvoLayerList[i]->Eval(m);
               clveo_write(i, m);
            }

            ColVector cv;
            cv = LayerList[0]->Eval(m[0].col(0));
            lveo_write(0, cv);
            for (int i = 1; i < LayerList.size(); i++) {
               cv = LayerList[i]->Eval(cv);
               lveo_write(i, cv);
            }

            double le = loss->Eval(cv, dl[n].y);
            //if (le > e) { e = le; }
            double a = 1.0 / (double)(n + 1);
            double d = 1.0 - a;
            e = a * le + d * e;

            vector_of_matrix vm_backprop(1);
            RowVector g = loss->LossGradient();
            lvgo_write(LayerList.size(), g);
            for (int i = (int)LayerList.size() - 1; i >= 0; --i) {
               if (i == 0) {
                  vm_backprop[0] = LayerList[i]->BackProp(g);
                  clvgo_write(i, vm_backprop); // Debug
               }
               else {
                  // REVIEW: Add a visitor interface to BackProp that can be used
                  //         to produce metric's such as scale of dW.
                  g = LayerList[i]->BackProp(g);
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

            //eta = (1.0 / (1.0 + 0.001 * loop)) * eta;
            for (auto lli : ConvoLayerList) {
               lli->Update(eta);
            }
            for (auto lit : LayerList) {
               lit->Update(eta);
            }

         }
         err_out.Write(e);
         cout << "count: " << loop << " error:" << e << endl;
      }
   }

   MNISTReader reader1( dataroot + "\\test\\t10k-images-idx3-ubyte",
                        dataroot + "\\test\\t10k-labels-idx1-ubyte");

   ClassifierStats stat_class;

   ColVector X;
   ColVector Y;

   double avg_e = 0.0;
   int count = 0;

   while (reader1.read_next()) {
      X = reader1.data();
      Y = reader1.label();
      vector_of_matrix m(1);
      m[0].resize(28, 28);
      TrasformMNISTtoMatrix(m[0], X);
      ScaleToOne(m[0].data(), (int)(m[0].rows() * m[0].cols()));

      for (auto lli : ConvoLayerList) {
         m = lli->Eval(m);
      }

      ColVector cv;
      cv = LayerList[0]->Eval(m[0].col(0));
      for (int i = 1; i < LayerList.size(); i++) {
         cv = LayerList[i]->Eval(cv);
      }
      
      stat_class.Eval(cv, Y);
      
   }

   std::cout << " correct/incorrect " << stat_class.Correct << " , " << stat_class.Incorrect << endl;
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

   MNISTReader reader(dataroot + "\\test\\t10k-images-idx3-ubyte",
                      dataroot + "\\test\\t10k-labels-idx1-ubyte");

   ClassifierStats stat_class;

   ColVector X;
   ColVector Y;

   double avg_e = 0.0;
   int count = 0;

   while (reader.read_next()) {
      X = reader.data();
      Y = reader.label();
      vector_of_matrix m(1);

      m[0].resize(28, 28);
      TrasformMNISTtoMatrix(m[0], X);
      ScaleToOne(m[0].data(), (int)(m[0].rows() * m[0].cols()));

      for (auto lli : ConvoLayerList) {
         m = lli->Eval(m);
      }

      ColVector cv;
      cv = LayerList[0]->Eval(m[0].col(0));
      for (int i = 1; i < LayerList.size(); i++) {
         cv = LayerList[i]->Eval(cv);
      }

      double e = loss->Eval(cv, Y);
      stat_class.Eval(cv, Y);
      /*
      if (++count == 10) {
         count = 0;
         std::cout << " correct/incorrect " << stat_class.Correct << " , " << stat_class.Incorrect << endl;
      }
      */
   }
   std::cout << " correct/incorrect " << stat_class.Correct << " , " << stat_class.Incorrect << endl;
}

int main(int argc, char* argv[])
{
   try {
      std::cout << "Starting Convolution MNIST\n";
      string dataroot = "C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data";
      //TestSave(); ;
      //TestGradComp(dataroot);
      //MakeBiasErrorFunction("C:\\projects\\neuralnet\\simplenet\\SNCVMNIST\\bias_error");
      //exit(0);

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
      else {
         if (argc > 1) {
            dataroot = argv[1];
            path = argv[2];
         }
         else {
            dataroot = "C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data";
         }
         Test(dataroot);
      }
   }
   catch (std::exception ex) {
      cout << "Error:\n" << ex.what() << endl;
   }
}
