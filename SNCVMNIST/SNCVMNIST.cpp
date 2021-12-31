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
LossCrossEntropy loss;

string path = "C:\\projects\\neuralnet\\simplenet\\SNCVMNIST\\weights";
string model_name = "layer";

void MakeMNISTImage(string file, Matrix m)
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

   generateBitmapImage(pbytes, 28, 28, 28 * sizeof(pixel_data), file);
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


//---------------------------------------------------------------------------------
//                         Big Kernel Method
//

class InitBigKernelConvoLayer : public iInitWeights{
   int ReadCount;
   MNISTReader::MNIST_list dl;
   int itodl[10];
public:
   InitBigKernelConvoLayer(){
      ReadCount = 0;
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

   void Init(Matrix& m) {
      if (ReadCount > 9) {
         m.setZero();
         return;
      }
      assert(m.rows() == 28 && m.cols() == 28);
      TrasformMNISTtoMatrix(m, dl[ itodl[ReadCount] ].x );
      ScaleToOne(m.data(), (int)(m.rows() * m.cols()));
      ReadCount++;

      //cout << "Read " << pathname << endl;
      //cout << m << endl;
   }
};

class InitBigKernelFCLayer : public iInitWeights{

public:
   InitBigKernelFCLayer(){
   }

   void Init(Matrix& m) {
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
            if ((r2 * r2 + c2 * c2) <= pass_rad2) {
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
   }
};

class OMultiWeightsBMP : public iOutputWeights{
   string Path;
   string RootName;
   int WriteCount;
public:
   OMultiWeightsBMP(string path, string root_name) : RootName(root_name), Path(path) {
      WriteCount = 1;
   }
   bool Write(Matrix& m) {
      if (WriteCount > 10) {
         return true;
      }
      string str_count;
      str_count = to_string(WriteCount);
      WriteCount++;
      string pathname = Path + "\\" + RootName + "." + str_count + ".bmp";
      MakeMNISTImage(pathname, m);
      return true;
   }
};

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
                           new ReLu(size_out * size_out), 
                           restore ? dynamic_pointer_cast<iInitWeights>( make_shared<IOMultiWeightsBinary>(path, model_name + "." + to_string(l++))) : 
                                     dynamic_pointer_cast<iInitWeights>( make_shared<InitWeightsToRandom>(0.1,0.0,chn_out)),
                      //               dynamic_pointer_cast<iInitWeights>( make_shared<InitBigKernelConvoLayer>()),
                           true) ); // No bias

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
   LayerList.push_back(make_shared<Layer>(size_in, size_out, new SoftMax(size_out), 
                           restore ? dynamic_pointer_cast<iInitWeights>( make_shared<IOMultiWeightsBinary>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iInitWeights>( make_shared<InitWeightsToRandom>(0.1,0.0,chn_out))) );
                                //     dynamic_pointer_cast<iInitWeights>( make_shared<InitBigKernelFCLayer>())) );
   l++;

   // Loss Layer - Not part of network, must be called seperatly.
   // Type: LossCrossEntropy
   loss.Init(size_out, 1);   
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
                           new actSigmoid(size_out * size_out), 
                           restore ? dynamic_pointer_cast<iInitWeights>( make_shared<IOMultiWeightsBinary>(path, model_name + "." + to_string(l++))) : 
                                     dynamic_pointer_cast<iInitWeights>( make_shared<InitWeightsToRandom>(0.1,0.0,chn_out))) );

   // Pooling Layer 2
   // Type: MaxPool3D
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

   //                                    MaxPool3D(Size input_size, int input_channels, Size output_size, int output_channels, vector_of_colvector_i& output_map) 
   ConvoLayerList.push_back(make_shared<MaxPool3D>(iConvoLayer::Size(size_in, size_in), chn_in, iConvoLayer::Size(size_out, size_out), chn_out, maps));
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
                           new actSigmoid(size_out * size_out), 
                           restore ? dynamic_pointer_cast<iInitWeights>( make_shared<IOMultiWeightsBinary>(path, model_name + "." + to_string(l++))) : 
                                     dynamic_pointer_cast<iInitWeights>( make_shared<InitWeightsToRandom>(0.1,0.0,chn_out))) );

   // Pooling Layer 4
   // Type: MaxPool3D
    chn_in = chn_out;
    chn_out = 1;  // Pool all layers into one layer.
   vector_of_colvector_i maps4(1);
   maps4[0].resize(chn_in); 
   for (int i = 0; i < chn_in; i++) { maps4[0](i) = i; }
   size_in  = size_out;
   size_out = 5;

   assert(!(size_in % size_out));
   ConvoLayerList.push_back(make_shared<MaxPool3D>(iConvoLayer::Size(size_in, size_in), chn_in, iConvoLayer::Size(size_out, size_out), chn_out, maps4));
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
                           new actSigmoid(size_out * size_out), 
                           restore ? dynamic_pointer_cast<iInitWeights>( make_shared<IOMultiWeightsBinary>(path, model_name + "." + to_string(l++))) : 
                                     dynamic_pointer_cast<iInitWeights>( make_shared<InitWeightsToRandom>(0.1,0.0,chn_out))) );
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
   LayerList.push_back(make_shared<Layer>(size_in, size_out, new actSigmoid(size_out), 
                           restore ? dynamic_pointer_cast<iInitWeights>( make_shared<IOMultiWeightsBinary>(path, model_name + "." + to_string(l++))) : 
                                     dynamic_pointer_cast<iInitWeights>( make_shared<InitWeightsToRandom>(0.1,0.0))) );

   // Fully Connected Layer 2
   // Type: SoftMAX
   size_in = size_out;
   size_out = 10;
   LayerList.push_back(make_shared<Layer>(size_in, size_out, new SoftMax(size_out), 
                           restore ? dynamic_pointer_cast<iInitWeights>( make_shared<IOMultiWeightsBinary>(path, model_name + "." + to_string(l++))) : 
                                     dynamic_pointer_cast<iInitWeights>( make_shared<InitWeightsToRandom>(0.1,0.0))) );

   // Loss Layer - Not part of network, must be called seperatly.
   // Type: SoftMAX
   loss.Init(size_out, 1);   
   //--------------------------------------------------------------

}
//------------------------------- End LeNet-5 -------------------------------------

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
                           new ReLu(size_out * size_out), 
                           restore ? dynamic_pointer_cast<iInitWeights>( make_shared<IOMultiWeightsBinary>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iInitWeights>( make_shared<InitWeightsToRandom>(0.1,0.0,chn_out)),
                      //               dynamic_pointer_cast<iInitWeights>( make_shared<InitBigKernelConvoLayer>()),
                           false) ); // No bias - false -> use bias
   l++;
   //---------------------------------------------------------------
 
   // Pooling Layer 2 ----------------------------------------------
   // Type: MaxPool2D
   size_in  = size_out;
   size_out = 14;
   chn_in = chn_out;

   assert(!(size_in % size_out));
   ConvoLayerList.push_back(make_shared<MaxPool2D>(iConvoLayer::Size(size_in, size_in), chn_in, iConvoLayer::Size(size_out, size_out)));
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
                           new ReLu(size_out * size_out), 
                           restore ? dynamic_pointer_cast<iInitWeights>( make_shared<IOMultiWeightsBinary>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iInitWeights>( make_shared<InitWeightsToRandom>(0.1,0.0,chn_out))));
   l++;
   //---------------------------------------------------------------

   // Pooling Layer 4 ----------------------------------------------
   // Type: MaxPool2D
   size_in  = size_out;
   size_out = 7;
   chn_in = chn_out;

   assert(!(size_in % size_out));
   ConvoLayerList.push_back(make_shared<MaxPool2D>(iConvoLayer::Size(size_in, size_in), chn_in, iConvoLayer::Size(size_out, size_out)));
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
   LayerList.push_back(make_shared<Layer>(size_in, size_out, new ReLu(size_out), 
                           restore ? dynamic_pointer_cast<iInitWeights>( make_shared<IOMultiWeightsBinary>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iInitWeights>( make_shared<InitWeightsToRandom>(0.1,0.0,chn_out))) );
                                //     dynamic_pointer_cast<iInitWeights>( make_shared<InitBigKernelFCLayer>())) );
   l++;
   //---------------------------------------------------------------      

   // Fully Connected Layer 7 ---------------------------------------
   // Type: SoftMAX
   size_in = size_out;
   size_out = 10;
   LayerList.push_back(make_shared<Layer>(size_in, size_out, new SoftMax(size_out), 
                           restore ? dynamic_pointer_cast<iInitWeights>( make_shared<IOMultiWeightsBinary>(path, model_name + "." + to_string(l))) : 
                                     dynamic_pointer_cast<iInitWeights>( make_shared<InitWeightsToRandom>(0.1,0.0,chn_out))) );
   l++;
   //---------------------------------------------------------------      

   // Loss Layer - Not part of network, must be called seperatly.
   // Type: LossCrossEntropy
   loss.Init(size_out, 1);   
   //--------------------------------------------------------------

}
//---------------------------- End LeNet 5A -------------------------------------

typedef void (*InitModelFunction)(bool);

InitModelFunction InitModel = InitLeNet5AModel;
//InitModelFunction InitModel = InitLeNet5Model;
//InitModelFunction InitModel = InitBigKernelModel;
//InitModelFunction InitModel = InitGenericNetworkModel;

void SaveModelWeights()
{
   int l = 1;
   for (auto lli : ConvoLayerList) {
      //lli->Save(make_shared<OMultiWeightsCSV>(path, model_name + "." + to_string(l) ));
      lli->Save(make_shared<OMultiWeightsBMP>(path, model_name + "." + to_string(l) ));
      lli->Save(make_shared<IOMultiWeightsBinary>(path, model_name + "." + to_string(l++) ));
   }
   for (const auto& lit : LayerList) {
      lit->Save(make_shared<OMultiWeightsCSV>(path, model_name + "." + to_string(l) ));
      lit->Save(make_shared<IOMultiWeightsBinary>(path, model_name + "." + to_string(l++) ));
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
   e = loss.Eval(cv, dl[n].y);\
}

void MakeBiasErrorFunction( string fileroot )
{
   ofstream owf(fileroot + ".0.csv", ios::trunc);
   ofstream odwf(fileroot + ".0.dB.csv", ios::trunc);

   assert(owf.is_open());
   assert(odwf.is_open());

   ColVector w0(1000);
   w0.setLinSpaced( -0.2 , 1.2 );

   ColVector f(1000);
   ColVector df(1000);
   MNISTReader reader("C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data\\train\\train-images-idx3-ubyte",
      "C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data\\train\\train-labels-idx1-ubyte");

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
      RowVector g = loss.LossGradiant();
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

void TestGradComp()
{
   MNISTReader reader("C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data\\train\\train-images-idx3-ubyte",
      "C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data\\train\\train-labels-idx1-ubyte");

   InitModel(false);

   vector_of_matrix m(1);
   Matrix data;
   double e;
   MNISTReader::MNIST_list dl = reader.read_batch(10);

   // NOTE: This is a blind downcast to FilterLayer2D.  Normally is will resolve to a FilterLayer2D object because
   //       we are working with the top layer.  The assert will make sure the downcast is valid.
   shared_ptr<FilterLayer2D> ipcl = dynamic_pointer_cast<FilterLayer2D>(ConvoLayerList[0]);
   assert(ipcl);
   int kpc = (int)ipcl->W.size();
   int ks = ipcl->KernelSize.rows;
   assert( ks == ipcl->KernelSize.cols);

   Matrix dif(ks, ks);

   for (int kn = 0; kn < kpc; kn++) {
      ColVector cv;
      int n = 1;

      data.resize(28, 28);
      TrasformMNISTtoMatrix(data, dl[n].x);
      ScaleToOne(data.data(), (int)(data.rows() * data.cols()));

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
            RowVector g = loss.LossGradiant();
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
      RowVector g = loss.LossGradiant();

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
   int l;
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

   cout << "Enter key to exit." << endl;
   char c;
}

void Train(int nloop)
{
   MNISTReader reader("C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data\\train\\train-images-idx3-ubyte",
      "C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data\\train\\train-labels-idx1-ubyte");

   InitModel(false);
   //InitModel(true);  // Pickup where we left off.

   double eta = 1.0;  // Begining learning rate.
   double e = 0;
  // for (int c = 0; c < 5; c++) {
      for (int loop = 0; loop < nloop; loop++) {
         e = 0;
         MNISTReader::MNIST_list dl = reader.read_batch(60);
         //for (int n = 0; n < 10; n++) {
         for (int n = 0; n < 60; n++) {
            vector_of_matrix m(1);
            m[0].resize(28, 28);
            TrasformMNISTtoMatrix(m[0], dl[n].x);
            ScaleToOne(m[0].data(), (int)(m[0].rows() * m[0].cols()));

            for (auto lli : ConvoLayerList) {
               m = lli->Eval(m);
            }
            //FilterLayer2D::vecDebugOut("convo_out", m);

            ColVector cv;
            cv = LayerList[0]->Eval(m[0].col(0));
            for (int i = 1; i < LayerList.size(); i++) {
               cv = LayerList[i]->Eval(cv);
            }

            double le = loss.Eval(cv, dl[n].y);
            //if (le > e) { e = le; }
            double a = 1.0 / (double)(n + 1);
            double b = 1.0 - a;
            e = a * le + b * e;

            vector_of_matrix vm_backprop(1); 
            RowVector g = loss.LossGradiant();
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
         }

         eta = (1.0 / (1.0 + 0.001 * loop)) * eta;
         for (auto lli : ConvoLayerList) {
            lli->Update(eta);
            }
         for (auto lit : LayerList) {
            lit->Update(eta);
         }
         cout << "count: " << loop << " error:" << e << endl;
      }
   //}
      
   MNISTReader reader1("C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data\\test\\t10k-images-idx3-ubyte",
      "C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data\\test\\t10k-labels-idx1-ubyte");

   ColVector X;
   ColVector Y;

   double avg_e = 0.0;
   int count = 0;

   loss.ResetCounters();
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

      double e = loss.Eval(cv, Y);
      
   }
   
   std::cout << " correct/incorrect " << loss.Correct << " , " << loss.Incorrect << endl;
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

   MNISTReader reader("C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data\\test\\t10k-images-idx3-ubyte",
      "C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data\\test\\t10k-labels-idx1-ubyte");

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

      double e = loss.Eval(cv, Y);
      /*
      if (++count == 10) {
         count = 0;
         std::cout << " correct/incorrect " << loss.Correct << " , " << loss.Incorrect << endl;
      }
      */
   }
   std::cout << " correct/incorrect " << loss.Correct << " , " << loss.Incorrect << endl;
}

int main(int argc, char* argv[])
{
   std::cout << "Starting Convolution MNIST\n";
   //TestSave(); ;
   //TestGradComp();
   //MakeBiasErrorFunction("C:\\projects\\neuralnet\\simplenet\\SNCVMNIST\\bias_error");
   //exit(0);
  
   if (argc > 0 && string(argv[1]) == "train") {
      Train( atoi(argv[2]) );
      
   }
   else {
      Test();
   }

   //char c;
   //cin >> c;
}
