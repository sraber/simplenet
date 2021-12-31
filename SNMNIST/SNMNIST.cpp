// SNMNIST.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <Eigen>
#include <iostream>
#include <iomanip>
#include <MNISTReader.h>
#include <Layer.h>
#include <chrono>

typedef vector< shared_ptr<Layer> > layer_list;
layer_list LayerList;
LossCrossEntropy loss;

string path = "C:\\projects\\neuralnet\\simplenet\\SNMNIST\\weights";
string model_name = "layer";

void SaveModelWeights()
{
   int l = 1;
   for (const auto& lit : LayerList) {
      lit->Save(make_shared<OMultiWeightsCSV>(path, model_name + "." + to_string(l) ));
      lit->Save(make_shared<IOMultiWeightsBinary>(path, model_name + "." + to_string(l++) ));
   }
}

void InitFCModel(bool restore)
{
   model_name = "FC32";
   LayerList.clear();

   // FC Layer 1 -----------------------------------------
   // Type: FC Layer
   int size_in  = MNISTReader::DIM;
   int size_out = 32;

   int l = 1; // Layer counter
   LayerList.push_back(make_shared<Layer>(size_in, size_out, new ReLu(size_out),
                 restore ? dynamic_pointer_cast<iInitWeights>( make_shared<IOMultiWeightsBinary>(path, model_name + "." + to_string(l))) : 
                           dynamic_pointer_cast<iInitWeights>( make_shared<InitWeightsToRandom>(0.1,0.0))) );
                           //dynamic_pointer_cast<iInitWeights>( make_shared<InitWeightsToConstants>(0.1,0.0))) );
   l++;

   // FC Layer 2 -----------------------------------------
   // Type: FC Layer
   size_in  = size_out;
   size_out = 10;
   LayerList.push_back(make_shared<Layer>(size_in, size_out, new SoftMax(size_out),
                 restore ? dynamic_pointer_cast<iInitWeights>( make_shared<IOMultiWeightsBinary>(path, model_name + "." + to_string(l))) : 
                           dynamic_pointer_cast<iInitWeights>( make_shared<InitWeightsToRandom>(0.1,0.0))) );
                           //dynamic_pointer_cast<iInitWeights>( make_shared<InitWeightsToConstants>(0.1,0.0))) );

   loss.Init(size_out, 1);
}

typedef void (*InitModelFunction)(bool);

InitModelFunction InitModel = InitFCModel;

#define COMPUTE_LOSS {\
   cv = dl[n].x;\
   for (auto lli : LayerList) {\
      cv = lli->Eval(cv);\
   }\
   e = loss.Eval(cv, dl[n].y);\
}

void TestGradComp()
{
   MNISTReader reader("C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data\\train\\train-images-idx3-ubyte",
      "C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data\\train\\train-labels-idx1-ubyte");

   InitModel(false);

   double e;

   MNISTReader::MNIST_list dl = reader.read_batch(10);
   int rows = LayerList[0]->W.rows();
   int cols = LayerList[0]->W.cols();
   Matrix dif(rows,cols);
   ColVector cv;
   int n = 0;
   double max = 0.0;
   for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
         double f1, f2;
         double eta = 1.0e-12;
         double w1 = LayerList[0]->W(r, c);
         //----- Eval ------
         LayerList[0]->W(r, c) = w1 - eta;
         COMPUTE_LOSS
         f1 = e;

         LayerList[0]->W(r, c) = w1 + eta;
         COMPUTE_LOSS
         f2 = e;

         LayerList[0]->W(r, c) = w1;
         COMPUTE_LOSS
         RowVector g = loss.LossGradiant();
         for (int i = LayerList.size() - 1; i >= 0; --i) {
            LayerList[i]->Count = 0;
            g = LayerList[i]->BackProp(g);
         }

         double grad1 = LayerList[0]->grad_W.transpose()(r,c);
         if (grad1 > max) {
            max = grad1;
         }
         //-------------------------

         double grad = (f2 - f1) / (2.0*eta);
         dif(r, c) = abs(grad - grad1);
      } 
   }

   cout << "dW Max error: " << dif.maxCoeff() << " Max error percent: " << 100.0 * dif.maxCoeff() / max << endl;

   cout << "enter a key and press Enter" << endl;
   char c;
   cin >> c;
}

void MakeTrendErrorFunction( int r, int c, string fileroot )
{
   string sr = to_string(r);
   string sc = to_string(c);
   ofstream owf(fileroot + "." + sr + "." + sc + ".f.csv", ios::trunc);
   ofstream odwf(fileroot + "." + sr + "." + sc + ".df.csv", ios::trunc);

   assert(owf.is_open());
   assert(odwf.is_open());

   ColVector w0(1000);
   w0.setLinSpaced( -0.2 , 1.2 );

   ColVector f(1000);
   ColVector df(1000);
   MNISTReader reader("C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data\\train\\train-images-idx3-ubyte",
      "C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data\\train\\train-labels-idx1-ubyte");

   InitModel(false);

   double e;

   MNISTReader::MNIST_list dl = reader.read_batch(10);

   

   ColVector cv;
   int n = 0;

   /*
   for (int i = 0; i < MNISTReader::DIM; i++) {
      cout << i << " , " << dl[n].x(i) << endl;
   }

   char cc;
   cin >> cc;
   return;
   */

   double f1;
   int pos = 0;
   for (int i = 0; i < 1000; ++i) {
      LayerList[0]->W(r,c) = w0[i];
      cv = dl[n].x;
      for (auto lli : LayerList) { cv = lli->Eval(cv); }
      f(i) = loss.Eval(cv, dl[n].y);

      LayerList[0]->Count = 0;
      RowVector g = loss.LossGradiant();
      for (int i = LayerList.size() - 1; i >= 0; --i) {
         g = LayerList[i]->BackProp(g);
      } 
      df(i) = LayerList[0]->grad_W(c,r);
   }

   // octave file format
   const static Eigen::IOFormat OctaveFmt(6, 0, ", ", "\n", "", "", "", "");
   owf << f.format(OctaveFmt);
   owf.close();
   odwf << df.format(OctaveFmt);
   odwf.close();
}

void Train( int nloop )
{
   InitModel(false);

   MNISTReader reader("C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data\\train\\train-images-idx3-ubyte",
      "C:\\projects\\neuralnet\\cpp_nn_in_a_weekend-master\\data\\train\\train-labels-idx1-ubyte");
   ColVector X;
   ColVector Y;
   double avg_e = 0.0;

   for (int loop = 1; loop < nloop; loop++) {
      MNISTReader::MNIST_list dl = reader.read_batch(60);
      auto start = chrono::steady_clock::now();
      for (int k = 1; k <= 3; k++) {
         for (auto& dli : dl) {
            X = dli.x;
            for (const auto& lit : LayerList) {
               X = lit->Eval(X);
            }
            double error = loss.Eval(X, dli.y);
            double a = 1.0 / (double)loop;
            double b = 1 - a;
            avg_e = a * error + b * avg_e;

            //cout << "------ Error: " << error << "------------" << endl;
            RowVector g = loss.LossGradiant();

            for (layer_list::reverse_iterator riter = LayerList.rbegin();
               riter != LayerList.rend();
               riter++) {
               g = (*riter)->BackProp(g);
            }
         }
         for (const auto& lit : LayerList) {
            double eta = 1.0;
            lit->Update(eta);
         }
      }
      auto end = chrono::steady_clock::now();
      auto diff = end - start;
      //std::cout << std::setprecision(3) << avg_e << endl;
      std::cout << "avg error: " << std::setprecision(3) << avg_e << " correct/incorrect " << loss.Correct << " , " << loss.Incorrect << " Compute time: " << chrono::duration <double, milli>(diff).count() << " ms" << endl;
      int l = 1;
      for (const auto& lit : LayerList) {
         cout << "L " << l++ << "W: " << lit->W.cwiseAbs().maxCoeff() << endl;
      }
      loss.ResetCounters();
   }
   
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
      for (const auto& lit : LayerList) {
         X = lit->Eval(X);
      }
      double error = loss.Eval(X, Y);
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
   std::cout << "Starting simpleMNIST\n";

   //TestGradComp();
   MakeTrendErrorFunction(11, 156, "C:\\projects\\neuralnet\\simplenet\\SNMNIST\\weights\\trend");
   exit(0);

   if (argc > 0 && string(argv[1]) == "train") {
      Train( atoi(argv[2]) );  
   }
   else {
      Test();
   }


}

