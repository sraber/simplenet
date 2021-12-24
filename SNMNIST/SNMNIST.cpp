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

void Train( int nloop )
{
   //------------ setup the network ------------------------------
   LayerList.push_back(make_shared<Layer>(MNISTReader::DIM, 32, new ReLu(32), make_shared<InitWeightsToRandom>(0.1, 0.0)));
   LayerList.push_back(make_shared<Layer>(32, 10, new SoftMax(10), make_shared<InitWeightsToRandom>(0.1, 0.0)));

   loss.Init(10, 1);
   //-------------------------------------------------------------

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
            double eta = 0.5;
            lit->Update(eta);
         }
      }
      auto end = chrono::steady_clock::now();
      auto diff = end - start;
      std::cout << "avg error: " << std::setprecision(3) << avg_e << " correct/incorrect " << loss.Correct << " , " << loss.Incorrect << " Compute time: " << chrono::duration <double, milli>(diff).count() << " ms" << endl;
      loss.ResetCounters();
   }

   std::cout << "Save? y/n:  ";
   char c;
   std::cin >> c;
   if (c == 'y') {
      int l = 0;
      for (const auto& lit : LayerList) {
         l++;
         string layer = to_string(l);
         lit->Save(make_shared<IOWeightsBinary>("C:\\projects\\neuralnet\\simplenet\\SNMNIST\\Layer." + layer + ".wts"));
         lit->Save(make_shared<WriteWeightsCSV>("C:\\projects\\neuralnet\\simplenet\\SNMNIST\\Layer." + layer + ".wts.csv"));
      }
   }
}

void Test()
{
   //------------ setup the network ------------------------------
   LayerList.push_back(make_shared<Layer>(MNISTReader::DIM, 32, new ReLu(32), make_shared<IOWeightsBinary>("C:\\projects\\neuralnet\\simplenet\\SNMNIST\\Layer.1.wts")));
   LayerList.push_back(make_shared<Layer>(32, 10, new SoftMax(10), make_shared<IOWeightsBinary>("C:\\projects\\neuralnet\\simplenet\\SNMNIST\\Layer.2.wts")));

   loss.Init(10, 1);
   //-------------------------------------------------------------
   /*
      int l = 0;
      for (const auto& lit : LayerList) {
         l++;
         string layer = to_string(l);
         lit->Save(make_shared<WriteWeightsCSV>("C:\\projects\\neuralnet\\simplenet\\SNMNIST\\LayerT." + layer + ".wts.csv"));
      }

      exit(2);
      */
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
   /*
   Layer tl(5, 3, new ReLu(3),  make_shared<InitWeightsToConstants>(0.0, 0.0) );
   Matrix& w = tl.W;
   for (int r = 0; r < 3; r++) {
      for (int c = 0; c < 6; c++) {
         w(r, c) = r;
      }
   }

   tl.Save(make_shared<IOWeightsBinary>("C:\\projects\\neuralnet\\simplenet\\SNMNIST\\Layer.test.wts"));

   Layer rl(5, 3, new ReLu(3), make_shared<IOWeightsBinary>("C:\\projects\\neuralnet\\simplenet\\SNMNIST\\Layer.test.wts") );

   cout << rl.W << endl;
   exit(1);
   */
   /*
   IOWeightsBinary rd("C:\\projects\\neuralnet\\simplenet\\SNMNIST\\Layer.2.wts");
   Matrix w(10,32+1);
   rd.Init(w);
   for (int r = 0; r < 10; r++) {
      for (int c = 0; c < 10; c++) {
         cout << w(r, c) << ",";
      }
      cout << endl;
   }

   exit(0);
   */
   /*    Matrix m(5, 10);
       m.row(0).setConstant(1.0);
       m.row(1).setConstant(2.0);
       m.row(2).setConstant(3.0);
       m.row(3).setConstant(4.0);
       m.row(4).setConstant(5.0);

       WriteWeightsBinary wrt("C:\\projects\\neuralnet\\simpleMNIST\\Layer.1.wts");
       ReadWeightsBinary rd("C:\\projects\\neuralnet\\simpleMNIST\\Layer.test.wts");

       wrt.Write(m);

       Matrix n(5, 10);
       rd.Init(n);

       Matrix o = n - m;

       cout << o << endl;
   */
   if (argc > 0 && string(argv[1]) == "train") {
      Train( atoi(argv[2]) );
/*
      layer_list ll;
         ll.push_back(make_shared<Layer>(MNISTReader::DIM, 32, new ReLu(32), make_shared<IOWeightsBinary>("C:\\projects\\neuralnet\\simplenet\\SNMNIST\\Layer.1.wts")));
   ll.push_back(make_shared<Layer>(32, 10, new SoftMax(10), make_shared<IOWeightsBinary>("C:\\projects\\neuralnet\\simplenet\\SNMNIST\\Layer.2.wts")));

   Matrix d1(33, MNISTReader::DIM);
   Matrix d2(11, 32);

   d1 = LayerList[0]->W - ll[0]->W;
   d2 = LayerList[1]->W - ll[1]->W;
   d1.cwiseAbs();
   d2.cwiseAbs();

   cout << d1.maxCoeff() << endl << d2.maxCoeff();
   exit(2);
*/
      
   }
   else {
      Test();
   }


}

