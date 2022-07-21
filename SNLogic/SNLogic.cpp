// SNLogic.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <Layer.h>
#include <random>
#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;

typedef vector< shared_ptr<Layer> > layer_list;
layer_list LayerList;
shared_ptr<iLossLayer> loss;

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// Output file path.  Change it to suit your file structure.
string path = "C:\\projects\\neuralnet\\simplenet\\SNLogic";
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

struct tup {
   double x1;
   double x2;
   double y;
   tup(double _x1, double _x2, double _y) : x1(_x1), x2(_x2), y(_y) {}
};
typedef list<tup> tup_list;

void MakeDecsionSurface( string fileroot );
void MakeErrorSurface(tup_list points);
void MakeAvgErrorSurface(tup_list points);

void MakePointCloud(tup_list& tuples, double xc, double yc, int label, int n)
{
   random_device rd;
   mt19937 mt(rd());
   uniform_real_distribution<double> dst(-2.0, 2.0);
   for (int i = 1; i <= n; i++) {
      double x = dst(mt);
      double y = dst(mt);
      tuples.push_back({ xc + x, yc + y, (double)label });
   }
}

void MakePointRing(tup_list& tuples, double xc, double yc, double radius, int label, int n)
{
   double rad_step = 2.0 * M_PI / (double)n;
   for (int i = 0; i < n; i++) {
      double rad = rad_step * (double)i;
      double x = radius * cos(rad);
      double y = radius * sin(rad);
      tuples.push_back({ xc + x, yc + y, (double)label });
   }
}

#define COMPUTE_LOSS {\
   cv = x;\
   for (auto lli : LayerList) {\
      lli->Count = 0;\
      cv = lli->Eval(cv);\
   }\
   e = loss->Eval(cv, y);\
}

void TestGradComp(tup t)
{
   ColVector x(2);
   ColVector y(1);
   x(0) = t.x1;
   x(1) = t.x2;
   y(0) = t.y;

   // NOTE: This is a blind downcast to Layer.
   shared_ptr<Layer> ipcl = dynamic_pointer_cast<Layer>(LayerList[0]);
   assert(ipcl);
   int rows = (int)ipcl->W.rows();
   int cols = (int)ipcl->W.cols();

   Matrix dif(rows, cols);
   Matrix dwt(rows, cols);

   ColVector cv;
   double e;

   COMPUTE_LOSS
   RowVector g = loss->LossGradient();
   for (layer_list::reverse_iterator riter = LayerList.rbegin();
      riter != LayerList.rend();
      riter++) {
      g = (*riter)->BackProp(g);
   } 
   //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   // The strength of the L2 regularize is hardcoded here       !!
   //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   dwt = ipcl->dW.transpose() + 0.1*ipcl->W;

   for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
         double f1, f2;
         double eta = 1.0e-5;

         double w1 = ipcl->W(r, c);
         //----- Eval ------
         ipcl->W(r, c) = w1 - eta;
         COMPUTE_LOSS
         f1 = e;

         ipcl->W(r, c) = w1 + eta;
         COMPUTE_LOSS
         f2 = e;

         ipcl->W(r, c) = w1;

         double grad1 = dwt(r, c);
         double grad = (f2 - f1) / (2.0*eta);

         dif(r, c) = abs(grad - grad1);
      }
   }

   cout << "dW Max error: " << dif.maxCoeff() << endl;

   OWeightsCSVFile out_file(path, "dif");
   out_file.Write(dif, 1);

   cout << "enter a key and press Enter" << endl;
   char c;
   cin >> c;
}


int main(int argc, char* argv[])
{
   tup_list points;
   ErrorOutput err_out(path, "snlogic");
   /*
   // Neural Networks paper - simple example
   points.push_back({ 6.5, 4.0, (double)0 });
   points.push_back({ 4.5, -4.0, (double)1 });
   points.push_back({ -4.0, 4.5, (double)1 });
   //------------ setup the network ------------------------------
   LayerList.push_back(make_shared<Layer>(2, 1, new actSigmoid(1), make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier,1),make_shared<penalityL2Weight>(0.1)));
   //LayerList.push_back(make_shared<Layer>(2, 1, new actSigmoid(1), make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier,1) ));
   */

   
   // Tougher example ---------------------------------------
   // This example puts a cloud of 1's at the center, with a
   // ring of 0's around it, followed by a ring of 1's around that.

   MakePointRing(points, 0.0, 0.0, 8.0, 1, 80);
   MakePointRing(points, 0.0, 0.0, 5.0, 0, 160);
   MakePointCloud(points, 0.0, 0.0, 1, 80);

   // Try 2d+2 in hidden layer for Point Ring problem.  
   //     See: APPROACH TO THE SYNTHESIS OF NEURAL NETWORK STRUCTURE DURING CLASSIFICATION
   // A two layer network with a Sigmoid activation can apparently fit any data set given enough
   // interior nodes.  Experiment with the number of interior nodes by changing the value of
   // a1.
   //------------ setup the network ------------------------------
   int a1 = 9;
   LayerList.push_back(make_shared<Layer>(2, a1, new actSigmoid(), make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier,1)));
   LayerList.push_back(make_shared<Layer>(a1, 1, new actSigmoid(), make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier,1)));
   //--  End Tough example ------------------------------------
   

   //------ Back-propagation test setup.
   // The setup was never run to see if it would converge.  It was
   // used along with the function TestGradComp to test the accuracy
   // of the Back-Propagation algorithm.
   //LayerList.push_back(make_shared<Layer>(2, 5, new actSigmoid(), make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier,1)));
   //LayerList.push_back(make_shared<Layer>(5, 9, new actSigmoid(), make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier,1)));
   //LayerList.push_back(make_shared<Layer>(9, 11, new actSigmoid(), make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier,1)));
   //LayerList.push_back(make_shared<Layer>(11, 1, new actSigmoid(), make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier,1)));
   //-- End Backprop test setup ---------------------

   // Setup the loss layer. 
   loss = make_shared<LossL2>(1, 1);   

   //-------------------------------------------------------------

   //***************  Test Gradient Computation ********************
   //TestGradComp( points.front() );
   //return 0;
   //***************************************************************

   //************ Point output to file ****************
   //{
   //    ofstream ofp(path + "\\points.csv", ios::trunc);
   //    for (const tup& t : points) {
   //       ofp << t.x1 << "," << t.x2 << "," << t.y << endl;
   //    }
   //}

   for (int loop = 1; loop <= 10000; loop++) {
      double avg_error = 0.0;
      int count = 0;
      for (const tup& t : points) {
         count++;
         ColVector X(2);
         ColVector Y(1);
         X(0) = t.x1;
         X(1) = t.x2;
         Y(0) = t.y;

         for (const auto& lit : LayerList) {
            X = lit->Eval(X);
         }
         double error = loss->Eval(X, Y);
         //if (!(loop % 100)) {
         //   cout << error << ",";
         //}
         double a = 1.0 / (double)count;
         double b = 1.0 - a;
         avg_error = a * error + b * avg_error;
        // cout << "------ Error: " << error << "------------" << endl;
         RowVector g = loss->LossGradient();

         for (layer_list::reverse_iterator riter = LayerList.rbegin();
            riter != LayerList.rend();
            riter++) {
            g = (*riter)->BackProp(g);
         }
      }

      err_out.Write(avg_error);
      if ( !(loop % 100) ) {
         cout << "avg error " << avg_error << " " << endl;
      }
      for (const auto& lit : LayerList) {
         double eta = 0.75;
         lit->Update(eta);
         //cout << lit->W(0, 0) << " , " << lit->W(0, 1) << endl;
      }
   }

   int l = 0;
   for (const auto& lit : LayerList) {
      l++;
      string layer = to_string(l);
      lit->Save(make_shared<OWeightsCSVFile>(path, "simple." + layer ));
      lit->Save(make_shared<IOWeightsBinaryFile>(path,"simple." + layer ));
   }

   MakeDecsionSurface(path + "\\ds");


   MakeAvgErrorSurface( points );

   cout << "Output complete. Hit a key and press Enter to close." << endl;

   char _c;
   cin >> _c;
}

void MakeDecsionSurface( string fileroot )
{
   ofstream owf(fileroot + ".csv", ios::trunc);

   assert(owf.is_open());

   ColVector w0(99);
   ColVector w1(99);
   w0.setLinSpaced(double{ -10.0 }, double{ 10.0 });
   w1.setLinSpaced(double{ -10.0 }, double{ 10.0 });

   Matrix f(99, 99);
   for (int r = 0; r < 99; r++) {
      for (int c = 0; c < 99; c++) {
         ColVector X(2);
         X(0) = w0(c);
         X(1) = w1(r);
         for (const auto& lit : LayerList) {
            X = lit->Eval(X);
         }
         f(r, c) = X(0);
      }
   }

   // octave file format
   const static Eigen::IOFormat OctaveFmt(6, 0, ", ", ";\n", "", "", "", "");
   owf << f.format(OctaveFmt);
   owf.close();
}

void MakeErrorSurface( tup_list points )
{
   const int grid = 99;
   ColVector w0(grid);
   ColVector w1(grid);
   w0.setLinSpaced(-2.0, 2.0);
   w1.setLinSpaced(-2.0, 2.0);

   Matrix w = LayerList[0]->W;

   for (int i = 0; i < grid; i++) {
      w0(i) = w0(i) + w(0, 0);
      w1(i) = w1(i) + w(0, 1);
   }

   int count = 1;
   for (const tup& t : points) {
      ofstream owf(path + "\\es." + to_string(count) + ".csv", ios::trunc);
      assert(owf.is_open());
      Matrix f(grid, grid);
      for (int r = 0; r < grid; r++) {
         for (int c = 0; c < grid; c++) {
            ColVector X(2);
            ColVector Y(1);
            X(0) = t.x1;
            X(1) = t.x2;
            Y(0) = t.y;
            LayerList[0]->W(0, 0) = w0(r);
            LayerList[0]->W(0, 1) = w1(c);
            for (const auto& lit : LayerList) {
               X = lit->Eval(X);
            }
            double e = loss->Eval(X, Y);
            f(r, c) = e;
         }
      }
      count++;

      // octave file format
      const static Eigen::IOFormat OctaveFmt(6, 0, ", ", ";\n", "", "", "", "");
      owf << f.format(OctaveFmt);
      owf.close();
   }
}

void MakeAvgErrorSurface( tup_list points )
{
   const int grid = 99;
   ColVector w0(grid);
   ColVector w1(grid);
   w0.setLinSpaced(-2.0, 2.0);
   w1.setLinSpaced(-2.0, 2.0);

   Matrix w = LayerList[0]->W;

   for (int i = 0; i < grid; i++) {
      w0(i) = w0(i) + w(0, 0);
      w1(i) = w1(i) + w(0, 1);
   }

   Matrix f(grid, grid);
   f.setZero();

   int count = 1;
   for (const tup& t : points) {


      for (int r = 0; r < grid; r++) {
         for (int c = 0; c < grid; c++) {
            ColVector X(2);
            ColVector Y(1);
            X(0) = t.x1;
            X(1) = t.x2;
            Y(0) = t.y;
            LayerList[0]->W(0, 0) = w0(r);
            LayerList[0]->W(0, 1) = w1(c);
            for (const auto& lit : LayerList) {
               X = lit->Eval(X);
            }
            double e = loss->Eval(X, Y);
            double a = 1.0 / (double)count;
            double b = 1.0 - a;
            f(r, c) = a*e + b*f(r, c);
         }
      }
      count++;
   }
   ofstream owf(path + "\\esa.csv", ios::trunc);
   assert(owf.is_open());
   // octave file format
   const static Eigen::IOFormat OctaveFmt(6, 0, ", ", ";\n", "", "", "", "");
   owf << f.format(OctaveFmt);
   owf.close();
}