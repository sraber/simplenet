// SNLogic.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <memory>
#include <Layer.h>
#include <random>
#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;

void MakeDecsionSurface( string fileroot );
void MakeErrorSurface(ColVector _x, ColVector _y, string fileroot);


typedef vector< shared_ptr<Layer> > layer_list;
layer_list LayerList;
shared_ptr<iLossLayer> loss;

struct tup {
   double x1;
   double x2;
   double y;
   tup(double _x1, double _x2, double _y) : x1(_x1), x2(_x2), y(_y) {}
};
typedef list<tup> tup_list;

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


string path = "C:\\projects\\neuralnet\\simplenet\\SNLogic";

int main(int argc, char* argv[])
{
   tup_list points;
   ofstream ofp(path + "\\points.csv", ios::trunc);

   /*
   MakePointCloud(points, -5.0, 5.0, 1, 10);
   MakePointCloud(points, -5.0, 0.0, 1, 10);
   MakePointCloud(points, -5.0, -5.0, 1, 10);
   MakePointCloud(points, 0.0, 5.0, 0, 10);
   MakePointCloud(points, 0.0, 0.0, 0, 10);
   MakePointCloud(points, 0.0, -5.0, 0, 10);
   MakePointCloud(points, 5.0, 5.0, 1, 10);
   MakePointCloud(points, 5.0, 0.0, 1, 10);
   MakePointCloud(points, 5.0, -5.0, 1, 10);
   */
   //MakePointCloud(points, 5.0, 5.0, 0, 10);
   //MakePointCloud(points, 5.0, -5.0, 0, 10);
   //MakePointCloud(points, -5.0, 5.0, 1, 10);
   //MakePointCloud(points, -5.0, -5.0, 1, 10);

   //MakePointCloud(points, 5.0, 5.0, 0, 2);
   //MakePointCloud(points, 5.0, -5.0, 1, 2);

   points.push_back({ 6.5, 4.0, (double)0 });
   points.push_back({ 5.0, 4.5, (double)0 });
   points.push_back({ 4.0, -3.0, (double)1 });
   points.push_back({ 4.5, -4.0, (double)1 });


  // MakePointRing(points, 0.0, 0.0, 8.0, 1, 40);
  // MakePointRing(points, 0.0, 0.0, 5.0, 0, 40);
  // MakePointCloud(points, 0.0, 0.0, 1, 20);


   //LayerList.push_back(make_shared<Layer>(2, 1, new actSigmoid(1), make_shared<InitWeightsToRandom>(0.1)));

   //------------ setup the network ------------------------------
   //LayerList.push_back(make_shared<Layer>(2, 12, new actSigmoid(12), make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier,1)));
   //LayerList.push_back(make_shared<Layer>(12, 7, new actSigmoid(7), make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier,1)));
   //LayerList.push_back(make_shared<Layer>(7, 1, new actSigmoid(1), make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier,1)));
   //LayerList.push_back(make_shared<Layer>(12, 1, new actSigmoid(1), make_shared<InitWeightsToRandom>(0.1, 0.0)));

   LayerList.push_back(make_shared<Layer>(2, 1, new actSigmoid(1), make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier,1)));

   loss = make_shared<LossL2>(1, 1);   

   //-------------------------------------------------------------

   for (const tup& t : points) {
      ofp << t.x1 << "," << t.x2 << "," << t.y << endl;
   }

   for (int loop = 1; loop < 5000; loop++) {
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
         double a = 1.0 / (double)count;
         double b = 1 - a;
         avg_error = a * error + b * avg_error;
        // cout << "------ Error: " << error << "------------" << endl;
         RowVector g = loss->LossGradient();

         for (layer_list::reverse_iterator riter = LayerList.rbegin();
            riter != LayerList.rend();
            riter++) {
            g = (*riter)->BackProp(g);
         }
      }

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
      lit->Save(make_shared<OWeightsCSVFile>(path, "parallel." + layer ));
      lit->Save(make_shared<IOWeightsBinaryFile>(path,"parallel." + layer ));
   }

   MakeDecsionSurface("C:\\projects\\neuralnet\\simplenet\\SNLogic\\ds");

  ColVector X(2);
  ColVector Y(1);
  int count = 1;
  for (const tup& t : points) {
     X(0) = t.x1;
     X(1) = t.x2;
     Y(0) = t.y;
     MakeErrorSurface(X, Y, path + "\\es." + to_string(count) );
     count++;
   }

   cout << "Output complete" << endl;

   char _c;
   cin >> _c;
}

void MakeDecsionSurface( string fileroot )
{
   ofstream owf(fileroot + ".csv", ios::trunc);

   assert(owf.is_open());

   ColVector w0(100);
   ColVector w1(100);
   w0.setLinSpaced(double{ -10.0 }, double{ 10.0 });
   w1.setLinSpaced(double{ -10.0 }, double{ 10.0 });

   Matrix f(100, 100);
   for (int r = 0; r < 100; r++) {
      for (int c = 0; c < 100; c++) {
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

void MakeErrorSurface( ColVector _x, ColVector _y, string fileroot )
{
   ofstream owf(fileroot + ".csv", ios::trunc);

   assert(owf.is_open());

   ColVector w0(100);
   ColVector w1(100);
   w0.setLinSpaced(-2.0, 2.0);
   w1.setLinSpaced(-2.0, 2.0);

   Matrix w = LayerList[0]->W;

   Matrix f(100, 100);
   for (int r = 0; r < 100; r++) {
      for (int c = 0; c < 100; c++) {
         ColVector X(2);
         X = _x;
         LayerList[0]->W(0, 0) = w0(r);
         LayerList[0]->W(0, 1) = w1(c);
         for (const auto& lit : LayerList) {
            X = lit->Eval(X);
         }
         f(r, c) = loss->Eval(X,_y);
      }
   }

   // octave file format
   const static Eigen::IOFormat OctaveFmt(6, 0, ", ", ";\n", "", "", "", "");
   owf << f.format(OctaveFmt);
   owf.close();
}