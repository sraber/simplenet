// SNMulticlass.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#include <fstream>
#include <list>
#include <memory>
#include <Layer.h>
#include <random>
#define _USE_MATH_DEFINES
#include <math.h>
#include <bmp.h>

using namespace std;

void MakeDecsionSurface(string fileroot, int k);
void MakeDecsionColorMap(string fileroot, int k);

typedef list< shared_ptr<Layer> > layer_list;
layer_list LayerList;

struct tup {
   double x1;
   double x2;
   ColVector y;
   tup(double _x1, double _x2, ColVector _y) : x1(_x1), x2(_x2), y(_y) {}
};
typedef list<tup> tup_list;

void MakePointCloud(tup_list& tuples, double xc, double yc, int lab, int k, int n)
{
   random_device rd;
   mt19937 mt(rd());
   uniform_real_distribution<double> dst(-2.0, 2.0);
   for (int i = 1; i <= n; i++) {
      double x = dst(mt);
      double y = dst(mt);
      ColVector label(k);
      label.setZero();
      label(lab) = 1.0;
      tuples.push_back({ xc + x, yc + y, label });
   }
}

void MakePointRing(tup_list& tuples, double xc, double yc, double radius, int lab, int k, int n)
{
   double rad_step = 2.0 * M_PI / (double)n;
   for (int i = 0; i < n; i++) {
      double rad = rad_step * (double)i;
      double x = radius * cos(rad);
      double y = radius * sin(rad);
      ColVector label(k);
      label.setZero();
      label(lab) = 1.0;
      tuples.push_back({ xc + x, yc + y, label });
   }
}

int main(int argc, char* argv[])
{
   tup_list points;
   int k = 3;

   // The label is a 3 element vector.  1 in the lable type, zero elsewhere.

   MakePointCloud(points, 5.0, 5.0, 0, k, 20);
   MakePointCloud(points, -5.0, -5.0, 1, k, 20);
   //MakePointCloud(points, -5.0, -5.0, 2, k, 20);
   //MakePointCloud(points, 0.0, 0.0, 2, k, 20);
   MakePointRing(points, 0.0, 0.0, 9, 2, k, 40);

   //------------ setup the network ------------------------------
   LayerList.push_back(make_shared<Layer>(2, 10, make_unique<actSigmoid>(10), make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier,1)));
   LayerList.push_back(make_shared<Layer>(10, k, make_unique<actSoftMax>(k), make_shared<IWeightsToNormDist>(IWeightsToNormDist::Xavier,1)));

   LossCrossEntropy loss(k, 1);
   //-------------------------------------------------------------

   for (int loop = 1; loop < 10000; loop++) {
      double avg_error = 0.0;
      int count = 0;
      for (const tup& t : points) {
         count++;
         ColVector X(2);
         ColVector Y(3);
         X(0) = t.x1;
         X(1) = t.x2;
         Y = t.y;

         for (const auto& lit : LayerList) {
            X = lit->Eval(X);
         }
         double error = loss.Eval(X, Y);
         double a = 1.0 / (double)count;
         double b = 1 - a;
         avg_error = a * error + b * avg_error;
         // cout << "------ Error: " << error << "------------" << endl;
         RowVector g = loss.LossGradient();

         for (layer_list::reverse_iterator riter = LayerList.rbegin();
            riter != LayerList.rend();
            riter++) {
            g = (*riter)->BackProp(g);
         }
      }

      if (!(loop % 100)) {
         cout << "avg error " << avg_error << " " << endl;
      }
      for (const auto& lit : LayerList) {
         double eta = 0.75;
         lit->Update(eta);
         //cout << lit->W(0, 0) << " , " << lit->W(0, 1) << endl;
      }
   }

   /*
   int l = 0;
   for (const auto& lit : LayerList) {
      l++;
      string layer = to_string(l);
      lit->Save(make_shared<WriteWeightsCSV>("C:\\projects\\neuralnet\\simplenet\\SNMulticlass\\paralell." + layer + ".wts.csv"));
   }
   */
   
   //MakeDecsionSurface("C:\\projects\\neuralnet\\simplenet\\SNMulticlass\\ds", k);

   MakeDecsionColorMap("C:\\projects\\neuralnet\\simplenet\\SNMulticlass\\ds", k);

   cout << "Output complete" << endl;

   char c;
   cin >> c;
}

void MakeDecsionSurface(string fileroot, int k)
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
         int i_max = 0;
         double max = 0.0;
         for (int i = 0; i < k; i++) {
            if (X(i) > max) {
               max = X(i);
               i_max = i;
            }
         }

         f(r, c) = (double)i_max;
      }
   }

   // octave file format
   const static Eigen::IOFormat OctaveFmt(6, 0, ", ", ";\n", "", "", "", "");
   owf << f.format(OctaveFmt);
   owf.close();
}

void MakeDecsionColorMap(string fileroot, int k)
{
   struct pixel_data {
      unsigned char r;
      unsigned char g;
      unsigned char b;
   }pixel;

   assert(k == 3);

   ColVector w0(100);
   ColVector w1(100);
   w0.setLinSpaced(double{ -10.0 }, double{ 10.0 });
   w1.setLinSpaced(double{ -10.0 }, double{ 10.0 });

   unsigned char* pbytes = new unsigned char[100 * 100 * sizeof(pixel_data)]; // 24 bit BMP
   unsigned char* pbs = pbytes;
   for (int r = 0; r < 100; r++) {
      for (int c = 99; c >= 0; c--) {
         ColVector X(2);
         X(0) = w0(c);
         X(1) = w1(r);
         for (const auto& lit : LayerList) {
            X = lit->Eval(X);
         }

         pixel.r = static_cast<unsigned char>(X(0) * 255);
         pixel.g = static_cast<unsigned char>(X(1) * 255);
         pixel.b = static_cast<unsigned char>(X(2) * 255);

         std::memcpy(pbs, &pixel, sizeof(pixel_data));
         pbs += sizeof(pixel_data);
      }
   }

   generateBitmapImage(pbytes, 100, 100, 100 * sizeof(pixel_data), fileroot + ".bmp");
}