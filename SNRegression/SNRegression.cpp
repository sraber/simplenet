// SNRegression.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#include <fstream>
#include <list>
#include <memory>
#include "CSVReader.h"
#include <Layer.h>

using namespace std;

typedef list<MPG_Data> data_list;

//-------- Utilities --------------------
void write_weight_mpg_to_file(data_list& llist, string filename)
{
   std::ofstream fout(filename, ios::trunc);
   assert(fout.is_open());

   for (const MPG_Data& lit : llist) {
      fout << lit.weight << "," << lit.mpg << endl;
   }
   fout.close();
}

void normalize_weight_mpg(data_list& dat)
{
   double wsum = 0.0;
   double msum = 0.0;
   int n = (int)dat.size();
   for (const MPG_Data& idat : dat) {
      wsum += idat.weight;
      msum += idat.mpg;
   }

   wsum /= (double)n;
   msum /= (double)n;

   double wstd = 0.0;
   double mstd = 0.0;
   for (const MPG_Data& idat : dat) {
      wstd += (wsum - idat.weight) * (wsum - idat.weight);
      mstd += (msum - idat.mpg) * (msum - idat.mpg);
   }

   wstd = sqrt(wstd / (double)n);
   mstd = sqrt(mstd / (double)n);

   for (MPG_Data& idat : dat) {
      idat.weight = (float)(((double)idat.weight - wsum) / wstd);
      //idat.mpg = (float)(((double)idat.mpg - msum) / mstd);
   }
}

//---------------------------------------
string path = "C:\\projects\\neuralnet\\simplenet\\SNRegression";

typedef list< shared_ptr<Layer> > layer_list;
layer_list LayerList;

void TestMPG(void)
{
   LayerList.push_back(make_shared<Layer>(1, 1, make_unique<actLinear>(1),
                       make_shared<IOWeightsBinaryFile>(path, "SNRegress.1") ));
   ColVector X(1);
   X(0) = -0.5;
   for (const auto& lit : LayerList) {
      X = lit->Eval(X);
   }

   cout << X << endl;
}

void MakeSurface(ColVector& x, ColVector& y, string file);


int main(int argc, char* argv[])
{
   //TestMPG();
   //exit(0);

   list<MPG_Data> dat;
   dat = ReadMPGData("C:\\papers\\ai\\data\\auto-mpg.data");
   normalize_weight_mpg(dat);

   //write_weight_mpg_to_file(dat, "C:\\projects\\neuralnet\\simplenet\\SNRegression\\mpgdat1.csv" );
   //exit(0);


   //------------ setup the network ------------------------------
   LayerList.push_back( make_shared<Layer>(1, 1, make_unique<actLinear>(1), make_shared<IWeightsToConstants>(0.1, 0.0)
      /*, "C:\\projects\\neuralnet\\simplenet\\SNRegression\\weights.csv"*/));
   LossL2 loss(1, 1);
   //-------------------------------------------------------------
   /*
   Layer ly(1, 1, make_unique<actLinear>(1), make_shared<InitWeightsToConstants>(-6.5, 23.515));

   ColVector X(1);
   ColVector Y(1);
   X(0) = 0.461382;
   Y(0) = 20.6;

   double e = 1.0e-6;
   double w = 1.0;
   ly.W(0,0) = w + e;
   double g1 = loss.Eval(ly.Eval(X), Y);

   ly.W(0, 0) = w - e;
   double g2 = loss.Eval(ly.Eval(X), Y);

   cout << (g1 - g2) / (2.0 * e) << endl;
   cout << ly.BackProp( loss.LossGradient(), true );

   exit(0);
   */
   //ColVector X(1);
   //ColVector Y(1);
   //X(0) = 0.631615;
   //Y(0) = 18.0;
   //MakeSurface(X, Y, "C:\\projects\\neuralnet\\simplenet\\SNRegression\\surf1");
   //exit(0);

   for (int loop = 1; loop < 20; loop++) {
      for (std::list<MPG_Data>::iterator diter = dat.begin(); diter != dat.end(); diter++) {
         ColVector X(1);
         ColVector Y(1);
         X(0) = diter->weight;
         Y(0) = diter->mpg;

         for (const auto& lit : LayerList) {
            X = lit->Eval(X);
         }
         double error = loss.Eval(X, Y);
         //cout << "------ Error: " << error << "------------" << endl;
         RowVector g = loss.LossGradient();

         for (layer_list::reverse_iterator riter = LayerList.rbegin();
            riter != LayerList.rend();
            riter++) {
            g = (*riter)->BackProp(g);
         }
      }
      for (const auto& lit : LayerList) {
         double eta = 0.5;
         lit->Update(eta);
         cout << lit->W(0, 0) << " , " << lit->W(0, 1) << endl;
      }
   }

   int l = 0;
   for (const auto& lit : LayerList) {
      l++;
      string layer = to_string(l);
      lit->Save(make_shared<IOWeightsBinaryFile>(path, "SNRegress." + layer ));
   }

   char c;
   cin >> c;

}

void MakeSurface(ColVector& x, ColVector& y, string fileroot)
{
   ofstream owf(fileroot + ".L.oct", ios::trunc);
   ofstream owe(fileroot + ".E.oct", ios::trunc);

   assert(owf.is_open());
   assert(owe.is_open());

   ColVector w0(100);
   ColVector w1(100);
   w0.setLinSpaced(double{ -26.0 }, double{ 13.0 });
   w1.setLinSpaced(double{ 12.0 }, double{ 36.0 });

   shared_ptr<Layer> pl = *LayerList.begin();
   LossL2 loss(1, 1);

   Matrix f(100, 100);
   Matrix e(100, 100);
   for (int r = 0; r < 100; r++) {
      for (int c = 0; c < 100; c++) {
         pl->W(0, 0) = w0[r];
         pl->W(0, 1) = w1[c];
         ColVector lf = pl->Eval(x);
         f(r, c) = lf[0];
         e(r, c) = loss.Eval(lf, y);
      }
   }

   // octave file format
   const static Eigen::IOFormat OctaveFmt(6, 0, ", ", ";\n", "", "", "", "");
   owf << f.format(OctaveFmt);
   owe << e.format(OctaveFmt);
   owf.close();
   owe.close();
}
