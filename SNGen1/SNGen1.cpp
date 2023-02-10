// SNGen1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <Eigen>
#include <iostream>
#include <strstream>
#include <iomanip>
//#include <MNISTReader.h>
#include <Layer.h>
#include <utility.h>

using namespace std;

typedef list< shared_ptr<Layer> > layer_list;
layer_list LayerList;

int main()
{
   //------------ setup the network ------------------------------
   const double s = 1.5;
   const int k = 16;
   LayerList.push_back(make_shared<Layer>(2, k, make_unique<actTanh>(k), make_shared<IWeightsToRandom>(s)));
   LayerList.push_back(make_shared<Layer>(k, k, make_unique<actTanh>(k), make_shared<IWeightsToRandom>(s)));
   LayerList.push_back(make_shared<Layer>(k, k, make_unique<actTanh>(k), make_shared<IWeightsToRandom>(s)));
   LayerList.push_back(make_shared<Layer>(k, k, make_unique<actTanh>(k), make_shared<IWeightsToRandom>(s)));
   LayerList.push_back(make_shared<Layer>(k, k, make_unique<actTanh>(k), make_shared<IWeightsToRandom>(s)));
   LayerList.push_back(make_shared<Layer>(k, k, make_unique<actTanh>(k), make_shared<IWeightsToRandom>(s)));
   LayerList.push_back(make_shared<Layer>(k, k, make_unique<actTanh>(k), make_shared<IWeightsToRandom>(s)));
   LayerList.push_back(make_shared<Layer>(k, k, make_unique<actTanh>(k), make_shared<IWeightsToRandom>(s)));
   LayerList.push_back(make_shared<Layer>(k, k, make_unique<actTanh>(k), make_shared<IWeightsToRandom>(s)));
   LayerList.push_back(make_shared<Layer>(k, 3, make_unique<actSigmoid>(k), make_shared<IWeightsToRandom>(s)));
   //-------------------------------------------------------------

   const int w = 1080;
   const int h = 1920;

   Matrix r(w, h);
   Matrix g(w, h);
   Matrix b(w, h);

   for (int i = 0; i < w; i++) {
      for (int j = 0; j < h; j++) {
         ColVector v(2);
         v(0) = ((double)i/w) - 0.5;
         v(1) = ((double)j / h) - 0.5;
         for (auto li : LayerList) {
            v = li->Eval(v);
         }
         r(i, j) = v(0);
         g(i, j) = v(1);
         b(i, j) = v(2);
      }
   }

   string path = "C:\\projects\\neuralnet\\simplenet\\SNGen1";

   MakeMatrixRGBImage(path + "\\pic.bmp", r, g, b);
}


