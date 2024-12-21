// SNcifar.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <Eigen>
#include <iostream>
#include <iomanip>
#include <Layer.h>
#include <SpacialTransformer.h>
#include <CFARReader.h>
#include <utility.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
   // Define static optomizer variables.
double optoADAM::B1 = 0.0;
double optoADAM::B2 = 0.0;
double optoLowPass::Momentum = 0.0;

int gModelBranches = 0;
int gAutoSave = 0;
int gTrainingBranch = 4;
int gWarmupOn = 0;

string path = "C:\\projects\\neuralnet\\simplenet\\SNcifar\\weights";
string model_name = "layer";

int main()
{
   string dataroot = "C:\\projects\\neuralnet\\data\\cifar\\cifar-10-batches-bin";

    std::cout << "Hello World!\n";

    CFARReader reader( dataroot + "\\data_batch_1.bin" );

    auto render = [&reader](int i) {
       Matrix red(32, 32);
       Matrix grn(32, 32);
       Matrix blu(32, 32);
       ColVector ldg(10);
       reader.read_next_ex(blu, grn, red, ldg);
       MakeMatrixRGBImage(path + "\\cf" + to_string(i) + ".bmp", red, grn, blu);
    };

    for (int i = 1; i <= 100; i++) {
       render(i);
    }
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
