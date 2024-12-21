#pragma once

#include <Eigen>
#include <iostream>
#include <fstream>
#include <string>
#include <list>
#include <vector>
// #define NDEBUG
#include <cassert>

using namespace std;

typedef double num_t;
typedef Eigen::VectorXd ColVector;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;

class CFARReader
{
public:
   constexpr static size_t DIM = 32 * 32;
   constexpr static size_t ISIZE = 3 * DIM;
   constexpr static size_t SAMPLESIZE = ISIZE + 1;
private:
   std::ifstream images;
   // Data from the images file is read as one-byte unsigned values which are
   // converted to num_t after
   unsigned char buf_[SAMPLESIZE]; // One extra byte for the label.

   void TrasformVectorToMatrix(unsigned char* pbuf, Matrix& m)
   {      // Can use for scaling.
      const double scale = 1.0 /  255.0;
      for (int r = 31; r >= 0; r--) {
         for (int c = 0; c < 32; c++) {
            m(r, c) = static_cast<double>(*pbuf) * scale;
            pbuf++;
         }
      }
   }

public:
   struct CFAR_Pair {
      Matrix r;
      Matrix g;
      Matrix b;
      ColVector y;
      CFAR_Pair() : r(32,32), g(32, 32), b(32, 32), y(10) {}
      CFAR_Pair(Matrix _r, Matrix _g, Matrix _b, ColVector _y) : r(_r), g(_g), b(_b), y(_y) {}
   };

   typedef vector< CFAR_Pair> CFAR_list;

   CFARReader(){}

   CFARReader(string images_file)
   {
      images.open(images_file, std::ios::binary);

      assert(images.is_open());

      reset(false);
   }
   void open(string images_file, string labels_file) {
      images.open(images_file, std::ios::binary);

      assert(images.is_open());

      reset(false);
   }
   void reset(bool _reset = true) {
      if (_reset) {
         images.clear();
         images.seekg(0, ios::beg);
      }

      cout << "Loaded CFAR images file.\n";
   }

   // Parse the next image and label into memory
   bool read_next() {
      if (!images.read( (char*)buf_, SAMPLESIZE)) {
         return false;
      }
      return true;
   }

   bool read_next_ex(Matrix& r, Matrix& g, Matrix& b, ColVector& y) {
      if ( !read_next() ) {
         return false;
      }

      r = red();
      g = green();
      b = blue();
      y = label();

      return true;
   }

   CFAR_list read_batch(int batch)
   {
      CFAR_list ml;

      for (int i = 1; i <= batch; i++) {
         CFAR_Pair mp;
         if (!read_next_ex(mp.r, mp.g, mp.b, mp.y)) {
            reset();
            bool res = read_next_ex(mp.r, mp.g, mp.b, mp.y); // read_next can't go inside assert.  It won't be compiled on Release build.
            assert(("Error resetting batch", res));
         }
         ml.emplace_back(move(mp));
      }
      return ml;
   }

   Matrix red()
   {
      Matrix d(32,32);
      unsigned char* ptr = buf_ + 1;
      TrasformVectorToMatrix(ptr, d);
      return d;
   }

   Matrix green()
   {
      Matrix d(32, 32);
      unsigned char* ptr = buf_ + 1 + DIM;
      TrasformVectorToMatrix(ptr, d);
      return d;
   }

   Matrix blue()
   {
      Matrix d(32, 32);
      unsigned char* ptr = buf_ + 1 + DIM + DIM;
      TrasformVectorToMatrix(ptr, d);
      return d;
   }
   
   ColVector label()
   {
      ColVector l(10);
      l.setZero();
      l[ static_cast<uint8_t>(buf_[0]) ] = 1.0;
      return l;
   }

};