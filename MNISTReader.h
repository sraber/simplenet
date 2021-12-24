#pragma once
#include <Eigen>
#include <iostream>
#include <fstream>
//#include <filesystem>
#include <string>
#include <list>
#include <vector>
// #define NDEBUG
#include <cassert>

using namespace std;

typedef double num_t;
typedef Eigen::VectorXd ColVector;

class MNISTReader
{
public:
   constexpr static size_t DIM = 28 * 28;
private:
   std::ifstream images;
   std::ifstream labels;
   uint32_t image_count_;
   // Data from the images file is read as one-byte unsigned values which are
   // converted to num_t after
   char buf_[DIM];
   // All images are resized (with antialiasing) to a 28 x 28 row-major raster
   ColVector data_;
   // One-hot encoded label
   ColVector label_;

   void read_be(std::ifstream& in, uint32_t* out)
   {
      char* buf = reinterpret_cast<char*>(out);
      in.read(buf, 4);

      std::swap(buf[0], buf[3]);
      std::swap(buf[1], buf[2]);
   }

public:
   struct MNIST_Pair {
      ColVector x;
      ColVector y;
      MNIST_Pair() {}
      MNIST_Pair(ColVector _x, ColVector _y) : x(_x), y(_y) {}
   };

   typedef vector< MNIST_Pair> MNIST_list;

   MNISTReader(string images_file, string labels_file ) : data_(DIM) , label_(10)
   {
      images.open(images_file, std::ios::binary);
      labels.open(labels_file, std::ios::binary);

      assert(images.is_open());
      assert(labels.is_open());

      reset(false);
   }

   void reset(bool _reset = true ) {
      if (_reset) {
         images.clear();
         images.seekg(0, ios::beg);
         labels.clear();
         labels.seekg(0, ios::beg);
      }

      // Confirm that passed input file streams are well-formed MNIST data sets
      uint32_t image_magic;
      read_be(images, &image_magic);
      if (image_magic != 2051)
      {
         throw std::runtime_error{ "Images file appears to be malformed" };
      }
      read_be(images, &image_count_);

      uint32_t labels_magic;
      read_be(labels, &labels_magic);
      if (labels_magic != 2049)
      {
         throw std::runtime_error{ "Labels file appears to be malformed" };
      }

      uint32_t label_count;
      read_be(labels, &label_count);
      if (label_count != image_count_)
      {
         throw std::runtime_error(
            "Label count did not match the number of images supplied");
      }

      uint32_t rows;
      uint32_t columns;
      read_be(images, &rows);
      read_be(images, &columns);
      if (rows != 28 || columns != 28)
      {
         throw std::runtime_error{
             "Expected 28x28 images, non-MNIST data supplied" };
      }

      printf("Loaded images file with %d entries\n", image_count_);
   }

   // Parse the next image and label into memory
   bool read_next() {
      // Note: The magic number, the image count, the rows number and the columns number
      //       must be read prior to this call as it is simply a continuation of reading 
      //       the file stream.
      if (!images.read(buf_, DIM)) {
         return false;
      }
      num_t inv = num_t{ 1.0 } / num_t{ 255.0 };
      for (size_t i = 0; i != DIM; ++i)
      {
         data_[i] = static_cast<uint8_t>(buf_[i]) * inv;
      }

      char label;
      labels.read(&label, 1);

      label_.setZero();
      label_[static_cast<uint8_t>(label)] = 1.0;

      return true;
   }

   MNIST_list read_batch(int batch)
   {
      MNIST_list ml;

      for (int i = 1; i <= batch; i++) {
         if (!read_next()) {
            reset();
            bool res = read_next(); // read_next can't go inside assert.  It won't be compiled on Release build.
            assert(("Error resetting batch",res));
         }
         ml.emplace_back( MNIST_Pair(data(), label()) );
      }
      return ml;
   }

   size_t size()
   {
      return image_count_;
   }

   ColVector data()
   {
      return data_;
   }

   ColVector label()
   {
      return label_;
   }

};
