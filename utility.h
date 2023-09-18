#ifndef _UTILITY_H
#define _UTILITY_H

#include <layer.h>
#include <bmp.h>

void MakeMatrixRGBImage(string file, Matrix rm, Matrix gm, Matrix bm);
void MakeMatrixImage(std::string file, Matrix m);
void ScaleToOne(double* pdata, int size);

template<typename MatrixType>
void PutMatrix(const MatrixType& m, string filename) {
   struct MatHeader {
      int rows;
      int cols;
      int step;
   };
   ofstream file(filename, ios::trunc | ios::binary | ios::out);
   runtime_assert(file.is_open());
   MatHeader header;
   header.rows = (int)m.rows();
   header.cols = (int)m.cols();
   header.step = sizeof(Matrix::Scalar);

   file.write(reinterpret_cast<char*>(&header), sizeof(MatHeader));
   for (int r = 0; r < header.rows; r++) {
      for (int c = 0; c < header.cols; c++) {
         double v = m(r, c);
         file.write((char*)&v, header.step);
      }
   }
   file.close();
}

template<typename MatrixType>
void GetMatrix(MatrixType& m, string filename) {
   struct MatHeader {
      int rows;
      int cols;
      int step;
   };
   ifstream file(filename, ios::in | ios::binary);
   runtime_assert(file.is_open());

   MatHeader header;
   file.read(reinterpret_cast<char*>((&header)), sizeof(MatHeader));
   runtime_assert(header.step == sizeof(typename Matrix::Scalar));
   m.resize(header.rows, header.cols);

   for (int r = 0; r < header.rows; r++) {
      for (int c = 0; c < header.cols; c++) {
         double v;
         file.read((char*)&v, header.step);
         m(r, c) = static_cast<int>(v);
      }
   }

   file.close();
}

class OMultiWeightsBMP : public iPutWeights{
   string Path;
   string RootName;
public:
   OMultiWeightsBMP(string path, string root_name) : RootName(root_name), Path(path) {}
   void Write(const Matrix& m, int k) {
      string str_count;
      str_count = to_string(k);
      string pathname = Path + "\\" + RootName + "." + str_count + ".bmp";
      Matrix temp;
      temp = m;
      ScaleToOne(temp.data(), (int)temp.size());
      MakeMatrixImage(pathname, temp);
   }
};

#endif // !_UTILITY_H
