#ifndef _UTILITY_H
#define _UTILITY_H

#include <layer.h>
#include <bmp.h>

void MakeMatrixRGBImage(string file, Matrix rm, Matrix gm, Matrix bm);
void MakeMatrixImage(std::string file, Matrix m);
void ScaleToOne(double* pdata, int size);

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
