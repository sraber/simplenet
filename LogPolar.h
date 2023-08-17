#pragma once

typedef std::vector < std::vector< std::tuple<int, int, int, int, double, double>>> LogPolarSupportMatrix;

struct LogPolarSupportMatrixCenter
{
   double row_center;
   double col_center;
   double rad_max;
};

LogPolarSupportMatrix PrecomputeLogPolarSupportMatrix(int in_rows, int in_cols, int out_rows, int out_cols, LogPolarSupportMatrixCenter* pcenter = NULL);
LogPolarSupportMatrix PrecomputeLogPolarSupportMatrix1(int in_rows, int in_cols, int out_rows, int out_cols, LogPolarSupportMatrixCenter* pcenter = NULL);
void ConvertToLogPolar(Matrix& m, Matrix& out, LogPolarSupportMatrix& lpsm);
void ConvertLogPolarToCart(const Matrix& m, Matrix& out, LogPolarSupportMatrixCenter lpsmc);

