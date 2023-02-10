#pragma once
#include <layer.h>

void fft2convolve(const Matrix& m, const Matrix& h, Matrix& o, int con_cor, 
                  bool force_row_pad = false, bool force_col_pad = false,
                  bool sum_into_output = false );

