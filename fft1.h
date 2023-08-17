#pragma once

int rfftsine(double * pdata, const int n, const int isign);
int cfftsine( double * pdata, const int n, const int isign);
void real_four1(double * pdata, const int n, const int isign);

void MakeRabinerChirp( double* signal, long L, long n, int is );
void MakeRabinerChirp1( double* signal, long L, long n, int is );

unsigned int nearest_power_floor(unsigned int x);

unsigned int nearest_power_ceil(unsigned int x);