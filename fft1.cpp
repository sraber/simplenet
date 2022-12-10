#include <windows.h>
#include <math.h>
#include <memory>
#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr

#define PI 3.141592653589793

void four1(double * pdata, const int n, const int isign)
{
	int mmax,m,j,i;
	double wr,wi;
   double tempr, tempi;
   double theta;
   double wpr,wpi,wtemp;
   
	j=1;
	for (i=1;i<n;i+=2) {
	   if (j > i) {   // bit reversal. 
		   SWAP(pdata[j-1],pdata[i-1]);
		   SWAP(pdata[j],pdata[i]);
	      }
	   m=n >> 1;
	   while (m >= 2 && j > m){
		   j -= m;
		   m >>= 1;
	      }
	   j += m;
	   }

	for (mmax = 2; mmax<n; mmax <<= 1){
      theta = (double)isign * (6.28318530717959/(double)mmax);
      wtemp=(double)sin(0.5*theta);
      wpr = -2.0f*wtemp*wtemp;
      wpi=(double)sin(theta);
      wr=1.0f;
      wi=0.0f;
	   for (m=0;m<mmax;m+=2) {
	      for (i=m;i<n; i += 2*mmax) {
		      j=i+mmax;   
		      tempr = wr*pdata[j]-wi*pdata[j+1];
		      tempi = wr*pdata[j+1]+wi*pdata[j];

		      pdata[j] = pdata[i]-tempr; 
		      pdata[j+1] = pdata[i+1]-tempi;
		      pdata[i]  += tempr; 
		      pdata[i+1] += tempi; 
	         }
         wtemp=wr;
         wr=wr*wpr-wi*wpi+wr;
         wi=wi*wpr+wtemp*wpi+wi;
	      }
	   }
}
void real_four1(double * pdata, const int n, const int isign)
{
	int i,i1,i3;
	double h1r,h1i,h2r,h2i,g1,g2;
	double wr,wi;
   double theta;
   double wpr,wpi,wtemp;

   theta = PI / (double)(n>>1);
   if( isign == (-1) ){ theta = -theta; }

   // Initialize the trig recursion relation.
   wtemp = sin(0.5*theta);
   wpr = -2.0 * wtemp*wtemp;
   wpi = sin(theta);
   wr = 1.0 + wpr;
   wi = wpi;

	i1 = 0;
	i3 = n;
	for (i=1; i<n/4; i++ ) { 
		i1 += 2 ;      // i1 walks through from 2 ... 2*(n/4-1)
		i3 -= 2;       // i3 walks through from n-2 ... n-2*(n/4-1)

		h2r = isign *(pdata[i3+1] + pdata[i1+1])/2.0f;
		h2i = isign *(pdata[i3] - pdata[i1])/2.0f;
		g1 = (double)(wr*h2r-wi*h2i);
		g2 = (double)(wr*h2i+wi*h2r);

		h1r = ( pdata[i1] + pdata[i3]) * 0.5f;
		h1i = ( pdata[i1+1] - pdata[i3+1]) * 0.5f;
		pdata[i1] = h1r + g1; 
		pdata[i3] = h1r - g1;
		pdata[i1+1] = g2 + h1i;
		pdata[i3+1] = g2 - h1i ;

      // trig recursion
      wtemp=wr;
      wr=wr*wpr-wi*wpi+wr;
      wi=wi*wpr+wtemp*wpi+wi;
	   }

	h1r = pdata[0]; 
	pdata[0] = h1r + pdata[1];
	pdata[1] = h1r - pdata[1];

	if (isign == -1){
		pdata[0] /= 2.0f;
		pdata[1] /= 2.0f;
	   }
}


//--------------------------------
// From http://www.gamedev.net/community/forums/topic.asp?topic_id=229831

// Supposedly the fastest way to find the nearest power of 2.
// Couldn't resist using it.
// It fails on zero so we must check for that.
//int __declspec( naked) __fastcall NearestPow2(int n)
//{
//	_asm
//	{
//		dec ecx
//		mov eax, 2
//		bsr ecx, ecx
//		rol eax, cl
//		ret
//	}
//}

inline 
BOOL IsPower2(int n)
{
return ((n) > 0 && !((n) & (n-1)));
}

// NOTE: See std::bit_floor( T x) implemented in C++20
unsigned nearest_power_floor(unsigned x) {
   if (IsPower2(x)) {
      return x;
   }
    int power = 1;
    while (x >>= 1) power <<= 1;
    return power;
}

// NOTE: See std::bit_ceil( T x) implemented in C++20
unsigned nearest_power_ceil(unsigned x) {
   if (IsPower2(x)) {
      return x;
   }
    if (x <= 1) return 1;
    int power = 2;
    x--;
    while (x >>= 1) power <<= 1;
    return power;
}

//---------------------------------



void ComplexMultiplyEquals( double* f, double* w, long n)
{
for(long i=0;i<2*n;i+=2){
   double rl,im;
   rl = f[i]*w[i] - f[i+1]*w[i+1];
   im = f[i]*w[i+1]+f[i+1]*w[i];
   f[i] = rl;
   f[i+1] = im;
   }
}

//***********************
// Input/Output:
//   signal: A vector of length equal to a power of 2 AND greater than or equal to (2*n+1).
//
// Input:
//   L: Length of signal.  The actual length of the array.  L/2 is what we will work with.
//   n: The number of the coefficients being z-transformed, presumably not a power of 2.
//      It is the number of complex pairs, so the array size is 2*n.
//   is:  Set to 1 - Chirp / set to -1 - Inverse Chirp
//
void MakeRabinerChirp( double* signal, long L, long n, int is )
{
double phi = PI/(double)(n);
double k;  
long i = 0;
const long m = L >> 1;

// REVIEW: Assert
// Assert(L%2==0)
// if( L < 4*n ){ cout << "Signal size is too small."; exit(0); }
// Assert( IsPower2(m) )

//NOTE: Triginometric recursion formula is dependant on the operand in sin and cos
//      to march "linearly" higher from some starting angle.  The operand in the chirp
//      function is not linear and so we cannot use the recursion formual to lessen the
//      computational load of repeatedly computing sin and cos.

for (long p=0; p<m; p++, i+=2){
   if( p < n ){
      k = (double)p;
      signal[i] = cos(phi*k*k);
      if( is>=0 ){
         signal[i+1] = sin(phi*k*k);
         }
      else{
         signal[i+1] = -sin(phi*k*k);
         }
      }
   else if( p >= m-n ){
      k = (double)(m-p);
      signal[i] = cos(phi*k*k);
      if( is>=0 ){
         signal[i+1] = sin(phi*k*k);
         }
      else{
         signal[i+1] = -sin(phi*k*k);
         }
      }
   else{
      signal[i] = 0.0f;    // When 2*n is not a power of 2 this zero
      signal[i+1] = 0.0f;  // padded area emerges.
      }
   }
}

//-------------------------------------
// NOTE: I'll leave this optomized version of the Chirp function here,
//       but after doing timing tests we found that even the long version
//       is blazing fast.  The optomized version here is about twice as fast
//       but the obviscated code doesn't seem worth it given that this function
//       is unlikly to be a performance bottleneck.
void MakeRabinerChirp1( double* signal, long L, long n, int is )
{
double phi = PI/(double)(n);
double k,t;  
const long m = L >> 1;

// REVIEW: Assert
// Assert(L%2==0)
// if( L < 4*n ){ cout << "Signal size is too small."; exit(0); }
// Assert( IsPower2(m) )

//NOTE: Triginometric recursion formula is dependant on the operand in sin and cos
//      to march "linearly" higher from some starting angle.  The operand in the chirp
//      function is not linear and so we cannot use the recursion formual to lessen the
//      computational load of repeatedly computing sin and cos.
long i = 0;
for (long p=0; p<n; p++, i+=2){
   t = phi*(double)(p*p);
   signal[i] = cos(t);
   if( is>=0 ){
      signal[i+1] = sin(t);
      }
   else{
      signal[i+1] = -sin(t);
      }
   }

double* p1r = signal + (L-2*n);
double* p1i = p1r + 1;
double* p2i = signal + (2*n-1);
double* p2r = p2i - 1;
for( ;p2i>signal;p2i-=2,p2r-=2,p1r+=2,p1i+=2 ){ 
   *p1r = *p2r; 
   *p1i = *p2i; 
   }

double* p1 = signal + 2*n;
double* e1 = signal + (L-2*n);
for( ;p1<e1;p1++ ){ *p1 = 0.0; }
}

//***********************
// Input/Output:
//   s: A complex vector of length equal to an even number n.
//      There are n/2 complex pairs.
//
// Input:
//   n: The length of the data set being z-transformed, presumably not a power of 2.
//      It is the number of complex pairs, so the array size is 2*n.
//  is: Set to 1 - Chirp / set to -1 - Inverse Chirp
//
void ChirpZ( double* s, long n, int is )
{
int n2 = n >> 1;
const int m = 2 * nearest_power_ceil(n+1);

std::auto_ptr<double> t(new double[m]);
std::auto_ptr<double> w(new double[m]);
std::auto_ptr<double> dw(new double[m]);

double *p1, *p2;
double *e1, *e2;

p1 = t.get();
e1 = p1 + n;
e2 = p1 + m;
p2 = s;
for(;p1<e1;p1++,p2++){ *p1 = *p2; }
for(;p1<e2;p1++){      *p1 = 0.0; }

MakeRabinerChirp1(w.get(),m,n2,is);
MakeRabinerChirp1(dw.get(),m,n2,-is);

ComplexMultiplyEquals( t.get(), w.get(), n2 ); // s = s * w for elements 0 to n-1

//******** Convolution ***************
four1(t.get(),m,1);
four1(dw.get(),m,1);
ComplexMultiplyEquals( t.get(), dw.get(), m>>1 );
four1(t.get(),m,-1);
//************************************

ComplexMultiplyEquals( t.get(), w.get(), n2 );

// Transfer results to input/output vector.
const long m2 = m >> 1;
const double mul = 1.0/(double)(m2);
p1 = s;
p2 = t.get();
e1 = p1 + n;
for(;p1<e1;p1++,p2++){ *p1 = *p2 * mul; }
}

int rfftsine(double * pdata, const int n, const int isign)
{
if( IsPower2(n) ){
    if (isign == 1){
	   four1(pdata,n,isign);
	   real_four1(pdata,n,isign);
      }
    else{
	   real_four1(pdata,n,-1);
	   four1(pdata,n,-1);
      }
   }
else if(!(n%2)){
   if (isign == 1){
      ChirpZ(pdata,n,isign);
      real_four1(pdata,n,isign);
      }
   else{
      real_four1(pdata,n,isign);
      ChirpZ(pdata,n,isign);
      }
   }
else{
   return 0;
   } 
	
 return 1;
}

int cfftsine( double * pdata, const int n, const int isign)
{
	if( !IsPower2(n) )  {
    	return -1; 
	}          
		
	four1(pdata,n,isign);
    return 1;
}
