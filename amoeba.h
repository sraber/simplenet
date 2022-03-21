#pragma once
#include <Eigen>
typedef double Doub;
typedef int Int;
typedef ColVector VecDoub, VecDoub_I, VecDoub_O, VecDoub_IO;
typedef Matrix MatDoub, MatDoub_I, MatDoub_O, MatDoub_IO;

template<class T>
inline void SWAP(T &a, T &b)
	{T dum=a; a=b; b=dum;}

using AmoFunc = double (*)(ColVector);

#define EIGMAT
#ifdef EIGMAT
	#define nrows rows
	#define ncols cols
#endif 

struct Amoeba {
	const Doub ftol;
	Int nfunc;
	Int mpts;
	Int ndim;
	Doub fmin;
	VecDoub y;
	MatDoub p;
	Amoeba(const Doub ftoll) : ftol(ftoll) {}
	template <class T>
	VecDoub minimize(VecDoub_I &point, const Doub del, T &func)
	{
		VecDoub dels(point.size(),del);
		return minimize(point,dels,func);
	}
	template <class T>
	VecDoub minimize(VecDoub_I &point, VecDoub_I &dels, T &func)
	{
		Int ndim=point.size();
		MatDoub pp(ndim+1,ndim);
#ifdef EIGMAT
		for (Int i=0;i<ndim+1;i++) {
			for (Int j = 0; j < ndim; j++) {
				pp(i,j) = point[j];
			}
			if (i !=0 ) pp(i,i-1) += dels[i-1];
		}
#else
		for (Int i=0;i<ndim+1;i++) {
			for (Int j = 0; j < ndim; j++) {
				pp[i][j] = point[j];
			}
			if (i !=0 ) pp[i][i-1] += dels[i-1];

		}
#endif
		return minimize(pp,func);
	}
	template <class T>
	VecDoub minimize(MatDoub_I &pp, T &func)
	{
		const Int NMAX=5000;
		const Doub TINY=1.0e-10;
		Int ihi,ilo,inhi;
		mpts=pp.nrows();
		ndim=pp.ncols();
		VecDoub psum(ndim),pmin(ndim),x(ndim);
		p=pp;
		y.resize(mpts);
		for (Int i=0;i<mpts;i++) {
#ifdef EIGMAT
			x=p.row(i);
#else
			for (Int j = 0; j < ndim; j++) { x[j] = p[i][j]; }
#endif
			y[i]=func(x);
		}
		nfunc=0;
		get_psum(p,psum);
		for (;;) {
			ilo=0;
			ihi = y[0]>y[1] ? (inhi=1,0) : (inhi=0,1);
			for (Int i=0;i<mpts;i++) {
				if (y[i] <= y[ilo]) ilo=i;
				if (y[i] > y[ihi]) {
					inhi=ihi;
					ihi=i;
				} else if (y[i] > y[inhi] && i != ihi) inhi=i;
			}
			Doub rtol=2.0*abs(y[ihi]-y[ilo])/(abs(y[ihi])+abs(y[ilo])+TINY);
			if (rtol < ftol) {
				SWAP(y[0],y[ilo]);
				for (Int i=0;i<ndim;i++) {
#ifdef EIGMAT
					SWAP(p(0,i),p(ilo,i));
					pmin[i]=p(0,i);
#else
					SWAP(p[0,i],p[ilo,i]);
					pmin[i]=p[0,i];
#endif
				}
				fmin=y[0];
				return pmin;
			}
			if (nfunc >= NMAX) throw("NMAX exceeded");
			nfunc += 2;
			Doub ytry=amotry(p,y,psum,ihi,-1.0,func);
			if (ytry <= y[ilo])
				ytry=amotry(p,y,psum,ihi,2.0,func);
			else if (ytry >= y[inhi]) {
				Doub ysave=y[ihi];
				ytry=amotry(p,y,psum,ihi,0.5,func);
				if (ytry >= ysave) {
					for (Int i=0;i<mpts;i++) {
						if (i != ilo) {
							for (Int j=0;j<ndim;j++)
#ifdef EIGMAT
								p(i,j)=psum[j]=0.5*(p(i,j)+p(ilo,j));
#else
								p[i][j]=psum[j]=0.5*(p[i][j]+p[ilo][j]);
#endif
							y[i]=func(psum);
						}
					}
					nfunc += ndim;
					get_psum(p,psum);
				}
			} else --nfunc;
		}
	}
	inline void get_psum(MatDoub_I &p, VecDoub_O &psum)
	{
		for (Int j=0;j<ndim;j++) {
			Doub sum=0.0;
			for (Int i=0;i<mpts;i++)
#ifdef EIGMAT
				sum += p(i,j);
#else
				sum += p[i][j];
#endif
			psum[j]=sum;
		}
	}
	template <class T>
	Doub amotry(MatDoub_IO &p, VecDoub_O &y, VecDoub_IO &psum,
		const Int ihi, const Doub fac, T &func)
	{
		VecDoub ptry(ndim);
		Doub fac1=(1.0-fac)/ndim;
		Doub fac2=fac1-fac;
		for (Int j=0;j<ndim;j++)
#ifdef EIGMAT
			ptry[j]=psum[j]*fac1-p(ihi,j)*fac2;
#else
			ptry[j]=psum[j]*fac1-p[ihi][j]*fac2;
#endif
		Doub ytry=func(ptry);
		if (ytry < y[ihi]) {
			y[ihi]=ytry;
			for (Int j=0;j<ndim;j++) {
#ifdef EIGMAT
				psum[j] += ptry[j]-p(ihi,j);
				p(ihi,j)=ptry[j];
#else
				psum[j] += ptry[j]-p[ihi][j];
				p[ihi][j]=ptry[j];
#endif
			}
		}
		return ytry;
	}
};
