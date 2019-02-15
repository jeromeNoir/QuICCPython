#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/LU>
#include <iostream>
#include "PolynomialTransforms/AssociatedLegendrePolynomial.hpp"
#include "PolynomialTransforms/WorlandPolynomial.hpp"
#include "Exceptions/Exception.hpp"
#include <shtns.h>

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

//worland polynomial for a given grid
void wnl( Eigen::Ref<QuICC::Matrix> op, int l, QuICC::Array igrid){
    QuICC::Matrix mOp(op);
    QuICC::Matrix ipoly(op);
    QuICC::Polynomial::WorlandPolynomial::Wnl(mOp, ipoly, l, igrid);
    op = mOp;
}

//diff worland
void dwnl( Eigen::Ref<QuICC::Matrix> diff, int l, QuICC::Array igrid){
    QuICC::Matrix mDiff(diff);
    QuICC::Matrix ipoly(diff);
    QuICC::Polynomial::WorlandPolynomial::dWnl(mDiff, ipoly, l, igrid);
    diff = mDiff;
}

//diff worland
void drwnl( Eigen::Ref<QuICC::Matrix> diff, int l, QuICC::Array igrid){
    QuICC::Matrix mDiff(diff);
    QuICC::Matrix ipoly(diff);
    QuICC::Polynomial::WorlandPolynomial::drWnl(mDiff, ipoly, l, igrid);
    diff = mDiff;
}

void r_1drwnl( Eigen::Ref<QuICC::Matrix> diff, int l, QuICC::Array igrid){
    QuICC::Matrix mDiff(diff);
    QuICC::Matrix ipoly(diff);
    QuICC::Polynomial::WorlandPolynomial::r_1drWnl(mDiff, ipoly, l, igrid);
    diff = mDiff;
}

void slaplwnl( Eigen::Ref<QuICC::Matrix> diff, int l, QuICC::Array igrid){
    QuICC::Matrix mDiff(diff);
    QuICC::Matrix ipoly(diff);
    QuICC::Polynomial::WorlandPolynomial::slaplWnl(mDiff, ipoly, l, igrid);
    diff = mDiff;
}

//Associated legendre polynomial
void plm( Eigen::Ref<QuICC::Matrix> op, int m, QuICC::Array igrid){
    QuICC::Matrix mOp(op);
    QuICC::Matrix ipoly(op);
    QuICC::Polynomial::AssociatedLegendrePolynomial::Plm(mOp, ipoly, m, igrid);
    op = mOp;
}

void dplm( Eigen::Ref<QuICC::Matrix> diff, int m, QuICC::Array igrid){
    QuICC::Matrix mDiff(diff);
    QuICC::Matrix iDiff(diff);
    QuICC::Matrix ipoly(diff);
    //QuICC::Polynomial::AssociatedLegendrePolynomial::dPlmA(mDiff, iDiff, m, ipoly, igrid);
    QuICC::Polynomial::AssociatedLegendrePolynomial::dPlmB(mDiff, ipoly, m, igrid);
    diff = mDiff;
}

void plm_sin( Eigen::Ref<QuICC::Matrix> op, int m, QuICC::Array igrid){
    QuICC::Matrix mOp(op);
    QuICC::Matrix ipoly(op);
    QuICC::Polynomial::AssociatedLegendrePolynomial::sin_1Plm(mOp, ipoly, m, igrid);
    op = mOp;
}

/*
 * Leo: something failing with Yrotate
void Zrotate( Eigen::Ref<QuICC::ArrayZ> Qlm, Eigen::Ref<QuICC::ArrayZ> Slm, QuICC::MHDFloat alpha, int LMAX, int MMAX){
    //std::cout<<"Qlm:" << Qlm << std::endl;
    //setting configuration
    int MRES = 1;
    enum shtns_norm shtnorm = sht_orthonormal;
    //internal arrays to compute rotations
    QuICC::ArrayZ mQlm, mSlm;
    //configuration file for shtns
    shtns_cfg shtns = shtns_create(LMAX, MMAX, MRES, shtnorm);
    int NLM  = shtns->nlm;    
    mQlm.resize(NLM);
    mSlm.resize(NLM);

    int ind = 0; 
    //transposing QuICC format [l+] into
    //TODO: Leo check limits of LMAX and MMAX  
    for(int l=0; l < LMAX + 1; ++l)
    {
        for(int m=0; m< fmin(l, MMAX)+1; ++m)
        {
        //std::cout<<"Qlm["<<ind<<"]:"<<Qlm[ind]<<std::endl; 
            mQlm[LiM(shtns, l, m)] = Qlm[ind];
            ++ind;     
        }
    }

    if(ind != Slm.size())
    {
        throw QuICC::Exception("Error in Z-rotation: Lmax and Mmax don't match SH Data!");
    }

    SH_Zrotate(shtns, mQlm.data(), alpha, mSlm.data());

    ind=0;
    //TODO: Leo check limits of LMAX and MMAX  
    for(int l=0; l < LMAX+1; ++l)
    {
        for(int m=0; m< fmin(l, MMAX)+1; ++m)
        {
            Slm[ind] = mSlm[LiM(shtns, l, m)]; 
            ++ind;     
        }
    }
    if(ind != Slm.size())
    {
        throw QuICC::Exception("Error in Z-rotation: Lmax and Mmax don't match SH Data!");
    }
}

//void Xrotate( Eigen::Ref<QuICC::ArrayZ> Qlm, Eigen::Ref<QuICC::ArrayZ> Slm, QuICC::MHDFloat beta, int LMAX, int MMAX){
void Xrotate( Eigen::Ref<QuICC::ArrayZ> Qlm, Eigen::Ref<QuICC::ArrayZ> Slm, QuICC::MHDFloat beta, int LMAX, int MMAX){
    //std::cout<<"Qlm:" << Qlm << std::endl;
    //setting configuration
    int MRES = 1;
    enum shtns_norm shtnorm = sht_orthonormal;

    //internal arrays to compute rotations in LM
    QuICC::ArrayZ mQlm, mSlm;
    int iMMAX; 

    //Padding vectors 
    if (MMAX<LMAX){iMMAX = LMAX;}
    else{iMMAX = MMAX;}

    shtns_cfg shtns = shtns_create(LMAX, iMMAX, MRES, shtnorm);
    int NLM  = shtns->nlm;    
    mQlm.resize(NLM);
    mSlm.resize(NLM);
    mQlm.setZero();

    int ind = 0; 
    
    //transposing data 
    //TODO: Leo check limits of LMAX and MMAX  
    for(int l=0; l < LMAX+1; ++l)
    {
        for(int m=0; m< fmin(l, MMAX)+1; ++m)
        {
            //std::cout<<"Qlm["<<ind<<"]:"<<Qlm[ind]<<std::endl; 
            mQlm[LiM(shtns, l, m)] = Qlm[ind];
            ++ind;     
        }
    }

    if(ind != Slm.size())
    {
        throw QuICC::Exception("Error in X-rotation: Lmax and Mmax don't match SH Data!");
    }

    //mSlm all zero, when m = 0  and Im>0 spectral coefficient 
    SH_Yrotate(shtns, mQlm.data(), -M_PI/2, mSlm.data()); 
    SH_Zrotate(shtns, mSlm.data(), beta, mQlm.data());
    SH_Yrotate(shtns, mQlm.data(), M_PI/2, mSlm.data());
 
    ind=0;
    //TODO: Leo check limits of LMAX and MMAX  
    for(int l=0; l < LMAX+1; ++l)
    {
        for(int m=0; m< fmin(l, MMAX)+1; ++m)
        {
            Slm[ind] = mSlm[LiM(shtns, l, m)]; 
            ++ind;     
        }
    }

    if(ind != Slm.size())
    {
        throw QuICC::Exception("Error in X-rotation: Lmax and Mmax don't match SH Data!");
    }
}

void XrotateFull( Eigen::Ref<QuICC::MatrixZ> QlmM, Eigen::Ref<QuICC::MatrixZ> SlmM, QuICC::MHDFloat beta, int LMAX, int MMAX){
    for( int j=0; j <QlmM.cols(); ++j)
      {
	QuICC::ArrayZ Qlm = QlmM.col(j);
	QuICC::ArrayZ Slm = SlmM.col(j);
	//std::cout<<"Qlm:" << Qlm << std::endl;
	//setting configuration
	int MRES = 1;
	enum shtns_norm shtnorm = sht_orthonormal;
	
	//internal arrays to compute rotations in LM
	QuICC::ArrayZ mQlm, mSlm;
	int iMMAX; 
	
	//Padding vectors 
	if (MMAX<LMAX){iMMAX = LMAX;}
	else{iMMAX = MMAX;}
	
	shtns_cfg shtns = shtns_create(LMAX, iMMAX, MRES, shtnorm);
	int NLM  = shtns->nlm;    
	mQlm.resize(NLM);
	mSlm.resize(NLM);
	mQlm.setZero();
	
	int ind = 0; 
	
	//transposing data 
	for(int l=0; l < LMAX+1; ++l)
	  {
	    for(int m=0; m< fmin(l, MMAX)+1; ++m)
	      {
		//std::cout<<"Qlm["<<ind<<"]:"<<Qlm[ind]<<std::endl; 
		mQlm[LiM(shtns, l, m)] = Qlm[ind];
		++ind;     
	      }
	  }
	
	
	//mSlm all zero, when m = 0  and Im>0 spectral coefficient 
	SH_Yrotate(shtns, mQlm.data(), -M_PI/2, mSlm.data()); 
	SH_Zrotate(shtns, mSlm.data(), beta, mQlm.data());
	SH_Yrotate(shtns, mQlm.data(), M_PI/2, mSlm.data());
	
	ind=0;
	for(int l=0; l < LMAX+1; ++l)
	  {
	    for(int m=0; m< fmin(l, MMAX)+1; ++m)
	      {
		Slm[ind] = mSlm[LiM(shtns, l, m)]; 
		++ind;     
	      }
	  }
	// restore the result vector into the matrix
	SlmM.col(j) = Slm;
      }
}


void ZrotateFull( Eigen::Ref<QuICC::MatrixZ> QlmM, Eigen::Ref<QuICC::MatrixZ> SlmM, QuICC::MHDFloat alpha, int LMAX, int MMAX){
    for(int  j=0; j<QlmM.cols(); ++j)
      {
	QuICC::ArrayZ Qlm = QlmM.col(j);
	QuICC::ArrayZ Slm = SlmM.col(j);
	//std::cout<<"Qlm:" << Qlm << std::endl;
	//setting configuration
	int MRES = 1;
	enum shtns_norm shtnorm = sht_orthonormal;
	//internal arrays to compute rotations
	QuICC::ArrayZ mQlm, mSlm;
	//configuration file for shtns
	shtns_cfg shtns = shtns_create(LMAX, MMAX, MRES, shtnorm);
	int NLM  = shtns->nlm;    
	mQlm.resize(NLM);
	mSlm.resize(NLM);
      
	int ind = 0; 
	//transposing QuICC format [l+] into
	for(int l=0; l < LMAX + 1; ++l)
	  {
	    for(int m=0; m< fmin(l, MMAX)+1; ++m)
	      {
		//std::cout<<"Qlm["<<ind<<"]:"<<Qlm[ind]<<std::endl; 
		mQlm[LiM(shtns, l, m)] = Qlm[ind];
		++ind;     
	      }
	  }
      
	SH_Zrotate(shtns, mQlm.data(), alpha, mSlm.data());
      
	ind=0;
	for(int l=0; l < LMAX+1; ++l)
	  {
	    for(int m=0; m< fmin(l, MMAX)+1; ++m)
	      {
		Slm[ind] = mSlm[LiM(shtns, l, m)]; 
		++ind;     
	      }
	  }
	SlmM.col(j) = Slm;
      }
}
*/

PYBIND11_MODULE(quicc_bind, m) {
    m.doc() = R"pbdoc(
        QuICC Pybind11 plugin
        -----------------------

        .. currentmodule:: quicc_bind

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");
    m.def("wnl", &wnl);
    m.def("dwnl", &dwnl);
    m.def("drwnl", &drwnl);
    m.def("r_1drwnl", &r_1drwnl);
    m.def("slaplwnl", &slaplwnl);
    m.def("plm", &plm);
    m.def("dplm", &dplm);
    m.def("plm_sin", &plm_sin);
    //m.def("Xrotate", &Xrotate);
    //m.def("Zrotate", &Zrotate);
    //m.def("XrotateFull", &XrotateFull);
    //m.def("ZrotateFull", &ZrotateFull);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}

