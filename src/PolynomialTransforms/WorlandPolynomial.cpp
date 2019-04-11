/** 
 * @file WorlandPolynomial.cpp
 * @brief Source of the implementation of the Jones-Worland polynomial
 * @author Philippe Marti \<philippe.marti@colorado.edu\>
 */

// System includes
//

// External includes
//

// Class include
//
#include "PolynomialTransforms/WorlandPolynomial.hpp"

// Project includes
//
#include "Exceptions/Exception.hpp"

#define QUICC_WORLAND_NORM_UNITY
//#define QUICC_WORLAND_NORM_NATURAL

namespace QuICC {

namespace Polynomial {

   internal::MHDFloat WorlandPolynomial::alpha(const int l)
   {
      return MHD_MP(-0.5);
   }

   internal::MHDFloat WorlandPolynomial::beta(const int l)
   {
      return internal::MHDFloat(l) - MHD_MP(0.5);
   }

   //Worland Polynomial
   void WorlandPolynomial::Wnl(Matrix& poly, internal::Matrix& ipoly, const int l, const internal::Array& igrid)
   {
      int gN = poly.rows();
      int nPoly = poly.cols();

      if (l < 0)
      {
         throw Exception("Tried to compute Worland polynomial W_n^l with l < 0");
      }

      if (nPoly < 1)
      {
         throw Exception("Operator matrix should have at least 1 column");
      }

      if (gN != igrid.size())
      {
         throw Exception("Operator matrix does not mach grid size");
      }

      internal::MHDFloat a = WorlandPolynomial::alpha(l);
      internal::MHDFloat b = WorlandPolynomial::beta(l);

      ipoly.resize(gN, nPoly);
      WorlandPolynomial::W0l(ipoly.col(0), l, a, b, igrid, WorlandPolynomial::normWP0ab());

      // Make X grid in [-1, 1]
      internal::Array ixgrid = MHD_MP(2.0)*igrid.array()*igrid.array() - MHD_MP(1.0);

      if(nPoly > 1)
      {
         ThreeTermRecurrence::P1(ipoly.col(1), a, b, ipoly.col(0), ixgrid, WorlandPolynomial::normWP1ab());
      }

      for(int i = 2; i < nPoly; ++i)
      {
         ThreeTermRecurrence::Pn(ipoly.col(i), i, a, b, ipoly.col(i-1), ipoly.col(i-2), ixgrid, WorlandPolynomial::normWPnab());
      }

      poly = Precision::cast(ipoly);
   }

   void WorlandPolynomial::dWn0(Matrix& diff, internal::Matrix& idiff, const internal::Array& igrid)
   {
      int gN = diff.rows();
      int nPoly = diff.cols();

      if (nPoly < 1)
      {
         throw Exception("Operator matrix should have at least 1 column");
      }

      if (gN != igrid.size())
      {
         throw Exception("Operator matrix does not mach grid size");
      }

      internal::MHDFloat a1 = WorlandPolynomial::alpha(0) + MHD_MP(1.0);
      internal::MHDFloat b1 = WorlandPolynomial::beta(0) + MHD_MP(1.0);

      // Make X grid in [-1, 1]
      internal::Array ixgrid = MHD_MP(2.0)*igrid.array()*igrid.array() - MHD_MP(1.0);

      idiff.resize(gN, nPoly);
      idiff.col(0).setZero();

      if(nPoly > 1)
      {
         WorlandPolynomial::W0l(idiff.col(1), 1, a1, b1, igrid, WorlandPolynomial::normWDP0ab());
      }

      if(nPoly > 2)
      {
         ThreeTermRecurrence::P1(idiff.col(2), a1, b1, idiff.col(1), ixgrid, WorlandPolynomial::normWDP1ab());
      }

      for(int i = 3; i < nPoly; ++i)
      {
         ThreeTermRecurrence::Pn(idiff.col(i), i-1, a1, b1, idiff.col(i-1), idiff.col(i-2), ixgrid, WorlandPolynomial::normWDPnab());
      }

      diff = Precision::cast(idiff);
   }

   void WorlandPolynomial::dWnl(Matrix& diff, internal::Matrix& idiff, const int l, const internal::Array& igrid)
   {
      if(l < 0)
      {
         throw Exception("Tried to compute Worland derivative with l < 0");
      }

      if(l == 0)
      {
         WorlandPolynomial::dWn0(diff, idiff, igrid);
      } else
      {
         int gN = diff.rows();
         int nPoly = diff.cols();

         if (nPoly < 1)
         {
            throw Exception("Operator matrix should have at least 1 column");
         }

         if (gN != igrid.size())
         {
            throw Exception("Operator matrix does not mach grid size");
         }

         internal::MHDFloat a = WorlandPolynomial::alpha(l);
         internal::MHDFloat b = WorlandPolynomial::beta(l);
         internal::MHDFloat a1 = WorlandPolynomial::alpha(l) + MHD_MP(1.0);
         internal::MHDFloat b1 = WorlandPolynomial::beta(l) + MHD_MP(1.0);
         internal::MHDFloat dl = internal::MHDFloat(l);

         // Make X grid in [-1, 1]
         internal::Array ixgrid = MHD_MP(2.0)*igrid.array()*igrid.array() - MHD_MP(1.0);

         // Storage for P_n^{(alpha,beta)} and dP_n{(alpha,beta)}
         internal::Matrix ipnab(gN,2);
         internal::Matrix idpnab(gN,2);
         idiff.resize(gN, nPoly);

         // Compute P_0
         WorlandPolynomial::W0l(ipnab.col(0), l-1, a, b, igrid, WorlandPolynomial::normWP0ab());
         ipnab.col(0) *= dl;

         // Compute DP_0
         idpnab.col(0).setZero();

         // Compute l P
         idiff.col(0) = ipnab.col(0);

         if(nPoly > 1)
         {
            // Compute P_0
            ThreeTermRecurrence::P1(ipnab.col(1), a, b, ipnab.col(0), ixgrid, WorlandPolynomial::normWP1ab());

            // Compute DP_1
            WorlandPolynomial::W0l(idpnab.col(0), l+1, a1, b1, igrid, WorlandPolynomial::normWDP0ab());

            // Compute e P + 4r^2 DP
            idiff.col(1) = ipnab.col(1) + idpnab.col(0);
         }

         if(nPoly > 2)
         {
            // Increment P_n
            ThreeTermRecurrence::Pn(ipnab.col(0), 2, a, b, ipnab.col(1), ipnab.col(0), ixgrid, WorlandPolynomial::normWPnab());
            ipnab.col(0).swap(ipnab.col(1));

            // Compute DP_2
            ThreeTermRecurrence::P1(idpnab.col(1), a1, b1, idpnab.col(0), ixgrid, WorlandPolynomial::normWDP1ab());

            // Compute e P + 2(x+1) DP
            idiff.col(2) = ipnab.col(1) + idpnab.col(1);
         }

         for(int i = 3; i < nPoly; ++i)
         {
            // Increment P_n
            ThreeTermRecurrence::Pn(ipnab.col(0), i, a, b, ipnab.col(1), ipnab.col(0), ixgrid, WorlandPolynomial::normWPnab());
            ipnab.col(0).swap(ipnab.col(1));

            // Increment DP_n
            ThreeTermRecurrence::Pn(idpnab.col(0), i-1, a1, b1, idpnab.col(1), idpnab.col(0), ixgrid, WorlandPolynomial::normWDPnab());
            idpnab.col(0).swap(idpnab.col(1));

            // Compute e P + 2(x+1) DP
            idiff.col(i) = ipnab.col(1) + idpnab.col(1);
         }

         diff = Precision::cast(idiff);
      }
   }

   void WorlandPolynomial::drWnl(Matrix& diff, internal::Matrix& idiff, const int l, const internal::Array& igrid)
   {
      int gN = diff.rows();
      int nPoly = diff.cols();

      if(l < 0)
      {
         throw Exception("Tried to compute Worland D r operator with l < 0");
      }

      if (nPoly < 1)
      {
         throw Exception("Operator matrix should have at least 1 column");
      }

      if (gN != igrid.size())
      {
         throw Exception("Operator matrix does not mach grid size");
      }

      internal::MHDFloat a = WorlandPolynomial::alpha(l);
      internal::MHDFloat b = WorlandPolynomial::beta(l);
      internal::MHDFloat a1 = WorlandPolynomial::alpha(l) + MHD_MP(1.0);
      internal::MHDFloat b1 = WorlandPolynomial::beta(l) + MHD_MP(1.0);
      internal::MHDFloat dl1 = internal::MHDFloat(l+1);

      // Make X grid in [-1, 1]
      internal::Array ixgrid = MHD_MP(2.0)*igrid.array()*igrid.array() - MHD_MP(1.0);

      // Storage for P_n^{(alpha,beta)} and dP_n{(alpha,beta)}
      internal::Matrix ipnab(gN,2);
      internal::Matrix idpnab(gN,2);
      idiff.resize(gN, nPoly);

      // Compute P_0
      WorlandPolynomial::W0l(ipnab.col(0), l, a, b, igrid, WorlandPolynomial::normWP0ab());
      ipnab.col(0) *= dl1;

      // Compute DP_0
      idpnab.col(0).setZero();

      // Compute l P
      idiff.col(0) = ipnab.col(0);

      if(nPoly > 1)
      {
         // Compute P_0
         ThreeTermRecurrence::P1(ipnab.col(1), a, b, ipnab.col(0), ixgrid, WorlandPolynomial::normWP1ab());

         // Compute DP_1
         WorlandPolynomial::W0l(idpnab.col(0), l+2, a1, b1, igrid, WorlandPolynomial::normWDP0ab());

         // Compute e P + 4r^2 DP
         idiff.col(1) = ipnab.col(1) + idpnab.col(0);
      }

      if(nPoly > 2)
      {
         // Increment P_n
         ThreeTermRecurrence::Pn(ipnab.col(0), 2, a, b, ipnab.col(1), ipnab.col(0), ixgrid, WorlandPolynomial::normWPnab());
         ipnab.col(0).swap(ipnab.col(1));

         // Compute DP_2
         ThreeTermRecurrence::P1(idpnab.col(1), a1, b1, idpnab.col(0), ixgrid, WorlandPolynomial::normWDP1ab());

         // Compute e P + 2(x+1) DP
         idiff.col(2) = ipnab.col(1) + idpnab.col(1);
      }

      for(int i = 3; i < nPoly; ++i)
      {
         // Increment P_n
         ThreeTermRecurrence::Pn(ipnab.col(0), i, a, b, ipnab.col(1), ipnab.col(0), ixgrid, WorlandPolynomial::normWPnab());
         ipnab.col(0).swap(ipnab.col(1));

         // Increment DP_n
         ThreeTermRecurrence::Pn(idpnab.col(0), i-1, a1, b1, idpnab.col(1), idpnab.col(0), ixgrid, WorlandPolynomial::normWDPnab());
         idpnab.col(0).swap(idpnab.col(1));

         // Compute e P + 2(x+1) DP
         idiff.col(i) = ipnab.col(1) + idpnab.col(1);
      }

      diff = Precision::cast(idiff);
   }

   void WorlandPolynomial::r_1drWnl(Matrix& diff, internal::Matrix& idiff, const int l, const internal::Array& igrid)
   {
      int gN = diff.rows();
      int nPoly = diff.cols();

      if(l < 0)
      {
         throw Exception("Tried to compute Worland 1/r D r operator with l < 0");
      }

      if (nPoly < 1)
      {
         throw Exception("Operator matrix should have at least 1 column");
      }

      if (gN != igrid.size())
      {
         throw Exception("Operator matrix does not mach grid size");
      }

      internal::MHDFloat a = WorlandPolynomial::alpha(l);
      internal::MHDFloat b = WorlandPolynomial::beta(l);
      internal::MHDFloat a1 = WorlandPolynomial::alpha(l) + MHD_MP(1.0);
      internal::MHDFloat b1 = WorlandPolynomial::beta(l) + MHD_MP(1.0);
      internal::MHDFloat dl1 = internal::MHDFloat(l+1);

      // Make X grid in [-1, 1]
      internal::Array ixgrid = MHD_MP(2.0)*igrid.array()*igrid.array() - MHD_MP(1.0);

      // Storage for P_n^{(alpha,beta)} and dP_n{(alpha,beta)}
      internal::Matrix ipnab(gN,2);
      internal::Matrix idpnab(gN,2);
      idiff.resize(gN, nPoly);

      // Compute P_0
      WorlandPolynomial::W0l(ipnab.col(0), l-1, a, b, igrid, WorlandPolynomial::normWP0ab());
      ipnab.col(0) *= dl1;

      // Compute DP_0
      idpnab.col(0).setZero();

      // Compute l P
      idiff.col(0) = ipnab.col(0);

      if(nPoly > 1)
      {
         // Compute P_0
         ThreeTermRecurrence::P1(ipnab.col(1), a, b, ipnab.col(0), ixgrid, WorlandPolynomial::normWP1ab());

         // Compute DP_1
         WorlandPolynomial::W0l(idpnab.col(0), l+1, a1, b1, igrid, WorlandPolynomial::normWDP0ab());

         // Compute e P + 4r^2 DP
         idiff.col(1) = ipnab.col(1) + idpnab.col(0);
      }

      if(nPoly > 2)
      {
         // Increment P_n
         ThreeTermRecurrence::Pn(ipnab.col(0), 2, a, b, ipnab.col(1), ipnab.col(0), ixgrid, WorlandPolynomial::normWPnab());
         ipnab.col(0).swap(ipnab.col(1));

         // Compute DP_2
         ThreeTermRecurrence::P1(idpnab.col(1), a1, b1, idpnab.col(0), ixgrid, WorlandPolynomial::normWDP1ab());

         // Compute e P + 2(x+1) DP
         idiff.col(2) = ipnab.col(1) + idpnab.col(1);
      }

      for(int i = 3; i < nPoly; ++i)
      {
         // Increment P_n
         ThreeTermRecurrence::Pn(ipnab.col(0), i, a, b, ipnab.col(1), ipnab.col(0), ixgrid, WorlandPolynomial::normWPnab());
         ipnab.col(0).swap(ipnab.col(1));

         // Increment DP_n
         ThreeTermRecurrence::Pn(idpnab.col(0), i-1, a1, b1, idpnab.col(1), idpnab.col(0), ixgrid, WorlandPolynomial::normWDPnab());
         idpnab.col(0).swap(idpnab.col(1));

         // Compute e P + 2(x+1) DP
         idiff.col(i) = ipnab.col(1) + idpnab.col(1);
      }

      diff = Precision::cast(idiff);
   }

   void WorlandPolynomial::slaplWnl(Matrix& diff, internal::Matrix& idiff, const int l, const internal::Array& igrid)
   {
      int gN = diff.rows();
      int nPoly = diff.cols();

      if(l < 0)
      {
         throw Exception("Tried to compute Worland spherical laplacian with l < 0");
      }

      if (nPoly < 1)
      {
         throw Exception("Operator matrix should have at least 1 column");
      }

      if (gN != igrid.size())
      {
         throw Exception("Operator matrix does not mach grid size");
      }

      internal::MHDFloat a1 = WorlandPolynomial::alpha(l) + MHD_MP(1.0);
      internal::MHDFloat b1 = WorlandPolynomial::beta(l) + MHD_MP(1.0);
      internal::MHDFloat a2 = WorlandPolynomial::alpha(l) + MHD_MP(2.0);
      internal::MHDFloat b2 = WorlandPolynomial::beta(l) + MHD_MP(2.0);
      internal::MHDFloat dl = internal::MHDFloat(l);

      // Make X grid in [-1, 1]
      internal::Array ixgrid = MHD_MP(2.0)*igrid.array()*igrid.array() - MHD_MP(1.0);

      // Storage for P_n^{(alpha,beta)} and dP_n{(alpha,beta)}
      internal::Matrix idpnab(gN,2);
      internal::Matrix id2pnab(gN,2);
      idiff.resize(gN, nPoly);

      // Compute spherical laplacian P_0
      idiff.col(0).setZero();

      if(nPoly > 1)
      {
         // Compute DP_1
         WorlandPolynomial::W0l(idpnab.col(0), l, a1, b1, igrid, WorlandPolynomial::normWDP0ab());
         idpnab.col(0) *= (MHD_MP(2.0)*dl + MHD_MP(3.0)); 

         // Compute spherical laplacian P_1
         idiff.col(1) = idpnab.col(0);
      }

      if(nPoly > 2)
      {
         // Increment DP_2
         ThreeTermRecurrence::P1(idpnab.col(1), a1, b1, idpnab.col(0), ixgrid, WorlandPolynomial::normWDP1ab());

         // Compute D2P_2
         WorlandPolynomial::W0l(id2pnab.col(0), l+2, a2, b2, igrid, WorlandPolynomial::normWD2P0ab());

         // Compute e P + 2(x+1) DP
         idiff.col(2) = id2pnab.col(0) + idpnab.col(1);
      }

      if(nPoly > 3)
      {
         // Increment DP_n
         ThreeTermRecurrence::Pn(idpnab.col(0), 2, a1, b1, idpnab.col(1), idpnab.col(0), ixgrid, WorlandPolynomial::normWDPnab());
         idpnab.col(0).swap(idpnab.col(1));

         // Compute D2P_2
         ThreeTermRecurrence::P1(id2pnab.col(1), a2, b2, id2pnab.col(0), ixgrid, WorlandPolynomial::normWD2P1ab());

         // Compute e P + 2(x+1) DP
         idiff.col(3) = id2pnab.col(1) + idpnab.col(1);
      }

      for(int i = 4; i < nPoly; ++i)
      {
         // Increment DP_n
         ThreeTermRecurrence::Pn(idpnab.col(0), i-1, a1, b1, idpnab.col(1), idpnab.col(0), ixgrid, WorlandPolynomial::normWDPnab());
         idpnab.col(0).swap(idpnab.col(1));

         // Increment D2P_n
         ThreeTermRecurrence::Pn(id2pnab.col(0), i-2, a2, b2, id2pnab.col(1), id2pnab.col(0), ixgrid, WorlandPolynomial::normWD2Pnab());
         id2pnab.col(0).swap(id2pnab.col(1));

         // Compute e P + 2(x+1) DP
         idiff.col(i) = id2pnab.col(1) + idpnab.col(1);
      }

      diff = Precision::cast(idiff);
   }

   void WorlandPolynomial::dr_1drWnl(Matrix& diff, internal::Matrix& idiff, const int l, const internal::Array& igrid)
   {
      int gN = diff.rows();
      int nPoly = diff.cols();

      if(l < 0)
      {
         throw Exception("Tried to compute Worland D 1/r D r operator with l < 0");
      }

      if (nPoly < 1)
      {
         throw Exception("Operator matrix should have at least 1 column");
      }

      if (gN != igrid.size())
      {
         throw Exception("Operator matrix does not mach grid size");
      }

      internal::MHDFloat a = WorlandPolynomial::alpha(l);
      internal::MHDFloat b = WorlandPolynomial::beta(l);
      internal::MHDFloat a1 = WorlandPolynomial::alpha(l) + MHD_MP(1.0);
      internal::MHDFloat b1 = WorlandPolynomial::beta(l) + MHD_MP(1.0);
      internal::MHDFloat a2 = WorlandPolynomial::alpha(l) + MHD_MP(2.0);
      internal::MHDFloat b2 = WorlandPolynomial::beta(l) + MHD_MP(2.0);
      internal::MHDFloat dl = internal::MHDFloat(l);

      // Make X grid in [-1, 1]
      internal::Array ixgrid = MHD_MP(2.0)*igrid.array()*igrid.array() - MHD_MP(1.0);

      // Storage for P_n^{(alpha,beta)} and dP_n{(alpha,beta)}
      internal::Matrix ipnab(gN,2);
      internal::Matrix idpnab(gN,2);
      internal::Matrix id2pnab(gN,2);
      idiff.resize(gN, nPoly);

      // Compute P_0
      WorlandPolynomial::W0l(ipnab.col(0), l-2, a, b, igrid, WorlandPolynomial::normWP0ab());
      ipnab.col(0) *= (dl - MHD_MP(1.0))*(dl + MHD_MP(1.0)); 

      // Compute l P
      if(l == 1)
      {
         idiff.col(0).setZero();
      } else
      {
         idiff.col(0) = ipnab.col(0);
      }

      if(nPoly > 1)
      {
         // Compute P_0
         if(l != 1)
         {
            ThreeTermRecurrence::P1(ipnab.col(1), a, b, ipnab.col(0), ixgrid, WorlandPolynomial::normWP1ab());
         }

         // Compute DP_1
         WorlandPolynomial::W0l(idpnab.col(0), l, a1, b1, igrid, WorlandPolynomial::normWDP0ab());
         idpnab.col(0) *= MHD_MP(2.0)*(dl + MHD_MP(1.0)); 

         // Compute e P + 4r^2 DP
         if(l == 1)
         {
            idiff.col(1) = idpnab.col(0);
         } else
         {
            idiff.col(1) = ipnab.col(1) + idpnab.col(0);
         }
      }

      if(nPoly > 2)
      {
         if(l != 1)
         {
            // Increment P_n
            ThreeTermRecurrence::Pn(ipnab.col(0), 2, a, b, ipnab.col(1), ipnab.col(0), ixgrid, WorlandPolynomial::normWPnab());
            ipnab.col(0).swap(ipnab.col(1));
         }

         // Compute DP_1
         ThreeTermRecurrence::P1(idpnab.col(1), a1, b1, idpnab.col(0), ixgrid, WorlandPolynomial::normWDP1ab());

         // Compute D2P_0
         WorlandPolynomial::W0l(id2pnab.col(0), l+2, a2, b2, igrid, WorlandPolynomial::normWD2P0ab());

         // Compute e P + 2(x+1) DP
         if(l == 1)
         {
            idiff.col(2) = idpnab.col(1) + id2pnab.col(0);
         } else
         {
            idiff.col(2) = ipnab.col(1) + idpnab.col(1) + id2pnab.col(0);
         }
      }

      if(nPoly > 3)
      {
         if(l != 1)
         {
            // Increment P_3
            ThreeTermRecurrence::Pn(ipnab.col(0), 3, a, b, ipnab.col(1), ipnab.col(0), ixgrid, WorlandPolynomial::normWPnab());
            ipnab.col(0).swap(ipnab.col(1));
         }

         // Compute DP_2
         ThreeTermRecurrence::Pn(idpnab.col(0), 2, a1, b1, idpnab.col(1), idpnab.col(0), ixgrid, WorlandPolynomial::normWDPnab());
         idpnab.col(0).swap(idpnab.col(1));

         // Compute D2P_1
         ThreeTermRecurrence::P1(id2pnab.col(1), a2, b2, id2pnab.col(0), ixgrid, WorlandPolynomial::normWD2P1ab());

         // Compute e P + 2(x+1) DP
         if(l == 1)
         { 
            idiff.col(3) = idpnab.col(1) + id2pnab.col(1);
         } else
         {
            idiff.col(3) = ipnab.col(1) + idpnab.col(1) + id2pnab.col(1);
         }
      }

      for(int i = 4; i < nPoly; ++i)
      {
         if(l != 1)
         {
            // Increment P_n
            ThreeTermRecurrence::Pn(ipnab.col(0), i, a, b, ipnab.col(1), ipnab.col(0), ixgrid, WorlandPolynomial::normWPnab());
            ipnab.col(0).swap(ipnab.col(1));
         }

         // Increment DP_n
         ThreeTermRecurrence::Pn(idpnab.col(0), i-1, a1, b1, idpnab.col(1), idpnab.col(0), ixgrid, WorlandPolynomial::normWDPnab());
         idpnab.col(0).swap(idpnab.col(1));

         // Increment D2P_n
         ThreeTermRecurrence::Pn(id2pnab.col(0), i-2, a2, b2, id2pnab.col(1), id2pnab.col(0), ixgrid, WorlandPolynomial::normWD2Pnab());
         id2pnab.col(0).swap(id2pnab.col(1));

         // Compute e P + 2(x+1) DP
         if(l == 1)
         {
            idiff.col(i) = idpnab.col(1) + id2pnab.col(1);
         } else
         {
            idiff.col(i) = ipnab.col(1) + idpnab.col(1) + id2pnab.col(1);
         }
      }

      diff = Precision::cast(idiff);
   }

   void WorlandPolynomial::claplhWnl(Matrix& diff, internal::Matrix& idiff, const int l, const internal::Array& igrid)
   {
      int gN = diff.rows();
      int nPoly = diff.cols();

      if(l < 0)
      {
         throw Exception("Tried to compute Worland cylindrical laplacian with l < 0");
      }

      if (nPoly < 1)
      {
         throw Exception("Operator matrix should have at least 1 column");
      }

      if (gN != igrid.size())
      {
         throw Exception("Operator matrix does not mach grid size");
      }

      internal::MHDFloat a1 = WorlandPolynomial::alpha(l) + MHD_MP(1.0);
      internal::MHDFloat b1 = WorlandPolynomial::beta(l) + MHD_MP(1.0);
      internal::MHDFloat a2 = WorlandPolynomial::alpha(l) + MHD_MP(2.0);
      internal::MHDFloat b2 = WorlandPolynomial::beta(l) + MHD_MP(2.0);
      internal::MHDFloat dl = internal::MHDFloat(l);

      // Make X grid in [-1, 1]
      internal::Array ixgrid = MHD_MP(2.0)*igrid.array()*igrid.array() - MHD_MP(1.0);

      // Storage for P_n^{(alpha,beta)} and dP_n{(alpha,beta)}
      internal::Matrix idpnab(gN,2);
      internal::Matrix id2pnab(gN,2);
      idiff.resize(gN, nPoly);

      // Compute spherical laplacian P_0
      idiff.col(0).setZero();

      if(nPoly > 1)
      {
         // Compute DP_1
         WorlandPolynomial::W0l(idpnab.col(0), l, a1, b1, igrid, WorlandPolynomial::normWDP0ab());
         idpnab.col(0) *= MHD_MP(2.0)*(dl + MHD_MP(1.0)); 

         // Compute spherical laplacian P_1
         idiff.col(1) = idpnab.col(0);
      }

      if(nPoly > 2)
      {
         // Increment DP_2
         ThreeTermRecurrence::P1(idpnab.col(1), a1, b1, idpnab.col(0), ixgrid, WorlandPolynomial::normWDP1ab());

         // Compute D2P_2
         WorlandPolynomial::W0l(id2pnab.col(0), l+2, a2, b2, igrid, WorlandPolynomial::normWD2P0ab());

         // Compute e P + 2(x+1) DP
         idiff.col(2) = id2pnab.col(0) + idpnab.col(1);
      }

      if(nPoly > 3)
      {
         // Increment DP_n
         ThreeTermRecurrence::Pn(idpnab.col(0), 2, a1, b1, idpnab.col(1), idpnab.col(0), ixgrid, WorlandPolynomial::normWDPnab());
         idpnab.col(0).swap(idpnab.col(1));

         // Compute D2P_2
         ThreeTermRecurrence::P1(id2pnab.col(1), a2, b2, id2pnab.col(0), ixgrid, WorlandPolynomial::normWD2P1ab());

         // Compute e P + 2(x+1) DP
         idiff.col(3) = id2pnab.col(1) + idpnab.col(1);
      }

      for(int i = 4; i < nPoly; ++i)
      {
         // Increment DP_n
         ThreeTermRecurrence::Pn(idpnab.col(0), i-1, a1, b1, idpnab.col(1), idpnab.col(0), ixgrid, WorlandPolynomial::normWDPnab());
         idpnab.col(0).swap(idpnab.col(1));

         // Increment D2P_n
         ThreeTermRecurrence::Pn(id2pnab.col(0), i-2, a2, b2, id2pnab.col(1), id2pnab.col(0), ixgrid, WorlandPolynomial::normWD2Pnab());
         id2pnab.col(0).swap(id2pnab.col(1));

         // Compute e P + 2(x+1) DP
         idiff.col(i) = id2pnab.col(1) + idpnab.col(1);
      }

      diff = Precision::cast(idiff);
   }

   void WorlandPolynomial::r_1claplhWnl(Matrix& diff, internal::Matrix& idiff, const int l, const internal::Array& igrid)
   {
      int gN = diff.rows();
      int nPoly = diff.cols();

      if(l < 0)
      {
         throw Exception("Tried to compute Worland 1/r cylindrical laplacian with l < 0");
      }

      if (nPoly < 1)
      {
         throw Exception("Operator matrix should have at least 1 column");
      }

      if (gN != igrid.size())
      {
         throw Exception("Operator matrix does not mach grid size");
      }

      internal::MHDFloat a1 = WorlandPolynomial::alpha(l) + MHD_MP(1.0);
      internal::MHDFloat b1 = WorlandPolynomial::beta(l) + MHD_MP(1.0);
      internal::MHDFloat a2 = WorlandPolynomial::alpha(l) + MHD_MP(2.0);
      internal::MHDFloat b2 = WorlandPolynomial::beta(l) + MHD_MP(2.0);
      internal::MHDFloat dl = internal::MHDFloat(l);

      // Make X grid in [-1, 1]
      internal::Array ixgrid = MHD_MP(2.0)*igrid.array()*igrid.array() - MHD_MP(1.0);

      // Storage for P_n^{(alpha,beta)} and dP_n{(alpha,beta)}
      internal::Matrix idpnab(gN,2);
      internal::Matrix id2pnab(gN,2);
      idiff.resize(gN, nPoly);

      // Compute spherical laplacian P_0
      idiff.col(0).setZero();

      if(nPoly > 1)
      {
         // Compute DP_1
         WorlandPolynomial::W0l(idpnab.col(0), l-1, a1, b1, igrid, WorlandPolynomial::normWDP0ab());
         idpnab.col(0) *= MHD_MP(2.0)*(dl + MHD_MP(1.0)); 

         // Compute spherical laplacian P_1
         idiff.col(1) = idpnab.col(0);
      }

      if(nPoly > 2)
      {
         // Increment DP_2
         ThreeTermRecurrence::P1(idpnab.col(1), a1, b1, idpnab.col(0), ixgrid, WorlandPolynomial::normWDP1ab());

         // Compute D2P_2
         WorlandPolynomial::W0l(id2pnab.col(0), l+1, a2, b2, igrid, WorlandPolynomial::normWD2P0ab());

         // Compute e P + 2(x+1) DP
         idiff.col(2) = id2pnab.col(0) + idpnab.col(1);
      }

      if(nPoly > 3)
      {
         // Increment DP_n
         ThreeTermRecurrence::Pn(idpnab.col(0), 2, a1, b1, idpnab.col(1), idpnab.col(0), ixgrid, WorlandPolynomial::normWDPnab());
         idpnab.col(0).swap(idpnab.col(1));

         // Compute D2P_2
         ThreeTermRecurrence::P1(id2pnab.col(1), a2, b2, id2pnab.col(0), ixgrid, WorlandPolynomial::normWD2P1ab());

         // Compute e P + 2(x+1) DP
         idiff.col(3) = id2pnab.col(1) + idpnab.col(1);
      }

      for(int i = 4; i < nPoly; ++i)
      {
         // Increment DP_n
         ThreeTermRecurrence::Pn(idpnab.col(0), i-1, a1, b1, idpnab.col(1), idpnab.col(0), ixgrid, WorlandPolynomial::normWDPnab());
         idpnab.col(0).swap(idpnab.col(1));

         // Increment D2P_n
         ThreeTermRecurrence::Pn(id2pnab.col(0), i-2, a2, b2, id2pnab.col(1), id2pnab.col(0), ixgrid, WorlandPolynomial::normWD2Pnab());
         id2pnab.col(0).swap(id2pnab.col(1));

         // Compute e P + 2(x+1) DP
         idiff.col(i) = id2pnab.col(1) + idpnab.col(1);
      }

      diff = Precision::cast(idiff);
   }

   void WorlandPolynomial::dclaplhWnl(Matrix& diff, internal::Matrix& idiff, const int l, const internal::Array& igrid)
   {
      int gN = diff.rows();
      int nPoly = diff.cols();

      if(l < 0)
      {
         throw Exception("Tried to compute Worland derivative of cylindrical laplacian with l < 0");
      }

      if (nPoly < 1)
      {
         throw Exception("Operator matrix should have at least 1 column");
      }

      if (gN != igrid.size())
      {
         throw Exception("Operator matrix does not mach grid size");
      }

      internal::MHDFloat a1 = WorlandPolynomial::alpha(l) + MHD_MP(1.0);
      internal::MHDFloat b1 = WorlandPolynomial::beta(l) + MHD_MP(1.0);
      internal::MHDFloat a2 = WorlandPolynomial::alpha(l) + MHD_MP(2.0);
      internal::MHDFloat b2 = WorlandPolynomial::beta(l) + MHD_MP(2.0);
      internal::MHDFloat a3 = WorlandPolynomial::alpha(l) + MHD_MP(3.0);
      internal::MHDFloat b3 = WorlandPolynomial::beta(l) + MHD_MP(3.0);
      internal::MHDFloat dl = internal::MHDFloat(l);

      // Make X grid in [-1, 1]
      internal::Array ixgrid = MHD_MP(2.0)*igrid.array()*igrid.array() - MHD_MP(1.0);

      // Storage for P_n^{(alpha,beta)} and dP_n{(alpha,beta)}
      internal::Matrix idpnab(gN,2);
      internal::Matrix id2pnab(gN,2);
      internal::Matrix id3pnab(gN,2);
      idiff.resize(gN, nPoly);

      // Compute spherical laplacian P_0
      idiff.col(0).setZero();

      if(nPoly > 1)
      {
         // Compute DP_1
         WorlandPolynomial::W0l(idpnab.col(0), l-1, a1, b1, igrid, WorlandPolynomial::normWDP0ab());
         idpnab.col(0) *= MHD_MP(2.0)*dl*(dl + MHD_MP(1.0)); 

         // Compute spherical laplacian P_1
         idiff.col(1) = idpnab.col(0);
      }

      if(nPoly > 2)
      {
         // Increment DP_2
         ThreeTermRecurrence::P1(idpnab.col(1), a1, b1, idpnab.col(0), ixgrid, WorlandPolynomial::normWDP1ab());

         // Compute D2P_2
         WorlandPolynomial::W0l(id2pnab.col(0), l+1, a2, b2, igrid, WorlandPolynomial::normWD2P0ab());
         id2pnab.col(0) *= (MHD_MP(3.0)*dl + MHD_MP(4.0)); 

         // Compute e P + 2(x+1) DP
         idiff.col(2) = id2pnab.col(0) + idpnab.col(1);
      }

      if(nPoly > 3)
      {
         // Increment DP_n
         ThreeTermRecurrence::Pn(idpnab.col(0), 2, a1, b1, idpnab.col(1), idpnab.col(0), ixgrid, WorlandPolynomial::normWDPnab());
         idpnab.col(0).swap(idpnab.col(1));

         // Compute D2P_2
         ThreeTermRecurrence::P1(id2pnab.col(1), a2, b2, id2pnab.col(0), ixgrid, WorlandPolynomial::normWD2P1ab());

         // Compute D3P_2
         WorlandPolynomial::W0l(id3pnab.col(0), l+3, a3, b3, igrid, WorlandPolynomial::normWD3P0ab());

         // Compute e P + 2(x+1) DP
         idiff.col(3) = id3pnab.col(0) + id2pnab.col(1) + idpnab.col(1);
      }

      if(nPoly > 4)
      {
         // Increment DP_n
         ThreeTermRecurrence::Pn(idpnab.col(0), 3, a1, b1, idpnab.col(1), idpnab.col(0), ixgrid, WorlandPolynomial::normWDPnab());
         idpnab.col(0).swap(idpnab.col(1));

         // Compute D2P_2
         ThreeTermRecurrence::Pn(id2pnab.col(0), 2, a2, b2, id2pnab.col(1), id2pnab.col(0), ixgrid, WorlandPolynomial::normWD2Pnab());
         id2pnab.col(0).swap(id2pnab.col(1));

         // Compute D3P_2
         ThreeTermRecurrence::P1(id3pnab.col(1), a3, b3, id3pnab.col(0), ixgrid, WorlandPolynomial::normWD3P1ab());

         // Compute e P + 2(x+1) DP
         idiff.col(4) = id3pnab.col(1) + id2pnab.col(1) + idpnab.col(1);
      }

      for(int i = 5; i < nPoly; ++i)
      {
         // Increment DP_n
         ThreeTermRecurrence::Pn(idpnab.col(0), i-1, a1, b1, idpnab.col(1), idpnab.col(0), ixgrid, WorlandPolynomial::normWDPnab());
         idpnab.col(0).swap(idpnab.col(1));

         // Increment D2P_n
         ThreeTermRecurrence::Pn(id2pnab.col(0), i-2, a2, b2, id2pnab.col(1), id2pnab.col(0), ixgrid, WorlandPolynomial::normWD2Pnab());
         id2pnab.col(0).swap(id2pnab.col(1));

         // Increment D3P_n
         ThreeTermRecurrence::Pn(id3pnab.col(0), i-3, a3, b3, id3pnab.col(1), id3pnab.col(0), ixgrid, WorlandPolynomial::normWD3Pnab());
         id3pnab.col(0).swap(id3pnab.col(1));

         // Compute e P + 2(x+1) DP
         idiff.col(i) = id3pnab.col(1) + id2pnab.col(1) + idpnab.col(1);
      }

      diff = Precision::cast(idiff);
   }

   void WorlandPolynomial::r_1Wnl(Matrix& poly, internal::Matrix& ipoly, const int l, const internal::Array& igrid)
   {
      int gN = poly.rows();
      int nPoly = poly.cols();

      if(l < 0)
      {
         throw Exception("Tried to compute Worland 1/r operator with l < 0");
      }

      if (nPoly < 1)
      {
         throw Exception("Operator matrix should have at least 1 column");
      }

      if (gN != igrid.size())
      {
         throw Exception("Operator matrix does not mach grid size");
      }

      ipoly.resize(gN, nPoly);
      WorlandPolynomial::W0l(ipoly.col(0), l-1, WorlandPolynomial::alpha(l), WorlandPolynomial::beta(l), igrid, WorlandPolynomial::normWP0ab());

      // Make X grid in [-1, 1]
      internal::Array ixgrid = MHD_MP(2.0)*igrid.array()*igrid.array() - MHD_MP(1.0);

      if(nPoly > 1)
      {
         ThreeTermRecurrence::P1(ipoly.col(1), WorlandPolynomial::alpha(l), WorlandPolynomial::beta(l), ipoly.col(0), ixgrid, WorlandPolynomial::normWP1ab());
      }

      for(int i = 2; i < nPoly; ++i)
      {
         ThreeTermRecurrence::Pn(ipoly.col(i), i, WorlandPolynomial::alpha(l), WorlandPolynomial::beta(l), ipoly.col(i-1), ipoly.col(i-2), ixgrid, WorlandPolynomial::normWPnab());
      }

      poly = Precision::cast(ipoly);
   }

   void WorlandPolynomial::W0l(Eigen::Ref<internal::Matrix> iw0l, const int l, const internal::MHDFloat alpha, const internal::MHDFloat beta, const internal::Array& igrid, ThreeTermRecurrence::NormalizerAB norm)
   {
      internal::Array cs = norm(alpha, beta);

      if(l == 0)
      {
         iw0l.setConstant(MHD_MP(1.0));
      } else
      {
         iw0l.array() = igrid.array().pow(l);
      }

      iw0l.array() *= cs(0);
   }

   //
   // General polynomial normalizer
   //
   ThreeTermRecurrence::NormalizerNAB  WorlandPolynomial::normWPnab()
   {
      #ifdef QUICC_WORLAND_NORM_UNITY
         return &WorlandPolynomial::unitWPnab;
      #else 
         return &WorlandPolynomial::naturalWPnab;
      #endif //QUICC_WORLAND_NORM_UNITY
   }

   ThreeTermRecurrence::NormalizerAB  WorlandPolynomial::normWP1ab()
   {
      #ifdef QUICC_WORLAND_NORM_UNITY
         return &WorlandPolynomial::unitWP1ab;
      #else 
         return &WorlandPolynomial::naturalWP1ab;
      #endif //QUICC_WORLAND_NORM_UNITY
   }

   ThreeTermRecurrence::NormalizerAB  WorlandPolynomial::normWP0ab()
   {
      #ifdef QUICC_WORLAND_NORM_UNITY
         return &WorlandPolynomial::unitWP0ab;
      #else 
         return &WorlandPolynomial::naturalWP0ab;
      #endif //QUICC_WORLAND_NORM_UNITY
   }

   ThreeTermRecurrence::NormalizerNAB  WorlandPolynomial::normWDPnab()
   {
      #ifdef QUICC_WORLAND_NORM_UNITY
         return &WorlandPolynomial::unitWDPnab;
      #else 
         return &WorlandPolynomial::naturalWDPnab;
      #endif //QUICC_WORLAND_NORM_UNITY
   }

   ThreeTermRecurrence::NormalizerAB  WorlandPolynomial::normWDP1ab()
   {
      #ifdef QUICC_WORLAND_NORM_UNITY
         return &WorlandPolynomial::unitWDP1ab;
      #else 
         return &WorlandPolynomial::naturalWDP1ab;
      #endif //QUICC_WORLAND_NORM_UNITY
   }

   ThreeTermRecurrence::NormalizerAB  WorlandPolynomial::normWDP0ab()
   {
      #ifdef QUICC_WORLAND_NORM_UNITY
         return &WorlandPolynomial::unitWDP0ab;
      #else 
         return &WorlandPolynomial::naturalWDP0ab;
      #endif //QUICC_WORLAND_NORM_UNITY
   }

   ThreeTermRecurrence::NormalizerNAB  WorlandPolynomial::normWD2Pnab()
   {
      #ifdef QUICC_WORLAND_NORM_UNITY
         return &WorlandPolynomial::unitWD2Pnab;
      #else 
         return &WorlandPolynomial::naturalWD2Pnab;
      #endif //QUICC_WORLAND_NORM_UNITY
   }

   ThreeTermRecurrence::NormalizerAB  WorlandPolynomial::normWD2P1ab()
   {
      #ifdef QUICC_WORLAND_NORM_UNITY
         return &WorlandPolynomial::unitWD2P1ab;
      #else 
         return &WorlandPolynomial::naturalWD2P1ab;
      #endif //QUICC_WORLAND_NORM_UNITY
   }

   ThreeTermRecurrence::NormalizerAB  WorlandPolynomial::normWD2P0ab()
   {
      #ifdef QUICC_WORLAND_NORM_UNITY
         return &WorlandPolynomial::unitWD2P0ab;
      #else 
         return &WorlandPolynomial::naturalWD2P0ab;
      #endif //QUICC_WORLAND_NORM_UNITY
   }

   ThreeTermRecurrence::NormalizerNAB  WorlandPolynomial::normWD3Pnab()
   {
      #ifdef QUICC_WORLAND_NORM_UNITY
         return &WorlandPolynomial::unitWD3Pnab;
      #else 
         return &WorlandPolynomial::naturalWD3Pnab;
      #endif //QUICC_WORLAND_NORM_UNITY
   }

   ThreeTermRecurrence::NormalizerAB  WorlandPolynomial::normWD3P1ab()
   {
      #ifdef QUICC_WORLAND_NORM_UNITY
         return &WorlandPolynomial::unitWD3P1ab;
      #else 
         return &WorlandPolynomial::naturalWD3P1ab;
      #endif //QUICC_WORLAND_NORM_UNITY
   }

   ThreeTermRecurrence::NormalizerAB  WorlandPolynomial::normWD3P0ab()
   {
      #ifdef QUICC_WORLAND_NORM_UNITY
         return &WorlandPolynomial::unitWD3P0ab;
      #else 
         return &WorlandPolynomial::naturalWD3P0ab;
      #endif //QUICC_WORLAND_NORM_UNITY
   }

   //
   // Unit Worland polynomial normalizers
   //

   internal::Array WorlandPolynomial::unitWPnab(const internal::MHDFloat n, const internal::MHDFloat a, const internal::MHDFloat b)
   {
      internal::Array cs(4);

      cs(0) = -(MHD_MP(2.0)*n + a + b)/(MHD_MP(2.0)*n + a + b - MHD_MP(2.0))*precision::sqrt((n - MHD_MP(1.0))*(n + a - MHD_MP(1.0))*(n + b - MHD_MP(1.0))/(n + a + b));
      if (n > MHD_MP(2.0))
      {
         cs(0) *= precision::sqrt((n + a + b - MHD_MP(1.0))/(MHD_MP(2.0)*n + a + b - MHD_MP(3.0)));
      }
      cs(1) = ((MHD_MP(2.0)*n + a + b)/MHD_MP(2.0))*precision::sqrt((MHD_MP(2.0)*n + a + b - MHD_MP(1.0))/(n + a + b));
      cs(2) = (a*a - b*b)/(MHD_MP(2.0)*(MHD_MP(2.0)*n + a + b - MHD_MP(2.0)))*precision::sqrt((MHD_MP(2.0)*n + a + b - MHD_MP(1.0))/(n + a + b));
      cs(3) = precision::sqrt((MHD_MP(2.0)*n + a + b + MHD_MP(1.0))/(n*(n + a)*(n + b)));

      assert(!std::isnan(cs.sum()));

      return cs;
   }

   internal::Array WorlandPolynomial::unitWP1ab(const internal::MHDFloat a, const internal::MHDFloat b)
   {
      internal::Array cs(3);

      cs(0) = (a + b + MHD_MP(2.0));
      cs(1) = (a - b);
      cs(2) = precision::sqrt((a + b + MHD_MP(3.0))/(MHD_MP(4.0)*(a + MHD_MP(1.0))*(b + MHD_MP(1.0))));

      assert(!std::isnan(cs.sum()));

      return cs;
   }

   internal::Array WorlandPolynomial::unitWP0ab(const internal::MHDFloat a, const internal::MHDFloat b)
   {
      internal::Array cs(1);

      cs(0) = precision::sqrt(MHD_MP(2.0))*precision::exp(MHD_MP(0.5)*(precisiontr1::lgamma(a + b + MHD_MP(2.0)) - precisiontr1::lgamma(a + MHD_MP(1.0)) - precisiontr1::lgamma(b + MHD_MP(1.0))));

      assert(!std::isnan(cs.sum()));

      return cs;
   }

   //
   // Unit Worland first derivative normalizers
   //
   internal::Array WorlandPolynomial::unitWDPnab(const internal::MHDFloat n, const internal::MHDFloat a, const internal::MHDFloat b)
   {
      internal::Array cs(4);

      cs(0) = -((MHD_MP(2.0)*n + a + b)/(MHD_MP(2.0)*n + a + b - MHD_MP(2.0)))*precision::sqrt((n + a - MHD_MP(1.0))*(n + b - MHD_MP(1.0))*(n + a + b - MHD_MP(1.0))/(n*(n + a + b - MHD_MP(2.0))*(MHD_MP(2.0)*n + a + b - MHD_MP(3.0))));
      cs(1) = ((MHD_MP(2.0)*n + a + b)/(MHD_MP(2.0)*n))*precision::sqrt((MHD_MP(2.0)*n + a + b - MHD_MP(1.0))/(n + a + b - MHD_MP(1.0)));
      cs(2) = ((a*a - b*b)/(MHD_MP(2.0)*n*(MHD_MP(2.0)*n + a + b - MHD_MP(2.0))))*precision::sqrt((MHD_MP(2.0)*n + a + b - MHD_MP(1.0))/(n + a + b - MHD_MP(1.0)));
      cs(3) = precision::sqrt((n + MHD_MP(1.0))*(MHD_MP(2.0)*n + a + b + MHD_MP(1.0))/((n + a)*(n + b)));

      assert(!std::isnan(cs.sum()));

      return cs;
   }

   internal::Array WorlandPolynomial::unitWDP1ab(const internal::MHDFloat a, const internal::MHDFloat b)
   {
      internal::Array cs(3);

      cs(0) = (a + b + MHD_MP(2.0));
      cs(1) = (a - b);

      cs(2) = precision::sqrt((a + b + MHD_MP(1.0))*(a + b + MHD_MP(3.0))/(MHD_MP(2.0)*(a + MHD_MP(1.0))*(b + MHD_MP(1.0))*(a + b)));

      assert(!std::isnan(cs.sum()));

      return cs;
   }

   internal::Array WorlandPolynomial::unitWDP0ab(const internal::MHDFloat a, const internal::MHDFloat b)
   {
      internal::Array cs(1);

      cs(0) = MHD_MP(2.0)*precision::sqrt(MHD_MP(2.0)*(a+b))*precision::exp(MHD_MP(0.5)*(precisiontr1::lgamma(a + b + MHD_MP(2.0)) - precisiontr1::lgamma(a + MHD_MP(1.0)) - precisiontr1::lgamma(b + MHD_MP(1.0))));

      assert(!std::isnan(cs.sum()));

      return cs;
   }

   //
   // Unit Worland second derivative normalizers
   //
   internal::Array WorlandPolynomial::unitWD2Pnab(const internal::MHDFloat n, const internal::MHDFloat a, const internal::MHDFloat b)
   {
      internal::Array cs(4);

      cs(0) = -((MHD_MP(2.0)*n + a + b)*(n + a + b - MHD_MP(1.0))/((MHD_MP(2.0)*n + a + b - MHD_MP(2.0))))*precision::sqrt((n+1)*(n + a - MHD_MP(1.0))*(n + b - MHD_MP(1.0))/((n + a + b - MHD_MP(3.0))*(MHD_MP(2.0)*n + a + b - MHD_MP(3.0))));
      cs(1) = ((MHD_MP(2.0)*n + a + b)/(MHD_MP(2.0)))*precision::sqrt(MHD_MP(2.0)*n + a + b - MHD_MP(1.0));
      cs(2) = ((a*a - b*b)/(MHD_MP(2.0)*(MHD_MP(2.0)*n + a + b - MHD_MP(2.0))))*precision::sqrt(MHD_MP(2.0)*n + a + b - MHD_MP(1.0));
      cs(3) = precision::sqrt((n + MHD_MP(2.0))*(MHD_MP(2.0)*n + a + b + MHD_MP(1.0))/(n*n*(n + a)*(n + b)*(n + a + b - MHD_MP(2.0))));

      assert(!std::isnan(cs.sum()));

      return cs;
   }

   internal::Array WorlandPolynomial::unitWD2P1ab(const internal::MHDFloat a, const internal::MHDFloat b)
   {
      internal::Array cs(3);

      cs(0) = (a + b + MHD_MP(2.0));
      cs(1) = (a - b);
      cs(2) = (precision::sqrt(MHD_MP(3.0))/MHD_MP(2.0))*precision::sqrt((a + b + MHD_MP(1.0))*(a + b + MHD_MP(3.0))/((a + MHD_MP(1.0))*(b + MHD_MP(1.0))*(a + b - MHD_MP(1.0))));

      assert(!std::isnan(cs.sum()));

      return cs;
   }

   internal::Array WorlandPolynomial::unitWD2P0ab(const internal::MHDFloat a, const internal::MHDFloat b)
   {
      internal::Array cs(1);

      cs(0) = MHD_MP(8.0)*precision::sqrt((a + b)*(a + b - MHD_MP(1.0)))*precision::exp(MHD_MP(0.5)*(precisiontr1::lgamma(a + b + MHD_MP(2.0)) - precisiontr1::lgamma(a + MHD_MP(1.0)) - precisiontr1::lgamma(b + MHD_MP(1.0))));

      assert(!std::isnan(cs.sum()));

      return cs;
   }

   //
   // Unit Worland third derivative normalizers
   //
   internal::Array WorlandPolynomial::unitWD3Pnab(const internal::MHDFloat n, const internal::MHDFloat a, const internal::MHDFloat b)
   {
      internal::Array cs(4);

      cs(0) = -((MHD_MP(2.0)*n + a + b)*(n + a + b - MHD_MP(1.0))/((MHD_MP(2.0)*n + a + b - MHD_MP(2.0))))*precision::sqrt((n+2)*(n + a - MHD_MP(1.0))*(n + b - MHD_MP(1.0))/((n + a + b - MHD_MP(4.0))*(MHD_MP(2.0)*n + a + b - MHD_MP(3.0))*(MHD_MP(2.0)*n + a + b - MHD_MP(1.0))));
      cs(1) = ((MHD_MP(2.0)*n + a + b)/(MHD_MP(2.0)));
      cs(2) = ((a*a - b*b)/(MHD_MP(2.0)*(MHD_MP(2.0)*n + a + b - MHD_MP(2.0))));
      cs(3) = precision::sqrt((n + MHD_MP(3.0))*(MHD_MP(2.0)*n + a + b + MHD_MP(1.0))*(MHD_MP(2.0)*n + a + b - MHD_MP(1.0))/(n*n*(n + a)*(n + b)*(n + a + b - MHD_MP(3.0))));

      assert(!std::isnan(cs.sum()));

      return cs;
   }

   internal::Array WorlandPolynomial::unitWD3P1ab(const internal::MHDFloat a, const internal::MHDFloat b)
   {
      internal::Array cs(3);

      cs(0) = (a + b + MHD_MP(2.0));
      cs(1) = (a - b);
      cs(2) = precision::sqrt((a + b + MHD_MP(1.0))*(a + b + MHD_MP(3.0))/((a + MHD_MP(1.0))*(b + MHD_MP(1.0))*(a + b - MHD_MP(2.0))));

      assert(!std::isnan(cs.sum()));

      return cs;
   }

   internal::Array WorlandPolynomial::unitWD3P0ab(const internal::MHDFloat a, const internal::MHDFloat b)
   {
      internal::Array cs(1);

      cs(0) = MHD_MP(16.0)*precision::sqrt(MHD_MP(3.0)*(a + b)*(a + b - MHD_MP(1.0))*(a + b - MHD_MP(2.0)))*precision::exp(MHD_MP(0.5)*(precisiontr1::lgamma(a + b + MHD_MP(2.0)) - precisiontr1::lgamma(a + MHD_MP(1.0)) - precisiontr1::lgamma(b + MHD_MP(1.0))));

      assert(!std::isnan(cs.sum()));

      return cs;
   }

   //
   // Natural polynomial normalizer
   //
   internal::Array WorlandPolynomial::naturalWPnab(const internal::MHDFloat n, const internal::MHDFloat a, const internal::MHDFloat b)
   {
      internal::Array cs(4);

      cs(0) = -((n + a - MHD_MP(1.0))*(n + b - MHD_MP(1.0))*(MHD_MP(2.0)*n + a + b))/(n*(n + a + b)*(MHD_MP(2.0)*n + a + b - MHD_MP(2.0)));
      cs(1) = ((MHD_MP(2.0)*n + a + b - MHD_MP(1.0))*(MHD_MP(2.0)*n + a + b))/(MHD_MP(2.0)*n*(n + a + b));
      cs(2) = ((MHD_MP(2.0)*n + a + b - MHD_MP(1.0))*(a*a - b*b))/(MHD_MP(2.0)*n*(n + a + b)*(MHD_MP(2.0)*n + a + b - MHD_MP(2.0)));
      cs(3) = MHD_MP(1.0);

      assert(!std::isnan(cs.sum()));

      return cs;
   }

   internal::Array WorlandPolynomial::naturalWP1ab(const internal::MHDFloat a, const internal::MHDFloat b)
   {
      internal::Array cs(3);

      cs(0) = (a + b + MHD_MP(2.0));
      cs(1) = (a - b);
      cs(2) = MHD_MP(0.5);

      assert(!std::isnan(cs.sum()));

      return cs;
   }

   internal::Array WorlandPolynomial::naturalWP0ab(const internal::MHDFloat a, const internal::MHDFloat b)
   {
      internal::Array cs(1);

      cs(0) = MHD_MP(1.0);

      assert(!std::isnan(cs.sum()));

      return cs;
   }

   //
   // Natural first derivative normalizer
   //
   internal::Array WorlandPolynomial::naturalWDPnab(const internal::MHDFloat n, const internal::MHDFloat a, const internal::MHDFloat b)
   {
      internal::Array cs(4);

      cs(0) = -((n + a + b - MHD_MP(1.0))*(n + a - MHD_MP(1.0))*(n + b - MHD_MP(1.0))*(MHD_MP(2.0)*n + a + b))/(n*(n + a + b - MHD_MP(2.0))*(MHD_MP(2.0)*n + a + b - MHD_MP(2.0)));
      cs(1) = ((MHD_MP(2.0)*n + a + b - MHD_MP(1.0))*(MHD_MP(2.0)*n + a + b))/(MHD_MP(2.0)*n);
      cs(2) = ((MHD_MP(2.0)*n + a + b - MHD_MP(1.0))*(a*a - b*b))/(MHD_MP(2.0)*n*(MHD_MP(2.0)*n + a + b - MHD_MP(2.0)));
      cs(3) = MHD_MP(1.0)/(n + a + b - MHD_MP(1.0));

      assert(!std::isnan(cs.sum()));
      
      return cs;
   }

   internal::Array WorlandPolynomial::naturalWDP1ab(const internal::MHDFloat a, const internal::MHDFloat b)
   {
      internal::Array cs(3);

      cs(0) = (a + b + MHD_MP(2.0));
      cs(1) = (a - b);
      cs(2) = (a + b + MHD_MP(1.0))/(MHD_MP(2.0)*(a + b));

      assert(!std::isnan(cs.sum()));

      return cs;
   }

   internal::Array WorlandPolynomial::naturalWDP0ab(const internal::MHDFloat a, const internal::MHDFloat b)
   {
      internal::Array cs(1);

      cs(0) = MHD_MP(2.0)*(a + b);

      assert(!std::isnan(cs.sum()));

      return cs;
   }

   //
   // Natural second derivative normalizer
   //
   internal::Array WorlandPolynomial::naturalWD2Pnab(const internal::MHDFloat n, const internal::MHDFloat a, const internal::MHDFloat b)
   {
      internal::Array cs(4);

      cs(0) = -((n + a + b - MHD_MP(1.0))*(n + a - MHD_MP(1.0))*(n + b - MHD_MP(1.0))*(MHD_MP(2.0)*n + a + b))/(n*(n + a + b - MHD_MP(3.0))*(MHD_MP(2.0)*n + a + b - MHD_MP(2.0)));
      cs(1) = ((MHD_MP(2.0)*n + a + b - MHD_MP(1.0))*(MHD_MP(2.0)*n + a + b))/(MHD_MP(2.0)*n);
      cs(2) = ((MHD_MP(2.0)*n + a + b - MHD_MP(1.0))*(a*a - b*b))/(MHD_MP(2.0)*n*(MHD_MP(2.0)*n + a + b - MHD_MP(2.0)));
      cs(3) = MHD_MP(1.0)/(n + a + b - MHD_MP(2.0));

      assert(!std::isnan(cs.sum()));
      
      return cs;
   }

   internal::Array WorlandPolynomial::naturalWD2P1ab(const internal::MHDFloat a, const internal::MHDFloat b)
   {
      internal::Array cs(3);

      cs(0) = (a + b + MHD_MP(2.0));
      cs(1) = (a - b);
      cs(2) = (a + b + MHD_MP(1.0))/(MHD_MP(2.0)*(a + b - MHD_MP(1.0)));

      assert(!std::isnan(cs.sum()));

      return cs;
   }

   internal::Array WorlandPolynomial::naturalWD2P0ab(const internal::MHDFloat a, const internal::MHDFloat b)
   {
      internal::Array cs(1);

      cs(0) = MHD_MP(4.0)*(a + b)*(a + b - MHD_MP(1.0));

      assert(!std::isnan(cs.sum()));

      return cs;
   }

   //
   // Natural third derivative normalizer
   //
   internal::Array WorlandPolynomial::naturalWD3Pnab(const internal::MHDFloat n, const internal::MHDFloat a, const internal::MHDFloat b)
   {
      internal::Array cs(4);

      cs(0) = -((n + a + b - MHD_MP(1.0))*(n + a - MHD_MP(1.0))*(n + b - MHD_MP(1.0))*(MHD_MP(2.0)*n + a + b))/(n*(n + a + b - MHD_MP(4.0))*(MHD_MP(2.0)*n + a + b - MHD_MP(2.0)));
      cs(1) = ((MHD_MP(2.0)*n + a + b - MHD_MP(1.0))*(MHD_MP(2.0)*n + a + b))/(MHD_MP(2.0)*n);
      cs(2) = ((MHD_MP(2.0)*n + a + b - MHD_MP(1.0))*(a*a - b*b))/(MHD_MP(2.0)*n*(MHD_MP(2.0)*n + a + b - MHD_MP(2.0)));
      cs(3) = MHD_MP(1.0)/(n + a + b - MHD_MP(3.0));

      assert(!std::isnan(cs.sum()));
      
      return cs;
   }

   internal::Array WorlandPolynomial::naturalWD3P1ab(const internal::MHDFloat a, const internal::MHDFloat b)
   {
      internal::Array cs(3);

      cs(0) = (a + b + MHD_MP(2.0));
      cs(1) = (a - b);
      cs(2) = (a + b + MHD_MP(1.0))/(MHD_MP(2.0)*(a + b - MHD_MP(2.0)));

      assert(!std::isnan(cs.sum()));

      return cs;
   }

   internal::Array WorlandPolynomial::naturalWD3P0ab(const internal::MHDFloat a, const internal::MHDFloat b)
   {
      internal::Array cs(1);

      cs(0) = MHD_MP(8.0)*(a + b)*(a + b - MHD_MP(1.0))*(a + b - MHD_MP(2.0));

      assert(!std::isnan(cs.sum()));

      return cs;
   }

   WorlandPolynomial::WorlandPolynomial()
   {
   }

   WorlandPolynomial::~WorlandPolynomial()
   {
   }

}
}
