// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <vector>
#include "journal/journal.h"
#include "BoundedMesh.h"
#include "CartesianCoord.h"


// Transformation from Spherical to Euclidean coordinate system

void CartesianCoord::coordinate(BoundedBox& bbox) const
{
    //for(int i=0; i<2; i++)
    //bbox[i][DIM-1] *= length_factor;
}


void CartesianCoord::coordinate(Array2D<double,DIM>& X) const
{
    std::vector<double> xt(DIM);

    for(int i=0; i<X.size(); ++i) {

        for(int j=0; j<DIM; j++)
            xt[j] = X[j][i];

        X[0][i] = xt[2] * sin(xt[0]) * cos(xt[1]);
        X[1][i] = xt[2] * sin(xt[0]) * sin(xt[1]);
        X[2][i] = xt[2] * cos(xt[0]);
    }
}


void CartesianCoord::vector(Array2D<double,DIM>& V,
			    const Array2D<double,DIM>& X) const
{
    std::vector<double> vt(DIM);
    std::vector<double> xt(DIM);

    // sanity check
    if(V.size() != X.size()) {
	journal::firewall_t firewall("Exchanger");
	firewall << journal::loc(__HERE__)
		 << "size of vectors mismatch" << journal::end;
    }

    for(int i=0; i<V.size(); ++i) {

        for(int j=0; j<DIM; j++) {
            vt[j] = V[j][i];
            xt[j] = X[j][i];
        }

        V[0][i] = cos(xt[0]) * cos(xt[1]) * vt[0]
	        - sin(xt[1]) * vt[1]
	        + sin(xt[0]) * cos(xt[1]) * vt[2];
        V[1][i] = cos(xt[0]) * sin(xt[1]) * vt[0]
	        + cos(xt[1]) * vt[1]
	        + sin(xt[0]) * sin(xt[1]) * vt[2];
        V[2][i] = -sin(xt[0]) * vt[0]
	        + cos(xt[0]) * vt[2];
    }
}


// Transformation from Euclidean to Spherical coordinate system

void CartesianCoord::xcoordinate(BoundedBox& bbox) const
{
    //for(int i=0; i<2; i++)
    //bbox[i][DIM-1] *= length_factor;
}


void CartesianCoord::xcoordinate(Array2D<double,DIM>& X) const
{
    std::vector<double> xt(DIM);

    for(int i=0; i<X.size(); ++i) {

        for(int j=0; j<DIM; j++)
            xt[j] = X[j][i];

        X[2][i] = std::sqrt(xt[0]*xt[0] + xt[1]*xt[1] + xt[2]*xt[2]);
        X[1][i] = std::atan(xt[1] / xt[0]);
        X[0][i] = std::acos(xt[2] / X[2][i]);
    }
}


void CartesianCoord::xvector(Array2D<double,DIM>& V,
			     const Array2D<double,DIM>& X) const
{
    std::vector<double> vt(DIM);
    std::vector<double> xt(DIM);

    // sanity check
    if(V.size() != X.size()) {
	journal::firewall_t firewall("Exchanger");
	firewall << journal::loc(__HERE__)
		 << "size of vectors mismatch" << journal::end;
    }

    for(int i=0; i<V.size(); ++i) {

        for(int j=0; j<DIM; j++) {
            vt[j] = V[j][i];
            xt[j] = X[j][i];
        }

        V[0][i] = cos(xt[0]) * cos(xt[1]) * vt[0]
	        + cos(xt[0]) * sin(xt[1]) * vt[1]
	        - sin(xt[0]) * vt[2];
        V[1][i] = -sin(xt[0]) * vt[0]
	        + cos(xt[1]) * vt[1];
        V[2][i] = sin(xt[0]) * cos(xt[1]) * vt[0]
	        + sin(xt[0]) * sin(xt[1]) * vt[1]
	        + cos(xt[0]) * vt[2];
    }
}


// version
// $Id: CartesianCoord.cc,v 1.2 2004/01/07 21:54:00 tan2 Exp $

// End of file
