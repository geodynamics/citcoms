// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <stdexcept>
#include "global_defs.h"
#include "journal/journal.h"
#include "Transformational.h"

Transformational& Transformational::instance()
{
    static Transformational* handle;

    if(!handle) {
	if(!E) {
	    journal::firewall_t firewall("Transformational");
	    firewall << journal::loc(__HERE__)
		     << "All_variables* E == 0; Forget to call setE() first?"
		     << journal::end;
	    //throw std::error;
	}

	journal::debug_t debug("Exchanger");
	debug << journal::loc(__HERE__)
	      << "creating Transformational singleton"
	      << journal::end;

	handle = new Transformational(E);
    }
    return *handle;
}


void Transformational::setE(const All_variables* e)
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << journal::end;

    E = e;
}


Transformational::Transformational
{}

// Transformation from Spherical to Euclidean coordinate system
void Transformational::coordinate(BoundedBox& bbox) const
{
    for(int i=0; i<3; i++)
	bbox[i][DIM-1] *= length_factor;
}

void Transformational::coordinate(Array2D<double,DIM>& X) const
{
    std::vector<double> xt(DIM);
    for(int i=0; i<X.size(); ++i)
    {        
        for(int j=0; j<3 ; j++)
        {                
            xt[j]=X[j][i];
        }
        X[0][i]=xt[2]*sin(xt[0])*cos(xt[1]);
        X[1][i]=xt[2]*sin(xt[0])*sin(xt[1]);
        X[2][i]=xt[2]*cos(xt[0]);
        
    }    
}

void Transformational::velocity(Array2D<double,DIM>& V) const
{
    std::vector<double> tmp(DIM);
    std::vector<double> xt(DIM);
    for(int i=0; i<X.size(); ++i)
    {
        for(int j=0; j<3 ; j++)
        {                
            tmp[j]=V[j][i];
            xt[j] = X[j][i];
        }
        
        V[0][i]=cos(xt[0])*cos(xt[1])*tmp[0]-sin(xt[1])*tmp[1]+sin(xt[0])*cos(xt[1])*tmp[2];
        V[1][i]=cos(xt[0])*sin(xt[1])*tmp[0]+cos(xt[1])*tmp[1]+sin(xt[0])*sin(xt[1])*tmp[2];
        V[2][i]=-sin(xt[0])*tmp[0]+cos(xt[0])*tmp[2]; 
    }  
}
void Transformational::traction(Array2D<double,DIM>& F) const
{
    std::vector<double> tmp(DIM);
    std::vector<double> xt(DIM);
    for(int i=0; i<X.size(); ++i)
    {
        for(int j=0; j<3 ; j++)
        {                
            tmp[j]=F[j][i];
            xt[j] = X[j][i];
        }
        
        F[0][i]=cos(xt[0])*cos(xt[1])*tmp[0]-sin(xt[1])*tmp[1]+sin(xt[0])*cos(xt[1])*tmp[2];
        F[1][i]=cos(xt[0])*sin(xt[1])*tmp[0]+cos(xt[1])*tmp[1]+sin(xt[0])*sin(xt[1])*tmp[2];
        F[2][i]=-sin(xt[0])*tmp[0]+cos(xt[0])*tmp[2]; 
    }
    
}

// Transformation from Euclidean to Spherical coordinate system
void Transformational::xcoordinate(BoundedBox& bbox) const
{
    for(int i=0; i<3; i++)
	bbox[i][DIM-1] *= length_factor;
}

void Transformational::xcoordinate(Array2D<double,DIM>& X) const
{
    std::vector<double> xt(DIM);
    for(int i=0; i<X.size(); ++i)
    {        
        for(int j=0; j<3 ; j++)
        {                
            xt[j]=X[j][i];
        }
        X[2][i]=sqrt(xt[0]*xt[0]+xt[1]*xt[1]+xt[2]*xt[2]);
        X[1][i]=atan(xt[1]/xt[0]);
        X[0][i]=acos(xt[2]/X[2][i]);
        
    }
}

void Transformational::xvelocity(Array2D<double,DIM>& V) const
{
    std::vector<double> tmp(DIM);
    std::vector<double> xt(DIM);
    for(int i=0; i<X.size(); ++i)
    {
        for(int j=0; j<3 ; j++)
        {                
            tmp[j]=V[j][i];
            xt[j] = X[j][i];
        }
        
        V[0][i]=cos(xt[0])*cos(xt[1])*tmp[0]+cos(xt[0])*sin(xt[1])*tmp[1]-sin(xt[0])*tmp[2];
        V[1][i]=-sin(xt[0])*tmp[0]+cos(xt[1])*tmp[1];
        V[2][i]=sin(xt[0])*cos(xt[1])*tmp[0]+sin(xt[0])*sin(xt[1])*tmp[1]+cos(xt[0])*tmp[2]; 
    }   
}

void Transformational::xtraction(Array2D<double,DIM>& F) const
{
    std::vector<double> tmp(DIM);
    std::vector<double> xt(DIM);
    for(int i=0; i<X.size(); ++i)
    {
        for(int j=0; j<3 ; j++)
        {                
            tmp[j]=F[j][i];
            xt[j] = X[j][i];
        }
        F[0][i]=cos(xt[0])*cos(xt[1])*tmp[0]+cos(xt[0])*sin(xt[1])*tmp[1]-sin(xt[0])*tmp[2];
        F[1][i]=-sin(xt[0])*tmp[0]+cos(xt[1])*tmp[1];
        F[2][i]=sin(xt[0])*cos(xt[1])*tmp[0]+sin(xt[0])*sin(xt[1])*tmp[1]+cos(xt[0])*tmp[2];    
    }   
}
