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
#include "SIUnit.h"


SIUnit::SIUnit(const All_variables* E) :
    length_factor(E->data.radius_km * 1000),
    velocity_factor(E->data.therm_diff / length_factor),
    temperature_factor(1),
    time_factor(length_factor / velocity_factor),
    traction_factor(E->data.ref_viscosity * E->data.therm_diff)
{}



void SIUnit::coordinate(BoundedBox& bbox) const
{
    for(int i=0; i<2; ++i)
	bbox[i][DIM-1] *= length_factor;
}


void SIUnit::coordinate(Array2D<double,DIM>& X) const
{
    for(int i=0; i<X.size(); ++i)
	X[DIM-1][i] *= length_factor;
}


void SIUnit::temperature(Array2D<double,1>& T) const
{
    for(int i=0; i<T.size(); ++i)
	T[0][i] *= temperature_factor;
}


void SIUnit::time(double& t) const
{
    t *= time_factor;
}


void SIUnit::traction(Array2D<double,DIM>& F) const
{
    for(int i=0; i<F.size(); ++i)
	for(int d=0; d<DIM; ++d)
	    F[d][i] *= traction_factor;
}


void SIUnit::velocity(Array2D<double,DIM>& V) const
{
    for(int i=0; i<V.size(); ++i)
	for(int d=0; d<DIM; ++d)
	    V[d][i] *= velocity_factor;
}


void SIUnit::xcoordinate(BoundedBox& bbox) const
{
    for(int i=0; i<2; ++i)
	bbox[i][DIM-1] /= length_factor;
}


void SIUnit::xcoordinate(Array2D<double,DIM>& X) const
{
    for(int i=0; i<X.size(); ++i)
	X[DIM-1][i] /= length_factor;
}


void SIUnit::xtemperature(Array2D<double,1>& T) const
{
    for(int i=0; i<T.size(); ++i)
	T[0][i] /= temperature_factor;
}


void SIUnit::xtime(double& t) const
{
    t /= time_factor;
}


void SIUnit::xtraction(Array2D<double,DIM>& F) const
{
    for(int i=0; i<F.size(); ++i)
	for(int d=0; d<DIM; ++d)
	    F[d][i] /= traction_factor;
}


void SIUnit::xvelocity(Array2D<double,DIM>& V) const
{
    for(int i=0; i<V.size(); ++i)
	for(int d=0; d<DIM; ++d)
	    V[d][i] /= velocity_factor;
}


// version
// $Id: SIUnit.cc,v 1.2 2004/01/07 21:54:00 tan2 Exp $

// End of file
