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
#include "Dimensional.h"


// definition of static variable
const All_variables* Dimensional::E;


// the singleton
Dimensional& Dimensional::instance()
{
    static Dimensional* handle;

    if(!handle) {
	if(!E) {
	    journal::firewall_t firewall("Dimensional");
	    firewall << journal::loc(__HERE__)
		     << "All_variables* E == 0; Forget to call setE() first?"
		     << journal::end;
	    //throw std::error;
	}

	journal::debug_t debug("Exchanger");
	debug << journal::loc(__HERE__)
	      << "creating Dimensional singleton"
	      << journal::end;

	handle = new Dimensional(E);
    }
    return *handle;
}


void Dimensional::setE(const All_variables* e)
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << journal::end;

    E = e;
}


Dimensional::Dimensional(const All_variables* E) :
    length_factor(E->data.radius_km * 1000),
    velocity_factor(E->data.therm_diff / length_factor),
    temperature_factor(1),
    time_factor(length_factor / velocity_factor),
    traction_factor(E->data.ref_viscosity * E->data.therm_diff)
{}



void Dimensional::coordinate(BoundedBox& bbox) const
{
    for(int i=0; i<2; ++i)
	bbox[i][DIM-1] *= length_factor;
}


void Dimensional::coordinate(Array2D<double,DIM>& X) const
{
    for(int i=0; i<X.size(); ++i)
	X[DIM-1][i] *= length_factor;
}


void Dimensional::temperature(Array2D<double,1>& T) const
{
    for(int i=0; i<T.size(); ++i)
	T[0][i] *= temperature_factor;
}


void Dimensional::time(double& t) const
{
    t *= time_factor;
}


void Dimensional::traction(Array2D<double,DIM>& F) const
{
    for(int i=0; i<F.size(); ++i)
	for(int d=0; d<DIM; ++d)
	    F[d][i] *= traction_factor;
}


void Dimensional::velocity(Array2D<double,DIM>& V) const
{
    for(int i=0; i<V.size(); ++i)
	for(int d=0; d<DIM; ++d)
	    V[d][i] *= velocity_factor;
}


void Dimensional::xcoordinate(BoundedBox& bbox) const
{
    for(int i=0; i<2; ++i)
	bbox[i][DIM-1] /= length_factor;
}


void Dimensional::xcoordinate(Array2D<double,DIM>& X) const
{
    for(int i=0; i<X.size(); ++i)
	X[DIM-1][i] /= length_factor;
}


void Dimensional::xtemperature(Array2D<double,1>& T) const
{
    for(int i=0; i<T.size(); ++i)
	T[0][i] /= temperature_factor;
}


void Dimensional::xtime(double& t) const
{
    t /= time_factor;
}


void Dimensional::xtraction(Array2D<double,DIM>& F) const
{
    for(int i=0; i<F.size(); ++i)
	for(int d=0; d<DIM; ++d)
	    F[d][i] /= traction_factor;
}


void Dimensional::xvelocity(Array2D<double,DIM>& V) const
{
    for(int i=0; i<V.size(); ++i)
	for(int d=0; d<DIM; ++d)
	    V[d][i] /= velocity_factor;
}


// version
// $Id: SIUnit.cc,v 1.1 2003/12/30 21:44:32 tan2 Exp $

// End of file
