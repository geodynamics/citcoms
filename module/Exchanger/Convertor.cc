// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include "CartesianCoord.h"
#include "SIUnit.h"
#include "Convertor.h"


// definition of static variables
std::auto_ptr<SIUnit> Convertor::si;
std::auto_ptr<CartesianCoord> Convertor::cart;
bool inited = false;


// the singleton
Convertor& Convertor::instance()
{
    static Convertor* handle;

    if(!handle) {
	if(!inited) {
	    journal::firewall_t firewall("Convertor");
	    firewall << journal::loc(__HERE__)
		     << "instance() is called before init()" << journal::end;
	}

	journal::debug_t debug("Exchanger");
	debug << journal::loc(__HERE__)
	      << "creating Convertor singleton"
	      << journal::end;

	handle = new Convertor();
    }
    return *handle;
}


void Convertor::init(bool dimensional, bool transformational,
		     const All_variables* E)
{
    if(dimensional)
	si.reset(new SIUnit(E));

    if(transformational)
	cart.reset(new CartesianCoord());

    inited = true;
}


// internal representation ==> standard representation

void Convertor::coordinate(BoundedBox& bbox) const
{
    if(si.get()) si->coordinate(bbox);
    if(cart.get()) cart->coordinate(bbox);
}


void Convertor::coordinate(Array2D<double,DIM>& X) const
{
    if(si.get()) si->coordinate(X);
    if(cart.get()) cart->coordinate(X);
}


void Convertor::temperature(Array2D<double,1>& T) const
{
    if(si.get()) si->temperature(T);
}


void Convertor::time(double& t) const
{
    if(si.get()) si->time(t);
}


void Convertor::traction(Array2D<double,DIM>& F,
			 const Array2D<double,DIM>& X) const
{
    if(si.get()) si->traction(F);
    if(cart.get()) cart->vector(F, X);
}


void Convertor::velocity(Array2D<double,DIM>& V,
			 const Array2D<double,DIM>& X) const
{
    if(si.get()) si->velocity(V);
    if(cart.get()) cart->vector(V, X);
}


// standard representation ==> internal representation

void Convertor::xcoordinate(BoundedBox& bbox) const
{
    if(si.get()) si->xcoordinate(bbox);
    if(cart.get()) cart->xcoordinate(bbox);
}


void Convertor::xcoordinate(Array2D<double,DIM>& X) const
{
    if(si.get()) si->xcoordinate(X);
    if(cart.get()) cart->xcoordinate(X);
}


void Convertor::xtemperature(Array2D<double,1>& T) const
{
    if(si.get()) si->xtemperature(T);
}


void Convertor::xtime(double& t) const
{
    if(si.get()) si->xtime(t);
}


void Convertor::xtraction(Array2D<double,DIM>& F,
			  const Array2D<double,DIM>& X) const
{
    if(si.get()) si->xtraction(F);
    if(cart.get()) cart->xvector(F, X);
}


void Convertor::xvelocity(Array2D<double,DIM>& V,
			  const Array2D<double,DIM>& X) const
{
    if(si.get()) si->xvelocity(V);
    if(cart.get()) cart->xvector(V, X);
}


// version
// $Id: Convertor.cc,v 1.1 2004/01/07 21:54:00 tan2 Exp $

// End of file
