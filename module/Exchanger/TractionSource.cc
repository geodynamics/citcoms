// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include "journal/journal.h"
#include "BoundedBox.h"
#include "BoundedMesh.h"
#include "TractionInterpolator.h"
#include "TractionSource.h"


TractionSource::TractionSource(MPI_Comm comm, int sink,
			       BoundedMesh& mesh, const All_variables* e,
			       const BoundedBox& mybbox) :
    AbstractSource(comm, sink, mesh, mybbox),
    E(e)
{
    init(mesh, mybbox);
}


TractionSource::~TractionSource()
{}


void TractionSource::interpolateTraction(Array2D<double,DIM>& F) const
{
    if(size())
	interp->interpolateTraction(F);
}


void TractionSource::domain_cutout()
{
    (dynamic_cast<TractionInterpolator*>(interp))->domain_cutout();
}


// private functions

void TractionSource::createInterpolator(const BoundedMesh& mesh)
{
    interp = new TractionInterpolator(mesh, E, meshNode_);
}


// version
// $Id: TractionSource.cc,v 1.5 2004/02/25 23:07:35 tan2 Exp $

// End of file
