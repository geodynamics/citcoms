// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include "global_defs.h"
#include "Boundary.h"
#include "Sink.h"
#include "Source.h"
#include "BoundaryCondition.h"
#include "dimensionalization.h"

double dimensional_len,dimensional_vel,dimensional_traction,dimesional_temp;

extern "C" {
    void check_bc_consistency(const All_variables *E);
    void construct_id(const All_variables *E);
    void temperatures_conform_bcs(All_variables* E);
}



//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////


BoundaryConditionSink::BoundaryConditionSink(const MPI_Comm& c,
					     const Boundary& b,
					     const Sink& s,
					     All_variables* e) :
    comm(c),
    E(e),
    boundary(b),
    sink(s),
    awnormal(comm, boundary, sink, E),
    vbc(sink.size()),
    old_vbc(sink.size()),
    tbc(sink.size()),
    old_tbc(sink.size())
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in BoundaryConditionSink::c'tor"
	  << " sink.size = " << sink.size()
	  << journal::end;

    fge_t = cge_t = 0;

    setVBCFlag();
    setTBCFlag();
}


void BoundaryConditionSink::recvTandV()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in BoundaryConditionSink::recvTandV" << journal::end;

    // store bc from previous timestep
    tbc.swap(old_tbc);
    vbc.swap(old_vbc);

    sink.recvArray2D(tbc, vbc);

// TODO : non-dimensionalizing temeperature

    for(int i=0; i<sink.size(); i++) {
	for(int d=0; d<DIM; d++)vbc[d][i]/=dimensional_vel;
    }

    tbc.print("TBC");
    vbc.print("VBC");

    imposeConstraint();
    vbc.print("VBC_i");
}


void BoundaryConditionSink::imposeBC()
{
    imposeTBC();
    imposeVBC();
}


void BoundaryConditionSink::storeTimestep(double fge_time, double cge_time)
{
    fge_t = fge_time;
    cge_t = cge_time;
}


// private functions

void BoundaryConditionSink::setVBCFlag()
{
    // Because CitcomS' default side BC is reflecting,
    // here we should change to velocity BC.
    const int m = 1;
    for(int i=0; i<boundary.size(); i++) {
	int n = boundary.nodeID(i);
	E->node[m][n] = E->node[m][n] | VBX;
	E->node[m][n] = E->node[m][n] | VBY;
	E->node[m][n] = E->node[m][n] | VBZ;
	E->node[m][n] = E->node[m][n] & (~SBX);
	E->node[m][n] = E->node[m][n] & (~SBY);
	E->node[m][n] = E->node[m][n] & (~SBZ);
    }

    check_bc_consistency(E);
    // reconstruct ID array to reflect changes in VBC
    construct_id(E);
}


void BoundaryConditionSink::setTBCFlag()
{

    const int m = 1;
    for(int i=0; i<boundary.size(); i++) {
	int n = boundary.nodeID(i);
	E->node[m][n] = E->node[m][n] | TBX;
	E->node[m][n] = E->node[m][n] | TBY;
	E->node[m][n] = E->node[m][n] | TBZ;
	E->node[m][n] = E->node[m][n] & (~FBX);
	E->node[m][n] = E->node[m][n] & (~FBY);
	E->node[m][n] = E->node[m][n] & (~FBZ);
    }

    check_bc_consistency(E);

}


void BoundaryConditionSink::imposeConstraint()
{
    awnormal.imposeConstraint(vbc, comm, sink);
}


void BoundaryConditionSink::imposeTBC()
{

    journal::debug_t debugBC("imposeTBC");
    debugBC << journal::loc(__HERE__);

    double N1, N2;

    if(cge_t == 0) {
        N1 = 0.0;
        N2 = 1.0;
    } else {
        N1 = (cge_t - fge_t) / cge_t;
        N2 = fge_t / cge_t;
    }

    const int m = 1;
    for(int i=0; i<sink.size(); i++) {
	int n = boundary.nodeID(sink.meshNode(i));
	for(int d=0; d<DIM; d++)
	    E->sphere.cap[m].TB[d+1][n] = N1 * old_tbc[0][i]
		                        + N2 * tbc[0][i];

	debugBC << E->sphere.cap[m].TB[1][n] << " "
		<< E->sphere.cap[m].TB[2][n] << " "
		<< E->sphere.cap[m].TB[3][n] << journal::newline;

    }
    debugBC << journal::end;
    temperatures_conform_bcs(E);
}


void BoundaryConditionSink::imposeVBC()
{
    journal::debug_t debugBC("imposeVBC");
    debugBC << journal::loc(__HERE__);

    double N1, N2;

    if(cge_t == 0) {
        N1 = 0.0;
        N2 = 1.0;
    } else {
        N1 = (cge_t - fge_t) / cge_t;
        N2 = fge_t / cge_t;
    }

    const int m = 1;
    for(int i=0; i<sink.size(); i++) {
	int n = boundary.nodeID(sink.meshNode(i));
	for(int d=0; d<DIM; d++)
	    E->sphere.cap[m].VB[d+1][n] = N1 * old_vbc[d][i]
		                        + N2 * vbc[d][i];
	debugBC << E->sphere.cap[m].VB[1][n] << " "
		<< E->sphere.cap[m].VB[2][n] << " "
		<< E->sphere.cap[m].VB[3][n] << journal::newline;
    }
    debugBC << journal::end;
}


//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////


BoundaryConditionSource::BoundaryConditionSource(const Source& s,
						 All_variables* e) :
    E(e),
    source(s),
    vbc(source.size()),
    tbc(source.size()),
    fbc(source.size())
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in BoundaryConditionSource::c'tor"
	  << " source.size = " << source.size()
	  << journal::end;
}


void BoundaryConditionSource::sendTandV()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in BoundaryConditionSource::sendTandV" << journal::end;

    source.interpolateT(tbc, E);
    //tbc.print("TBC");
    source.interpolateV(vbc, E);
    //vbc.print("VBC");

    for(int i=0; i<source.size(); i++) {
	for(int d=0; d<DIM; d++)vbc[d][i]*=dimensional_vel;
    }

// TODO dimensionalize temperature

    source.sendArray2D(tbc, vbc);
}

void BoundaryConditionSource::sendTraction()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in BoundaryConditionSource::sendTraction" << journal::end;

    //source.interpolateF(fbc, E);
    //fbc.print("FBC");

    //source.sendArray2D(fbc);
}

void BoundaryConditionSource::domain_cutout()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in BoundaryConditionSource::domain_cutout" << journal::end;

    //source.domain_cutout(E);

}
// version
// $Id: BoundaryCondition.cc,v 1.10 2003/12/30 21:46:48 tan2 Exp $

// End of file
