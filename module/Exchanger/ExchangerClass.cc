// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <algorithm>
#include "Array2D.h"
#include "Boundary.h"
#include "global_defs.h"
#include "journal/journal.h"
#include "ExchangerClass.h"


Exchanger::Exchanger(const MPI_Comm communicator,
		     const MPI_Comm icomm,
		     const int leaderRank,
		     const int remote,
		     const All_variables *e):
    comm(communicator),
    intercomm(icomm),
    leader(leaderRank),
    remoteLeader(remote),
    E(e),
    boundary(NULL)
{
    MPI_Comm_rank(comm, const_cast<int*>(&rank));
    fge_t = cge_t = 0;
}


Exchanger::~Exchanger() {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in Exchanger::~Exchanger" << journal::end;
}


void Exchanger::initTemperature() {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in Exchanger::initTemperature" << journal::end;

    journal::debug_t debugInitT("initTemperature");
    debugInitT << journal::loc(__HERE__);

    // put a hot blob in the center of fine grid mesh and T=0 elsewhere

    // center of fine grid mesh
    double theta_center = 0.5 * (boundary->theta_max() + boundary->theta_min());
    double fi_center = 0.5 * (boundary->fi_max() + boundary->fi_min());
    double r_center = 0.5 * (boundary->ro() + boundary->ri());

    double x_center = r_center * sin(fi_center) * cos(theta_center);
    double y_center = r_center * sin(fi_center) * sin(theta_center);
    double z_center = r_center * cos(fi_center);

    // radius of the blob is one third of the smallest dimension
    double d = std::min(std::min(boundary->theta_max() - boundary->theta_min(),
				 boundary->fi_max() - boundary->fi_min()),
			boundary->ro() - boundary->ri()) / 3;

    // compute temperature field according to nodal coordinate
    for(int m=1;m<=E->sphere.caps_per_proc;m++)
        for(int k=1;k<=E->lmesh.noy;k++)
	    for(int j=1;j<=E->lmesh.nox;j++)
		for(int i=1;i<=E->lmesh.noz;i++)  {
		    int node = i + (j-1)*E->lmesh.noz
			     + (k-1)*E->lmesh.noz*E->lmesh.nox;

 		    double theta = E->sx[m][1][node];
		    double fi = E->sx[m][2][node];
		    double r = E->sx[m][3][node];

		    double x = r * sin(fi) * cos(theta);
		    double y = r * sin(fi) * sin(theta);
		    double z = r * cos(fi);

		    double distance = sqrt((x - x_center)*(x - x_center) +
					   (y - y_center)*(y - y_center) +
					   (z - z_center)*(z - z_center));

		    if (distance <= d)
			E->T[m][node] = 0.5 + 0.5*cos(distance/d * M_PI);
		    else
			E->T[m][node] = 0;

		    if (rank == leader) {
			debugInitT << "(theta,fi,r,T) = "
				   << theta << "  "
				   << fi << "  "
				   << r << "  "
				   << E->T[m][node] << journal::newline;
		    }
		}
    debugInitT << journal::end;
}


void Exchanger::sendTemperature() {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in Exchanger::sendTemperature"
	  << "  rank = " << rank
	  << "  receiver = "<< remoteLeader
	  << journal::end;

    if(rank == leader) {
	outgoingT.send(intercomm, remoteLeader);
    }
}


void Exchanger::receiveTemperature() {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in Exchanger::receiveTemperature"
	  << "  rank = " << rank
	  << "  receiver = "<< remoteLeader
	  << journal::end;

    if(rank == leader) {
	incomingT.receive(intercomm, remoteLeader);
    }
}


void Exchanger::sendVelocities() {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in Exchanger::sendVelocities" << journal::end;

    if(rank == leader) {
	outgoingV.send(intercomm, remoteLeader);
    }
}


void Exchanger::receiveVelocities() {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in Exchanger::receiveVelocities" << journal::end;

    if(rank == leader) {
	// store previously received V
	incomingV.swap(old_incomingV);

	incomingV.receive(intercomm, remoteLeader);
	//incomingV.print("incomingV");
    }
}


void Exchanger::storeTimestep(const double fge_time, const double cge_time) {
    fge_t = fge_time;
    cge_t = cge_time;
}


double Exchanger::exchangeTimestep(const double dt) const {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in Exchanger::exchangeTimestep"
	  << "  rank = " << rank
	  << "  receiver = "<< remoteLeader << journal::end;
    return exchangeDouble(dt, 1);
}


int Exchanger::exchangeSignal(const int sent) const {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in Exchanger::exchangeSignal" << journal::end;
    return exchangeInt(sent, 1);
}


// helper functions



double Exchanger::exchangeDouble(const double &sent, const int len) const {
    double received;
    if (rank == leader) {
	const int tag = 350;
	MPI_Status status;

	MPI_Sendrecv((void*)&sent, len, MPI_DOUBLE,
		     remoteLeader, tag,
		     &received, len, MPI_DOUBLE,
		     remoteLeader, tag,
		     intercomm, &status);
    }

    MPI_Bcast(&received, 1, MPI_DOUBLE, leader, comm);
    return received;
}


float Exchanger::exchangeFloat(const float &sent, const int len) const {
    float received;
    if (rank == leader) {
	const int tag = 351;
	MPI_Status status;

	MPI_Sendrecv((void*)&sent, len, MPI_FLOAT,
		     remoteLeader, tag,
		     &received, len, MPI_FLOAT,
		     remoteLeader, tag,
		     intercomm, &status);
    }

    MPI_Bcast(&received, 1, MPI_FLOAT, leader, comm);
    return received;
}


int Exchanger::exchangeInt(const int &sent, const int len) const {
    int received;
    if (rank == leader) {
	const int tag = 352;
	MPI_Status status;

	MPI_Sendrecv((void*)&sent, len, MPI_INT,
		     remoteLeader, tag,
		     &received, len, MPI_INT,
		     remoteLeader, tag,
		     intercomm, &status);
    }

    MPI_Bcast(&received, 1, MPI_INT, leader, comm);
    return received;
}


// version
// $Id: ExchangerClass.cc,v 1.39 2003/10/30 23:36:07 tan2 Exp $

// End of file

