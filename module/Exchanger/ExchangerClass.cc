// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <portinfo>
#include <iostream>
#include <fstream>
#include <cmath>
#include "Array2D.h"
#include "Array2D.cc"
#include "Boundary.h"
#include "global_defs.h"
#include "ExchangerClass.h"

extern "C" {
    void check_bc_consistency(const All_variables *E);
    void construct_id(const All_variables *E);
}

Exchanger::Exchanger(const MPI_Comm communicator,
		     const MPI_Comm icomm,
		     const int leaderRank,
		     const int local,
		     const int remote,
		     const All_variables *e):
    comm(communicator),
    intercomm(icomm),
    leader(leaderRank),
    localLeader(local),
    remoteLeader(remote),
    E(e),
    boundary(NULL) {

    MPI_Comm_rank(comm, const_cast<int*>(&rank));
    fge_t = cge_t = 0;
}


Exchanger::~Exchanger() {
    std::cout << "in Exchanger::~Exchanger" << std::endl;
}


void Exchanger::createDataArrays() {
    std::cout << "in Exchanger::createDataArrays" << std::endl;

    int size = boundary->size;

    outgoingT = Temper(new Array2D<1>(size));
    incomingT = Temper(new Array2D<1>(size));

    incomingV = Velo(new Array2D<3>(size));
    old_incomingV = Velo(new Array2D<3>(size));
}


void Exchanger::deleteDataArrays() {
    std::cout << "in Exchanger::deleteDataArrays" << std::endl;
}


void Exchanger::initTemperature() {
    std::cout << "in Exchanger::initTemperature" << std::endl;
    // put a hot blob in the center of fine grid mesh and T=0 elsewhere

    // center of fine grid mesh
    double theta_center = 0.5 * (boundary->theta_max + boundary->theta_min);
    double fi_center = 0.5 * (boundary->fi_max + boundary->fi_min);
    double r_center = 0.5 * (boundary->ro + boundary->ri);

    double x_center = r_center * sin(fi_center) * cos(theta_center);
    double y_center = r_center * sin(fi_center) * sin(theta_center);
    double z_center = r_center * cos(fi_center);

    // radius of the blob is one third of the smallest dimension
    double d = min(min(boundary->theta_max - boundary->theta_min,
		       boundary->fi_max - boundary->fi_min),
		   boundary->ro - boundary->ri) / 3;

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

// 		    if (rank == leader) {
// 			std::cout << "(theta,fi,r,T) = "
// 				  << theta << "  "
// 				  << fi << "  "
// 				  << r << "  "
// 				  << E->T[m][node] << std::endl;
// 		    }
		}
}


void Exchanger::sendTemperature() {
    std::cout << "in Exchanger::sendTemperature"
	      << "  rank = " << rank
	      << "  leader = "<< localLeader
	      << "  receiver = "<< remoteLeader
	      << std::endl;

    if(rank == leader) {
	outgoingT->send(intercomm, remoteLeader);
    }
}


void Exchanger::receiveTemperature() {
    std::cout << "in Exchanger::receiveTemperature"
	      << "  rank = " << rank
	      << "  leader = "<< localLeader
	      << "  receiver = "<< remoteLeader
	      << std::endl;

    if(rank == leader) {
	incomingT->receive(intercomm, remoteLeader);
    }
}


void Exchanger::sendVelocities() {
    std::cout << "in Exchanger::sendVelocities" << std::endl;

    if(rank == leader) {
	outgoingV->send(intercomm, remoteLeader);
    }
}


void Exchanger::receiveVelocities() {
    std::cout << "in Exchanger::receiveVelocities" << std::endl;

    if(rank == leader) {
	// store previously received V
	std::swap(incomingV, old_incomingV);

	incomingV->receive(intercomm, remoteLeader);
	//incomingV->print();
    }

    imposeConstraint();
}


void Exchanger::imposeConstraint(){
    std::cout << "in Exchanger::imposeConstraint" << std::endl;

    if(rank == leader) {
	// this function is for 3D velocity field only
        const int dim = 3;

	// area-weighted normal vector
        double* nwght  = new double[boundary->size * dim];
	computeWeightedNormal(nwght);

        double outflow = computeOutflow(incomingV, nwght);
        std::cout << " Net outflow before boundary velocity correction " << outflow << std::endl;

	if (outflow > E->control.tole_comp) {
	    reduceOutflow(outflow, nwght);

	    outflow = computeOutflow(incomingV, nwght);
	    std::cout << " Net outflow after boundary velocity correction (SHOULD BE ZERO !) " << outflow << std::endl;
	}

        delete [] nwght;
    }
}


void Exchanger::imposeBC() {
    std::cout << "in Exchanger::imposeBC" << std::endl;

    double N1,N2;

    if(cge_t == 0) {
        N1 = 0.0;
        N2 = 1.0;
    } else {
        N1 = (cge_t - fge_t) / cge_t;
        N2 = fge_t / cge_t;
    }

    // setup aliases
    const int dim = 3;
    Array2D<dim>& oldV = *old_incomingV;
    Array2D<dim>& newV = *incomingV;

    for(int m=1; m<=E->sphere.caps_per_proc; m++) {
	for(int i=0; i<boundary->size; i++) {
	    int n = boundary->bid2gid[i];
	    int p = boundary->bid2proc[i];
	    if (p == rank) {
 		for(int d=0; d<dim; d++)
 		    E->sphere.cap[m].VB[d+1][n] = N1 * oldV(d,i)
 		                                + N2 * newV(d,i);
//                 std::cout << E->sphere.cap[m].VB[1][n] << " "
// 			  << E->sphere.cap[m].VB[2][n] << " "
// 			  <<  E->sphere.cap[m].VB[3][n] << std::endl;
	    }
	}
    }
}


void Exchanger::setBCFlag() {
    std::cout << "in Exchanger::setBCFlag" << std::endl;

    // Because CitcomS is defaulted to have reflecting side BC,
    // here we should change to velocity BC.
    for(int m=1; m<=E->sphere.caps_per_proc; m++)
	for(int i=0; i<boundary->size; i++) {
	    int n = boundary->bid2gid[i];
	    int p = boundary->bid2proc[i];
	    if (p == rank) {
// 		std::cout << "    before: " << std::hex
// 			  << E->node[m][n] << std::dec << std::endl;
		E->node[m][n] = E->node[m][n] | VBX;
		E->node[m][n] = E->node[m][n] | VBY;
		E->node[m][n] = E->node[m][n] | VBZ;
		E->node[m][n] = E->node[m][n] & (~SBX);
		E->node[m][n] = E->node[m][n] & (~SBY);
		E->node[m][n] = E->node[m][n] & (~SBZ);
// 		std::cout << "    after : "  << std::hex
// 			  << E->node[m][n] << std::dec << std::endl;
	    }
	}

    // reconstruct ID array to reflect changes in BC
    construct_id(E);
}


void Exchanger::storeTimestep(const double fge_time, const double cge_time) {
    fge_t = fge_time;
    cge_t = cge_time;
}


double Exchanger::exchangeTimestep(const double dt) const {
    std::cout << "in Exchanger::exchangeTimestep"
	      << "  rank = " << rank
	      << "  leader = "<< localLeader
	      << "  receiver = "<< remoteLeader << std::endl;
    return exchangeDouble(dt, 1);
}


int Exchanger::exchangeSignal(const int sent) const {
    std::cout << "in Exchanger::exchangeSignal" << std::endl;
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


void Exchanger::computeWeightedNormal(double* nwght) const {
    const int dim = 3;
    const int enodes = 2 << (dim-1); // # of nodes per element
    const int facenodes[]={0, 1, 5, 4,
                           2, 3, 7, 6,
                           1, 2, 6, 5,
                           0, 4, 7, 3,
                           4, 5, 6, 7,
                           0, 3, 2, 1};

    for(int i=0; i<boundary->size*dim; i++) nwght[i] = 0.0;

    int nodest = enodes * E->lmesh.nel;
    int* bnodes = new int[nodest];
    for(int i=0; i< nodest; i++) bnodes[i] = -1;

    // Assignment of the local boundary node numbers
    // to bnodes elements array
    for(int n=0; n<E->lmesh.nel; n++) {
	for(int j=0; j<enodes; j++) {
	    int gnode = E->IEN[E->mesh.levmax][1][n+1].node[j+1];
	    for(int k=0; k<boundary->size; k++) {
		if(gnode == boundary->bid2gid[k]) {
		    bnodes[n*enodes+j] = k;
		    break;
		}
	    }
	}
    }

    double garea[dim][2];
    for(int i=0; i<dim; i++)
	for(int j=0; j<2; j++)
	    garea[i][j] = 0.0;

    for(int n=0; n<E->lmesh.nel; n++) {
	// Loop over element faces
	for(int i=0; i<6; i++) {
	    // Checking of diagonal nodal faces
	    if((bnodes[n*enodes+facenodes[i*4]] >=0) &&
	       (bnodes[n*enodes+facenodes[i*4+1]] >=0) &&
	       (bnodes[n*enodes+facenodes[i*4+2]] >=0) &&
	       (bnodes[n*enodes+facenodes[i*4+3]] >=0)) {

		double xc[12], normal[dim];
		for(int j=0; j<4; j++) {
		    int lnode = bnodes[n*enodes+facenodes[i*4+j]];
		    if(lnode >= boundary->size)
			std::cout <<" lnode = " << lnode
				  << " size " << boundary->size
				  << std::endl;
		    for(int l=0; l<dim; l++)
			xc[j*dim+l] = boundary->X[l][lnode];
		}

		normal[0] = (xc[4]-xc[1])*(xc[11]-xc[2])
		          - (xc[5]-xc[2])*(xc[10]-xc[1]);
		normal[1] = (xc[5]-xc[2])*(xc[9]-xc[0])
			  - (xc[3]-xc[0])*(xc[11]-xc[2]);
		normal[2] = (xc[3]-xc[0])*(xc[10]-xc[1])
			  - (xc[4]-xc[1])*(xc[9]-xc[0]);
		double area = sqrt(normal[0]*normal[0]
				   + normal[1]*normal[1]
				   + normal[2]*normal[2]);

		for(int l=0; l<dim; l++)
		    normal[l] /= area;


		if(xc[0] == xc[6])
		    area = fabs(0.5 * (xc[2]+xc[8]) * (xc[8]-xc[2])
				* (xc[7]-xc[1]) * sin(xc[0]));
		if(xc[1] == xc[7])
		    area = fabs(0.5 * (xc[2]+xc[8]) * (xc[8]-xc[2])
				* (xc[6]-xc[0]));
		if(xc[2] == xc[8])
		    area = fabs(xc[2] * xc[8] * (xc[7]-xc[1])
				* (xc[6]-xc[0]) * sin(0.5*(xc[0]+xc[6])));

		for(int l=0; l<dim; l++) {
		    if(normal[l] > 0.999 ) garea[l][0] += area;
                        if(normal[l] < -0.999 ) garea[l][1] += area;
		}
		for(int j=0; j<4; j++) {
		    int lnode = bnodes[n*enodes+facenodes[i*4+j]];
		    for(int l=0; l<dim; l++)
			nwght[lnode*dim+l] += normal[l] * area/4.;
		}
	    } // end of check of nodes
	} // end of loop over faces
    } // end of loop over elements
    delete [] bnodes;
}


double Exchanger::computeOutflow(const Velo& V,
				 const double* nwght) const {
    const int dim = 3;

    double outflow = 0;
    for(int n=0; n<V->size(); n++)
	for(int j=0; j<dim; j++)
	    outflow += (*V)(j,n) * nwght[n*dim+j];

    return outflow;
}


void Exchanger::reduceOutflow(const double outflow, const double* nwght) {

	const int dim = 3;
	const int size = boundary->size;

	double total_area = 0;
	for(int n=0; n<size; n++)
	    for(int j=0; j<dim; j++)
		total_area += fabs(nwght[n*3+j]);

	auto_array_ptr<double> tmp(new double[dim*size]);

	for(int i=0; i<size; i++)
	    for(int j=0; j<dim; j++)
		tmp[i*dim+j] = (*incomingV)(i,j);

	for(int n=0; n<size; n++) {
            for(int j=0; j<dim; j++)
                if(fabs(nwght[n*dim+j]) > 1.e-10) {
                    tmp[n*dim+j] = (*incomingV)(j,n)
			         - outflow * nwght[n*dim+j]
			           / (total_area * fabs(nwght[n*dim+j]));
	    }
	}

	incomingV = Velo(new Array2D<dim>(tmp, size));
	//incomingV->print();
}


// version
// $Id: ExchangerClass.cc,v 1.32 2003/10/10 18:14:49 tan2 Exp $

// End of file

