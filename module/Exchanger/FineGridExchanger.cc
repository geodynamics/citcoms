// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <portinfo>
#include <iostream>

#include "Array2D.h"
#include "Array2D.cc"
#include "Boundary.h"
#include "Mapping.h"
#include "FineGridExchanger.h"
#include "global_defs.h"

extern "C" {
    void check_bc_consistency(const All_variables *E);
    void construct_id(const All_variables *E);
}


FineGridExchanger::FineGridExchanger(const MPI_Comm comm,
				     const MPI_Comm intercomm,
				     const int leader,
				     const int localLeader,
				     const int remoteLeader,
				     const All_variables *E):
    Exchanger(comm, intercomm, leader, localLeader, remoteLeader, E)
{
    std::cout << "in FineGridExchanger::FineGridExchanger" << std::endl;
}


FineGridExchanger::~FineGridExchanger() {
    std::cout << "in FineGridExchanger::~FineGridExchanger" << std::endl;
}


void FineGridExchanger::gather() {
    std::cout << "in FineGridExchanger::gather" << std::endl;
}


void FineGridExchanger::distribute() {
    std::cout << "in FineGridExchanger::distribute" << std::endl;
}


void FineGridExchanger::interpretate() {
    std::cout << "in FineGridExchanger::interpretate" << std::endl;
}


void FineGridExchanger::mapBoundary() {
    std::cout << "in FineGridExchanger::mapBoundary" << std::endl;

    // Assuming all boundary nodes are inside localLeader!
    // assumption will be relaxed in future
    if (rank == leader) {
	boundary->initBound(E);
	createMapping();
	createDataArrays();
    }
}


void FineGridExchanger::createMapping() {
    fgmapping = new FineGridMapping(boundary, E, comm, rank, leader);
    mapping = fgmapping;
}


void FineGridExchanger::createBoundary() {
    std::cout << "in FineGridExchanger::createBoundary" << std::endl;

    if (rank == leader) {
	// boundary = all - interior
	int size = E->mesh.nno - (E->mesh.nox-2)*(E->mesh.noy-2)*(E->mesh.noz-2);
	boundary = new Boundary(size);
    }
}


void FineGridExchanger::sendBoundary() {
    std::cout << "in FineGridExchanger::sendBoundary"
	      << "  rank = " << rank
	      << "  leader = "<< localLeader
	      << "  receiver = "<< remoteLeader << std::endl;

    if (rank == leader) {
	int tag = 0;
	int itmp = boundary->size();
	MPI_Send(&itmp, 1, MPI_INT,
		 remoteLeader, tag, intercomm);

	boundary->send(intercomm, remoteLeader);
    }
}


void FineGridExchanger::setBCFlag() {
    std::cout << "in FineGridExchanger::setBCFlag" << std::endl;

    // Because CitcomS is defaulted to have reflecting side BC,
    // here we should change to velocity BC.
    for(int m=1; m<=E->sphere.caps_per_proc; m++)
	for(int i=0; i<boundary->size(); i++) {
	    int n = fgmapping->bid2gid(i);
	    int p = fgmapping->bid2proc(i);
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

    check_bc_consistency(E);
    // reconstruct ID array to reflect changes in BC
    construct_id(E);
}


void FineGridExchanger::imposeConstraint(){
    std::cout << "in FineGridExchanger::imposeConstraint" << std::endl;

    if(rank == leader) {
	// this function is for 3D velocity field only
        const int dim = 3;

	// area-weighted normal vector
        double* nwght  = new double[boundary->size() * dim];
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


void FineGridExchanger::imposeBC() {
    std::cout << "in FineGridExchanger::imposeBC" << std::endl;

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
	for(int i=0; i<boundary->size(); i++) {
	    int n = fgmapping->bid2gid(i);
	    int p = fgmapping->bid2proc(i);
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


void FineGridExchanger::computeWeightedNormal(double* nwght) const {
    const int dim = 3;
    const int enodes = 2 << (dim-1); // # of nodes per element
    const int facenodes[]={0, 1, 5, 4,
                           2, 3, 7, 6,
                           1, 2, 6, 5,
                           0, 4, 7, 3,
                           4, 5, 6, 7,
                           0, 3, 2, 1};

    for(int i=0; i<boundary->size()*dim; i++) nwght[i] = 0.0;

    int nodest = enodes * E->lmesh.nel;
    int* bnodes = new int[nodest];
    for(int i=0; i< nodest; i++) bnodes[i] = -1;

    // Assignment of the local boundary node numbers
    // to bnodes elements array
    for(int n=0; n<E->lmesh.nel; n++) {
	for(int j=0; j<enodes; j++) {
	    int gnode = E->IEN[E->mesh.levmax][1][n+1].node[j+1];
	    for(int k=0; k<boundary->size(); k++) {
		if(gnode == fgmapping->bid2gid(k)) {
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
		    if(lnode >= boundary->size())
			std::cout <<" lnode = " << lnode
				  << " size " << boundary->size()
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


double FineGridExchanger::computeOutflow(const Velo& V,
					 const double* nwght) const {
    const int dim = 3;

    double outflow = 0;
    for(int n=0; n<V->size(); n++)
	for(int j=0; j<dim; j++)
	    outflow += (*V)(j,n) * nwght[n*dim+j];

    return outflow;
}


void FineGridExchanger::reduceOutflow(const double outflow,
				      const double* nwght) {

	const int dim = 3;
	const int size = boundary->size();

	double total_area = 0;
	for(int n=0; n<size; n++)
	    for(int j=0; j<dim; j++)
		total_area += fabs(nwght[n*3+j]);

	auto_array_ptr<double> tmp(new double[dim*size]);

	for(int n=0; n<size; n++)
	    for(int j=0; j<dim; j++)
		tmp[n*dim+j] = (*incomingV)(j,n);

	for(int n=0; n<size; n++) {
            for(int j=0; j<dim; j++)
                if(fabs(nwght[n*dim+j]) > 1.e-10) {
                    tmp[n*dim+j] = (*incomingV)(j,n)
			         - outflow * nwght[n*dim+j]
			           / (total_area * fabs(nwght[n*dim+j]));
	    }
	}

	incomingV = Velo(new Array2D<dim>(tmp, size));
	//incomingV->print("incomingV");
}



// version
// $Id: FineGridExchanger.cc,v 1.23 2003/10/15 18:51:24 tan2 Exp $

// End of file
