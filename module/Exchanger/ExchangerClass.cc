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
    incoming.size = size;
    incoming.T = new double[size];
    outgoing.size = size;
    outgoing.T = new double[size];
    poutgoing.size = size;
    poutgoing.T = new double[size];
    for(int i=0; i < boundary->dim; i++) {
	incoming.v[i] = new double[size];
	outgoing.v[i] = new double[size];
        poutgoing.v[i] = new double[size];
    }
}


void Exchanger::deleteDataArrays() {
    std::cout << "in Exchanger::deleteDataArrays" << std::endl;

      delete [] incoming.T;
      delete [] outgoing.T;
      delete [] poutgoing.T;
      for(int i=0; i < boundary->dim; i++) {
	  delete [] incoming.v[i];
	  delete [] outgoing.v[i];
	  delete [] poutgoing.v[i];
      }
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


void Exchanger::sendTemperature(void) {
    std::cout << "in Exchanger::sendTemperature"
	      << "  rank = " << rank
	      << "  leader = "<< localLeader
	      << "  receiver = "<< remoteLeader
	      << std::endl;

    if(rank == leader) {

//       std::cout << "nox = " << E->mesh.nox << std::endl;
//       for(int j=0; j < boundary->size; j++)
// 	{
// 	  n=boundary->bid2gid[j];
// 	  outgoing.T[j]=E->T[1][n];

// 	  // Test
// 	  std::cout << "Temperature sent" << std::endl;
// 	  std::cout << j << " " << n << "  " << outgoing.T[j] << std::endl;

// 	}
      MPI_Send(outgoing.T,outgoing.size,MPI_DOUBLE,remoteLeader,0,intercomm);
    }

//     delete outgoing.T;

    return;
}


void Exchanger::receiveTemperature(void) {
    std::cout << "in Exchanger::receiveTemperature"
	      << "  rank = " << rank
	      << "  leader = "<< localLeader
	      << "  receiver = "<< remoteLeader
	      << std::endl;
    int n,success;

    MPI_Status status;
    MPI_Request request;

    if(rank == localLeader) {
//       std::cout << "nox = " << E->nox << std::endl;

      MPI_Irecv(incoming.T,incoming.size,MPI_DOUBLE,remoteLeader,0,intercomm,&request);
      std::cout << "Exchanger::receiveTemperature ===> Posted" << std::endl;
    }
    // Test
    MPI_Wait(&request,&status);
    MPI_Test(&request,&success,&status);
    if(success)
      std::cout << "Temperature transfer Succeeded!!" << std::endl;

    for(int j=0; j < boundary->size; j++)
      {
	n=boundary->bid2gid[j];
	std::cout << "Temperature received" << std::endl;
	std::cout << j << " " << n << "  " << incoming.T[j] << std::endl;
      }
    // Don' forget to delete incoming.T
    return;
}


void Exchanger::sendVelocities() {
    std::cout << "in Exchanger::sendVelocities" << std::endl;

    if(rank == leader) {
	int tag = 0;
	for(int i=0; i < boundary->dim; i++) {
	    MPI_Send(outgoing.v[i], outgoing.size, MPI_DOUBLE,
		     remoteLeader, tag, intercomm);
	    tag ++;
	}
    }
}


void Exchanger::receiveVelocities() {
    std::cout << "in Exchanger::receiveVelocities" << std::endl;

    if(rank == leader) {
	int tag = 0;
	MPI_Status status;
	for(int i=0; i < boundary->dim; i++) {
            for(int n=0; n < incoming.size; n++)
            {
                if(!((fge_t==0)&&(cge_t==0)))poutgoing.v[i][n]=incoming.v[i][n];
            }
	    MPI_Recv(incoming.v[i], incoming.size, MPI_DOUBLE,
		     remoteLeader, tag, intercomm, &status);
	    tag ++;
            if((fge_t==0)&&(cge_t==0))
            {
                for(int n=0; n < incoming.size; n++)poutgoing.v[i][n]=incoming.v[i][n];
            }

	}
    }
    imposeConstraint();

//printDataV(incoming);
    //printDataV(poutgoing);

    // Don't forget to delete inoming.v
    return;
}
void Exchanger::imposeConstraint(){
    std::cout << "in Exchanger::imposeConstraint" << std::endl;

    int nodest,gnode,lnode;
    int *bnodes;
    double xc[12],avgV[3],normal[3],garea[3][2],tarea;
    double outflow,area,factr,*nwght;

    int facenodes[]={0, 1, 5, 4,
		     2, 3, 7, 6,
                     1, 2, 6, 5,
                     0, 4, 7, 3,
                     4, 5, 6, 7,
                     0, 3, 2, 1};
    if(rank == leader) {

        nodest = 8*E->lmesh.nel;
        bnodes = new int[nodest];
        nwght  = new double[boundary->size*3];
        for(int i=0; i< nodest; i++) bnodes[i]=-1;
        for(int i=0; i< boundary->size*3; i++) nwght[i]=0.0;
// Assignment of the local boundary node numbers to bnodes elements array
        for(int n=0; n<E->lmesh.nel; n++)
        {
            for( int j=0; j<8; j++)
            {
                gnode = E->IEN[E->mesh.levmax][1][n+1].node[j+1];
                for(int k=0; k < incoming.size; k++)
                {
                    if(gnode==boundary->bid2gid[k])bnodes[n*8+j]=k;
                    if(gnode==boundary->bid2gid[k])break;
                }

            }
        }

        outflow=0.0;
       	for( int i=0;i<3;i++)
		for(int j=0; j<2 ;j++)
			garea[i][j]=0.0;

        for(int n=0; n<E->lmesh.nel; n++)
        {
// Loop over element faces
            for(int i=0; i<6; i++)
            {
                    // Checking of diagonal nodal faces
                if((bnodes[n*8+facenodes[i*4]] >=0)&&(bnodes[n*8+facenodes[i*4+1]] >=0) &&(bnodes[n*8+facenodes[i*4+2]] >=0)&&(bnodes[n*8+facenodes[i*4+3]] >=0))
                {
                    avgV[0]=avgV[1]=avgV[2]=0.0;
                    for(int j=0;j<4;j++)
                    {
                        lnode=bnodes[n*8+facenodes[i*4+j]];
 			if(lnode >= boundary->size) std::cout <<" lnode = " << lnode << " size " << boundary->size << std::endl;
                        for(int l=0; l<3; l++)
                        {
                            xc[j*3+l]=boundary->X[l][lnode];
                            avgV[l]+=incoming.v[l][lnode]/4.0;
                        }

                    }
                    normal[0]=(xc[4]-xc[1])*(xc[11]-xc[2])-(xc[5]-xc[2])*(xc[10]-xc[1]);
                    normal[1]=(xc[5]-xc[2])*(xc[9]-xc[0])-(xc[3]-xc[0])*(xc[11]-xc[2]);
                    normal[2]=(xc[3]-xc[0])*(xc[10]-xc[1])-(xc[4]-xc[1])*(xc[9]-xc[0]);
                    area=sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);

                    for(int l=0; l<3; l++)
                    {
                        normal[l]/=area;
                    }
		    if(xc[0]==xc[6]) area=fabs(0.5*(xc[2]+xc[8])*(xc[8]-xc[2])*(xc[7]-xc[1]));
		    if(xc[1]==xc[7]) area=fabs(0.5*(xc[2]+xc[8])*(xc[8]-xc[2])*(xc[6]-xc[0])*sin(0.5*(xc[7]+xc[1])));
		    if(xc[2]==xc[8]) area=fabs(xc[2]*xc[8]*(xc[7]-xc[1])*(xc[6]-xc[0])*sin(0.5*(xc[0]+xc[6])));


                    for(int l=0; l<3; l++)
                    {
                        if(normal[l] > 0.999 ) garea[l][0]+=area;
                        if(normal[l] < -0.999 ) garea[l][1]+=area;
                    }
                    for(int j=0;j<4;j++)
                    {
                        lnode=bnodes[n*8+facenodes[i*4+j]];
                        for(int l=0; l<3; l++)nwght[lnode*3+l]+=normal[l]*area/4.;
                    }

                    for(int l=0; l<3; l++)
                    {
                        outflow+=avgV[l]*normal[l]*area;
                    }

                }

            }
        }

        outflow=0.0;
        tarea=0.0;
        for(int n=0; n<boundary->size;n++)
        {
            for(int j=0; j < 3; j++)
            {
                outflow+=incoming.v[j][n]*nwght[n*3+j];
                tarea+=fabs(nwght[n*3+j]);
            }
        }

        std::cout << " Net outflow before boundary velocity correction in imposeConstraint" << outflow << std::endl;
        for(int n=0; n<boundary->size;n++)
        {
            for(int j=0; j < 3; j++)
            {
                if(fabs(nwght[n*3+j]) > 1.e-10)
                    incoming.v[j][n]-=outflow*nwght[n*3+j]/(tarea*fabs(nwght[n*3+j]));
            }
        }
        outflow=0.0;
        for(int n=0; n<boundary->size;n++)
        {
            for(int j=0; j < 3; j++)
            {
                outflow+=incoming.v[j][n]*nwght[n*3+j];
            }
        }
         std::cout << " Net outflow after boundary velocity correction in imposeConstraint (SHOULD BE ZERO !)" << outflow << std::endl;
        delete [] bnodes;
        delete [] nwght;
    }

//printDataV(incoming);
    //printDataV(poutgoing);

    // Don't forget to delete inoming.v
    return;
}

void Exchanger::imposeBC() {
    std::cout << "in Exchanger::imposeBC" << std::endl;

    double N1,N2;

    if(cge_t==0)
    {
        N1=1.0;
        N2=0.0;
    }
    else
    {
        N1=(cge_t-fge_t)/cge_t;
        N2=fge_t/cge_t;
    }

    for(int m=1;m<=E->sphere.caps_per_proc;m++) {
	for(int i=0;i<boundary->size;i++) {
	    int n = boundary->bid2gid[i];
	    int p = boundary->bid2proc[i];
	    if (p == rank) {
		E->sphere.cap[m].VB[1][n] = N1*poutgoing.v[0][i]+N2*incoming.v[0][i];
		E->sphere.cap[m].VB[2][n] = N1*poutgoing.v[1][i]+N2*incoming.v[1][i];
		E->sphere.cap[m].VB[3][n] = N1*poutgoing.v[2][i]+N2*incoming.v[2][i];
                //std::cout << E->sphere.cap[m].VB[1][n] << " " << E->sphere.cap[m].VB[2][n] << " " <<  E->sphere.cap[m].VB[3][n] << std::endl;
	    }
	}
    }

    return;
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


void Exchanger::printDataT(const Data &data) const {
    for (int n=0; n<data.size; n++) {
	std::cout << "  Data.T:  " << n << ":  "
		  << data.T[n] << std::endl;
    }
}


void Exchanger::printDataV(const Data &data) const {
    for (int n=0; n<data.size; n++) {
	std::cout << "  Data.v:  " << n << ":  ";
	for (int j=0; j<boundary->dim; j++)
	    std::cout << data.v[j][n] << "  ";
	std::cout << std::endl;
    }
}


// version
// $Id: ExchangerClass.cc,v 1.30 2003/10/05 18:55:48 tan2 Exp $

// End of file

