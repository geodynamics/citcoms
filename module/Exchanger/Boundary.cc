// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <iostream>
#include <memory>
#include "global_defs.h"
#include "Boundary.h"


Boundary::Boundary(const int n) :
    size(n),
    bid2gid(new int[size]),
    bid2elem(new int[size]),
    bid2proc(new int[size]),
    shape(new double[size*8])
{
    std::cout << "in Boundary::Boundary  size = " << size << std::endl;

    for(int i=0; i<dim; i++) {
	auto_array_ptr<double> tmp(new double[size]);
	X[i] = tmp;
    }
}


Boundary::~Boundary() {
    std::cout << "in Boundary::~Boundary" << std::endl;
}


void Boundary::mapFineGrid(const All_variables *E) {
    initBound(E);
    findBoundaryNodes(E);
}


void Boundary::initBound(const All_variables *E) {
    theta_max = E->control.theta_max;
    theta_min = E->control.theta_min;
    fi_max = E->control.fi_max;
    fi_min = E->control.fi_min;
    ro = E->sphere.ro;
    ri = E->sphere.ri;

    //printBound();
}


void Boundary::findBoundaryNodes(const All_variables *E) {
    int node1,node2;
    int nodest = E->lmesh.nox * E->lmesh.noy * E->lmesh.noz;

    int *nid = new int[nodest];
    for(int i=0;i<nodest;i++) nid[i]=0;

    int nodes=0;

    //  for two YOZ planes

    if (E->parallel.me_loc[1]==0 || E->parallel.me_loc[1]==E->parallel.nprocx-1)
	for (int m=1;m<=E->sphere.caps_per_proc;m++)
	    for(int j=1;j<=E->lmesh.noy;j++)
		for(int i=1;i<=E->lmesh.noz;i++)  {
		    node1 = i + (j-1)*E->lmesh.noz*E->lmesh.nox;
		    node2 = node1 + (E->lmesh.nox-1)*E->lmesh.noz;

		    if ((E->parallel.me_loc[1]==0) && (!nid[node1-1]))  {
			for(int k=0;k<dim;k++)X[k][nodes]=E->sx[m][k+1][node1];
			bid2gid[nodes] = node1;
			nid[node1-1]++;
			nodes++;
		    }
		    if ((E->parallel.me_loc[1]==E->parallel.nprocx-1) && (!nid[node2-1])) {
			for(int k=0;k<dim;k++)X[k][nodes]=E->sx[m][k+1][node2];
			bid2gid[nodes] = node2;
			nid[node2-1]++;
			nodes++;
		    }
		}

    //  for two XOZ planes

    if (E->parallel.me_loc[2]==0 || E->parallel.me_loc[2]==E->parallel.nprocy-1)
	for (int m=1;m<=E->sphere.caps_per_proc;m++)
	    for(int j=1;j<=E->lmesh.nox;j++)
		for(int i=1;i<=E->lmesh.noz;i++)  {
		    node1 = i + (j-1)*E->lmesh.noz;
		    node2 = node1 + (E->lmesh.noy-1)*E->lmesh.noz*E->lmesh.nox;
		    if ((E->parallel.me_loc[2]==0) && (!nid[node1-1]))  {
			for(int k=0;k<dim;k++)X[k][nodes]=E->sx[m][k+1][node1];
			bid2gid[nodes] = node1;
			nid[node1-1]++;
			nodes++;
		    }
		    if((E->parallel.me_loc[2]==E->parallel.nprocy-1)&& (!nid[node2-1]))  {
			for(int k=0;k<dim;k++)X[k][nodes]=E->sx[m][k+1][node2];
			bid2gid[nodes] = node2;
			nid[node2-1]++;
			nodes++;
		    }
		}
    //  for two XOY planes
    if (E->parallel.me_loc[3]==0 || E->parallel.me_loc[3]==E->parallel.nprocz-1)
	for (int m=1;m<=E->sphere.caps_per_proc;m++)
	    for(int j=1;j<=E->lmesh.noy;j++)
		for(int i=1;i<=E->lmesh.nox;i++)  {
		    node1 = 1 + (i-1)*E->lmesh.noz+(j-1)*E->lmesh.nox*E->lmesh.noz;
		    node2 = node1 + E->lmesh.noz-1;

		    if ((E->parallel.me_loc[3]==0 ) && (!nid[node1-1])) {
			for(int k=0;k<dim;k++)X[k][nodes]=E->sx[m][k+1][node1];
			bid2gid[nodes] = node1;
			nid[node1-1]++;
			nodes++;
		    }
		    if ((E->parallel.me_loc[3]==E->parallel.nprocz-1) &&(!nid[node2-1])) {
			for(int k=0;k<dim;k++)X[k][nodes]=E->sx[m][k+1][node2];
			bid2gid[nodes] = node2;
			nid[node2-1]++;
			nodes++;
		    }
		}
    if(nodes != size) std::cout << " nodes != size ";

    delete [] nid;

    //printX();
    //printBid2gid();
}


void Boundary::mapCoarseGrid(const All_variables *E, const int rank) {
    std::cout << "in Boundary::mapCoarseGrid" << std::endl;

    double xt[3],xc[24],dett,det[4],x1[3],x2[3],x3[3],x4[3];
    int nsub[]={0, 2, 3, 7,
		0, 1, 2, 5,
		4, 7, 5, 0,
		5, 7, 6, 2,
		5, 7, 2, 0};
    for(int i=0; i<size; i++)
    {
	bid2proc[i] = E->parallel.nproc;  // nproc is always an invalid rank
	bid2elem[i] = 0;
    }

    // find max. grid spacing
    const double pi = 4*atan(1);
    const double cap_side = 0.5*pi / sqrt(2);  // side length of a spherical cap
    double elem_side = cap_side / E->mesh.elx;

    double theta_tol = elem_side;
    double fi_tol = elem_side;
    double r_tol = 0;
    for(int n=0; n<E->lmesh.nel; n++) {
	const int m = 1;
	int gnode1 = E->IEN[E->mesh.levmax][m][n+1].node[1];
	int gnode5 = E->IEN[E->mesh.levmax][m][n+1].node[5];
	r_tol = max(r_tol, fabs(E->sx[m][3][gnode5] - E->sx[m][3][gnode1]));
    }
    //std::cout << " tol: " << theta_tol << " " << fi_tol << " " << r_tol << std::endl;

    for(int i=0; i< size; i++) {
	for(int j=0; j< dim; j++)xt[j]=X[j][i];
	// loop over 5 sub tets in a brick element
        int ind = 0;

        for(int mm=1;mm<=E->sphere.caps_per_proc;mm++)
            for(int n=0;n<E->lmesh.nel;n++) {
                for(int j=0; j < 8; j++) {
		    int gnode = E->IEN[E->mesh.levmax][mm][n+1].node[j+1];
                    for(int k=0; k < dim; k++) {
                        xc[j*dim+k]=E->sx[mm][k+1][gnode];
                    }
		}
                int in = 0;
                for(int j=0; j<8; j++) {
		    if(((theta_min - xc[j*3]) <= theta_tol) &&
		       ((xc[j*3] - theta_max) <= theta_tol) &&
		       ((fi_min - xc[j*3+1]) <= fi_tol) &&
		       ((xc[j*3+1] - fi_max) <= fi_tol) &&
		       ((ri - xc[j*3+2]) <= r_tol) &&
		       ((xc[j*3+2] - ro) <= r_tol))
			in = 1;
                }
                if(in==0)continue;
                for(int k=0; k < 5; k++) {
                    for(int m=0; m < dim; m++) {
                        x1[m]=xc[nsub[k*4]*dim+m];
                        x2[m]=xc[nsub[k*4+1]*dim+m];
                        x3[m]=xc[nsub[k*4+2]*dim+m];
                        x4[m]=xc[nsub[k*4+3]*dim+m];
                    }
                    dett=Tetrahedronvolume(x1,x2,x3,x4);
                    det[0]=Tetrahedronvolume(x2,x4,x3,xt);
                    det[1]=Tetrahedronvolume(x3,x4,x1,xt);
                    det[2]=Tetrahedronvolume(x1,x4,x2,xt);
                    det[3]=Tetrahedronvolume(x1,x2,x3,xt);
                    if(dett < 0)
                    {
                        std::cout << " Determinant evaluation is wrong " << in << std::endl;
		       std::cout << " node " << i << " " << xt[0] << " " << xt[1] << " " << xt[2] << std::endl;
                       for(int j=0;j<8;j++)
                            std::cout << xc[j*3] <<" "<<xc[j*3+1] <<" " << xc[j*3+2] << std::endl;
                    }

                    if(det[0] < 0.0 || det[1] <0.0 || det[2] < 0.0 || det[3] < 0.0) continue;
//                    std::cout << "node" << i <<"Found the  element "<< n+1 <<std::endl;

//		       std::cout << " node " << i << " " << xt[0] << " " << xt[1] << " " << xt[2] << std::endl;
//                       for(int j=0;j<8;j++)
//                            std::cout << xc[j*3] <<" "<<xc[j*3+1] <<" " << xc[j*3+2] << std::endl;
                    ind=1;
                    bid2elem[i]=n+1;
                    bid2proc[i]=rank;
                    for(int j=0;j<8;j++)shape[i*8+j]=0.0;
                    shape[i*8+nsub[k*4]]=det[0]/dett;
                    shape[i*8+nsub[k*4+1]]=det[1]/dett;
                    shape[i*8+nsub[k*4+2]]=det[2]/dett;
                    shape[i*8+nsub[k*4+3]]=det[3]/dett;
                    break;
                }
                if(ind) break;
            }
    }
    //printBid2proc();
    //printBid2elem();

    testMapping(E);

}


void Boundary::testMapping(const All_variables *E) const {
    double xc[24], xi[3], xt[3],tshape;

    for(int i=0; i< size; i++) {
	if(!bid2elem[i])continue;
        for(int j=0; j< dim; j++) xt[j]=X[j][i];

        int n1=bid2elem[i];

        for(int j=0; j < 8; j++) {
            for(int k=0; k < dim; k++) {
                xc[j*dim+k]=E->sx[1][k+1][E->IEN[E->mesh.levmax][1][n1].node[j+1]];
            }
	    //std::cout <<" " <<xc[j*dim] << " " << xc[j*dim+1] << " " << xc[j*dim+2] <<" "<< shape[i*8+j] << std::endl;
        }
        for(int k=0; k<dim; k++)xi[k]=0.0;
        for(int k=0; k<dim; k++)
            for(int j=0; j < 8; j++) {
                xi[k]+=xc[j*dim+k]*shape[i*8+j];
            }
	//std::cout << " "<< xt[0] <<" "<< xi[0] <<" "<< xt[1] << " "<< xi[1] << " " << xt[2] << " " << xi[2] << std::endl;
        double norm = 0.0;
        for(int k=0; k < dim; k++) norm+=(xt[k]-xi[k])*(xt[k]-xi[k]);
        if(norm > 1.e-10) {
            tshape=0.0;
            for(int j=0; j < 8; j++) tshape+=shape[i*8+j];
            std::cout << "\n in Boundary::testMappingfor bid2elem interpolation functions are wrong " << norm <<std::endl;
            std::cout << "node #"<< i <<"tshape" << tshape <<std::endl;
            std::cout << xi[0] <<" " <<xt[0] <<" "<< xi[1] <<" " <<xt[1] << " "<<xi[2] <<" " <<xt[2] << std::endl;
        }
    }
}


double Boundary::Tetrahedronvolume(double  *x1, double *x2, double *x3, double *x4)  const
{
    double vol;
//    xx[0] = x2;  xx[1] = x3;  xx[2] = x4;
    vol = det3_sub(x2,x3,x4);
//    xx[0] = x1;  xx[1] = x3;  xx[2] = x4;
    vol -= det3_sub(x1,x3,x4);
//    xx[0] = x1;  xx[1] = x2;  xx[2] = x4;
    vol += det3_sub(x1,x2,x4);
//    xx[0] = x1;  xx[1] = x2;  xx[2] = x3;
    vol -= det3_sub(x1,x2,x3);
    vol /= 6.;
    return vol;
}


double Boundary::det3_sub(double *x1, double *x2, double *x3) const
{
    return (x1[0]*(x2[1]*x3[2]-x3[1]*x2[2])
            -x1[1]*(x2[0]*x3[2]-x3[0]*x2[2])
            +x1[2]*(x2[0]*x3[1]-x3[0]*x2[1]));
}



void Boundary::send(const MPI_Comm comm, const int receiver) const {
    int tag = 1;

    for (int i=0; i<dim; i++) {
	MPI_Send(X[i].get(), size, MPI_DOUBLE,
		 receiver, tag, comm);
	tag ++;
    }

    const int size_temp = 6;
    double temp[size_temp] = {theta_max,
			      theta_min,
			      fi_max,
			      fi_min,
			      ro,
			      ri};

    MPI_Send(temp, size_temp, MPI_DOUBLE,
	     receiver, tag, comm);
    tag ++;

    return;
}


void Boundary::receive(const MPI_Comm comm, const int sender) {
    MPI_Status status;
    int tag = 1;

    for (int i=0; i<dim; i++) {
	MPI_Recv(X[i].get(), size, MPI_DOUBLE,
		 sender, tag, comm, &status);
	tag ++;
    }
    //printX();

    const int size_temp = 6;
    double temp[size_temp];

    MPI_Recv(&temp, size_temp, MPI_DOUBLE,
	     sender, tag, comm, &status);
    tag ++;

    theta_max = temp[0];
    theta_min = temp[1];
    fi_max = temp[2];
    fi_min = temp[3];
    ro = temp[4];
    ri = temp[5];
    //printBound();

    return;
}


void Boundary::broadcast(const MPI_Comm comm, const int broadcaster) {

    for (int i=0; i<dim; i++) {
      MPI_Bcast(X[i].get(), size, MPI_DOUBLE, broadcaster, comm);
    }
    //printX();

    MPI_Bcast(&theta_max, 1, MPI_DOUBLE, broadcaster, comm);
    MPI_Bcast(&theta_min, 1, MPI_DOUBLE, broadcaster, comm);
    MPI_Bcast(&fi_max, 1, MPI_DOUBLE, broadcaster, comm);
    MPI_Bcast(&fi_min, 1, MPI_DOUBLE, broadcaster, comm);
    MPI_Bcast(&ro, 1, MPI_DOUBLE, broadcaster, comm);
    MPI_Bcast(&ri, 1, MPI_DOUBLE, broadcaster, comm);
    //printBound();

    return;
}


void Boundary::sendBid2proc(const MPI_Comm comm,
			    const int rank, const int leader) {
    std::cout << "in Boundary::sendBid2proc" << std::endl;

    if (rank == leader) {
	int nproc;
	MPI_Comm_size(comm, &nproc);

	std::auto_ptr<int> tmp = auto_ptr<int>(new int[size]);
	int *ptmp = tmp.get();

	for (int i=0; i<nproc; i++) {
	    if (i == leader) continue; // skip leader itself

	    MPI_Status status;
	    MPI_Recv(ptmp, size, MPI_INT,
	    	     i, i, comm, &status);
	    for (int n=0; n<size; n++) {
		if (ptmp[n] != nproc) bid2proc[n] = ptmp[n];
	    }
	}
	// whether all boundary nodes are mapped to a processor?
	for (int n=0; n<size; n++)
	    if (bid2proc[n] == nproc) {
		// nproc is an invalid rank
		printBid2proc();
		break;
	    }

	//printBid2proc();
    }
    else {
	MPI_Send(bid2proc.get(), size, MPI_INT,
		 leader, rank, comm);
    }

}


void Boundary::printX() const {
    for(int j=0; j<size; j++) {
	std::cout << "  X:  " << j << ":  ";
	    for(int i=0; i<dim; i++)
		std::cout << X[i][j] << " ";
	std::cout << std::endl;
    }
}


void Boundary::printBid2gid() const {
    int *c = bid2gid.get();
    for(int j=0; j<size; j++)
	std::cout << "  B:  " << j << ":  " << c[j] << std::endl;
}


void Boundary::printBid2proc() const {
    int *c = bid2proc.get();
    for(int j=0; j<size; j++)
	std::cout << "  proc:  " << j << ":  " << c[j] << std::endl;
}


void Boundary::printBid2elem() const {
    int *c = bid2elem.get();
    for(int j=0; j<size; j++)
	std::cout << "  elem:  " << j << ":  " << c[j] << std::endl;
}


void Boundary::printBound() const {
    std::cout << "theta= " << theta_min
	      << " : " << theta_max << std::endl;
    std::cout << "fi   = " << fi_min
	      << " : " << fi_max << std::endl;
    std::cout << "r    = " << ri
	      << " : " << ro  << std::endl;
}



// version
// $Id: Boundary.cc,v 1.30 2003/10/03 18:36:17 tan2 Exp $

// End of file
