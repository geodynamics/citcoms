// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <portinfo>
#include <iostream>
#include <cmath>
#include "global_defs.h"
#include "Array2D.cc"
#include "Boundary.h"
#include "Mapping.h"


Mapping::Mapping(const int dim, const int size) :
    dim_(dim),
    size_(size),
    bid2proc_(size)
{}


Mapping::~Mapping() {};


void Mapping::printBid2proc(const std::string& prefix) const {
    bid2proc_.print(prefix + "  proc");
}


void Mapping::sendBid2proc(const MPI_Comm comm,
			   const int rank, const int leader) {

    if (rank == leader) {
	int nproc;
	MPI_Comm_size(comm, &nproc);

	Array2D<int,1> tmp(size_);

	for (int i=0; i<nproc; i++) {
	    if (i == leader) continue; // skip leader itself

	    tmp.receive(comm, i);

 	    for (int n=0; n<size_; n++) {
 		if (tmp[0][n] != nproc) bid2proc_[0][n] = tmp[0][n];
 	    }
	}

	// whether all boundary nodes are mapped to a processor?
	for (int n=0; n<size_; n++)
	    if (bid2proc_[0][n] == nproc) { // nproc is an invalid rank
		printBid2proc();
		const std::string msg = "Unmapped boundary node";
		std::cerr << msg << std::endl;
		throw msg;
	    }

	printBid2proc();
    }
    else {
	bid2proc_.send(comm, leader);
    }
}


void Mapping::resize(const int n) {
    if (n == size_) return;

    bid2proc_.resize(n);
    size_ = n;
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


CoarseGridMapping::CoarseGridMapping(const Boundary* boundary,
				     const All_variables* E,
				     const MPI_Comm comm,
				     const int rank, const int leader) :
    Mapping(boundary->dim(), boundary->size()),
    bid2elem_(size_),
    shape_(8*size_)
{
    double theta_tol = 0;
    double fi_tol = 0;
    double r_tol = 0;

    findMaxGridSpacing(E, theta_tol, fi_tol, r_tol);
    findBoundaryElements(boundary, E, rank,
     			 theta_tol, fi_tol, r_tol);
    sendBid2proc(comm, rank, leader);
}


void CoarseGridMapping::printBid2elem(const std::string& prefix) const {
    bid2elem_.print(prefix + " elem");
}


void CoarseGridMapping::resize(const int n) {
    if (n == size_) return;

    bid2elem_.resize(n);
    shape_.resize(8*n);
    Mapping::resize(n);
}


void CoarseGridMapping::findMaxGridSpacing(const All_variables* E,
					   double& theta_tol,
					   double& fi_tol,
					   double& r_tol) const {
    const double pi = 4*atan(1);
    const double cap_side = 0.5*pi / sqrt(2);  // side length of a spherical cap
    double elem_side = cap_side / E->mesh.elx;

    theta_tol = elem_side;
    fi_tol = elem_side;

    r_tol = 0;
    for(int n=0; n<E->lmesh.nel; n++) {
	const int m = 1;
	int gnode1 = E->IEN[E->mesh.levmax][m][n+1].node[1];
	int gnode5 = E->IEN[E->mesh.levmax][m][n+1].node[5];
	r_tol = max(r_tol, fabs(E->sx[m][3][gnode5] - E->sx[m][3][gnode1]));
    }
}


void CoarseGridMapping::findBoundaryElements(const Boundary* boundary,
					     const All_variables* E,
					     const int rank,
					     const double theta_tol,
					     const double fi_tol,
					     const double r_tol) {

    const int nsub[] = {0, 2, 3, 7,
			0, 1, 2, 5,
			4, 7, 5, 0,
			5, 7, 6, 2,
			5, 7, 2, 0};

    double xt[3], xc[24], x1[3], x2[3], x3[3], x4[3];

    for(int i=0; i<size_; i++) {
	bid2proc_[0][i] = E->parallel.nproc;  // nproc is always an invalid rank
	bid2elem_[0][i] = 0;
    }

    const double theta_max = boundary->theta_max();
    const double theta_min = boundary->theta_min();
    const double fi_max = boundary->fi_max();
    const double fi_min = boundary->fi_min();
    const double ro = boundary->ro();
    const double ri = boundary->ri();

    for(int i=0; i<size_; i++) {
	for(int j=0; j<dim_; j++) xt[j] = boundary->X(j,i);
	// loop over 5 sub tets in a brick element
        int ind = 0;

        for(int mm=1; mm<=E->sphere.caps_per_proc; mm++)
            for(int n=0; n<E->lmesh.nel; n++) {
                for(int j=0; j<8; j++) {
		    int gnode = E->IEN[E->mesh.levmax][mm][n+1].node[j+1];
                    for(int k=0; k<dim_; k++) {
                        xc[j*dim_+k] = E->sx[mm][k+1][gnode];
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
                if(in == 0)continue;
                for(int k=0; k<5; k++) {
                    for(int m=0; m<dim_; m++) {
                        x1[m] = xc[nsub[k*4]*dim_+m];
                        x2[m] = xc[nsub[k*4+1]*dim_+m];
                        x3[m] = xc[nsub[k*4+2]*dim_+m];
                        x4[m] = xc[nsub[k*4+3]*dim_+m];
                    }

		    double dett, det[4];
                    dett = TetrahedronVolume(x1,x2,x3,x4);
                    det[0] = TetrahedronVolume(x2,x4,x3,xt);
                    det[1] = TetrahedronVolume(x3,x4,x1,xt);
                    det[2] = TetrahedronVolume(x1,x4,x2,xt);
                    det[3] = TetrahedronVolume(x1,x2,x3,xt);
                    if(dett < 0) {
			std::cout << " node " << i
				  << " " << xt[0]
				  << " " << xt[1]
				  << " " << xt[2] << std::endl;
			for(int j=0; j<8; j++)
                            std::cout << xc[j*3]
				      << " " << xc[j*3+1]
				      << " " << xc[j*3+2] << std::endl;
                        const std::string msg = "Determinant evaluation is wrong ";
			std::cerr << msg << std::endl;
			throw msg;
                    }

		    if (det[0] < -1.e-10 ||
			det[1] < -1.e-10 ||
			det[2] < -1.e-10 ||
			det[3] < -1.e-10) continue;

                    ind = 1;
                    bid2elem_[0][i] = n+1;
                    bid2proc_[0][i] = rank;
                    for(int j=0; j<8; j++) shape_[0][i*8+j] = 0.0;
                    shape_[0][i*8+nsub[k*4]] = det[0]/dett;
                    shape_[0][i*8+nsub[k*4+1]] = det[1]/dett;
                    shape_[0][i*8+nsub[k*4+2]] = det[2]/dett;
                    shape_[0][i*8+nsub[k*4+3]] = det[3]/dett;
                    break;
                }
                if(ind) break;
            }
    }
    //printBid2proc();
    //printBid2elem();

    selfTest(boundary, E);

}


void CoarseGridMapping::selfTest(const Boundary* boundary,
				 const All_variables *E) const {
    double xc[24], xi[3], xt[3];

    for(int i=0; i<size_; i++) {
	if(!bid2elem_[0][i]) continue;
        for(int j=0; j<dim_; j++) xt[j] = boundary->X(j,i);

        int n1 = bid2elem_[0][i];

        for(int j=0; j<8; j++) {
            for(int k=0; k<dim_; k++) {
                xc[j*dim_+k] = E->sx[1][k+1][E->IEN[E->mesh.levmax][1][n1].node[j+1]];
            }
        }
        for(int k=0; k<dim_; k++) xi[k] = 0.0;
        for(int k=0; k<dim_; k++)
            for(int j=0; j<8; j++) {
                xi[k] += xc[j*dim_+k]*shape_[0][i*8+j];
            }

        double norm = 0.0;
        for(int k=0; k<dim_; k++) norm += (xt[k]-xi[k]) * (xt[k]-xi[k]);
        if(norm > 1.e-10) {
            double tshape = 0.0;
            for(int j=0; j<8; j++) tshape += shape_[0][i*8+j];
            std::cout << "node #" << i << " tshape = " << tshape <<std::endl;
            std::cout << xi[0] << " " << xt[0] << " "
		      << xi[1] << " " << xt[1] << " "
		      << xi[2] << " " << xt[2] << " "
		      << " norm = " << norm << std::endl;
            const std::string msg = "bid2elem interpolation functions are wrong ";
	    std::cerr << msg << std::endl;
	    throw msg;
        }
    }
}


double CoarseGridMapping::TetrahedronVolume(double *x1, double *x2,
					    double *x3, double *x4)  const
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


double CoarseGridMapping::det3_sub(double *x1, double *x2, double *x3) const
{
    return (x1[0]*(x2[1]*x3[2]-x3[1]*x2[2])
            -x1[1]*(x2[0]*x3[2]-x3[0]*x2[2])
            +x1[2]*(x2[0]*x3[1]-x3[0]*x2[1]));
}




//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


FineGridMapping::FineGridMapping(Boundary* boundary,
				 const All_variables* E,
				 const MPI_Comm comm,
				 const int rank, const int leader) :
    Mapping(boundary->dim(), boundary->size()),
    bid2gid_(size_)
{
    findBoundaryNodes(boundary, E);
}


void FineGridMapping::printBid2gid(const std::string& prefix) const {
    bid2gid_.print(prefix + "  bid");
}


void FineGridMapping::resize(const int n) {
    if (n == size_) return;

    bid2gid_.resize(n);
    Mapping::resize(n);
}


void FineGridMapping::findBoundaryNodes(Boundary* boundary,
					const All_variables* E) {

    int nodest = E->lmesh.nox * E->lmesh.noy * E->lmesh.noz;
    int *nid = new int[nodest];
    for(int i=0;i<nodest;i++) nid[i] = 0;

    int nodes = 0;

    //  for two YOZ planes

    if (E->parallel.me_loc[1]==0 || E->parallel.me_loc[1]==E->parallel.nprocx-1)
	for (int m=1;m<=E->sphere.caps_per_proc;m++)
	    for(int j=1;j<=E->lmesh.noy;j++)
		for(int i=1;i<=E->lmesh.noz;i++)  {
		    int node1 = i + (j-1)*E->lmesh.noz*E->lmesh.nox;
		    int node2 = node1 + (E->lmesh.nox-1)*E->lmesh.noz;

		    if ((E->parallel.me_loc[1]==0) && (!nid[node1-1]))  {
			for(int k=0;k<dim_;k++)
			    boundary->setX(k, nodes, E->sx[m][k+1][node1]);
			bid2gid_[0][nodes] = node1;
			nid[node1-1]++;
			nodes++;
		    }
		    if ((E->parallel.me_loc[1]==E->parallel.nprocx-1) && (!nid[node2-1])) {
			for(int k=0;k<dim_;k++)
			    boundary->setX(k, nodes, E->sx[m][k+1][node2]);
			bid2gid_[0][nodes] = node2;
			nid[node2-1]++;
			nodes++;
		    }
		}

    //  for two XOZ planes

    if (E->parallel.me_loc[2]==0 || E->parallel.me_loc[2]==E->parallel.nprocy-1)
	for (int m=1;m<=E->sphere.caps_per_proc;m++)
	    for(int j=1;j<=E->lmesh.nox;j++)
		for(int i=1;i<=E->lmesh.noz;i++)  {
		    int node1 = i + (j-1)*E->lmesh.noz;
		    int node2 = node1 + (E->lmesh.noy-1)*E->lmesh.noz*E->lmesh.nox;
		    if ((E->parallel.me_loc[2]==0) && (!nid[node1-1]))  {
			for(int k=0;k<dim_;k++)
			    boundary->setX(k, nodes, E->sx[m][k+1][node1]);
			bid2gid_[0][nodes] = node1;
			nid[node1-1]++;
			nodes++;
		    }
		    if((E->parallel.me_loc[2]==E->parallel.nprocy-1)&& (!nid[node2-1]))  {
			for(int k=0;k<dim_;k++)
			    boundary->setX(k, nodes, E->sx[m][k+1][node2]);
			bid2gid_[0][nodes] = node2;
			nid[node2-1]++;
			nodes++;
		    }
		}
    //  for two XOY planes
    if (E->parallel.me_loc[3]==0 || E->parallel.me_loc[3]==E->parallel.nprocz-1)
	for (int m=1;m<=E->sphere.caps_per_proc;m++)
	    for(int j=1;j<=E->lmesh.noy;j++)
		for(int i=1;i<=E->lmesh.nox;i++)  {
		    int node1 = 1 + (i-1)*E->lmesh.noz+(j-1)*E->lmesh.nox*E->lmesh.noz;
		    int node2 = node1 + E->lmesh.noz-1;

		    if ((E->parallel.me_loc[3]==0 ) && (!nid[node1-1])) {
			for(int k=0;k<dim_;k++)
			    boundary->setX(k, nodes, E->sx[m][k+1][node1]);
			bid2gid_[0][nodes] = node1;
			nid[node1-1]++;
			nodes++;
		    }
		    if ((E->parallel.me_loc[3]==E->parallel.nprocz-1) &&(!nid[node2-1])) {
			for(int k=0;k<dim_;k++)
			    boundary->setX(k, nodes, E->sx[m][k+1][node2]);
			bid2gid_[0][nodes] = node2;
			nid[node2-1]++;
			nodes++;
		    }
		}

    delete [] nid;
    boundary->printX();
    printBid2gid();

    if(nodes != size_) {
	if(nodes < size_) {
	    this->resize(nodes);
	    boundary->resize(nodes);
	}
	else {
	    const std::string msg = " nodes > size ";
	    std::cerr << msg << std::endl;
	    throw msg;
	}
    }
}


// version
// $Id: Mapping.cc,v 1.4 2003/10/22 01:15:47 tan2 Exp $

// End of file
