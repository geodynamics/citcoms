// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include "global_defs.h"
#include "journal/journal.h"
#include "Boundary.h"


Boundary::Boundary(const All_variables* E) :
    // boundary = all - interior
    size_(E->lmesh.nno - (E->lmesh.nox-2)*(E->lmesh.noy-2)*(E->lmesh.noz-2)),
    bounds_(2),
    X_(size_)
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in Boundary::Boundary  size = " << size_ << journal::end;
    initBounds(E);
}


Boundary::Boundary(const int n) :
    size_(n),
    bounds_(2),
    X_(n)
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in Boundary::Boundary  size = " << size_ << journal::end;
}


Boundary::Boundary() :
    size_(0),
    bounds_(2)
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in Boundary::Boundary  size = " << 0 << journal::end;
}


void Boundary::initBounds(const All_variables *E) {

    bounds_[0][0] = E->control.theta_max;
    bounds_[0][1] = E->control.theta_min;
    bounds_[1][0] = E->control.fi_max;
    bounds_[1][1] = E->control.fi_min;
    bounds_[2][0] = E->sphere.ro;
    bounds_[2][1] = E->sphere.ri;

    printBounds();
}


void Boundary::send(const MPI_Comm comm, const int receiver) const {
    int tag = 0;
    MPI_Send(const_cast<int*>(&size_), 1, MPI_INT,
	     receiver, tag, comm);

    X_.send(comm, receiver);
    bounds_.send(comm, receiver);
}


void Boundary::receive(const MPI_Comm comm, const int sender) {
    int tag = 0;
    MPI_Status status;
    MPI_Recv(&size_, 1, MPI_INT,
	     sender, tag, comm, &status);

    X_.resize(size_);
    X_.receive(comm, sender);
    //printX();

    bounds_.receive(comm, sender);
    //printBounds();

}


void Boundary::broadcast(const MPI_Comm comm, const int broadcaster) {
    MPI_Bcast(&size_, 1, MPI_INT, broadcaster, comm);

    X_.resize(size_);
    X_.broadcast(comm, broadcaster);
    //printX();

    bounds_.broadcast(comm, broadcaster);
    //printBounds();
}


void Boundary::resize(const int n) {
    if (n == size_) return;

    X_.resize(n);
    size_ = n;
}


void Boundary::printBounds(const std::string& prefix) const {
    bounds_.print(prefix + "  bounds");
}


void Boundary::printX(const std::string& prefix) const {
    X_.print(prefix + "  X");
}

Boundary::Boundary(const All_variables* E, Boundary *b) :
    // boundary = all - interior
    size_(0),
    bounds_(2),
    X_(size_)
{
    int node,n,l;
    
    size_=0;
    for (int m=1;m<=E->sphere.caps_per_proc;m++)
        for(int i=1;i<=E->lmesh.nox;i++) 
	    for(int j=1;j<=E->lmesh.noy;j++)
		for(int k=1;k<=E->lmesh.noz;k++)   
                {
                    node = k + (i-1)*E->lmesh.noz+(j-1)*E->lmesh.nox*E->lmesh.noz;
                    if((E->sx[m][1][node]> b->theta_min()) &&
                       (E->sx[m][1][node]< b->theta_max()) &&
                       (E->sx[m][2][node]> b->fi_min()) &&
                       (E->sx[m][2][node]< b->fi_max()) &&
                       (E->sx[m][3][node]> b->ri()) &&
                       (E->sx[m][3][node]< b->ro()))
                    {
                        size_++;                        
                    }
                    
                }
    X_.resize(size_);
     n=0;
    for (int m=1;m<=E->sphere.caps_per_proc;m++)
        for(int i=1;i<=E->lmesh.nox;i++) 
	    for(int j=1;j<=E->lmesh.noy;j++)
		for(int k=1;k<=E->lmesh.noz;k++)   
                {
                    node = k + (i-1)*E->lmesh.noz+(j-1)*E->lmesh.nox*E->lmesh.noz;
                    if((E->sx[m][1][node]> b->theta_min()) &&
                       (E->sx[m][1][node]< b->theta_max()) &&
                       (E->sx[m][2][node]> b->fi_min()) &&
                       (E->sx[m][2][node]< b->fi_max()) &&
                       (E->sx[m][3][node]> b->ri()) &&
                       (E->sx[m][3][node]< b->ro()))
                    {
                        for(l=0;l<dim_;l++) setX(k,n,E->sx[m][l+1][node]);
                        n++;
                    }                    
                }
    if(n != size_) {
	journal::firewall_t firewall("Mapping");
	firewall << "error in CoarseGridMapping::findinteriornodes" << journal::end;
    }
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in Interior::Interior  size = " << size_ << journal::end;
    initBounds(E);
}
// version
// $Id: Boundary.cc,v 1.38 2003/10/28 02:34:37 puru Exp $

// End of file
