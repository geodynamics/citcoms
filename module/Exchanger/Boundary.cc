// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <iostream>
#include <memory>
#include "global_defs.h"
#include "Array2D.cc"
#include "Boundary.h"


Boundary::Boundary(const All_variables* E) :
    size_(E->mesh.nno - (E->mesh.nox-2)*(E->mesh.noy-2)*(E->mesh.noz-2)),
    bounds_(2),
    X_(size_)
{
    std::cout << "in Boundary::Boundary  size = " << size_ << std::endl;
    initBounds(E);
}


Boundary::Boundary(const int n) :
    size_(n),
    bounds_(2),
    X_(n)
{
    std::cout << "in Boundary::Boundary  size = " << size_ << std::endl;
}


Boundary::Boundary() :
    size_(0),
    bounds_(2)
{
    std::cout << "in Boundary::Boundary  size = " << 0 << std::endl;
}


void Boundary::initBounds(const All_variables *E) {

    bounds_[0][0] = E->control.theta_max;
    bounds_[0][1] = E->control.theta_min;
    bounds_[1][0] = E->control.fi_max;
    bounds_[1][1] = E->control.fi_min;
    bounds_[2][0] = E->sphere.ro;
    bounds_[2][1] = E->sphere.ri;
    //printBounds();
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


void Boundary::printBounds(const std::string& prefix) const {
    bounds_.print(prefix + "  bounds");
}


void Boundary::printX(const std::string& prefix) const {
    X_.print(prefix + "  X");
}


// version
// $Id: Boundary.cc,v 1.35 2003/10/20 17:13:08 tan2 Exp $

// End of file
