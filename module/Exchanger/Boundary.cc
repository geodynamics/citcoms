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
#include "Boundary.h"


Boundary::Boundary(const int n) : size_(n) {
    std::cout << "in Boundary::Boundary  size = " << size_ << std::endl;

    for(int i=0; i<dim; i++) {
	auto_array_ptr<double> tmp(new double[size_]);
	X[i] = tmp;
    }
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


void Boundary::send(const MPI_Comm comm, const int receiver) const {
    int tag = 1;

    for (int i=0; i<dim; i++) {
	MPI_Send(X[i].get(), size_, MPI_DOUBLE,
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
}


void Boundary::receive(const MPI_Comm comm, const int sender) {
    MPI_Status status;
    int tag = 1;

    for (int i=0; i<dim; i++) {
	MPI_Recv(X[i].get(), size_, MPI_DOUBLE,
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
}


void Boundary::broadcast(const MPI_Comm comm, const int broadcaster) {

    for (int i=0; i<dim; i++) {
      MPI_Bcast(X[i].get(), size_, MPI_DOUBLE, broadcaster, comm);
    }
    //printX();

    MPI_Bcast(&theta_max, 1, MPI_DOUBLE, broadcaster, comm);
    MPI_Bcast(&theta_min, 1, MPI_DOUBLE, broadcaster, comm);
    MPI_Bcast(&fi_max, 1, MPI_DOUBLE, broadcaster, comm);
    MPI_Bcast(&fi_min, 1, MPI_DOUBLE, broadcaster, comm);
    MPI_Bcast(&ro, 1, MPI_DOUBLE, broadcaster, comm);
    MPI_Bcast(&ri, 1, MPI_DOUBLE, broadcaster, comm);
    //printBound();
}


void Boundary::printBound() const {
    std::cout << "theta= " << theta_min
	      << " : " << theta_max << std::endl;
    std::cout << "fi   = " << fi_min
	      << " : " << fi_max << std::endl;
    std::cout << "r    = " << ri
	      << " : " << ro  << std::endl;
}


void Boundary::printX() const {
    for(int j=0; j<size_; j++) {
	std::cout << "  X:  " << j << ":  ";
	    for(int i=0; i<dim; i++)
		std::cout << X[i][j] << " ";
	std::cout << std::endl;
    }
}


// version
// $Id: Boundary.cc,v 1.33 2003/10/11 00:38:46 tan2 Exp $

// End of file
