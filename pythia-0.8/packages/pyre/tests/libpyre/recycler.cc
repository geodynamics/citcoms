// -*- C++ -*-
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                              Michael A.G. Aivazis
//                       California Institute of Technology
//                       (C) 1998-2005  All Rights Reserved
//
// <LicenseText>
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
#include <portinfo>

#include <cmath>
#include <iostream>
#include <iomanip>

#include "pyre/memory/FixedAllocator.h"
#include "pyre/memory/Recycler.h"

const int CAPACITY = 8; // * 1024 * 1024;
typedef pyre::memory::Recycler<> Recycler;
typedef pyre::memory::FixedAllocator<CAPACITY, sizeof(double)> FixedAllocator;

class bin_t {

    // interface
public:

    void push_back(double d) {
	double * dptr = static_cast<double *>(_bin->allocate());
	*dptr = d;
	return;
	}

    // temporary interface from bin

    double * allocate() {
	double * d_ptr = static_cast<double *>(_recycler.reuse());
	if (d_ptr) {
	    return d_ptr;
	}
	return static_cast<double *>(_bin->allocate());
    }


    double * next() {
	return static_cast<double *>(_bin->next());
    }


    void deallocate(double *dptr) {
	_recycler.recycle(dptr);
	return;
    }

public:

    ~bin_t() { delete _bin; }

    bin_t():
	_recycler(),
	_bin(new FixedAllocator)
	{ }

private:

    bin_t(const bin_t &);
    const bin_t & operator=(const bin_t &);

private:
    Recycler _recycler;
    FixedAllocator * _bin;
};

// prototypes
void fill(bin_t & bin);

// main
int main()
{
    std::cout << "sizeof(bin_t) = " << sizeof(bin_t) << std::endl;

    bin_t bin;
    double *start = (double *)bin.next();

    fill(bin);

    std::cout << "deallocating some slots" << std::endl;
    bin.deallocate(start + 3);
    bin.deallocate(start + 5);
    bin.deallocate(start + 6);

    fill(bin);
    
    return 0;
}

// helpers
void fill(bin_t & bin) {


    int i = 0;
    double * d = 0;
    std::cout << "filling the container with multiples of pi" << std::endl;
    while ((d = (double *) bin.allocate())) {

	*d = 4 * i++ * std::atan(1.0);
	std::cout << "&d = " << d 
		  << ", d = " << std::setw(15) << std::setprecision(9) << *d
		  << ", next available = " << bin.next() << std::endl;
    }

    std::cout << "-- capacity exceeded" << std::endl;

    return;
}

// version
// $Id: recycler.cc,v 1.1.1.1 2005/03/08 16:13:50 aivazis Exp $

// End of file
