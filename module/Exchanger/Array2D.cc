// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <portinfo>
#include <iostream>
#include "Array2D.h"


template <int N>
Array2D<N>::Array2D(const int n) :
    size_(n),
    a_(new double[n*N])
{}


template <int N>
Array2D<N>::Array2D(const Array2D<N>& rhs) :
    size_(rhs.size_),
    a_(new double[rhs.size_*N])
{
    for(int i=0; i<rhs.size_*N; i++)
	a_[i] = rhs.a_[i];
}


template <int N>
Array2D<N>::Array2D(auto_array_ptr<double> array, const int n) :
    size_(n),
    a_(array.release())
{}


template <int N>
Array2D<N>& Array2D<N>::operator=(const Array2D<N>& rhs) {
    std::cout << "in Array2D<" << dim << "> operator=" << std::endl;

    if(this == &rhs) return;  // if rhs is itself, do nothing

    if(size_ != rhs.size_) {
	int n = N*rhs.size_;
	double* tmp = new double[n];

	for(int i=0; i<n; i++)
	    tmp[i] = rhs.a_[i];

	a_.reset(tmp);
	size_ = rhs.size_;  // reset size
	return;
    }

    for(int i=0; i<n; i++)
	a_[i] = rhs.a_[i];
}


template <int N>
void Array2D<N>::send(const MPI_Comm comm, const int receiver) const {
    int tag = 10;
    MPI_Send(a_.get(), N*size_, MPI_DOUBLE,
	     receiver, tag, comm);
    tag ++;
    //print();
}


template <int N>
void Array2D<N>::receive(const MPI_Comm comm, const int sender) {
    int tag = 10;
    MPI_Status status;
    MPI_Recv(a_.get(), N*size_, MPI_DOUBLE,
	     sender, tag, comm, &status);
    tag ++;
    //print();
}


template <int N>
void Array2D<N>::broadcast(const MPI_Comm comm, const int broadcaster) {

    MPI_Bcast(a_.get(), N*size_, MPI_DOUBLE, broadcaster, comm);
    //print();
}


template <int N>
void Array2D<N>::print() const {
    for (int n=0; n<size_; n++) {
	std::cout << "  Array2D:  " << n << ":  ";
	for (int j=0; j<N; j++)
	    std::cout << a_[n*N + j] << "  ";
	std::cout << std::endl;
    }
}



// version
// $Id: Array2D.cc,v 1.1 2003/10/10 18:14:49 tan2 Exp $

// End of file
