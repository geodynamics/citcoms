// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <portinfo>
#include <iostream>
#include "Array2D.h"


template <class T, int N>
Array2D<T,N>::Array2D(const int n) :
    size_(n),
    a_(new T[n*N])
{}


template <class T, int N>
Array2D<T,N>::Array2D(const Array2D<T,N>& rhs) :
    size_(rhs.size_),
    a_(new T[size_*N])
{
    for(int i=0; i<size_*N; i++)
	a_[i] = rhs.a_[i];
}


template <class T, int N>
Array2D<T,N>::Array2D(T* array, const int size) :
    size_(size),
    a_(array)
{}


template <class T, int N>
Array2D<T,N>::~Array2D()
{
    delete [] a_;
}


template <class T, int N>
Array2D<T,N>& Array2D<T,N>::operator=(const Array2D<T,N>& rhs) {
    std::cout << "in Array2D<" << dim << "> operator=" << std::endl;

    if(this == &rhs) return;  // if rhs is itself, do nothing

    if(size_ != rhs.size_) {
	int n = N*rhs.size_;
	T* tmp = new T[n];

	for(int i=0; i<n; i++)
	    tmp[i] = rhs.a_[i];

	delete [] a_;
	a_ = tmp;
	size_ = rhs.size_;
	return;
    }

    for(int i=0; i<N*size_; i++)
	a_[i] = rhs.a_[i];
}


template <class T, int N>
void Array2D<T,N>::send(const MPI_Comm comm, const int receiver) const {

    const int tag = 10;
    MPI_Datatype datatype = typeofT();
    int ok = MPI_Send(a_, N*size_, datatype,
		      receiver, tag, comm);
    if (ok != MPI_SUCCESS) {
	std::cerr << " send error!" << std::endl;
	throw ok;
    }
}


template <class T, int N>
void Array2D<T,N>::receive(const MPI_Comm comm, const int sender) {

    const int tag = 10;
    MPI_Status status;
    MPI_Datatype datatype = typeofT();
    int ok = MPI_Recv(a_, N*size_, datatype,
		      sender, tag, comm, &status);
    if (ok != MPI_SUCCESS) {
	std::cerr << " receive error!" << std::endl;
	throw ok;
    }
}


template <class T, int N>
void Array2D<T,N>::broadcast(const MPI_Comm comm, const int broadcaster) {

    MPI_Datatype datatype = typeofT();
    int ok = MPI_Bcast(a_, N*size_, datatype, broadcaster, comm);
    if (ok != MPI_SUCCESS) {
	std::cerr << " broadcast error!" << std::endl;
	throw ok;
    }
}


template <class T, int N>
void Array2D<T,N>::print(const std::string& prefix) const {
    for (int n=0; n<size_; n++) {
	std::cout << "  " << prefix << ":  " << n << ":  ";
	for (int j=0; j<N; j++)
	    std::cout << a_[n*N + j] << "  ";
	std::cout << std::endl;
    }
}


template <class T, int N>
MPI_Datatype Array2D<T,N>::typeofT() {

    if (typeid(T) == typeid(double))
	return MPI_DOUBLE;

    if (typeid(T) == typeid(float))
	return MPI_FLOAT;

    if (typeid(T) == typeid(int))
	return MPI_INT;

    if (typeid(T) == typeid(char))
	return MPI_CHAR;

    const std::string msg = "unexpected Array2D datatype";
    std::cerr << msg << std::endl;
    throw msg;
}

// version
// $Id: Array2D.cc,v 1.3 2003/10/16 20:06:02 tan2 Exp $

// End of file
