// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <portinfo>
#include <iostream>
#include "Array2D.h"
#include "auto_array_ptr.h"


template <class T, int N>
Array2D<T,N>::Array2D() :
    size_(0),
    a_(NULL)
{
    std::cout << "in Array2D<" << N << ">.c'tor()" << std::endl;
}


template <class T, int N>
Array2D<T,N>::Array2D(const int n) :
    size_(n),
    a_(new T[n*N])
{
    std::cout << "in Array2D<" << N << ">.c'tor(int)" << std::endl;
}


template <class T, int N>
Array2D<T,N>::Array2D(const Array2D<T,N>& rhs) :
    size_(rhs.size_),
    a_(new T[size_*N])
{
    std::cout << "in Array2D<" << N << ">.c'tor(Array2D)" << std::endl;
    for(int i=0; i<size_*N; i++)
	a_[i] = rhs.a_[i];
}


template <class T, int N>
Array2D<T,N>::Array2D(T* array, const int size) :
    size_(size),
    a_(array)
{
    std::cout << "in Array2D<" << N << ">.c'tor(T*,int)" << std::endl;
}


template <class T, int N>
Array2D<T,N>::~Array2D()
{
    std::cout << "in Array2D<" << N << ">.d'tor()" << std::endl;
    delete [] a_;
}


template <class T, int N>
Array2D<T,N>& Array2D<T,N>::operator=(const Array2D<T,N>& rhs) {
    std::cout << "in Array2D<" << N << ">.operator=" << std::endl;
    if(this == &rhs) return *this;  // if rhs is itself, do nothing

    if(size_ != rhs.size_) {
	// copy rhs.a_
	int n = N*rhs.size_;
	auto_array_ptr<T> tmp(new T[n]);
	for(int i=0; i<n; i++)
	    tmp[i] = rhs.a_[i];

	this->reset(tmp.release(), rhs.size_);

	return *this;
    }

    for(int i=0; i<N*size_; i++)
	a_[i] = rhs.a_[i];

    return *this;
}


template <class T, int N>
void Array2D<T,N>::resize(const int size) {
    std::cout << "in Array2D<" << N << ">.resize" << std::endl;
    if (size_ == size) return;

    T* tmp = new T[N*size];
    int n = (size < size_)? size : size_;  // find minimum
    for(int i=0; i<N*n; i++)
	tmp[i] = a_[i];

    this->reset(tmp, size);
}


template <class T, int N>
void Array2D<T,N>::send(const MPI_Comm comm, const int receiver) const {

    const int tag = 10;
    MPI_Datatype datatype = typeofT();
    int result = MPI_Send(a_, N*size_, datatype,
			  receiver, tag, comm);
    if (result != MPI_SUCCESS) {
	std::cerr << " send error!" << std::endl;
	throw result;
    }
}


template <class T, int N>
void Array2D<T,N>::receive(const MPI_Comm comm, const int sender) {

    const int tag = 10;
    MPI_Status status;
    MPI_Datatype datatype = typeofT();
    int result = MPI_Recv(a_, N*size_, datatype,
			  sender, tag, comm, &status);
    if (result != MPI_SUCCESS) {
	std::cerr << " receive error!" << std::endl;
	throw result;
    }
}


template <class T, int N>
void Array2D<T,N>::broadcast(const MPI_Comm comm, const int broadcaster) {

    MPI_Datatype datatype = typeofT();
    int result = MPI_Bcast(a_, N*size_, datatype, broadcaster, comm);
    if (result != MPI_SUCCESS) {
	std::cerr << " broadcast error!" << std::endl;
	throw result;
    }
}


template <class T, int N>
void Array2D<T,N>::print(const std::string& prefix) const {
    std::cout << "  " << prefix << ":  addr = " << a_ << std::endl;

    for (int n=0; n<size_; n++) {
	std::cout << "  " << prefix << ":  " << n << ":  ";
	for (int j=0; j<N; j++)
	    std::cout << a_[n*N + j] << "  ";
	std::cout << std::endl;
    }
}


// private functions

template <class T, int N>
void Array2D<T,N>::reset(T* array, const int size) {
    if (a_ == array) return;

    delete [] a_;
    a_ = array;
    size_ = size;
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


// friend functions

template <class T, int N>
void swap(Array2D<T,N>& lhs, Array2D<T,N>& rhs) {
    std::cout << "swapping Array2D<" << N << ">" << std::endl;

    if(&lhs == &rhs) return;

    std::swap(lhs.a_, rhs.a_);
    std::swap(lhs.size_, rhs.size_);
}



// version
// $Id: Array2D.cc,v 1.6 2003/10/22 01:13:56 tan2 Exp $

// End of file
