// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include "auto_array_ptr.h"
#include "journal/journal.h"


template <class T, int N>
Array2D<T,N>::Array2D() :
    memsize_(0),
    size_(0),
    a_(NULL)
{
    journal::debug_t debug("Array2D");
    debug << journal::loc(__HERE__)
	  << "in Array2D<" << N << ">.c'tor()" << journal::end;
}


template <class T, int N>
Array2D<T,N>::Array2D(const int n) :
    memsize_(n),
    size_(n),
    a_(new T[n*N])
{
    journal::debug_t debug("Array2D");
    debug << journal::loc(__HERE__)
	  << "in Array2D<" << N << ">.c'tor(int)" << journal::end;
}


template <class T, int N>
Array2D<T,N>::Array2D(const Array2D<T,N>& rhs) :
    memsize_(rhs.size_),
    size_(rhs.size_),
    a_(new T[size_*N])
{
    journal::debug_t debug("Array2D");
    debug << journal::loc(__HERE__)
	  << "in Array2D<" << N << ">.c'tor(Array2D)" << journal::end;
    for(int i=0; i<size_*N; i++)
	a_[i] = rhs.a_[i];
}


template <class T, int N>
Array2D<T,N>::Array2D(T* array, const int size) :
    memsize_(size),
    size_(size),
    a_(array)
{
    journal::debug_t debug("Array2D");
    debug << journal::loc(__HERE__)
	  << "in Array2D<" << N << ">.c'tor(T*,int)" << journal::end;
}


template <class T, int N>
Array2D<T,N>::~Array2D()
{
    journal::debug_t debug("Array2D");
    debug << journal::loc(__HERE__)
	  << "in Array2D<" << N << ">.d'tor()" << journal::end;
    delete [] a_;
}


template <class T, int N>
Array2D<T,N>& Array2D<T,N>::operator=(const Array2D<T,N>& rhs) {
    journal::debug_t debug("Array2D");
    debug << journal::loc(__HERE__)
	  << "in Array2D<" << N << ">.operator=" << journal::end;
    if(this == &rhs) return *this;  // if rhs is itself, do nothing

    if(memsize_ < rhs.size_) {
	// copy rhs.a_
	int n = N*rhs.size_;
	auto_array_ptr<T> tmp(new T[n]);
	for(int i=0; i<n; i++)
	    tmp[i] = rhs.a_[i];

	this->reset(tmp.release(), rhs.size_);

	return *this;
    }

    size_ = rhs.size_;
    for(int i=0; i<N*size_; i++)
	a_[i] = rhs.a_[i];

    return *this;
}


template <class T, int N>
void Array2D<T,N>::resize(const int size) {
    journal::debug_t debug("Array2D");
    debug << journal::loc(__HERE__)
	  << "in Array2D<" << N << ">.resize" << journal::end;
    if (size_ == size) return;

    if (size_ > size)
	downsize(size);
    else
	upsize(size);
}


template <class T, int N>
void Array2D<T,N>::shrink() {
    // shrink memory so that memsize_ == size_
    if (memsize_ == size_) return;

    T* tmp = new T[N*size_];
    for(int i=0; i<N*size_; i++)
	tmp[i] = a_[i];

    this->reset(tmp, size_);
}


template <class T, int N>
void Array2D<T,N>::send(const MPI_Comm comm, const int receiver) const {

    const int tag = 10;
    MPI_Datatype datatype = typeofT();
    int result = MPI_Send(a_, N*size_, datatype,
			  receiver, tag, comm);
    if (result != MPI_SUCCESS) {
	journal::error_t error("Array2D");
	error << journal::loc(__HERE__)
	      << " send error!" << journal::end;
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
	journal::error_t error("Array2D");
	error << journal::loc(__HERE__)
	      << " receive error!" << journal::end;
	throw result;
    }
}


template <class T, int N>
void Array2D<T,N>::broadcast(const MPI_Comm comm, const int broadcaster) {

    MPI_Datatype datatype = typeofT();
    int result = MPI_Bcast(a_, N*size_, datatype, broadcaster, comm);
    if (result != MPI_SUCCESS) {
	journal::error_t error("Array2D");
	error << journal::loc(__HERE__)
	      << " broadcast error!" << journal::end;
	throw result;
    }
}


template <class T, int N>
void Array2D<T,N>::print(const std::string& prefix) const {
    journal::info_t info(prefix);
    info << "  " << prefix << ":  addr = " << a_;

    for (int n=0; n<size_; n++) {
	info <<  journal::newline << "  " << prefix << ":  " << n << ":  ";
	for (int j=0; j<N; j++)
	    info << a_[n*N + j] << "  ";
    }
    info << journal::newline << journal::end;
}


// private functions

template <class T, int N>
void Array2D<T,N>::upsize(const int size) {
    if (memsize_ >= size)
	size_ = size;
    else {
	T* tmp = new T[N*size];
	this->reset(tmp, size);
    }
}


template <class T, int N>
void Array2D<T,N>::downsize(const int size) {
    size_ = size;
}


template <class T, int N>
void Array2D<T,N>::reset(T* array, const int size) {
    if (a_ == array) return;

    delete [] a_;
    a_ = array;
    memsize_ = size;
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

    journal::firewall_t firewall("Array2D");
    firewall << journal::loc(__HERE__)
	     << "unexpected Array2D datatype" << journal::end;
    // firewall will throw an exception or terminate the job
    return 0;
}


// friend functions

template <class T, int N>
void swap(Array2D<T,N>& lhs, Array2D<T,N>& rhs) {
    journal::debug_t debug("Array2D");
    debug << journal::loc(__HERE__)
	  << "swapping Array2D<" << N << ">" << journal::end;

    if(&lhs == &rhs) return;

    std::swap(lhs.a_, rhs.a_);
    std::swap(lhs.memsize_, rhs.memsize_);
    std::swap(lhs.size_, rhs.size_);
}



// version
// $Id: Array2D.cc,v 1.7 2003/10/24 04:51:53 tan2 Exp $

// End of file
