// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <iostream>
#include "auto_array_ptr.h"
#include "journal/journal.h"


template <class T, int N>
Array2D<T,N>::Array2D() :
    a_()
{
    journal::debug_t debug("Array2D");
    debug << journal::loc(__HERE__)
          << "in Array2D<" << N << ">.c'tor()" << journal::end;
}


template <class T, int N>
Array2D<T,N>::Array2D(const int n) :
    a_(n*N)
{
    journal::debug_t debug("Array2D");
    debug << journal::loc(__HERE__)
          << "in Array2D<" << N << ">.c'tor(int)" << journal::end;
}


template <class T, int N>
Array2D<T,N>::Array2D(const Array2D<T,N>& rhs) :
    a_(rhs.a_)
{
    journal::debug_t debug("Array2D");
    debug << journal::loc(__HERE__)
          << "in Array2D<" << N << ">.c'tor(Array2D)" << journal::end;
}


template <class T, int N>
Array2D<T,N>::~Array2D()
{
    journal::debug_t debug("Array2D");
    debug << journal::loc(__HERE__)
          << "in Array2D<" << N << ">.d'tor()" << journal::end;
}


template <class T, int N>
Array2D<T,N>& Array2D<T,N>::operator=(const Array2D<T,N>& rhs) {
    journal::debug_t debug("Array2D");
    debug << journal::loc(__HERE__)
          << "in Array2D<" << N << ">.operator=" << journal::end;

    if(this != &rhs)
	a_ = rhs.a_;
    return *this;
}


template <class T, int N>
void Array2D<T,N>::swap(Array2D<T,N>& rhs) {
    journal::debug_t debug("Array2D");
    debug << journal::loc(__HERE__)
          << "in Array2D<" << N << ">.swap" << journal::end;

    if(this != &rhs)
	a_.swap(rhs.a_);
}


template <class T, int N>
void Array2D<T,N>::reserve(const int n) {
    journal::debug_t debug("Array2D");
    debug << journal::loc(__HERE__)
          << "in Array2D<" << N << ">.reserve" << journal::end;
    a_.reserve(N*n);
}


template <class T, int N>
void Array2D<T,N>::resize(const int n) {
    journal::debug_t debug("Array2D");
    debug << journal::loc(__HERE__)
          << "in Array2D<" << N << ">.resize" << journal::end;
    a_.resize(N*n);
}


template <class T, int N>
void Array2D<T,N>::shrink() {
    journal::debug_t debug("Array2D");
    debug << journal::loc(__HERE__)
          << "in Array2D<" << N << ">.shrink" << journal::end;
    std::vector<T>(a_).swap(a_);
}


template <class T, int N>
int Array2D<T,N>::size() const {
    return a_.size()/N;
}


template <class T, int N>
void Array2D<T,N>::send(const MPI_Comm comm, const int receiver) const {

    const int tag = 10;
    MPI_Datatype datatype = typeofT();
    int result = MPI_Send(const_cast<T*>(&a_[0]), a_.size(), datatype,
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
    int result = MPI_Recv(&a_[0], a_.size(), datatype,
			  sender, tag, comm, &status);
    if (result != MPI_SUCCESS) {
	std::cerr << " receive error!" << std::endl;
	throw result;
    }
}


template <class T, int N>
void Array2D<T,N>::broadcast(const MPI_Comm comm, const int broadcaster) {

    MPI_Datatype datatype = typeofT();
    int result = MPI_Bcast(&a_[0], a_.size(), datatype, broadcaster, comm);
    if (result != MPI_SUCCESS) {
	std::cerr << " broadcast error!" << std::endl;
	throw result;
    }
}


template <class T, int N>
void Array2D<T,N>::print(const std::string& prefix) const {
    journal::info_t info(prefix);
    info << "  " << prefix << ":  addr = " << &a_;

    for (int n=0; n<size(); n++) {
        info <<  journal::newline << "  " << prefix << ":  " << n << ":  ";
        for (int j=0; j<N; j++)
            info << a_[n*N + j] << "  ";
    }
    info << journal::newline << journal::end;
}


// private functions


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
// $Id: Array2D.cc,v 1.8 2003/10/28 01:51:05 tan2 Exp $

// End of file
