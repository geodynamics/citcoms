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
bool Array2D<T,N>::empty() const {
    return (a_.size() == 0);
}


template <class T, int N>
void Array2D<T,N>::sendSize(const MPI_Comm comm, const int receiver) const {

    const int tag = 10;
    int n = size();
    int result = MPI_Send(&n, 1, MPI_INT,
			  receiver, tag, comm);
    if (result != MPI_SUCCESS) {
        journal::error_t error("Array2D");
        error << journal::loc(__HERE__)
              << " sendSize error!" << journal::end;
	throw result;
    }
}


template <class T, int N>
int Array2D<T,N>::receiveSize(const MPI_Comm comm, const int sender) const {

    const int tag = 10;
    MPI_Status status;
    int n;
    int result = MPI_Recv(&n, 1, MPI_INT,
			  sender, tag, comm, &status);
    if (result != MPI_SUCCESS) {
        journal::error_t error("Array2D");
        error << journal::loc(__HERE__)
              << " receiveSize error!" << journal::end;
	throw result;
    }
    return n;
}


template <class T, int N>
int Array2D<T,N>::broadcastSize(const MPI_Comm comm, const int broadcaster) const {
    int n = size();
    int result = MPI_Bcast(&n, 1, MPI_INT, broadcaster, comm);
    if (result != MPI_SUCCESS) {
        journal::error_t error("Array2D");
        error << journal::loc(__HERE__)
              << " receiveSize error!" << journal::end;
	throw result;
    }
    return n;
}


template <class T, int N>
void Array2D<T,N>::send(const MPI_Comm comm, const int receiver) const {

    sendSize(comm, receiver);

    const int tag = 11;
    MPI_Datatype datatype = typeofT();
    int result = MPI_Send(const_cast<T*>(&a_[0]), a_.size(), datatype,
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

    // resize to accommodate incoming data
    resize(receiveSize(comm, sender));

    const int tag = 11;
    MPI_Status status;
    MPI_Datatype datatype = typeofT();
    int result = MPI_Recv(&a_[0], a_.size(), datatype,
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

    // resize to accommodate incoming data
    resize(broadcastSize(comm, broadcaster));

    MPI_Datatype datatype = typeofT();
    int result = MPI_Bcast(&a_[0], a_.size(), datatype, broadcaster, comm);
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

    journal::firewall_t firewall("Array2D");
    firewall << journal::loc(__HERE__)
             << "unexpected Array2D datatype" << journal::end;
    // firewall will throw an exception or terminate the job
    return 0;
}



// version
// $Id: Array2D.cc,v 1.9 2003/10/28 19:57:45 tan2 Exp $

// End of file
