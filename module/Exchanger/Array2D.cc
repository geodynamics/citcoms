// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <algorithm>
#include <iostream>
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
Array2D<T,N>::Array2D(int n) :
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
Array2D<T,N>& Array2D<T,N>::operator=(const Array2D<T,N>& rhs)
{
    journal::debug_t debug("Array2D");
    debug << journal::loc(__HERE__)
          << "in Array2D<" << N << ">.operator=" << journal::end;

    if(this != &rhs)
	a_ = rhs.a_;
    return *this;
}


template <class T, int N>
void Array2D<T,N>::swap(Array2D<T,N>& rhs)
{
    journal::debug_t debug("Array2D");
    debug << journal::loc(__HERE__)
          << "in Array2D<" << N << ">.swap" << journal::end;

    if(this != &rhs)
	a_.swap(rhs.a_);
}


template <class T, int N>
void Array2D<T,N>::reserve(int n)
{
    journal::debug_t debug("Array2D");
    debug << journal::loc(__HERE__)
          << "in Array2D<" << N << ">.reserve" << journal::end;
    a_.reserve(N*n);
}


template <class T, int N>
void Array2D<T,N>::resize(int n)
{
    journal::debug_t debug("Array2D");
    debug << journal::loc(__HERE__)
          << "in Array2D<" << N << ">.resize" << journal::end;
    a_.resize(N*n);
}


template <class T, int N>
void Array2D<T,N>::shrink()
{
    journal::debug_t debug("Array2D");
    debug << journal::loc(__HERE__)
          << "in Array2D<" << N << ">.shrink" << journal::end;
    std::vector<T>(a_).swap(a_);
}


template <class T, int N>
int Array2D<T,N>::size() const
{
    return a_.size()/N;
}


template <class T, int N>
int Array2D<T,N>::capacity() const
{
    return a_.capacity()/N;
}


template <class T, int N>
bool Array2D<T,N>::empty() const
{
    return (a_.size() == 0);
}


template <class T, int N>
void Array2D<T,N>::push_back(const std::vector<T>& val)
{
#ifdef DEBUG
    if (val.size() != N) {
	journal::firewall_t firewall("Array2D");
	firewall << journal::loc(__HERE__)
		 << "Array2D: push_back element out of range" << journal::end;
	throw std::out_of_range("Array2D");
    }
#endif
    reserve(size()+1);
    copy(val.begin(), val.end(), back_inserter(a_));
}


template <class T, int N>
void Array2D<T,N>::push_back(const T& val)
{
    reserve(size()+1);
    fill_n(back_inserter(a_), N, val);
}


template <class T, int N>
void Array2D<T,N>::sendSize(const MPI_Comm& comm, int receiver) const
{
    sendSize(comm, receiver, size());
}


template <class T, int N>
void Array2D<T,N>::sendSize(const MPI_Comm& comm, int receiver, int n) const
{
    int result = MPI_Send(&n, 1, MPI_INT,
			  receiver, SIZETAG_, comm);
    testResult(result, "sendSize error!");
}


template <class T, int N>
int Array2D<T,N>::receiveSize(const MPI_Comm& comm, int sender) const
{
    int n;
    MPI_Status status;
    int result = MPI_Recv(&n, 1, MPI_INT,
			  sender, SIZETAG_, comm, &status);
    testResult(result, "receiveSize error!");
    return n;
}


template <class T, int N>
int Array2D<T,N>::broadcastSize(const MPI_Comm& comm, int broadcaster) const
{
    int n = size();
    int result = MPI_Bcast(&n, 1, MPI_INT, broadcaster, comm);
    testResult(result, "broadcastSize error!");
    return n;
}


template <class T, int N>
void Array2D<T,N>::send(const MPI_Comm& comm, int receiver) const
    // blocking send the whole vector
{
    sendSize(comm, receiver);

    MPI_Datatype datatype = typeofT();
    int result = MPI_Send(const_cast<T*>(&a_[0]), a_.size(), datatype,
			  receiver, TAG_, comm);
    testResult(result, "send error!");
}


template <class T, int N>
void Array2D<T,N>::send(const MPI_Comm& comm, int receiver,
			MPI_Request& request) const
    // non-blocking send the whole vector
{
    sendSize(comm, receiver, size());
    send(comm, receiver, 0, size(), request);
}


template <class T, int N>
void Array2D<T,N>::send(const MPI_Comm& comm, int receiver,
			int begin, int sendsize, MPI_Request& request) const
    // non-blocking send the vector[begin ~ begin+sendsize-1]
    // the caller must guarantee a_ is of sufficent size
{
    MPI_Datatype datatype = typeofT();
    int result = MPI_Isend(const_cast<T*>(&a_[begin*N]), sendsize*N, datatype,
			   receiver, TAG_, comm, &request);
    testResult(result, "send error!");
}


template <class T, int N>
void Array2D<T,N>::receive(const MPI_Comm& comm, int sender)
    // blocking receive the whole vector
{
    // resize to accommodate incoming data
    resize(receiveSize(comm, sender));

    MPI_Status status;
    MPI_Datatype datatype = typeofT();
    int result = MPI_Recv(&a_[0], a_.size(), datatype,
			  sender, TAG_, comm, &status);
    testResult(result, "receive error!");
}


template <class T, int N>
void Array2D<T,N>::receive(const MPI_Comm& comm, int sender,
			   MPI_Request& request)
    // non-blocking receive the whole vector
{
    resize(receiveSize(comm, sender));
    receive(comm, sender, 0, size(), request);
}


template <class T, int N>
void Array2D<T,N>::receive(const MPI_Comm& comm, int sender,
			   int begin, int recvsize, MPI_Request& request)
    // non-blocking receive the vector[begin ~ begin+recvsize-1]
    // the caller must guarantee a_ is of sufficent size
{
    MPI_Datatype datatype = typeofT();
    int result = MPI_Irecv(&a_[begin*N], recvsize*N, datatype,
			   sender, TAG_, comm, &request);
    testResult(result, "receive error!");
}


template <class T, int N>
void Array2D<T,N>::broadcast(const MPI_Comm& comm, int broadcaster)
{
    // resize to accommodate incoming data
    resize(broadcastSize(comm, broadcaster));

    MPI_Datatype datatype = typeofT();
    int result = MPI_Bcast(&a_[0], a_.size(), datatype, broadcaster, comm);
    testResult(result, "broadcast error!");
}


template <class T, int N>
void Array2D<T,N>::print(const std::string& prefix) const
{
    journal::info_t info(prefix);
    info << "  " << prefix << ":  addr = " << &a_;

    for (int n=0; n<size(); n++) {
        info <<  journal::newline << "  " << prefix << ":  " << n << ":  ";
        for (int j=0; j<N; j++)
            info << a_[n*N + j] << "  ";
    }
    info << journal::newline << journal::end;
}


template <class T, int N>
typename Array2D<T,N>::Array1D
Array2D<T,N>::operator[](size_t index)
{
#ifdef DEBUG
    if (index >= N) {
	journal::firewall_t firewall("Array2D");
	firewall << journal::loc(__HERE__)
		 << "Array2D: first index out of range" << journal::end;
	throw std::out_of_range("Array2D");
    }
#endif
    return Array1D(a_, index);
}


template <class T, int N>
const typename Array2D<T,N>::Array1D
Array2D<T,N>::operator[](size_t index) const
{
#ifdef DEBUG
    if (index >= N) {
	journal::firewall_t firewall("Array2D");
	firewall << journal::loc(__HERE__)
		 << "Array2D: first index out of range" << journal::end;
	throw std::out_of_range("Array2D");
    }
#endif
    return Array1D(const_cast<std::vector<T>&>(a_), index);
}


// Proxy class


template <class T, int N>
Array2D<T,N>::Array1D::Array1D(std::vector<T>& a, size_t n) :
    p_(a),
    n_(n)
{}


template <class T, int N>
T& Array2D<T,N>::Array1D::operator[](size_t index)
{
#ifdef DEBUG
    if (index*N+n_ >= p_.size()) {
	journal::firewall_t firewall("Array2D");
	firewall << journal::loc(__HERE__)
		 << "Array2D: second index out of range" << journal::end;
	throw std::out_of_range("Array2D");
    }
#endif
    return p_[index*N+n_];
}


template <class T, int N>
const T& Array2D<T,N>::Array1D::operator[](size_t index) const
{
#ifdef DEBUG
    if (index*N+n_ >= p_.size()) {
	journal::firewall_t firewall("Array2D");
	firewall << journal::loc(__HERE__)
		 << "Array2D: second index out of range" << journal::end;
	throw std::out_of_range("Array2D");
    }
#endif
    return p_[index*N+n_];
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


template <class T, int N>
void Array2D<T,N>::testResult(int result, const std::string& errmsg)
{
    if (result != MPI_SUCCESS) {
        journal::error_t error("Array2D");
        error << journal::loc(__HERE__)
              << errmsg << journal::end;
	throw result;
    }
}


// version
// $Id: Array2D.cc,v 1.11 2003/10/30 22:45:37 tan2 Exp $

// End of file
