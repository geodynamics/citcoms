// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include "journal/journal.h"
#include "utility.h"
#include "Array2D.h"


template <class T>
void util::broadcast(const MPI_Comm& comm, int broadcaster, T& data)
{
    int size;
    MPI_Comm_size(comm, &size);
    if(size == 1) return;

    int result = MPI_Bcast(&data, 1, datatype(data), broadcaster, comm);
    testResult(result, "broadcast error!");
}


template <class T, int N>
void util::broadcast(const MPI_Comm& comm, int broadcaster, Array2D<T,N>& data)
{
    int size;
    MPI_Comm_size(comm, &size);
    if(size == 1) return;

    data.broadcast(comm, broadcaster);
}


template <class T>
void util::exchange(const MPI_Comm& comm, int target, T& data)
{
    const int tag = 352;
    MPI_Status status;

    int result = MPI_Sendrecv_replace(&data, 1, datatype(data),
				      target, tag,
				      target, tag,
				      comm, &status);
    testResult(result, "exchange error!");
}


template <class T, int N>
void util::exchange(const MPI_Comm& comm, int target, Array2D<T,N>& data)
{
    // non-blocking send
    MPI_Request request;
    Array2D<T,N> data2(data);
    data2.send(comm, target, request);

    // blocking receive
    data.receive(comm, target);

    waitRequest(request);
}


template <class T>
void util::gatherSum(const MPI_Comm& comm, T& data)
{
    int size;
    MPI_Comm_size(comm, &size);
    if(size == 1) return;

    const int root = 0;
    T tmp(data);
    int result = MPI_Reduce(&tmp, &data, 1, datatype(data),
			    MPI_SUM, root, comm);
    testResult(result, "gatherSum error!");

    broadcast(comm, root, data);
}


template <class T>
MPI_Datatype util::datatype(const T&)
{
    if (typeid(T) == typeid(double))
	return MPI_DOUBLE;

    if (typeid(T) == typeid(float))
	return MPI_FLOAT;

    if (typeid(T) == typeid(int))
	return MPI_INT;

    if (typeid(T) == typeid(char))
	return MPI_CHAR;

    journal::firewall_t firewall("utilTemplate");
    firewall << journal::loc(__HERE__)
             << "unexpected datatype" << journal::end;
    throw std::domain_error("unexpected datatype");
}



// version
// $Id: utilTemplate.cc,v 1.2 2003/11/10 21:55:28 tan2 Exp $

// End of file
