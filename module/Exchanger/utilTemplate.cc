// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include "Array2D.h"
#include "utilTemplate.h"


template <class T>
void broadcast(const MPI_Comm& comm, int broadcaster, T& data)
{
    MPI_Bcast(&data, 1, datatype(data), broadcaster, comm);
}


template <class T, int N>
void broadcast(const MPI_Comm& comm, int broadcaster, Array2D<T,N>& data)
{
    data.broadcast(comm, broadcaster);
}


template <class T>
void exchange(const MPI_Comm& comm, int target, T& data)
{
    const int tag = 352;
    MPI_Status status;

    MPI_Sendrecv_replace(&data, 1, datatype(data),
			 target, tag,
			 target, tag,
			 comm, &status);
}


template <class T, int N>
void exchange(const MPI_Comm& comm, int target, Array2D<T,N>& data)
{
    // non-blocking send
    MPI_Request request;
    Array2D<T,N> data2(data);
    data2.send(comm, target, request);

    // blocking receive
    data.receive(comm, target);

    data2.wait(request);
}


template <class T>
MPI_Datatype datatype(const T& data)
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
// $Id: utilTemplate.cc,v 1.1 2003/11/07 01:08:01 tan2 Exp $

// End of file
