// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_utilTemplate_h)
#define pyCitcom_utilTemplate_h

#include "mpi.h"

template <class T, int N> class Array2D;

//


template <class T>
void broadcast(const MPI_Comm& comm, int broadcaster, T& data);

template <class T, int N>
void broadcast(const MPI_Comm& comm, int broadcaster, Array2D<T,N>& data);

template <class T>
void exchange(const MPI_Comm& comm, int target, T& data);

template <class T, int N>
void exchange(const MPI_Comm& comm, int target, Array2D<T,N>& data);

template <class T>
MPI_Datatype datatype(const T& data);


#include "utilTemplate.cc"

#endif

// version
// $Id: utilTemplate.h,v 1.1 2003/11/07 01:08:01 tan2 Exp $

// End of file
