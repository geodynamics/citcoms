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

namespace util {

    template <class T>
    void broadcast(const MPI_Comm& comm, int broadcaster, T& data);

    template <class T, int N>
    void broadcast(const MPI_Comm& comm, int broadcaster, Array2D<T,N>& data);

    template <class T>
    void exchange(const MPI_Comm& comm, int target, T& data);

    template <class T, int N>
    void exchange(const MPI_Comm& comm, int target, Array2D<T,N>& data);

    template <class T>
    void gatherSum(const MPI_Comm& comm, T& data);

    template <class T>
    inline MPI_Datatype datatype(const T&);

}

#include "utilTemplate.cc"


#endif

// version
// $Id: utilTemplate.h,v 1.2 2003/11/10 21:55:28 tan2 Exp $

// End of file
