// -*- C++ -*-
//
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                              Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2005 All Rights Reserved
//
// <LicenseText>
//
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pympi_Communicator_h__)
#define pympi_Communicator_h__

namespace mpi {

    class Group;
    class Communicator;
}

class mpi::Communicator {
// interface
public:
    int size() const;
    int rank() const;

    void barrier() const;
    void cartesianCoordinates(int rank, int dim, int * coordinates) const;

    MPI_Comm handle() const;

    // factories
    Communicator * communicator(const Group & group) const;
    Communicator * cartesian(int size, int * procs, int * periods, int reorder) const;


// meta-methods
public:
    Communicator(MPI_Comm handle);
    virtual ~Communicator();

// hide these
private:
    Communicator(const Communicator &);
    Communicator & operator=(const Communicator &);

// instance atributes
protected:

    MPI_Comm _communicator;
};

#endif

// version
// $Id: Communicator.h,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $

//
// End of file
