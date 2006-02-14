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

#if !defined(pympi_groups_h)
#define pympi_groups_h

// create a communicator group (MPI_Comm_group)
extern char pympi_groupCreate__doc__[];
extern char pympi_groupCreate__name__[];
extern "C"
PyObject * pympi_groupCreate(PyObject *, PyObject *);

// destroy a communicator group (MPI_Group_free)
extern char pympi_groupDestroy__doc__[];
extern char pympi_groupDestroy__name__[];
extern "C"
PyObject * pympi_groupDestroy(PyObject *, PyObject *);

// return the communicator group size (MPI_Group_size)
extern char pympi_groupSize__doc__[];
extern char pympi_groupSize__name__[];
extern "C"
PyObject * pympi_groupSize(PyObject *, PyObject *);

// return the process rank in a given communicator group (MPI_Group_rank)
extern char pympi_groupRank__doc__[];
extern char pympi_groupRank__name__[];
extern "C"
PyObject * pympi_groupRank(PyObject *, PyObject *);

// return the process rank in a given communicator group (MPI_Group_incl)
extern char pympi_groupInclude__doc__[];
extern char pympi_groupInclude__name__[];
extern "C"
PyObject * pympi_groupInclude(PyObject *, PyObject *);

// return the process rank in a given communicator group (MPI_Group_excl)
extern char pympi_groupExclude__doc__[];
extern char pympi_groupExclude__name__[];
extern "C"
PyObject * pympi_groupExclude(PyObject *, PyObject *);

#endif

// version
// $Id: groups.h,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $

// End of file
