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


#if !defined(pympi_communicator_h_)
#define pympi_communicator_h_


// create a communicator (MPI_Comm_create)
extern char pympi_communicatorCreate__doc__[];
extern char pympi_communicatorCreate__name__[];
extern "C"
PyObject * pympi_communicatorCreate(PyObject *, PyObject *);

// destroy a communicator (MPI_Comm_free)
extern char pympi_communicatorDestroy__doc__[];
extern char pympi_communicatorDestroy__name__[];
extern "C"
PyObject * pympi_communicatorDestroy(PyObject *, PyObject *);

// return the communicator size (MPI_Comm_size)
extern char pympi_communicatorSize__doc__[];
extern char pympi_communicatorSize__name__[];
extern "C"
PyObject * pympi_communicatorSize(PyObject *, PyObject *);

// return the process rank in a given communicator (MPI_Comm_rank)
extern char pympi_communicatorRank__doc__[];
extern char pympi_communicatorRank__name__[];
extern "C"
PyObject * pympi_communicatorRank(PyObject *, PyObject *);

// set a communicator barrier (MPI_Barrier)
extern char pympi_communicatorBarrier__doc__[];
extern char pympi_communicatorBarrier__name__[];
extern "C"
PyObject * pympi_communicatorBarrier(PyObject *, PyObject *);

// create a cartesian communicator (MPI_Cart_create)
extern char pympi_communicatorCreateCartesian__doc__[];
extern char pympi_communicatorCreateCartesian__name__[];
extern "C"
PyObject * pympi_communicatorCreateCartesian(PyObject *, PyObject *);

// return the coordinates of the process in the cartesian communicator (MPI_Cart_coords)
extern char pympi_communicatorCartesianCoordinates__doc__[];
extern char pympi_communicatorCartesianCoordinates__name__[];
extern "C"
PyObject * pympi_communicatorCartesianCoordinates(PyObject *, PyObject *);

#endif

// version
// $Id: communicators.h,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $

// End of file
