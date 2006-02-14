// -*- C++ -*-
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//                               Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2005  All Rights Reserved
// 
//  <LicenseText>
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 

#if !defined(pyelc_via_mpi_h)
#define pyelc_via_mpi_h

// sendBoundaryMPI
extern char pyelc_sendBoundaryMPI__name__[];
extern char pyelc_sendBoundaryMPI__doc__[];
extern "C"
PyObject * pyelc_sendBoundaryMPI(PyObject *, PyObject *);

// sendFieldMPI
extern char pyelc_sendFieldMPI__name__[];
extern char pyelc_sendFieldMPI__doc__[];
extern "C"
PyObject * pyelc_sendFieldMPI(PyObject *, PyObject *);

// receiveBoundaryMPI
extern char pyelc_receiveBoundaryMPI__name__[];
extern char pyelc_receiveBoundaryMPI__doc__[];
extern "C"
PyObject * pyelc_receiveBoundaryMPI(PyObject *, PyObject *);

// receiveFieldMPI
extern char pyelc_receiveFieldMPI__name__[];
extern char pyelc_receiveFieldMPI__doc__[];
extern "C"
PyObject * pyelc_receiveFieldMPI(PyObject *, PyObject *);

#endif


// $Id: via_mpi.h,v 1.1.1.1 2005/03/08 16:13:29 aivazis Exp $

// End of file
