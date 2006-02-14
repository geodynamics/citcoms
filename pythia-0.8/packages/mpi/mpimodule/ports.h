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

#if !defined(pympi_ports_h)
#define pympi_ports_h

// send a string
extern char pympi_sendString__doc__[];
extern char pympi_sendString__name__[];
extern "C"
PyObject * pympi_sendString(PyObject *, PyObject *);

// receive a string
extern char pympi_receiveString__doc__[];
extern char pympi_receiveString__name__[];
extern "C"
PyObject * pympi_receiveString(PyObject *, PyObject *);

#endif

// version
// $Id: ports.h,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $

// End of file
