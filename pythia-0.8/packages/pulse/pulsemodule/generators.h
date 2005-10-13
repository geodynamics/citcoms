// -*- C++ -*-
//
//-----------------------------------------------------------------------------
//
//                              Michael A.G. Aivazis
//                       California Institute of Technology
//                       (C) 1998-2005  All Rights Reserved
//
// <LicenseText>
//
//-----------------------------------------------------------------------------
//

#if !defined(pypulse_generators_h)
#define pypulse_generators_h

// the routine that paints a heaviside pressure pulse
extern char pypulse_heaviside__doc__[];
extern char pypulse_heaviside__name__[];
extern "C"
PyObject * pypulse_heaviside(PyObject *, PyObject *);

// pressure as a function of depth
extern char pypulse_bath__doc__[];
extern char pypulse_bath__name__[];
extern "C"
PyObject * pypulse_bath(PyObject *, PyObject *);


#endif // pypulse_generators_h

// $Id: generators.h,v 1.1.1.1 2005/03/08 16:13:57 aivazis Exp $

// End of file
