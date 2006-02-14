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

#include <portinfo>
#include <Python.h>

#include "facility.h"

#include "journal/diagnostics.h"

// firewall

char pyjournal_firewall__doc__[] = "";
char pyjournal_firewall__name__[] = "firewall";


PyObject * pyjournal_firewall(PyObject *, PyObject * args)
{
    char * facility;
    int ok = PyArg_ParseTuple(args, "s:initialize", &facility);
    if (!ok) {
        return 0;
    }

    journal::SeverityFirewall::state_t * state = &journal::SeverityFirewall::lookup(facility);

    // return
    return PyCObject_FromVoidPtr(state, 0);
}
    
// debug

char pyjournal_debug__doc__[] = "";
char pyjournal_debug__name__[] = "debug";


PyObject * pyjournal_debug(PyObject *, PyObject * args)
{
    char * facility;
    int ok = PyArg_ParseTuple(args, "s:initialize", &facility);
    if (!ok) {
        return 0;
    }

    journal::SeverityDebug::state_t * state = &journal::SeverityDebug::lookup(facility);

    // return
    return PyCObject_FromVoidPtr(state, 0);
}
    
// info

char pyjournal_info__doc__[] = "";
char pyjournal_info__name__[] = "info";


PyObject * pyjournal_info(PyObject *, PyObject * args)
{
    char * facility;
    int ok = PyArg_ParseTuple(args, "s:initialize", &facility);
    if (!ok) {
        return 0;
    }

    journal::SeverityInfo::state_t * state = &journal::SeverityInfo::lookup(facility);

    // return
    return PyCObject_FromVoidPtr(state, 0);
}
    
// warning

char pyjournal_warning__doc__[] = "";
char pyjournal_warning__name__[] = "warning";


PyObject * pyjournal_warning(PyObject *, PyObject * args)
{
    char * facility;
    int ok = PyArg_ParseTuple(args, "s:initialize", &facility);
    if (!ok) {
        return 0;
    }

    journal::SeverityWarning::state_t * state = &journal::SeverityWarning::lookup(facility);

    // return
    return PyCObject_FromVoidPtr(state, 0);
}
    
// error

char pyjournal_error__doc__[] = "";
char pyjournal_error__name__[] = "error";


PyObject * pyjournal_error(PyObject *, PyObject * args)
{
    char * facility;
    int ok = PyArg_ParseTuple(args, "s:initialize", &facility);
    if (!ok) {
        return 0;
    }

    journal::SeverityError::state_t * state = &journal::SeverityError::lookup(facility);

    // return
    return PyCObject_FromVoidPtr(state, 0);
}
    
// version
// $Id: facility.cc,v 1.1.1.1 2005/03/08 16:13:54 aivazis Exp $

// End of file
