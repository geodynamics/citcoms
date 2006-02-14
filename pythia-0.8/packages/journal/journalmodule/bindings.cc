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

#include "bindings.h"

#include "facility.h"      // facility lookups
#include "journal.h"       // journal initialization
#include "misc.h"          // miscellaneous methods
#include "state.h"         // facility state accessors

// the method table

struct PyMethodDef pyjournal_methods[] = {

    // facility
    {pyjournal_firewall__name__, pyjournal_firewall, METH_VARARGS, pyjournal_firewall__doc__},
    {pyjournal_debug__name__, pyjournal_debug, METH_VARARGS, pyjournal_debug__doc__},
    {pyjournal_info__name__, pyjournal_info, METH_VARARGS, pyjournal_info__doc__},
    {pyjournal_warning__name__, pyjournal_warning, METH_VARARGS, pyjournal_warning__doc__},
    {pyjournal_error__name__, pyjournal_error, METH_VARARGS, pyjournal_error__doc__},

    // journal
    {pyjournal_initialize__name__, pyjournal_initialize,
     METH_VARARGS, pyjournal_initialize__doc__},

    // state
    {pyjournal_getState__name__, pyjournal_getState, METH_VARARGS, pyjournal_getState__doc__},
    {pyjournal_setState__name__, pyjournal_setState, METH_VARARGS, pyjournal_setState__doc__},
    {pyjournal_flip__name__, pyjournal_flip, METH_VARARGS, pyjournal_flip__doc__},
    {pyjournal_activate__name__, pyjournal_activate, METH_VARARGS, pyjournal_activate__doc__},
    {pyjournal_deactivate__name__,
     pyjournal_deactivate, METH_VARARGS, pyjournal_deactivate__doc__},

    // copyright
    {pyjournal_copyright__name__, pyjournal_copyright,
     METH_VARARGS, pyjournal_copyright__doc__},


// Sentinel
    {0, 0}
};

// version
// $Id: bindings.cc,v 1.1.1.1 2005/03/08 16:13:54 aivazis Exp $

// End of file
