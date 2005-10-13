// -*- C++ -*-
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//                              Michael A.G. Aivazis
//                       California Institute of Technology
//                       (C) 1998-2005  All Rights Reserved
// 
//  <LicenseText>
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 

#include <portinfo>
#include <Python.h>

#include <map>
#include <string>
#include <sstream>

#include "journal.h"

// necessary so that we can attach the ProxyDevice to the journal singleton
#include "../libjournal/Diagnostic.h"
#include "../libjournal/Device.h"
#include "../libjournal/Journal.h"
#include "ProxyDevice.h"

// initialize
char pyjournal_initialize__doc__[] = "";
char pyjournal_initialize__name__[] = "initialize";

PyObject * pyjournal_initialize(PyObject *, PyObject * args)
{
    PyObject * py_journal;

    int ok = PyArg_ParseTuple(args, "O:initialize", &py_journal);

    if (!ok) {
        return 0;
    }

    // attach the proxy device so that output can be redirected by the python module
    journal::Diagnostic::journal().device(new ProxyDevice(py_journal));

    // return
    Py_INCREF(Py_None);
    return Py_None;
}

// $Id: journal.cc,v 1.1.1.1 2005/03/08 16:13:54 aivazis Exp $

// End of file
