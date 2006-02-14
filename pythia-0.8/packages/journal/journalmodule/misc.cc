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

#include "misc.h"


// copyright

char pyjournal_copyright__doc__[] = "";
char pyjournal_copyright__name__[] = "copyright";

static char pyjournal_copyright_note[] = 
    "pythia.journal module: Copyright (c) 1998-2005 Michael A.G. Aivazis";


PyObject * pyjournal_copyright(PyObject *, PyObject *)
{
    return Py_BuildValue("s", pyjournal_copyright_note);
}
    
// version
// $Id: misc.cc,v 1.1.1.1 2005/03/08 16:13:54 aivazis Exp $

// End of file
