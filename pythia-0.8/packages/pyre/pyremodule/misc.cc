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

char pyremodule_copyright__doc__[] = "";
char pyremodule_copyright__name__[] = "copyright";

static char pyremodule_copyright_note[] = 
    "pyre python module: Copyright (c) 1998-2005 Michael A.G. Aivazis";


PyObject * pyremodule_copyright(PyObject *, PyObject *)
{
    return Py_BuildValue("s", pyremodule_copyright_note);
}
    
// version
// $Id: misc.cc,v 1.1.1.1 2005/03/08 16:13:52 aivazis Exp $

// End of file
