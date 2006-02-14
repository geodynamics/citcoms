// -*- C++ -*-
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//                               Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2005 All Rights Reserved
// 
//  <LicenseText>
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 

#include <portinfo>
#include <Python.h>

#include "misc.h"


// copyright

char pytabulator_copyright__doc__[] = "";
char pytabulator_copyright__name__[] = "copyright";

static char pytabulator_copyright_note[] = 
    "tabulator python module: Copyright (c) 1998-2005 Michael A.G. Aivazis";


PyObject * pytabulator_copyright(PyObject *, PyObject *)
{
    return Py_BuildValue("s", pytabulator_copyright_note);
}
    
// version
// $Id: misc.cc,v 1.1.1.1 2005/03/17 20:03:02 aivazis Exp $

// End of file
