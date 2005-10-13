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

char pypulse_copyright__doc__[] = "";
char pypulse_copyright__name__[] = "copyright";

static char pypulse_copyright_note[] = 
    "pulse python module: Copyright (c) 1998-2005 Michael A.G. Aivazis";


PyObject * pypulse_copyright(PyObject *, PyObject *)
{
    return Py_BuildValue("s", pypulse_copyright_note);
}
    
// version
// $Id: misc.cc,v 1.1.1.1 2005/03/08 16:13:57 aivazis Exp $

// End of file
