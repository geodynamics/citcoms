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

char pyrigid_copyright__doc__[] = "";
char pyrigid_copyright__name__[] = "copyright";

static char pyrigid_copyright_note[] = 
    "rigid python module: Copyright (c) 1998-2005 Michael A.G. Aivazis";


PyObject * pyrigid_copyright(PyObject *, PyObject *)
{
    return Py_BuildValue("s", pyrigid_copyright_note);
}
    
// version
// $Id: misc.cc,v 1.1.1.1 2005/03/08 16:13:58 aivazis Exp $

// End of file
