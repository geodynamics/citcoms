// -*- C++ -*-
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//                               Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2003 All Rights Reserved
// 
//  <LicenseText>
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 

#include <portinfo>
#include <Python.h>

#include "misc.h"

extern "C" double return1_test();

// copyright

char pyCitcomSRegional_copyright__doc__[] = "";
char pyCitcomSRegional_copyright__name__[] = "copyright";

static char pyCitcomSRegional_copyright_note[] = 
    "CitcomSRegional python module: Copyright (c) 1998-2003 California Institute of Technology";


PyObject * pyCitcomSRegional_copyright(PyObject *, PyObject *)
{
    return Py_BuildValue("s", pyCitcomSRegional_copyright_note);
}




// hello

char pyCitcomSRegional_return1_test__doc__[] = "";
char pyCitcomSRegional_return1_test__name__[] = "return1_test";

PyObject * pyCitcomSRegional_return1_test(PyObject *, PyObject *)
{
    double a;
    a = return1_test();
    return Py_BuildValue("d", a);
}





// version
// $Id: misc.cc,v 1.1.1.1 2003/03/24 01:46:37 tan2 Exp $

// End of file
