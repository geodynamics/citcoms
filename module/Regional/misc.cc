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
#include "global_defs.h"
//#include "citcom_init.h"

extern "C" double return1_test();
extern "C" struct All_variables* Citcom_Init();
struct All_variables *E;


// copyright

char pyCitcomSRegional_copyright__doc__[] = "";
char pyCitcomSRegional_copyright__name__[] = "copyright";

static char pyCitcomSRegional_copyright_note[] = 
    "CitcomSRegional python module: Copyright (c) 1998-2003 California Institute of Technology";


PyObject * pyCitcomSRegional_copyright(PyObject *, PyObject *)
{
    return Py_BuildValue("s", pyCitcomSRegional_copyright_note);
}



char pyCitcomSRegional_return1_test__doc__[] = "";
char pyCitcomSRegional_return1_test__name__[] = "return1_test";

PyObject * pyCitcomSRegional_return1_test(PyObject *, PyObject *)
{
    double a;
    a = return1_test();
    return Py_BuildValue("d", a);
}


char pyCitcomSRegional_Citcom_Init__doc__[] = "";
char pyCitcomSRegional_Citcom_Init__name__[] = "Citcom_Init";

PyObject * pyCitcomSRegional_Citcom_Init(PyObject *, PyObject *)
{
    E = Citcom_Init();
    return Py_BuildValue("O", E);
}




// version
// $Id: misc.cc,v 1.2 2003/04/04 00:42:50 tan2 Exp $

// End of file
