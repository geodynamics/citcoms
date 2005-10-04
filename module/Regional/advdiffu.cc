// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//<LicenseText>
//=====================================================================
//
//                             CitcomS.py
//                 ---------------------------------
//
//                              Authors:
//            Eh Tan, Eun-seo Choi, and Pururav Thoutireddy 
//          (c) California Institute of Technology 2002-2005
//
//        By downloading and/or installing this software you have
//       agreed to the CitcomS.py-LICENSE bundled with this software.
//             Free for non-commercial academic research ONLY.
//      This program is distributed WITHOUT ANY WARRANTY whatsoever.
//
//=====================================================================
//
//  Copyright June 2005, by the California Institute of Technology.
//  ALL RIGHTS RESERVED. United States Government Sponsorship Acknowledged.
// 
//  Any commercial use must be negotiated with the Office of Technology
//  Transfer at the California Institute of Technology. This software
//  may be subject to U.S. export control laws and regulations. By
//  accepting this software, the user agrees to comply with all
//  applicable U.S. export laws and regulations, including the
//  International Traffic and Arms Regulations, 22 C.F.R. 120-130 and
//  the Export Administration Regulations, 15 C.F.R. 730-744. User has
//  the responsibility to obtain export licenses, or other export
//  authority as may be required before exporting such information to
//  foreign countries or providing access to foreign nationals.  In no
//  event shall the California Institute of Technology be liable to any
//  party for direct, indirect, special, incidental or consequential
//  damages, including lost profits, arising out of the use of this
//  software and its documentation, even if the California Institute of
//  Technology has been advised of the possibility of such damage.
// 
//  The California Institute of Technology specifically disclaims any
//  warranties, including the implied warranties or merchantability and
//  fitness for a particular purpose. The software and documentation
//  provided hereunder is on an "as is" basis, and the California
//  Institute of Technology has no obligations to provide maintenance,
//  support, updates, enhancements or modifications.
//
//=====================================================================
//</LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <Python.h>
#include <cstdio>

#include "advdiffu.h"

#include "global_defs.h"
#include "advection_diffusion.h"

extern "C" {
    void set_convection_defaults(struct All_variables *);
}


char pyCitcom_PG_timestep_init__doc__[] = "";
char pyCitcom_PG_timestep_init__name__[] = "PG_timestep_init";
PyObject * pyCitcom_PG_timestep_init(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:PG_timestep_init", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    PG_timestep_init(E);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_PG_timestep_solve__doc__[] = "";
char pyCitcom_PG_timestep_solve__name__[] = "PG_timestep_solve";
PyObject * pyCitcom_PG_timestep_solve(PyObject *self, PyObject *args)
{
    PyObject *obj;
    double dt;

    if (!PyArg_ParseTuple(args, "Od:PG_timestep_solve", &obj, &dt))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    E->monitor.solution_cycles++;
    if(E->monitor.solution_cycles>E->control.print_convergence)
	E->control.print_convergence=1;

    E->advection.timestep = dt;
    PG_timestep_solve(E);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_set_convection_defaults__doc__[] = "";
char pyCitcom_set_convection_defaults__name__[] = "set_convection_defaults";
PyObject * pyCitcom_set_convection_defaults(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:set_convection_defaults", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    E->control.CONVECTION = 1;
    set_convection_defaults(E);

    // copied from advection_diffusion_parameters()
    E->advection.total_timesteps = 1;
    E->advection.sub_iterations = 1;
    E->advection.last_sub_iterations = 1;
    E->advection.gamma = 0.5;
    E->monitor.T_maxvaried = 1.05;

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_stable_timestep__doc__[] = "";
char pyCitcom_stable_timestep__name__[] = "stable_timestep";
PyObject * pyCitcom_stable_timestep(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:stable_timestep", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));


    std_timestep(E);

    return Py_BuildValue("d", E->advection.timestep);
}




// version
// $Id$

// End of file
