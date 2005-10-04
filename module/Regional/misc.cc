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


#include "exceptions.h"
#include "misc.h"

#include "mpi.h"
#include "global_defs.h"
#include "citcom_init.h"
#include "advection_diffusion.h"

extern "C" {

    double return1_test();
    void read_instructions(struct All_variables*, char*);
    double CPU_time0();

    void global_default_values(struct All_variables*);
    void parallel_process_termination();
    void read_mat_from_file(struct All_variables*);
    void read_velocity_boundary_from_file(struct All_variables*);
    void set_signal();
    void tracer_advection(struct All_variables*);
    void velocities_conform_bcs(struct All_variables*, double **);

}

#include "mpi/Communicator.h"
#include "mpi/Group.h"

// copyright

char pyCitcom_copyright__doc__[] = "";
char pyCitcom_copyright__name__[] = "copyright";

static char pyCitcom_copyright_note[] =
"CitcomS python module: Copyright (c) 1998-2003 California Institute of Technology";


PyObject * pyCitcom_copyright(PyObject *, PyObject *)
{
  return Py_BuildValue("s", pyCitcom_copyright_note);
}

//////////////////////////////////////////////////////////////////////////
// This section is for testing or temporatory implementation
//////////////////////////////////////////////////////////////////////////



char pyCitcom_return1_test__doc__[] = "";
char pyCitcom_return1_test__name__[] = "return1_test";

PyObject * pyCitcom_return1_test(PyObject *, PyObject *)
{
    double a;
    a = return1_test();
    return Py_BuildValue("d", a);
}


char pyCitcom_read_instructions__doc__[] = "";
char pyCitcom_read_instructions__name__[] = "read_instructions";

PyObject * pyCitcom_read_instructions(PyObject *self, PyObject *args)
{
    PyObject *obj;
    char *filename;

    if (!PyArg_ParseTuple(args, "Os:read_instructions", &obj, &filename))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    read_instructions(E, filename);

    // test
    fprintf(stderr,"output file prefix: %s\n", E->control.data_file);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_CPU_time__doc__[] = "";
char pyCitcom_CPU_time__name__[] = "CPU_time";

PyObject * pyCitcom_CPU_time(PyObject *, PyObject *)
{
    return Py_BuildValue("d", CPU_time0());
}


//////////////////////////////////////////////////////////////////////////
// This section is for finished implementation
//////////////////////////////////////////////////////////////////////////

char pyCitcom_citcom_init__doc__[] = "";
char pyCitcom_citcom_init__name__[] = "citcom_init";

PyObject * pyCitcom_citcom_init(PyObject *self, PyObject *args)
{
    PyObject *Obj;

    if (!PyArg_ParseTuple(args, "O:citcom_init", &Obj))
        return NULL;

    mpi::Communicator * comm = (mpi::Communicator *) PyCObject_AsVoidPtr(Obj);
    if (comm == NULL)
        return PyErr_Format(pyCitcom_runtimeError,
                            "%s: 'mpi::Communicator *' argument is null",
                            pyCitcom_citcom_init__name__);
        
    MPI_Comm world = comm->handle();

    // Allocate global pointer E
    struct All_variables* E = citcom_init(&world);

    // if E is NULL, raise an exception here.
    if (E == NULL)
        return PyErr_Format(pyCitcom_runtimeError,
                            "%s: 'libCitcomSCommon.citcom_init' failed",
                            pyCitcom_citcom_init__name__);

    PyObject *cobj = PyCObject_FromVoidPtr(E, NULL);

    return Py_BuildValue("O", cobj);
}


char pyCitcom_global_default_values__doc__[] = "";
char pyCitcom_global_default_values__name__[] = "global_default_values";

PyObject * pyCitcom_global_default_values(PyObject *, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:global_default_values", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    global_default_values(E);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_set_signal__doc__[] = "";
char pyCitcom_set_signal__name__[] = "set_signal";

PyObject * pyCitcom_set_signal(PyObject *, PyObject *)
{
    set_signal();

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_velocities_conform_bcs__doc__[] = "";
char pyCitcom_velocities_conform_bcs__name__[] = "velocities_conform_bcs";

PyObject * pyCitcom_velocities_conform_bcs(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:velocities_conform_bcs", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    velocities_conform_bcs(E, E->U);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_BC_update_plate_velocity__doc__[] = "";
char pyCitcom_BC_update_plate_velocity__name__[] = "BC_update_plate_velocity";

PyObject * pyCitcom_BC_update_plate_velocity(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:BC_update_plate_velocity", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    if(E->control.vbcs_file==1)
      read_velocity_boundary_from_file(E);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_Tracer_tracer_advection__doc__[] = "";
char pyCitcom_Tracer_tracer_advection__name__[] = "Tracer_tracer_advection";

PyObject * pyCitcom_Tracer_tracer_advection(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:Tracer_tracer_advection", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    if(E->control.tracer==1)
      tracer_advection(E);

    Py_INCREF(Py_None);
    return Py_None;
}


char pyCitcom_Visc_update_material__doc__[] = "";
char pyCitcom_Visc_update_material__name__[] = "Visc_update_material";

PyObject * pyCitcom_Visc_update_material(PyObject *self, PyObject *args)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O:Visc_update_material", &obj))
        return NULL;

    struct All_variables* E = static_cast<struct All_variables*>(PyCObject_AsVoidPtr(obj));

    if(E->control.mat_control==1)
      read_mat_from_file(E);

    Py_INCREF(Py_None);
    return Py_None;
}




//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////



// version
// $Id$

// End of file
