// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_misc_h)
#define pyCitcom_misc_h

// copyright
extern char pyCitcom_copyright__name__[];
extern char pyCitcom_copyright__doc__[];
extern "C"
PyObject * pyCitcom_copyright(PyObject *, PyObject *);


extern char pyCitcom_return1_test__name__[];
extern char pyCitcom_return1_test__doc__[];
extern "C"
PyObject * pyCitcom_return1_test(PyObject *, PyObject *);


extern char pyCitcom_read_instructions__name__[];
extern char pyCitcom_read_instructions__doc__[];
extern "C"
PyObject * pyCitcom_read_instructions(PyObject *, PyObject *);


extern char pyCitcom_CPU_time__name__[];
extern char pyCitcom_CPU_time__doc__[];
extern "C"
PyObject * pyCitcom_CPU_time(PyObject *, PyObject *);


//
//

extern char pyCitcom_citcom_init__doc__[];
extern char pyCitcom_citcom_init__name__[];
extern "C"
PyObject * pyCitcom_citcom_init(PyObject *, PyObject *);


extern char pyCitcom_global_default_values__name__[];
extern char pyCitcom_global_default_values__doc__[];
extern "C"
PyObject * pyCitcom_global_default_values(PyObject *, PyObject *);


extern char pyCitcom_set_signal__name__[];
extern char pyCitcom_set_signal__doc__[];
extern "C"
PyObject * pyCitcom_set_signal(PyObject *, PyObject *);


extern char pyCitcom_velocities_conform_bcs__name__[];
extern char pyCitcom_velocities_conform_bcs__doc__[];
extern "C"
PyObject * pyCitcom_velocities_conform_bcs(PyObject *, PyObject *);


extern char pyCitcom_BC_update_plate_velocity__name__[];
extern char pyCitcom_BC_update_plate_velocity__doc__[];
extern "C"
PyObject * pyCitcom_BC_update_plate_velocity(PyObject *, PyObject *);


extern char pyCitcom_Tracer_tracer_advection__name__[];
extern char pyCitcom_Tracer_tracer_advection__doc__[];
extern "C"
PyObject * pyCitcom_Tracer_tracer_advection(PyObject *, PyObject *);


extern char pyCitcom_Visc_update_material__name__[];
extern char pyCitcom_Visc_update_material__doc__[];
extern "C"
PyObject * pyCitcom_Visc_update_material(PyObject *, PyObject *);


#endif

// version
// $Id: misc.h,v 1.17 2005/01/19 02:01:41 tan2 Exp $

// End of file
