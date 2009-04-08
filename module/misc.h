/*
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//<LicenseText>
//
// CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
// Copyright (C) 2002-2005, California Institute of Technology.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//</LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

#if !defined(pyCitcom_misc_h)
#define pyCitcom_misc_h

extern char pyCitcom_copyright__name__[];
extern char pyCitcom_copyright__doc__[];
PyObject * pyCitcom_copyright(PyObject *, PyObject *);


extern char pyCitcom_return1_test__name__[];
extern char pyCitcom_return1_test__doc__[];
PyObject * pyCitcom_return1_test(PyObject *, PyObject *);


extern char pyCitcom_CPU_time__name__[];
extern char pyCitcom_CPU_time__doc__[];
PyObject * pyCitcom_CPU_time(PyObject *, PyObject *);


extern char pyCitcom_citcom_init__doc__[];
extern char pyCitcom_citcom_init__name__[];
PyObject * pyCitcom_citcom_init(PyObject *, PyObject *);


extern char pyCitcom_full_solver_init__doc__[];
extern char pyCitcom_full_solver_init__name__[];
PyObject * pyCitcom_full_solver_init(PyObject *, PyObject *);


extern char pyCitcom_regional_solver_init__doc__[];
extern char pyCitcom_regional_solver_init__name__[];
PyObject * pyCitcom_regional_solver_init(PyObject *, PyObject *);


extern char pyCitcom_global_default_values__name__[];
extern char pyCitcom_global_default_values__doc__[];
PyObject * pyCitcom_global_default_values(PyObject *, PyObject *);


extern char pyCitcom_set_signal__name__[];
extern char pyCitcom_set_signal__doc__[];
PyObject * pyCitcom_set_signal(PyObject *, PyObject *);


extern char pyCitcom_check_settings_consistency__name__[];
extern char pyCitcom_check_settings_consistency__doc__[];
PyObject * pyCitcom_check_settings_consistency(PyObject *, PyObject *);


extern char pyCitcom_velocities_conform_bcs__name__[];
extern char pyCitcom_velocities_conform_bcs__doc__[];
PyObject * pyCitcom_velocities_conform_bcs(PyObject *, PyObject *);


extern char pyCitcom_BC_update_plate_temperature__name__[];
extern char pyCitcom_BC_update_plate_temperature__doc__[];
PyObject * pyCitcom_BC_update_plate_temperature(PyObject *, PyObject *);


extern char pyCitcom_BC_update_plate_velocity__name__[];
extern char pyCitcom_BC_update_plate_velocity__doc__[];
PyObject * pyCitcom_BC_update_plate_velocity(PyObject *, PyObject *);


extern char pyCitcom_Tracer_tracer_advection__name__[];
extern char pyCitcom_Tracer_tracer_advection__doc__[];
PyObject * pyCitcom_Tracer_tracer_advection(PyObject *, PyObject *);


extern char pyCitcom_Visc_update_material__name__[];
extern char pyCitcom_Visc_update_material__doc__[];
PyObject * pyCitcom_Visc_update_material(PyObject *, PyObject *);


extern char pyCitcom_return_dt__name__[];
extern char pyCitcom_return_dt__doc__[];
PyObject * pyCitcom_return_dt(PyObject *, PyObject *);


extern char pyCitcom_return_step__name__[];
extern char pyCitcom_return_step__doc__[];
PyObject * pyCitcom_return_step(PyObject *, PyObject *);


extern char pyCitcom_return_t__name__[];
extern char pyCitcom_return_t__doc__[];
PyObject * pyCitcom_return_t(PyObject *, PyObject *);


extern char pyCitcom_return_rank__name__[];
extern char pyCitcom_return_rank__doc__[];
PyObject * pyCitcom_return_rank(PyObject *, PyObject *);


extern char pyCitcom_return_pid__name__[];
extern char pyCitcom_return_pid__doc__[];
PyObject * pyCitcom_return_pid(PyObject *, PyObject *);


#endif

/* $Id$ */

/* End of file */
