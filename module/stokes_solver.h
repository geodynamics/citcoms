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

#if !defined(pyCitcom_stokes_solver_h)
#define pyCitcom_stokes_solver_h

extern char pyCitcom_assemble_forces__name__[];
extern char pyCitcom_assemble_forces__doc__[];
PyObject * pyCitcom_assemble_forces(PyObject *, PyObject *);


extern char pyCitcom_assemble_forces_pseudo_surf__name__[];
extern char pyCitcom_assemble_forces_pseudo_surf__doc__[];
PyObject * pyCitcom_assemble_forces_pseudo_surf(PyObject *, PyObject *);


extern char pyCitcom_construct_stiffness_B_matrix__name__[];
extern char pyCitcom_construct_stiffness_B_matrix__doc__[];
PyObject * pyCitcom_construct_stiffness_B_matrix(PyObject *, PyObject *);


extern char pyCitcom_general_stokes_solver__name__[];
extern char pyCitcom_general_stokes_solver__doc__[];
PyObject * pyCitcom_general_stokes_solver(PyObject *, PyObject *);


extern char pyCitcom_general_stokes_solver_setup__name__[];
extern char pyCitcom_general_stokes_solver_setup__doc__[];
PyObject * pyCitcom_general_stokes_solver_setup(PyObject *, PyObject *);


extern char pyCitcom_get_system_viscosity__name__[];
extern char pyCitcom_get_system_viscosity__doc__[];
PyObject * pyCitcom_get_system_viscosity(PyObject *, PyObject *);


extern char pyCitcom_set_cg_defaults__name__[];
extern char pyCitcom_set_cg_defaults__doc__[];
PyObject * pyCitcom_set_cg_defaults(PyObject *, PyObject *);


extern char pyCitcom_set_mg_defaults__name__[];
extern char pyCitcom_set_mg_defaults__doc__[];
PyObject * pyCitcom_set_mg_defaults(PyObject *, PyObject *);


extern char pyCitcom_set_mg_el_defaults__name__[];
extern char pyCitcom_set_mg_el_defaults__doc__[];
PyObject * pyCitcom_set_mg_el_defaults(PyObject *, PyObject *);


extern char pyCitcom_solve_constrained_flow_iterative__name__[];
extern char pyCitcom_solve_constrained_flow_iterative__doc__[];
PyObject * pyCitcom_solve_constrained_flow_iterative(PyObject *, PyObject *);


extern char pyCitcom_solve_constrained_flow_iterative_pseudo_surf__name__[];
extern char pyCitcom_solve_constrained_flow_iterative_pseudo_surf__doc__[];
PyObject * pyCitcom_solve_constrained_flow_iterative_pseudo_surf(PyObject *, PyObject *);

#endif

/* $Id$ */

/* End of file */
