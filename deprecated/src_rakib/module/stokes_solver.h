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

extern char pyCitcom_general_stokes_solver__name__[];
extern char pyCitcom_general_stokes_solver__doc__[];
PyObject * pyCitcom_general_stokes_solver(PyObject *, PyObject *);


extern char pyCitcom_general_stokes_solver_setup__name__[];
extern char pyCitcom_general_stokes_solver_setup__doc__[];
PyObject * pyCitcom_general_stokes_solver_setup(PyObject *, PyObject *);


extern char pyCitcom_set_cg_defaults__name__[];
extern char pyCitcom_set_cg_defaults__doc__[];
PyObject * pyCitcom_set_cg_defaults(PyObject *, PyObject *);


extern char pyCitcom_set_mg_defaults__name__[];
extern char pyCitcom_set_mg_defaults__doc__[];
PyObject * pyCitcom_set_mg_defaults(PyObject *, PyObject *);

#endif

/* $Id: stokes_solver.h 16087 2009-12-08 21:43:15Z tan2 $ */

/* End of file */
