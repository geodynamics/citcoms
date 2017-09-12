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

#if !defined(pyCitcom_advection_diffusion_h)
#define pyCitcom_advection_diffusion_h

/* advection_diffustion routines */
extern char pyCitcom_PG_timestep_init__doc__[];
extern char pyCitcom_PG_timestep_init__name__[];
PyObject * pyCitcom_PG_timestep_init(PyObject *, PyObject *);

extern char pyCitcom_PG_timestep_solve__doc__[];
extern char pyCitcom_PG_timestep_solve__name__[];
PyObject * pyCitcom_PG_timestep_solve(PyObject *, PyObject *);

extern char pyCitcom_set_convection_defaults__doc__[];
extern char pyCitcom_set_convection_defaults__name__[];
PyObject * pyCitcom_set_convection_defaults(PyObject *, PyObject *);

extern char pyCitcom_stable_timestep__doc__[];
extern char pyCitcom_stable_timestep__name__[];
PyObject * pyCitcom_stable_timestep(PyObject *, PyObject *);

#endif

/* $Id: advdiffu.h 4957 2006-10-12 14:48:43Z leif $ */

/* End of file */
