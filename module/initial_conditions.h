// -*- C++ -*-
//
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
//

#if !defined(pyCitcom_initial_conditions_h)
#define pyCitcom_initial_conditions_h


extern char pyCitcom_ic_constructTemperature__name__[];
extern char pyCitcom_ic_constructTemperature__doc__[];
extern "C"
PyObject * pyCitcom_ic_constructTemperature(PyObject *, PyObject *);


extern char pyCitcom_ic_restartTemperature__name__[];
extern char pyCitcom_ic_restartTemperature__doc__[];
extern "C"
PyObject * pyCitcom_ic_restartTemperature(PyObject *, PyObject *);


extern char pyCitcom_ic_initPressure__name__[];
extern char pyCitcom_ic_initPressure__doc__[];
extern "C"
PyObject * pyCitcom_ic_initPressure(PyObject *, PyObject *);


extern char pyCitcom_ic_initVelocity__name__[];
extern char pyCitcom_ic_initVelocity__doc__[];
extern "C"
PyObject * pyCitcom_ic_initVelocity(PyObject *, PyObject *);


extern char pyCitcom_ic_initViscosity__name__[];
extern char pyCitcom_ic_initViscosity__doc__[];
extern "C"
PyObject * pyCitcom_ic_initViscosity(PyObject *, PyObject *);

#endif

// version
// $Id$

// End of file
