// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <Python.h>
#include <iostream>

extern "C" {
#include "global_defs.h"
}

#include "misc.h"


// copyright

char pyExchanger_copyright__doc__[] = "";
char pyExchanger_copyright__name__[] = "copyright";

static char pyExchanger_copyright_note[] =
    "Exchanger python module: Copyright (c) 1998-2003 California Institute of Technology";


PyObject * pyExchanger_copyright(PyObject *, PyObject *)
{
    return Py_BuildValue("s", pyExchanger_copyright_note);
}

// hello

char pyExchanger_hello__doc__[] = "";
char pyExchanger_hello__name__[] = "hello";

PyObject * pyExchanger_hello(PyObject *, PyObject *)
{
    return Py_BuildValue("s", "hello");
}

// return (All_variables* E)

char pyExchanger_FinereturnE__doc__[] = "";
char pyExchanger_FinereturnE__name__[] = "FinereturnE";

PyObject * pyExchanger_FinereturnE(PyObject *, PyObject *)
{
    All_variables *E = new All_variables;

    E->parallel.me = 1;
    E->parallel.me_loc[1] = 0;
    E->parallel.me_loc[2] = 0;
    E->parallel.me_loc[3] = 0;
    E->parallel.nprocx = 1;
    E->parallel.nprocy = 1;
    E->parallel.nprocz = 1;

    E->sphere.caps_per_proc = 1;

    E->mesh.levmax = 1;
    E->mesh.dof = 3;
    E->mesh.nox = E->lmesh.nox = 4;
    E->mesh.noy = E->lmesh.noy = 4;
    E->mesh.noz = E->lmesh.noz = 3;

    E->control.theta_max=2.0;
    E->control.theta_min=1.0;
    E->control.fi_max=2.0;
    E->control.fi_min=1.0;
    E->sphere.ro=2.0;
    E->sphere.ri=1.0;


    const int n = E->lmesh.nox * E->lmesh.noy * E->lmesh.noz;
    for(int m=1;m<=E->sphere.caps_per_proc;m++)
	for(int i=1; i<=E->mesh.dof; i++) {
	    E->X[E->mesh.levmax][m][i] = new double[n+1];
    }

    for(int m=1;m<=E->sphere.caps_per_proc;m++)
	for(int k=1;k<=E->lmesh.noy;k++)
	    for(int j=1;j<=E->lmesh.nox;j++)
		for(int i=1;i<=E->lmesh.noz;i++)  {
		    int node = i + (j-1)*E->lmesh.noz
			     + (k-1)*E->lmesh.noz*E->lmesh.nox;
		    E->X[E->mesh.levmax][m][1][node] = 1.0/(E->lmesh.nox-1)*(j-1)+1.0;
		    E->X[E->mesh.levmax][m][2][node] = 1.0/(E->lmesh.noy-1)*(k-1)+1.0;
		    E->X[E->mesh.levmax][m][3][node] = 1.0/(E->lmesh.noz-1)*(i-1)+1.0;

//  		    std::cout << "Fine Grid " <<  node << " "
//  			      << E->X[E->mesh.levmax][m][1][node] << " "
//  			      << E->X[E->mesh.levmax][m][2][node] << " "
//  			      << E->X[E->mesh.levmax][m][3][node] << " "
//  			      << std::endl;
		}

    PyObject *cobj = PyCObject_FromVoidPtr(E, NULL);
    return Py_BuildValue("O", cobj);
}


char pyExchanger_CoarsereturnE__doc__[] = "";
char pyExchanger_CoarsereturnE__name__[] = "CoarsereturnE";

PyObject * pyExchanger_CoarsereturnE(PyObject *, PyObject *)
{
    All_variables *E = new All_variables;

    E->parallel.me = 1;
    E->parallel.me_loc[1] = 0;
    E->parallel.me_loc[2] = 0;
    E->parallel.me_loc[3] = 0;
    E->parallel.nprocx = 1;
    E->parallel.nprocy = 1;
    E->parallel.nprocz = 1;

    E->sphere.caps_per_proc = 1;

    E->mesh.levmax = 1;
    E->mesh.dof = 3;
    E->mesh.nox = E->lmesh.nox = 4;
    E->mesh.noy = E->lmesh.noy = 4;
    E->mesh.noz = E->lmesh.noz = 3;

    E->control.theta_max=3.0;
    E->control.theta_min=0.0;
    E->control.fi_max=3.0;
    E->control.fi_min=0.0;
    E->sphere.ro=2.0;
    E->sphere.ri=0.0;

    const int n = E->lmesh.nox * E->lmesh.noy * E->lmesh.noz;
    for(int m=1;m<=E->sphere.caps_per_proc;m++)
	for(int i=1; i<=E->mesh.dof; i++) {
	    E->X[E->mesh.levmax][m][i] = new double[n+1];
    }

    for(int m=1;m<=E->sphere.caps_per_proc;m++)
	for(int k=1;k<=E->lmesh.noy;k++)
	    for(int j=1;j<=E->lmesh.nox;j++)
		for(int i=1;i<=E->lmesh.noz;i++)  {
		    int node = i + (j-1)*E->lmesh.noz
			     + (k-1)*E->lmesh.noz*E->lmesh.nox;
		    E->X[E->mesh.levmax][m][1][node] = j-1;
		    E->X[E->mesh.levmax][m][2][node] = k-1;
		    E->X[E->mesh.levmax][m][3][node] = i-1;

// 		    std::cout << "Coarse Grid " <<  node << " "
//  			      << E->X[E->mesh.levmax][m][1][node] << " "
//  			      << E->X[E->mesh.levmax][m][2][node] << " "
//  			      << E->X[E->mesh.levmax][m][3][node] << " "
//  			      << std::endl;
		}

    PyObject *cobj = PyCObject_FromVoidPtr(E, NULL);
    return Py_BuildValue("O", cobj);
}

// version
// $Id: misc.cc,v 1.5 2003/09/18 22:03:48 ces74 Exp $

// End of file
