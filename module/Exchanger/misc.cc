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
#include "element_definitions.h"
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
    int nox,noy,noz,p,q,r,lev,i,j,rr;
    int elx,ely,elz,nel,nno,element,start;

    E->parallel.me = 1;
    E->parallel.me_loc[1] = 0;
    E->parallel.me_loc[2] = 0;
    E->parallel.me_loc[3] = 0;
    E->parallel.nprocx = 1;
    E->parallel.nprocy = 1;
    E->parallel.nprocz = 1;

    E->sphere.caps_per_proc = 1;

    E->mesh.levmax = 1;
    E->mesh.levmin = 1;

    E->mesh.dof = 3;
    E->mesh.nox = 4;
    E->mesh.noy = 4;
    E->mesh.noz = 3;

    E->mesh.elx = E->mesh.nox - 1;
    E->mesh.ely = E->mesh.noy - 1;
    E->mesh.elz = E->mesh.noz - 1;

    E->lmesh.elx = E->mesh.elx/E->parallel.nprocx;
    E->lmesh.elz = E->mesh.elz/E->parallel.nprocz;
    E->lmesh.ely = E->mesh.ely/E->parallel.nprocy;
    E->lmesh.nox = E->lmesh.elx + 1;
    E->lmesh.noz = E->lmesh.elz + 1;
    E->lmesh.noy = E->lmesh.ely + 1;
    
    E->lmesh.nno = E->lmesh.noz*E->lmesh.nox*E->lmesh.noy;
    E->lmesh.nel = E->lmesh.ely*E->lmesh.elx*E->lmesh.elz;
    E->lmesh.npno = E->lmesh.nel;
    
    noz = E->lmesh.noz;
    noy = E->mesh.noy;
    nox = E->mesh.nox;
    
    E->lmesh.ELX[E->mesh.levmax] = nox-1;
    E->lmesh.ELY[E->mesh.levmax] = noy-1;
    E->lmesh.ELZ[E->mesh.levmax] = noz-1;
    E->lmesh.NOZ[E->mesh.levmax] = noz;
    E->lmesh.NOY[E->mesh.levmax] = noy;
    E->lmesh.NOX[E->mesh.levmax] = nox;
    E->lmesh.NNO[E->mesh.levmax] = nox * noz * noy;
    E->lmesh.NEL[E->mesh.levmax] = (nox-1) * (noz-1) * (noy-1);
    
    for (lev=E->mesh.levmax;lev>=E->mesh.levmin;lev--)  {
      for (j=1;j<=E->sphere.caps_per_proc;j++)  {	
	E->IEN[lev][j] = new IEN [E->lmesh.nel+1];
      }
    }

    for (lev=E->mesh.levmax;lev>=E->mesh.levmin;lev--)  {
      for (j=1;j<=E->sphere.caps_per_proc;j++)  {

	elx = E->lmesh.ELX[lev];
	elz = E->lmesh.ELZ[lev];
	ely = E->lmesh.ELY[lev];
	nox = E->lmesh.NOX[lev];
	noz = E->lmesh.NOZ[lev];
	noy = E->lmesh.NOY[lev];
	nel=E->lmesh.NEL[lev];
	nno=E->lmesh.NNO[lev];
	
	for(r=1;r<=ely;r++)
	  for(q=1;q<=elx;q++)
	    for(p=1;p<=elz;p++)     {
	      element = (r-1)*elx*elz + (q-1)*elz  + p;
	      start = (r-1)*noz*nox + (q-1)*noz + p;
	      for(rr=1;rr<=8;rr++)
		E->IEN[lev][j][element].node[rr]= start
                  + offset[rr].vector[0]
                  + offset[rr].vector[1]*noz
                  + offset[rr].vector[2]*noz*nox;
	    }
      }     /* end for cap j */
    }     /* end loop for lev */

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
    int nox,noy,noz,p,q,r,lev,i,j,rr,k,m;
    int elx,ely,elz,nel,nno,element,start;

    E->parallel.me = 1;
    E->parallel.me_loc[1] = 0;
    E->parallel.me_loc[2] = 0;
    E->parallel.me_loc[3] = 0;
    E->parallel.nprocx = 1;
    E->parallel.nprocy = 1;
    E->parallel.nprocz = 1;

    E->sphere.caps_per_proc = 1;

    E->mesh.levmax = 1;
    E->mesh.levmin = 1;
    
    E->mesh.dof = 3;
    E->mesh.nox = 4;
    E->mesh.noy = 4;
    E->mesh.noz = 3;

    E->mesh.elx = E->mesh.nox - 1;
    E->mesh.ely = E->mesh.noy - 1;
    E->mesh.elz = E->mesh.noz - 1;

    E->lmesh.elx = E->mesh.elx/E->parallel.nprocx;
    E->lmesh.elz = E->mesh.elz/E->parallel.nprocz;
    E->lmesh.ely = E->mesh.ely/E->parallel.nprocy;
    E->lmesh.nox = E->lmesh.elx + 1;
    E->lmesh.noz = E->lmesh.elz + 1;
    E->lmesh.noy = E->lmesh.ely + 1;
    
    E->lmesh.nno = E->lmesh.noz*E->lmesh.nox*E->lmesh.noy;
    E->lmesh.nel = E->lmesh.ely*E->lmesh.elx*E->lmesh.elz;
    E->lmesh.npno = E->lmesh.nel;
    
    noz = E->lmesh.noz;
    noy = E->mesh.noy;
    nox = E->mesh.nox;

    E->lmesh.ELX[E->mesh.levmax] = nox-1;
    E->lmesh.ELY[E->mesh.levmax] = noy-1;
    E->lmesh.ELZ[E->mesh.levmax] = noz-1;
    E->lmesh.NOZ[E->mesh.levmax] = noz;
    E->lmesh.NOY[E->mesh.levmax] = noy;
    E->lmesh.NOX[E->mesh.levmax] = nox;
    E->lmesh.NNO[E->mesh.levmax] = nox * noz * noy;
    E->lmesh.NEL[E->mesh.levmax] = (nox-1) * (noz-1) * (noy-1);
    
    for (lev=E->mesh.levmax;lev>=E->mesh.levmin;lev--)  {
      for (j=1;j<=E->sphere.caps_per_proc;j++)  {	
	E->IEN[lev][j] = new IEN [E->lmesh.nel+1];
      }
    }
    for (lev=E->mesh.levmax;lev>=E->mesh.levmin;lev--)  {
      for (j=1;j<=E->sphere.caps_per_proc;j++)  {	
	elx = E->lmesh.ELX[lev];
	elz = E->lmesh.ELZ[lev];
	ely = E->lmesh.ELY[lev];
	nox = E->lmesh.NOX[lev];
	noz = E->lmesh.NOZ[lev];
	noy = E->lmesh.NOY[lev];
	nel=E->lmesh.NEL[lev];
	nno=E->lmesh.NNO[lev];

	for(r=1;r<=ely;r++)
	  for(q=1;q<=elx;q++)
	    for(p=1;p<=elz;p++)     {
	      element = (r-1)*elx*elz + (q-1)*elz  + p;
	      start = (r-1)*noz*nox + (q-1)*noz + p;
	      for(rr=1;rr<=8;rr++)
		E->IEN[lev][j][element].node[rr]= start
                  + offset[rr].vector[0]
                  + offset[rr].vector[1]*noz
                  + offset[rr].vector[2]*noz*nox;
	    }
      }     /* end for cap j */
    }     /* end loop for lev */

    E->control.theta_max=3.0;
    E->control.theta_min=0.0;
    E->control.fi_max=3.0;
    E->control.fi_min=0.0;
    E->sphere.ro=2.0;
    E->sphere.ri=0.0;

    const int n = E->lmesh.nno;
    for(int m=1;m<=E->sphere.caps_per_proc;m++) {
	for(i=1; i<=E->mesh.dof; i++) {
	  // Don't forget to delete these later
	    E->X[E->mesh.levmax][m][i] = new double [n+1];
	    E->V[m][i] = new float [n+1];
	}
  	E->T[m] = new double [n+1];
    }

    for(m=1;m<=E->sphere.caps_per_proc;m++)
        for(k=1;k<=E->lmesh.noy;k++)
 	  for(j=1;j<=E->lmesh.nox;j++) 
	    for(i=1;i<=E->lmesh.noz;i++)  {
	      int node = i + (j-1)*E->lmesh.noz
			     + (k-1)*E->lmesh.noz*E->lmesh.nox;

		    E->X[E->mesh.levmax][m][1][node] = j-1;
		    E->X[E->mesh.levmax][m][2][node] = k-1;
		    E->X[E->mesh.levmax][m][3][node] = i-1;
		    
 		    E->T[m][node] = E->X[E->mesh.levmax][m][1][node]
 		      + E->X[E->mesh.levmax][m][2][node]
 		      + E->X[E->mesh.levmax][m][3][node];

		    E->V[m][1][node] = E->T[m][node];
		    E->V[m][2][node] = 2.0*E->T[m][node];
		    E->V[m][3][node] = 3.0*E->T[m][node];

//  		    std::cout << "in CoarsereturnE (T, v1,v2,v3): " <<  node << " "
//   			      << E->T[m][node] << " "
//   			      << E->V[m][1][node] << " "
//   			      << E->V[m][2][node] << " "
//   			      << E->V[m][3][node] << " "
//   			      << std::endl;
		}

    PyObject *cobj = PyCObject_FromVoidPtr(E, NULL);
    return Py_BuildValue("O", cobj);
}

// version
// $Id: misc.cc,v 1.8 2003/09/21 22:24:00 ces74 Exp $

// End of file
