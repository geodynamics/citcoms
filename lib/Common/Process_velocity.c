/*  Here are the routines which process the results of each velocity solution, and call
    the relevant output routines. At this point, the velocity and pressure fields have
    been calculated and stored at the nodes. The only properties of the velocity field
    which are already known are those required to check convergence of the iterative
    scheme and so on. */


#include "element_definitions.h"
#include "global_defs.h"


/* ===============================================   */

void get_ele_visc(E, EV,m)
  struct All_variables *E;
  float *EV;
  int m;
  {

  int el,j,lev;

  const int nel=E->lmesh.nel;
  const int vpts=vpoints[E->mesh.nsd];

  lev = E->mesh.levmax;

  for(m=1;m<=E->sphere.caps_per_proc;m++)
    for (el=1;el<=nel;el++)   {
      EV[el] = 0.0;
      for (j=1;j<=vpts;j++)   {
        EV[el] +=  E->EVI[lev][m][(el-1)*vpts+j];
      }

      EV[el] /= vpts;
      }

  return;
  }

void get_surf_stress(E,SXX,SYY,SZZ,SXY,SXZ,SZY)
  struct All_variables *E;
  float **SXX,**SYY,**SZZ,**SXY,**SXZ,**SZY;
  {
  int m,i,node,stride;

  stride = E->lmesh.nsf*6;

  for(m=1;m<=E->sphere.caps_per_proc;m++)
    for (node=1;node<=E->lmesh.nno;node++)
      if ( (node%E->lmesh.noz)==0 )  {
        i = node/E->lmesh.noz;
        E->stress[m][(i-1)*6+1] = SXX[m][node];
        E->stress[m][(i-1)*6+2] = SZZ[m][node];
        E->stress[m][(i-1)*6+3] = SYY[m][node];
        E->stress[m][(i-1)*6+4] = SXY[m][node];
        E->stress[m][(i-1)*6+5] = SXZ[m][node];
        E->stress[m][(i-1)*6+6] = SZY[m][node];
        }
     else if ( ((node+1)%E->lmesh.noz)==0 )  {
        i = (node+1)/E->lmesh.noz;
        E->stress[m][stride+(i-1)*6+1] = SXX[m][node];
        E->stress[m][stride+(i-1)*6+2] = SZZ[m][node];
        E->stress[m][stride+(i-1)*6+3] = SYY[m][node];
        E->stress[m][stride+(i-1)*6+4] = SXY[m][node];
        E->stress[m][stride+(i-1)*6+5] = SXZ[m][node];
        E->stress[m][stride+(i-1)*6+6] = SZY[m][node];
        }

  return;
  }

