/*  Here are the routines which process the results of each velocity solution, and call
    the relevant output routines. At this point, the velocity and pressure fields have
    been calculated and stored at the nodes. The only properties of the velocity field
    which are already known are those required to check convergence of the iterative
    scheme and so on. */

#include <math.h>
#include <sys/types.h>
#include <stdlib.h> /* for "system" command */

#include "element_definitions.h"
#include "global_defs.h"

void process_new_velocity(E,ii)
    struct All_variables *E;
    int ii;
{
    void output_velo_related();
    void get_STD_topo();
    void get_CBF_topo();
    void parallel_process_sync();

    int m,i,it;


    E->monitor.length_scale = E->data.layer_km/E->mesh.layer[2]; /* km */
    E->monitor.time_scale = pow(E->data.layer_km*1000.0,2.0)/   /* Million years */
      (E->data.therm_diff*3600.0*24.0*365.25*1.0e6);

    if ( (ii == 0) || ((ii % E->control.record_every) == 0)
		|| E->control.DIRECTII)     {
      get_STD_topo(E,E->slice.tpg,E->slice.tpgb,E->slice.divg,E->slice.vort,ii);
      parallel_process_sync();
      output_velo_related(E,ii);         /* also topo */
    }

    return;
}

void process_output_field(E,ii)
    struct All_variables *E;
    int ii;
{
  void output_velo_related();
  void parallel_process_sync();

  int ll,mm,m,i,it,j,k,snode,node,lev;
  float x,*power[100],z,th,fi,*TG;
  FILE *fp1;
  char output_file[255];

  lev = E->mesh.levmax;

  return;
}


/* ===============================================   */

void get_surface_velo(E, SV,m)
  struct All_variables *E;
  float *SV;
  int m;
  {

  int el,els,i,node,lev;
  char output_file[255];
  FILE *fp;

/*   const int dims=E->mesh.nsd; */
/*   const int ends=enodes[dims]; */
/*   const int nno=E->lmesh.nno; */

/*   lev = E->mesh.levmax; */

/*   for(m=1;m<=E->sphere.caps_per_proc;m++)     */
/*     for (node=1;node<=nno;node++) */
/*       if (node%E->lmesh.noz==0)   { */
/*         i = node/E->lmesh.noz; */
/*         SV[(i-1)*2+1] = E->sphere.cap[m].V[1][node]; */
/*         SV[(i-1)*2+2] = E->sphere.cap[m].V[2][node]; */
/*       } */

  return;
  }

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
