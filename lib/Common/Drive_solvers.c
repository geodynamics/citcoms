#include <math.h>
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"
#include "drive_solvers.h"

float global_fvdot();
float vnorm_nonnewt();


//***********************************************************

void general_stokes_solver_setup(struct All_variables *E)
{
  int i, m;

  if (E->control.NMULTIGRID || E->control.NASSEMBLE)
    construct_node_maps(E);
  else
    for (i=E->mesh.gridmin;i<=E->mesh.gridmax;i++)
      for (m=1;m<=E->sphere.caps_per_proc;m++)
	E->elt_k[i][m]=(struct EK *)malloc((E->lmesh.NEL[i]+1)*sizeof(struct EK));


  return;
}




void general_stokes_solver(struct All_variables *E)
{
  void construct_stiffness_B_matrix();
  void velocities_conform_bcs();
  void assemble_forces();
  void sphere_harmonics_layer();
  double global_vdot(),kineticE_radial();
  float global_fvdot();
  float vnorm_nonnewt();
  void get_system_viscosity();

  float vmag;

  double Udot_mag, dUdot_mag;
  int m,count,i,j,k;

  float *oldU[NCS], *delta_U[NCS];

  const int nno = E->lmesh.nno;
  const int nel = E->lmesh.nel;
  const int nnov = E->lmesh.nnov;
  const int neq = E->lmesh.neq;
  const int vpts = vpoints[E->mesh.nsd];
  const int dims = E->mesh.nsd;
  const int addi_dof = additional_dof[dims];

  E->monitor.elapsed_time_vsoln = E->monitor.elapsed_time;

  velocities_conform_bcs(E,E->U);

  assemble_forces(E,0);

  if(E->monitor.solution_cycles==0 || E->viscosity.update_allowed) {
    get_system_viscosity(E,1,E->EVI[E->mesh.levmax],E->VI[E->mesh.levmax]);
    construct_stiffness_B_matrix(E);
  }

  solve_constrained_flow_iterative(E);

  if (E->viscosity.SDEPV) {

    for (m=1;m<=E->sphere.caps_per_proc;m++)  {
      delta_U[m] = (float *)malloc((neq+2)*sizeof(float));
      oldU[m] = (float *)malloc((neq+2)*sizeof(float));
      for(i=0;i<=neq;i++)
	oldU[m][i]=0.0;
    }

    Udot_mag=dUdot_mag=0.0;
    count=1;

    while (1) {

      for (m=1;m<=E->sphere.caps_per_proc;m++)
	for (i=0;i<neq;i++) {
	  delta_U[m][i] = E->U[m][i] - oldU[m][i];
	  oldU[m][i] = E->U[m][i];
	}

      Udot_mag  = sqrt(global_fvdot(E,oldU,oldU,E->mesh.levmax));
      dUdot_mag = vnorm_nonnewt(E,delta_U,oldU,E->mesh.levmax);

      if(E->parallel.me==0){
	fprintf(stderr,"Stress dependent viscosity: DUdot = %.4e (%.4e) for iteration %d\n",dUdot_mag,Udot_mag,count);
	fprintf(E->fp,"Stress dependent viscosity: DUdot = %.4e (%.4e) for iteration %d\n",dUdot_mag,Udot_mag,count);
	fflush(E->fp);
      }

      if (count>50 || dUdot_mag>E->viscosity.sdepv_misfit)
	break;

      get_system_viscosity(E,1,E->EVI[E->mesh.levmax],E->VI[E->mesh.levmax]);
      construct_stiffness_B_matrix(E);
      solve_constrained_flow_iterative(E);

      count++;

    } //end while

    for (m=1;m<=E->sphere.caps_per_proc;m++)  {
      free((void *) oldU[m]);
      free((void *) delta_U[m]);
    }

  } //end if SDEPV

  return;
}
