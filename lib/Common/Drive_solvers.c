#include <math.h>
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"

void general_stokes_solver(E)
     struct All_variables *E;

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
    float *delta_U[NCS];
    double Udot_mag, dUdot_mag;
    double CPU_time0(),time;
    int m,count,i,j,k;

    static float *oldU[NCS];
    static int visits=0;

    const int nno = E->lmesh.nno;
    const int nel = E->lmesh.nel;
    const int nnov = E->lmesh.nnov;
    const int neq = E->lmesh.neq;
    const int vpts = vpoints[E->mesh.nsd];
    const int dims = E->mesh.nsd;
    const int addi_dof = additional_dof[dims];

    if(visits==0) {
      for (m=1;m<=E->sphere.caps_per_proc;m++)  {
        oldU[m] = (float *)malloc((neq+2)*sizeof(float));
	for(i=0;i<=neq;i++) 
	    oldU[m][i]=0.0;
        }
    visits ++;
    }

    for (m=1;m<=E->sphere.caps_per_proc;m++)  {
      delta_U[m] = (float *)malloc((neq+2)*sizeof(float));
      }
     
    /* FIRST store the old velocity field */
    E->monitor.elapsed_time_vsoln = E->monitor.elapsed_time;

    if(E->parallel.me==0) time=CPU_time0();

    velocities_conform_bcs(E,E->U);

    assemble_forces(E,0);
 
    Udot_mag=dUdot_mag=0.0;
    count=1;
          
    do  {

      if(E->viscosity.update_allowed)
         get_system_viscosity(E,1,E->EVI[E->mesh.levmax],E->VI[E->mesh.levmax]);

      construct_stiffness_B_matrix(E);
      solve_constrained_flow_iterative(E);	

/*      Udot_mag = kineticE_radial(E,E->U,E->mesh.levmax);
      if(E->parallel.me==0)
          fprintf(E->fp_out,"%g %g \n",E->monitor.elapsed_time,Udot_mag);
      fflush(E->fp_out);

      if(E->parallel.me==0)
          fprintf(stderr,"kinetic energy= %g time4= %g seconds \n",Udot_mag,CPU_time0()-time);
*/
      if (  E->viscosity.SDEPV  )   {
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
        count++;
        }         /* end for SDEPV   */

      } while((count < 50) && (dUdot_mag>E->viscosity.sdepv_misfit) && E->viscosity.SDEPV);
    	
  for (m=1;m<=E->sphere.caps_per_proc;m++)  {
    free((void *) delta_U[m]);
    }
      
  return;
}
