/*  Here are the routines which process the results of each buoyancy solution, and call
    any relevant output routines. Much of the information has probably been output along
    with the velocity field. (So the velocity vectors and other data are fully in sync).
    However, heat fluxes and temperature averages are calculated here (even when they
    get output the next time around the velocity solver);
    */


#include "element_definitions.h"
#include "global_defs.h"



void post_processing(struct All_variables *E)
{
  return;
}



/* ===================
    Surface heat flux
   =================== */

void heat_flux(E)
     struct All_variables *E;
{
  int m,e,el,i,j,node,lnode;
  float *flux[NCS],*SU[NCS],*RU[NCS];
  float VV[4][9],u[9],T[9],dTdz[9],area,uT;
  float *sum_h;
  double rtf[4][9];

  struct Shape_function GN;
  struct Shape_function_dA dOmega;
  struct Shape_function_dx GNx;
  void get_global_shape_fn();
  void exchange_node_f();

  void velo_from_element();
  void sum_across_surface();
  void return_horiz_ave();
  void return_horiz_ave_f();

  const int dims=E->mesh.nsd,dofs=E->mesh.dof;
  const int vpts=vpoints[dims];
  const int ppts=ppoints[dims];
  const int ends=enodes[dims];
  const int nno=E->lmesh.nno;
  const int lev = E->mesh.levmax;
  const int sphere_key=1;


  sum_h = (float *) malloc((5)*sizeof(float));
  for(i=0;i<=4;i++)
    sum_h[i] = 0.0;

  for(m=1;m<=E->sphere.caps_per_proc;m++) {

    flux[m] = (float *) malloc((1+nno)*sizeof(float));

    for(i=1;i<=nno;i++)   {
      flux[m][i] = 0.0;
    }

    for(e=1;e<=E->lmesh.nel;e++) {
      get_global_shape_fn(E,e,&GN,&GNx,&dOmega,0,sphere_key,rtf,lev,m);

      velo_from_element(E,VV,m,e,sphere_key);

      for(i=1;i<=vpts;i++)   {
	u[i] = 0.0;
	T[i] = 0.0;
	dTdz[i] = 0.0;
	for(j=1;j<=ends;j++)  {
	  u[i] += VV[3][j]*E->N.vpt[GNVINDEX(j,i)];
	  T[i] += E->T[m][E->ien[m][e].node[j]]*E->N.vpt[GNVINDEX(j,i)];
	  dTdz[i] += -E->T[m][E->ien[m][e].node[j]]*GNx.vpt[GNVXINDEX(2,j,i)];
	}
      }

      uT = 0.0;
      area = 0.0;
      for(i=1;i<=vpts;i++)   {
	uT += u[i]*T[i]*dOmega.vpt[i] + dTdz[i]*dOmega.vpt[i];
      }

      uT /= E->eco[m][e].area;

      for(j=1;j<=ends;j++)
	flux[m][E->ien[m][e].node[j]] += uT*E->TWW[lev][m][e].node[j];

    }             /* end of e */
  }             /* end of m */

  /*
    exchange_node_f(E,flux,lev);
  */

  exchange_node_f(E,flux,lev);

  for(m=1;m<=E->sphere.caps_per_proc;m++)
    for(i=1;i<=nno;i++)
      flux[m][i] *= E->MASS[lev][m][i];

  /*
    if (E->parallel.me_loc[3]==E->parallel.nprocz-1)
  */
  if (E->parallel.me_loc[3]==E->parallel.nprocz-1)

    for(m=1;m<=E->sphere.caps_per_proc;m++)
      for(i=1;i<=E->lmesh.nsf;i++)
	E->slice.shflux[m][i]=2*flux[m][E->surf_node[m][i]]-flux[m][E->surf_node[m][i]-1];
  /*
    if (E->parallel.me_loc[3]==0)
  */

  if (E->parallel.me_loc[3]==0)

    for(m=1;m<=E->sphere.caps_per_proc;m++)
      for(i=1;i<=E->lmesh.nsf;i++)
	E->slice.bhflux[m][i] = 2*flux[m][E->surf_node[m][i]-E->lmesh.noz+1]
	  - flux[m][E->surf_node[m][i]-E->lmesh.noz+2];

  for(m=1;m<=E->sphere.caps_per_proc;m++)
    for(e=1;e<=E->lmesh.snel;e++) {
      uT =(E->slice.shflux[m][E->sien[m][e].node[1]] +
	   E->slice.shflux[m][E->sien[m][e].node[2]] +
	   E->slice.shflux[m][E->sien[m][e].node[3]] +
	   E->slice.shflux[m][E->sien[m][e].node[4]])*0.25;
      el = e*E->lmesh.elz;
      sum_h[0] += uT*E->eco[m][el].area;
      sum_h[1] += E->eco[m][el].area;

      uT =(E->slice.bhflux[m][E->sien[m][e].node[1]] +
	   E->slice.bhflux[m][E->sien[m][e].node[2]] +
	   E->slice.bhflux[m][E->sien[m][e].node[3]] +
	   E->slice.bhflux[m][E->sien[m][e].node[4]])*0.25;
      el = (e-1)*E->lmesh.elz+1;
      sum_h[2] += uT*E->eco[m][el].area;
      sum_h[3] += E->eco[m][el].area;
    }

  sum_across_surface(E,sum_h,4);

  /*
    if (E->parallel.me_loc[3]==E->parallel.nprocz-1)   {
    sum_h[0] = sum_h[0]/sum_h[1];
    if (E->parallel.me==E->parallel.nprocz-1) fprintf(E->fp_out,"surface heat flux= %f %f\n",sum_h[0],E->monitor.elapsed_time);
    if (E->parallel.me==E->parallel.nprocz-1) fprintf(stderr,"surface heat flux= %f\n",sum_h[0]);
    }

    if (E->parallel.me_loc[3]==0)    {
    sum_h[2] = sum_h[2]/sum_h[3];
    if (E->parallel.me==0) fprintf(E->fp_out,"bottom heat flux= %f %f\n",sum_h[2],E->monitor.elapsed_time);
    if (E->parallel.me==0) fprintf(stderr,"bottom heat flux= %f\n",sum_h[2]);
    }
  */

  if (E->parallel.me_loc[3]==E->parallel.nprocz-1) {
    sum_h[0] = sum_h[0]/sum_h[1];
    if (E->parallel.me==E->parallel.nprocz-1) fprintf(E->fp,"surface heat flux= %f %f\n",sum_h[0],E->monitor.elapsed_time);
    if (E->parallel.me==E->parallel.nprocz-1) fprintf(stderr,"surface heat flux= %f\n",sum_h[0]);

  }

  if (E->parallel.me_loc[3]==0)    {
    sum_h[2] = sum_h[2]/sum_h[3];
    if (E->parallel.me==0) fprintf(E->fp,"bottom heat flux= %f %f\n",sum_h[2],E->monitor.elapsed_time);
    if (E->parallel.me==0) fprintf(stderr,"bottom heat flux= %f\n",sum_h[2]);
  }
  for(m=1;m<=E->sphere.caps_per_proc;m++)
    free((void *)flux[m]);
  free((void *)sum_h);

  return;
}
