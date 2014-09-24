/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 *<LicenseText>
 *
 * CitcomS by Louis Moresi, Shijie Zhong, Lijie Han, Eh Tan,
 * Clint Conrad, Michael Gurnis, and Eun-seo Choi.
 * Copyright (C) 1994-2005, California Institute of Technology.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *</LicenseText>
 *
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
/*  Here are the routines which process the results of each buoyancy solution, and call
    any relevant output routines. Much of the information has probably been output along
    with the velocity field. (So the velocity vectors and other data are fully in sync).
    However, heat fluxes and temperature averages are calculated here (even when they
    get output the next time around the velocity solver);
    */


#include "element_definitions.h"
#include "global_defs.h"
#include <math.h>		/* for sqrt */

void parallel_process_termination(void);

void post_processing(struct All_variables *E)
{
}



/* ===================
    Surface heat flux
   =================== */

void heat_flux(E)
    struct All_variables *E;
{
    int m,e,el,i,j,node,lnode,nz;
    float *flux,*SU[NCS],*RU[NCS];
    float VV[4][9],u[9],T[9],dTdz[9],rho[9],area,uT;
    float *sum_h;

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

    flux = (float *) malloc((1+nno)*sizeof(float));

    for(i=1;i<=nno;i++)   {
      flux[i] = 0.0;
      }

    for(e=1;e<=E->lmesh.nel;e++) {

      velo_from_element(E,VV,e,sphere_key);

      for(i=1;i<=vpts;i++)   {
        u[i] = 0.0;
        T[i] = 0.0;
        dTdz[i] = 0.0;
        rho[i] = 0.0;
        for(j=1;j<=ends;j++)  {
          nz = ((E->ien[e].node[j]-1) % E->lmesh.noz)+1;
          rho[i] += E->refstate.rho[nz]*E->N.vpt[GNVINDEX(j,i)];
          u[i] += VV[3][j]*E->N.vpt[GNVINDEX(j,i)];
          T[i] += E->T[E->ien[e].node[j]]*E->N.vpt[GNVINDEX(j,i)];
          dTdz[i] += -E->T[E->ien[e].node[j]]*E->gNX[e].vpt[GNVXINDEX(2,j,i)];
          }
        }

      uT = 0.0;
      area = 0.0;
      for(i=1;i<=vpts;i++)   {
        /* XXX: missing unit conversion, heat capacity and thermal conductivity */
        uT += rho[i]*u[i]*T[i]*E->gDA[e].vpt[i] + dTdz[i]*E->gDA[e].vpt[i];
        }

      uT /= E->eco[e].area;

      for(j=1;j<=ends;j++)
        flux[E->ien[e].node[j]] += uT*E->TWW[lev][e].node[j];

      }             /* end of e */


  (E->exchange_node_f)(E,flux,lev);

   for(i=1;i<=nno;i++)
     flux[i] *= E->MASS[lev][i];

  if (E->parallel.me_loc[3]==E->parallel.nprocz-1)
    for(i=1;i<=E->lmesh.nsf;i++)
      E->slice.shflux[i]=2*flux[E->surf_node[i]]-flux[E->surf_node[i]-1];

  if (E->parallel.me_loc[3]==0)
    for(i=1;i<=E->lmesh.nsf;i++)
      E->slice.bhflux[i] = 2*flux[E->surf_node[i]-E->lmesh.noz+1]
                              - flux[E->surf_node[i]-E->lmesh.noz+2];

    for(e=1;e<=E->lmesh.snel;e++) {
         uT =(E->slice.shflux[E->sien[e].node[1]] +
              E->slice.shflux[E->sien[e].node[2]] +
              E->slice.shflux[E->sien[e].node[3]] +
              E->slice.shflux[E->sien[e].node[4]])*0.25;
         el = e*E->lmesh.elz;
         sum_h[0] += uT*E->eco[el].area;
         sum_h[1] += E->eco[el].area;

         uT =(E->slice.bhflux[E->sien[e].node[1]] +
              E->slice.bhflux[E->sien[e].node[2]] +
              E->slice.bhflux[E->sien[e].node[3]] +
              E->slice.bhflux[E->sien[e].node[4]])*0.25;
         el = (e-1)*E->lmesh.elz+1;
         sum_h[2] += uT*E->eco[el].area;
         sum_h[3] += E->eco[el].area;
         }

  sum_across_surface(E,sum_h,4);

  if (E->parallel.me_loc[3]==E->parallel.nprocz-1)   {
    sum_h[0] = sum_h[0]/sum_h[1];
    /*     if (E->control.verbose && E->parallel.me==E->parallel.nprocz-1) {
	     fprintf(E->fp_out,"surface heat flux= %f %f\n",sum_h[0],E->monitor.elapsed_time);
             fflush(E->fp_out);
    } */
    if (E->parallel.me==E->parallel.nprocz-1) {
      fprintf(stderr,"surface heat flux= %f\n",sum_h[0]);
      //fprintf(E->fp,"surface heat flux= %f\n",sum_h[0]); //commented out because E->fp is only on CPU 0 

      if(E->output.write_q_files > 0){
	/* format: time heat_flow sqrt(v.v)  */
	fprintf(E->output.fpqt,"%13.5e %13.5e %13.5e\n",E->monitor.elapsed_time,sum_h[0],sqrt(E->monitor.vdotv));
	fflush(E->output.fpqt);
      }
    }
  }

  if (E->parallel.me_loc[3]==0)    {
    sum_h[2] = sum_h[2]/sum_h[3];
/*     if (E->control.verbose && E->parallel.me==0) fprintf(E->fp_out,"bottom heat flux= %f %f\n",sum_h[2],E->monitor.elapsed_time); */
    if (E->parallel.me==0) {
      fprintf(stderr,"bottom heat flux= %f\n",sum_h[2]);
      fprintf(E->fp,"bottom heat flux= %f\n",sum_h[2]);
      if(E->output.write_q_files > 0){
	fprintf(E->output.fpqb,"%13.5e %13.5e %13.5e\n",
		E->monitor.elapsed_time,sum_h[2],sqrt(E->monitor.vdotv));
	fflush(E->output.fpqb);
      }

    }
  }


  free((void *)flux);
  free((void *)sum_h);

}



/*
  compute horizontal average of temperature, composition and rms velocity
*/
void compute_horiz_avg(struct All_variables *E)
{
    void return_horiz_ave_f();

    int m, n, i;
    float *S1,*S2,*S3;

	S1 = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
	S2 = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
	S3 = (float *)malloc((E->lmesh.nno+1)*sizeof(float));

	for(i=1;i<=E->lmesh.nno;i++) {
	    S1[i] = E->T[i];
	    S2[i] = E->sphere.cap[1].V[1][i]*E->sphere.cap[1].V[1][i]
          	+ E->sphere.cap[1].V[2][i]*E->sphere.cap[1].V[2][i];
	    S3[i] = E->sphere.cap[1].V[3][i]*E->sphere.cap[1].V[3][i];
	}

    return_horiz_ave_f(E,S1,E->Have.T);
    return_horiz_ave_f(E,S2,E->Have.V[1]);
    return_horiz_ave_f(E,S3,E->Have.V[2]);

    if (E->composition.on) {
        for(n=0; n<E->composition.ncomp; n++) {
            for(i=1;i<=E->lmesh.nno;i++)
                S1[i] = E->composition.comp_node[n][i];
            return_horiz_ave_f(E,S1,E->Have.C[n]);
        }
    }

	free((void *)S1);
	free((void *)S2);
	free((void *)S3);

    for (i=1;i<=E->lmesh.noz;i++) {
	E->Have.V[1][i] = sqrt(E->Have.V[1][i]);
	E->Have.V[2][i] = sqrt(E->Have.V[2][i]);
    }

}
