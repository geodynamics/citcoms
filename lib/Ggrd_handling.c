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
/*

routines that deal with GMT/netcdf grd I/O as supported through
the ggrd subroutines of the hc package

*/

#include <math.h>
#include "global_defs.h"
#include "parsing.h"
#include "parallel_related.h"
#include "composition_related.h"

#ifdef USE_GGRD

#include "hc.h"			/* ggrd and hc packages */

void ggrd_init_tracer_flavors(struct All_variables *);
int layers_r(struct All_variables *,float );
void myerror(struct All_variables *,char *);
void ggrd_full_temp_init(struct All_variables *);
void ggrd_reg_temp_init(struct All_variables *);
void ggrd_temp_init_general(struct All_variables *,char *);

/* 

assign tracer flavor based on its depth (within top n layers), 
and the grd value


*/
void ggrd_init_tracer_flavors(struct All_variables *E)
{
  int j, kk, number_of_tracers;
  double rad,theta,phi,indbl;
  char char_dummy[1],error[255],gmt_bc[10];
  struct ggrd_gt ggrd_ict[1];	
  /* for dealing with several processors */
  MPI_Status mpi_stat;
  int mpi_rc;  
  int mpi_inmsg, mpi_success_message = 1;

  report(E,"ggrd_init_tracer_flavors: ggrd mat init");

  /* 
     are we global?
  */
  if (E->parallel.nprocxy == 12){
    /* use GMT's geographic boundary conditions */
    sprintf(gmt_bc,"-Lg");
  }else{			/* regional */
    sprintf(gmt_bc,"");
  }
    
  /* 
     initialize the ggrd control 
  */
  if(E->parallel.me > 0){	
    /* wait for previous processor */
    mpi_rc = MPI_Recv(&mpi_inmsg, 1, MPI_INT, (E->parallel.me-1), 
		      0, MPI_COMM_WORLD, &mpi_stat);
  }
  if(ggrd_grdtrack_init_general(FALSE,E->trace.ggrd_file,
				char_dummy,gmt_bc,
				ggrd_ict,FALSE,FALSE)){
    myerror(E,"ggrd tracer init error");
  }
  if(E->parallel.me <  E->parallel.nproc-1){ 
    /* tell the next proc to go ahead */
    mpi_rc = MPI_Send(&mpi_success_message, 1, 
		      MPI_INT, (E->parallel.me+1), 0, MPI_COMM_WORLD);
  }else{
    report(E,"ggrd_init_tracer_flavors: last processor done with ggrd mat init");
  }
  /* init done */
  
  /* assign values to each tracer based on grd file */
  for (j=1;j<=E->sphere.caps_per_proc;j++) {
    number_of_tracers = E->trace.ntracers[j];
    for (kk=1;kk <= number_of_tracers;kk++) {
      rad = E->trace.basicq[j][2][kk]; /* tracer radius */
      if(layers_r(E,rad) <= E->trace.ggrd_layers){
	/* 
	   in top layers 
	*/
	phi =   E->trace.basicq[j][1][kk]; 
	theta = E->trace.basicq[j][0][kk]; 
	/* interpolate from grid */
	if(!ggrd_grdtrack_interpolate_tp((double)theta,(double)phi,
					 ggrd_ict,&indbl,FALSE)){
	  snprintf(error,255,"read_mat_from_ggrd_file: interpolation error at theta: %g phi: %g",
		   theta,phi);
	  myerror(E,error);
	}
	/* limit to 0 or 1 */
	if(indbl < .5)
	  indbl = 0.0;
	else
	  indbl = 1.0;
	E->trace.extraq[j][0][kk]= indbl;
      }else{
	/* below */
	E->trace.extraq[j][0][kk]=0.0;
      }
    }
  }

  /* free grd structure */
  ggrd_grdtrack_free_gstruc(ggrd_ict);
  report(E,"ggrd tracer init done");
}

void ggrd_full_temp_init(struct All_variables *E)
{
  ggrd_temp_init_general(E,"-L");
}
void ggrd_reg_temp_init(struct All_variables *E)
{
  ggrd_temp_init_general(E,"");
}



/* 

initialize temperatures from grd files for spherical geometry

*/

void ggrd_temp_init_general(struct All_variables *E,char *gmtflag)
{
  
  MPI_Status mpi_stat;
  int mpi_rc;
  int mpi_inmsg, mpi_success_message = 1;
  double temp1,tbot,tgrad,tmean,tadd,rho_prem;

  int i,j,k,m,node,noxnoz,nox,noy,noz;

  noy=E->lmesh.noy;
  nox=E->lmesh.nox;
  noz=E->lmesh.noz;
  noxnoz = nox * noz;

  if(E->parallel.me == 0)  
    fprintf(stderr,"ggrd_temp_init_general: using GMT grd files for temperatures, gmtflag: %s\n",gmtflag);
  /* 
     
  
  read in tempeatures/density from GMT grd files
  
  
  */
  /* 
     
  begin MPI synchronization part
  
  */
  if(E->parallel.me > 0){
    /* 
       wait for the previous processor 
    */
    mpi_rc = MPI_Recv(&mpi_inmsg, 1, MPI_INT, (E->parallel.me-1), 
		      0, MPI_COMM_WORLD, &mpi_stat);
  }
  
  if(E->convection.ggrd_tinit_scale_with_prem){/* initialize PREM */
    if(prem_read_model(E->convection.prem.model_filename,
		       &E->convection.prem, (E->parallel.me==0)))
      myerror(E,"PREM init error");
  }
  /* 
     initialize the GMT grid files 
  */
  E->convection.ggrd_tinit_d[0].init = FALSE;
  if(ggrd_grdtrack_init_general(TRUE,E->convection.ggrd_tinit_gfile, 
				E->convection.ggrd_tinit_dfile,gmtflag, 
				E->convection.ggrd_tinit_d,(E->parallel.me == 0),
				FALSE))
    myerror(E,"grd init error");
  if(E->parallel.me <  E->parallel.nproc-1){
    /* tell the next processor to go ahead with the init step	*/
    mpi_rc = MPI_Send(&mpi_success_message, 1, MPI_INT, (E->parallel.me+1), 0, MPI_COMM_WORLD);
  }else{
    fprintf(stderr,"ggrd_temp_init_general: last processor (%i) done with grd init\n",
	    E->parallel.me);
  }     
  /* 
     
  interpolate densities to temperature given PREM variations
  
  */
  if(E->mesh.bottbc == 1){
    /* bottom has specified temperature */
    tbot =  E->control.TBCbotval;
  }else{
    /* 
       bottom has specified heat flux start with unity bottom temperature
    */ 
    tbot = 1.0;
  }
  /* 
     mean temp is (top+bot)/2 + offset 
  */
  tmean = (tbot + E->control.TBCtopval)/2.0 +  E->convection.ggrd_tinit_offset;


  for(m=1;m <= E->sphere.caps_per_proc;m++)
    for(i=1;i <= noy;i++)  
      for(j=1;j <= nox;j++) 
	for(k=1;k <= noz;k++)  {
	  /* node numbers */
	  node=k+(j-1)*noz+(i-1)*noxnoz;

	  /* 
	     get interpolated velocity anomaly 
	  */
	  if(!ggrd_grdtrack_interpolate_rtp((double)E->sx[m][3][node],
					    (double)E->sx[m][1][node],
					    (double)E->sx[m][2][node],
					    E->convection.ggrd_tinit_d,&tadd,
					    FALSE))
	    myerror(E,"ggrd_temp_init_general");
	  if(E->convection.ggrd_tinit_scale_with_prem){
	    /* 
	       get the PREM density at r for additional scaling  
	    */
	    prem_get_rho(&rho_prem,(double)E->sx[m][3][node],&E->convection.prem);
	    if(rho_prem < 3200.0)
	      rho_prem = 3200.0; /* we don't want the density of water */
	    /* 
	       assign temperature 
	    */
	    E->T[m][node] = tmean + tadd * E->convection.ggrd_tinit_scale * 
	      rho_prem / E->data.density;
	  }else{
	    /* no PREM scaling */
	    E->T[m][node] = tmean + tadd * E->convection.ggrd_tinit_scale;
	  }

	  if(E->convection.ggrd_tinit_limit_trange){
	    /* limit to 0 < T < 1 ?*/
	    E->T[m][node] = min(max(E->T[m][node], 0.0),1.0);
	  }
	  //fprintf(stderr,"z: %11g T: %11g\n",E->sx[m][3][node],E->T[m][node]);
	  if(E->convection.ggrd_tinit_override_tbc){
	    if((k == 1) && (E->mesh.bottbc == 1)){ /* bottom TBC */
	      E->sphere.cap[m].TB[1][node] =  E->T[m][node];
	      E->sphere.cap[m].TB[2][node] =  E->T[m][node];
	      E->sphere.cap[m].TB[3][node] =  E->T[m][node];
	      //fprintf(stderr,"z: %11g TBB: %11g\n",E->sx[m][3][node],E->T[m][node]);
	    }
	    if((k == noz) && (E->mesh.toptbc == 1)){ /* top TBC */
	      E->sphere.cap[m].TB[1][node] =  E->T[m][node];
	      E->sphere.cap[m].TB[2][node] =  E->T[m][node];
	      E->sphere.cap[m].TB[3][node] =  E->T[m][node];
	      //fprintf(stderr,"z: %11g TBT: %11g\n",E->sx[m][3][node],E->T[m][node]);
	    }
	  }

	  

	}
  /* 
     free the structure, not needed anymore since T should now
     change internally
  */
  ggrd_grdtrack_free_gstruc(E->convection.ggrd_tinit_d);
  /* 
     end temperature/density from GMT grd init
  */
  temperatures_conform_bcs(E);
}



#endif


