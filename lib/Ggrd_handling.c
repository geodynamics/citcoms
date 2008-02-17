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
#include "element_definitions.h"

#ifdef USE_GGRD

#include "hc.h"			/* ggrd and hc packages */

void ggrd_init_tracer_flavors(struct All_variables *);
int layers_r(struct All_variables *,float );
void myerror(struct All_variables *,char *);
void ggrd_full_temp_init(struct All_variables *);
void ggrd_reg_temp_init(struct All_variables *);
void ggrd_temp_init_general(struct All_variables *,char *);
void ggrd_read_mat_from_file(struct All_variables *, int );
void xyz2rtp(float ,float ,float ,float *);
float find_age_in_MY();

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
	  snprintf(error,255,"ggrd_init_tracer_flavors: interpolation error at theta: %g phi: %g",
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
	E->trace.extraq[j][0][kk] = 0.0;
      }
    }
  }

  /* free grd structure */
  ggrd_grdtrack_free_gstruc(ggrd_ict);
  report(E,"ggrd tracer init done");
  if(E->parallel.me == 0)
    fprintf(stderr,"ggrd tracer init OK\n");
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
  
  if(E->control.ggrd.temp_init.scale_with_prem){/* initialize PREM */
    if(prem_read_model(E->control.ggrd.temp_init.prem.model_filename,
		       &E->control.ggrd.temp_init.prem, (E->parallel.me == 0)))
      myerror(E,"PREM init error");
  }
  /* 
     initialize the GMT grid files 
  */
  E->control.ggrd.temp_init.d[0].init = FALSE;
  if(ggrd_grdtrack_init_general(TRUE,E->control.ggrd.temp_init.gfile, 
				E->control.ggrd.temp_init.dfile,gmtflag, 
				E->control.ggrd.temp_init.d,(E->parallel.me == 0),
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
  tmean = (tbot + E->control.TBCtopval)/2.0 +  E->control.ggrd.temp_init.offset;


  for(m=1;m <= E->sphere.caps_per_proc;m++)
    for(i=1;i <= noy;i++)  
      for(j=1;j <= nox;j++) 
	for(k=1;k <= noz;k++)  {
	  /* node numbers */
	  node=k+(j-1)*noz+(i-1)*noxnoz;

	  /* 
	     get interpolated velocity anomaly 
	  */
	  if(!ggrd_grdtrack_interpolate_rtp((double)E->sx[m][3][node],(double)E->sx[m][1][node],
					    (double)E->sx[m][2][node],
					    E->control.ggrd.temp_init.d,&tadd,
					    FALSE))
	    myerror(E,"ggrd_temp_init_general");
	  if(E->control.ggrd.temp_init.scale_with_prem){
	    /* 
	       get the PREM density at r for additional scaling  
	    */
	    prem_get_rho(&rho_prem,(double)E->sx[m][3][node],&E->control.ggrd.temp_init.prem);
	    if(rho_prem < 3200.0)
	      rho_prem = 3200.0; /* we don't want the density of water */
	    /* 
	       assign temperature 
	    */
	    E->T[m][node] = tmean + tadd * E->control.ggrd.temp_init.scale * 
	      rho_prem / E->data.density;
	  }else{
	    /* no PREM scaling */
	    E->T[m][node] = tmean + tadd * E->control.ggrd.temp_init.scale;
	  }

	  if(E->control.ggrd.temp_init.limit_trange){
	    /* limit to 0 < T < 1 ?*/
	    E->T[m][node] = min(max(E->T[m][node], 0.0),1.0);
	  }
	  //fprintf(stderr,"z: %11g T: %11g\n",E->sx[m][3][node],E->T[m][node]);
	  if(E->control.ggrd.temp_init.override_tbc){
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
  ggrd_grdtrack_free_gstruc(E->control.ggrd.temp_init.d);
  /* 
     end temperature/density from GMT grd init
  */
  temperatures_conform_bcs(E);
}

/* 


read in material, i.e. viscosity prefactor from ggrd file, this will get assigned if 

layer <=  E->control.ggrd.mat_control



 */
void ggrd_read_mat_from_file(struct All_variables *E, int is_global)
{
  MPI_Status mpi_stat;
  int mpi_rc;
  int mpi_inmsg, mpi_success_message = 1;
  int m,el,i,j,k,inode,i1,i2,elxlz,elxlylz,ind;
  int llayer,nox,noy,noz,nox1,noz1,noy1,lev,lselect,idim,elx,ely,elz;
  char gmt_string[10],char_dummy;
  double indbl,indbl2,age,f1,f2,vip;
  float rout[3],xloc[4];
  char tfilename[1000];

  const int dims=E->mesh.nsd;
  const int ends = enodes[dims];

  


  nox=E->mesh.nox;
  noy=E->mesh.noy;
  noz=E->mesh.noz;
  nox1=E->lmesh.nox;
  noz1=E->lmesh.noz;
  noy1=E->lmesh.noy;
  elx=E->lmesh.elx;
  elz=E->lmesh.elz;
  ely=E->lmesh.ely;

  elxlz = elx * elz;
  elxlylz = elxlz * ely;

  lev=E->mesh.levmax;
  
  /* 
     if we have not initialized the time history structure, do it now
  */
  if(!E->control.ggrd.time_hist.init){
    /* 
       init times, if available
    */
    ggrd_init_thist_from_file(&E->control.ggrd.time_hist,
			      E->control.ggrd.time_hist.file,TRUE,(E->parallel.me == 0));
    E->control.ggrd.time_hist.init = 1;
  }
  if(!E->control.ggrd.mat_control_init){

    /* assign the general depth dependent material group */
    construct_mat_group(E);

    if(E->parallel.me==0)
      fprintf(stderr,"ggrd_read_mat_from_file: initializing ggrd materials\n");

    if(is_global)		/* decide on GMT flag */
      sprintf(gmt_string,"-Lg"); /* global */
    else
      sprintf(gmt_string,"");
    /* 
       
    initialization steps
    
    */
    if(E->parallel.me > 0)	/* wait for previous processor */
      mpi_rc = MPI_Recv(&mpi_inmsg, 1, MPI_INT, (E->parallel.me-1), 
			0, MPI_COMM_WORLD, &mpi_stat);
    /* 
       read in the material file(s)
    */
    E->control.ggrd.mat = (struct  ggrd_gt *)calloc(E->control.ggrd.time_hist.nvtimes,sizeof(struct ggrd_gt));
    for(i=0;i < E->control.ggrd.time_hist.nvtimes;i++){
      if(E->control.ggrd.time_hist.nvtimes == 1)
	sprintf(tfilename,"%s",E->control.ggrd.mat_file);
      else
	sprintf(tfilename,"%s/%i/weak.grd",E->control.ggrd.mat_file,i+1);
      /* 2D file init */
      if(ggrd_grdtrack_init_general(FALSE,tfilename,&char_dummy,
				    gmt_string,(E->control.ggrd.mat+i),(E->parallel.me == 0),FALSE))
	myerror(E,"ggrd init error");
    }
    if(E->parallel.me <  E->parallel.nproc-1){ /* tell the next proc to go ahead */
      mpi_rc = MPI_Send(&mpi_success_message, 1, 
			MPI_INT, (E->parallel.me+1), 0, MPI_COMM_WORLD);
    }else{
      fprintf(stderr,"ggrd_read_mat_from_file: last processor done with ggrd mat init\n");
      fprintf(stderr,"ggrd_read_mat_from_file: WARNING: assuming a regular grid geometry\n");
    }
    
    /* end init */
  }
  if((E->control.ggrd.time_hist.nvtimes > 1)||(!E->control.ggrd.mat_control_init)){
    /* 
       loop through all elements and assign
    */
    for (m=1;m <= E->sphere.caps_per_proc;m++) {
      for (j=1;j <= elz;j++)  {	/* this assumes a regular grid sorted as in (1)!!! */
	if(E->mat[m][j] <= E->control.ggrd.mat_control ){
	  /* 
	     lithosphere or asthenosphere 
	  */
	  for (k=1;k <= ely;k++){
	    for (i=1;i <= elx;i++)   {
	      /* eq.(1) */
	      el = j + (i-1) * elz + (k-1)*elxlz;
	      /* 
		 find average horizontal coordinate 
		 
		 (DO WE HAVE THIS STORED ALREADY, E.G. FROM PRESSURE
		 EVAL FORM FUNCTION???)
	      */
	      xloc[1] = xloc[2] = xloc[3] = 0.0;
	      for(inode=1;inode <= 4;inode++){
		ind = E->ien[m][el].node[inode];
		xloc[1] += E->x[m][1][ind];xloc[2] += E->x[m][2][ind];xloc[3] += E->x[m][3][ind];
	      }
	      xloc[1]/=4.;xloc[2]/=4.;xloc[3]/=4.;
	      xyz2rtp(xloc[1],xloc[2],xloc[3],rout);
	      /* interpolate material */
	      if(E->control.ggrd.time_hist.nvtimes == 1){
		if(!ggrd_grdtrack_interpolate_tp((double)rout[1],(double)rout[2],E->control.ggrd.mat,
						 &vip,FALSE)){
		  fprintf(stderr,"ggrd_read_mat_from_ggrd_file: interpolation error at %g, %g\n",
			  xloc[1],xloc[2]);
		  parallel_process_termination();
		}
	      }else{		
		/* 
		   interpolate by time 
		*/
		age = find_age_in_MY(E);
		/*  */
		ggrd_interpol_time(age,&E->control.ggrd.time_hist,&i1,&i2,&f1,&f2,
				   E->control.ggrd.time_hist.vstage_transition);
		/*  */
		if(!ggrd_grdtrack_interpolate_tp((double)rout[1],(double)rout[2],(E->control.ggrd.mat+i1),&indbl,FALSE)){
		  fprintf(stderr,"ggrd_read_mat_from_ggrd_file: interpolation error at %g, %g\n",
			  xloc[1],xloc[2]);
		  parallel_process_termination();
		}
		if(!ggrd_grdtrack_interpolate_tp((double)rout[1],(double)rout[2],(E->control.ggrd.mat+i2),&indbl2,FALSE)){
		  fprintf(stderr,"ggrd_read_mat_from_ggrd_file: interpolation error at %g, %g\n",
			  xloc[1],xloc[2]);
		  parallel_process_termination();
		}
		/* average smoothly between the two tectonic stages */

		vip = exp((f1*log(indbl)+f2*log(indbl2)));
		//fprintf(stderr,"%g %i %i %g %g %g %g -> %g\n",age, i1,i2,f1,f2,indbl,indbl2,vip);
	      }
	      /* limit the input scaling? */
	      if(vip < 1e-5)
	      	vip = 1e-5;
	      if(vip > 1e5)
	      	vip = 1e5;
	      E->VIP[m][el] = vip;
	    }
	  }
	}else{
	  /* outside the lithosphere */
	  for (k=1;k <= ely;k++){
	    for (i=1;i <= elx;i++)   {
	      el = j + (i-1) * elz + (k-1)*elxlz;
	      /* no scaling else */
	      E->VIP[m][el] = 1.0;
	    }
	  }
	}
      }	/* end elz loop */
    } /* end m loop */
  } /* end assignment loop */

  E->control.ggrd.mat_control_init = 1;
}

#endif


