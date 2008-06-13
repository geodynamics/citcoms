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
#include "ggrd_handling.h"

void report(struct All_variables *,char *);
int layers_r(struct All_variables *,float );
void construct_mat_group(struct All_variables *);
void temperatures_conform_bcs(struct All_variables *);
int layers(struct All_variables *,int ,int );

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
    sprintf(gmt_bc,GGRD_GMT_GLOBAL_STRING);
  }else{			/* regional */
    sprintf(gmt_bc,"");
  }
    
  /* 
     initialize the ggrd control 
  */
  if(E->parallel.me > 0){	
    /* wait for previous processor */
    mpi_rc = MPI_Recv(&mpi_inmsg, 1, MPI_INT, (E->parallel.me-1), 
		      0, E->parallel.world, &mpi_stat);
  }
  if(ggrd_grdtrack_init_general(FALSE,E->trace.ggrd_file,
				char_dummy,gmt_bc,
				ggrd_ict,FALSE,FALSE)){
    myerror(E,"ggrd tracer init error");
  }
  if(E->parallel.me <  E->parallel.nproc-1){
    /* tell the next proc to go ahead */
    mpi_rc = MPI_Send(&mpi_success_message, 1,
		      MPI_INT, (E->parallel.me+1), 0, E->parallel.world);
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
  ggrd_temp_init_general(E,1);
}
void ggrd_reg_temp_init(struct All_variables *E)
{
  ggrd_temp_init_general(E,0);
}



/*

initialize temperatures from grd files for spherical geometry

*/

void ggrd_temp_init_general(struct All_variables *E,int is_global)
{

  MPI_Status mpi_stat;
  int mpi_rc;
  int mpi_inmsg, mpi_success_message = 1;
  double temp1,tbot,tgrad,tmean,tadd,rho_prem;
  char gmt_string[10];
  int i,j,k,m,node,noxnoz,nox,noy,noz;
  if(is_global)		/* decide on GMT flag */
    sprintf(gmt_string,GGRD_GMT_GLOBAL_STRING); /* global */
  else
    sprintf(gmt_string,"");

  noy=E->lmesh.noy;
  nox=E->lmesh.nox;
  noz=E->lmesh.noz;
  noxnoz = nox * noz;

  if(E->parallel.me == 0)
    fprintf(stderr,"ggrd_temp_init_general: using GMT grd files for temperatures, gmtflag: %s\n",gmt_string);
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
		      0, E->parallel.world, &mpi_stat);
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
				E->control.ggrd.temp_init.dfile,gmt_string,
				E->control.ggrd.temp_init.d,(E->parallel.me == 0),
				FALSE))
    myerror(E,"grd init error");
  if(E->parallel.me <  E->parallel.nproc-1){
    /* tell the next processor to go ahead with the init step	*/
    mpi_rc = MPI_Send(&mpi_success_message, 1, MPI_INT, (E->parallel.me+1), 0, E->parallel.world);
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
  int mpi_rc,timedep,interpolate;
  int mpi_inmsg, mpi_success_message = 1;
  int m,el,i,j,k,inode,i1,i2,elxlz,elxlylz,ind;
  int llayer,nox,noy,noz,nox1,noz1,noy1,lev,lselect,idim,elx,ely,elz;
  char gmt_string[10],char_dummy;
  double indbl,indbl2,age,f1,f2,vip,rout[3],xloc[4];
  char tfilename[1000];

  const int dims=E->mesh.nsd;
  const int ends = enodes[dims];

  nox=E->mesh.nox;noy=E->mesh.noy;noz=E->mesh.noz;
  nox1=E->lmesh.nox;noz1=E->lmesh.noz;noy1=E->lmesh.noy;
  elx=E->lmesh.elx;elz=E->lmesh.elz;ely=E->lmesh.ely;
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
  /* time dependent? */
  timedep = (E->control.ggrd.time_hist.nvtimes > 1)?(1):(0);
  if(!E->control.ggrd.mat_control_init){
    /* assign the general depth dependent material group */
    construct_mat_group(E);
    if(E->parallel.me==0)
      fprintf(stderr,"ggrd_read_mat_from_file: initializing ggrd materials\n");
    if(is_global)		/* decide on GMT flag */
      sprintf(gmt_string,GGRD_GMT_GLOBAL_STRING); /* global */
    else
      sprintf(gmt_string,"");
    /*

    initialization steps

    */
    if(E->parallel.me > 0)	/* wait for previous processor */
      mpi_rc = MPI_Recv(&mpi_inmsg, 1, MPI_INT, (E->parallel.me-1),
			0, E->parallel.world, &mpi_stat);
    /*
       read in the material file(s)
    */
    E->control.ggrd.mat = (struct  ggrd_gt *)calloc(E->control.ggrd.time_hist.nvtimes,sizeof(struct ggrd_gt));
    for(i=0;i < E->control.ggrd.time_hist.nvtimes;i++){
      if(!timedep)		/* constant */
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
			MPI_INT, (E->parallel.me+1), 0, E->parallel.world);
    }else{
      fprintf(stderr,"ggrd_read_mat_from_file: last processor done with ggrd mat init\n");
      fprintf(stderr,"ggrd_read_mat_from_file: WARNING: assuming a regular grid geometry\n");
    }

    /* end init */
  }
  if(timedep || (!E->control.ggrd.mat_control_init)){
    age = find_age_in_MY(E);
    if(E->parallel.me == 0)
      fprintf(stderr,"ggrd_read_mat_from_ggrd_file: assigning at age %g\n",age);
    if(timedep){
      ggrd_interpol_time(age,&E->control.ggrd.time_hist,&i1,&i2,&f1,&f2,
			 E->control.ggrd.time_hist.vstage_transition);
      interpolate = 1;
    }else{
      interpolate = 0;
      i1 = 0;
    }
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
	      xyz2rtpd(xloc[1],xloc[2],xloc[3],rout);
	      /* material */
	      if(!ggrd_grdtrack_interpolate_tp(rout[1],rout[2],(E->control.ggrd.mat+i1),&indbl,FALSE)){
		  fprintf(stderr,"ggrd_read_mat_from_ggrd_file: interpolation error at %g, %g\n",
			  rout[1],rout[2]);
		  parallel_process_termination();
	      }
	      if(interpolate){
		if(!ggrd_grdtrack_interpolate_tp((double)rout[1],(double)rout[2],
						 (E->control.ggrd.mat+i2),&indbl2,FALSE)){
		  fprintf(stderr,"ggrd_read_mat_from_ggrd_file: interpolation error at %g, %g\n",
			  rout[1],rout[2]);
		  parallel_process_termination();
		}
		/* average smoothly between the two tectonic stages */
		vip = exp((f1*log(indbl)+f2*log(indbl2)));
	      }else{
		vip = indbl;
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
  if((!timedep) && (!E->control.ggrd.mat_control_init)){			/* forget the grid */
    ggrd_grdtrack_free_gstruc(E->control.ggrd.mat);
  }
  E->control.ggrd.mat_control_init = 1;
} /* end mat control */


/*


read in Rayleigh number prefactor from file, this will get assigned if

layer <= E->control.ggrd.ray_control


*/
void ggrd_read_ray_from_file(struct All_variables *E, int is_global)
{
  MPI_Status mpi_stat;
  int mpi_rc,timedep,interpolate;
  int mpi_inmsg, mpi_success_message = 1;
  int m,el,i,j,k,node,i1,i2,elxlz,elxlylz,ind;
  int llayer,nox,noy,noz,nox1,noz1,noy1,lev,lselect,idim,elx,ely,elz;
  char gmt_string[10],char_dummy;
  double indbl,indbl2,age,f1,f2,vip,rout[3],xloc[4];
  char tfilename[1000];
  const int dims=E->mesh.nsd;
  const int ends = enodes[dims];
  /* dimensional ints */
  nox=E->mesh.nox;noy=E->mesh.noy;noz=E->mesh.noz;
  nox1=E->lmesh.nox;noz1=E->lmesh.noz;noy1=E->lmesh.noy;
  elx=E->lmesh.elx;elz=E->lmesh.elz;ely=E->lmesh.ely;
  elxlz = elx * elz;
  elxlylz = elxlz * ely;
  lev=E->mesh.levmax;
  /*
     if we have not initialized the time history structure, do it now
     any function can do that

     we could only use the surface processors, but maybe the rayleigh
     number is supposed to be changed at large depths
  */
  if(!E->control.ggrd.time_hist.init){
    ggrd_init_thist_from_file(&E->control.ggrd.time_hist,
			      E->control.ggrd.time_hist.file,TRUE,(E->parallel.me == 0));
    E->control.ggrd.time_hist.init = 1;
  }
  timedep = (E->control.ggrd.time_hist.nvtimes > 1)?(1):(0);
  if(!E->control.ggrd.ray_control_init){
    /* init step */
    if(E->parallel.me==0)
      fprintf(stderr,"ggrd_read_ray_from_file: initializing from %s\n",E->control.ggrd.ray_file);
    if(is_global)		/* decide on GMT flag */
      sprintf(gmt_string,GGRD_GMT_GLOBAL_STRING); /* global */
    else
      sprintf(gmt_string,"");
    if(E->parallel.me > 0)	/* wait for previous processor */
      mpi_rc = MPI_Recv(&mpi_inmsg, 1, MPI_INT, (E->parallel.me-1),
			0, E->parallel.world, &mpi_stat);
    E->control.ggrd.ray = (struct  ggrd_gt *)calloc(E->control.ggrd.time_hist.nvtimes,sizeof(struct ggrd_gt));
    for(i=0;i < E->control.ggrd.time_hist.nvtimes;i++){
      if(!timedep)		/* constant */
	sprintf(tfilename,"%s",E->control.ggrd.ray_file);
      else
	sprintf(tfilename,"%s/%i/rayleigh.grd",E->control.ggrd.ray_file,i+1);
      if(ggrd_grdtrack_init_general(FALSE,tfilename,&char_dummy,
				    gmt_string,(E->control.ggrd.ray+i),(E->parallel.me == 0),FALSE))
	myerror(E,"ggrd init error");
    }
    if(E->parallel.me <  E->parallel.nproc-1){ /* tell the next proc to go ahead */
      mpi_rc = MPI_Send(&mpi_success_message, 1,
			MPI_INT, (E->parallel.me+1), 0, E->parallel.world);
    }else{
      fprintf(stderr,"ggrd_read_ray_from_file: last processor done with ggrd ray init\n");
    }
    E->control.surface_rayleigh = (float *)malloc(sizeof(float)*(E->lmesh.nsf+2));
    if(!E->control.surface_rayleigh)
      myerror(E,"ggrd rayleigh mem error");
  }
  if(timedep || (!E->control.ggrd.ray_control_init)){
    if(timedep){
      age = find_age_in_MY(E);
      ggrd_interpol_time(age,&E->control.ggrd.time_hist,&i1,&i2,&f1,&f2,
			 E->control.ggrd.time_hist.vstage_transition);
      interpolate = 1;
    }else{
      interpolate = 0;i1 = 0;
    }
    if(E->parallel.me == 0)
      fprintf(stderr,"ggrd_read_ray_from_ggrd_file: assigning at time %g\n",age);
    for (m=1;m <= E->sphere.caps_per_proc;m++) {
      /* loop through all surface nodes */
      for (j=1;j <= E->lmesh.nsf;j++)  {
	node = j * E->lmesh.noz ;
	rout[1] = (double)E->sx[m][1][node];
	rout[2] = (double)E->sx[m][2][node];
	if(!ggrd_grdtrack_interpolate_tp(rout[1],rout[2],(E->control.ggrd.ray+i1),&indbl,FALSE)){
	  fprintf(stderr,"ggrd_read_ray_from_ggrd_file: interpolation error at %g, %g\n",
		  rout[1],rout[2]);
	  parallel_process_termination();
	}
	//fprintf(stderr,"%i %i %g %g %g\n",j,E->lmesh.nsf,rout[1],rout[2],indbl);
	if(interpolate){
	  if(!ggrd_grdtrack_interpolate_tp(rout[1],rout[2],
					   (E->control.ggrd.ray+i2),&indbl2,FALSE)){
	    fprintf(stderr,"ggrd_read_ray_from_ggrd_file: interpolation error at %g, %g\n",
		    rout[1],rout[2]);
	    parallel_process_termination();
	  }
	  /* average smoothly between the two tectonic stages */
	  vip = f1*indbl+f2*indbl2;
	}else{
	  vip = indbl;
	}
	E->control.surface_rayleigh[j] = vip;
      }	/* end node loop */
    } /* end cap loop */
  } /* end assign loop */
  if((!timedep) && (!E->control.ggrd.ray_control_init)){			/* forget the grid */
    ggrd_grdtrack_free_gstruc(E->control.ggrd.ray);
  }
  E->control.ggrd.ray_control_init = 1;
} /* end ray control */




void ggrd_read_vtop_from_file(struct All_variables *E, int is_global)
{
  MPI_Status mpi_stat;
  int mpi_rc,interpolate,timedep;
  int mpi_inmsg, mpi_success_message = 1;
  int m,el,i,k,i1,i2,ind,nodel;
  int nox1,noz1,noy1,lev,lselect,idim,nox1noz1;
  char gmt_string[10],char_dummy;

  double vin1[2],vin2[2],age,f1,f2,vp,vt,vscale,rout[3];
  char tfilename1[1000],tfilename2[1000];

  const int dims=E->mesh.nsd;

  nox1 = E->lmesh.nox;noz1=E->lmesh.noz;noy1=E->lmesh.noy;
  nox1noz1 = nox1*noz1;
  lev = E->mesh.levmax;

  /* velocity scaling, assuming input is cm/yr  */
  vscale = E->data.scalev * E->data.timedir;

  /*
     if we have not initialized the time history structure, do it now
     if this file is not found, will use constant velocities
  */
  if(!E->control.ggrd.time_hist.init){/* init times, if available*/
    ggrd_init_thist_from_file(&E->control.ggrd.time_hist,E->control.ggrd.time_hist.file,
			      TRUE,(E->parallel.me == 0));
    E->control.ggrd.time_hist.init = 1;
  }

  timedep = (E->control.ggrd.time_hist.nvtimes > 1)?(1):(0);

  if(!E->control.ggrd.vtop_control_init){
    if(E->parallel.me==0)
      fprintf(stderr,"ggrd_read_vtop_from_file: initializing ggrd velocities\n");
    if(is_global){		/* decide on GMT flag */
      //sprintf(gmt_string,GGRD_GMT_XPERIODIC_STRING); /* periodic */
      sprintf(gmt_string,GGRD_GMT_GLOBAL_STRING); /* global */
    }else
      sprintf(gmt_string,"");
    /*

    initialization steps

    */
    if(E->parallel.me > 0)	/* wait for previous processor */
      mpi_rc = MPI_Recv(&mpi_inmsg, 1, MPI_INT, (E->parallel.me-1),
			0, E->parallel.world, &mpi_stat);
    /*
       read in the velocity file(s)
    */
    E->control.ggrd.svt = (struct  ggrd_gt *)calloc(E->control.ggrd.time_hist.nvtimes,sizeof(struct ggrd_gt));
    E->control.ggrd.svp = (struct  ggrd_gt *)calloc(E->control.ggrd.time_hist.nvtimes,sizeof(struct ggrd_gt));
    for(i=0;i < E->control.ggrd.time_hist.nvtimes;i++){
      if(!timedep){ /* constant */
	sprintf(tfilename1,"%s/vt.grd",E->control.ggrd.vtop_dir);
	sprintf(tfilename2,"%s/vp.grd",E->control.ggrd.vtop_dir);
      } else {			/* f(t) */
	sprintf(tfilename1,"%s/%i/vt.grd",E->control.ggrd.vtop_dir,i+1);
	sprintf(tfilename2,"%s/%i/vp.grd",E->control.ggrd.vtop_dir,i+1);
      }
      if(ggrd_grdtrack_init_general(FALSE,tfilename1,&char_dummy,
				    gmt_string,(E->control.ggrd.svt+i),(E->parallel.me == 0),FALSE))
	myerror(E,"ggrd init error vt");
      if(ggrd_grdtrack_init_general(FALSE,tfilename2,&char_dummy,
				    gmt_string,(E->control.ggrd.svp+i),(E->parallel.me == 0),FALSE))
	myerror(E,"ggrd init error vp");
    }
    if(E->parallel.me <  E->parallel.nproc-1){ /* tell the next proc to go ahead */
      mpi_rc = MPI_Send(&mpi_success_message, 1,
			MPI_INT, (E->parallel.me+1), 0, E->parallel.world);
    }else{
      fprintf(stderr,"ggrd_read_vtop_from_file: last processor done with ggrd vtop BC init, %i timesteps\n",
	      E->control.ggrd.time_hist.nvtimes);
    }
    /* end init */
  }
  if((E->control.ggrd.time_hist.nvtimes > 1)||
     (!E->control.ggrd.vtop_control_init)){
    age = find_age_in_MY(E);
    if(timedep){
      /* 
	 interpolate by time 
      */
      if(age < 0){		/* Opposite of other method */
	interpolate = 0;
	/* present day should be last file*/
	i1 = E->control.ggrd.time_hist.nvtimes - 1;
	if(E->parallel.me == 0)
	  fprintf(stderr,"ggrd_read_vtop_from_file: using present day vtop for age = %g\n",age);
      }else{
	/*  */
	ggrd_interpol_time(age,&E->control.ggrd.time_hist,&i1,&i2,&f1,&f2,
			   E->control.ggrd.time_hist.vstage_transition);
	interpolate = 1;
	if(E->parallel.me == 0)
	  fprintf(stderr,"ggrd_read_vtop_from_file: interpolating vtop for age = %g\n",age);
      }
    }else{
      interpolate = 0;		/* single timestep, use single file */
      i1 = 0;
      if(E->parallel.me == 0)
	fprintf(stderr,"ggrd_read_vtop_from_file: constant velocity BC \n");
    }
    if(E->parallel.me==0)
      fprintf(stderr,"ggrd_read_vtop_from_file: assigning velocities BC, timedep: %i time: %g\n",
	      timedep,age);
    /* 
       loop through all elements and assign
    */
    for (m=1;m <= E->sphere.caps_per_proc;m++) {
      if(E->parallel.me_loc[3] == E->parallel.nprocz-1 )  { /* if on top */
	for(k=1;k <= noy1;k++)	{/* loop through surface nodes */
	  for(i=1;i <= nox1;i++)    {
	    nodel = (k-1)*nox1noz1 + (i-1)*noz1+noz1;

	    rout[1] = E->sx[m][1][nodel]; /* theta,phi coordinates */
	    rout[2] = E->sx[m][2][nodel];

	    if(!ggrd_grdtrack_interpolate_tp(rout[1],rout[2],(E->control.ggrd.svt+i1),
					     vin1,FALSE)){
	      fprintf(stderr,"ggrd_read_vtop_from_ggrd_file: interpolation error at %g, %g\n",
		      rout[1],rout[2]);
	      parallel_process_termination();
	    }
	    if(!ggrd_grdtrack_interpolate_tp(rout[1],rout[2],(E->control.ggrd.svp+i1),
					     (vin1+1),FALSE)){
	      fprintf(stderr,"ggrd_read_vtop_from_ggrd_file: interpolation error at %g, %g\n",
		      rout[1],rout[2]);
	      parallel_process_termination();
	    }
	    if(interpolate){	/* second time */
	      if(!ggrd_grdtrack_interpolate_tp(rout[1],rout[2],(E->control.ggrd.svt+i2),vin2,FALSE)){
		fprintf(stderr,"ggrd_read_mat_from_ggrd_file: interpolation error at %g, %g\n",
			rout[1],rout[2]);
		parallel_process_termination();
	      }
	      if(!ggrd_grdtrack_interpolate_tp(rout[1],rout[2],(E->control.ggrd.svp+i2),(vin2+1),FALSE)){
		fprintf(stderr,"ggrd_read_mat_from_ggrd_file: interpolation error at %g, %g\n",
			rout[1],rout[2]);
		parallel_process_termination();
	      }
	      vt = (f1*vin1[0] + f2*vin2[0])*vscale;
	      vp = (f1*vin1[1] + f2*vin2[1])*vscale;
	    }else{
	      vt = vin1[0]*vscale;
	      vp = vin1[1]*vscale;
	    }
	    /* assign as boundary condition */
	    E->sphere.cap[m].VB[1][nodel] = vt;	/* theta */
	    E->sphere.cap[m].VB[2][nodel] = vp;	/* phi */
	    E->sphere.cap[m].VB[3][nodel] = 0.0; /* r */
	  }
	}	/* end surface node loop */
      }	/* end top proc branch */
    } /* end cap loop */
  } /* end assignment branch */
  if((!timedep)&&(!E->control.ggrd.vtop_control_init)){			/* forget the grids */
    ggrd_grdtrack_free_gstruc(E->control.ggrd.svt);
    ggrd_grdtrack_free_gstruc(E->control.ggrd.svp);
  }
  E->control.ggrd.vtop_control_init = 1;
}



void ggrd_read_age_from_file(struct All_variables *E, int is_global)
{
  myerror(E,"not implemented yet");
} /* end age control */

/* adjust Ra in top boundary layer  */
void ggrd_adjust_tbl_rayleigh(struct All_variables *E,
			      double **buoy)
{
  int m,snode,node,i;
  double xloc,fac,bnew;
  if(!E->control.ggrd.ray_control_init)
    myerror(E,"ggrd rayleigh not initialized, but in adjust tbl");
  if(E->parallel.me == 0)
    fprintf(stderr,"ggrd__adjust_tbl_rayleigh: adjusting Rayleigh in top %i layers\n",
	    E->control.ggrd.ray_control);

  /* 
     need to scale buoy with the material determined rayleigh numbers
  */
  for(m=1;m <= E->sphere.caps_per_proc;m++){
    for(snode=1;snode <= E->lmesh.nsf;snode++){ /* loop through surface nodes */
      if(fabs(E->control.surface_rayleigh[snode]-1.0)>1e-6){
	for(i=1;i <= E->lmesh.noz;i++){ /* go through depth layers */
	  node = (snode-1)*E->lmesh.noz + i; /* global node number */
	  if(layers(E,m,node) <= E->control.ggrd.ray_control){ 
	    /* 
	       node is in top layers 
	    */
	    /* depth factor, cos^2 tapered */
	    xloc=1.0 + ((1 - E->sx[m][3][node]) - 
			E->viscosity.zbase_layer[E->control.ggrd.ray_control])/
	      E->viscosity.zbase_layer[E->control.ggrd.ray_control];
	    fac = cos(xloc*1.5707963267);fac *= fac; /* cos^2
							tapering,
							factor
							decrease from
							1 at surface
							to zero at
							boundary */
	    bnew = buoy[m][node] * E->control.surface_rayleigh[snode]; /* modified rayleigh */
	    /* debugging */
	    //fprintf(stderr,"z: %11g tl: %i zm: %11g fac: %11g sra: %11g bnew: %11g bold: %11g\n",
	    //	    (1 - E->sx[m][3][node])*6371,E->control.ggrd.ray_control,
	    //	    E->viscosity.zbase_layer[E->control.ggrd.ray_control]*6371,
	    //	    fac,E->control.surface_rayleigh[snode],(fac * bnew + (1-fac)*buoy[m][node]),buoy[m][node]);
	    buoy[m][node] = fac * bnew + (1-fac)*buoy[m][node];
	  }
	}
      }
    }
  }

}



#endif


