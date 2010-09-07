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
#ifdef USE_GZDIR
#include <zlib.h>
gzFile *gzdir_output_open(char *,char *);

#endif

#include <math.h>
#include "global_defs.h"
#include "parsing.h"
#include "parallel_related.h"
#include "composition_related.h"
#include "element_definitions.h"

#ifdef CITCOM_ALLOW_ORTHOTROPIC_VISC
#include "anisotropic_viscosity.h"
#endif
#ifdef USE_GGRD

#include "hc.h"			/* ggrd and hc packages */
#include "ggrd_handling.h"

void report(struct All_variables *,char *);
int layers_r(struct All_variables *,float );
void construct_mat_group(struct All_variables *);
void temperatures_conform_bcs(struct All_variables *);
int layers(struct All_variables *,int ,int );
void ggrd_vtop_helper_decide_on_internal_nodes(struct All_variables *,	/* input */
					       int ,int ,int, int,int ,int *, int *,int *);
void convert_pvec_to_cvec_d(double ,double , double , double *,double *);
void calc_cbase_at_tp_d(double , double , double *);
void xyz2rtpd(float ,float ,float ,double *);

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
  static ggrd_boolean shift_to_pos_lon = FALSE;	/* this should not be needed anymore */
  report(E,"ggrd_init_tracer_flavors: ggrd mat init");
  int only_one_layer,this_layer;
  /* 
     are we global?
  */
  if (E->parallel.nprocxy == 12){
    /* use GMT's geographic boundary conditions */
    sprintf(gmt_bc,GGRD_GMT_GLOBAL_STRING);
  }else{			/* regional */
    sprintf(gmt_bc,"");
  }
  only_one_layer = ((E->trace.ggrd_layers > 0)?(0):(1));

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
  /* shold we decide on shifting to positive longitudes, ie. 0...360? */
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

      this_layer = layers_r(E,rad);
      if((only_one_layer && (this_layer == -E->trace.ggrd_layers)) ||
	 ((!only_one_layer)&&(this_layer <= E->trace.ggrd_layers))){
	/*
	   in top layers
	*/
	phi =   E->trace.basicq[j][1][kk];
	theta = E->trace.basicq[j][0][kk];
	/* interpolate from grid */
	if(!ggrd_grdtrack_interpolate_tp((double)theta,(double)phi,
					 ggrd_ict,&indbl,FALSE,shift_to_pos_lon)){
	  snprintf(error,255,"ggrd_init_tracer_flavors: interpolation error at lon: %g lat: %g",
		   phi*180/M_PI, 90-theta*180/M_PI);
	  myerror(E,error);
	}
	if(!E->control.ggrd_comp_smooth){
	  /* limit to 0 or 1 */
	  if(indbl < .5)
	    indbl = 0.0;
	  else
	    indbl = 1.0;
	}
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
  double temp1,tbot,tgrad,tmean,tadd,rho_prem,depth,loc_scale;
  char gmt_string[10];
  int i,j,k,m,node,noxnoz,nox,noy,noz;
  static ggrd_boolean shift_to_pos_lon = FALSE;
  
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

  if(E->control.ggrd.temp.scale_with_prem){/* initialize PREM */
    if(prem_read_model(E->control.ggrd.temp.prem.model_filename,
		       &E->control.ggrd.temp.prem, (E->parallel.me == 0)))
      myerror(E,"PREM init error");
  }
  /*
     initialize the GMT grid files
  */
  E->control.ggrd.temp.d[0].init = FALSE;
  if(ggrd_grdtrack_init_general(TRUE,E->control.ggrd.temp.gfile,
				E->control.ggrd.temp.dfile,gmt_string,
				E->control.ggrd.temp.d,(E->parallel.me == 0),
				FALSE))
    myerror(E,"grd init error");
  /*  */
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


  for(m=1;m <= E->sphere.caps_per_proc;m++)
    for(i=1;i <= noy;i++)
      for(j=1;j <= nox;j++)
	for(k=1;k <= noz;k++)  {
	  /* node numbers */
	  node=k+(j-1)*noz+(i-1)*noxnoz;

	  /*
	     get interpolated velocity anomaly
	  */
	  depth = (1-E->sx[m][3][node])*6371;
	  if(!ggrd_grdtrack_interpolate_rtp((double)E->sx[m][3][node],
					    (double)E->sx[m][1][node],
					    (double)E->sx[m][2][node],
					    E->control.ggrd.temp.d,&tadd,
					    FALSE,shift_to_pos_lon)){
	    fprintf(stderr,"%g %g %g\n",E->sx[m][2][node]*57.29577951308232087,
		    90-E->sx[m][1][node]*57.29577951308232087,depth);
		    
	    myerror(E,"ggrd__temp_init_general: interpolation error");

	  }
	  
	  if(depth < E->control.ggrd_lower_depth_km){
	    /*
	      mean temp is (top+bot)/2 + offset
	    */
	    tmean = (tbot + E->control.TBCtopval)/2.0 +  E->control.ggrd.temp.offset;
	    loc_scale =  E->control.ggrd.temp.scale;
	  }else{		/* lower mantle */
	    tmean = (tbot + E->control.TBCtopval)/2.0 +  E->control.ggrd_lower_offset;
	    loc_scale = E->control.ggrd_lower_scale;
	  }
	  if(E->control.ggrd.temp.scale_with_prem){
	    /*
	       get the PREM density at r for additional scaling
	    */
	    prem_get_rho(&rho_prem,(double)E->sx[m][3][node],&E->control.ggrd.temp.prem);
	    if(rho_prem < GGRD_DENS_MIN){
	      fprintf(stderr,"WARNING: restricting minimum density to %g, would have been %g\n",
		      GGRD_DENS_MIN,rho_prem);
	      rho_prem = GGRD_DENS_MIN; /* we don't want the density of water or crust*/
	    }
	    /*
	       assign temperature
	    */
	    E->T[m][node] = tmean + tadd * loc_scale * rho_prem / E->data.density;
	  }else{
	    /* no PREM scaling */
	    E->T[m][node] = tmean + tadd * loc_scale;
	  }

	  if(E->control.ggrd.temp.limit_trange){
	    /* limit to 0 < T < 1 ?*/
	    E->T[m][node] = min(max(E->T[m][node], 0.0),1.0);
	  }
	  //fprintf(stderr,"z: %11g T: %11g\n",E->sx[m][3][node],E->T[m][node]);
	  if(E->control.ggrd.temp.override_tbc){
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
  ggrd_grdtrack_free_gstruc(E->control.ggrd.temp.d);
  /*
     end temperature/density from GMT grd init
  */
  temperatures_conform_bcs(E);
}

/*


read in material, i.e. viscosity prefactor from ggrd file, this will
get assigned for all nodes if their 

layer <=  E->control.ggrd.mat_control for  E->control.ggrd.mat_control > 0

or 

layer ==  -E->control.ggrd.mat_control for  E->control.ggrd.mat_control < 0


the grd model can be 2D (a layer in itself), or 3D (a model with
several layers)

*/
void ggrd_read_mat_from_file(struct All_variables *E, int is_global)
{
  MPI_Status mpi_stat;
  int mpi_rc,timedep,interpolate;
  int mpi_inmsg, mpi_success_message = 1;
  int m,el,i,j,k,inode,i1,i2,elxlz,elxlylz,ind;
  int llayer,nox,noy,noz,level,lselect,idim,elx,ely,elz;
  char gmt_string[10],char_dummy;
  double indbl,indbl2,age,f1,f2,vip,rout[3],xloc[4];
  char tfilename[1000];
  static ggrd_boolean shift_to_pos_lon = FALSE;
  const int dims=E->mesh.nsd;
  const int ends = enodes[dims];
  FILE *in;

  nox=E->mesh.nox;noy=E->mesh.noy;noz=E->mesh.noz;
  elx=E->lmesh.elx;elz=E->lmesh.elz;ely=E->lmesh.ely;
  elxlz = elx * elz;
  elxlylz = elxlz * ely;

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
    /* 
       is this 3D?
    */
    if((in = fopen(E->control.ggrd_mat_depth_file,"r"))!=NULL) /* expect 3D setup */
      E->control.ggrd_mat_is_3d = TRUE;
    else
      E->control.ggrd_mat_is_3d = FALSE;

    if(E->parallel.me==0)
      if(E->control.ggrd.mat_control > 0)
	fprintf(stderr,"ggrd_read_mat_from_file: initializing, assigning to all above %g km, input is %s, %s\n",
		E->data.radius_km*E->viscosity.zbase_layer[E->control.ggrd.mat_control-1],
		(is_global)?("global"):("regional"),(E->control.ggrd_mat_is_3d)?("3D"):("single layer"));
      else
	fprintf(stderr,"ggrd_read_mat_from_file: initializing, assigning to single layer at %g km, input is %s, %s\n",
		E->data.radius_km*E->viscosity.zbase_layer[-E->control.ggrd.mat_control-1],
		(is_global)?("global"):("regional"),(E->control.ggrd_mat_is_3d)?("3D"):("single layer"));

    for(i=0;i < E->control.ggrd.time_hist.nvtimes;i++){
      if(!timedep)		/* constant */
	sprintf(tfilename,"%s",E->control.ggrd.mat_file);
      else{
	if(E->control.ggrd_mat_is_3d)
	  sprintf(tfilename,"%s/%i/weak",E->control.ggrd.mat_file,i+1);
	else
	  sprintf(tfilename,"%s/%i/weak.grd",E->control.ggrd.mat_file,i+1);
      }
      if(ggrd_grdtrack_init_general(E->control.ggrd_mat_is_3d,tfilename,E->control.ggrd_mat_depth_file,
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
      fprintf(stderr,"ggrd_read_mat_from_file: assigning at age %g\n",age);
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
	if(((E->control.ggrd.mat_control > 0) && (E->mat[m][j] <=  E->control.ggrd.mat_control )) || 
	   ((E->control.ggrd.mat_control < 0) && (E->mat[m][j] == -E->control.ggrd.mat_control ))){
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
	      /* 
		 material 
	      */
	      if(E->control.ggrd_mat_is_3d){
		if(!ggrd_grdtrack_interpolate_rtp((double)rout[0],(double)rout[1],(double)rout[2],(E->control.ggrd.mat+i1),&indbl,
						 FALSE,shift_to_pos_lon)){
		  fprintf(stderr,"ggrd_read_mat_from_file: interpolation error at lon: %g lat: %g depth: %g\n",
			  rout[2]*180/M_PI,90-rout[1]*180/M_PI,(1.0-rout[0]) * 6371.0);
		  parallel_process_termination();
		}

	      }else{
		if(!ggrd_grdtrack_interpolate_tp((double)rout[1],(double)rout[2],(E->control.ggrd.mat+i1),&indbl,
						 FALSE,shift_to_pos_lon)){
		  fprintf(stderr,"ggrd_read_mat_from_file: interpolation error at lon: %g lat: %g\n",
			  rout[2]*180/M_PI,90-rout[1]*180/M_PI);
		  parallel_process_termination();
		}
	      }

	      if(interpolate){
		if(E->control.ggrd_mat_is_3d){
		  if(!ggrd_grdtrack_interpolate_rtp((double)rout[0],(double)rout[1],(double)rout[2],
						   (E->control.ggrd.mat+i2),&indbl2,
						   FALSE,shift_to_pos_lon)){
		    fprintf(stderr,"ggrd_read_mat_from_file: interpolation error at lon: %g lat: %g depth: %g\n",
			    rout[2]*180/M_PI,90-rout[1]*180/M_PI,(1.0-rout[0]) * 6371.0);
		    parallel_process_termination();
		  }
		}else{
		  if(!ggrd_grdtrack_interpolate_tp((double)rout[1],(double)rout[2],
						   (E->control.ggrd.mat+i2),&indbl2,
						   FALSE,shift_to_pos_lon)){
		    fprintf(stderr,"ggrd_read_mat_from_file: interpolation error at lon: %g lat: %g\n",
			    rout[2]*180/M_PI,90-rout[1]*180/M_PI);
		    parallel_process_termination();
		  }
		}
		/* average smoothly between the two tectonic stages */
		vip = exp((f1*log(indbl)+f2*log(indbl2)));
	      }else{
		vip = indbl;
	      }
	      if(E->control.ggrd_mat_limit_prefactor){
		/* limit the input scaling? */
		if(vip < 1e-5)
		  vip = 1e-5;
		if(vip > 1e5)
		  vip = 1e5;
	      }
	      //fprintf(stderr,"lon %11g lat %11g depth %11g vip %11g\n",rout[2]*180/M_PI,90-rout[1]*180/M_PI,(1.0-rout[0]) * 6371.0,vip);
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


I.e. this function can be used to assign a laterally varying prefactor
to the rayleigh number in the surface layers, e.g. to have a simple
way to represent stationary, chemical heterogeneity

*/
void ggrd_read_ray_from_file(struct All_variables *E, int is_global)
{
  MPI_Status mpi_stat;
  int mpi_rc,timedep,interpolate;
  int mpi_inmsg, mpi_success_message = 1;
  int m,el,i,j,k,node,i1,i2,elxlz,elxlylz,ind;
  int llayer,nox,noy,noz,lev,lselect,idim,elx,ely,elz;
  char gmt_string[10],char_dummy;
  double indbl,indbl2,age,f1,f2,vip,rout[3],xloc[4];
  char tfilename[1000];
  static ggrd_boolean shift_to_pos_lon = FALSE;

  const int dims=E->mesh.nsd;
  const int ends = enodes[dims];
  /* dimensional ints */
  nox=E->mesh.nox;noy=E->mesh.noy;noz=E->mesh.noz;
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
      fprintf(stderr,"ggrd_read_ray_from_file: assigning at time %g\n",age);
    for (m=1;m <= E->sphere.caps_per_proc;m++) {
      /* loop through all surface nodes */
      for (j=1;j <= E->lmesh.nsf;j++)  {
	node = j * E->lmesh.noz ;
	rout[1] = (double)E->sx[m][1][node];
	rout[2] = (double)E->sx[m][2][node];
	if(!ggrd_grdtrack_interpolate_tp(rout[1],rout[2],(E->control.ggrd.ray+i1),&indbl,
					 FALSE,shift_to_pos_lon)){
	  fprintf(stderr,"ggrd_read_ray_from_file: interpolation error at %g, %g\n",
		  rout[1],rout[2]);
	  parallel_process_termination();
	}
	//fprintf(stderr,"%i %i %g %g %g\n",j,E->lmesh.nsf,rout[1],rout[2],indbl);
	if(interpolate){
	  if(!ggrd_grdtrack_interpolate_tp(rout[1],rout[2],
					   (E->control.ggrd.ray+i2),&indbl2,
					   FALSE,shift_to_pos_lon)){
	    fprintf(stderr,"ggrd_read_ray_from_file: interpolation error at %g, %g\n",
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


/*  

read surface boundary conditions from netcdf grd files

if topvbc=1, will apply to velocities
if topvbc=0, will apply to tractions

*/

void ggrd_read_vtop_from_file(struct All_variables *E, int is_global)
{
  MPI_Status mpi_stat;
  int mpi_rc,interpolate,timedep,use_codes,code,assign,ontop;
  int mpi_inmsg, mpi_success_message = 1;
  int m,el,i,k,i1,i2,ind,nodel,j,level, verbose;
  int nox,noz,noy,noxl,noyl,nozl,lselect,idim,noxnoz,
    noxlnozl,save_codes,topnode,botnode;
  char gmt_string[10],char_dummy;
  static int lc =0;			/* only for debugging */
  double vin1[2],vin2[2],age,f1,f2,vscale,rout[3],cutoff,v[3],sin_theta,vx[4],
    cos_theta,sin_phi,cos_phi,theta_max,theta_min;
  char tfilename1[1000],tfilename2[1000];
  static pole_warned = FALSE, mode_warned = FALSE;
  static ggrd_boolean shift_to_pos_lon = FALSE;
  const int dims=E->mesh.nsd;
  int top_proc,nfree,nfixed,use_vel,allow_internal;
#ifdef USE_GZDIR
  gzFile *fp1;
#else
  myerror(E,"ggrd_read_vtop_from_file needs to use GZDIR (set USE_GZDIR flag) because of code output");
#endif

  /* number of nodes for this processor at highest MG level */
  nox = E->lmesh.nox;
  noz = E->lmesh.noz;
  noy = E->lmesh.noy;
  noxnoz = nox*noz;
  
  if(E->mesh.toplayerbc != 0)
    allow_internal = TRUE;
  else
    allow_internal = FALSE;

  /* top processor check */
  top_proc = E->parallel.nprocz-1;
  /* output of warning messages */
  if((allow_internal && (E->parallel.me == 0))||(E->parallel.me == top_proc))
    verbose = TRUE;
  else
    verbose = FALSE;
  /* velocities or tractions? */
  switch(E->mesh.topvbc){
  case 0:
    use_vel = FALSE;
    break;
  case 1:
    use_vel = TRUE;
    break;
  default:
    myerror(E,"ggrd_read_vtop_from_file: cannot handle topvbc other than 1 (vel) or 0 (stress)");
    break;
  }

  /* 

     read in plate code files?  this will assign Euler vector
     respective velocities

  */
  use_codes = (E->control.ggrd_vtop_omega[0] > 1e-7)?(1):(0);
  save_codes = 0;
  if(use_codes && (!use_vel)){
    myerror(E,"ggrd_read_vtop_from_file: looking for Euler codes but in traction mode, likely no good");
  }


  if(verbose)
    fprintf(stderr,"ggrd_read_vtop_from_file: init stage, assigning %s, mixed mode: %i\n",
	    ((E->mesh.topvbc)?("velocities"):("tractions")),E->control.ggrd_allow_mixed_vbcs);
	    

  if(use_vel){
    /* 
       velocity scaling, assuming input is cm/yr  
    */
    vscale = E->data.scalev * E->data.timedir;
    if(use_codes)
      vscale *=  E->data.radius_km*1e3/1e6*1e2*M_PI/180.;		/* for deg/Myr -> cm/yr conversion */
    if(E->parallel.me == 0)
      fprintf(stderr,"ggrd_read_vtop_from_file: expecting velocity grids in cm/yr, scaling factor: %g\n",vscale);
  }else{
    /* stress scale, from MPa */
    vscale =  1e6/(E->data.ref_viscosity*E->data.therm_diff/(E->data.radius_km*E->data.radius_km*1e6));
    if((!mode_warned) && (verbose)){
      fprintf(stderr,"ggrd_read_vtop_from_file: WARNING: make sure traction control is what you want, not free slip\n");
      fprintf(stderr,"ggrd_read_vtop_from_file: expecting traction grids in MPa, scaling factor: %g\n",vscale);
      mode_warned = TRUE;
    }
  }
  if (allow_internal || (E->parallel.me_loc[3] == top_proc)) { 
    
    /* 
       top processors for regular operations, all for internal

    */
    
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
      /* 
	 read in grd files (only needed for top processors, really, but
	 leave as is for now
      */
      if(verbose)
	fprintf(stderr,"ggrd_read_vtop_from_file: initializing ggrd velocities/tractions for %s setup\n",
		is_global?("global"):("regional"));
      if(is_global){		/* decide on GMT flag */
	//sprintf(gmt_string,""); /* periodic */
	sprintf(gmt_string,GGRD_GMT_GLOBAL_STRING); /* global */
      }else
	sprintf(gmt_string,"");

      /*
	
      initialization steps
      
      */
      /*
	read in the velocity/traction file(s)
      */
      E->control.ggrd.svt = (struct  ggrd_gt *)calloc(E->control.ggrd.time_hist.nvtimes,sizeof(struct ggrd_gt));
      E->control.ggrd.svp = (struct  ggrd_gt *)calloc(E->control.ggrd.time_hist.nvtimes,sizeof(struct ggrd_gt));
      /* for detecting the actual max */
      E->control.ggrd.svt->bandlim = E->control.ggrd.svp->bandlim = 1e6;
      for(i=0;i < E->control.ggrd.time_hist.nvtimes;i++){
	/* 
	   
	by default, all velocity/traction grids will be stored in memory, this
	may or may not be ideal
	
	*/
	if(!timedep){ /* constant */
	  if(use_codes)
	    sprintf(tfilename1,"%s/code.grd",E->control.ggrd.vtop_dir);
	  else{
	    sprintf(tfilename1,"%s/vt.grd",E->control.ggrd.vtop_dir);
	    sprintf(tfilename2,"%s/vp.grd",E->control.ggrd.vtop_dir);
	  }
	} else {			/* f(t) */
	  if(use_codes)
	    sprintf(tfilename1,"%s/%i/code.grd",E->control.ggrd.vtop_dir,i+1);
	  else{
	    sprintf(tfilename1,"%s/%i/vt.grd",E->control.ggrd.vtop_dir,i+1);
	    sprintf(tfilename2,"%s/%i/vp.grd",E->control.ggrd.vtop_dir,i+1);
	  }
	}
	if(use_codes){
	  if(ggrd_grdtrack_init_general(FALSE,tfilename1,&char_dummy,
					gmt_string,(E->control.ggrd.svt+i),(E->parallel.me == 0),FALSE))
	    myerror(E,"ggrd init error codes");

	}else{
	  if(ggrd_grdtrack_init_general(FALSE,tfilename1,&char_dummy,
					gmt_string,(E->control.ggrd.svt+i),(E->parallel.me == 0),FALSE))
	    myerror(E,"ggrd init error vt");
	  if(ggrd_grdtrack_init_general(FALSE,tfilename2,&char_dummy,
					gmt_string,(E->control.ggrd.svp+i),(E->parallel.me == 0),FALSE))
	    myerror(E,"ggrd init error vp"); 
	}
      }/* all grids read */
      if(use_codes){
	save_codes = 1;
	snprintf(tfilename1,1000,"%s/codes.%d.gz", E->control.data_dir,E->parallel.me);
	fp1 = gzdir_output_open(tfilename1,"w");
      }
      if(verbose)
	if(use_codes)
	  fprintf(stderr,"ggrd_read_vtop_from_file: assigning Euler vector %g, %g, %g to plates with code %i\n",
		  E->control.ggrd_vtop_omega[1],
		  E->control.ggrd_vtop_omega[2],
		  E->control.ggrd_vtop_omega[3],
		  (int)E->control.ggrd_vtop_omega[0]);
	else
	  fprintf(stderr,"ggrd_read_vtop_from_file: done with ggrd vtop BC init, %i timesteps, vp band lim max: %g\n",
		  E->control.ggrd.time_hist.nvtimes,E->control.ggrd.svp->fmaxlim[0]);
    } /* end init part */
    
    /* 
       geographic bounds 
    */
    theta_max = (90-E->control.ggrd.svp[0].south)*M_PI/180-1e-5;
    theta_min = (90-E->control.ggrd.svp[0].north)*M_PI/180+1e-5;
    if(verbose && is_global){
      fprintf(stderr,"ggrd_read_vtop_from_file: determined South/North range: %g/%g\n",
	      E->control.ggrd.svp[0].south,E->control.ggrd.svp[0].north);
    }

    if((E->control.ggrd.time_hist.nvtimes > 1)|| (!E->control.ggrd.vtop_control_init)){
      /* 
	 either first time around, or time-dependent assignment
      */
      age = find_age_in_MY(E);
      if(timedep){
	/* 
	   interpolate by time 
	*/
	if(age < 0){		/* Opposite of other method */
	  interpolate = 0;
	  /* present day should be last file*/
	  i1 = E->control.ggrd.time_hist.nvtimes - 1;
	  if(verbose)
	    fprintf(stderr,"ggrd_read_vtop_from_file: using present day vtop for age = %g\n",age);
	}else{
	  /*  */
	  ggrd_interpol_time(age,&E->control.ggrd.time_hist,&i1,&i2,&f1,&f2,
			     E->control.ggrd.time_hist.vstage_transition);
	  interpolate = 1;
	  if(verbose)
	    fprintf(stderr,"ggrd_read_vtop_from_file: interpolating vtop for age = %g\n",age);
	}
	
      }else{
	interpolate = 0;		/* single timestep, use single file */
	i1 = 0;
	if(verbose)
	  fprintf(stderr,"ggrd_read_vtop_from_file: temporally constant velocity BC \n");
      }
      
      if(verbose)
	fprintf(stderr,"ggrd_read_vtop_from_file: assigning %s BC, timedep: %i time: %g\n",
		(use_vel)?("velocities"):("tractions"),	timedep,age);
      
      /* if mixed BCs are allowed, need to reassign the boundary
	 condition */
      if(E->control.ggrd_allow_mixed_vbcs){
	nfree = nfixed = 0;
	/* 
	   
	mixed BC part

	*/
	if(use_codes)
	  myerror(E,"cannot mix Euler velocities for plate codes and mixed vbcs");
	if(verbose)
	  fprintf(stderr,"WARNING: allowing mixed velocity BCs\n");
	/* velocities larger than the cutoff will be assigned as free
	   slip */
	cutoff = E->control.ggrd.svp->fmaxlim[0] + 1e-5;	  
	for(level=E->mesh.gridmax;level >= E->mesh.gridmin;level--){/* multigrid levels */
	  /* assign BCs to all levels */
	  noxl = E->lmesh.NOX[level];
	  noyl = E->lmesh.NOY[level];
	  nozl = E->lmesh.NOZ[level];
	  noxlnozl = noxl*nozl;

	  for (m=1;m <= E->sphere.caps_per_proc;m++) {
	    /* determine vertical nodes */
	    ggrd_vtop_helper_decide_on_internal_nodes(E,allow_internal,nozl,level,m,verbose,
						      &assign,&botnode,&topnode);
	    /* 
	       loop through all horizontal nodes and assign boundary
	       conditions for all required levels
	    */
	    if(assign){
	      for(i=1;i <= noyl;i++){
		for(j=1;j <= noxl;j++) {
		  nodel =  nozl + (j-1) * nozl + (i-1)*noxlnozl; /* top node =  nozl + (j-1) * nozl + (i-1)*noxlnozl; */
		  /* node location */
		  rout[1] = E->SX[level][m][1][nodel]; /* theta,phi */
		  rout[2] = E->SX[level][m][2][nodel];
		  /* 
		     
		  for global grid, shift theta if too close to poles
		  
		  */
		  if((is_global)&&(rout[1] > theta_max)){
		    if(!pole_warned){
		      fprintf(stderr,"ggrd_read_vtop_from_file: WARNING: shifting theta from %g (%g) to max theta %g (%g)\n",
			      rout[1],90-180/M_PI*rout[1],theta_max,90-180/M_PI*theta_max);
		      pole_warned = TRUE;
		    }
		    rout[1] = theta_max;
		  }
		  if((is_global)&&(rout[1] < theta_min)){
		    if(!pole_warned){
		      fprintf(stderr,"ggrd_read_vtop_from_file: WARNING: shifting theta from %g (%g) to min theta %g (%g)\n",
			      rout[1],90-180/M_PI*rout[1],theta_min,90-180/M_PI*theta_min);
		      pole_warned = TRUE;
		    }
		    rout[1] = theta_min;
		  }
		  /* find vp */
		  if(!ggrd_grdtrack_interpolate_tp(rout[1],rout[2],(E->control.ggrd.svp+i1),
						   vin1,FALSE,shift_to_pos_lon)){
		    fprintf(stderr,"ggrd_read_vtop_from_file: interpolation error at %g, %g\n",
			    rout[1],rout[2]);parallel_process_termination();
		  }
		  if(interpolate){	/* second time */
		    if(!ggrd_grdtrack_interpolate_tp(rout[1],rout[2],(E->control.ggrd.svp+i2),vin2,
						     FALSE,shift_to_pos_lon)){
		      fprintf(stderr,"ggrd_read_mat_from_file: interpolation error at %g, %g\n",
			      rout[1],rout[2]);parallel_process_termination();
		    }
		    v[2] = (f1*vin1[0] + f2*vin2[0]); /* vphi unscaled! */
		  }else{
		    v[2] = vin1[0]; /* vphi */
		  }
		  /* 

		  depth dependent factor goes here 

		  XXX
		  
		  */
		  
		  for(k = botnode;k <= topnode;k++){
		    ontop = ((k==nozl) && (E->parallel.me_loc[3]==E->parallel.nprocz-1))?(TRUE):(FALSE);
		    /* depth loop */
		    nodel =  k + (j-1) * nozl + (i-1)*noxlnozl; /* top node =  nozl + (j-1) * nozl + (i-1)*noxlnozl; */
		    if(fabs(v[2]) > cutoff){
		      /* free slip */
		      nfree++;
		      E->NODE[level][m][nodel] = E->NODE[level][m][nodel] & (~VBX);
		      E->NODE[level][m][nodel] = E->NODE[level][m][nodel] | SBX;
		      E->NODE[level][m][nodel] = E->NODE[level][m][nodel] & (~VBY);
		      E->NODE[level][m][nodel] = E->NODE[level][m][nodel] | SBY;
		    }else{
		      nfixed++;
		      if(use_vel){
			/* no slip */
			E->NODE[level][m][nodel] = E->NODE[level][m][nodel] | VBX;
			E->NODE[level][m][nodel] = E->NODE[level][m][nodel] & (~SBX);
			E->NODE[level][m][nodel] = E->NODE[level][m][nodel] | VBY;
			E->NODE[level][m][nodel] = E->NODE[level][m][nodel] & (~SBY);
		      }else{			fprintf(stderr,"t %i %i\n",level,nodel);

			/* prescribed tractions */
			E->NODE[level][m][nodel] = E->NODE[level][m][nodel] & (~VBX);
			E->NODE[level][m][nodel] = E->NODE[level][m][nodel] | SBX;
			E->NODE[level][m][nodel] = E->NODE[level][m][nodel] & (~VBY);
			E->NODE[level][m][nodel] = E->NODE[level][m][nodel] | SBY;
		      }
		    }
		  } /* depth loop */
		}	/* end x loop */
	      } /* end y loop */
	    } /* actually assign */
	  } /* cap */
	} /* MG level */
	fprintf(stderr,"ggrd_read_vtop_from_file: mixed_bc: %i free %i fixed for CPU %i\n",nfree,nfixed,E->parallel.me);
      }	 /* end mixed BC assign */

      /* 
	 
      now loop through all nodes and assign velocity boundary
      condition values
      
      */
      /* scaled cutoff velocity */
      if(!use_codes)		/* else, is not defined */
	cutoff = E->control.ggrd.svp->fmaxlim[0] * vscale + 1e-5;
      else{
	cutoff = 1e30;
	if(save_codes)	/* those will be surface nodes only */
	  gzprintf(fp1,"%3d %7d\n",m,E->lmesh.nsf);
      }

      /* top leevl */

      for (m=1;m <= E->sphere.caps_per_proc;m++) {
	/* top level only */
	ggrd_vtop_helper_decide_on_internal_nodes(E,allow_internal,E->lmesh.NOZ[E->mesh.gridmax],E->mesh.gridmax,m,verbose,
						  &assign,&botnode,&topnode);
	if(assign){	
	  for(i=1;i <= noy;i++)	{/* loop through surface nodes */
	    for(j=1;j <= nox;j++)    {
	      nodel =  noz + (j-1) * noz + (i-1)*noxnoz; /* top node =  nozg + (j-1) * nozg + (i-1)*noxgnozg; */	
	      /*  */
	      rout[1] = E->sx[m][1][nodel]; /* theta,phi coordinates */
	      rout[2] = E->sx[m][2][nodel];
	      /* 
		 
	      for global grid, shift theta if too close to poles
	      
	      */
	      if((is_global)&&(rout[1] > theta_max)){
		if(!pole_warned){
		  fprintf(stderr,"WARNING: shifting theta from %g (%g) to max theta %g (%g)\n",
			  rout[1],90-180/M_PI*rout[1],theta_max,90-180/M_PI*theta_max);
		  pole_warned = TRUE;
		}
		rout[1] = theta_max;
	      }
	      if((is_global)&&(rout[1] < theta_min)){
		if(!pole_warned){
		  fprintf(stderr,"WARNING: shifting theta from %g (%g) to min theta %g (%g)\n",
			  rout[1],90-180/M_PI*rout[1],theta_min,90-180/M_PI*theta_min);
		  pole_warned = TRUE;
		}
		rout[1] = theta_min;
	      }
	      if(!ggrd_grdtrack_interpolate_tp(rout[1],rout[2],(E->control.ggrd.svt+i1),
					       vin1,FALSE,shift_to_pos_lon)){
		fprintf(stderr,"ggrd_read_vtop_from_file: interpolation error at %g, %g\n",
			rout[1],rout[2]);
		parallel_process_termination();
	      }
	      if(!use_codes)
		if(!ggrd_grdtrack_interpolate_tp(rout[1],rout[2],(E->control.ggrd.svp+i1),
						 (vin1+1),FALSE,shift_to_pos_lon)){
		  fprintf(stderr,"ggrd_read_vtop_from_file: interpolation error at %g, %g\n",
			  rout[1],rout[2]);
		  parallel_process_termination();
		}
	      if(interpolate){	/* second time */
		if(!ggrd_grdtrack_interpolate_tp(rout[1],rout[2],(E->control.ggrd.svt+i2),vin2,
						 FALSE,shift_to_pos_lon)){
		  fprintf(stderr,"ggrd_read_mat_from_file: interpolation error at %g, %g\n",
			  rout[1],rout[2]);
		  parallel_process_termination();
		}
		if(!use_codes){
		  if(!ggrd_grdtrack_interpolate_tp(rout[1],rout[2],(E->control.ggrd.svp+i2),(vin2+1),
						   FALSE,shift_to_pos_lon)){
		    fprintf(stderr,"ggrd_read_mat_from_file: interpolation error at %g, %g\n",
			    rout[1],rout[2]);
		    parallel_process_termination();
		  }
		  v[1] = (f1*vin1[0] + f2*vin2[0])*vscale; /* theta */
		  v[2] = (f1*vin1[1] + f2*vin2[1])*vscale; /* phi */
		}else{
		  v[1] = (f1*vin1[0] + f2*vin2[0]); /* theta */
		}
	      }else{
		if(!use_codes){
		  v[1] = vin1[0]*vscale; /* theta */
		  v[2] = vin1[1]*vscale; /* phi */
		}else{
		  v[1] = vin1[0];	/* theta */
		}
	      }
	      
	      /* 
		 
	      depth dependent factor goes here 
	      
	      XXX
	      
	      */
	      for(k = botnode;k <= topnode;k++){
		ontop = ((k==noz) && (E->parallel.me_loc[3]==E->parallel.nprocz-1))?(TRUE):(FALSE);
		nodel = k + (j-1) * noz + (i-1)*noxnoz ; /*  node =  k + (j-1) * nozg + (i-1)*noxgnozg; */	
		if(use_codes){
		  /* find code from v[1], theta */
		  code = (int)(v[1] + 0.5);
		  if(save_codes)	/* lon lat code */
		    gzprintf(fp1, "%9.4f %9.4f %i\n",rout[2]/M_PI*180,90-rout[1]*180/M_PI,code);
		  if((int)E->control.ggrd_vtop_omega[0] == code){
		    /* within plate */
		    sin_theta=sin(rout[1]);cos_theta=cos(rout[1]);
		    sin_phi  =sin(rout[2]);cos_phi=  cos(rout[2]);
		    /* compute spherical velocities in cm/yr at this
		       location, assuming rotation pole is in deg/Myr */
		    vx[1]=E->control.ggrd_vtop_omega[2]*E->x[m][3][nodel] - E->control.ggrd_vtop_omega[3]*E->x[m][2][nodel]; 
		    vx[2]=E->control.ggrd_vtop_omega[3]*E->x[m][1][nodel] - E->control.ggrd_vtop_omega[1]*E->x[m][3][nodel]; 
		    vx[3]=E->control.ggrd_vtop_omega[1]*E->x[m][2][nodel] - E->control.ggrd_vtop_omega[2]*E->x[m][1][nodel]; 
		    /*  */
		    v[1]= cos_theta*cos_phi*vx[1] + cos_theta*sin_phi*vx[2] - sin_theta*vx[3]; /* theta */
		    v[2]=-          sin_phi*vx[1] +           cos_phi*vx[2]; /* phie */
		    /* scale */
		    v[1] *= vscale;v[2] *= vscale;
		  }else{
		    v[1] = v[2] = 0.0;
		  }
		}
		/* assign velociites */
		if(fabs(v[2]) > cutoff){
		  /* huge velocitie - free slip */
		  E->sphere.cap[m].VB[1][nodel] = 0;	/* theta */
		  E->sphere.cap[m].VB[2][nodel] = 0;	/* phi */
		}else{
		  /* regular no slip , assign velocities/tractions as BCs */
		  E->sphere.cap[m].VB[1][nodel] = v[1];	/* theta */
		  E->sphere.cap[m].VB[2][nodel] = v[2];	/* phi */
		}
		if(use_vel && ontop)
		  E->sphere.cap[m].VB[3][nodel] = 0.0; /* r */
	      }	/* end z */
	    } /* end x */
	  } /* end y */
	} /* end assign */
      } /* end cap loop */
      
      if((!timedep)&&(!E->control.ggrd.vtop_control_init)){			/* forget the grids */
	ggrd_grdtrack_free_gstruc(E->control.ggrd.svt);
	ggrd_grdtrack_free_gstruc(E->control.ggrd.svp);
      }
    } /* end assignment branch */
    if(use_codes && save_codes){
      save_codes = 0;
      gzclose(fp1);
    }
  } /* end top proc branch */

  /*  */
  E->control.ggrd.vtop_control_init = TRUE;
  //if(E->parallel.me == 0)
  //fprintf(stderr,"vtop from grd done: %i\n",lc++);
}


void ggrd_vtop_helper_decide_on_internal_nodes(struct All_variables *E,	/* input */
					       int allow_internal,
					       int nozl,int level,int m,int verbose,
					       int *assign, /* output */
					       int *botnode,int *topnode)
{
  int k;
  /* default is assign to top */
  *assign = TRUE;
  *botnode = *topnode = nozl;
  /* determine vertical nodes */
  if(allow_internal){
    /* internal */
    if(E->mesh.toplayerbc > 0){
      /* check for internal nodes in layers */
      for(k=nozl;k >= 1;k--){
	if(E->SX[level][m][3][k] < E->mesh.toplayerbc_r) /* assume regular mesh structure */
	  break;
      }
      if(k == nozl){	/*  */
	*assign = FALSE;
      }else{
	*assign = TRUE;*topnode = nozl;*botnode = k+1;
      }
    }else if(E->mesh.toplayerbc < 0){
      /* only one internal node */
      if(level == E->mesh.gridmax)
	*botnode = nozl + E->mesh.toplayerbc;
      else{
	*botnode = nozl + (int)((float)E->mesh.toplayerbc / pow(2.,(float)(E->mesh.gridmax-level)));
      }
      *assign = TRUE;
      *topnode = *botnode;
    }else{
      fprintf(stderr,"ggrd_vtop_helper_decide_on_internal_nodes: toplayerbc %i, r_min: %g\n", 
	      E->mesh.toplayerbc,E->mesh.toplayerbc_r);
      myerror(E,"ggrd_vtop_helper_decide_on_internal_nodes: logic error");
    }
  }
  if(verbose)
    fprintf(stderr,"ggrd_vtop_helper_decide_on_internal_nodes: mixed: internal: %i assign: %i k: %i to %i (%i), r_min: %g\n",
	    allow_internal,*assign,*botnode,*topnode,nozl,E->mesh.toplayerbc_r);
  
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
			E->viscosity.zbase_layer[E->control.ggrd.ray_control-1])/
	      E->viscosity.zbase_layer[E->control.ggrd.ray_control-1];
	    fac = cos(xloc*1.5707963267);fac *= fac; /* cos^2
							tapering,
							factor
							decrease from
							1 at surface
							to zero at
							boundary */
	    bnew = buoy[m][node] * E->control.surface_rayleigh[snode]; /* modified rayleigh */
	    /* debugging */
	    /*   fprintf(stderr,"z: %11g tl: %i zm: %11g fac: %11g sra: %11g bnew: %11g bold: %11g\n", */
	    /* 	    	    (1 - E->sx[m][3][node])*E->data.radius_km,E->control.ggrd.ray_control, */
	    /* 	    	    E->viscosity.zbase_layer[E->control.ggrd.ray_control-1]*E->data.radius_km, */
	    /* 	    	    fac,E->control.surface_rayleigh[snode],(fac * bnew + (1-fac)*buoy[m][node]),buoy[m][node]); */
	    buoy[m][node] = fac * bnew + (1-fac)*buoy[m][node];
	  }
	}
      }
    }
  }

}



/*


read in anisotropic viscosity from a directory which holds


vis2.grd for the viscosity factors  (1 - eta_S/eta)

nr.grd, nt.grd, np.grd for the directors

*/
void ggrd_read_anivisc_from_file(struct All_variables *E, int is_global)
{
  MPI_Status mpi_stat;
  int mpi_rc;
  int mpi_inmsg, mpi_success_message = 1;
  int m,el,i,j,k,l,inode,i1,i2,elxlz,elxlylz,ind,nel;
  int llayer,nox,noy,noz,level,lselect,idim,elx,ely,elz;
  char gmt_string[10],char_dummy;
  double vis2,ntheta,nphi,nr,rout[3],xloc[4],nlen;
  double cvec[3],base[9];
  char tfilename[1000];
  static ggrd_boolean shift_to_pos_lon = FALSE;
  const int dims=E->mesh.nsd;
  const int ends = enodes[dims];
  FILE *in;
  struct ggrd_gt *vis2_grd,*ntheta_grd,*nphi_grd,*nr_grd;
  const int vpts = vpoints[E->mesh.nsd];

  
  nox=E->mesh.nox;noy=E->mesh.noy;noz=E->mesh.noz;
  elx=E->lmesh.elx;elz=E->lmesh.elz;ely=E->lmesh.ely;
  elxlz = elx * elz;
  elxlylz = elxlz * ely;

#ifndef CITCOM_ALLOW_ORTHOTROPIC_VISC
  fprintf(stderr,"ggrd_read_anivisc_from_file: error, need to compile with CITCOM_ALLOW_ORTHOTROPIC_VISC\n");
  parallel_process_termination();
#endif
  if(!E->viscosity.allow_orthotropic_viscosity)
    myerror(E,"ggrd_read_anivisc_from_file: called, but allow_orthotropic_viscosity is FALSE?!");
  
  /* isotropic default */
  for(i=E->mesh.gridmin;i <= E->mesh.gridmax;i++){
    nel  = E->lmesh.NEL[i];
    for (j=1;j<=E->sphere.caps_per_proc;j++) {
      for(k=1;k <= nel;k++){
	for(l=1;l <= vpts;l++){ /* assign to all integration points */
	  ind = (k-1)*vpts + l;
	  E->EVI2[i][j][ind] = 0.0;
	  E->EVIn1[i][j][ind] = 1.0; E->EVIn2[i][j][ind] = E->EVIn3[i][j][ind] = 0.0;
	}
      }
    }
  }
  /* 
     
  rest

  */
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
  vis2_grd =   (struct  ggrd_gt *)calloc(1,sizeof(struct ggrd_gt));
  nphi_grd =   (struct  ggrd_gt *)calloc(1,sizeof(struct ggrd_gt));
  nr_grd =     (struct  ggrd_gt *)calloc(1,sizeof(struct ggrd_gt));
  ntheta_grd = (struct  ggrd_gt *)calloc(1,sizeof(struct ggrd_gt));


  if(E->parallel.me==0)
    if(E->viscosity.anivisc_layer > 0)
      fprintf(stderr,"ggrd_read_anivisc_from_file: initializing, assigning to all elements above %g km, input is %s\n",
	      E->data.radius_km*E->viscosity.zbase_layer[E->viscosity.anivisc_layer - 1],
	      (is_global)?("global"):("regional"));
    else
      fprintf(stderr,"ggrd_read_anivisc_from_file: initializing, assigning to all elements between  %g and %g km, input is %s\n",
	      E->data.radius_km*((E->viscosity.anivisc_layer<-1)?(E->viscosity.zbase_layer[-E->viscosity.anivisc_layer - 2]):(0)),
	      E->data.radius_km*E->viscosity.zbase_layer[-E->viscosity.anivisc_layer - 1],
	      (is_global)?("global"):("regional"));

  /* 
     read viscosity ratio, and east/north direction of normal azimuth 
  */
  /* viscosity factor */
  sprintf(tfilename,"%s/vis2.grd",E->viscosity.anisotropic_init_dir);
  if(ggrd_grdtrack_init_general(FALSE,tfilename,"",gmt_string,
				vis2_grd,(E->parallel.me == 0),FALSE))
    myerror(E,"ggrd init error");
  /* n_r */
  sprintf(tfilename,"%s/nr.grd",E->viscosity.anisotropic_init_dir);
  if(ggrd_grdtrack_init_general(FALSE,tfilename,"",gmt_string,
				nr_grd,(E->parallel.me == 0),FALSE))
    myerror(E,"ggrd init error");
  /* n_theta */
  sprintf(tfilename,"%s/nt.grd",E->viscosity.anisotropic_init_dir);
  if(ggrd_grdtrack_init_general(FALSE,tfilename,"",gmt_string,
				ntheta_grd,(E->parallel.me == 0),FALSE))
    myerror(E,"ggrd init error");
  /* n_phi */
  sprintf(tfilename,"%s/np.grd",E->viscosity.anisotropic_init_dir);
  if(ggrd_grdtrack_init_general(FALSE,tfilename,"",gmt_string,
				nphi_grd,(E->parallel.me == 0),FALSE))
    myerror(E,"ggrd init error");

  /* done */
  if(E->parallel.me <  E->parallel.nproc-1){ /* tell the next proc to go ahead */
    mpi_rc = MPI_Send(&mpi_success_message, 1,
		      MPI_INT, (E->parallel.me+1), 0, E->parallel.world);
  }else{
    fprintf(stderr,"ggrd_read_anivisc_from_file: last processor done with ggrd init\n");
    fprintf(stderr,"ggrd_read_anivisc_from_file: WARNING: assuming a regular grid geometry\n");
  }
  /*

  loop through all elements and assign

  */
  for (m=1;m <= E->sphere.caps_per_proc;m++) {
    for (j=1;j <= elz;j++)  {	/* this assumes a regular grid sorted as in (1)!!! */
      if(((E->viscosity.anivisc_layer > 0)&&(E->mat[m][j] <=   E->viscosity.anivisc_layer))||
	 ((E->viscosity.anivisc_layer < 0)&&(E->mat[m][j] ==  -E->viscosity.anivisc_layer))){
	/* within top layers */
	for (k=1;k <= ely;k++){
	  for (i=1;i <= elx;i++)   {
	    /* eq.(1) */
	    el = j + (i-1) * elz + (k-1)*elxlz;
	    /*
	      find average coordinates
	    */
	    xloc[1] = xloc[2] = xloc[3] = 0.0;
	    for(inode=1;inode <= 4;inode++){
	      ind = E->ien[m][el].node[inode];
	      xloc[1] += E->x[m][1][ind];
	      xloc[2] += E->x[m][2][ind];
	      xloc[3] += E->x[m][3][ind];
	    }
	    xloc[1]/=4.;xloc[2]/=4.;xloc[3]/=4.;
	    xyz2rtpd(xloc[1],xloc[2],xloc[3],rout); /* convert to spherical */

	    /* vis2 */
	    if(!ggrd_grdtrack_interpolate_tp(rout[1],rout[2],
					     vis2_grd,&vis2,FALSE,shift_to_pos_lon)){
		fprintf(stderr,"ggrd_read_anivisc_from_file: interpolation error at lon: %g lat: %g\n",
			rout[2]*180/M_PI,90-rout[1]*180/M_PI);
		parallel_process_termination();
	    }
	    /* nr */
	    if(!ggrd_grdtrack_interpolate_tp(rout[1],rout[2],
					     nr_grd,&nr,FALSE,shift_to_pos_lon)){
		fprintf(stderr,"ggrd_read_anivisc_from_file: interpolation error at lon: %g lat: %g\n",
			rout[2]*180/M_PI,90-rout[1]*180/M_PI);
		parallel_process_termination();
	    }
	    /* ntheta */
	    if(!ggrd_grdtrack_interpolate_tp(rout[1],rout[2],
					     ntheta_grd,&ntheta,FALSE,shift_to_pos_lon)){
		fprintf(stderr,"ggrd_read_anivisc_from_file: interpolation error at lon: %g lat: %g\n",
			rout[2]*180/M_PI,90-rout[1]*180/M_PI);
		parallel_process_termination();
	    }
	    /* nphi */
	    if(!ggrd_grdtrack_interpolate_tp(rout[1],rout[2],
					     nphi_grd,&nphi,FALSE,shift_to_pos_lon)){
		fprintf(stderr,"ggrd_read_anivisc_from_file: interpolation error at lon: %g lat: %g\n",
			rout[2]*180/M_PI,90-rout[1]*180/M_PI);
		parallel_process_termination();
	    }
	    nlen = sqrt(nphi*nphi + ntheta*ntheta + nr*nr); /* correct,
							       because
							       interpolation
							       might
							       have
							       screwed
							       up
							       initialization */
	    nphi /= nlen; ntheta /= nlen;nr /= nlen;
	    calc_cbase_at_tp_d(rout[1],rout[2],base); /* convert from
							 spherical
							 coordinates
							 to
							 Cartesian */
	    convert_pvec_to_cvec_d(nr,ntheta,nphi,base,cvec);
	    for(l=1;l <= vpts;l++){ /* assign to all integration points */
	      ind = (el-1)*vpts + l;
	      E->EVI2[E->mesh.gridmax][m][ind]  =   vis2;
	      E->EVIn1[E->mesh.gridmax][m][ind]  = cvec[0];
	      E->EVIn2[E->mesh.gridmax][m][ind]  = cvec[1];
	      E->EVIn3[E->mesh.gridmax][m][ind]  = cvec[2];
	    }
	  }
	}
      }	/* end insize lith */
    }	/* end elz loop */
  } /* end m loop */


  ggrd_grdtrack_free_gstruc(vis2_grd);
  ggrd_grdtrack_free_gstruc(nr_grd);
  ggrd_grdtrack_free_gstruc(ntheta_grd);
  ggrd_grdtrack_free_gstruc(nphi_grd);
  
  
}


#endif


