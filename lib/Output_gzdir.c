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
/* Routine to process the output of the finite element cycles
   and to turn them into a coherent suite  files  */

/*

this version uses gzipped, ascii output to subdirectories for the
ascii-gz option

if, additionally, gzdir.vtk_io = 1, will write different format files
                                    for later post-processing into VTK

		  gzdir.vtk_io = 2, will try to write legacy serial VTK (experimental)

		  gzdir.vtk_io = 3, will try to write to legacy parallel VTK (experimental)



		  the VTK output is the "legacy" type, requires that
		  all processors see the same filesystem, and will
		  likely lead to a bottleneck for large CPU
		  computations as each processor has to wait til the
		  previous is done.

TWB

*/
#ifdef USE_GZDIR


//#define ASCII_DEBUG

#include <zlib.h>

#define BE_WERROR {myerror(E,"write error be output");}
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "element_definitions.h"
#include "global_defs.h"
#include "parsing.h"
#include "parallel_related.h"
#include "output.h"
/* Big endian crap */
#include <string.h>
#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif


void be_flipit(void *, void *, size_t );
void be_flip_byte_order(void *, size_t );
int be_is_little_endian(void);
int be_write_float_to_file(float *, int , FILE *);
int be_write_int_to_file(int *, int , FILE *);
void myfprintf(FILE *,char *);
void calc_cbase_at_node(int , float *,struct All_variables *);

/*  */
void get_vtk_filename(char *,int,struct All_variables *,int);

gzFile *gzdir_output_open(char *,char *);
void gzdir_output(struct All_variables *, int );
void gzdir_output_comp_nd(struct All_variables *, int);
void gzdir_output_comp_el(struct All_variables *, int);
void gzdir_output_coord(struct All_variables *);
void gzdir_output_mat(struct All_variables *);
void gzdir_output_velo_temp(struct All_variables *, int);
void gzdir_output_visc_prepare(struct All_variables *, float **);
void gzdir_output_visc(struct All_variables *, int);
void gzdir_output_surf_botm(struct All_variables *, int);
void gzdir_output_geoid(struct All_variables *, int);
void gzdir_output_stress(struct All_variables *, int);
void gzdir_output_horiz_avg(struct All_variables *, int);
void gzdir_output_tracer(struct All_variables *, int);
void gzdir_output_pressure(struct All_variables *, int);
void gzdir_output_heating(struct All_variables *, int);


void sub_netr(float, float, float, float *, float *, double *);
double determine_model_net_rotation(struct All_variables *,double *);


void restart_tic_from_gzdir_file(struct All_variables *);

void calc_cbase_at_tp(float , float , float *);
void rtp2xyz(float , float , float, float *);
void convert_pvec_to_cvec(float ,float , float , float *,float *);
void *safe_malloc (size_t );

int open_file_zipped(char *, FILE **,struct All_variables *);
void gzip_file(char *);

#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC
#include "anisotropic_viscosity.h"
void gzdir_output_avisc(struct All_variables *, int);
#endif

extern void temperatures_conform_bcs(struct All_variables *);
extern void myerror(struct All_variables *,char *);
extern void mkdatadir(const char *);
extern void heat_flux(struct All_variables *);
extern void get_STD_topo(struct All_variables *, float**, float**,
                         float**, float**, int);
extern void get_CBF_topo(struct All_variables *, float**, float**);

/**********************************************************************/


void gzdir_output(struct All_variables *E, int out_cycles)
{
  char output_dir[255];

  if (out_cycles == 0 ){
    /* initial I/O */
    
    gzdir_output_coord(E);
    output_domain(E);
    /*gzdir_output_mat(E);*/

    if (E->output.coord_bin)
        output_coord_bin(E);
  }

  /*
     make a new directory for all the other output

     (all procs need to do that, because we might be using a local tmp
     dir)

  */
  /* make a directory */
  snprintf(output_dir,255,"%s/%d",E->control.data_dir,out_cycles);

  mkdatadir(output_dir);


  /* output */

  gzdir_output_velo_temp(E, out_cycles); /* don't move this around,
					else new VTK output won't
					work */
  gzdir_output_visc(E, out_cycles);
#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC
  gzdir_output_avisc(E, out_cycles);
#endif

  gzdir_output_surf_botm(E, out_cycles);

  /* optiotnal output below */
  /* compute and output geoid (in spherical harmonics coeff) */
  if (E->output.geoid)
      gzdir_output_geoid(E, out_cycles);

  if (E->output.stress){
      gzdir_output_stress(E, out_cycles);
  }
  if (E->output.pressure)
    gzdir_output_pressure(E, out_cycles);

  if (E->output.horiz_avg)
      gzdir_output_horiz_avg(E, out_cycles);

  if(E->control.tracer){
    if(E->output.tracer ||
       (out_cycles == E->advection.max_timesteps))
      gzdir_output_tracer(E, out_cycles);
  }

  if (E->output.comp_nd && E->composition.on)
      gzdir_output_comp_nd(E, out_cycles);

  if (E->output.comp_el && E->composition.on)
      gzdir_output_comp_el(E, out_cycles);

  if(E->output.heating && E->control.disptn_number != 0)
      gzdir_output_heating(E, out_cycles);

}


gzFile *gzdir_output_open(char *filename,char *mode)
{
  gzFile *fp1;

  if (*filename) {
    fp1 = (gzFile *)gzopen(filename,mode);
    if (!fp1) {
      fprintf(stderr,"gzdir: cannot open file '%s'\n",filename);
      parallel_process_termination();
    }
  }else{
      fprintf(stderr,"gzdir: no file name given '%s'\n",filename);
      parallel_process_termination();
  }
  return fp1;
}

/*

initialization output of geometries, only called once


 */
void gzdir_output_coord(struct All_variables *E)
{
  int i, j, offset,ix[9],out;
  char output_file[255],ostring[255],message[255];
  float x[3];
  gzFile *gz1;
  FILE *fp1;
  MPI_Status mpi_stat;
  int mpi_rc, mpi_inmsg, mpi_success_message = 1;
  if((E->output.gzdir.vtk_io == 2)||(E->output.gzdir.vtk_io == 3)){
    /*
       direct VTK file output
    */
    if(E->output.gzdir.vtk_io == 2) /* serial */
      parallel_process_sync(E);
    /*

    start geometry pre-file, to which data will get appended later

    */
    E->output.gzdir.vtk_ocount = -1;
    get_vtk_filename(output_file,1,E,0); /* geometry file */
    if(E->parallel.me == 0){
      /* start log file */
      snprintf(message,255,"%s/vtk_time.log",E->control.data_dir);
      E->output.gzdir.vtk_fp = output_open(message,"w");
    }
    if((E->parallel.me == 0) || (E->output.gzdir.vtk_io == 3)){
      /* either in first CPU or parallel output */
      /* start geo file */
      fp1 = output_open(output_file,"w");
      myfprintf(fp1,"# vtk DataFile Version 2.0\n");
      myfprintf(fp1,"model name, extra info\n");
#ifdef ASCII_DEBUG
      myfprintf(fp1,"ASCII\n");
#else
      myfprintf(fp1,"BINARY\n");
#endif
      myfprintf(fp1,"DATASET UNSTRUCTURED_GRID\n");
      if(E->output.gzdir.vtk_io == 2) /* serial */
	sprintf(message,"POINTS %i float\n", /* total number of nodes */
		E->lmesh.nno * E->parallel.nproc *
		E->sphere.caps_per_proc);
      else			/* parallel */
	sprintf(message,"POINTS %i float\n",
		E->lmesh.nno * E->sphere.caps_per_proc);
      myfprintf(fp1,message);
    }else{			/* serial output */
      /* if not first CPU, wait for previous before appending */
      mpi_rc = MPI_Recv(&mpi_inmsg, 1, MPI_INT, (E->parallel.me-1), 0, E->parallel.world, &mpi_stat);
      /* open for append */
      fp1 = output_open(output_file,"a");
    }
    out = 0;
    /* write nodal coordinate to file, big endian */
      for(i=1;i <= E->lmesh.nno;i++) {
	/* cartesian coordinates */
	x[0]=E->x[CPPR][1][i];x[1]=E->x[CPPR][2][i];x[2]=E->x[CPPR][3][i];
	if(be_write_float_to_file(x,3,fp1) != 3)
	  BE_WERROR;
	out++;
      }
    if(E->output.gzdir.vtk_io == 2){ /* serial output, close and have
					next one write */
      fclose(fp1);fflush(fp1);		/* close file and flush buffer */
      if(E->parallel.me <  E->parallel.nproc-1){/* send to next if not last*/
	mpi_rc = MPI_Send(&mpi_success_message, 1, MPI_INT, (E->parallel.me+1), 0, E->parallel.world);
      }
      /*
	 node numbers for all the elements
      */
      parallel_process_sync(E);
    }
    if((E->output.gzdir.vtk_io == 3) || (E->parallel.me == 0)){ /* in first CPU, or parallel output */
      if(E->output.gzdir.vtk_io == 2){ /* need to reopen, serial */
	fp1 = output_open(output_file,"a");
	j = E->parallel.nproc * E->lmesh.nel *
	  E->sphere.caps_per_proc; /* total number of elements */
      }else{			/* parallel */
	j = E->lmesh.nel * E->sphere.caps_per_proc;
      }
      sprintf(message,"CELLS %i %i\n", /* number of elements
				      total number of int entries

				       */
	      j,j*(enodes[E->mesh.nsd]+1));
      myfprintf(fp1,message);
    }else{
      /* if not first, wait for previous */
      mpi_rc = MPI_Recv(&mpi_inmsg, 1, MPI_INT, (E->parallel.me-1), 0, E->parallel.world, &mpi_stat);
      fp1 = output_open(output_file,"a");
    }
    /*
       write CELL element nodes
    */
    if(enodes[E->mesh.nsd] != 8)
      myerror(E,"vtk error, only eight node hexes supported");
    if(E->output.gzdir.vtk_io == 2){ /* serial, global node numbers */
      offset = E->lmesh.nno * E->parallel.me - 1;
    }else{			/* parallel, only use local node numbers? */
      offset = -1;
    }
    ix[0] = enodes[E->mesh.nsd];
      for(i=1;i <= E->lmesh.nel;i++) {
	/*
	   need to add offset according to the processor for global
	   node numbers
	*/
	ix[1]= E->ien[CPPR][i].node[1]+offset;ix[2] = E->ien[CPPR][i].node[2]+offset;
	ix[3]= E->ien[CPPR][i].node[3]+offset;ix[4] = E->ien[CPPR][i].node[4]+offset;
	ix[5]= E->ien[CPPR][i].node[5]+offset;ix[6] = E->ien[CPPR][i].node[6]+offset;
	ix[7]= E->ien[CPPR][i].node[7]+offset;ix[8] = E->ien[CPPR][i].node[8]+offset;
	if(be_write_int_to_file(ix,9,fp1)!=9)
	  BE_WERROR;
      }
    if(E->output.gzdir.vtk_io == 2){ /* serial IO */
      fclose(fp1);fflush(fp1);		/* close file and flush buffer */
      if(E->parallel.me <  E->parallel.nproc-1)
	mpi_rc = MPI_Send(&mpi_success_message, 1, MPI_INT, (E->parallel.me+1), 0, E->parallel.world);
      parallel_process_sync(E);
    }
    if((E->output.gzdir.vtk_io==3) || (E->parallel.me == 0) ){
      if(E->output.gzdir.vtk_io == 2){ /* serial */
	fp1 = output_open(output_file,"a");
	j=E->parallel.nproc*E->lmesh.nel*E->sphere.caps_per_proc;
      }else{			/* parallel */
	j = E->lmesh.nel*E->sphere.caps_per_proc;
      }
      sprintf(message,"CELL_TYPES %i\n",j); /* number of elements*/
      myfprintf(fp1,message);
      ix[0] = 12;
      for(i=0;i<j;i++)
	if(be_write_int_to_file(ix,1,fp1)!=1)BE_WERROR;
      fclose(fp1);fflush(fp1);		/* all procs close file and flush buffer */
      if(E->parallel.me == 0)
	fprintf(stderr,"vtk_io: vtk geometry done for %s\n",output_file);
    }
    /* done straight VTK output, geometry part */
  }else{
    /*

    either zipped regular, or old VTK type for post-processing

    */
    /*
       don't use data file name
    */
    snprintf(output_file,255,"%s/coord.%d.gz",
	   E->control.data_dir,E->parallel.me);
    gz1 = gzdir_output_open(output_file,"w");

    /* nodal coordinates */
      gzprintf(gz1,"%3d %7d\n",CPPR,E->lmesh.nno);
      for(i=1;i<=E->lmesh.nno;i++)
	gzprintf(gz1,"%.6e %.6e %.6e\n",
		 E->sx[CPPR][1][i],E->sx[CPPR][2][i],E->sx[CPPR][3][i]);

    gzclose(gz1);
    if(E->output.gzdir.vtk_io == 1){
      /*

      output of Cartesian coordinates and element connectivitiy for
      vtk visualization

      */
      /*
	 nodal coordinates in Cartesian
      */
      snprintf(output_file,255,"%s/vtk_ecor.%d.gz",
	       E->control.data_dir,E->parallel.me);
      gz1 = gzdir_output_open(output_file,"w");
	for(i=1;i <= E->lmesh.nno;i++) {
	  gzprintf(gz1,"%9.6f %9.6f %9.6f\n", /* cartesian nodal coordinates */
		   E->x[CPPR][1][i],E->x[CPPR][2][i],E->x[CPPR][3][i]);
	}
      gzclose(gz1);
      /*
	 connectivity for all elements
      */
      offset = E->lmesh.nno * E->parallel.me - 1;
      snprintf(output_file,255,"%s/vtk_econ.%d.gz",
	       E->control.data_dir,E->parallel.me);
      gz1 = gzdir_output_open(output_file,"w");
	for(i=1;i <= E->lmesh.nel;i++) {
	  gzprintf(gz1,"%2i\t",enodes[E->mesh.nsd]);
	  if(enodes[E->mesh.nsd] != 8){
	    gzprintf(stderr,"gzdir: Output: error, only eight node hexes supported");
	    parallel_process_termination();
	  }
	  /*
	     need to add offset according to the processor for global
	     node numbers
	  */
	  gzprintf(gz1,"%6i %6i %6i %6i %6i %6i %6i %6i\n",
		   E->ien[CPPR][i].node[1]+offset,E->ien[CPPR][i].node[2]+offset,
		   E->ien[CPPR][i].node[3]+offset,E->ien[CPPR][i].node[4]+offset,
		   E->ien[CPPR][i].node[5]+offset,E->ien[CPPR][i].node[6]+offset,
		   E->ien[CPPR][i].node[7]+offset,E->ien[CPPR][i].node[8]+offset);
	}
      gzclose(gz1);
    } /* end vtkio = 1 (pre VTK) */
  }

}

/*

this needs to be called after the geometry files have been
established, and before any of the other stuff if VTK straight output
is chosen


*/
void gzdir_output_velo_temp(struct All_variables *E, int cycles)
{
  int i, j, k,os;
  char output_file[255],output_file2[255],message[255],geo_file[255];
  float cvec[3],vcorr[3];
  double omega[3],oamp;
  gzFile *gzout;
  FILE *fp1;
  /* for dealing with several processors */
  MPI_Status mpi_stat;
  int mpi_rc;
  int mpi_inmsg, mpi_success_message = 1;


  if(E->output.gzdir.vtk_io){	/* all VTK modes need basis vectors */
    os = E->lmesh.nno*9;
    if((!E->output.gzdir.vtk_base_init) ||(!E->output.gzdir.vtk_base_save)){
      /* either not computed, or need to compute anew */
      if(!E->output.gzdir.vtk_base_init) /* init space */
	E->output.gzdir.vtk_base = (float *)safe_malloc(sizeof(float)*os*E->sphere.caps_per_proc);
      /* compute */
	for(i=1,k=0;i <= E->lmesh.nno;i++,k += 9){
	  /* cartesian basis vectors at theta, phi */
	  calc_cbase_at_node(i,(E->output.gzdir.vtk_base+k),E);
	}
      E->output.gzdir.vtk_base_init = 1;
    }
  }

  if(E->output.gzdir.rnr){	/* remove the whole model net rotation */
    if((E->control.remove_rigid_rotation || E->control.remove_angular_momentum) &&
       (E->parallel.me == 0))	/* that's not too terrible but wastes time */
      fprintf(stderr,"WARNING: both gzdir.rnr and remove_rigid_rotation are switched on!\n");
    oamp = determine_model_net_rotation(E,omega);
    if(E->parallel.me == 0)
      fprintf(stderr,"gzdir_output_velo_temp: removing net rotation: |%8.3e, %8.3e, %8.3e| = %8.3e\n",
	      omega[0],omega[1],omega[2],oamp);
  }
  if((E->output.gzdir.vtk_io == 2) || (E->output.gzdir.vtk_io == 3)){
    /*

    direct VTK

    */
    if(E->output.gzdir.vtk_io == 2)
      parallel_process_sync(E);	/* serial needs sync */

    E->output.gzdir.vtk_ocount++; /* regular output file name */
    get_vtk_filename(geo_file,1,E,cycles);
    get_vtk_filename(output_file,0,E,cycles);
    /*

    start with temperature

    */
    if((E->parallel.me == 0) || (E->output.gzdir.vtk_io == 3)){
      /* copy geo file over to start out vtk file */
      snprintf(output_file2,255,"cp %s %s",geo_file,output_file);
      system(output_file2);
      /* should we do something to check if this has worked? */
      if(E->parallel.me == 0){
	/* write a time log */
	fprintf(E->output.gzdir.vtk_fp,"%12i %12i %12.6e %s\n",
		E->output.gzdir.vtk_ocount,cycles,E->monitor.elapsed_time,output_file);
      }
      fp1 = output_open(output_file,"a");
      if(E->output.gzdir.vtk_io == 2) /* serial */
	sprintf(message,"POINT_DATA %i\n",E->lmesh.nno*E->parallel.nproc*E->sphere.caps_per_proc);
      else			/* parallel */
	sprintf(message,"POINT_DATA %i\n",E->lmesh.nno*E->sphere.caps_per_proc);
      myfprintf(fp1,message);
      myfprintf(fp1,"SCALARS temperature float 1\n");
      myfprintf(fp1,"LOOKUP_TABLE default\n");
    }else{
      /* if not first, wait for previous */
      mpi_rc = MPI_Recv(&mpi_inmsg, 1, MPI_INT, (E->parallel.me-1), 7, E->parallel.world, &mpi_stat);
      /* open for append */
      fp1 = output_open(output_file,"a");
    }
      for(i=1;i<=E->lmesh.nno;i++){
	cvec[0] = E->T[CPPR][i];
	if(be_write_float_to_file(cvec,1,fp1)!=1)
	  BE_WERROR;
      }
    if(E->output.gzdir.vtk_io == 2){
      fclose(fp1);fflush(fp1);		/* close file and flush buffer */
      if(E->parallel.me <  E->parallel.nproc-1){
	mpi_rc = MPI_Send(&mpi_success_message, 1, MPI_INT, (E->parallel.me+1), 7, E->parallel.world);
      }else{
	mpi_rc = MPI_Send(&mpi_success_message, 1, MPI_INT, 0, 6, E->parallel.world); /* tell m=0 to go ahead */
      }
    }
    /*
       velocities second
    */
    if((E->output.gzdir.vtk_io == 3) || (E->parallel.me == 0)){
      if(E->output.gzdir.vtk_io == 2){
	mpi_rc = MPI_Recv(&mpi_inmsg, 1, MPI_INT, E->parallel.nproc-1 , 6, E->parallel.world, &mpi_stat);
	fp1 = output_open(output_file,"a"); /* append velocities */
      }
      sprintf(message,"VECTORS velocity float\n");myfprintf(fp1,message);
    }else{
      mpi_rc = MPI_Recv(&mpi_inmsg, 1, MPI_INT, (E->parallel.me-1), 5, E->parallel.world, &mpi_stat);
      fp1 = output_open(output_file,"a");
    }
      if(E->output.gzdir.rnr){
	/* remove NR */
	for(i=1,k=0;i<=E->lmesh.nno;i++,k += 9) {
	  vcorr[0] = E->sphere.cap[CPPR].V[1][i]; /* vtheta */
	  vcorr[1] = E->sphere.cap[CPPR].V[2][i]; /* vphi */
	  /* remove the velocity that corresponds to a net rotation of omega[0..2] at location
	     r,t,p from the t,p velocities in vcorr[0..1]
	  */
	  sub_netr(E->sx[CPPR][3][i],E->sx[CPPR][1][i],E->sx[CPPR][2][i],(vcorr+0),(vcorr+1),omega);

	  convert_pvec_to_cvec(E->sphere.cap[CPPR].V[3][i],vcorr[0],vcorr[1],
			       (E->output.gzdir.vtk_base+k),cvec);
	  if(be_write_float_to_file(cvec,3,fp1)!=3)BE_WERROR;
	}
      }else{
	/* regular output */
	for(i=1,k=0;i<=E->lmesh.nno;i++,k += 9) {
	  convert_pvec_to_cvec(E->sphere.cap[CPPR].V[3][i],E->sphere.cap[CPPR].V[1][i],E->sphere.cap[CPPR].V[2][i],
			       (E->output.gzdir.vtk_base+k),cvec);
	  if(be_write_float_to_file(cvec,3,fp1)!=3)BE_WERROR;
	}
      }
    fclose(fp1);fflush(fp1);		/* close file and flush buffer */
    if(E->output.gzdir.vtk_io == 2){
      if(E->parallel.me <  E->parallel.nproc-1){
	mpi_rc = MPI_Send(&mpi_success_message, 1, MPI_INT, (E->parallel.me+1), 5, E->parallel.world);
      }else{
	fprintf(stderr,"vtk_io: geo, temp, & vel writtend to %s\n",output_file);
      }
    }else{
      if(E->parallel.me == 0)
	fprintf(stderr,"vtk_io: geo, temp, & vel written to %s\n",output_file);
    }
    /* new VTK velo and temp done */
  }else{
    /*

    modified zipped output

    */
    /*


    temperatures are printed along with velocities for old type of
    output

    if VTK is selected, will generate a separate temperature file

    */
    if(E->output.gzdir.vtk_io == 1) {
      /*
	 for VTK, only print temperature
      */
      snprintf(output_file2,255,"%s/%d/t.%d.%d",
	       E->control.data_dir,
	       cycles,E->parallel.me,cycles);
    }else{				/* vel + T for old output */
      snprintf(output_file2,255,"%s/%d/velo.%d.%d",
	       E->control.data_dir,cycles,
	       E->parallel.me,cycles);
    }
    snprintf(output_file,255,"%s.gz",output_file2); /* add the .gz */

    gzout = gzdir_output_open(output_file,"w");
    gzprintf(gzout,"%d %d %.5e\n",
	     cycles,E->lmesh.nno,E->monitor.elapsed_time);
      gzprintf(gzout,"%3d %7d\n",CPPR,E->lmesh.nno);
      if(E->output.gzdir.vtk_io){
	/* VTK */
	for(i=1;i<=E->lmesh.nno;i++)
	  gzprintf(gzout,"%.6e\n",E->T[CPPR][i]);
      } else {
	/* old velo + T output */
	if(E->output.gzdir.rnr){
	  /* remove NR */
	  for(i=1;i<=E->lmesh.nno;i++){
	    vcorr[0] = E->sphere.cap[CPPR].V[1][i]; /* vt */
	    vcorr[1] = E->sphere.cap[CPPR].V[2][i]; /* vphi */
	    sub_netr(E->sx[CPPR][3][i],E->sx[CPPR][1][i],E->sx[CPPR][2][i],(vcorr+0),(vcorr+1),omega);
	    gzprintf(gzout,"%.6e %.6e %.6e %.6e\n",
		     vcorr[0],vcorr[1],
		     E->sphere.cap[CPPR].V[3][i],E->T[CPPR][i]);

	  }
	}else{
	  for(i=1;i<=E->lmesh.nno;i++)
	    gzprintf(gzout,"%.6e %.6e %.6e %.6e\n",
		     E->sphere.cap[CPPR].V[1][i],
		     E->sphere.cap[CPPR].V[2][i],
		     E->sphere.cap[CPPR].V[3][i],E->T[CPPR][i]);
	}
      }
    gzclose(gzout);
    if(E->output.gzdir.vtk_io){
      /*
	 write Cartesian velocities to file
      */
      snprintf(output_file,255,"%s/%d/vtk_v.%d.%d.gz",
	       E->control.data_dir,cycles,E->parallel.me,cycles);
      gzout = gzdir_output_open(output_file,"w");
	if(E->output.gzdir.rnr){
	  /* remove NR */
	  for(i=1,k=0;i<=E->lmesh.nno;i++,k += 9) {
	    vcorr[0] = E->sphere.cap[CPPR].V[1][i];
	    vcorr[1] = E->sphere.cap[CPPR].V[2][i];
	    sub_netr(E->sx[CPPR][3][i],E->sx[CPPR][1][i],E->sx[CPPR][2][i],(vcorr+0),(vcorr+1),omega);
	    convert_pvec_to_cvec(E->sphere.cap[CPPR].V[3][i],vcorr[0],vcorr[1],
				 (E->output.gzdir.vtk_base+k),cvec);
	    gzprintf(gzout,"%10.4e %10.4e %10.4e\n",cvec[0],cvec[1],cvec[2]);
	  }
	}else{
	  /* regular output */
	  for(i=1,k=0;i<=E->lmesh.nno;i++,k += 9) {
	    /* convert r,theta,phi vector to x,y,z at base location */
	    convert_pvec_to_cvec(E->sphere.cap[CPPR].V[3][i],
				 E->sphere.cap[CPPR].V[1][i],
				 E->sphere.cap[CPPR].V[2][i],
				 (E->output.gzdir.vtk_base+k),cvec);
	    /* output of cartesian vector */
	    gzprintf(gzout,"%10.4e %10.4e %10.4e\n",
		     cvec[0],cvec[1],cvec[2]);
	  }
	}
      gzclose(gzout);

     }
  } /* end gzipped and old VTK out */
  if(E->output.gzdir.vtk_io){	/* all VTK modes */
    /* free memory */
    if(!E->output.gzdir.vtk_base_save)
      free(E->output.gzdir.vtk_base);
  }
}

/*
   viscosity
*/
void gzdir_output_visc(struct All_variables *E, int cycles)
{
  int i, j;
  char output_file[255];
  gzFile *gz1;
  FILE *fp1;
  int lev = E->mesh.levmax;
  float ftmp;
  /* for dealing with several processors */
  MPI_Status mpi_stat;
  int mpi_rc;
  int mpi_inmsg, mpi_success_message = 1;


  if(E->output.gzdir.vtk_io < 2){
    snprintf(output_file,255,
	     "%s/%d/visc.%d.%d.gz", E->control.data_dir,
	     cycles,E->parallel.me, cycles);
    gz1 = gzdir_output_open(output_file,"w");
      gzprintf(gz1,"%3d %7d\n",CPPR,E->lmesh.nno);
      for(i=1;i<=E->lmesh.nno;i++)
	gzprintf(gz1,"%.4e\n",E->VI[lev][CPPR][i]);

    gzclose(gz1);
  }else{
    if(E->output.gzdir.vtk_io == 2)
      parallel_process_sync(E);
      /* new legacy VTK */
    get_vtk_filename(output_file,0,E,cycles);
    if((E->parallel.me == 0) || (E->output.gzdir.vtk_io == 3)){
      fp1 = output_open(output_file,"a");
      myfprintf(fp1,"SCALARS log10(visc) float 1\n");
      myfprintf(fp1,"LOOKUP_TABLE default\n");
    }else{
      /* if not first, wait for previous */
      mpi_rc = MPI_Recv(&mpi_inmsg, 1, MPI_INT, (E->parallel.me-1), 0, E->parallel.world, &mpi_stat);
      /* open for append */
      fp1 = output_open(output_file,"a");
    }
      for(i=1;i<=E->lmesh.nno;i++){
	ftmp = log10(E->VI[lev][CPPR][i]);
	if(fabs(ftmp) < 5e-7)ftmp = 0.0;
	if(be_write_float_to_file(&ftmp,1,fp1)!=1)BE_WERROR;
      }
    fclose(fp1);fflush(fp1);		/* close file and flush buffer */
    if(E->output.gzdir.vtk_io == 2)
      if(E->parallel.me <  E->parallel.nproc-1){
	mpi_rc = MPI_Send(&mpi_success_message, 1, MPI_INT, (E->parallel.me+1), 0, E->parallel.world);
      }
  }
}

#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC

/*
   anisotropic viscosity
*/
void gzdir_output_avisc(struct All_variables *E, int cycles)
{
  int i, j;
  char output_file[255];
  gzFile *gz1;
  FILE *fp1;
  int lev = E->mesh.levmax;
  float ftmp;
  /* for dealing with several processors */
  MPI_Status mpi_stat;
  int mpi_rc;
  int mpi_inmsg, mpi_success_message = 1;
  if(E->viscosity.allow_anisotropic_viscosity){
    
    if(E->output.gzdir.vtk_io < 2){
      snprintf(output_file,255,
	       "%s/%d/avisc.%d.%d.gz", E->control.data_dir,
	       cycles,E->parallel.me, cycles);
      gz1 = gzdir_output_open(output_file,"w");
	gzprintf(gz1,"%3d %7d\n",CPPR,E->lmesh.nno);
	for(i=1;i<=E->lmesh.nno;i++)
	  gzprintf(gz1,"%.4e %.4e %.4e %.4e\n",E->VI2[lev][CPPR][i],E->VIn1[lev][CPPR][i],E->VIn2[lev][CPPR][i],E->VIn3[lev][CPPR][i]);
      
      gzclose(gz1);
    }else{
      if(E->output.gzdir.vtk_io == 2)
	parallel_process_sync(E);
      /* new legacy VTK */
      get_vtk_filename(output_file,0,E,cycles);
      if((E->parallel.me == 0) || (E->output.gzdir.vtk_io == 3)){
	fp1 = output_open(output_file,"a");
	myfprintf(fp1,"SCALARS vis2 float 1\n");
	myfprintf(fp1,"LOOKUP_TABLE default\n");
      }else{
	/* if not first, wait for previous */
	mpi_rc = MPI_Recv(&mpi_inmsg, 1, MPI_INT, (E->parallel.me-1), 0, E->parallel.world, &mpi_stat);
	/* open for append */
	fp1 = output_open(output_file,"a");
      }
	for(i=1;i<=E->lmesh.nno;i++){
	  ftmp = E->VI2[lev][CPPR][i];
	  if(be_write_float_to_file(&ftmp,1,fp1)!=1)
	    BE_WERROR;
	}
      fclose(fp1);fflush(fp1);		/* close file and flush buffer */
      if(E->output.gzdir.vtk_io == 2)
	if(E->parallel.me <  E->parallel.nproc-1){
	  mpi_rc = MPI_Send(&mpi_success_message, 1, MPI_INT, (E->parallel.me+1), 0, E->parallel.world);
	}
    }
  }
}

#endif

void gzdir_output_surf_botm(struct All_variables *E, int cycles)
{
  int i, j, s;
  char output_file[255];
  gzFile *fp2;
  float *topo;

  if((E->output.write_q_files == 0) || (cycles == 0) ||
     (cycles % E->output.write_q_files)!=0)
      heat_flux(E);
  /* else, the heat flux will have been computed already */

  if(E->control.use_cbf_topo){
    get_CBF_topo(E,E->slice.tpg,E->slice.tpgb);
  }else{
    get_STD_topo(E,E->slice.tpg,E->slice.tpgb,E->slice.divg,E->slice.vort,cycles);
  }


  if (E->output.surf && (E->parallel.me_loc[3]==E->parallel.nprocz-1)) {
    snprintf(output_file,255,"%s/%d/surf.%d.%d.gz", E->control.data_dir,
	    cycles,E->parallel.me, cycles);
    fp2 = gzdir_output_open(output_file,"w");

        /* choose either STD topo or pseudo-free-surf topo */
        if(E->control.pseudo_free_surf)
            topo = E->slice.freesurf[CPPR];
        else
            topo = E->slice.tpg[CPPR];

        gzprintf(fp2,"%3d %7d\n",CPPR,E->lmesh.nsf);
        for(i=1;i<=E->lmesh.nsf;i++)   {
            s = i*E->lmesh.noz;
            gzprintf(fp2,"%.4e %.4e %.4e %.4e\n",
		     topo[i],E->slice.shflux[CPPR][i],E->sphere.cap[CPPR].V[1][s],E->sphere.cap[CPPR].V[2][s]);
        }
    gzclose(fp2);
  }


  if (E->output.botm && (E->parallel.me_loc[3]==0)) {
    snprintf(output_file,255,"%s/%d/botm.%d.%d.gz", E->control.data_dir,
	    cycles,E->parallel.me, cycles);
    fp2 = gzdir_output_open(output_file,"w");

      gzprintf(fp2,"%3d %7d\n",CPPR,E->lmesh.nsf);
      for(i=1;i<=E->lmesh.nsf;i++)  {
        s = (i-1)*E->lmesh.noz + 1;
        gzprintf(fp2,"%.4e %.4e %.4e %.4e\n",
		 E->slice.tpgb[CPPR][i],E->slice.bhflux[CPPR][i],E->sphere.cap[CPPR].V[1][s],E->sphere.cap[CPPR].V[2][s]);
      }
    gzclose(fp2);
  }

}


void gzdir_output_geoid(struct All_variables *E, int cycles)
{
    void compute_geoid();
    int ll, mm, p;
    char output_file[255];
    gzFile *fp1;

    compute_geoid(E);

    if (E->parallel.me == (E->parallel.nprocz-1))  {
        snprintf(output_file, 255,
		 "%s/%d/geoid.%d.%d.gz", E->control.data_dir,
		cycles,E->parallel.me, cycles);
        fp1 = gzdir_output_open(output_file,"w");

        /* write headers */
        gzprintf(fp1, "%d %d %.5e\n", cycles, E->output.llmax,
                E->monitor.elapsed_time);

        /* write sph harm coeff of geoid and topos */
        for (ll=0; ll<=E->output.llmax; ll++)
            for(mm=0; mm<=ll; mm++)  {
                p = E->sphere.hindex[ll][mm];
                gzprintf(fp1,"%d %d %.4e %.4e %.4e %.4e %.4e %.4e\n",
                        ll, mm,
                        E->sphere.harm_geoid[0][p],
                        E->sphere.harm_geoid[1][p],
                        E->sphere.harm_geoid_from_tpgt[0][p],
                        E->sphere.harm_geoid_from_tpgt[1][p],
                        E->sphere.harm_geoid_from_bncy[0][p],
                        E->sphere.harm_geoid_from_bncy[1][p]);

            }

        gzclose(fp1);
    }
}



void gzdir_output_stress(struct All_variables *E, int cycles)
{
  int m, node;
  char output_file[255];
  gzFile *fp1;
  /* for stress computation */
  void allocate_STD_mem();
  void compute_nodal_stress();
  void free_STD_mem();
  float *SXX[NCS],*SYY[NCS],*SXY[NCS],*SXZ[NCS],*SZY[NCS],*SZZ[NCS];
  float *divv[NCS],*vorv[NCS];
  /*  */
  if(E->control.use_cbf_topo)	{/* for CBF topo, stress will not have been computed */
    allocate_STD_mem(E, SXX, SYY, SZZ, SXY, SXZ, SZY, divv, vorv);
    compute_nodal_stress(E, SXX, SYY, SZZ, SXY, SXZ, SZY, divv, vorv);
    free_STD_mem(E, SXX, SYY, SZZ, SXY, SXZ, SZY, divv, vorv);
  }

  snprintf(output_file,255,"%s/%d/stress.%d.%d.gz", E->control.data_dir,
	  cycles,E->parallel.me, cycles);
  fp1 = gzdir_output_open(output_file,"w");

  gzprintf(fp1,"%d %d %.5e\n",cycles,E->lmesh.nno,E->monitor.elapsed_time);

    gzprintf(fp1,"%3d %7d\n",CPPR,E->lmesh.nno);
    for (node=1;node<=E->lmesh.nno;node++)
      gzprintf(fp1, "%.4e %.4e %.4e %.4e %.4e %.4e\n",
              E->gstress[CPPR][(node-1)*6+1], /*  stt */
              E->gstress[CPPR][(node-1)*6+2], /*  spp */
              E->gstress[CPPR][(node-1)*6+3], /*  srr */
              E->gstress[CPPR][(node-1)*6+4], /*  stp */
              E->gstress[CPPR][(node-1)*6+5], /*  str */
              E->gstress[CPPR][(node-1)*6+6]); /* srp */
  gzclose(fp1);
}


void gzdir_output_horiz_avg(struct All_variables *E, int cycles)
{
  /* horizontal average output of temperature, composition and rms velocity*/
  void compute_horiz_avg();

  int j;
  char output_file[255];
  gzFile *fp1;

  /* compute horizontal average here.... */
  compute_horiz_avg(E);

  /* only the first nprocz processors need to output */

  if (E->parallel.me<E->parallel.nprocz)  {
    snprintf(output_file,255,"%s/%d/horiz_avg.%d.%d.gz", E->control.data_dir,
	    cycles,E->parallel.me, cycles);
    fp1=gzdir_output_open(output_file,"w");
    for(j=1;j<=E->lmesh.noz;j++)  { /* format: r <T> <vh> <vr> (<C>) */
        gzprintf(fp1,"%.4e %.4e %.4e %.4e",E->sx[1][3][j],E->Have.T[j],E->Have.V[1][j],E->Have.V[2][j]);

        if (E->composition.on) {
            int n;
            for(n=0; n<E->composition.ncomp; n++)
                gzprintf(fp1," %.4e", E->Have.C[n][j]);
        }
        gzprintf(fp1,"\n");
    }
    gzclose(fp1);
  }

}


/* only called once */
void gzdir_output_mat(struct All_variables *E)
{
  int m, el;
  char output_file[255];
  gzFile* fp;

  snprintf(output_file,255,"%s/mat.%d.gz", E->control.data_dir,E->parallel.me);
  fp = gzdir_output_open(output_file,"w");

  for(el=1;el<=E->lmesh.nel;el++)
    gzprintf(fp,"%d %d %f\n", el,E->mat[CPPR][el],E->VIP[CPPR][el]);

  gzclose(fp);

}



void gzdir_output_pressure(struct All_variables *E, int cycles)
{
  int i, j;
  float ftmp;
  char output_file[255];
  gzFile *gz1;
  FILE *fp1;
  /* for dealing with several processors */
  MPI_Status mpi_stat;
  int mpi_rc;
  int mpi_inmsg, mpi_success_message = 1;

  if(E->output.gzdir.vtk_io < 2){ /* old */
    snprintf(output_file,255,"%s/%d/pressure.%d.%d.gz", E->control.data_dir,cycles,
	     E->parallel.me, cycles);
    gz1 = gzdir_output_open(output_file,"w");
    gzprintf(gz1,"%d %d %.5e\n",cycles,E->lmesh.nno,E->monitor.elapsed_time);
      gzprintf(gz1,"%3d %7d\n",CPPR,E->lmesh.nno);
      for(i=1;i<=E->lmesh.nno;i++)
	gzprintf(gz1,"%.6e\n",E->NP[CPPR][i]);
    gzclose(gz1);
  }else{/* new legacy VTK */
    if(E->output.gzdir.vtk_io == 2)
      parallel_process_sync(E);
    get_vtk_filename(output_file,0,E,cycles);
    if((E->parallel.me == 0) || (E->output.gzdir.vtk_io == 3)){
      fp1 = output_open(output_file,"a");
      myfprintf(fp1,"SCALARS pressure float 1\n");
      myfprintf(fp1,"LOOKUP_TABLE default\n");
    }else{
      mpi_rc = MPI_Recv(&mpi_inmsg, 1, MPI_INT, (E->parallel.me-1), 0, E->parallel.world, &mpi_stat);
      fp1 = output_open(output_file,"a");
    }
      for(i=1;i<=E->lmesh.nno;i++){
	ftmp = E->NP[CPPR][i];
	if(be_write_float_to_file(&ftmp,1,fp1)!=1)BE_WERROR;
      }
    fclose(fp1);fflush(fp1);		/* close file and flush buffer */
    if(E->output.gzdir.vtk_io == 2)
      if(E->parallel.me <  E->parallel.nproc-1){
	mpi_rc = MPI_Send(&mpi_success_message, 1, MPI_INT, (E->parallel.me+1), 0, E->parallel.world);
      }
  }
}



void gzdir_output_tracer(struct All_variables *E, int cycles)
{
  int i, j, n, ncolumns;
  char output_file[255];
  gzFile *fp1;

  snprintf(output_file,255,"%s/%d/tracer.%d.%d.gz",
	   E->control.data_dir,cycles,
	   E->parallel.me, cycles);
  fp1 = gzdir_output_open(output_file,"w");

  ncolumns = 3 + E->trace.number_of_extra_quantities;

  gzprintf(fp1,"%d %d %d %.5e\n", cycles, E->trace.ntracers[CPPR], ncolumns, E->monitor.elapsed_time);

      for(n=1;n<=E->trace.ntracers[CPPR];n++) {
          /* write basic quantities (coordinate) */
          gzprintf(fp1,"%9.5e %9.5e %9.5e",
                  E->trace.basicq[CPPR][0][n],
                  E->trace.basicq[CPPR][1][n],
                  E->trace.basicq[CPPR][2][n]);

          /* write extra quantities */
          for (i=0; i<E->trace.number_of_extra_quantities; i++) {
              gzprintf(fp1," %9.5e", E->trace.extraq[CPPR][i][n]);
          }
          gzprintf(fp1, "\n");
      }

  gzclose(fp1);
}


void gzdir_output_comp_nd(struct All_variables *E, int cycles)
{
  int i, j, k;
  char output_file[255],message[255];
  gzFile *gz1;
  FILE *fp1;
  float ftmp;
  /* for dealing with several processors */
  MPI_Status mpi_stat;
  int mpi_rc;
  int mpi_inmsg, mpi_success_message = 1;

  if(E->output.gzdir.vtk_io < 2){
    snprintf(output_file,255,"%s/%d/comp_nd.%d.%d.gz",
	     E->control.data_dir,cycles,
	     E->parallel.me, cycles);
    gz1 = gzdir_output_open(output_file,"w");
      gzprintf(gz1,"%3d %7d %.5e %.5e %.5e\n",
	       CPPR, E->lmesh.nel,
	       E->monitor.elapsed_time,
	       E->composition.initial_bulk_composition,
	       E->composition.bulk_composition);
      for(i=1;i<=E->lmesh.nno;i++) {
	for(k=0;k < E->composition.ncomp;k++)
	  gzprintf(gz1,"%.6e ",E->composition.comp_node[CPPR][k][i]);
	gzprintf(gz1,"\n");
      }
    gzclose(gz1);
  }else{/* new legacy VTK */
    if(E->output.gzdir.vtk_io == 2)
      parallel_process_sync(E);
    get_vtk_filename(output_file,0,E,cycles);
    if((E->output.gzdir.vtk_io==3) || (E->parallel.me == 0)){
      fp1 = output_open(output_file,"a");
      if(E->composition.ncomp > 4)
	myerror(E,"vtk out error: ncomp out of bounds (needs to be < 4)");
      sprintf(message,"SCALARS composition float %d\n",E->composition.ncomp);
      myfprintf(fp1,message);
      myfprintf(fp1,"LOOKUP_TABLE default\n");
    }else{			/* serial wait */
      mpi_rc = MPI_Recv(&mpi_inmsg, 1, MPI_INT, (E->parallel.me-1), 0, E->parallel.world, &mpi_stat);
      fp1 = output_open(output_file,"a");
    }
      for(i=1;i<=E->lmesh.nno;i++){
	for(k=0;k<E->composition.ncomp;k++){
	  ftmp = E->composition.comp_node[CPPR][k][i];
	  if(be_write_float_to_file(&ftmp,1,fp1)!=1)BE_WERROR;
	}
      }
    fclose(fp1);fflush(fp1);		/* close file and flush buffer */
    if(E->output.gzdir.vtk_io == 2) /* serial */
      if(E->parallel.me <  E->parallel.nproc-1){
	mpi_rc = MPI_Send(&mpi_success_message, 1, MPI_INT, (E->parallel.me+1), 0, E->parallel.world);
      }
  }
}


void gzdir_output_comp_el(struct All_variables *E, int cycles)
{
    int i, j, k;
    char output_file[255];
    gzFile *fp1;

    snprintf(output_file,255,"%s/%d/comp_el.%d.%d.gz", E->control.data_dir,
	    cycles,E->parallel.me, cycles);
    fp1 = gzdir_output_open(output_file,"w");

        gzprintf(fp1,"%3d %7d %.5e %.5e %.5e\n",
                CPPR, E->lmesh.nel,
                E->monitor.elapsed_time,
                E->composition.initial_bulk_composition,
                E->composition.bulk_composition);

        for(i=1;i<=E->lmesh.nel;i++) {
	  for(k=0;k<E->composition.ncomp;k++)
            gzprintf(fp1,"%.6e ",E->composition.comp_el[CPPR][k][i]);
	  gzprintf(fp1,"\n");
        }

    gzclose(fp1);
}


void gzdir_output_heating(struct All_variables *E, int cycles)
{
    int j, e;
    char output_file[255];
    gzFile *fp1;

    snprintf(output_file,255,"%s/%d/heating.%d.%d.gz", E->control.data_dir,
	    cycles,E->parallel.me, cycles);
    fp1 = gzdir_output_open(output_file,"w");

    gzprintf(fp1,"%.5e\n",E->monitor.elapsed_time);

    gzprintf(fp1,"%3d %7d\n", CPPR, E->lmesh.nel);
    for(e=1; e<=E->lmesh.nel; e++)
        gzprintf(fp1, "%.4e %.4e %.4e\n", E->heating_adi[CPPR][e],
                  E->heating_visc[CPPR][e], E->heating_latent[CPPR][e]);
    gzclose(fp1);
}


/*

restart facility for zipped/VTK style , will init temperature

*/
void restart_tic_from_gzdir_file(struct All_variables *E)
{
  int ii, ll, mm,rezip;
  float restart_elapsed_time;
  int i, m;
  char output_file[255], input_s[1000];
  FILE *fp;

  float v1, v2, v3, g;

  ii = E->monitor.solution_cycles_init;
  switch(E->output.gzdir.vtk_io){
  case 2:
  case 3:
    myerror(E,"sorry, restart with vtk_io 2 or 3 not implemented yet");
    break;
  case 1:
    /* VTK I/O */
    snprintf(output_file,255,"%s/%d/t.%d.%d",
	     E->control.data_dir_old,
	     ii,E->parallel.me,ii);
    break;
  default:
    snprintf(output_file,255,"%s/%d/velo.%d.%d",
	     E->control.data_dir_old,ii,
	     E->parallel.me,ii);
    break;
  }
  /* open file */
  rezip = open_file_zipped(output_file,&fp,E);
  if (E->parallel.me==0){
    fprintf(stderr,"restart_tic_from_gzdir_file: using  %s for restarted temperature\n",
	    output_file);
    fprintf(E->fp,"restart_tic_from_gzdir_file: using  %s for restarted temperature\n",
	    output_file);
  }
  if(fscanf(fp,"%i %i %f",&ll,&mm,&restart_elapsed_time) != 3)
    myerror(E,"restart vtkl read error 0");
  if(mm != E->lmesh.nno){
    fprintf(stderr,"%i %i\n",mm, E->lmesh.nno);
    myerror(E,"lmesh.nno mismatch in restart files");
  }
  
  switch(E->output.gzdir.vtk_io) {
  case 1: /* VTK */
      if(fscanf(fp,"%i %i",&ll,&mm) != 2)
	myerror(E,"restart vtkl read error 1");
      for(i=1;i<=E->lmesh.nno;i++){
	if(fscanf(fp,"%f",&g) != 1)
	  myerror(E,"restart vtkl read error 2");
	if(!finite(g)){
	  fprintf(stderr,"WARNING: found a NaN in input temperatures\n");
	  g=0.0;
	}
	E->T[CPPR][i] = g;
      }
    break;
  default:			/* old style velo */
      fscanf(fp,"%i %i",&ll,&mm);
      for(i=1;i<=E->lmesh.nno;i++)  {
        if(fscanf(fp,"%f %f %f %f",&v1,&v2,&v3,&g) != 4)
	  myerror(E,"restart velo read error 1");
	/*  E->sphere.cap[m].V[1][i] = v1;
	    E->sphere.cap[m].V[1][i] = v2;
	    E->sphere.cap[m].V[1][i] = v3;  */
	/* I don't like that  */
	//E->T[m][i] = max(0.0,min(g,1.0));
	E->T[CPPR][i] = g;
      }
    break;
  }
  fclose (fp);
  if(rezip)			/* rezip */
    gzip_file(output_file);

  temperatures_conform_bcs(E);
  
}


/*

tries to open 'name'. if name exists, out will be pointer to file and
return 0. if name doesn't exist, will check for name.gz.  if this
exists, will unzip and open, and return 1

the idea is to preserve the initial file state

*/
int open_file_zipped(char *name, FILE **in,
		     struct All_variables *E)
{
  char mstring[1000];
  *in = fopen(name,"r");
  if (*in == NULL) {
    /*
       unzipped file not found
    */
    snprintf(mstring,1000,"%s.gz",name);
    *in= fopen(mstring,"r");
    if(*in != NULL){
      /*
	 zipped version was found
      */
      fclose(*in);
      snprintf(mstring,1000,"gunzip -f %s.gz",name); /* brutal */
      system(mstring);	/* unzip */
      /* open unzipped file for read */
      *in = fopen(name,"r");
      if(*in == NULL)
          myerror(E,"open_file_zipped: unzipping error");
      return 1;
    }else{
      /*
	 no file, either zipped or unzipped
      */
      snprintf(mstring,1000,"no files %s and %s.gz were found, exiting",
	       name,name);
      myerror(E,mstring);
      return 0;
    }
  }else{
    /*
       file was found unzipped
    */
    return 0;
  }
}

/* compress a file using the sytem command  */
void gzip_file(char *output_file)
{
  char command_string[300];
  snprintf(command_string,300,"gzip -f %s",output_file); /* brutal */
  system(command_string);
}




void get_vtk_filename(char *output_file,
		      int geo,struct All_variables *E,
		      int cycles)
{
  if(E->output.gzdir.vtk_io == 2){ /* serial */
    if(geo)			/* geometry */
      sprintf(output_file,"%s/vtk_geo",
	       E->control.data_dir);
    else			/* data part */
      sprintf(output_file,"%s/d.%08i.vtk",
	       E->control.data_dir, E->output.gzdir.vtk_ocount);
  }else{			/* parallel */
    if(geo)			/* geometry */
      sprintf(output_file,"%s/vtk_geo.%i",
	       E->control.data_dir,E->parallel.me);
    else			/* data part */
      sprintf(output_file,"%s/%d/d.%08i.%i.vtk",
	       E->control.data_dir,cycles,
	       E->output.gzdir.vtk_ocount,
	       E->parallel.me);
  }
}




/*


big endian I/O (needed for vtk)


*/

/*

write the x[n] array to file, making sure it is written big endian

*/
int be_write_float_to_file(float *x, int n, FILE *out)
{
  int i,nout;
  static size_t len = sizeof(float);
  size_t bsize;
  float ftmp;
#ifdef ASCII_DEBUG
  for(i=0;i<n;i++)
    fprintf(out,"%11g ",x[i]);
  fprintf(out,"\n");
  nout = n;
#else
  /*
     do we need to flip?
  */
  if(be_is_little_endian()){
    nout = 0;
    for(i=0;i < n;i++){
      ftmp = x[i];
      be_flip_byte_order((void *)(&ftmp),len);
      nout += fwrite(&ftmp,len,(size_t)1,out);		/* write to file */
    }
  }else{			/* operate on x */
    nout = fwrite(x,len,(size_t)n,out);		/* write to file */
  }
#endif
  return nout;
}
int be_write_int_to_file(int *x, int n, FILE *out)
{
  int i,nout;
  static size_t len = sizeof(int);
  size_t bsize;
  int itmp;
#ifdef ASCII_DEBUG
  for(i=0;i<n;i++)
    fprintf(out,"%11i ",x[i]);
  fprintf(out,"\n");
  nout = n;
#else
  /*
     do we need to flip?
  */
  if(be_is_little_endian()){
    nout = 0;
    for(i=0;i < n;i++){
      itmp = x[i];
      be_flip_byte_order((void *)(&itmp),len);
      nout += fwrite(&itmp,len,(size_t)1,out);		/* write to file */
    }
  }else{			/* operate on x */
    nout = fwrite(x,len,(size_t)n,out);		/* write to file */
  }
#endif
  return nout;
}


/* does this make a difference? nope, didn't, and why would it */
void myfprintf(FILE *out,char *string)
{
#ifdef ASCII_DEBUG
  fprintf(out,string);
#else
  fwrite(string, sizeof(char), strlen(string), out);
#endif
}

int be_is_little_endian(void)
{
  static const unsigned long a = 1;
  return *(const unsigned char *)&a;
}

/*


flip endian-ness


*/
/*

flip endianness of x

*/
void be_flip_byte_order(void *x, size_t len)
{
  void *copy;
  int i;
  copy = (void *)malloc(len);	/* don't check here for speed */
  memcpy(copy,x,len);
  be_flipit(x,copy,len);
  free(copy);
}

/* this should not be called with (i,i,size i) */
void be_flipit(void *d, void *s, size_t len)
{
  unsigned char *dest = d;
  unsigned char *src  = s;
  src += len - 1;
  for (; len; len--)
    *dest++ = *src--;
}


#undef BE_WERROR
#endif /* gzdir switch */
