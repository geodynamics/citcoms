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

this version uses gzipped, ascii output to subdirectories

TWB

*/

#ifdef USE_GZDIR

#include <zlib.h>


#include <stdlib.h>
#include <math.h>
#include "element_definitions.h"
#include "global_defs.h"
#include "parsing.h"
#include "output.h"

gzFile *gzdir_output_open(char *,char *);
void gzdir_output(struct All_variables *, int );
void gzdir_output_comp_nd(struct All_variables *, int);
void gzdir_output_comp_el(struct All_variables *, int);
void gzdir_output_coord(struct All_variables *);
void gzdir_output_mat(struct All_variables *);
void gzdir_output_velo(struct All_variables *, int);
void gzdir_output_visc_prepare(struct All_variables *, float **);
void gzdir_output_visc(struct All_variables *, int);
void gzdir_output_surf_botm(struct All_variables *, int);
void gzdir_output_geoid(struct All_variables *, int);
void gzdir_output_stress(struct All_variables *, int);
void gzdir_output_horiz_avg(struct All_variables *, int);
void gzdir_output_tracer(struct All_variables *, int);
void gzdir_output_pressure(struct All_variables *, int);


void calc_cbase_at_tp(float , float , float *);
void rtp2xyz(float , float , float, float *);
void convert_pvec_to_cvec(float ,float , float , float *,float *);
void *safe_malloc (size_t );



extern void parallel_process_termination();
extern void heat_flux(struct All_variables *);
extern void get_STD_topo(struct All_variables *, float**, float**,
                         float**, float**, int);

/**********************************************************************/


void gzdir_output(struct All_variables *E, int cycles)
{
  char output_dir[255];
  
  if (cycles == 0) {
    /* initial I/O */
    
    gzdir_output_coord(E);
    /*gzdir_output_mat(E);*/
  }
  /* 
     make a new directory for all the other output 
     
     I thought I could just have the first proc do this and sync, but
     the syncing didn't work ?!

  */
  //if(E->parallel.me == 0){
    /* make a directory */
    snprintf(output_dir,255,"%s/%d",
	    E->control.data_dir,cycles);
    mkdatadir(output_dir);
    //  }
  /* and wait for the other jobs */
    //  parallel_process_sync();

  /* output */

  gzdir_output_velo(E, cycles);
  gzdir_output_visc(E, cycles);

  gzdir_output_surf_botm(E, cycles);

  /* optiotnal output below */
  /* compute and output geoid (in spherical harmonics coeff) */
  if (E->output.geoid == 1)
      gzdir_output_geoid(E, cycles);

  if (E->output.stress == 1)
    gzdir_output_stress(E, cycles);

  if (E->output.pressure == 1)
    gzdir_output_pressure(E, cycles);

  if (E->output.horiz_avg == 1)
      gzdir_output_horiz_avg(E, cycles);

  if(E->output.tracer == 1 && E->control.tracer == 1)
      gzdir_output_tracer(E, cycles);

  if (E->output.comp_nd == 1 && E->composition.on)
      gzdir_output_comp_nd(E, cycles);

  if (E->output.comp_el == 1 && E->composition.on)
          gzdir_output_comp_el(E, cycles);

  return;
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


void gzdir_output_coord(struct All_variables *E)
{
  int i, j, offset;
  char output_file[255];
  float locx[3];
  gzFile *fp1;
  /* 
     don't use data file name
  */
  snprintf(output_file,255,"%s/coord.%d.gz",
	  E->control.data_dir,E->parallel.me);
  fp1 = gzdir_output_open(output_file,"w");

  /* nodal coordinates */
  for(j=1;j<=E->sphere.caps_per_proc;j++)     {
    gzprintf(fp1,"%3d %7d\n",j,E->lmesh.nno);
    for(i=1;i<=E->lmesh.nno;i++)
      gzprintf(fp1,"%.6e %.6e %.6e\n",
	      E->sx[j][1][i],E->sx[j][2][i],E->sx[j][3][i]);
  }

  gzclose(fp1);


  if(E->output.gzdir_vtkio){
    /* 
       
    output of Cartesian coordinates and element connectivitiy for
    vtk visualization
    
    */
    /* 
       nodal coordinates in Cartesian
    */
    snprintf(output_file,255,"%s/vtk_ecor.%d.gz",
	    E->control.data_dir,E->parallel.me);
    fp1 = gzdir_output_open(output_file,"w");
    for(j=1;j <= E->sphere.caps_per_proc;j++)     {
      for(i=1;i <= E->lmesh.nno;i++) {
	rtp2xyz(E->sx[j][3][i],E->sx[j][1][i],E->sx[j][2][i],locx);
	gzprintf(fp1,"%9.6f %9.6f %9.6f\n",
		 locx[0],locx[1],locx[2]);
      }
    }
    gzclose(fp1);
    /* 
       connectivity for all elements
    */
    offset = E->lmesh.nno * E->parallel.me - 1;
    snprintf(output_file,255,"%s/vtk_econ.%d.gz",
	    E->control.data_dir,E->parallel.me);
    fp1 = gzdir_output_open(output_file,"w");
    for(j=1;j <= E->sphere.caps_per_proc;j++)     {
      for(i=1;i <= E->lmesh.nel;i++) {
	gzprintf(fp1,"%2i\t",enodes[E->mesh.nsd]);
	if(enodes[E->mesh.nsd] != 8){
	  gzprintf(stderr,"gzdir: Output: error, only eight node hexes supported");
	  parallel_process_termination();
	}
	/* 
	   need to add offset according to the processor for global
	   node numbers
	*/
	gzprintf(fp1,"%6i %6i %6i %6i %6i %6i %6i %6i\n",
		 E->ien[j][i].node[1]+offset,E->ien[j][i].node[2]+offset,
		 E->ien[j][i].node[3]+offset,E->ien[j][i].node[4]+offset,
		 E->ien[j][i].node[5]+offset,E->ien[j][i].node[6]+offset,
		 E->ien[j][i].node[7]+offset,E->ien[j][i].node[8]+offset);
      }
    }
    gzclose(fp1);
  } /* end vtkio */



  return;
}


void gzdir_output_visc(struct All_variables *E, int cycles)
{
  int i, j;
  char output_file[255];
  gzFile *fp1;
  int lev = E->mesh.levmax;

  snprintf(output_file,255,
	   "%s/%d/visc.%d.%d.gz", E->control.data_dir,
	  cycles,E->parallel.me, cycles);
  fp1 = gzdir_output_open(output_file,"w");
  for(j=1;j<=E->sphere.caps_per_proc;j++) {
    gzprintf(fp1,"%3d %7d\n",j,E->lmesh.nno);
    for(i=1;i<=E->lmesh.nno;i++)
      gzprintf(fp1,"%.4e\n",E->VI[lev][j][i]);
  }
  
  gzclose(fp1);

  return;
}


void gzdir_output_velo(struct All_variables *E, int cycles)
{
  int i, j, k,os;
  char output_file[255],output_file2[255];
  float cvec[3];
  gzFile *gzout;
  /*
    

  temperatures are printed along with velocities for old type of
  output

  if VTK is selected, will generate a separate temperature file
  
  */
  if(E->output.gzdir_vtkio) {
    /* 
       for VTK, only print temperature 
    */
    snprintf(output_file2,255,"%s/%d/t.%d.%d",
	    E->control.data_dir,
	    cycles,E->parallel.me,cycles);
  }else{				/* vel + T */
    snprintf(output_file2,255,"%s/%d/velo.%d.%d",
	    E->control.data_dir,cycles,
	    E->parallel.me,cycles);
  }
  snprintf(output_file,255,"%s.gz",output_file2); /* add the .gz */

  gzout = gzdir_output_open(output_file,"w");
  gzprintf(gzout,"%d %d %.5e\n",
	   cycles,E->lmesh.nno,E->monitor.elapsed_time);
  for(j=1; j<= E->sphere.caps_per_proc;j++)     {
    gzprintf(gzout,"%3d %7d\n",j,E->lmesh.nno);
    if(E->output.gzdir_vtkio){
      /* VTK */
      for(i=1;i<=E->lmesh.nno;i++)           
	gzprintf(gzout,"%.6e\n",E->T[j][i]); 
    } else {			
      /* old */
      for(i=1;i<=E->lmesh.nno;i++)           
	gzprintf(gzout,"%.6e %.6e %.6e %.6e\n",
		 E->sphere.cap[j].V[1][i],
		 E->sphere.cap[j].V[2][i],
		 E->sphere.cap[j].V[3][i],E->T[j][i]); 
    }
  }
  gzclose(gzout);
  if(E->output.gzdir_vtkio){
    /* 
       cartesian velocity output  
    */
    os = E->lmesh.nno*9;
    /* 
       get base vectors if first pass or if we're not saving
    */
    if((!E->output.gzdir_vtkbase_init) ||
       (!E->output.gzdir_vtkbase_save)){
      if(!E->output.gzdir_vtkbase_init){
	/* allocate */
	E->output.gzdir_vtkbase = (float *)
	  safe_malloc(sizeof(float)*os*E->sphere.caps_per_proc);
      }
      for(k=0,j=1;j <= E->sphere.caps_per_proc;j++,k += os)     {
	for(i=1;i <= E->lmesh.nno;i++,k += 9){
	  /* cartesian basis vectors at theta, phi */
	  calc_cbase_at_tp(E->sx[j][1][i],
			   E->sx[j][2][i],
			   (E->output.gzdir_vtkbase+k));
	}
      }
      E->output.gzdir_vtkbase_init = 1;
    }
    /* 
       write Cartesian velocities to file 
    */
    snprintf(output_file,255,"%s/%d/vtk_v.%d.%d.gz",
	    E->control.data_dir,cycles,E->parallel.me,cycles);
    gzout = gzdir_output_open(output_file,"w");
    for(k=0,j=1;j <= E->sphere.caps_per_proc;j++,k += os)     {
      for(i=1;i<=E->lmesh.nno;i++,k += 9) {
	/* convert r,theta,phi vector to x,y,z at base location */
	convert_pvec_to_cvec(E->sphere.cap[j].V[3][i],
			     E->sphere.cap[j].V[1][i],
			     E->sphere.cap[j].V[2][i],
			     (E->output.gzdir_vtkbase+k),cvec);
	/* output of cartesian vector */
	gzprintf(gzout,"%10.4e %10.4e %10.4e\n",
		 cvec[0],cvec[1],cvec[2]);
      }
    }
    gzclose(gzout);
    
    /* free memory */
    if(!E->output.gzdir_vtkbase_save)
      free(E->output.gzdir_vtkbase);
  }

  return;
}


void gzdir_output_surf_botm(struct All_variables *E, int cycles)
{
  int i, j, s;
  char output_file[255];
  gzFile *fp2;
  float *topo;

  heat_flux(E);
  get_STD_topo(E,E->slice.tpg,E->slice.tpgb,E->slice.divg,E->slice.vort,cycles);

  if (E->output.surf && (E->parallel.me_loc[3]==E->parallel.nprocz-1)) {
    snprintf(output_file,255,"%s/%d/surf.%d.%d.gz", E->control.data_dir,
	    cycles,E->parallel.me, cycles);
    fp2 = gzdir_output_open(output_file,"w");

    for(j=1;j<=E->sphere.caps_per_proc;j++)  {
        /* choose either STD topo or pseudo-free-surf topo */
        if(E->control.pseudo_free_surf)
            topo = E->slice.freesurf[j];
        else
            topo = E->slice.tpg[j];

        gzprintf(fp2,"%3d %7d\n",j,E->lmesh.nsf);
        for(i=1;i<=E->lmesh.nsf;i++)   {
            s = i*E->lmesh.noz;
            gzprintf(fp2,"%.4e %.4e %.4e %.4e\n",
		     topo[i],E->slice.shflux[j][i],E->sphere.cap[j].V[1][s],E->sphere.cap[j].V[2][s]);
        }
    }
    gzclose(fp2);
  }


  if (E->output.botm && (E->parallel.me_loc[3]==0)) {
    snprintf(output_file,255,"%s/%d/botm.%d.%d.gz", E->control.data_dir,
	    cycles,E->parallel.me, cycles);
    fp2 = gzdir_output_open(output_file,"w");

    for(j=1;j<=E->sphere.caps_per_proc;j++)  {
      gzprintf(fp2,"%3d %7d\n",j,E->lmesh.nsf);
      for(i=1;i<=E->lmesh.nsf;i++)  {
        s = (i-1)*E->lmesh.noz + 1;
        gzprintf(fp2,"%.4e %.4e %.4e %.4e\n",
		 E->slice.tpgb[j][i],E->slice.bhflux[j][i],E->sphere.cap[j].V[1][s],E->sphere.cap[j].V[2][s]);
      }
    }
    gzclose(fp2);
  }

  return;
}


void gzdir_output_geoid(struct All_variables *E, int cycles)
{
    void compute_geoid();
    int ll, mm, p;
    char output_file[255];
    gzFile *fp1;

    compute_geoid(E, E->sphere.harm_geoid,
                  E->sphere.harm_geoid_from_bncy,
                  E->sphere.harm_geoid_from_tpgt,
                  E->sphere.harm_geoid_from_tpgb);

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

  snprintf(output_file,255,"%s/%d/stress.%d.%d.gz", E->control.data_dir,
	  cycles,E->parallel.me, cycles);
  fp1 = gzdir_output_open(output_file,"w");

  gzprintf(fp1,"%d %d %.5e\n",cycles,E->lmesh.nno,E->monitor.elapsed_time);

  for(m=1;m<=E->sphere.caps_per_proc;m++) {
    gzprintf(fp1,"%3d %7d\n",m,E->lmesh.nno);
    for (node=1;node<=E->lmesh.nno;node++)
      gzprintf(fp1, "%.4e %.4e %.4e %.4e %.4e %.4e\n",
              E->gstress[m][(node-1)*6+1],
              E->gstress[m][(node-1)*6+2],
              E->gstress[m][(node-1)*6+3],
              E->gstress[m][(node-1)*6+4],
              E->gstress[m][(node-1)*6+5],
              E->gstress[m][(node-1)*6+6]);
  }
  gzclose(fp1);
}


void gzdir_output_horiz_avg(struct All_variables *E, int cycles)
{
  /* horizontal average output of temperature and rms velocity*/
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
    for(j=1;j<=E->lmesh.noz;j++)  {
        gzprintf(fp1,"%.4e %.4e %.4e %.4e\n",E->sx[1][3][j],E->Have.T[j],E->Have.V[1][j],E->Have.V[2][j]);
    }
    gzclose(fp1);
  }

  return;
}


/* only called once */
void gzdir_output_mat(struct All_variables *E)
{
  int m, el;
  char output_file[255];
  gzFile* fp;

  snprintf(output_file,255,"%s/mat.%d.gz", E->control.data_dir,E->parallel.me);
  fp = gzdir_output_open(output_file,"w");

  for (m=1;m<=E->sphere.caps_per_proc;m++)
    for(el=1;el<=E->lmesh.nel;el++)
      gzprintf(fp,"%d %d %f\n", el,E->mat[m][el],E->VIP[m][el]);

  gzclose(fp);

  return;
}



void gzdir_output_pressure(struct All_variables *E, int cycles)
{
  int i, j;
  char output_file[255];
  gzFile *fp1;

  snprintf(output_file,255,"%s/%d/pressure.%d.%d.gz", E->control.data_dir,cycles,
          E->parallel.me, cycles);
  fp1 = gzdir_output_open(output_file,"w");

  gzprintf(fp1,"%d %d %.5e\n",cycles,E->lmesh.nno,E->monitor.elapsed_time);

  for(j=1;j<=E->sphere.caps_per_proc;j++) {
    gzprintf(fp1,"%3d %7d\n",j,E->lmesh.nno);
    for(i=1;i<=E->lmesh.nno;i++)
      gzprintf(fp1,"%.6e\n",E->NP[j][i]);
  }

  gzclose(fp1);

  return;
}



void gzdir_output_tracer(struct All_variables *E, int cycles)
{
  int i, j, n, ncolumns;
  char output_file[255];
  gzFile *fp1;

  snprintf(output_file,255,"%s/%d/tracer.%d.%d.gz", E->control.data_dir,
	  cycles,
          E->parallel.me, cycles);
  fp1 = gzdir_output_open(output_file,"w");

  ncolumns = 3 + E->trace.number_of_extra_quantities;

  for(j=1;j<=E->sphere.caps_per_proc;j++) {
      gzprintf(fp1,"%d %d %d %.5e\n", cycles, E->trace.ntracers[j],
              ncolumns, E->monitor.elapsed_time);

      for(n=1;n<=E->trace.ntracers[j];n++) {
          /* write basic quantities (coordinate) */
          gzprintf(fp1,"%9.5e %9.5e %9.5e",
                  E->trace.basicq[j][0][n],
                  E->trace.basicq[j][1][n],
                  E->trace.basicq[j][2][n]);

          /* write extra quantities */
          for (i=0; i<E->trace.number_of_extra_quantities; i++) {
              gzprintf(fp1," %9.5e", E->trace.extraq[j][i][n]);
          }
          gzprintf(fp1, "\n");
      }

  }

  gzclose(fp1);
  return;
}


void gzdir_output_comp_nd(struct All_variables *E, int cycles)
{
    int i, j;
    char output_file[255];
    gzFile *fp1;

    snprintf(output_file,255,"%s/%d/comp_nd.%d.%d.gz", 
	    E->control.data_dir,
	    cycles,
            E->parallel.me, cycles);
    fp1 = gzdir_output_open(output_file,"w");

    for(j=1;j<=E->sphere.caps_per_proc;j++) {
        gzprintf(fp1,"%3d %7d %.5e %.5e %.5e\n",
                j, E->lmesh.nel,
                E->monitor.elapsed_time,
                E->composition.initial_bulk_composition,
                E->composition.bulk_composition);

        for(i=1;i<=E->lmesh.nno;i++) {
            gzprintf(fp1,"%.6e\n",E->composition.comp_node[j][i]);
        }

    }

    gzclose(fp1);
    return;
}


void gzdir_output_comp_el(struct All_variables *E, int cycles)
{
    int i, j;
    char output_file[255];
    gzFile *fp1;

    snprintf(output_file,255,"%s/%d/comp_el.%d.%d.gz", E->control.data_dir,
	    cycles,E->parallel.me, cycles);
    fp1 = gzdir_output_open(output_file,"w");

    for(j=1;j<=E->sphere.caps_per_proc;j++) {
        gzprintf(fp1,"%3d %7d %.5e %.5e %.5e\n",
                j, E->lmesh.nel,
                E->monitor.elapsed_time,
                E->composition.initial_bulk_composition,
                E->composition.bulk_composition);

        for(i=1;i<=E->lmesh.nel;i++) {
            gzprintf(fp1,"%.6e\n",E->composition.comp_el[j][i]);
        }
    }

    gzclose(fp1);
    return;
}




#endif /* gzdir switch */
