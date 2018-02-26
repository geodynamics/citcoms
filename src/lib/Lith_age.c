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

#include <math.h>

#include "global_defs.h"

/*#include "age_related.h"*/
#include "parallel_related.h"
#include "parsing.h"
#include "lith_age.h"

float find_age_in_MY();
void lith_age_update_tbc(struct All_variables *E);


void lith_age_input(struct All_variables *E)
{
  int m = E->parallel.me;

  E->control.lith_age = 0;
  E->control.lith_age_time = 0;
  E->control.temperature_bound_adj = 0;

  E->control.slab_assim = 0; // DJB SLAB

  input_int("lith_age",&(E->control.lith_age),"0",m);
  input_boolean("slab_assim",&(E->control.slab_assim),"0",m); // DJB SLAB

#ifdef USE_GGRD
  input_int("ggrd_age_control",&(E->control.ggrd.age_control),"0",m); /* if > 0, will use top  E->control.ggrd.mat_control layers and assign a prefactor for the viscosity */
  if(E->control.ggrd.age_control){
    E->control.lith_age = 1;	
  }
#endif

  if (E->control.lith_age) {
    input_int("lith_age_time",&(E->control.lith_age_time),"0",m);
    input_string("lith_age_file",E->control.lith_age_file,"",m);
    input_float("lith_age_depth",&(E->control.lith_age_depth),"0.0471",m);
    /* DJB SLAB */
    input_boolean("lith_age_depth_function",&(E->control.lith_age_depth_function),"0",m);
    input_float("lith_age_exponent",&(E->control.lith_age_exponent),"0",m);
    input_float("lith_age_min",&(E->control.lith_age_min),"0",m);
    input_float("lith_age_stencil_value",&(E->control.lith_age_stencil_value),"-999",m);

    input_int("temperature_bound_adj",&(E->control.temperature_bound_adj),"0",m);
    if (E->control.temperature_bound_adj) {
      input_float("depth_bound_adj",&(E->control.depth_bound_adj),"0.1570",m);
      input_float("width_bound_adj",&(E->control.width_bound_adj),"0.08727",m);
    }
  }

  // DJB SLAB
  if (E->control.slab_assim) {
    input_string("slab_assim_file",E->control.slab_assim_file,"",m);
  }

  return;
}


void lith_age_init(struct All_variables *E)
{
  char output_file[255];
  FILE *fp1;
  int node, i, j, output;

  int gnox, gnoy;
  gnox=E->mesh.nox;
  gnoy=E->mesh.noy;

  if (E->parallel.me == 0 ) fprintf(stderr,"INSIDE lith_age_init\n");

  /* DJB SLAB, DJB OUT */
  E->control.lith_age_min_nondim=E->control.lith_age_min/E->data.scalet;
  E->control.lith_age_stencil_value_nondim=E->control.lith_age_stencil_value/E->data.scalet;

  E->age_t=(float*) malloc((gnox*gnoy+1)*sizeof(float));

  /* DJB SLAB */
  /* age-dependent lith_age_depth */
  E->lith_age_depth_t=(float*) malloc((gnox*gnoy+1)*sizeof(float));

  if(E->control.lith_age_time==1)   {
    /* to open files every timestep */
    E->control.lith_age_old_cycles = E->monitor.solution_cycles;
    output = 1;
    (E->solver.lith_age_read_files)(E,output);
  }
  else {
    /* otherwise, just open for the first timestep */
    /* NOTE: This is only used if we are adjusting the boundaries */
    sprintf(output_file,"%s",E->control.lith_age_file);
    fp1=fopen(output_file,"r");
    if (fp1 == NULL) {
      fprintf(E->fp,"(Boundary_conditions #1) Can't open %s\n",output_file);
      parallel_process_termination();
    }
    for(i=1;i<=gnoy;i++)
      for(j=1;j<=gnox;j++) {
	node=j+(i-1)*gnox;
	if(fscanf(fp1,"%f",&(E->age_t[node])) != 1) {
          fprintf(stderr,"Error while reading file '%s'\n",output_file);
          exit(8);
        }
        /* TODO: DJB SLAB should this be divide by scalet?  In original code
           it is, but it looks like I corrected it during slab assimilation */
        /* UPDATE: in the original svn code, I write:
           bug fix.  For time-independent age (lithosphere assimilation) the age was being
           non-dimensionalised incorrectly (*scalet rather than /scalet).  This has likely
           never caused a problem because we all use time-dependent lithosphere assimilation.
           But I've corrected it nonetheless. */
	E->age_t[node]=E->age_t[node]/E->data.scalet;
      }
    fclose(fp1);
  } /* end E->control.lith_age_time == false */
}


void lith_age_construct_tic(struct All_variables *E)
{
  int i, j, k, m, node, nodeg;
  int nox, noy, noz, gnox, gnoy, gnoz;
  double r1, temp;
  float age;
  void temperatures_conform_bcs();

  noy=E->lmesh.noy;
  nox=E->lmesh.nox;
  noz=E->lmesh.noz;

  gnox=E->mesh.nox;
  gnoy=E->mesh.noy;
  gnoz=E->mesh.noz;

  for(m=1;m<=E->sphere.caps_per_proc;m++)
    for(i=1;i<=noy;i++)
      for(j=1;j<=nox;j++)
	for(k=1;k<=noz;k++)  {
	  nodeg=E->lmesh.nxs-1+j+(E->lmesh.nys+i-2)*gnox;
	  node=k+(j-1)*noz+(i-1)*nox*noz;
	  r1=E->sx[m][3][node];
	  E->T[m][node] = E->control.mantle_temp;
	  if( r1 >= E->sphere.ro-E->control.lith_age_depth )
	    { /* if closer than (lith_age_depth) from top */
              /* DJB SLAB */
              /* must be greater than lith_age_min */
              if(E->age_t[nodeg] <= E->control.lith_age_min_nondim)
                  temp = (E->sphere.ro-r1) *0.5 /sqrt(E->control.lith_age_min_nondim);
              else
	          temp = (E->sphere.ro-r1) *0.5 /sqrt(E->age_t[nodeg]);
	      E->T[m][node] = E->control.mantle_temp * erf(temp);
	    }
	}

  /* modify temperature BC to be concorded with read in T */
  lith_age_update_tbc(E);

  temperatures_conform_bcs(E);

  return;
}


void lith_age_update_tbc(struct All_variables *E)
{
  int i, j, k, m, node;
  int nox, noy, noz;
  double r1, rout, rin;
  const float e_4=1.e-4;

  noy = E->lmesh.noy;
  nox = E->lmesh.nox;
  noz = E->lmesh.noz;
  rout = E->sphere.ro;
  rin = E->sphere.ri;

  for(m=1;m<=E->sphere.caps_per_proc;m++)
    for(i=1;i<=noy;i++)
      for(j=1;j<=nox;j++)
	for(k=1;k<=noz;k++)  {
	  node=k+(j-1)*noz+(i-1)*nox*noz;
	  r1=E->sx[m][3][node];

	  if(fabs(r1-rout)>=e_4 && fabs(r1-rin)>=e_4)  {
	    E->sphere.cap[m].TB[1][node]=E->T[m][node];
	    E->sphere.cap[m].TB[2][node]=E->T[m][node];
	    E->sphere.cap[m].TB[3][node]=E->T[m][node];
	  }
	}

  return;
}


void lith_age_temperature_bound_adj(struct All_variables *E, int lv)
{
  int j,node,nno;
  float ttt2,ttt3,fff2,fff3;

  nno=E->lmesh.nno;

/* NOTE: To start, the relevent bits of "node" are zero. Thus, they only
get set to TBX/TBY/TBZ if the node is in one of the bounding regions.
Also note that right now, no matter which bounding region you are in,
all three get set to true. CPC 6/20/00 */

  if (E->control.temperature_bound_adj) {
    ttt2=E->control.theta_min + E->control.width_bound_adj;
    ttt3=E->control.theta_max - E->control.width_bound_adj;
    fff2=E->control.fi_min + E->control.width_bound_adj;
    fff3=E->control.fi_max - E->control.width_bound_adj;

    if(lv==E->mesh.gridmax)
      for(j=1;j<=E->sphere.caps_per_proc;j++)
	for(node=1;node<=E->lmesh.nno;node++)  {
	  if( ((E->sx[j][1][node]<=ttt2) && (E->sx[j][3][node]>=E->sphere.ro-E->control.depth_bound_adj)) || ((E->sx[j][1][node]>=ttt3) && (E->sx[j][3][node]>=E->sphere.ro-E->control.depth_bound_adj)) )
	    /* if < (width) from x bounds AND (depth) from top */
	    {
	      E->node[j][node]=E->node[j][node] | TBX;
	      E->node[j][node]=E->node[j][node] & (~FBX);
	      E->node[j][node]=E->node[j][node] | TBY;
	      E->node[j][node]=E->node[j][node] & (~FBY);
	      E->node[j][node]=E->node[j][node] | TBZ;
	      E->node[j][node]=E->node[j][node] & (~FBZ);
	    }

	  if( ((E->sx[j][2][node]<=fff2) && (E->sx[j][3][node]>=E->sphere.ro-E->control.depth_bound_adj)) )
	    /* if fi is < (width) from side AND z is < (depth) from top */
	    {
	      E->node[j][node]=E->node[j][node] | TBX;
	      E->node[j][node]=E->node[j][node] & (~FBX);
	      E->node[j][node]=E->node[j][node] | TBY;
	      E->node[j][node]=E->node[j][node] & (~FBY);
	      E->node[j][node]=E->node[j][node] | TBZ;
	      E->node[j][node]=E->node[j][node] & (~FBZ);
	    }

	  if( ((E->sx[j][2][node]>=fff3) && (E->sx[j][3][node]>=E->sphere.ro-E->control.depth_bound_adj)) )
	    /* if fi is < (width) from side AND z is < (depth) from top */
	    {
	      E->node[j][node]=E->node[j][node] | TBX;
	      E->node[j][node]=E->node[j][node] & (~FBX);
	      E->node[j][node]=E->node[j][node] | TBY;
	      E->node[j][node]=E->node[j][node] & (~FBY);
	      E->node[j][node]=E->node[j][node] | TBZ;
	      E->node[j][node]=E->node[j][node] & (~FBZ);
	    }

	}
  } /* end E->control.temperature_bound_adj */

  /* DJB SLAB */
  /* This sets the temperature flag(s) for lithosphere assimilation
     in the usual CIG version.  However, we do not use flags because
     they are more difficult to implement for deforming regions (GPlates)
     where we want to ignore assimilation.  Hence this code block is 
     commented out */
#if 0
  if (E->control.lith_age_time) {
    if(lv==E->mesh.gridmax)
      for(j=1;j<=E->sphere.caps_per_proc;j++)
	for(node=1;node<=E->lmesh.nno;node++)  {
	  if(E->sx[j][3][node]>=E->sphere.ro-E->control.lith_age_depth)
	    { /* if closer than (lith_age_depth) from top */
	      E->node[j][node]=E->node[j][node] | TBX;
	      E->node[j][node]=E->node[j][node] & (~FBX);
	      E->node[j][node]=E->node[j][node] | TBY;
	      E->node[j][node]=E->node[j][node] & (~FBY);
	      E->node[j][node]=E->node[j][node] | TBZ;
	      E->node[j][node]=E->node[j][node] & (~FBZ);
	    }

	}
  } /* end E->control.lith_age_time */
#endif

  return;
}

/* DJB SLAB */
static void get_lith_age_depth(struct All_variables *E)
{
  int m,i,j,nox,noy,nodeg,gnox;
  float age_t;

  gnox=E->mesh.nox;
  nox=E->lmesh.nox;
  noy=E->lmesh.noy;

  for(m=1;m<=E->sphere.caps_per_proc;m++)
    for(i=1;i<=noy;i++)
      for(j=1;j<=nox;j++) {
        nodeg=E->lmesh.nxs-1+j+(E->lmesh.nys+i-2)*gnox;

        age_t = E->age_t[nodeg];

        /* assimilation regions */
        if( age_t >= E->control.lith_age_stencil_value_nondim) {

          /* minimum age, which correlates to a minimum depth constraint */
          age_t = max( age_t, E->control.lith_age_min_nondim );

          if( E->control.lith_age_depth_function)
            /* from Turcotte and Schubert */
            /* here, E-control.lith_age_depth is a PREFACTOR */
            E->lith_age_depth_t[nodeg] = 2.32*sqrt(age_t);
          else
            /* to default to constant depth (lith_age_depth) */
            E->lith_age_depth_t[nodeg] = 1.0;

          /* for control.lith_age_depth_function, E->lith_age_depth is a PREFACTOR */
          E->lith_age_depth_t[nodeg] *= E->control.lith_age_depth;

        }
        /* for no assimilation regions */
        else
          E->lith_age_depth_t[nodeg] = -1.0; // arbitrary, but easy to spot problems

        /* for testing */
        /*if (E->parallel.me == 3){
          fprintf(stderr,"nodeg=%d, E->lith_age_depth_t[nodeg]=%f\n",nodeg,E->lith_age_depth_t[nodeg]);
          //fprintf(stderr,"slab_temp=%f, E->T[j][node]=%f\n",E->sphere.cap[j].slab_temp[node],E->T[j][node]);
        }*/


  }

  return;
}

void lith_age_conform_tbc(struct All_variables *E)
{
  void get_lith_age_depth(struct All_variables *E); // DJB SLAB

  int m,j,node,nox,noz,noy,gnox,gnoy,gnoz,nodeg,i,k;
  float ttt2,ttt3,fff2,fff3;
  float r1,t1,f1,t0,temp;
  float depth;
  float e_4;
  FILE *fp1;
  char output_file[255];
  int output;
  float lith_age_depth; // DJB SLAB age-dependent

  e_4=1.e-4;
  output = 0;

  gnox=E->mesh.nox;
  gnoy=E->mesh.noy;
  gnoz=E->mesh.noz;
  nox=E->lmesh.nox;
  noy=E->lmesh.noy;
  noz=E->lmesh.noz;

  if(E->control.lith_age_time==1)   {
    /* to open files every timestep */
    if (E->control.lith_age_old_cycles != E->monitor.solution_cycles) {
      /*update so that output only happens once*/
      output = 1;
      E->control.lith_age_old_cycles = E->monitor.solution_cycles;
    }
    if (E->parallel.me == 0) fprintf(stderr,"INSIDE lith_age_conform_tbc\n");
    (E->solver.lith_age_read_files)(E,output);
  }

  /* NOW SET THE TEMPERATURES IN THE BOUNDARY REGIONS */
  if(E->monitor.solution_cycles>1 && E->control.temperature_bound_adj) {
    ttt2=E->control.theta_min + E->control.width_bound_adj;
    ttt3=E->control.theta_max - E->control.width_bound_adj;
    fff2=E->control.fi_min + E->control.width_bound_adj;
    fff3=E->control.fi_max - E->control.width_bound_adj;

    for(m=1;m<=E->sphere.caps_per_proc;m++)
      for(i=1;i<=noy;i++)
	for(j=1;j<=nox;j++)
	  for(k=1;k<=noz;k++)  {
	    nodeg=E->lmesh.nxs-1+j+(E->lmesh.nys+i-2)*gnox;
	    node=k+(j-1)*noz+(i-1)*nox*noz;
	    t1=E->sx[m][1][node];
	    f1=E->sx[m][2][node];
	    r1=E->sx[m][3][node];

	    if(fabs(r1-E->sphere.ro)>=e_4 && fabs(r1-E->sphere.ri)>=e_4)  { /* if NOT right on the boundary */
	      if( ((E->sx[m][1][node]<=ttt2) && (E->sx[m][3][node]>=E->sphere.ro-E->control.depth_bound_adj)) || ((E->sx[m][1][node]>=ttt3) && (E->sx[m][3][node]>=E->sphere.ro-E->control.depth_bound_adj)) ) {
		/* if < (width) from x bounds AND (depth) from top */
                // DJB SLAB
                if(E->age_t[nodeg] <= E->control.lith_age_min_nondim)
                   temp = (E->sphere.ro-r1) *0.5 /sqrt(E->control.lith_age_min_nondim);
                else
                   temp = (E->sphere.ro-r1) *0.5 /sqrt(E->age_t[nodeg]);
		t0 = E->control.mantle_temp * erf(temp);

		/* keep the age the same! */
		E->sphere.cap[m].TB[1][node]=t0;
		E->sphere.cap[m].TB[2][node]=t0;
		E->sphere.cap[m].TB[3][node]=t0;
	      }

	      if( ((E->sx[m][2][node]<=fff2) || (E->sx[m][2][node]>=fff3)) && (E->sx[m][3][node]>=E->sphere.ro-E->control.depth_bound_adj) ) {
		/* if < (width) from y bounds AND (depth) from top */
                /* keep the age the same */
                // DJB SLAB
                if(E->age_t[nodeg] <= E->control.lith_age_min_nondim)
                    temp = (E->sphere.ro-r1) *0.5 /sqrt(E->control.lith_age_min_nondim);
                else
                    temp = (E->sphere.ro-r1) *0.5 /sqrt(E->age_t[nodeg]);
		t0 = E->control.mantle_temp * erf(temp);

		E->sphere.cap[m].TB[1][node]=t0;
		E->sphere.cap[m].TB[2][node]=t0;
		E->sphere.cap[m].TB[3][node]=t0;

	      }

	    }

	  } /* end k   */

  }   /*  end of solution cycles  && temperature_bound_adj */

  /* DJB SLAB */
  get_lith_age_depth( E );

  /* NOW SET THE TEMPERATURES IN THE LITHOSPHERE IF CHANGING EVERY TIME STEP */
  if(E->monitor.solution_cycles>0 && E->control.lith_age_time)   {
    for(m=1;m<=E->sphere.caps_per_proc;m++)
      for(i=1;i<=noy;i++)
	for(j=1;j<=nox;j++)
	  for(k=1;k<=noz;k++)  {
	    nodeg=E->lmesh.nxs-1+j+(E->lmesh.nys+i-2)*gnox;
	    node=k+(j-1)*noz+(i-1)*nox*noz;
	    t1=E->sx[m][1][node];
	    f1=E->sx[m][2][node];
	    r1=E->sx[m][3][node];

            /* DJB SLAB */
            lith_age_depth = E->lith_age_depth_t[nodeg];

            /* DJB SLAB */
	    if(fabs(r1-E->sphere.ro)>=e_4 && fabs(r1-E->sphere.ri)>=e_4)  { /* if NOT right on the boundary */
              depth = E->sphere.ro - r1;

              if(depth <= lith_age_depth && E->age_t[nodeg]>=E->control.lith_age_stencil_value_nondim) {
                  /* if closer than (lith_age_depth) from top */

                       /* set a new age from the file */
                        temp = depth * 0.5 /sqrt(E->age_t[nodeg]);
                        t0 = E->control.mantle_temp * erf(temp);

                        E->sphere.cap[m].TB[1][node]=t0;
                        E->sphere.cap[m].TB[2][node]=t0;
                        E->sphere.cap[m].TB[3][node]=t0;
	      }
	    }
	  }     /* end k   */
  }   /*  end of solution cycles  && lith_age_time */

  return;
}


void assimilate_lith_conform_bcs(struct All_variables *E)
{
  void temperatures_conform_bcs2(struct All_variables *); // DJB SLAB
  float depth, daf, assimilate_new_temp;
  int m,j,nno,node,nox,noz,noy,gnox,gnoy,gnoz,nodeg,ii,i,k;
  unsigned int type;
  float znd;
  float lith_age_depth;

  nno=E->lmesh.nno;
  gnox=E->mesh.nox;
  gnoy=E->mesh.noy;
  gnoz=E->mesh.noz;
  nox=E->lmesh.nox;
  noy=E->lmesh.noy;
  noz=E->lmesh.noz;

  /* DJB SLAB */
  /* First, assimilate lithosphere temperature for all depths less than lith_age_depth */
  
  for(m=1;m<=E->sphere.caps_per_proc;m++)
      for(i=1;i<=noy;i++)
        for(j=1;j<=nox;j++)
          for(k=1;k<=noz;k++)  {
            nodeg=E->lmesh.nxs-1+j+(E->lmesh.nys+i-2)*gnox;
            node=k+(j-1)*noz+(i-1)*nox*noz;
            
            depth = E->sphere.ro - E->sx[m][3][node];
            
            lith_age_depth = E->lith_age_depth_t[nodeg];
            
            if(depth <= lith_age_depth && E->age_t[nodeg]>=E->control.lith_age_stencil_value_nondim) {
                znd = depth/lith_age_depth;
                daf = pow(znd,E->control.lith_age_exponent);
                /* this depth assimilation factor was used prior to svn r29 */
                /* daf = 0.5*depth/E->control.lith_age_depth; */
                assimilate_new_temp = E->sphere.cap[m].TB[3][node];
                
                E->T[m][node] = daf*E->T[m][node] + (1.0-daf)*assimilate_new_temp;
                // for testing
                // E->T[m][node] = assimilate_new_temp;
             }

           } /* next node */

  /* Second, apply thermal bcs to top and bottom surface.  This will over-ride
     E->T[m][node] for the uppermost nodes that was applied above. */
  temperatures_conform_bcs2(E);

return;
}

void assimilate_slab_conform_bcs(struct All_variables *E)
{
  float depth;
  int m,j,nno,node,nox,noz,noy,gnox,gnoy,gnoz,nodeg,ii,i,k;
  unsigned int type;
  float r1,t1,f1, assimilate_slab_temp;
  int output;
  double alpha;

  output = 1;

  (E->solver.slab_temperature_read_files)(E,output);

  nno=E->lmesh.nno;
  gnox=E->mesh.nox;
  gnoy=E->mesh.noy;
  gnoz=E->mesh.noz;
  nox=E->lmesh.nox;
  noy=E->lmesh.noy;
  noz=E->lmesh.noz;

  for(j=1;j<=E->sphere.caps_per_proc;j++)
    for(node=1;node<=E->lmesh.nno;node++)  {
      alpha=E->sphere.cap[j].slab_sten[node];
      E->T[j][node] = (1.0-alpha)*E->T[j][node] + alpha*E->sphere.cap[j].slab_temp[node];
    } /* next node */

return;
}

