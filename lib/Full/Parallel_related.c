/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * 
 *<LicenseText>
 *=====================================================================
 *
 *                              CitcomS
 *                 ---------------------------------
 *
 *                              Authors:
 *           Louis Moresi, Shijie Zhong, Lijie Han, Eh Tan,
 *           Clint Conrad, Michael Gurnis, and Eun-seo Choi
 *          (c) California Institute of Technology 1994-2005
 *
 *        By downloading and/or installing this software you have
 *       agreed to the CitcomS.py-LICENSE bundled with this software.
 *             Free for non-commercial academic research ONLY.
 *      This program is distributed WITHOUT ANY WARRANTY whatsoever.
 *
 *=====================================================================
 *
 *  Copyright June 2005, by the California Institute of Technology.
 *  ALL RIGHTS RESERVED. United States Government Sponsorship Acknowledged.
 * 
 *  Any commercial use must be negotiated with the Office of Technology
 *  Transfer at the California Institute of Technology. This software
 *  may be subject to U.S. export control laws and regulations. By
 *  accepting this software, the user agrees to comply with all
 *  applicable U.S. export laws and regulations, including the
 *  International Traffic and Arms Regulations, 22 C.F.R. 120-130 and
 *  the Export Administration Regulations, 15 C.F.R. 730-744. User has
 *  the responsibility to obtain export licenses, or other export
 *  authority as may be required before exporting such information to
 *  foreign countries or providing access to foreign nationals.  In no
 *  event shall the California Institute of Technology be liable to any
 *  party for direct, indirect, special, incidental or consequential
 *  damages, including lost profits, arising out of the use of this
 *  software and its documentation, even if the California Institute of
 *  Technology has been advised of the possibility of such damage.
 * 
 *  The California Institute of Technology specifically disclaims any
 *  warranties, including the implied warranties or merchantability and
 *  fitness for a particular purpose. The software and documentation
 *  provided hereunder is on an "as is" basis, and the California
 *  Institute of Technology has no obligations to provide maintenance,
 *  support, updates, enhancements or modifications.
 *
 *=====================================================================
 *</LicenseText>
 * 
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
/* Routines here are for intel paragon with MPI */

#include <mpi.h>
#include <math.h>

#include "element_definitions.h"
#include "global_defs.h"
#include "sphere_communication.h"

#include "parallel_related.h"

void set_horizontal_communicator(struct All_variables*);
void set_vertical_communicator(struct All_variables*);



/* ============================================ */
/* ============================================ */

void parallel_process_termination()
{

  MPI_Finalize();
  exit(8);
  return;
  }

/* ============================================ */
/* ============================================ */

void parallel_process_sync(struct All_variables *E)
{

  MPI_Barrier(E->parallel.world);
  return;
  }


/* ==========================   */

 double CPU_time0()
{
 double time, MPI_Wtime();
 time = MPI_Wtime();
 return (time);
}

/* ============================================ */
/* ============================================ */

void parallel_processor_setup(struct All_variables *E)
  {

  int i,j,k,m,me,temp,pid_surf;
  int cap_id_surf;
  int surf_proc_per_cap, proc_per_cap, total_proc;

  me = E->parallel.me;

  if ( E->parallel.nprocx != E->parallel.nprocy ) {
    if (E->parallel.me==0) fprintf(stderr,"!!!! nprocx must equal to nprocy \n");
    parallel_process_termination();
  }

  surf_proc_per_cap = E->parallel.nprocx * E->parallel.nprocy;
  proc_per_cap = surf_proc_per_cap * E->parallel.nprocz;
  total_proc = E->sphere.caps * proc_per_cap;
  E->parallel.total_surf_proc = E->sphere.caps * surf_proc_per_cap;

  if ( total_proc != E->parallel.nproc ) {
    if (E->parallel.me==0) fprintf(stderr,"!!!! # of requested CPU is incorrect \n");
    parallel_process_termination();
    }

  E->sphere.caps_per_proc = max(1,E->sphere.caps*E->parallel.nprocz/E->parallel.nproc);

  if (E->sphere.caps_per_proc > 1) {
    if (E->parallel.me==0) fprintf(stderr,"!!!! # caps per proc > 1 is not supported.\n \n");
    parallel_process_termination();
  }

  /* determine the location of processors in each cap */
  cap_id_surf = me / proc_per_cap;

 /* z-direction first*/
  E->parallel.me_loc[3] = (me - cap_id_surf*proc_per_cap) % E->parallel.nprocz;

 /* x-direction then*/
  E->parallel.me_loc[1] = ((me - cap_id_surf*proc_per_cap - E->parallel.me_loc[3])/E->parallel.nprocz) % E->parallel.nprocx;

 /* y-direction then*/
  E->parallel.me_loc[2] = ((((me - cap_id_surf*proc_per_cap - E->parallel.me_loc[3])/E->parallel.nprocz) - E->parallel.me_loc[1])/E->parallel.nprocx) % E->parallel.nprocy;


/*
the numbering of proc in each caps is as so (example for an xyz = 2x2x2 box):
NOTE: This is different (in a way) than the numbering of the nodes:
the nodeal number has the first oordinate as theta, which goes N-S and
the second oordinate as fi, which goes E-W. Here we use R-L as the first
oordinate and F-B
 0 = lower  left front corner=[000]  4 = lower  left back corner=[010]
 1 = upper  left front corner=[001]  5 = upper  left back corner=[011]
 2 = lower right front corner=[100]  6 = lower right back corner=[110]
 3 = upper right front corner=[101]  7 = upper right back corner=[111]
[xyz] is x=E->parallel.me_loc[1],y=E->parallel.me_loc[2],z=E->parallel.me_loc[3]
*/

  /* determine cap id for each cap in a given processor  */
  pid_surf = me/proc_per_cap; /* cap number (0~11) */
  i = cases[E->sphere.caps_per_proc]; /* 1 for more than 12 processors */

  for (j=1;j<=E->sphere.caps_per_proc;j++)  {
    temp = pid_surf*E->sphere.caps_per_proc + j-1; /* cap number (out of 12) */
    E->sphere.capid[j] = incases1[i].links[temp]; /* id (1~12) of the current cap */
    }

  /* determine which caps are linked with each of 12 caps  */
  /* if the 12 caps are broken, set these up instead */
  if (surf_proc_per_cap > 1) {
     E->sphere.max_connections = 8;
  }

  /* steup location-to-processor map */
  E->parallel.loc2proc_map = (int ****) malloc(E->sphere.caps*sizeof(int ***));
  for (m=0;m<E->sphere.caps;m++)  {
    E->parallel.loc2proc_map[m] = (int ***) malloc(E->parallel.nprocx*sizeof(int **));
    for (i=0;i<E->parallel.nprocx;i++) {
      E->parallel.loc2proc_map[m][i] = (int **) malloc(E->parallel.nprocy*sizeof(int *));
      for (j=0;j<E->parallel.nprocy;j++)
	E->parallel.loc2proc_map[m][i][j] = (int *) malloc(E->parallel.nprocz*sizeof(int));
    }
  }

  for (m=0;m<E->sphere.caps;m++)
    for (i=0;i<E->parallel.nprocx;i++)
      for (j=0;j<E->parallel.nprocy;j++)
	for (k=0;k<E->parallel.nprocz;k++) {
	  if (E->sphere.caps_per_proc>1) {
	    temp = cases[E->sphere.caps_per_proc];
	    E->parallel.loc2proc_map[m][i][j][k] = incases2[temp].links[m-1];
	  }
	  else
	    E->parallel.loc2proc_map[m][i][j][k] = m*proc_per_cap
	      + j*E->parallel.nprocx*E->parallel.nprocz
	      + i*E->parallel.nprocz + k;
	}

  if (E->control.verbose) {
    fprintf(E->fp_out,"me=%d loc1=%d loc2=%d loc3=%d\n",me,E->parallel.me_loc[1],E->parallel.me_loc[2],E->parallel.me_loc[3]);
    for (j=1;j<=E->sphere.caps_per_proc;j++) {
      fprintf(E->fp_out,"capid[%d]=%d \n",j,E->sphere.capid[j]);
    }
    for (m=0;m<E->sphere.caps;m++)
      for (j=0;j<E->parallel.nprocy;j++)
	for (i=0;i<E->parallel.nprocx;i++)
	  for (k=0;k<E->parallel.nprocz;k++)
	    fprintf(E->fp_out,"loc2proc_map[cap=%d][x=%d][y=%d][z=%d] = %d\n",
		    m,i,j,k,E->parallel.loc2proc_map[m][i][j][k]);

    fflush(E->fp_out);
  }

  set_vertical_communicator(E);
  set_horizontal_communicator(E);

  E->exchange_node_d = exchange_node_d;
  E->exchange_node_f = exchange_node_f;

  return;
  }



void set_horizontal_communicator(struct All_variables *E)
{
  MPI_Group world_g, horizon_g;
  int i,j,k,m,n;
  int *processors;

  processors = (int *) malloc((E->parallel.total_surf_proc+1)*sizeof(int));

  k = E->parallel.me_loc[3];
  n = 0;
  for (m=0;m<E->sphere.caps;m++)
    for (i=0;i<E->parallel.nprocx;i++)
      for (j=0;j<E->parallel.nprocy;j++) {
	processors[n] = E->parallel.loc2proc_map[m][i][j][k];
	n++;
      }

  MPI_Comm_group(E->parallel.world, &world_g);
  MPI_Group_incl(world_g, E->parallel.total_surf_proc, processors, &horizon_g);
  MPI_Comm_create(E->parallel.world, horizon_g, &(E->parallel.horizontal_comm));

  if (E->control.verbose) {
    fprintf(E->fp_out,"horizontal group of me=%d loc3=%d\n",E->parallel.me,E->parallel.me_loc[3]);
    for (j=0;j<E->parallel.total_surf_proc;j++) {
      fprintf(E->fp_out,"%d proc=%d\n",j,processors[j]);
    }
    fflush(E->fp_out);
  }

  MPI_Group_free(&horizon_g);
  MPI_Group_free(&world_g);
  free((void *) processors);

  return;
}


void set_vertical_communicator(struct All_variables *E)
{
  MPI_Group world_g, vertical_g;
  int i,j,k,m;
  int *processors;

  processors = (int *)malloc((E->parallel.nprocz+2)*sizeof(int));

  m = E->sphere.capid[1] - 1;  // assume 1 cap per proc.
  i = E->parallel.me_loc[1];
  j = E->parallel.me_loc[2];

  for (k=0;k<E->parallel.nprocz;k++) {
      processors[k] = E->parallel.loc2proc_map[m][i][j][k];
  }

  MPI_Comm_group(E->parallel.world, &world_g);
  MPI_Group_incl(world_g, E->parallel.nprocz, processors, &vertical_g);
  MPI_Comm_create(E->parallel.world, vertical_g, &(E->parallel.vertical_comm));

  if (E->control.verbose) {
    fprintf(E->fp_out,"vertical group of me=%d loc1=%d loc2=%d\n",E->parallel.me,E->parallel.me_loc[1],E->parallel.me_loc[2]);
    for (j=0;j<E->parallel.nprocz;j++) {
      fprintf(E->fp_out,"%d proc=%d\n",j,processors[j]);
    }
    fflush(E->fp_out);
  }

  MPI_Group_free(&vertical_g);
  MPI_Group_free(&world_g);
  free((void *) processors);
}


/* =========================================================================
get element information for each processor.
 ========================================================================= */

void parallel_domain_decomp0(struct All_variables *E)
  {

  int i,nox,noz,noy,me;

  me = E->parallel.me;

  E->lmesh.elx = E->mesh.elx/E->parallel.nprocx;
  E->lmesh.elz = E->mesh.elz/E->parallel.nprocz;
  E->lmesh.ely = E->mesh.ely/E->parallel.nprocy;
  E->lmesh.nox = E->lmesh.elx + 1;
  E->lmesh.noz = E->lmesh.elz + 1;
  E->lmesh.noy = E->lmesh.ely + 1;

  E->lmesh.exs = E->parallel.me_loc[1]*E->lmesh.elx;
  E->lmesh.eys = E->parallel.me_loc[2]*E->lmesh.ely;
  E->lmesh.ezs = E->parallel.me_loc[3]*E->lmesh.elz;
  E->lmesh.nxs = E->parallel.me_loc[1]*E->lmesh.elx+1;
  E->lmesh.nys = E->parallel.me_loc[2]*E->lmesh.ely+1;
  E->lmesh.nzs = E->parallel.me_loc[3]*E->lmesh.elz+1;

  E->lmesh.nno = E->lmesh.noz*E->lmesh.nox*E->lmesh.noy;
  E->lmesh.nel = E->lmesh.ely*E->lmesh.elx*E->lmesh.elz;
  E->lmesh.npno = E->lmesh.nel;

  E->lmesh.nsf = E->lmesh.nno/E->lmesh.noz;
  E->lmesh.snel = E->lmesh.elx*E->lmesh.ely;


  i = cases[E->sphere.caps_per_proc];

  E->parallel.nproc_sph[1] = incases3[i].xy[0];
  E->parallel.nproc_sph[2] = incases3[i].xy[1];

  E->sphere.lelx = E->sphere.elx/E->parallel.nproc_sph[1];
  E->sphere.lely = E->sphere.ely/E->parallel.nproc_sph[2];
  E->sphere.lsnel = E->sphere.lely*E->sphere.lelx;
  E->sphere.lnox = E->sphere.lelx + 1;
  E->sphere.lnoy = E->sphere.lely + 1;
  E->sphere.lnsf = E->sphere.lnox*E->sphere.lnoy;

/* NOTE: These are for spherical harmonics - they may not be correct! CPC */
  for (i=0;i<=E->parallel.nprocz-1;i++)
    if (E->parallel.me_loc[3] == i)    {
      E->parallel.me_sph = (E->parallel.me-i)/E->parallel.nprocz;
      E->parallel.me_loc_sph[1] = E->parallel.me_sph%E->parallel.nproc_sph[1];
      E->parallel.me_loc_sph[2] = E->parallel.me_sph/E->parallel.nproc_sph[1];
      }

  E->sphere.lexs = E->sphere.lelx * E->parallel.me_loc_sph[1];
  E->sphere.leys = E->sphere.lely * E->parallel.me_loc_sph[2];



  for(i=E->mesh.levmax;i>=E->mesh.levmin;i--)   {

     if (E->control.NMULTIGRID||E->control.EMULTIGRID)  {
        nox = E->lmesh.elx/(int)pow(2.0,(double)(E->mesh.levmax-i))+1;
        noy = E->lmesh.ely/(int)pow(2.0,(double)(E->mesh.levmax-i))+1;
        noz = E->lmesh.elz/(int)pow(2.0,(double)(E->mesh.levmax-i))+1;
        E->parallel.redundant[i]=0;
        }
     else
        { noz = E->lmesh.noz;
          noy = E->lmesh.noy;
          nox = E->lmesh.nox;
        }

     E->lmesh.ELX[i] = nox-1;
     E->lmesh.ELY[i] = noy-1;
     E->lmesh.ELZ[i] = noz-1;
     E->lmesh.NOZ[i] = noz;
     E->lmesh.NOY[i] = noy;
     E->lmesh.NOX[i] = nox;
     E->lmesh.NNO[i] = nox * noz * noy;
     E->lmesh.NNOV[i] = E->lmesh.NNO[i];
     E->lmesh.SNEL[i] = E->lmesh.ELX[i]*E->lmesh.ELY[i];

     E->lmesh.NEL[i] = (nox-1) * (noz-1) * (noy-1);
     E->lmesh.NPNO[i] = E->lmesh.NEL[i] ;

     E->lmesh.NEQ[i] = E->mesh.nsd * E->lmesh.NNOV[i] ;

     E->lmesh.EXS[i] = E->parallel.me_loc[1]*E->lmesh.ELX[i];
     E->lmesh.EYS[i] = E->parallel.me_loc[2]*E->lmesh.ELY[i];
     E->lmesh.EZS[i] = E->parallel.me_loc[3]*E->lmesh.ELZ[i];
     E->lmesh.NXS[i] = E->parallel.me_loc[1]*E->lmesh.ELX[i]+1;
     E->lmesh.NYS[i] = E->parallel.me_loc[2]*E->lmesh.ELY[i]+1;
     E->lmesh.NZS[i] = E->parallel.me_loc[3]*E->lmesh.ELZ[i]+1;
     }

/*
fprintf(stderr,"b %d %d %d %d %d %d %d\n",E->parallel.me,E->parallel.me_loc[1],E->parallel.me_loc[2],E->parallel.me_loc[3],E->lmesh.nzs,E->lmesh.nys,E->lmesh.noy);
*/
/* parallel_process_termination();
*/
  return;
  }



/* ============================================
 determine boundary nodes for
 exchange info across the boundaries
 ============================================ */

void parallel_domain_boundary_nodes(E)
  struct All_variables *E;
  {

  void parallel_process_termination();

  int m,i,ii,j,k,l,node,el,lnode;
  int lev,ele,elx,elz,ely,nel,nno,nox,noz,noy;
  FILE *fp;
  char output_file[255];

  for(lev=E->mesh.gridmin;lev<=E->mesh.gridmax;lev++)   {
    for(m=1;m<=E->sphere.caps_per_proc;m++)   {
      nel = E->lmesh.NEL[lev];
      elx = E->lmesh.ELX[lev];
      elz = E->lmesh.ELZ[lev];
      ely = E->lmesh.ELY[lev];
      nox = E->lmesh.NOX[lev];
      noy = E->lmesh.NOY[lev];
      noz = E->lmesh.NOZ[lev];
      nno = E->lmesh.NNO[lev];

/* do the ZOY boundary elements first */
      lnode = 0;
      ii =1;              /* left */
      for(j=1;j<=noz;j++)
      for(k=1;k<=noy;k++)  {
        node = j + (k-1)*noz*nox;
        E->parallel.NODE[lev][m][++lnode].bound[ii] =  node;
        E->NODE[lev][m][node] = E->NODE[lev][m][node] | OFFSIDE;
        }

      E->parallel.NUM_NNO[lev][m].bound[ii] = lnode;


      lnode = 0;
      ii =2;              /* right */
      for(j=1;j<=noz;j++)
      for(k=1;k<=noy;k++)      {
        node = (nox-1)*noz + j + (k-1)*noz*nox;
        E->parallel.NODE[lev][m][++lnode].bound[ii] =  node;
        E->NODE[lev][m][node] = E->NODE[lev][m][node] | OFFSIDE;
        }

      E->parallel.NUM_NNO[lev][m].bound[ii] = lnode;


/* do XOY boundary elements */
      ii=5;                           /* bottom */
      lnode=0;
      for(k=1;k<=noy;k++)
      for(i=1;i<=nox;i++)   {
        node = (k-1)*nox*noz + (i-1)*noz + 1;
        E->parallel.NODE[lev][m][++lnode].bound[ii] = node;
        E->NODE[lev][m][node] = E->NODE[lev][m][node] | OFFSIDE;
        }

      E->parallel.NUM_NNO[lev][m].bound[ii] = lnode;

      ii=6;                           /* top  */
      lnode=0;
      for(k=1;k<=noy;k++)
      for(i=1;i<=nox;i++)  {
        node = (k-1)*nox*noz + i*noz;
        E->parallel.NODE[lev][m][++lnode].bound[ii] = node;
        E->NODE[lev][m][node] = E->NODE[lev][m][node] | OFFSIDE;
        }

      E->parallel.NUM_NNO[lev][m].bound[ii] = lnode;


/* do XOZ boundary elements for 3D */
      ii=3;                           /* front */
      lnode=0;
      for(j=1;j<=noz;j++)
      for(i=1;i<=nox;i++)   {
        node = (i-1)*noz +j;
        E->parallel.NODE[lev][m][++lnode].bound[ii] = node;
        E->NODE[lev][m][node] = E->NODE[lev][m][node] | OFFSIDE;
        }

      E->parallel.NUM_NNO[lev][m].bound[ii] = lnode;

      ii=4;                           /* rear */
      lnode=0;
      for(j=1;j<=noz;j++)
      for(i=1;i<=nox;i++)   {
        node = noz*nox*(noy-1) + (i-1)*noz +j;
        E->parallel.NODE[lev][m][++lnode].bound[ii] = node;
        E->NODE[lev][m][node] = E->NODE[lev][m][node] | OFFSIDE;
        }

      E->parallel.NUM_NNO[lev][m].bound[ii] = lnode;

         /* determine the overlapped nodes between caps or between proc */

    if (E->parallel.me_loc[3]!=E->parallel.nprocz-1 )
      for (lnode=1;lnode<=E->parallel.NUM_NNO[lev][m].bound[6];lnode++) {
        node = E->parallel.NODE[lev][m][lnode].bound[6];
        E->NODE[lev][m][node] = E->NODE[lev][m][node] | SKIP;
        }

    if (E->sphere.capid[m]==12 && E->parallel.me_loc[1]==E->parallel.nprocx-1 && E->parallel.me_loc[2]==E->parallel.nprocy-1) /* back right of cap number 12 */
      for (lnode=1;lnode<=E->parallel.NUM_NNO[lev][m].bound[2];lnode++) {
        node = E->parallel.NODE[lev][m][lnode].bound[2];
        if (node<=nno-noz)  {
           E->NODE[lev][m][node] = E->NODE[lev][m][node] | SKIP;
           if (E->parallel.me_loc[3]==E->parallel.nprocz-1 || node%noz!=0)
             E->NODE[lev][m][node] = E->NODE[lev][m][node] | SKIPS;
	   }
        }
    else
      for (lnode=1;lnode<=E->parallel.NUM_NNO[lev][m].bound[2];lnode++) {
        node = E->parallel.NODE[lev][m][lnode].bound[2];
        E->NODE[lev][m][node] = E->NODE[lev][m][node] | SKIP;
        if (E->parallel.me_loc[3]==E->parallel.nprocz-1 || node%noz!=0)
          E->NODE[lev][m][node] = E->NODE[lev][m][node] | SKIPS;
        }

    if (E->sphere.capid[m]==1 && E->parallel.me_loc[1]==0 && E->parallel.me_loc[2]==0) /* front left of cap number 1 */
      for (lnode=1;lnode<=E->parallel.NUM_NNO[lev][m].bound[3];lnode++) {
        node = E->parallel.NODE[lev][m][lnode].bound[3];
        if (node>noz)  {
           E->NODE[lev][m][node] = E->NODE[lev][m][node] | SKIP;
           if (E->parallel.me_loc[3]==E->parallel.nprocz-1 || node%noz!=0)
             E->NODE[lev][m][node] = E->NODE[lev][m][node] | SKIPS;
	   }
        }
    else
      for (lnode=1;lnode<=E->parallel.NUM_NNO[lev][m].bound[3];lnode++) {
        node = E->parallel.NODE[lev][m][lnode].bound[3];
        E->NODE[lev][m][node] = E->NODE[lev][m][node] | SKIP;
        if (E->parallel.me_loc[3]==E->parallel.nprocz-1 || node%noz!=0)
           E->NODE[lev][m][node] = E->NODE[lev][m][node] | SKIPS;
        }

      }       /* end for m */
    }   /* end for level */

/*  ii=0; */
/*  for (m=1;m<=E->sphere.caps_per_proc;m++) */
/*     for (node=1;node<=E->lmesh.nno;node++) */
/*       if(E->node[m][node] & SKIPS) */
/*         ii+=1; */

/*  MPI_Allreduce(&ii, &node  ,1,MPI_INT,MPI_SUM,E->parallel.world); */

/*  E->mesh.nno = E->lmesh.nno*E->parallel.nproc - node - 2*E->mesh.noz; */
/*  E->mesh.neq = E->mesh.nno*3; */

if (E->control.verbose) {
 fprintf(E->fp_out,"output_shared_nodes %d \n",E->parallel.me);
 for(lev=E->mesh.gridmax;lev>=E->mesh.gridmin;lev--)
   for (m=1;m<=E->sphere.caps_per_proc;m++)      {
    fprintf(E->fp_out,"lev=%d  me=%d capid=%d m=%d \n",lev,E->parallel.me,E->sphere.capid[m],m);
    for (ii=1;ii<=6;ii++)
      for (i=1;i<=E->parallel.NUM_NNO[lev][m].bound[ii];i++)
        fprintf(E->fp_out,"ii=%d   %d %d \n",ii,i,E->parallel.NODE[lev][m][i].bound[ii]);

    lnode=0;
    for (node=1;node<=E->lmesh.nno;node++)
      if((E->NODE[lev][m][node] & SKIP)) {
        lnode++;
        fprintf(E->fp_out,"skip %d %d \n",lnode,node);
        }
    fflush(E->fp_out);
    }
  }



  return;
  }


/* ============================================
 determine communication routs  and boundary ID for
 exchange info across the boundaries
 assuming fault nodes are in the top row of processors
 ============================================ */

void parallel_communication_routs_v(E)
  struct All_variables *E;
  {

  int m,i,ii,j,k,l,node,el,elt,lnode,jj,doff,target;
  int lev,elx,elz,ely,nno,nox,noz,noy,p,kkk,kk,kf,kkkp;
  int me, nprocx,nprocy,nprocz,nprocxz;
  int tscaps,cap,scap,large,npass,lx,ly,lz,temp,layer;

  void face_eqn_node_to_pass(struct All_variables *, int, int, int, int);
  void line_eqn_node_to_pass(struct All_variables *, int, int, int, int, int, int);

  const int dims=E->mesh.nsd;

  me = E->parallel.me;
  nprocx = E->parallel.nprocx;
  nprocy = E->parallel.nprocy;
  nprocz = E->parallel.nprocz;
  nprocxz = nprocx * nprocz;
  tscaps = E->parallel.total_surf_proc;
  lx = E->parallel.me_loc[1];
  ly = E->parallel.me_loc[2];
  lz = E->parallel.me_loc[3];

        /* determine the communications in horizontal direction        */
  for(lev=E->mesh.gridmax;lev>=E->mesh.gridmin;lev--)       {
    nox = E->lmesh.NOX[lev];
    noz = E->lmesh.NOZ[lev];
    noy = E->lmesh.NOY[lev];

    for(m=1;m<=E->sphere.caps_per_proc;m++)    {
      cap = E->sphere.capid[m] - 1;  /* which cap I am in (0~11) */

      /* left */
      npass = ii = 1;
      if (lx != 0)
	target = E->parallel.loc2proc_map[cap][lx-1][ly][lz];
      else
	if ( cap%3 != 0) {
	  temp = (cap+2) % 12;
	  target = E->parallel.loc2proc_map[temp][nprocx-1][ly][lz];
	}
	else {
	  temp = (cap+3) % 12;
	  target = E->parallel.loc2proc_map[temp][ly][0][lz];
	}

      E->parallel.PROCESSOR[lev][m].pass[npass] = target;
      face_eqn_node_to_pass(E,lev,m,npass,ii);

      /* right */
      npass = ii = 2;
      if (lx != nprocx-1)
	target = E->parallel.loc2proc_map[cap][lx+1][ly][lz];
      else
	if ( cap%3 != 2) {
	  temp = (12+cap-2) % 12;
	  target = E->parallel.loc2proc_map[temp][0][ly][lz];
	}
	else {
	  temp = (12+cap-3) % 12;
	  target = E->parallel.loc2proc_map[temp][ly][nprocy-1][lz];
	}
      E->parallel.PROCESSOR[lev][m].pass[npass] = target;
      face_eqn_node_to_pass(E,lev,m,npass,ii);

      /* front */
      npass = ii = 3;
      if (ly != 0)
	target = E->parallel.loc2proc_map[cap][lx][ly-1][lz];
      else
	if ( cap%3 != 0) {
	  temp = cap-1;
	  target = E->parallel.loc2proc_map[temp][lx][nprocy-1][lz];
	}
	else {
	  temp = (12+cap-3) % 12;
	  target = E->parallel.loc2proc_map[temp][0][lx][lz];
	}

      E->parallel.PROCESSOR[lev][m].pass[npass] = target;
      face_eqn_node_to_pass(E,lev,m,npass,ii);

      /* back */
      npass = ii = 4;
      if (ly != nprocy-1)
	target = E->parallel.loc2proc_map[cap][lx][ly+1][lz];
      else
	if ( cap%3 != 2) {
	  temp = cap+1;
	  target = E->parallel.loc2proc_map[temp][lx][0][lz];
	}
	else {
	  temp = (cap+3) % 12;
	  target = E->parallel.loc2proc_map[temp][nprocx-1][lx][lz];
	}

      E->parallel.PROCESSOR[lev][m].pass[npass] = target;
      face_eqn_node_to_pass(E,lev,m,npass,ii);

      /* do lines parallel to Z */

	/* front-left line */
	if (!( (cap%3==1) && (lx==0) && (ly==0) )) {
	  npass ++;
	  if ((cap%3==0) && (lx==0) && (ly==0)) {
	    temp = (cap+6) % 12;
	    target = E->parallel.loc2proc_map[temp][lx][ly][lz];
	  }
	  else if ((cap%3==0) && (lx==0))
	    target = E->parallel.PROCESSOR[lev][m].pass[1] - nprocz;
	  else if ((cap%3==0) && (ly==0))
	    target = E->parallel.PROCESSOR[lev][m].pass[3] - nprocxz;
	  else
	    target = E->parallel.PROCESSOR[lev][m].pass[1] - nprocxz;

	  E->parallel.PROCESSOR[lev][m].pass[npass] = target;
	  line_eqn_node_to_pass(E,lev,m,npass,noz,1,1);
	}

	/* back-right line */
	if (!( (cap%3==1) && (lx==nprocx-1) && (ly==nprocy-1) )) {
	  npass ++;
	  if ((cap%3==2) && (lx==nprocx-1) && (ly==nprocy-1)) {
	    temp = (cap+6) % 12;
	    target = E->parallel.loc2proc_map[temp][lx][ly][lz];
	  }
	  else if ((cap%3==2) && (lx==nprocx-1))
	    target = E->parallel.PROCESSOR[lev][m].pass[2] + nprocz;
	  else if ((cap%3==2) && (ly==nprocy-1))
	    target = E->parallel.PROCESSOR[lev][m].pass[4] + nprocxz;
	  else
	    target = E->parallel.PROCESSOR[lev][m].pass[2] + nprocxz;

	  E->parallel.PROCESSOR[lev][m].pass[npass] = target;
	  line_eqn_node_to_pass(E,lev,m,npass,noz,(noy*nox-1)*noz+1,1);
	}

	/* back-left line */
	if (!( (cap%3==2 || cap%3==0) && (lx==0) && (ly==nprocy-1) )) {
	  npass ++;
	  if ((cap%3==2) && (ly==nprocy-1))
	    target = E->parallel.PROCESSOR[lev][m].pass[4] - nprocxz;
	  else if ((cap%3==0) && (lx==0))
	    target = E->parallel.PROCESSOR[lev][m].pass[1] + nprocz;
	  else
	    target = E->parallel.PROCESSOR[lev][m].pass[1] + nprocxz;

	  E->parallel.PROCESSOR[lev][m].pass[npass] = target;
	  line_eqn_node_to_pass(E,lev,m,npass,noz,(noy-1)*nox*noz+1,1);
	}

	/* front-right line */
	if (!( (cap%3==2 || cap%3==0) && (lx==nprocx-1) && (ly==0) )) {
	  npass ++;
	  if ((cap%3==2) && (lx==nprocx-1))
	    target = E->parallel.PROCESSOR[lev][m].pass[2] - nprocz;
	  else if ((cap%3==0) && (ly==0))
	    target = E->parallel.PROCESSOR[lev][m].pass[3] + nprocxz;
	  else
	    target = E->parallel.PROCESSOR[lev][m].pass[2] - nprocxz;

	  E->parallel.PROCESSOR[lev][m].pass[npass] = target;
	  line_eqn_node_to_pass(E,lev,m,npass,noz,(nox-1)*noz+1,1);
	}


      E->parallel.TNUM_PASS[lev][m] = npass;

    }   /* end for m  */
  }   /* end for lev  */

  /* determine the communications in vertical direction        */
  for(lev=E->mesh.gridmax;lev>=E->mesh.gridmin;lev--)       {
    kkk = 0;
    for(ii=5;ii<=6;ii++)       {    /* do top & bottom */
      E->parallel.NUM_PASSz[lev].bound[ii] = 1;
      if(lz==0 && ii==5)
	E->parallel.NUM_PASSz[lev].bound[ii] = 0;
      else if(lz==nprocz-1 && ii==6)
	E->parallel.NUM_PASSz[lev].bound[ii] = 0;

      for (p=1;p<=E->parallel.NUM_PASSz[lev].bound[ii];p++)  {
	kkk ++;
	/* determine the pass ID for ii-th boundary and p-th pass */
	kkkp = kkk + E->sphere.max_connections;

	E->parallel.NUM_NODEz[lev].pass[kkk] = 0;
	E->parallel.NUM_NEQz[lev].pass[kkk] = 0;

	for(m=1;m<=E->sphere.caps_per_proc;m++)    {
	  cap = E->sphere.capid[m] - 1;  /* which cap I am in (0~11) */
	  E->parallel.PROCESSORz[lev].pass[kkk] =
	    E->parallel.loc2proc_map[cap][lx][ly][lz+((ii==5)?-1:1)];

	  jj=0;  kk=0;
	  for (k=1;k<=E->parallel.NUM_NNO[lev][m].bound[ii];k++)   {
	    node = E->parallel.NODE[lev][m][k].bound[ii];
	    E->parallel.EXCHANGE_NODE[lev][m][++kk].pass[kkkp] = node;
	    for(doff=1;doff<=dims;doff++)
	      E->parallel.EXCHANGE_ID[lev][m][++jj].pass[kkkp] =
		E->ID[lev][m][node].doff[doff];
	  }
	  E->parallel.NUM_NODE[lev][m].pass[kkkp] = kk;
	  E->parallel.NUM_NEQ[lev][m].pass[kkkp] = jj;
	  E->parallel.NUM_NODEz[lev].pass[kkk] += kk;
	  E->parallel.NUM_NEQz[lev].pass[kkk] += jj;
	}

      }   /* end for loop p */
    }     /* end for j */

    E->parallel.TNUM_PASSz[lev] = kkk;
  }        /* end for level */



  if(E->control.verbose) {
    for(lev=E->mesh.gridmax;lev>=E->mesh.gridmin;lev--) {
      fprintf(E->fp_out,"output_communication route surface for lev=%d \n",lev);
      for (m=1;m<=E->sphere.caps_per_proc;m++)  {
	fprintf(E->fp_out,"  me= %d cap=%d pass  %d \n",E->parallel.me,E->sphere.capid[m],E->parallel.TNUM_PASS[lev][m]);
	for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)   {
	  fprintf(E->fp_out,"proc %d and pass  %d to proc %d with %d eqn and %d node\n",E->parallel.me,k,E->parallel.PROCESSOR[lev][m].pass[k],E->parallel.NUM_NEQ[lev][m].pass[k],E->parallel.NUM_NODE[lev][m].pass[k]);
/* 	  fprintf(E->fp_out,"Eqn:\n");  */
/* 	  for (ii=1;ii<=E->parallel.NUM_NEQ[lev][m].pass[k];ii++)  */
/* 	    fprintf(E->fp_out,"%d %d\n",ii,E->parallel.EXCHANGE_ID[lev][m][ii].pass[k]);  */
/* 	  fprintf(E->fp_out,"Node:\n");  */
/* 	  for (ii=1;ii<=E->parallel.NUM_NODE[lev][m].pass[k];ii++)  */
/* 	    fprintf(E->fp_out,"%d %d\n",ii,E->parallel.EXCHANGE_NODE[lev][m][ii].pass[k]);  */
	}
      }

      fprintf(E->fp_out,"output_communication route vertical \n");
      fprintf(E->fp_out," me= %d pass  %d \n",E->parallel.me,E->parallel.TNUM_PASSz[lev]);
      for (k=1;k<=E->parallel.TNUM_PASSz[lev];k++)   {
	kkkp = k + E->sphere.max_connections;
	fprintf(E->fp_out,"proc %d and pass  %d to proc %d\n",E->parallel.me,k,E->parallel.PROCESSORz[lev].pass[k]);
	for (m=1;m<=E->sphere.caps_per_proc;m++)  {
	  fprintf(E->fp_out,"cap=%d eqn=%d node=%d\n",E->sphere.capid[m],E->parallel.NUM_NEQ[lev][m].pass[kkkp],E->parallel.NUM_NODE[lev][m].pass[kkkp]);
/* 	  for (ii=1;ii<=E->parallel.NUM_NEQ[lev][m].pass[kkkp];ii++) */
/* 	    fprintf(E->fp_out,"%d %d\n",ii,E->parallel.EXCHANGE_ID[lev][m][ii].pass[kkkp]); */
/* 	  for (ii=1;ii<=E->parallel.NUM_NODE[lev][m].pass[kkkp];ii++) */
/* 	    fprintf(E->fp_out,"%d %d\n",ii,E->parallel.EXCHANGE_NODE[lev][m][ii].pass[kkkp]); */
	}
      }
    }
    fflush(E->fp_out);
  }

  return;
  }


/* ============================================
 determine communication routs for
 exchange info across the boundaries on the surfaces
 assuming fault nodes are in the top row of processors
 ============================================ */

void parallel_communication_routs_s(E)
  struct All_variables *E;
  {

  int i,ii,j,k,l,node,el,elt,lnode,jj,doff;
  int lev,nno,nox,noz,noy,kkk,kk,kf;
  int me,m, nprocz;
  void parallel_process_termination();

  const int dims=E->mesh.nsd;

  /* This function is needed only for get_CBF_topo(), */
  /* which is obsolete.             Tan2 Feb. 24 2002 */
  return;


/*   me = E->parallel.me; */
/*   nprocz = E->parallel.nprocz; */

        /* determine the communications in horizontal direction        */
/*   for(lev=E->mesh.gridmax;lev>=E->mesh.gridmin;lev--)       { */
/*     nox = E->lmesh.NOX[lev]; */
/*     noz = E->lmesh.NOZ[lev]; */
/*     noy = E->lmesh.NOY[lev]; */

/*     for(m=1;m<=E->sphere.caps_per_proc;m++)    { */
/*       j = E->sphere.capid[m]; */

/*       for (kkk=1;kkk<=E->parallel.TNUM_PASS[lev][m];kkk++) { */
/*         if (kkk<=4) { */  /* first 4 communications are for planes */
/*           ii = kkk; */
/*           E->parallel.NUM_sNODE[lev][m].pass[kkk] =  */
/*                            E->parallel.NUM_NNO[lev][m].bound[ii]/noz; */

/*           for (k=1;k<=E->parallel.NUM_sNODE[lev][m].pass[kkk];k++)   { */
/*             lnode = k; */
/*             node = (E->parallel.NODE[lev][m][lnode].bound[ii]-1)/noz + 1; */
/*             E->parallel.EXCHANGE_sNODE[lev][m][k].pass[kkk] = node; */
/*             }  */ /* end for node k */
/*           }    */   /* end for first 4 communications */

/*         else  {    */      /* the last FOUR communications are for lines */
/*           E->parallel.NUM_sNODE[lev][m].pass[kkk]=1; */
/*           for (k=1;k<=E->parallel.NUM_sNODE[lev][m].pass[kkk];k++)   { */
/* 	    if (E->parallel.nprocx*E->parallel.nprocy > 1) { */ /* 4 or more horiz. proc*/
/* 	      switch(kkk) { */
/* 	      case 5: */
/* 		ii = 1; */
/* 		lnode = k; */
/* 		break; */
/* 	      case 6: */
/* 		ii = 2; */
/* 		lnode = k + (noy-1)*noz*noy; */
/* 		break; */
/* 	      case 7: */
/* 		ii = 1; */
/* 		lnode = k + (noy-1)*noz*noy; */
/* 		break; */
/* 	      case 8: */
/* 		ii = 2; */
/* 		lnode = k; */
/* 		break; */
/* 	      } */
/* 	    } */
/* 	    else { */ /* 1 or fewer horiz. processors per cap */
/* 	      ii = kkk-4; */
/* 	      if (j%3==2)  */       /* for the middle caps */
/* 		lnode = k + ((ii==1)?1:0)*(noy-1)*noz*noy; */
/* 	      else    */            /* for the caps linked to poles */
/* 		lnode = k + ((ii==1)?0:1)*(noy-1)*noz*noy; */
/* 	    } */



/*             node = (E->parallel.NODE[lev][m][lnode].bound[ii]-1)/noz + 1; */
/*             E->parallel.EXCHANGE_sNODE[lev][m][k].pass[kkk] = node; */
/*             } */  /* end for node k */
/*           } */  /* end for the last FOUR communications */

/*         }   */ /* end for kkk  */
/*       }    *//* end for m  */

/*     }   */ /* end for lev  */

/*   return; */
  }



/* ================================================ */
/* ================================================ */

void face_eqn_node_to_pass(E,lev,m,npass,bd)
  struct All_variables *E;
  int lev,m,npass,bd;
{
  int jj,kk,node,doff;
  const int dims=E->mesh.nsd;

  E->parallel.NUM_NODE[lev][m].pass[npass] = E->parallel.NUM_NNO[lev][m].bound[bd];

  jj = 0;
  for (kk=1;kk<=E->parallel.NUM_NODE[lev][m].pass[npass];kk++)   {
    node = E->parallel.NODE[lev][m][kk].bound[bd];
    E->parallel.EXCHANGE_NODE[lev][m][kk].pass[npass] = node;
    for(doff=1;doff<=dims;doff++)
      E->parallel.EXCHANGE_ID[lev][m][++jj].pass[npass] = E->ID[lev][m][node].doff[doff];
  }

  E->parallel.NUM_NEQ[lev][m].pass[npass] = jj;

  return;
}

/* ================================================ */
/* ================================================ */

void line_eqn_node_to_pass(E,lev,m,npass,num_node,offset,stride)
  struct All_variables *E;
  int lev,m,npass,num_node,offset,stride;
{
  int jj,kk,node,doff;
  const int dims=E->mesh.nsd;

  E->parallel.NUM_NODE[lev][m].pass[npass] = num_node;

  jj=0;
  for (kk=1;kk<=E->parallel.NUM_NODE[lev][m].pass[npass];kk++)   {
    node = (kk-1)*stride + offset;
    E->parallel.EXCHANGE_NODE[lev][m][kk].pass[npass] = node;
    for(doff=1;doff<=dims;doff++)
      E->parallel.EXCHANGE_ID[lev][m][++jj].pass[npass] = E->ID[lev][m][node].doff[doff];
  }

  E->parallel.NUM_NEQ[lev][m].pass[npass] = jj;

  return;
}

/* ================================================
WARNING: BUGS AHEAD

   for (m=1;m<=E->sphere.caps_per_proc;m++)    {
     for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)  {

       sizeofk = (1+E->parallel.NUM_NEQ[lev][m].pass[k])*sizeof(double);
       S[k]=(double *)malloc( sizeofk );
       R[k]=(double *)malloc( sizeofk );
       }
      }

This piece of code contain a bug. Arrays S and R are allocated for each m.
But most of the memory is leaked.

In this version of CitcomS, sphere.caps_per_proc is always equal to one.
So, this bug won't manifest itself. But in other version of CitcomS, it will.

by Tan2 7/21, 2003
================================================ */

void exchange_id_d(E, U, lev)
 struct All_variables *E;
 double **U;
 int lev;
 {

 int ii,j,jj,m,k,kk,t_cap,idb,msginfo[8];
 double *S[73],*R[73], *RV, *SV;
 int mid_recv, sizeofk;

 MPI_Status status[100];
 MPI_Status status1;
 MPI_Request request[100];

 for (m=1;m<=E->sphere.caps_per_proc;m++)    {
   for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)  {
     sizeofk = (1+E->parallel.NUM_NEQ[lev][m].pass[k])*sizeof(double);
     S[k]=(double *)malloc( sizeofk );
     R[k]=(double *)malloc( sizeofk );
   }
 }

 sizeofk = 0;
 for (k=1;k<=E->parallel.TNUM_PASSz[lev];k++)  {
   kk = (1+E->parallel.NUM_NEQz[lev].pass[k])*sizeof(double);
   sizeofk = max(sizeofk, kk);
 }
 RV=(double *)malloc( sizeofk );
 SV=(double *)malloc( sizeofk );

  idb=0;
  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)     {

      for (j=1;j<=E->parallel.NUM_NEQ[lev][m].pass[k];j++) {
        S[k][j-1] = U[m][ E->parallel.EXCHANGE_ID[lev][m][j].pass[k] ];
	}

      if (E->parallel.PROCESSOR[lev][m].pass[k] != E->parallel.me &&
	  E->parallel.PROCESSOR[lev][m].pass[k] != -1) {
	  idb ++;
          MPI_Isend(S[k], E->parallel.NUM_NEQ[lev][m].pass[k], MPI_DOUBLE,
		    E->parallel.PROCESSOR[lev][m].pass[k], 1,
		    E->parallel.world, &request[idb-1]);
      }
    }           /* for k */
  }     /* for m */         /* finish sending */

  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)   {

      if (E->parallel.PROCESSOR[lev][m].pass[k] != E->parallel.me &&
	  E->parallel.PROCESSOR[lev][m].pass[k] != -1) {
         idb++;
	 MPI_Irecv(R[k],E->parallel.NUM_NEQ[lev][m].pass[k], MPI_DOUBLE,
		   E->parallel.PROCESSOR[lev][m].pass[k], 1,
		   E->parallel.world, &request[idb-1]);
      }
      else {
	for (j=1;j<=E->parallel.NUM_NEQ[lev][m].pass[k];j++)
           U[m][ E->parallel.EXCHANGE_ID[lev][m][j].pass[k] ] += S[k][j-1];
      }
    }      /* for k */
  }     /* for m */         /* finish receiving */

  MPI_Waitall(idb,request,status);

  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)   {

      if (E->parallel.PROCESSOR[lev][m].pass[k] != E->parallel.me &&
	  E->parallel.PROCESSOR[lev][m].pass[k] != -1) {
	for (j=1;j<=E->parallel.NUM_NEQ[lev][m].pass[k];j++)
	  U[m][ E->parallel.EXCHANGE_ID[lev][m][j].pass[k] ] += R[k][j-1];
      }
    }
  }

  /* for vertical direction  */

  for (k=1;k<=E->parallel.TNUM_PASSz[lev];k++)  {
    jj = 0;
    kk = k + E->sphere.max_connections;

    for(m=1;m<=E->sphere.caps_per_proc;m++)
      for (j=1;j<=E->parallel.NUM_NEQ[lev][m].pass[kk];j++)
        SV[jj++] = U[m][ E->parallel.EXCHANGE_ID[lev][m][j].pass[kk] ];

    MPI_Sendrecv(SV, E->parallel.NUM_NEQz[lev].pass[k], MPI_DOUBLE,
		 E->parallel.PROCESSORz[lev].pass[k], 1,
                 RV, E->parallel.NUM_NEQz[lev].pass[k], MPI_DOUBLE,
		 E->parallel.PROCESSORz[lev].pass[k], 1,
		 E->parallel.world, &status1);

    jj = 0;
    for(m=1;m<=E->sphere.caps_per_proc;m++)
      for (j=1;j<=E->parallel.NUM_NEQ[lev][m].pass[kk];j++)
        U[m][ E->parallel.EXCHANGE_ID[lev][m][j].pass[kk] ] += RV[jj++];
  }

 for (m=1;m<=E->sphere.caps_per_proc;m++)    {
   for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)  {
     free((void*) S[k]);
     free((void*) R[k]);
   }
 }

 free((void*) SV);
 free((void*) RV);

 return;
 }


/* ================================================ */
/* ================================================ */
void exchange_node_d(E, U, lev)
 struct All_variables *E;
 double **U;
 int lev;
 {

 int ii,j,jj,m,k,kk,t_cap,idb,msginfo[8];
 double *S[73],*R[73], *RV, *SV;
 int mid_recv, sizeofk;

 MPI_Status status[100];
 MPI_Status status1;
 MPI_Request request[100];

 kk=0;
 for (m=1;m<=E->sphere.caps_per_proc;m++)    {
   for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)  {
     ++kk;
     sizeofk = (1+E->parallel.NUM_NODE[lev][m].pass[k])*sizeof(double);
     S[kk]=(double *)malloc( sizeofk );
     R[kk]=(double *)malloc( sizeofk );
   }
 }

 idb= 0;
 for (k=1;k<=E->parallel.TNUM_PASSz[lev];k++)  {
   sizeofk = (1+E->parallel.NUM_NODEz[lev].pass[k])*sizeof(double);
   idb = max(idb,sizeofk);
 }

 RV=(double *)malloc( idb );
 SV=(double *)malloc( idb );

  idb=0;
  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)     {
      kk=k;

      for (j=1;j<=E->parallel.NUM_NODE[lev][m].pass[k];j++)
        S[kk][j-1] = U[m][ E->parallel.EXCHANGE_NODE[lev][m][j].pass[k] ];

      if (E->parallel.PROCESSOR[lev][m].pass[k]!=E->parallel.me) {
	if (E->parallel.PROCESSOR[lev][m].pass[k]!=-1) {
         idb ++;
        MPI_Isend(S[kk],E->parallel.NUM_NODE[lev][m].pass[k],MPI_DOUBLE,
             E->parallel.PROCESSOR[lev][m].pass[k],1,E->parallel.world,&request[idb-1]);
	}
         }
      }           /* for k */
    }     /* for m */         /* finish sending */

  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)   {
      kk=k;

      if (E->parallel.PROCESSOR[lev][m].pass[k]!=E->parallel.me)  {
	if (E->parallel.PROCESSOR[lev][m].pass[k]!=-1) {
         idb++;
         MPI_Irecv(R[kk],E->parallel.NUM_NODE[lev][m].pass[k],MPI_DOUBLE,
         E->parallel.PROCESSOR[lev][m].pass[k],1,E->parallel.world,&request[idb-1]);
         }
      }

      else   {
	kk=k;
         for (j=1;j<=E->parallel.NUM_NODE[lev][m].pass[k];j++)
           U[m][ E->parallel.EXCHANGE_NODE[lev][m][j].pass[k] ] += S[kk][j-1];
         }
      }      /* for k */
    }     /* for m */         /* finish receiving */

  MPI_Waitall(idb,request,status);

  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)   {
      kk=k;

      if (E->parallel.PROCESSOR[lev][m].pass[k]!=E->parallel.me)
	if (E->parallel.PROCESSOR[lev][m].pass[k]!=-1) {
        for (j=1;j<=E->parallel.NUM_NODE[lev][m].pass[k];j++)
           U[m][ E->parallel.EXCHANGE_NODE[lev][m][j].pass[k] ] += R[kk][j-1];
      }
    }
    }

                /* for vertical direction  */

  for (k=1;k<=E->parallel.TNUM_PASSz[lev];k++)  {
    jj = 0;
    kk = k + E->sphere.max_connections;

    for(m=1;m<=E->sphere.caps_per_proc;m++)
      for (j=1;j<=E->parallel.NUM_NODE[lev][m].pass[kk];j++)
        SV[jj++] = U[m][ E->parallel.EXCHANGE_NODE[lev][m][j].pass[kk] ];

    MPI_Sendrecv(SV,E->parallel.NUM_NODEz[lev].pass[k],MPI_DOUBLE,
             E->parallel.PROCESSORz[lev].pass[k],1,
                 RV,E->parallel.NUM_NODEz[lev].pass[k],MPI_DOUBLE,
             E->parallel.PROCESSORz[lev].pass[k],1,E->parallel.world,&status1);

    jj = 0;
    for(m=1;m<=E->sphere.caps_per_proc;m++)
      for (j=1;j<=E->parallel.NUM_NODE[lev][m].pass[kk];j++)
        U[m][ E->parallel.EXCHANGE_NODE[lev][m][j].pass[kk] ] += RV[jj++];
    }

  kk = 0;
 for (m=1;m<=E->sphere.caps_per_proc;m++)    {
   for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)  {
     kk++;
     free((void*) S[kk]);
     free((void*) R[kk]);
   }
 }

 free((void*) SV);
 free((void*) RV);

 return;
}

/* ================================================ */
/* ================================================ */

void exchange_node_f(E, U, lev)
 struct All_variables *E;
 float **U;
 int lev;
 {

 int ii,j,jj,m,k,kk,t_cap,idb,msginfo[8];

 float *S[73],*R[73], *RV, *SV;
 int mid_recv, sizeofk;

 MPI_Status status[100];
 MPI_Status status1;
 MPI_Request request[100];

 kk=0;
 for (m=1;m<=E->sphere.caps_per_proc;m++)    {
   for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)  {
     ++kk;
     sizeofk = (1+E->parallel.NUM_NODE[lev][m].pass[k])*sizeof(float);
     S[kk]=(float *)malloc( sizeofk );
     R[kk]=(float *)malloc( sizeofk );
   }
 }

 idb= 0;
 for (k=1;k<=E->parallel.TNUM_PASSz[lev];k++)  {
   sizeofk = (1+E->parallel.NUM_NODEz[lev].pass[k])*sizeof(float);
   idb = max(idb,sizeofk);
 }

 RV=(float *)malloc( idb );
 SV=(float *)malloc( idb );

  idb=0;
  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)     {
      kk=k;

      for (j=1;j<=E->parallel.NUM_NODE[lev][m].pass[k];j++)
        S[kk][j-1] = U[m][ E->parallel.EXCHANGE_NODE[lev][m][j].pass[k] ];

      if (E->parallel.PROCESSOR[lev][m].pass[k]!=E->parallel.me) {
	if (E->parallel.PROCESSOR[lev][m].pass[k]!=-1) {
         idb ++;
        MPI_Isend(S[kk],E->parallel.NUM_NODE[lev][m].pass[k],MPI_FLOAT,
             E->parallel.PROCESSOR[lev][m].pass[k],1,E->parallel.world,&request[idb-1]);
	}
         }
      }           /* for k */
    }     /* for m */         /* finish sending */

  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)   {
      kk=k;

      if (E->parallel.PROCESSOR[lev][m].pass[k]!=E->parallel.me)  {
	if (E->parallel.PROCESSOR[lev][m].pass[k]!=-1) {
         idb++;
         MPI_Irecv(R[kk],E->parallel.NUM_NODE[lev][m].pass[k],MPI_FLOAT,
         E->parallel.PROCESSOR[lev][m].pass[k],1,E->parallel.world,&request[idb-1]);
         }
      }

      else   {
	kk=k;
         for (j=1;j<=E->parallel.NUM_NODE[lev][m].pass[k];j++)
           U[m][ E->parallel.EXCHANGE_NODE[lev][m][j].pass[k] ] += S[kk][j-1];
         }
      }      /* for k */
    }     /* for m */         /* finish receiving */

  MPI_Waitall(idb,request,status);

  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)   {
      kk=k;

      if (E->parallel.PROCESSOR[lev][m].pass[k]!=E->parallel.me)
	if (E->parallel.PROCESSOR[lev][m].pass[k]!=-1) {
        for (j=1;j<=E->parallel.NUM_NODE[lev][m].pass[k];j++)
           U[m][ E->parallel.EXCHANGE_NODE[lev][m][j].pass[k] ] += R[kk][j-1];
      }
    }
    }

                /* for vertical direction  */

  for (k=1;k<=E->parallel.TNUM_PASSz[lev];k++)  {
    jj = 0;
    kk = k + E->sphere.max_connections;

    for(m=1;m<=E->sphere.caps_per_proc;m++)
      for (j=1;j<=E->parallel.NUM_NODE[lev][m].pass[kk];j++)
        SV[jj++] = U[m][ E->parallel.EXCHANGE_NODE[lev][m][j].pass[kk] ];

    MPI_Sendrecv(SV,E->parallel.NUM_NODEz[lev].pass[k],MPI_FLOAT,
             E->parallel.PROCESSORz[lev].pass[k],1,
                 RV,E->parallel.NUM_NODEz[lev].pass[k],MPI_FLOAT,
             E->parallel.PROCESSORz[lev].pass[k],1,E->parallel.world,&status1);

    jj = 0;
    for(m=1;m<=E->sphere.caps_per_proc;m++)
      for (j=1;j<=E->parallel.NUM_NODE[lev][m].pass[kk];j++)
        U[m][ E->parallel.EXCHANGE_NODE[lev][m][j].pass[kk] ] += RV[jj++];
    }

  kk = 0;
 for (m=1;m<=E->sphere.caps_per_proc;m++)    {
   for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)  {
     kk++;
     free((void*) S[kk]);
     free((void*) R[kk]);
   }
 }

 free((void*) SV);
 free((void*) RV);

 return;
 }
/* ================================================ */
/* ================================================ */

void exchange_snode_f(E, U1, U2, lev)
 struct All_variables *E;
 float **U1,**U2;
 int lev;
 {

 int ii,j,k,m,kk,t_cap,idb,msginfo[8];
 float *S[73],*R[73];
 int mid_recv, sizeofk;

 MPI_Status status[100];
 MPI_Status status1;
 MPI_Request request[100];

   kk=0;
   for (m=1;m<=E->sphere.caps_per_proc;m++)    {
     for (k=1;k<=E->parallel.TNUM_PASS[E->mesh.levmax][m];k++)  {
       ++kk;
       sizeofk = (1+2*E->parallel.NUM_sNODE[E->mesh.levmax][m].pass[k])*sizeof(float);
       S[kk]=(float *)malloc( sizeofk );
       R[kk]=(float *)malloc( sizeofk );
       }
     }

  idb=0;
  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)     {
      kk=k;

      for (j=1;j<=E->parallel.NUM_sNODE[lev][m].pass[k];j++)  {
        S[kk][j-1] = U1[m][ E->parallel.EXCHANGE_sNODE[lev][m][j].pass[k] ];
        S[kk][j-1+E->parallel.NUM_sNODE[lev][m].pass[k]]
                   = U2[m][ E->parallel.EXCHANGE_sNODE[lev][m][j].pass[k] ];
        }

      if (E->parallel.PROCESSOR[lev][m].pass[k]!=E->parallel.me) {
	if (E->parallel.PROCESSOR[lev][m].pass[k]!=-1) {
         idb ++;
         MPI_Isend(S[kk],2*E->parallel.NUM_sNODE[lev][m].pass[k],MPI_FLOAT,
             E->parallel.PROCESSOR[lev][m].pass[k],1,E->parallel.world,&request[idb-1]);
         }
      }
      }           /* for k */
    }     /* for m */         /* finish sending */

  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)   {
      kk=k;

      if (E->parallel.PROCESSOR[lev][m].pass[k]!=E->parallel.me)  {
	if (E->parallel.PROCESSOR[lev][m].pass[k]!=-1) {

         idb ++;
         MPI_Irecv(R[kk],2*E->parallel.NUM_sNODE[lev][m].pass[k],MPI_FLOAT,
           E->parallel.PROCESSOR[lev][m].pass[k],1,E->parallel.world,&request[idb-1]);
         }
      }

      else   {
	kk=k;
         for (j=1;j<=E->parallel.NUM_sNODE[lev][m].pass[k];j++)     {
           U1[m][ E->parallel.EXCHANGE_sNODE[lev][m][j].pass[k] ] += S[kk][j-1];
           U2[m][ E->parallel.EXCHANGE_sNODE[lev][m][j].pass[k] ] +=
                               S[kk][j-1+E->parallel.NUM_sNODE[lev][m].pass[k]];
           }
         }
      }      /* for k */
    }     /* for m */         /* finish receiving */

  MPI_Waitall(idb,request,status);

  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)   {
      kk=k;

      if (E->parallel.PROCESSOR[lev][m].pass[k]!=E->parallel.me)
	if (E->parallel.PROCESSOR[lev][m].pass[k]!=-1) {
        for (j=1;j<=E->parallel.NUM_sNODE[lev][m].pass[k];j++)    {
           U1[m][ E->parallel.EXCHANGE_sNODE[lev][m][j].pass[k] ] += R[kk][j-1];
           U2[m][ E->parallel.EXCHANGE_sNODE[lev][m][j].pass[k] ] +=
                              R[kk][j-1+E->parallel.NUM_sNODE[lev][m].pass[k]];
           }
	}
      }
  }

  kk=0;
  for (m=1;m<=E->sphere.caps_per_proc;m++)    {
    for (k=1;k<=E->parallel.TNUM_PASS[E->mesh.levmax][m];k++)  {
      ++kk;
      free((void*) S[kk]);
      free((void*) R[kk]);
    }
  }

 return;
 }


