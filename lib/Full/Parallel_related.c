/* Routines here are for intel paragon with MPI */

#include <mpi.h>

#include <math.h>
#include "element_definitions.h"
#include "global_defs.h"
#include "sphere_communication.h"
#include "citcom_init.h"
#include "parallel_related.h"

extern void set_horizontal_communicator(struct All_variables*);
extern void set_vertical_communicator(struct All_variables*);


void parallel_process_initilization(E,argc,argv)
  struct All_variables *E;
  int argc;
  char **argv;
  {

  E->parallel.me = 0;
  E->parallel.nproc = 1;
  E->parallel.me_loc[1] = 0;
  E->parallel.me_loc[2] = 0;
  E->parallel.me_loc[3] = 0;

  /*  MPI_Init(&argc,&argv); moved to main{} in Citcom.c, CPC 6/16/00 */
  MPI_Comm_rank(E->parallel.world, &(E->parallel.me) );
  MPI_Comm_size(E->parallel.world, &(E->parallel.nproc) );

  return;
  }

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

void parallel_process_sync()
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

void parallel_processor_setup(E)
  struct All_variables *E;
  {
  
  int i,j,k,m,me,temp,pid_surf;
  int cap_id_surf;
  void parallel_process_termination();

  me = E->parallel.me;

  if ( E->parallel.nprocx != E->parallel.nprocy ) {
    if (E->parallel.me==0) fprintf(stderr,"!!!! nprocx must equal to nprocy \n");
    parallel_process_termination();
  }

  E->parallel.total_proc = E->parallel.nprocxy*E->parallel.nprocx*E->parallel.nprocy*E->parallel.nprocz;
  E->parallel.surf_proc_per_cap = E->parallel.nprocx*E->parallel.nprocy;
  E->sphere.total_sub_caps = E->sphere.caps*E->parallel.surf_proc_per_cap;
  E->parallel.proc_per_cap = E->parallel.nprocx*E->parallel.nprocy*E->parallel.nprocz;

  if ( E->parallel.total_proc != E->parallel.nproc ) {
    if (E->parallel.me==0) fprintf(stderr,"!!!! # of requested CPU is incorrect \n");
    parallel_process_termination();
    }

  E->sphere.caps_per_proc = max(1,E->sphere.caps/E->parallel.nprocxy);

  if ( (E->sphere.caps_per_proc > 1) && (E->parallel.surf_proc_per_cap > 1) ) {
    if (E->parallel.me==0) fprintf(stderr,"!!!! Shouldn't have # surface proc per cap > 1 if # caps per proc > 1\n \n");
    parallel_process_termination();
  }

  if (E->sphere.caps_per_proc > 1) {
    if (E->parallel.me==0) fprintf(stderr,"!!!! # caps per proc > 1 is not fully tested yet.\n \n");
    parallel_process_termination();
  }

  /* determine the location of processors in each cap */
  cap_id_surf = me / E->parallel.proc_per_cap;

  j = me % E->parallel.nprocz;

 /* z-direction first*/
  E->parallel.me_loc[3] = (me - cap_id_surf*E->parallel.proc_per_cap) % E->parallel.nprocz;

 /* x-direction then*/
  E->parallel.me_loc[1] = ((me - cap_id_surf*E->parallel.proc_per_cap - E->parallel.me_loc[3])/E->parallel.nprocz) % E->parallel.nprocx;

 /* y-direction then*/
  E->parallel.me_loc[2] = ((((me - cap_id_surf*E->parallel.proc_per_cap - E->parallel.me_loc[3])/E->parallel.nprocz) - E->parallel.me_loc[1])/E->parallel.nprocx) % E->parallel.nprocy;


/*the numbering of proc in each caps is as so (example for an xyz = 2x2x2 box):
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

  pid_surf = me/E->parallel.proc_per_cap; /* cap number (0~11) */
/*  pid_surf = me/E->parallel.nprocz; CPC 7/27/00 */
  i = cases[E->sphere.caps_per_proc]; /* 1 for more than 12 processors */

  for (j=1;j<=E->sphere.caps_per_proc;j++)  {
    temp = pid_surf*E->sphere.caps_per_proc + j-1; /* cap number (out of 12) */
    E->sphere.capid[j] = incases1[i].links[temp]; /* id (1~12) of the current cap */
    }

  /* determine which caps are linked with each of 12 caps  */
  /* if the 12 caps are broken, set these up instead */
  if (E->parallel.surf_proc_per_cap > 1) {
/*      E->sphere.caps = E->parallel.surf_proc_per_cap*12; */
     E->sphere.max_connections = 8;
  }

  /* steup location-to-processor map */

  E->parallel.loc2proc_map = (int ****) malloc(12*sizeof(int ***));
  for (m=0;m<12;m++)  {
    E->parallel.loc2proc_map[m] = (int ***) malloc(E->parallel.nprocx*sizeof(int **));
    for (i=0;i<E->parallel.nprocx;i++) {
      E->parallel.loc2proc_map[m][i] = (int **) malloc(E->parallel.nprocy*sizeof(int *));
      for (j=0;j<E->parallel.nprocy;j++) 
	E->parallel.loc2proc_map[m][i][j] = (int *) malloc(E->parallel.nprocz*sizeof(int));
    }
  }

  for (m=0;m<12;m++) 
    for (i=0;i<E->parallel.nprocx;i++)
      for (j=0;j<E->parallel.nprocy;j++) 
	for (k=0;k<E->parallel.nprocz;k++) {
	  if (E->sphere.caps_per_proc>1) {
	    temp = cases[E->sphere.caps_per_proc];
	    E->parallel.loc2proc_map[m][i][j][k] = incases2[temp].links[m-1];
	  }
	  else
	    E->parallel.loc2proc_map[m][i][j][k] = m*E->parallel.proc_per_cap
	      + j*E->parallel.nprocx*E->parallel.nprocz
	      + i*E->parallel.nprocz + k;
	}

          /* determine surface proc id for each cap  */
  i = cases[E->sphere.caps_per_proc]; /* =1 for proc >= 12 */
/*   for (j=1;j<=E->sphere.caps;j++)     { */
/*     if (E->parallel.surf_proc_per_cap > 1) */
/*       E->sphere.pid_surf[j] = j-1; */
/*     else */
/*       E->sphere.pid_surf[j] = incases2[i].links[j-1]; */
/*     } */
/*   E->sphere.pid_surf[0] = -1;  */
  /* used to indicate that this processor link
     should NOT be used!! CPC 8/25/00 */

/* still needs to be changed? I am pretty sure not. CPC 8/24/00 */


/** E->parallel.mst never used, commented out by Tan2 2/23/02 **/
/*   temp = 1; */
/*   for (j=1;j<=E->sphere.caps;j++)    */
/*     for (i=1;i<=j;i++)   */
/*       if (i!=j) */
/*         E->parallel.mst[j][i] = temp++;    */

/*   for (j=1;j<=E->sphere.caps;j++)    */
/*     for (i=1;i<=E->sphere.caps;i++)   */
/*       if (i>j) */
/*         E->parallel.mst[j][i] = E->parallel.mst[i][j];    */



  if (E->control.verbose) {
    fprintf(E->fp_out,"me=%d loc1=%d loc2=%d loc3=%d\n",me,E->parallel.me_loc[1],E->parallel.me_loc[2],E->parallel.me_loc[3]);
    for (j=1;j<=E->sphere.caps_per_proc;j++) {
      fprintf(E->fp_out,"capid[%d]=%d \n",j,E->sphere.capid[j]);
    }
    for (m=0;m<12;m++) 
      for (i=0;i<E->parallel.nprocx;i++)
	for (j=0;j<E->parallel.nprocy;j++) 
	  for (k=0;k<E->parallel.nprocz;k++) 
	    fprintf(E->fp_out,"loc2proc_map[cap=%d][x=%d][y=%d][z=%d] = %d\n",
		    m,i,j,k,E->parallel.loc2proc_map[m][i][j][k]);

    fflush(E->fp_out);
  }
/*   parallel_process_termination(); */


  return;
  }


/* =========================================================================
get element information for each processor. 
 ========================================================================= */

void parallel_domain_decomp0(E)
  struct All_variables *E;
  {
  
  int iy,eelx,i,j,k,nox,noz,noy,me,nel,lev,eaccumulatives,naccumulatives;
  int temp0,temp1;
  FILE *fp;
  char output_file[255];

  void parallel_process_termination();

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
 get numerical grid coordinates for each relevant processor 
 ============================================ */

void parallel_domain_decomp2(E,GX)
  struct All_variables *E;
  float *GX[4];
  {
  
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

 ii=0;
 for (m=1;m<=E->sphere.caps_per_proc;m++)
    for (node=1;node<=E->lmesh.nno;node++)
      if(E->node[m][node] & SKIPS)
        ii+=1; 

 MPI_Allreduce(&ii, &node  ,1,MPI_INT,MPI_SUM,E->parallel.world);

 E->mesh.nno = E->lmesh.nno*E->parallel.total_proc - node - 2*E->mesh.noz;
 /* the above line changed to accomodate multiple proc. per cap CPC 10/18/00 */
 /*E->mesh.nno -= node + 2*E->mesh.noz; */
 E->mesh.neq = E->mesh.nno*3;

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

  void face_eqn_node_to_pass();
  void line_eqn_node_to_pass();
  void parallel_process_termination();

  const int dims=E->mesh.nsd;

  me = E->parallel.me;
  nprocx = E->parallel.nprocx;
  nprocy = E->parallel.nprocy;
  nprocz = E->parallel.nprocz;
  nprocxz = nprocx * nprocz;
  tscaps = E->sphere.total_sub_caps;
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
/* 	    if (E->parallel.surf_proc_per_cap > 1) { */ /* 4 or more horiz. proc*/
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

/* ================================================ */
/* ================================================ */

void exchange_id_d(E, U, lev)
 struct All_variables *E;
 double **U;
 int lev;
 {

 int ii,j,jj,m,k,kk,t_cap,idb,msginfo[8];

 /*static int idd[NCS][NCS];*/
 static double *S[73],*R[73], *RV, *SV;

 static int been_here = 0;
 static int mid_recv, sizeofk;

 MPI_Status status[100];
 MPI_Status status1;
 MPI_Request request[100];

 if (been_here ==0 )   {
   kk=0;
   for (m=1;m<=E->sphere.caps_per_proc;m++)    {
     ii = E->sphere.capid[m];
     for (k=1;k<=E->parallel.TNUM_PASS[E->mesh.levmax][m];k++)  {
       ++kk;
       /*       t_cap = E->sphere.cap[ii].connection[k];
		idd[ii][t_cap] = kk;*/

       sizeofk = (1+E->parallel.NUM_NEQ[E->mesh.levmax][m].pass[k])*sizeof(double);
       S[kk]=(double *)malloc( sizeofk );
       R[kk]=(double *)malloc( sizeofk );
       }
     }

   idb= 0;
   for (k=1;k<=E->parallel.TNUM_PASSz[E->mesh.levmax];k++)  {
      sizeofk = (1+E->parallel.NUM_NEQz[E->mesh.levmax].pass[k])*sizeof(double);
      idb = max(idb,sizeofk);
      }

   RV=(double *)malloc( idb );
   SV=(double *)malloc( idb );

   been_here ++;
   }

   fflush(E->fp_out);

  idb=0;
  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    ii = E->sphere.capid[m];

    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)     {
      /*      t_cap = E->sphere.cap[ii].connection[k];
	      kk = idd[ii][t_cap];*/
      kk=k;

      for (j=1;j<=E->parallel.NUM_NEQ[lev][m].pass[k];j++) {
        S[kk][j-1] = U[m][ E->parallel.EXCHANGE_ID[lev][m][j].pass[k] ];
	}

      if (E->parallel.PROCESSOR[lev][m].pass[k]!=E->parallel.me) {
	if (E->parallel.PROCESSOR[lev][m].pass[k]!=-1) {
         idb ++;
	 /*         MPI_Isend(S[kk],E->parallel.NUM_NEQ[lev][m].pass[k],MPI_DOUBLE,
		    E->parallel.PROCESSOR[lev][m].pass[k],E->parallel.mst[ii][t_cap],E->parallel.world,&request[idb-1]);*/
          MPI_Isend(S[kk],E->parallel.NUM_NEQ[lev][m].pass[k],MPI_DOUBLE,
          E->parallel.PROCESSOR[lev][m].pass[k],1,E->parallel.world,&request[idb-1]); 
         }
      }
      }           /* for k */
    }     /* for m */         /* finish sending */

  /*MPI_Waitall(idb,request,status);

  idb = 0; */
  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    ii = E->sphere.capid[m];

    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)   {
      /*      t_cap = E->sphere.cap[ii].connection[k];
	      kk = idd[ii][t_cap];*/
      kk=k;

      if (E->parallel.PROCESSOR[lev][m].pass[k]!=E->parallel.me)  {
	if (E->parallel.PROCESSOR[lev][m].pass[k]!=-1) {

         idb++;
	 /*         MPI_Irecv(R[kk],E->parallel.NUM_NEQ[lev][m].pass[k],MPI_DOUBLE,
		    E->parallel.PROCESSOR[lev][m].pass[k],E->parallel.mst[ii][t_cap],E->parallel.world,&request[idb-1]);*/
	 MPI_Irecv(R[kk],E->parallel.NUM_NEQ[lev][m].pass[k],MPI_DOUBLE,
	   E->parallel.PROCESSOR[lev][m].pass[k],1,E->parallel.world,&request[idb-1]);
	}
         }

      else   {
	/*         kk = idd[t_cap][ii];*/
	kk=k;
         for (j=1;j<=E->parallel.NUM_NEQ[lev][m].pass[k];j++)
           U[m][ E->parallel.EXCHANGE_ID[lev][m][j].pass[k] ] += S[kk][j-1];
      }
      }      /* for k */
    }     /* for m */         /* finish receiving */

  MPI_Waitall(idb,request,status);

  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    ii = E->sphere.capid[m];

    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)   {
      /*      t_cap = E->sphere.cap[ii].connection[k];
	      kk = idd[ii][t_cap];*/
      kk=k;

      if (E->parallel.PROCESSOR[lev][m].pass[k]!=E->parallel.me)
	if (E->parallel.PROCESSOR[lev][m].pass[k]!=-1) {
        for (j=1;j<=E->parallel.NUM_NEQ[lev][m].pass[k];j++)
           U[m][ E->parallel.EXCHANGE_ID[lev][m][j].pass[k] ] += R[kk][j-1];
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

    MPI_Sendrecv(SV,E->parallel.NUM_NEQz[lev].pass[k],MPI_DOUBLE,
             E->parallel.PROCESSORz[lev].pass[k],1,
                 RV,E->parallel.NUM_NEQz[lev].pass[k],MPI_DOUBLE,
             E->parallel.PROCESSORz[lev].pass[k],1,E->parallel.world,&status1);

    jj = 0;
    for(m=1;m<=E->sphere.caps_per_proc;m++)    
      for (j=1;j<=E->parallel.NUM_NEQ[lev][m].pass[kk];j++)  
        U[m][ E->parallel.EXCHANGE_ID[lev][m][j].pass[kk] ] += RV[jj++];
    }

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

 /*static int idd[NCS][NCS];*/
 static double *S[73],*R[73], *RV, *SV;

 static int been_here = 0;
 static int mid_recv, sizeofk;

 MPI_Status status[100];
 MPI_Status status1;
 MPI_Request request[100];

 if (been_here ==0 )   {
   kk=0;
   for (m=1;m<=E->sphere.caps_per_proc;m++)    {
     ii = E->sphere.capid[m];
     for (k=1;k<=E->parallel.TNUM_PASS[E->mesh.levmax][m];k++)  {
       ++kk;
       /*       t_cap = E->sphere.cap[ii].connection[k];
		idd[ii][t_cap] = kk;*/

       sizeofk = (1+E->parallel.NUM_NODE[E->mesh.levmax][m].pass[k])*sizeof(double);
       S[kk]=(double *)malloc( sizeofk );
       R[kk]=(double *)malloc( sizeofk );
       }
     }

   idb= 0;
   for (k=1;k<=E->parallel.TNUM_PASSz[E->mesh.levmax];k++)  {
      sizeofk = (1+E->parallel.NUM_NODEz[E->mesh.levmax].pass[k])*sizeof(double);
      idb = max(idb,sizeofk);
      }

   RV=(double *)malloc( idb );
   SV=(double *)malloc( idb );

   been_here ++;
   }

  idb=0;
  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    ii = E->sphere.capid[m];

    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)     {
      /*      t_cap = E->sphere.cap[ii].connection[k];
	      kk = idd[ii][t_cap];*/
      kk=k;

      for (j=1;j<=E->parallel.NUM_NODE[lev][m].pass[k];j++)
        S[kk][j-1] = U[m][ E->parallel.EXCHANGE_NODE[lev][m][j].pass[k] ];

      if (E->parallel.PROCESSOR[lev][m].pass[k]!=E->parallel.me) {
	if (E->parallel.PROCESSOR[lev][m].pass[k]!=-1) {
         idb ++;
	 /*         MPI_Isend(S[kk],E->parallel.NUM_NODE[lev][m].pass[k],MPI_DOUBLE,
		    E->parallel.PROCESSOR[lev][m].pass[k],E->parallel.mst[ii][t_cap],E->parallel.world,&request[idb-1]);*/
        MPI_Isend(S[kk],E->parallel.NUM_NODE[lev][m].pass[k],MPI_DOUBLE,
             E->parallel.PROCESSOR[lev][m].pass[k],1,E->parallel.world,&request[idb-1]);
	}
         }
      }           /* for k */
    }     /* for m */         /* finish sending */

/*  MPI_Waitall(idb,request,status);  

  idb = 0; */
  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    ii = E->sphere.capid[m];

    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)   {
      /*      t_cap = E->sphere.cap[ii].connection[k];
	      kk = idd[ii][t_cap];*/
      kk=k;

      if (E->parallel.PROCESSOR[lev][m].pass[k]!=E->parallel.me)  {
	if (E->parallel.PROCESSOR[lev][m].pass[k]!=-1) {
         idb++;
	 /*         MPI_Irecv(R[kk],E->parallel.NUM_NODE[lev][m].pass[k],MPI_DOUBLE,
		    E->parallel.PROCESSOR[lev][m].pass[k],E->parallel.mst[ii][t_cap],E->parallel.world,&request[idb-1]);*/
         MPI_Irecv(R[kk],E->parallel.NUM_NODE[lev][m].pass[k],MPI_DOUBLE,
         E->parallel.PROCESSOR[lev][m].pass[k],1,E->parallel.world,&request[idb-1]);
         }
      }

      else   {
	/*         kk = idd[t_cap][ii];*/
	kk=k;
         for (j=1;j<=E->parallel.NUM_NODE[lev][m].pass[k];j++)
           U[m][ E->parallel.EXCHANGE_NODE[lev][m][j].pass[k] ] += S[kk][j-1];
         }
      }      /* for k */
    }     /* for m */         /* finish receiving */

  MPI_Waitall(idb,request,status);

  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    ii = E->sphere.capid[m];

    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)   {
      /*      t_cap = E->sphere.cap[ii].connection[k];
	      kk = idd[ii][t_cap];*/
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

 /* static int idd[NCS][NCS];*/
 static float *S[73],*R[73], *RV, *SV;

 static int been_here = 0;
 static int mid_recv, sizeofk;

 MPI_Status status[100];
 MPI_Status status1;
 MPI_Request request[100];

 if (been_here ==0 )   {
   kk=0;
   for (m=1;m<=E->sphere.caps_per_proc;m++)    {
     ii = E->sphere.capid[m];
     for (k=1;k<=E->parallel.TNUM_PASS[E->mesh.levmax][m];k++)  {
       ++kk;
       /*       t_cap = E->sphere.cap[ii].connection[k];
		idd[ii][t_cap] = kk;*/

       sizeofk = (1+E->parallel.NUM_NODE[E->mesh.levmax][m].pass[k])*sizeof(float);
       S[kk]=(float *)malloc( sizeofk );
       R[kk]=(float *)malloc( sizeofk );
       }
     }

   idb= 0;
   for (k=1;k<=E->parallel.TNUM_PASSz[E->mesh.levmax];k++)  {
      sizeofk = (1+E->parallel.NUM_NODEz[E->mesh.levmax].pass[k])*sizeof(float);
      idb = max(idb,sizeofk);
      }

   RV=(float *)malloc( idb );
   SV=(float *)malloc( idb );

   been_here ++;
   }

  idb=0;
  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    ii = E->sphere.capid[m];

    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)     {
      /*      t_cap = E->sphere.cap[ii].connection[k];
	      kk = idd[ii][t_cap];*/
      kk=k;

      for (j=1;j<=E->parallel.NUM_NODE[lev][m].pass[k];j++) {
        S[kk][j-1] = U[m][ E->parallel.EXCHANGE_NODE[lev][m][j].pass[k] ];
	}

      if (E->parallel.PROCESSOR[lev][m].pass[k]!=E->parallel.me) {
	if (E->parallel.PROCESSOR[lev][m].pass[k]!=-1) {
         idb ++;
	 /*         MPI_Isend(S[kk],E->parallel.NUM_NODE[lev][m].pass[k],MPI_FLOAT,
		    E->parallel.PROCESSOR[lev][m].pass[k],E->parallel.mst[ii][t_cap],E->parallel.world,&request[idb-1]); */
         MPI_Isend(S[kk],E->parallel.NUM_NODE[lev][m].pass[k],MPI_FLOAT,
          E->parallel.PROCESSOR[lev][m].pass[k],1,E->parallel.world,&request[idb-1]); 
         }
      }
      }           /* for k */
    }     /* for m */         /* finish sending */

/*  MPI_Waitall(idb,request,status);

  idb = 0; */
  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    ii = E->sphere.capid[m];

    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)   {
      /*      t_cap = E->sphere.cap[ii].connection[k];
	      kk = idd[ii][t_cap];*/
      kk=k;

      if (E->parallel.PROCESSOR[lev][m].pass[k]!=E->parallel.me)  {
	if (E->parallel.PROCESSOR[lev][m].pass[k]!=-1) {

         idb++;
	 /*         MPI_Irecv(R[kk],E->parallel.NUM_NODE[lev][m].pass[k],MPI_FLOAT,
		    E->parallel.PROCESSOR[lev][m].pass[k],E->parallel.mst[ii][t_cap],E->parallel.world,&request[idb-1]);*/
         MPI_Irecv(R[kk],E->parallel.NUM_NODE[lev][m].pass[k],MPI_FLOAT,
           E->parallel.PROCESSOR[lev][m].pass[k],1,E->parallel.world,&request[idb-1]);
         }
      }

      else   {
	/*         kk = idd[t_cap][ii];*/
	kk=k;
         for (j=1;j<=E->parallel.NUM_NODE[lev][m].pass[k];j++)
           U[m][ E->parallel.EXCHANGE_NODE[lev][m][j].pass[k] ] += S[kk][j-1];
         }
      }      /* for k */
    }     /* for m */         /* finish receiving */

  MPI_Waitall(idb,request,status);

  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    ii = E->sphere.capid[m];

    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)   {
      /*     t_cap = E->sphere.cap[ii].connection[k];
	     kk = idd[ii][t_cap];*/
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

 return;
 }
/* ================================================ */
/* ================================================ */

void exchange_snode_f(E, U1, U2, lev)
 struct All_variables *E;
 float **U1,**U2;
 int lev;
 {

 void parallel_process_sync();

 int ii,j,k,m,kk,t_cap,idb,msginfo[8];

 /*static int idd[NCS][NCS];*/
 static float *S[73],*R[73];

 static int been_here = 0;
 static int mid_recv, sizeofk;

 MPI_Status status[100];
 MPI_Status status1;
 MPI_Request request[100];

 if (been_here ==0 )   {
   kk=0;
   for (m=1;m<=E->sphere.caps_per_proc;m++)    {
     ii = E->sphere.capid[m];
     for (k=1;k<=E->parallel.TNUM_PASS[E->mesh.levmax][m];k++)  {
       ++kk;
       /*       t_cap = E->sphere.cap[ii].connection[k];
		idd[ii][t_cap] = kk;*/

       sizeofk = (1+2*E->parallel.NUM_sNODE[E->mesh.levmax][m].pass[k])*sizeof(float);
       S[kk]=(float *)malloc( sizeofk );
       R[kk]=(float *)malloc( sizeofk );
       }
     }
   been_here ++;
   }

  idb=0;
  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    ii = E->sphere.capid[m];

    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)     {
      /*      t_cap = E->sphere.cap[ii].connection[k];
	      kk = idd[ii][t_cap];*/
      kk=k;

      for (j=1;j<=E->parallel.NUM_sNODE[lev][m].pass[k];j++)  {
        S[kk][j-1] = U1[m][ E->parallel.EXCHANGE_sNODE[lev][m][j].pass[k] ];
        S[kk][j-1+E->parallel.NUM_sNODE[lev][m].pass[k]]
                   = U2[m][ E->parallel.EXCHANGE_sNODE[lev][m][j].pass[k] ];
        }

      if (E->parallel.PROCESSOR[lev][m].pass[k]!=E->parallel.me) {
	if (E->parallel.PROCESSOR[lev][m].pass[k]!=-1) {
         idb ++;
	 /*         MPI_Isend(S[kk],2*E->parallel.NUM_sNODE[lev][m].pass[k],MPI_FLOAT,
		    E->parallel.PROCESSOR[lev][m].pass[k],E->parallel.mst[ii][t_cap],E->parallel.world,&request[idb-1]);*/
         MPI_Isend(S[kk],2*E->parallel.NUM_sNODE[lev][m].pass[k],MPI_FLOAT,
             E->parallel.PROCESSOR[lev][m].pass[k],1,E->parallel.world,&request[idb-1]);
         }
      }
      }           /* for k */
    }     /* for m */         /* finish sending */

/*  MPI_Waitall(idb,request,status); 

  idb = 0; */
  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    ii = E->sphere.capid[m];

    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)   {
      /*      t_cap = E->sphere.cap[ii].connection[k];
	      kk = idd[ii][t_cap];*/
      kk=k;

      if (E->parallel.PROCESSOR[lev][m].pass[k]!=E->parallel.me)  {
	if (E->parallel.PROCESSOR[lev][m].pass[k]!=-1) {

         idb ++;
	 /*         MPI_Irecv(R[kk],2*E->parallel.NUM_sNODE[lev][m].pass[k],MPI_FLOAT,
		    E->parallel.PROCESSOR[lev][m].pass[k],E->parallel.mst[ii][t_cap],E->parallel.world,&request[idb-1]);*/
         MPI_Irecv(R[kk],2*E->parallel.NUM_sNODE[lev][m].pass[k],MPI_FLOAT,
           E->parallel.PROCESSOR[lev][m].pass[k],1,E->parallel.world,&request[idb-1]);
         }
      }

      else   {
	/*         kk = idd[t_cap][ii];*/
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
    ii = E->sphere.capid[m];

    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)   {
      /*      t_cap = E->sphere.cap[ii].connection[k];
	      kk = idd[ii][t_cap];*/
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

 return;
 }
/* ==========================================================  */
 void scatter_to_nlayer_id (E,AUi,AUo,lev) 
 struct All_variables *E;
 double **AUi,**AUo;
 int lev;
 {

 int i,j,k,k1,m,node1,node,eqn1,eqn,d;

 const int dims = E->mesh.nsd;

 static double *SD;
 static int been_here=0;
 static int *processors,rootid,nproc,NOZ;

 MPI_Status status;

 if (E->parallel.nprocz==1)  {
    if (E->parallel.me==0) fprintf(stderr,"scatter_to_nlayer should not be called\n");
    return;
    } 

 if (been_here==0)   {
   NOZ = E->lmesh.ELZ[lev]*E->parallel.nprocz + 1;

   processors = (int *)malloc((E->parallel.nprocz+2)*sizeof(int));

   SD = (double *)malloc((E->lmesh.NEQ[lev]+2)*sizeof(double));


   rootid = E->parallel.me_sph*E->parallel.nprocz;    /* which is the bottom cpu */
   nproc = 0;
   for (j=0;j<E->parallel.nprocz;j++) {
     d = rootid + j;
     processors[nproc] =  d;
     nproc ++;
     }

   been_here++;
   }

  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    if (E->parallel.me==rootid)
      for (d=0;d<E->parallel.nprocz;d++)  { 

        for (k=1;k<=E->lmesh.NOZ[lev];k++)   {
          k1 = k + d*E->lmesh.ELZ[lev]; 
          for (j=1;j<=E->lmesh.NOY[lev];j++)
            for (i=1;i<=E->lmesh.NOX[lev];i++)   {
              node = k + (i-1)*E->lmesh.NOZ[lev] + (j-1)*E->lmesh.NOZ[lev]*E->lmesh.NOX[lev];
              node1= k1+ (i-1)*NOZ + (j-1)*NOZ*E->lmesh.NOX[lev];
              SD[dims*(node-1)] = AUi[m][dims*(node1-1)];
              SD[dims*(node-1)+1] = AUi[m][dims*(node1-1)+1];
              SD[dims*(node-1)+2] = AUi[m][dims*(node1-1)+2];
              }
          }

        if (processors[d]!=rootid)  {
           MPI_Send(SD,E->lmesh.NEQ[lev],MPI_DOUBLE,processors[d],rootid,E->parallel.world);
	   }
        else
           for (i=0;i<=E->lmesh.NEQ[lev];i++)   
	      AUo[m][i] = SD[i];
        }
    else
        MPI_Recv(AUo[m],E->lmesh.NEQ[lev],MPI_DOUBLE,rootid,rootid,E->parallel.world,&status);
    }

 return;
 }

/* ==========================================================  */
 void gather_to_1layer_id (E,AUi,AUo,lev) 
 struct All_variables *E;
 double **AUi,**AUo;
 int lev;
 {

 int i,j,k,k1,m,node1,node,eqn1,eqn,d;

 const int dims = E->mesh.nsd;

 MPI_Status status;

 static double *RV;
 static int been_here=0;
 static int *processors,rootid,nproc,NOZ;

 if (E->parallel.nprocz==1)  {
    if (E->parallel.me==0) fprintf(stderr,"gather_to_1layer should not be called\n");
    return;
    } 

 if (been_here==0)   {
   NOZ = E->lmesh.ELZ[lev]*E->parallel.nprocz + 1;

   processors = (int *)malloc((E->parallel.nprocz+2)*sizeof(int));

   RV = (double *)malloc((E->lmesh.NEQ[lev]+2)*sizeof(double));


   rootid = E->parallel.me_sph*E->parallel.nprocz;    /* which is the bottom cpu */
   nproc = 0;
   for (j=0;j<E->parallel.nprocz;j++) {
     d = rootid + j;
     processors[nproc] =  d;
     nproc ++;
     }

   been_here++;
   }

  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    if (E->parallel.me!=rootid)
       MPI_Send(AUi[m],E->lmesh.NEQ[lev],MPI_DOUBLE,rootid,E->parallel.me,E->parallel.world);
    else
       for (d=0;d<E->parallel.nprocz;d++) {
         if (processors[d]!=rootid)
	   MPI_Recv(RV,E->lmesh.NEQ[lev],MPI_DOUBLE,processors[d],processors[d],E->parallel.world,&status);
         else
           for (node=0;node<E->lmesh.NEQ[lev];node++)
	      RV[node] = AUi[m][node];

         for (k=1;k<=E->lmesh.NOZ[lev];k++)   {
           k1 = k + d*E->lmesh.ELZ[lev];
           for (j=1;j<=E->lmesh.NOY[lev];j++)
             for (i=1;i<=E->lmesh.NOX[lev];i++)   {
               node = k + (i-1)*E->lmesh.NOZ[lev] + (j-1)*E->lmesh.NOZ[lev]*E->lmesh.NOX[lev];
               node1 = k1 + (i-1)*NOZ + (j-1)*NOZ*E->lmesh.NOX[lev];
	       
               AUo[m][dims*(node1-1)] = RV[dims*(node-1)];
               AUo[m][dims*(node1-1)+1] = RV[dims*(node-1)+1];
               AUo[m][dims*(node1-1)+2] = RV[dims*(node-1)+2];
	       }
	   }
	 }
       }

 return;
 }


/* ==========================================================  */
 void gather_to_1layer_node (E,AUi,AUo,lev) 
 struct All_variables *E;
 float **AUi,**AUo;
 int lev;
 {

 int i,j,k,k1,m,node1,node,d;

 MPI_Status status;

 static float *RV;
 static int been_here=0;
 static int *processors,rootid,nproc,NOZ,NNO;

 if (E->parallel.nprocz==1)  {
    if (E->parallel.me==0) fprintf(stderr,"gather_to_1layer should not be called\n");
    return;
    } 

 if (been_here==0)   {
   NOZ = E->lmesh.ELZ[lev]*E->parallel.nprocz + 1;
   NNO = NOZ*E->lmesh.NOX[lev]*E->lmesh.NOY[lev];

   processors = (int *)malloc((E->parallel.nprocz+2)*sizeof(int));
   RV = (float *)malloc((E->lmesh.NNO[lev]+2)*sizeof(float));


   rootid = E->parallel.me_sph*E->parallel.nprocz;    /* which is the bottom cpu */
   nproc = 0;
   for (j=0;j<E->parallel.nprocz;j++) {
     d = rootid + j;
     processors[nproc] =  d;
     nproc ++;
     }

   been_here++;
   }

  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    if (E->parallel.me!=rootid) {
       MPI_Send(AUi[m],E->lmesh.NNO[lev]+1,MPI_FLOAT,rootid,E->parallel.me,E->parallel.world);
         for (node=1;node<=NNO;node++)
           AUo[m][node] = 1.0;
       }
    else 
       for (d=0;d<E->parallel.nprocz;d++) {
	 if (processors[d]!=rootid)
           MPI_Recv(RV,E->lmesh.NNO[lev]+1,MPI_FLOAT,processors[d],processors[d],E->parallel.world,&status);
         else
	   for (node=1;node<=E->lmesh.NNO[lev];node++)
	      RV[node] = AUi[m][node];
	  
         for (k=1;k<=E->lmesh.NOZ[lev];k++)   {
           k1 = k + d*E->lmesh.ELZ[lev]; 
           for (j=1;j<=E->lmesh.NOY[lev];j++)
             for (i=1;i<=E->lmesh.NOX[lev];i++)   {
               node = k + (i-1)*E->lmesh.NOZ[lev] + (j-1)*E->lmesh.NOZ[lev]*E->lmesh.NOX[lev];
               node1 = k1 + (i-1)*NOZ + (j-1)*NOZ*E->lmesh.NOX[lev];
               AUo[m][node1] = RV[node];
               }
           }
         }
    }

 return;
 }

/* ==========================================================  */
 void gather_to_1layer_ele (E,AUi,AUo,lev) 
 struct All_variables *E;
 float **AUi,**AUo;
 int lev;
 {

 int i,j,k,k1,m,e,d,e1;

 MPI_Status status;

 static float *RV;
 static int been_here=0;
 static int *processors,rootid,nproc,NOZ,NNO;

 if (E->parallel.nprocz==1)  {
    if (E->parallel.me==0) fprintf(stderr,"gather_to_1layer should not be called\n");
    return;
    } 

 if (been_here==0)   {
   NOZ = E->lmesh.ELZ[lev]*E->parallel.nprocz;
   NNO = NOZ*E->lmesh.ELX[lev]*E->lmesh.ELY[lev];

   processors = (int *)malloc((E->parallel.nprocz+2)*sizeof(int));
   RV = (float *)malloc((E->lmesh.NEL[lev]+2)*sizeof(float));


   rootid = E->parallel.me_sph*E->parallel.nprocz;    /* which is the bottom cpu */
   nproc = 0;
   for (j=0;j<E->parallel.nprocz;j++) {
     d = rootid + j;
     processors[nproc] =  d;
     nproc ++;
     }

   been_here++;
   }

  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    if (E->parallel.me!=rootid) {
       MPI_Send(AUi[m],E->lmesh.NEL[lev]+1,MPI_FLOAT,rootid,E->parallel.me,E->parallel.world);
         for (e=1;e<=NNO;e++)
           AUo[m][e] = 1.0;
       }
    else 
       for (d=0;d<E->parallel.nprocz;d++) {
	 if (processors[d]!=rootid)
           MPI_Recv(RV,E->lmesh.NEL[lev]+1,MPI_FLOAT,processors[d],processors[d],E->parallel.world,&status);
         else
	   for (e=1;e<=E->lmesh.NEL[lev];e++)
	      RV[e] = AUi[m][e];
	  
         for (k=1;k<=E->lmesh.ELZ[lev];k++)   {
           k1 = k + d*E->lmesh.ELZ[lev]; 
           for (j=1;j<=E->lmesh.ELY[lev];j++)
             for (i=1;i<=E->lmesh.ELX[lev];i++)   {
               e = k + (i-1)*E->lmesh.ELZ[lev] + (j-1)*E->lmesh.ELZ[lev]*E->lmesh.ELX[lev];
               e1 = k1 + (i-1)*NOZ + (j-1)*NOZ*E->lmesh.ELX[lev];
               AUo[m][e1] = RV[e];
               }
           }
         }
    }

 return;
 }


/* ==========================================================  */
 void gather_TG_to_me0(E,TG)
 struct All_variables *E;
 float *TG;
 {

 void parallel_process_sync();
 int i,j,nsl,idb,to_everyone,from_proc,mst,me;

 static float *RG[20];
 static int been_here=0;
 const float e_16=1.e-16;

 MPI_Status status[100];
 MPI_Status status1;
 MPI_Request request[100];

 if (E->parallel.nprocxy==1)   return;

 nsl = E->sphere.nsf+1;
 me = E->parallel.me;

 if (been_here==0)   {
   been_here++;
   for (i=1;i<E->parallel.nprocxy*E->parallel.surf_proc_per_cap;i++)
     RG[i] = ( float *)malloc((E->sphere.nsf+1)*sizeof(float));
   }


 idb=0;
 for (i=1;i<=E->parallel.nprocxy*E->parallel.surf_proc_per_cap;i++)  {
   to_everyone = E->parallel.nprocz*(i-1) + E->parallel.me_loc[3];

   if (me!=to_everyone)    {  /* send TG */
     idb++;
     mst = me;
     MPI_Isend(TG,nsl,MPI_FLOAT,to_everyone,mst,E->parallel.world,&request[idb-1]);
     }
   }

/* parallel_process_sync(); */

 idb=0;
 for (i=1;i<=E->parallel.nprocxy*E->parallel.surf_proc_per_cap;i++)  {
   from_proc = E->parallel.nprocz*(i-1) + E->parallel.me_loc[3];
   if (me!=from_proc)   {    /* me==0 receive all TG and add them up */
      mst = from_proc;
      idb++;
      MPI_Irecv(RG[idb],nsl,MPI_FLOAT,from_proc,mst,E->parallel.world,&request[idb-1]);
      }
   }

 MPI_Waitall(idb,request,status);

 for (i=1;i<E->parallel.nprocxy*E->parallel.surf_proc_per_cap;i++)
   for (j=1;j<=E->sphere.nsf; j++)  {
        if (fabs(TG[j]) < e_16) TG[j] += RG[i][j];
        }

/* parallel_process_sync(); */

 return;
 }

/* ========================================= */
 void sum_across_depth_sph(E,sphc,sphs,dest_proc)
 struct All_variables *E;
 int dest_proc;
 float *sphc,*sphs;
 {

 void parallel_process_sync();
 int jumpp,i,j,nsl,idb,to_proc,from_proc,mst,me;

 float *RG,*TG;

 MPI_Status status[100];
 MPI_Status status1;
 MPI_Request request[100];

 if (E->parallel.nprocz==1)   return;

 jumpp = E->sphere.hindice;
 nsl = E->sphere.hindice*2+3;
 me = E->parallel.me;

 TG = ( float *)malloc((nsl+1)*sizeof(float));
 if (E->parallel.me_loc[3]==dest_proc) 
      RG = ( float *)malloc((nsl+1)*sizeof(float));

 for (i=0;i<E->sphere.hindice;i++)   {
    TG[i] = sphc[i];
    TG[i+jumpp] = sphs[i];
    }


 if (E->parallel.me_loc[3]!=dest_proc)    {  /* send TG */
     to_proc = E->parallel.me_sph*E->parallel.nprocz+E->parallel.nprocz-1;
     mst = me;
     MPI_Send(TG,nsl,MPI_FLOAT,to_proc,mst,E->parallel.world);
     }

 parallel_process_sync();

 if (E->parallel.me_loc[3]==dest_proc)  {
   for (i=1;i<E->parallel.nprocz;i++) {
      from_proc = me - i;
      mst = from_proc;
      MPI_Recv(RG,nsl,MPI_FLOAT,from_proc,mst,E->parallel.world,&status1);

      for (j=0;j<E->sphere.hindice;j++)   {
        sphc[j] += RG[j];
        sphs[j] += RG[j+jumpp];
        }
      }
   }

 free((void *) TG);
 if (E->parallel.me_loc[3]==dest_proc) 
   free((void *) RG);

 return;
 }
/* ========================================= */
 void sum_across_surf_sph(E,TG,loc_proc)
 struct All_variables *E;
 int loc_proc;
 float *TG;
 {

 void parallel_process_sync();
 int i,j,nsl,idb,to_everyone,from_proc,mst,me;

 float *RG[20];

 MPI_Status status[100];
 MPI_Status status1;
 MPI_Request request[100];

 if (E->parallel.nprocxy==1)   return;

 nsl = E->sphere.hindice*2+2;
 me = E->parallel.me;

 for (i=1;i<E->parallel.nprocxy*E->parallel.surf_proc_per_cap;i++)
    RG[i] = ( float *)malloc((nsl+1)*sizeof(float));


 idb=0;
 for (i=1;i<=E->parallel.nprocxy*E->parallel.surf_proc_per_cap;i++)  {
   to_everyone = E->parallel.nprocz*(i-1) + loc_proc;

   if (me!=to_everyone)    {  /* send TG */
     idb++;
     mst = me;
     MPI_Isend(TG,nsl,MPI_FLOAT,to_everyone,mst,E->parallel.world,&request[idb-1]);
     }
   }

/* parallel_process_sync(); */

 idb=0;
 for (i=1;i<=E->parallel.nprocxy*E->parallel.surf_proc_per_cap;i++)  {
   from_proc = E->parallel.nprocz*(i-1) + loc_proc;
   if (me!=from_proc)   {    /* me==0 receive all TG and add them up */
      mst = from_proc;
      idb++;
      MPI_Irecv(RG[idb],nsl,MPI_FLOAT,from_proc,mst,E->parallel.world,&request[idb-1]);
      }
   }

 MPI_Waitall(idb,request,status);

 for (i=1;i<E->parallel.nprocxy*E->parallel.surf_proc_per_cap;i++)
   for (j=0;j<nsl; j++)  {
        TG[j] += RG[i][j];
        }

/* parallel_process_sync(); */

 for (i=1;i<E->parallel.nprocxy*E->parallel.surf_proc_per_cap;i++)
       free((void *) RG[i]);

 return;
 }

/* ================================================== */

 void set_communication_sphereh(E)
 struct All_variables *E;
 {
  int i;

  i = cases[E->sphere.caps_per_proc];

  E->parallel.nproc_sph[1] = incases3[i].xy[0];
  E->parallel.nproc_sph[2] = incases3[i].xy[1];

  E->sphere.lelx = E->sphere.elx/E->parallel.nproc_sph[1];
  E->sphere.lely = E->sphere.ely/E->parallel.nproc_sph[2];
  E->sphere.lsnel = E->sphere.lely*E->sphere.lelx;
  E->sphere.lnox = E->sphere.lelx + 1;
  E->sphere.lnoy = E->sphere.lely + 1;
  E->sphere.lnsf = E->sphere.lnox*E->sphere.lnoy;

  for (i=0;i<=E->parallel.nprocz-1;i++)   
    if (E->parallel.me_loc[3] == i)    {
      E->parallel.me_sph = (E->parallel.me-i)/E->parallel.nprocz;
      E->parallel.me_loc_sph[1] = E->parallel.me_sph%E->parallel.nproc_sph[1];
      E->parallel.me_loc_sph[2] = E->parallel.me_sph/E->parallel.nproc_sph[1];
      }

  E->sphere.lexs = E->sphere.lelx * E->parallel.me_loc_sph[1];
  E->sphere.leys = E->sphere.lely * E->parallel.me_loc_sph[2];

 return;
 }



