#include <mpi.h>
#include <math.h>
#include "element_definitions.h"
#include "global_defs.h"
#include "sphere_communication.h"


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
  E->parallel.me_locl[1] = 0;
  E->parallel.me_locl[2] = 0;
  E->parallel.me_locl[3] = 0;

  /*  MPI_Init(&argc,&argv); moved to main{} in Citcom.c, cpc 12/24/00 */
  MPI_Comm_rank(MPI_COMM_WORLD, &(E->parallel.me) );
  MPI_Comm_size(MPI_COMM_WORLD, &(E->parallel.nproc) );

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

  MPI_Barrier(MPI_COMM_WORLD);
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
  
  int i,j,k,me,temp,pid_surf;
  void parallel_process_termination();

  me = E->parallel.me;

  E->parallel.nprocxy=E->parallel.nprocxl*E->parallel.nprocyl;
  E->parallel.nprocz=E->parallel.nproczl;

  if ( E->parallel.nprocxy*E->parallel.nprocz!=E->parallel.nproc ) {
    if (E->parallel.me==0) fprintf(stderr,"!!!! # of requested CPU is incorrect \n");
    parallel_process_termination();
    }

        /* z direction first */
  j = me % E->parallel.nprocz;
  E->parallel.me_loc[3] = j;

  E->parallel.me_loc[1] = 0;
  E->parallel.me_loc[2] = 0;

  E->parallel.nprocxzl = E->parallel.nprocxl*E->parallel.nproczl;
  E->parallel.nprocxyl = E->parallel.nprocxl*E->parallel.nprocyl;
  E->parallel.nproczyl = E->parallel.nproczl*E->parallel.nprocyl;

        /* z direction first */
  j = me % E->parallel.nproczl;
  E->parallel.me_locl[3] = j;

        /* y direction then */
  k = (me+1)/E->parallel.nprocxzl-(((me+1)%E->parallel.nprocxzl==0)?1:0);
  E->parallel.me_locl[2] = k;

        /* x direction then */
  i = (me+1-k*E->parallel.nprocxzl)/E->parallel.nproczl - ( ((me+1-k*E->parallel.nprocxzl)%E->parallel.nproczl==0)? 1:0 );
  E->parallel.me_locl[1] = i;

/*
  fprintf(stderr,"%d %d %d %d\n",E->parallel.me, E->parallel.me_locl[1],E->parallel.me_locl[2],E->parallel.me_locl[3]);
*/

/*
  E->sphere.caps_per_proc = max(1,E->sphere.caps/E->parallel.nprocxy);
*/
  E->sphere.caps_per_proc = 1;

          /* determine cap id for each cap in a given processor  */

  pid_surf = me/E->parallel.nprocz;
  i = cases[E->sphere.caps_per_proc];

  for (j=1;j<=E->sphere.caps_per_proc;j++)  {
    temp = pid_surf*E->sphere.caps_per_proc + j-1;
    E->sphere.capid[j] = incases1[i].links[temp]; 
    E->sphere.local_capid[ E->sphere.capid[j] ] = j;
    }

          /* determine which caps are linked with each of 12 cap  */
  for (j=1;j<=E->sphere.caps;j++)   
    for (i=1;i<=E->sphere.max_connections;i++)   
      E->sphere.cap[j].connection[i] = cap_connections[j][i];

          /* determine surface proc id for each cap  */
  i = cases[E->sphere.caps_per_proc];
  for (j=1;j<=E->sphere.caps;j++)     {
    E->sphere.pid_surf[j] = incases2[i].links[j-1];
    }

  temp = 1;
  for (j=1;j<=E->sphere.caps;j++)   
    for (i=1;i<=j;i++)  
      if (i!=j)
        E->parallel.mst[j][i] = temp++;   

  for (j=1;j<=E->sphere.caps;j++)   
    for (i=1;i<=E->sphere.caps;i++)  
      if (i>j)
        E->parallel.mst[j][i] = E->parallel.mst[i][j];   



if (E->control.verbose)
  for (j=1;j<=E->sphere.caps;j++)     {
    fprintf(E->fp_out,"j=%d pid_surf=%d capid=%d local_capid=%d \n",j,E->sphere.pid_surf[j],E->sphere.capid[j],E->sphere.local_capid[ E->sphere.capid[j] ]);
    for (i=1;i<=E->sphere.max_connections;i++)   
       fprintf(E->fp_out,"j=%d i=%d target_cap=%d \n",j,i,E->sphere.cap[j].connection[i]);
    }

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

  E->lmesh.elx = E->mesh.elx/E->parallel.nprocxl;
  E->lmesh.elz = E->mesh.elz/E->parallel.nproczl;
  E->lmesh.ely = E->mesh.ely/E->parallel.nprocyl;
  E->lmesh.nox = E->lmesh.elx + 1;
  E->lmesh.noz = E->lmesh.elz + 1;
  E->lmesh.noy = E->lmesh.ely + 1;

  E->lmesh.exs = E->parallel.me_locl[1]*E->lmesh.elx;
  E->lmesh.eys = E->parallel.me_locl[2]*E->lmesh.ely;
  E->lmesh.ezs = E->parallel.me_locl[3]*E->lmesh.elz;
  E->lmesh.nxs = E->parallel.me_locl[1]*E->lmesh.elx+1;
  E->lmesh.nys = E->parallel.me_locl[2]*E->lmesh.ely+1;
  E->lmesh.nzs = E->parallel.me_locl[3]*E->lmesh.elz+1;

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
        nox = E->mesh.mgunitx * (int) pow(2.0,(double)i) + 1;
        noy = E->mesh.mgunity * (int) pow(2.0,(double)i) + 1;
        noz = E->lmesh.elz/(int)pow(2.0,(double)(E->mesh.levmax-i))+1;
        E->parallel.redundant[i]=0;
        }
     else
        { noz = E->lmesh.noz;
          noy = E->mesh.mgunity * (int) pow(2.0,(double)i) + 1;
          nox = E->mesh.mgunitx * (int) pow(2.0,(double)i) + 1;
          if(i<E->mesh.levmax) noz=2;
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

     E->lmesh.EXS[i] = E->parallel.me_locl[1]*E->lmesh.ELX[i];
     E->lmesh.EYS[i] = E->parallel.me_locl[2]*E->lmesh.ELY[i];
     E->lmesh.EZS[i] = E->parallel.me_locl[3]*E->lmesh.ELZ[i];
     E->lmesh.NXS[i] = E->parallel.me_locl[1]*E->lmesh.ELX[i]+1;
     E->lmesh.NYS[i] = E->parallel.me_locl[2]*E->lmesh.ELY[i]+1;
     E->lmesh.NZS[i] = E->parallel.me_locl[3]*E->lmesh.ELZ[i]+1;

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
  FILE *fp,*fp1;
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

/*
    if (E->parallel.me_loc[3]!=E->parallel.nprocz-1 )
      for (lnode=1;lnode<=E->parallel.NUM_NNO[lev][m].bound[6];lnode++) {
        node = E->parallel.NODE[lev][m][lnode].bound[6];
        E->NODE[lev][m][node] = E->NODE[lev][m][node] | SKIP;
        }

    if (E->sphere.capid[m]==E->sphere.caps)
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

    if (E->sphere.capid[m]==1)
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

*/

/* ccccccc  */

    if (E->parallel.me_locl[1]!=E->parallel.nprocxl-1)
      for (lnode=1;lnode<=E->parallel.NUM_NNO[lev][m].bound[2];lnode++) {
        node = E->parallel.NODE[lev][m][lnode].bound[2];
        E->NODE[lev][m][node] = E->NODE[lev][m][node] | SKIP;
        }

    if (E->parallel.me_locl[2]!=E->parallel.nprocyl-1)
      for (lnode=1;lnode<=E->parallel.NUM_NNO[lev][m].bound[4];lnode++) {
        node = E->parallel.NODE[lev][m][lnode].bound[4];
        E->NODE[lev][m][node] = E->NODE[lev][m][node] | SKIP;
        }

    if (E->parallel.me_locl[3]!=E->parallel.nproczl-1)
      for (lnode=1;lnode<=E->parallel.NUM_NNO[lev][m].bound[6];lnode++) {
        node = E->parallel.NODE[lev][m][lnode].bound[6];
        E->NODE[lev][m][node] = E->NODE[lev][m][node] | SKIP;
        }

      }       /* end for m */
    }   /* end for level */

/*
 ii=0;
 for (m=1;m<=E->sphere.caps_per_proc;m++)
    for (node=1;node<=E->lmesh.nno;node++)
      if(E->node[m][node] & SKIPS)
        ii+=1; 

 MPI_Allreduce(&ii, &node  ,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);


 E->mesh.nno -= node + 2*E->mesh.noz;
 E->mesh.neq = E->mesh.nno*3;

*/

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

  int m,i,ii,j,k,l,node,el,elt,lnode,jj,doff,target_cap;
  int lev,elx,elz,ely,nno,nox,noz,noy,p,kkk,kk,kf,kkkp;
  int me, nproczl,nprocxl,nprocyl;
  int temp_dims,addi_doff;
  FILE *fp,*fp1,*fp2;
  char output_file[255];
  void parallel_process_termination();

  const int dims=E->mesh.nsd;

  me = E->parallel.me;
  nproczl = E->parallel.nproczl;
  nprocyl = E->parallel.nprocyl;
  nprocxl = E->parallel.nprocxl;

  
        /* determine the communications in horizontal direction        */
  for(lev=E->mesh.gridmax;lev>=E->mesh.gridmin;lev--)       {
    nox = E->lmesh.NOX[lev];
    noz = E->lmesh.NOZ[lev];
    noy = E->lmesh.NOY[lev];
    ii=0;
    kkk=0;


    for(m=1;m<=E->sphere.caps_per_proc;m++)    {
     
          for(i=1;i<=2;i++)       {       /* do YOZ boundaries & OY lines */

        ii ++;
        E->parallel.NUM_PASS[lev][m].bound[ii] = 1;
        if(E->parallel.me_locl[1]==0 && i==1)
          E->parallel.NUM_PASS[lev][m].bound[ii] = 0;
        else if(E->parallel.me_locl[1]==nprocxl-1 && i==2)
          E->parallel.NUM_PASS[lev][m].bound[ii] = 0;

        for (p=1;p<=E->parallel.NUM_PASS[lev][m].bound[ii];p++)  {
          kkk ++;
              /* determine the pass ID for ii-th boundary and p-th pass */

          E->parallel.PROCESSOR[lev][m].pass[kkk]=me-((i==1)?1:-1)*nproczl;

              E->parallel.NUM_NODE[lev][m].pass[kkk] = E->parallel.NUM_NNO[lev][m].bound[ii];
          jj = 0;   
          for (k=1;k<=E->parallel.NUM_NODE[lev][m].pass[kkk];k++)   {
            lnode = k;
            node = E->parallel.NODE[lev][m][lnode].bound[ii];
            E->parallel.EXCHANGE_NODE[lev][m][k].pass[kkk] = node;
            temp_dims = dims;

                    for(doff=1;doff<=temp_dims;doff++)
                         E->parallel.EXCHANGE_ID[lev][m][++jj].pass[kkk] = E->ID[lev][m][node].doff[doff];
            }  /* end for node k */

              E->parallel.NUM_NEQ[lev][m].pass[kkk] = jj;

          }   /* end for loop p */
            }  /* end for i */


        for(k=1;k<=2;k++)        {      /* do XOZ boundaries & OZ lines */
        ii ++;
        E->parallel.NUM_PASS[lev][m].bound[ii] = 1;
        if(E->parallel.me_locl[2]==0 && k==1)
          E->parallel.NUM_PASS[lev][m].bound[ii] = 0;
        else if(E->parallel.me_locl[2]==nprocyl-1 && k==2)
          E->parallel.NUM_PASS[lev][m].bound[ii] = 0;

        for (p=1;p<=E->parallel.NUM_PASS[lev][m].bound[ii];p++)  {

          kkk ++;
              /* determine the pass ID for ii-th boundary and p-th pass */

          E->parallel.PROCESSOR[lev][m].pass[kkk]=me-((k==1)?1:-1)*nprocxl*nproczl;
          E->parallel.NUM_NODE[lev][m].pass[kkk] = E->parallel.NUM_NNO[lev][m].bound[ii];

          jj = 0; kf = 0;
          for (kk=1;kk<=E->parallel.NUM_NODE[lev][m].pass[kkk];kk++)   {
            lnode = kk;
            node = E->parallel.NODE[lev][m][lnode].bound[ii];
            E->parallel.EXCHANGE_NODE[lev][m][kk].pass[kkk] = node;
            temp_dims = dims;
                    for(doff=1;doff<=temp_dims;doff++)
                         E->parallel.EXCHANGE_ID[lev][m][++jj].pass[kkk] = E->ID[lev][m][node].doff[doff];
            }  /* end for node kk */

              E->parallel.NUM_NEQ[lev][m].pass[kkk] = jj;

          }   /* end for loop p */

            }  /* end for k */


        for(j=1;j<=2;j++)       {       /* do XOY boundaries & OX lines */
        ii ++;
        E->parallel.NUM_PASS[lev][m].bound[ii] = 1;
        if(E->parallel.me_locl[3]==0 && j==1)
          E->parallel.NUM_PASS[lev][m].bound[ii] = 0;
        else if(E->parallel.me_locl[3]==nproczl-1 && j==2)
          E->parallel.NUM_PASS[lev][m].bound[ii] = 0;

        for (p=1;p<=E->parallel.NUM_PASS[lev][m].bound[ii];p++)  {
          kkk ++;
              /* determine the pass ID for ii-th boundary and p-th pass */

          E->parallel.PROCESSOR[lev][m].pass[kkk]=me-((j==1)?1:-1);
          E->parallel.NUM_NODE[lev][m].pass[kkk] = E->parallel.NUM_NNO[lev][m].bound[ii];

          jj = 0; kf = 0;
          for (kk=1;kk<=E->parallel.NUM_NODE[lev][m].pass[kkk];kk++)   {
            lnode = kk;
            node = E->parallel.NODE[lev][m][lnode].bound[ii];
            E->parallel.EXCHANGE_NODE[lev][m][kk].pass[kkk] = node;
            temp_dims = dims;
                    for(doff=1;doff<=temp_dims;doff++)
                         E->parallel.EXCHANGE_ID[lev][m][++jj].pass[kkk] = E->ID[lev][m][node].doff[doff];
            }  /* end for node k */

              E->parallel.NUM_NEQ[lev][m].pass[kkk] = jj;

          }   /* end for loop p */

            }     /* end for j */


      E->parallel.TNUM_PASS[lev][m] = kkk;


       }     /* end for m  */

      }        /* end for level */



fflush(E->fp_out);
  
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
  int p,me,m, nprocz;
  int nprocxl,nprocyl,nproczl;
  void parallel_process_termination();
  FILE *fp1,*fp2;

  char output_file[200];
  const int dims=E->mesh.nsd;

  me = E->parallel.me;
  nprocz = E->parallel.nprocz;

  nprocxl = E->parallel.nprocxl;
  nprocyl = E->parallel.nprocyl;
  nproczl = E->parallel.nproczl;
  

        /* determine the communications in horizontal direction        */
  for(lev=E->mesh.gridmax;lev>=E->mesh.gridmin;lev--)       {
    nox = E->lmesh.NOX[lev];
    noz = E->lmesh.NOZ[lev];
    noy = E->lmesh.NOY[lev];
    ii = 0;
    kkk = 0;

    for(m=1;m<=E->sphere.caps_per_proc;m++)    {

        for(i=1;i<=2;i++)       {       /* do YOZ boundaries & OY lines */

        ii ++;
        E->parallel.NUM_PASS[lev][m].bound[ii] = 1;
        if(E->parallel.me_locl[1]==0 && i==1)
          E->parallel.NUM_PASS[lev][m].bound[ii] = 0;
        else if(E->parallel.me_locl[1]==nprocxl-1 && i==2)
          E->parallel.NUM_PASS[lev][m].bound[ii] = 0;

        for (p=1;p<=E->parallel.NUM_PASS[lev][m].bound[ii];p++)  {
          kkk ++;
              /* determine the pass ID for ii-th boundary and p-th pass */

          E->parallel.sPROCESSOR[lev][m].pass[kkk]=me-((i==1)?1:-1)*nproczl;

              E->parallel.NUM_sNODE[lev][m].pass[kkk] =
                          E->parallel.NUM_NNO[lev][m].bound[ii]/noz;
          for (k=1;k<=E->parallel.NUM_sNODE[lev][m].pass[kkk];k++)   {
            lnode = k;             /* due to lnode increases in horizontal di first */
            node = (E->parallel.NODE[lev][m][lnode].bound[ii]-1)/noz+1;
            E->parallel.EXCHANGE_sNODE[lev][m][k].pass[kkk] = node;
            }  /* end for node k */

          }   /* end for loop p */
            }  /* end for i */

     ii = 2;
          for(k=1;k<=2;k++)        {      /* do XOZ boundaries & OX lines */

        ii ++;
        E->parallel.NUM_PASS[lev][m].bound[ii] = 1;
        if(E->parallel.me_locl[2]==0 && k==1)
          E->parallel.NUM_PASS[lev][m].bound[ii] = 0;
        else if(E->parallel.me_locl[2]==nprocyl-1 && k==2)
          E->parallel.NUM_PASS[lev][m].bound[ii] = 0;

        for (p=1;p<=E->parallel.NUM_PASS[lev][m].bound[ii];p++)  {

          kkk ++;
              /* determine the pass ID for ii-th boundary and p-th pass */

          E->parallel.sPROCESSOR[lev][m].pass[kkk]=me-((k==1)?1:-1)*nprocxl*nproczl;

              E->parallel.NUM_sNODE[lev][m].pass[kkk] =
                          E->parallel.NUM_NNO[lev][m].bound[ii]/noz;

          for (kk=1;kk<=E->parallel.NUM_sNODE[lev][m].pass[kkk];kk++)   {
            lnode = kk;             /* due to lnode increases in horizontal di first */
            node = (E->parallel.NODE[lev][m][lnode].bound[ii]-1)/noz+1;
            E->parallel.EXCHANGE_sNODE[lev][m][kk].pass[kkk] = node;
            }  /* end for node kk */

          }   /* end for loop p */

            }  /* end for k */
  

    E->parallel.sTNUM_PASS[lev][m] = kkk;


      }   /* end for m  */

    }   /* end for lev  */

  
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

 static int idd[NCS][NCS];
/* static double *S[73],*R[73], *RV, *SV;   */
 static double *S[27],*R[27], *RV, *SV;

 static int been_here = 0;
 static int mid_recv, sizeofk;

 MPI_Status status;

 if (been_here ==0 )   {
   for (m=1;m<=E->sphere.caps_per_proc;m++)    {
     for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)  {

       sizeofk = (1+E->parallel.NUM_NEQ[lev][m].pass[k])*sizeof(double);
       S[k]=(double *)malloc( sizeofk );
       R[k]=(double *)malloc( sizeofk );
       }
      }
   been_here ++;
     }

  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
     for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)  {
      for (j=1;j<=E->parallel.NUM_NEQ[lev][m].pass[k];j++)  
        S[k][j-1] = U[m][ E->parallel.EXCHANGE_ID[lev][m][j].pass[k] ];
      

	
        MPI_Sendrecv(S[k],E->parallel.NUM_NEQ[lev][m].pass[k],MPI_DOUBLE,
             E->parallel.PROCESSOR[lev][m].pass[k],1,
                     R[k],E->parallel.NUM_NEQ[lev][m].pass[k],MPI_DOUBLE,
             E->parallel.PROCESSOR[lev][m].pass[k],1,MPI_COMM_WORLD,&status);

        for (j=1;j<=E->parallel.NUM_NEQ[lev][m].pass[k];j++)
           U[m][ E->parallel.EXCHANGE_ID[lev][m][j].pass[k] ] += R[k][j-1];

      }           /* for k */
    }     /* for m */         /* finish sending */



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

 static int idd[NCS][NCS];
/*   static double *S[73],*R[73], *RV, *SV;   */
   static double *S[27],*R[27], *RV, *SV;   

 static int been_here = 0;
 static int mid_recv, sizeofk;

 MPI_Status status;

 if (been_here ==0 )   {
   for (m=1;m<=E->sphere.caps_per_proc;m++)    {
     for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)  {

       sizeofk = (1+E->parallel.NUM_NODE[lev][m].pass[k])*sizeof(double);
       S[k]=(double *)malloc( sizeofk );
       R[k]=(double *)malloc( sizeofk );

        }
       }   /* end for k */
      been_here ++;
     }

    for(m=1;m<=E->sphere.caps_per_proc;m++)     {
    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)   {
        for (j=1;j<=E->parallel.NUM_NODE[lev][m].pass[k];j++)
          S[k][j-1] = U[m][ E->parallel.EXCHANGE_NODE[lev][m][j].pass[k] ];

        MPI_Sendrecv(S[k],E->parallel.NUM_NODE[lev][m].pass[k],MPI_DOUBLE,
             E->parallel.PROCESSOR[lev][m].pass[k],1,
                     R[k],E->parallel.NUM_NODE[lev][m].pass[k],MPI_DOUBLE,
             E->parallel.PROCESSOR[lev][m].pass[k],1,MPI_COMM_WORLD,&status);

        for (j=1;j<=E->parallel.NUM_NODE[lev][m].pass[k];j++)
          U[m][ E->parallel.EXCHANGE_NODE[lev][m][j].pass[k] ] += R[k][j-1];
        }
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
 int levmax;

 static int idd[NCS][NCS];
/* static float *S[73],*R[73], *RV, *SV;   */
 static float *S[27],*R[27], *RV, *SV;

 static int been_here = 0;
 static int mid_recv, sizeofk;

 MPI_Status status;

 levmax=E->mesh.levmax;


 if (been_here ==0 )   {

   for (m=1;m<=E->sphere.caps_per_proc;m++)    {
     for (k=1;k<=E->parallel.TNUM_PASS[levmax][m];k++)  {

       sizeofk = (1+E->parallel.NUM_NODE[levmax][m].pass[k])*sizeof(float);
       S[k]=(float *)malloc( sizeofk );
       R[k]=(float *)malloc( sizeofk );

        }
       }   /* end for k */

      been_here ++;
     }


    for (m=1;m<=E->sphere.caps_per_proc;m++)     {
    for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)   {
        for (j=1;j<=E->parallel.NUM_NODE[lev][m].pass[k];j++)  
          S[k][j-1] = U[m][ E->parallel.EXCHANGE_NODE[lev][m][j].pass[k] ];

        
        MPI_Sendrecv(S[k],E->parallel.NUM_NODE[lev][m].pass[k],MPI_FLOAT,
             E->parallel.PROCESSOR[lev][m].pass[k],1,
                     R[k],E->parallel.NUM_NODE[lev][m].pass[k],MPI_FLOAT,
             E->parallel.PROCESSOR[lev][m].pass[k],1,MPI_COMM_WORLD,&status);

        for (j=1;j<=E->parallel.NUM_NODE[lev][m].pass[k];j++)
          U[m][ E->parallel.EXCHANGE_NODE[lev][m][j].pass[k] ] += R[k][j-1];


        }
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
 static int idd[NCS][NCS];
/* static float *S[73],*R[73];  */
 static float *S[27],*R[27];  
 static int been_here = 0;
 static int mid_recv, sizeofk;

 MPI_Status status;

 if (been_here ==0 )   {
   for (m=1;m<=E->sphere.caps_per_proc;m++)    {
     for (k=1;k<=E->parallel.sTNUM_PASS[lev][m];k++)  {
       sizeofk = (1+2*E->parallel.NUM_sNODE[lev][m].pass[k])*sizeof(float);
       S[k]=(float *)malloc( sizeofk );
       R[k]=(float *)malloc( sizeofk );
       }
     }
   been_here ++;
   }

    for (m=1;m<=E->sphere.caps_per_proc;m++)   {
       for (k=1;k<=E->parallel.sTNUM_PASS[lev][m];k++)  {
         for (j=1;j<=E->parallel.NUM_sNODE[lev][m].pass[k];j++)  {
          S[k][j-1] = U1[m][ E->parallel.EXCHANGE_sNODE[lev][m][j].pass[k] ];
          S[k][j-1+E->parallel.NUM_sNODE[lev][m].pass[k]]
                    = U2[m][ E->parallel.EXCHANGE_sNODE[lev][m][j].pass[k] ];
          }

        MPI_Sendrecv(S[k],2*E->parallel.NUM_sNODE[lev][m].pass[k],MPI_FLOAT,
             E->parallel.sPROCESSOR[lev][m].pass[k],1,
                     R[k],2*E->parallel.NUM_sNODE[lev][m].pass[k],MPI_FLOAT,
             E->parallel.sPROCESSOR[lev][m].pass[k],1,MPI_COMM_WORLD,&status);

        for (j=1;j<=E->parallel.NUM_sNODE[lev][m].pass[k];j++)   {
          U1[m][ E->parallel.EXCHANGE_sNODE[lev][m][j].pass[k] ] += R[k][j-1];
          U2[m][ E->parallel.EXCHANGE_sNODE[lev][m][j].pass[k] ] +=
                       R[k][j-1+E->parallel.NUM_sNODE[lev][m].pass[k]];
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


   rootid = E->parallel.me_sph*E->parallel.nprocz; /* which is the bottom cpu */
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
           MPI_Send(SD,E->lmesh.NEQ[lev],MPI_DOUBLE,processors[d],rootid,MPI_COMM_WORLD);
	   }
        else
           for (i=0;i<=E->lmesh.NEQ[lev];i++)   
	      AUo[m][i] = SD[i];
        }
    else
        MPI_Recv(AUo[m],E->lmesh.NEQ[lev],MPI_DOUBLE,rootid,rootid,MPI_COMM_WORLD,&status);
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
       MPI_Send(AUi[m],E->lmesh.NEQ[lev],MPI_DOUBLE,rootid,E->parallel.me,MPI_COMM_WORLD);
    else
       for (d=0;d<E->parallel.nprocz;d++) {
         if (processors[d]!=rootid)
	   MPI_Recv(RV,E->lmesh.NEQ[lev],MPI_DOUBLE,processors[d],processors[d],MPI_COMM_WORLD,&status);
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


   rootid = E->parallel.me_sph*E->parallel.nprocz; /* which is the bottom cpu */
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
       MPI_Send(AUi[m],E->lmesh.NNO[lev]+1,MPI_FLOAT,rootid,E->parallel.me,MPI_COMM_WORLD);
         for (node=1;node<=NNO;node++)
           AUo[m][node] = 1.0;
       }
    else 
       for (d=0;d<E->parallel.nprocz;d++) {
	 if (processors[d]!=rootid)
           MPI_Recv(RV,E->lmesh.NNO[lev]+1,MPI_FLOAT,processors[d],processors[d],MPI_COMM_WORLD,&status);
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
       MPI_Send(AUi[m],E->lmesh.NEL[lev]+1,MPI_FLOAT,rootid,E->parallel.me,MPI_COMM_WORLD);
         for (e=1;e<=NNO;e++)
           AUo[m][e] = 1.0;
       }
    else 
       for (d=0;d<E->parallel.nprocz;d++) {
	 if (processors[d]!=rootid)
           MPI_Recv(RV,E->lmesh.NEL[lev]+1,MPI_FLOAT,processors[d],processors[d],MPI_COMM_WORLD,&status);
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
   for (i=1;i<E->parallel.nprocxy;i++)
     RG[i] = ( float *)malloc((E->sphere.nsf+1)*sizeof(float));
   }

 idb=0;
 for (i=1;i<=E->parallel.nprocxy;i++)  {
   to_everyone = E->parallel.nprocz*(i-1) + E->parallel.me_loc[3];

   if (me!=to_everyone)    {  /* send TG */
     idb++;
     mst = me;
     MPI_Isend(TG,nsl,MPI_FLOAT,to_everyone,mst,MPI_COMM_WORLD,&request[idb-1]);
     }
   }

/* parallel_process_sync(); */

 idb=0;
 for (i=1;i<=E->parallel.nprocxy;i++)  {
   from_proc = E->parallel.nprocz*(i-1) + E->parallel.me_loc[3];
   if (me!=from_proc)   {    /* me==0 receive all TG and add them up */
      mst = from_proc;
      idb++;
      MPI_Irecv(RG[idb],nsl,MPI_FLOAT,from_proc,mst,MPI_COMM_WORLD,&request[idb-1]);
      }
   }

 MPI_Waitall(idb,request,status);

 for (i=1;i<E->parallel.nprocxy;i++)
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
     MPI_Send(TG,nsl,MPI_FLOAT,to_proc,mst,MPI_COMM_WORLD);
     }

 parallel_process_sync();

 if (E->parallel.me_loc[3]==dest_proc)  {
   for (i=1;i<E->parallel.nprocz;i++) {
      from_proc = me - i;
      mst = from_proc;
      MPI_Recv(RG,nsl,MPI_FLOAT,from_proc,mst,MPI_COMM_WORLD,&status1);

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

 for (i=1;i<E->parallel.nprocxy;i++)
    RG[i] = ( float *)malloc((nsl+1)*sizeof(float));


 idb=0;
 for (i=1;i<=E->parallel.nprocxy;i++)  {
   to_everyone = E->parallel.nprocz*(i-1) + loc_proc;

   if (me!=to_everyone)    {  /* send TG */
     idb++;
     mst = me;
     MPI_Isend(TG,nsl,MPI_FLOAT,to_everyone,mst,MPI_COMM_WORLD,&request[idb-1]);
     }
   }

/* parallel_process_sync(); */

 idb=0;
 for (i=1;i<=E->parallel.nprocxy;i++)  {
   from_proc = E->parallel.nprocz*(i-1) + loc_proc;
   if (me!=from_proc)   {    /* me==0 receive all TG and add them up */
      mst = from_proc;
      idb++;
      MPI_Irecv(RG[idb],nsl,MPI_FLOAT,from_proc,mst,MPI_COMM_WORLD,&request[idb-1]);
      }
   }

 MPI_Waitall(idb,request,status);

 for (i=1;i<E->parallel.nprocxy;i++)
   for (j=0;j<nsl; j++)  {
        TG[j] += RG[i][j];
        }

/* parallel_process_sync(); */

 for (i=1;i<E->parallel.nprocxy;i++)
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



