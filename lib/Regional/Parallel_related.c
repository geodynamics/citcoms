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

  int i,j,k,me,temp,pid_surf;

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

  set_horizontal_communicator(E);
  set_vertical_communicator(E);

if (E->control.verbose)
  for (j=1;j<=E->sphere.caps;j++)     {
    fprintf(E->fp_out,"j=%d pid_surf=%d capid=%d local_capid=%d \n",j,E->sphere.pid_surf[j],E->sphere.capid[j],E->sphere.local_capid[ E->sphere.capid[j] ]);
    for (i=1;i<=E->sphere.max_connections;i++)
       fprintf(E->fp_out,"j=%d i=%d target_cap=%d \n",j,i,E->sphere.cap[j].connection[i]);
    }

  return;
  }


void set_horizontal_communicator(struct All_variables *E)
{
  MPI_Group world_g, horizon_g;
  int j;
  int *processors;

  processors = (int *) malloc((E->parallel.nprocxy+2)*sizeof(int));

  for (j=0;j<E->parallel.nprocxy;j++) {
    processors[j] = E->parallel.me_loc[3] + j * E->parallel.nprocz;
  }

  MPI_Comm_group(E->parallel.world, &world_g);
  MPI_Group_incl(world_g, E->parallel.nprocxy, processors, &horizon_g);
  MPI_Comm_create(E->parallel.world, horizon_g, &(E->parallel.horizontal_comm));

  MPI_Group_free(&horizon_g);
  MPI_Group_free(&world_g);
  free((void *) processors);

  return;
}


void set_vertical_communicator(struct All_variables *E)
{
  MPI_Group world_g, vertical_g;
  int j;
  int *processors;

  processors = (int *)malloc((E->parallel.nprocz+2)*sizeof(int));

  for (j=0;j<E->parallel.nprocz;j++) {
    processors[j] = E->parallel.me_sph * E->parallel.nprocz
                  + E->parallel.nprocz - 1 - j;
  }

  MPI_Comm_group(E->parallel.world, &world_g);
  MPI_Group_incl(world_g, E->parallel.nprocz, processors, &vertical_g);
  MPI_Comm_create(E->parallel.world, vertical_g, &(E->parallel.vertical_comm));

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

 int ii,j,jj,m,k;
 double *S[27],*R[27];
 int sizeofk;

 MPI_Status status;

 for (m=1;m<=E->sphere.caps_per_proc;m++)    {
   for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)  {
     sizeofk = (1+E->parallel.NUM_NEQ[lev][m].pass[k])*sizeof(double);
     S[k]=(double *)malloc( sizeofk );
     R[k]=(double *)malloc( sizeofk );
   }
 }

 for (m=1;m<=E->sphere.caps_per_proc;m++)   {
   for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)  {

     for (j=1;j<=E->parallel.NUM_NEQ[lev][m].pass[k];j++)
       S[k][j-1] = U[m][ E->parallel.EXCHANGE_ID[lev][m][j].pass[k] ];

     MPI_Sendrecv(S[k],E->parallel.NUM_NEQ[lev][m].pass[k],MPI_DOUBLE,
		  E->parallel.PROCESSOR[lev][m].pass[k],1,
		  R[k],E->parallel.NUM_NEQ[lev][m].pass[k],MPI_DOUBLE,
		  E->parallel.PROCESSOR[lev][m].pass[k],1,
		  E->parallel.world,&status);

     for (j=1;j<=E->parallel.NUM_NEQ[lev][m].pass[k];j++)
       U[m][ E->parallel.EXCHANGE_ID[lev][m][j].pass[k] ] += R[k][j-1];

   }           /* for k */
 }     /* for m */         /* finish sending */

 for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)  {
   free((void*) S[k]);
   free((void*) R[k]);
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

 int ii,j,jj,m,k;
 double *S[27],*R[27];
 int sizeofk;

 MPI_Status status;

 for (m=1;m<=E->sphere.caps_per_proc;m++)    {
   for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)  {
     sizeofk = (1+E->parallel.NUM_NODE[lev][m].pass[k])*sizeof(double);
     S[k]=(double *)malloc( sizeofk );
     R[k]=(double *)malloc( sizeofk );
   }   /* end for k */
 }

 for(m=1;m<=E->sphere.caps_per_proc;m++)     {
   for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)   {

     for (j=1;j<=E->parallel.NUM_NODE[lev][m].pass[k];j++)
       S[k][j-1] = U[m][ E->parallel.EXCHANGE_NODE[lev][m][j].pass[k] ];

     MPI_Sendrecv(S[k],E->parallel.NUM_NODE[lev][m].pass[k],MPI_DOUBLE,
		  E->parallel.PROCESSOR[lev][m].pass[k],1,
		  R[k],E->parallel.NUM_NODE[lev][m].pass[k],MPI_DOUBLE,
		  E->parallel.PROCESSOR[lev][m].pass[k],1,
		  E->parallel.world,&status);

     for (j=1;j<=E->parallel.NUM_NODE[lev][m].pass[k];j++)
       U[m][ E->parallel.EXCHANGE_NODE[lev][m][j].pass[k] ] += R[k][j-1];
   }
 }

 for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)  {
   free((void*) S[k]);
   free((void*) R[k]);
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

 int ii,j,jj,m,k;
 float *S[27],*R[27];
 int sizeofk;

 MPI_Status status;

 for (m=1;m<=E->sphere.caps_per_proc;m++)    {
   for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)  {
     sizeofk = (1+E->parallel.NUM_NODE[lev][m].pass[k])*sizeof(float);
     S[k]=(float *)malloc( sizeofk );
     R[k]=(float *)malloc( sizeofk );
   }   /* end for k */
 }


 for (m=1;m<=E->sphere.caps_per_proc;m++)     {
   for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)   {

     for (j=1;j<=E->parallel.NUM_NODE[lev][m].pass[k];j++)
       S[k][j-1] = U[m][ E->parallel.EXCHANGE_NODE[lev][m][j].pass[k] ];

     MPI_Sendrecv(S[k],E->parallel.NUM_NODE[lev][m].pass[k],MPI_FLOAT,
		  E->parallel.PROCESSOR[lev][m].pass[k],1,
		  R[k],E->parallel.NUM_NODE[lev][m].pass[k],MPI_FLOAT,
		  E->parallel.PROCESSOR[lev][m].pass[k],1,
		  E->parallel.world,&status);

     for (j=1;j<=E->parallel.NUM_NODE[lev][m].pass[k];j++)
       U[m][ E->parallel.EXCHANGE_NODE[lev][m][j].pass[k] ] += R[k][j-1];
   }
 }


 for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)  {
   free((void*) S[k]);
   free((void*) R[k]);
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

 int ii,j,k,m,kk,t_cap,idb,msginfo[8];
 float *S[27],*R[27];
 int sizeofk;

 MPI_Status status;

 for (m=1;m<=E->sphere.caps_per_proc;m++)    {
   for (k=1;k<=E->parallel.sTNUM_PASS[lev][m];k++)  {
     sizeofk = (1+2*E->parallel.NUM_sNODE[lev][m].pass[k])*sizeof(float);
     S[k]=(float *)malloc( sizeofk );
     R[k]=(float *)malloc( sizeofk );
   }
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
		  E->parallel.sPROCESSOR[lev][m].pass[k],1,
		  E->parallel.world,&status);

     for (j=1;j<=E->parallel.NUM_sNODE[lev][m].pass[k];j++)   {
       U1[m][ E->parallel.EXCHANGE_sNODE[lev][m][j].pass[k] ] += R[k][j-1];
       U2[m][ E->parallel.EXCHANGE_sNODE[lev][m][j].pass[k] ] +=
	 R[k][j-1+E->parallel.NUM_sNODE[lev][m].pass[k]];
     }

   }
 }

 for (k=1;k<=E->parallel.sTNUM_PASS[lev][m];k++)  {
   free((void*) S[k]);
   free((void*) R[k]);
 }

 return;
 }

