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
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"

int layers_r(struct All_variables *,float );
int layers(struct All_variables *,int);


/*========================================================
  Function to make the IEN array for a mesh of given
  dimension. IEN is an externally defined structure array

  NOTE: this is not really general enough for new elements:
  it should be done through a pre-calculated lookup table.
  ======================================================== */

void construct_ien(E)
     struct All_variables *E;

{
  int lev,p,q,r,rr,j;
  int element,start,nel,nno;
  int elz,elx,ely,nox,noy,noz;

  const int dims=E->mesh.nsd;
  const int ends=enodes[dims];

  for (lev=E->mesh.levmax;lev>=E->mesh.levmin;lev--)  {

      elx = E->lmesh.ELX[lev];
      elz = E->lmesh.ELZ[lev];
      ely = E->lmesh.ELY[lev];
      nox = E->lmesh.NOX[lev];
      noz = E->lmesh.NOZ[lev];
      noy = E->lmesh.NOY[lev];
      nel=E->lmesh.NEL[lev];
      nno=E->lmesh.NNO[lev];

      for(r=1;r<=ely;r++)
        for(q=1;q<=elx;q++)
          for(p=1;p<=elz;p++)     {
             element = (r-1)*elx*elz + (q-1)*elz  + p;
             start = (r-1)*noz*nox + (q-1)*noz + p;
             for(rr=1;rr<=ends;rr++)
               E->IEN[lev][element].node[rr]= start
                  + offset[rr].vector[0]
                  + offset[rr].vector[1]*noz
                  + offset[rr].vector[2]*noz*nox;
	     }

    }     /* end loop for lev */


/* if(E->control.verbose)  { */
/*   for (lev=E->mesh.levmax;lev>=E->mesh.levmin;lev--)  { */
/*     fprintf(E->fp_out,"output_IEN_arrays me=%d lev=%d \n",E->parallel.me,lev); */
/*   for (j=1;j<=E->sphere.caps_per_proc;j++) { */
/*     fprintf(E->fp_out,"output_IEN_arrays me=%d %d %d\n",E->parallel.me,j,E->sphere.capid[j]); */
/*     for (i=1;i<=E->lmesh.NEL[lev];i++) */
/*        fprintf(E->fp_out,"%d %d %d %d %d %d %d %d %d\n",i,E->IEN[lev][j][i].node[1],E->IEN[lev][j][i].node[2],E->IEN[lev][j][i].node[3],E->IEN[lev][j][i].node[4],E->IEN[lev][j][i].node[5],E->IEN[lev][j][i].node[6],E->IEN[lev][j][i].node[7],E->IEN[lev][j][i].node[8]); */
/*     } */
/*     } */
/*   fflush (E->fp_out); */
/*   } */
}


/*  determine surface things */

void construct_surface( struct All_variables *E)
{
  int i, j, e, element;

    e = 0;
    for(element=1;element<=E->lmesh.nel;element++)
      if ( element%E->lmesh.elz==0) { /* top */
        e ++;
        E->sien[e].node[1] = E->ien[element].node[5]/E->lmesh.noz;
        E->sien[e].node[2] = E->ien[element].node[6]/E->lmesh.noz;
        E->sien[e].node[3] = E->ien[element].node[7]/E->lmesh.noz;
        E->sien[e].node[4] = E->ien[element].node[8]/E->lmesh.noz;
        E->surf_element[e] = element;
        }

    E->lmesh.snel = e;
    for (i=1;i<=E->lmesh.nsf;i++)
      E->surf_node[i] = i*E->lmesh.noz;


  if(E->control.verbose) {
      for(e=1;e<=E->lmesh.snel;e++) {
        fprintf(E->fp_out, "sien sel=%d node=%d %d %d %d\n",
		e, E->sien[e].node[1], E->sien[e].node[2], E->sien[e].node[3], E->sien[e].node[4]);
      }
  }
}


/*============================================
  Function to make the ID array for above case
  ============================================ */

void construct_id(E)
     struct All_variables *E;
{
    int i,j,k;
    int eqn_count,node,nno;
    int neq, gneq;
    unsigned int type,doff;
    int lev;
    void get_bcs_id_for_residual();

    const int dims=E->mesh.nsd,dofs=E->mesh.dof;
    const int ends=enodes[dims];

  for(lev=E->mesh.gridmax;lev>=E->mesh.gridmin;lev--)  {
      eqn_count = 0;

      for(node=1;node<=E->lmesh.NNO[lev];node++)
        for(doff=1;doff<=dims;doff++)  {
          E->ID[lev][node].doff[doff] = eqn_count;
          eqn_count ++;
          }

      E->lmesh.NEQ[lev] = eqn_count;

      i = 0;
      for(node=1;node<=E->lmesh.NNO[lev];node++) {
        if (E->NODE[lev][node] & SKIP)
        for(doff=1;doff<=dims;doff++)  {
	  i++;
          E->parallel.Skip_id[lev][i] = E->ID[lev][node].doff[doff];
          }
        }

      E->parallel.Skip_neq[lev] = i;

      /* global # of unskipped eqn */
      neq = E->lmesh.NEQ[lev] - E->parallel.Skip_neq[lev];
      MPI_Allreduce(&neq, &gneq, 1, MPI_INT, MPI_SUM, E->parallel.world);
      E->mesh.NEQ[lev] = gneq;

      get_bcs_id_for_residual(E,lev);

    }      /* end for lev */

    E->lmesh.neq = E->lmesh.NEQ[E->mesh.levmax];
    E->mesh.neq = E->mesh.NEQ[E->mesh.levmax];

/*     if (E->control.verbose) { */
/*       fprintf(E->fp_out,"output_ID_arrays \n"); */
/*       for(j=1;j<=E->sphere.caps_per_proc;j++)    */
/*         for (i=1;i<=E->lmesh.nno;i++) */
/*           fprintf(E->fp_out,"%d %d %d %d %d\n",eqn_count,i,E->ID[lev][j][i].doff[1],E->ID[lev][j][i].doff[2],E->ID[lev][j][i].doff[3]); */
/*       fflush(E->fp_out); */
/*       } */
}


void get_bcs_id_for_residual(E,level)
    struct All_variables *E;
    int level;
  {

    int i,j;

    const int nno=E->lmesh.NNO[level];

   j = 0;
   for(i=1;i<=nno;i++) {
      if ( (E->NODE[level][i] & VBX) != 0 )  {
	j++;
        E->zero_resid[level][j] = E->ID[level][i].doff[1];
	}
      if ( (E->NODE[level][i] & VBY) != 0 )  {
	j++;
        E->zero_resid[level][j] = E->ID[level][i].doff[2];
	}
      if ( (E->NODE[level][i] & VBZ) != 0 )  {
	j++;
        E->zero_resid[level][j] = E->ID[level][i].doff[3];
	}
      }

    E->num_zero_resid[level] = j;
}

/*==========================================================
  Function to construct  the LM array from the ID and IEN arrays
  ========================================================== */

void construct_lm(E)
     struct All_variables *E;
{
  int i,j,a,e;
  int lev,eqn_no;
  int nel, nel2;

  const int dims=E->mesh.nsd,dofs=E->mesh.dof;
  const int ends=enodes[dims];
}


/* =====================================================
   Function to build the local node matrix indexing maps
   ===================================================== */

void construct_node_maps(E)
    struct All_variables *E;
{
    double time1,CPU_time0();

    int ii,noz,noxz,m,n,nn,lev,i,j,k,jj,kk,ia,ja,is,ie,js,je,ks,ke,doff;
    int neq,nno,dims2,matrix,nox,noy;

    const int dims=E->mesh.nsd,dofs=E->mesh.dof;
    const int ends=enodes[dims];
    int max_eqn;

  dims2 = dims-1;
  for(lev=E->mesh.gridmax;lev>=E->mesh.gridmin;lev--) {
       neq=E->lmesh.NEQ[lev];
       nno=E->lmesh.NNO[lev];
       noxz = E->lmesh.NOX[lev]*E->lmesh.NOZ[lev];
       noz = E->lmesh.NOZ[lev];
       noy = E->lmesh.NOY[lev];
       nox = E->lmesh.NOX[lev];
       max_eqn = 14*dims; // Is this 14 supposed to be NCS?
       matrix = max_eqn*nno;
       E->Node_map[lev][CPPR]=(int *) malloc (matrix*sizeof(int));

       for(i=0;i<matrix;i++)
	   E->Node_map[lev][CPPR][i] = neq;  /* neq indicates an invalid eqn # */

       for (ii=1;ii<=noy;ii++)
       for (jj=1;jj<=nox;jj++)
       for (kk=1;kk<=noz;kk++)  {
	 nn = kk + (jj-1)*noz+ (ii-1)*noxz;
	 for(doff=1;doff<=dims;doff++)
	   E->Node_map[lev][CPPR][(nn-1)*max_eqn+doff-1] = E->ID[lev][nn].doff[doff];

         ia = 0;
	 is=1; ie=dims2;
	 js=1; je=dims;
	 ks=1; ke=dims;
	 if (kk==1  ) ks=2;
	 if (kk==noz) ke=2;
	 if (jj==1  ) js=2;
	 if (jj==nox) je=2;
	 if (ii==1  ) is=2;
	 if (ii==noy) ie=2;
         for (i=is;i<=ie;i++)
           for (j=js;j<=je;j++)
             for (k=ks;k<=ke;k++)  {
               ja = nn-((2-i)*noxz + (2-j)*noz + 2-k);
               if (ja<nn)   {
		 ia++;
                 for (doff=1;doff<=dims;doff++)
                   E->Node_map[lev][CPPR][(nn-1)*max_eqn+ia*dims+doff-1]=E->ID[lev][ja].doff[doff];
                 }
               }
         }

       E->Eqn_k1[lev][CPPR] = (higher_precision *)malloc(matrix*sizeof(higher_precision));
       E->Eqn_k2[lev][CPPR] = (higher_precision *)malloc(matrix*sizeof(higher_precision));
       E->Eqn_k3[lev][CPPR] = (higher_precision *)malloc(matrix*sizeof(higher_precision));

       E->mesh.matrix_size[lev] = matrix;

       if(E->control.verbose) {
           fprintf(E->fp_out, "output Node_map lev=%d m=%d\n", lev, CPPR);
           fprintf(E->fp_out, "neq=%d nno=%d max_eqn=%d matrix=%d\n", neq, nno, max_eqn, matrix);
           for(i=0;i<matrix;i++)
               fprintf(E->fp_out, "%d %d\n", i, E->Node_map[lev][CPPR][i]);
       }
  }
}


void construct_node_ks(E)
     struct All_variables *E;
{
    int m,level,i,j,k,e;
    int node,node1,eqn1,eqn2,eqn3,loc0,loc1,loc2,loc3,found,element,index,pp,qq;
    int neq,nno,nel,max_eqn;

    double elt_K[24*24];
    double w1,w2,w3,ww1,ww2,ww3,zero;

    higher_precision *B1,*B2,*B3;

    void get_elt_k();
    void get_aug_k();
    void build_diagonal_of_K();
    void parallel_process_termination();

    const int dims=E->mesh.nsd,dofs=E->mesh.dof;
    const int ends=enodes[dims];
    const int lms=loc_mat_size[E->mesh.nsd];

    zero = 0.0;
    max_eqn = 14*dims;

   for(level=E->mesh.gridmax;level>=E->mesh.gridmin;level--)   {


        neq=E->lmesh.NEQ[level];
        nel=E->lmesh.NEL[level];
        nno=E->lmesh.NNO[level];
	for(i=0;i<neq;i++)
	    E->BI[level][i] = zero;
        for(i=0;i<E->mesh.matrix_size[level];i++) {
            E->Eqn_k1[level][CPPR][i] = zero;
            E->Eqn_k2[level][CPPR][i] = zero;
            E->Eqn_k3[level][CPPR][i] = zero;
            }

        for(element=1;element<=nel;element++) {

	    get_elt_k(E,element,elt_K,level,0);

	    if (E->control.augmented_Lagr)
	         get_aug_k(E,element,elt_K,level);

            build_diagonal_of_K(E,element,elt_K,level);

	    for(i=1;i<=ends;i++) {  /* i, is the node we are storing to */
	       node=E->IEN[level][element].node[i];

	       pp=(i-1)*dims;
	       w1=w2=w3=1.0;

	       loc0=(node-1)*max_eqn;

	       if(E->NODE[level][node] & VBX) w1=0.0;
	       if(E->NODE[level][node] & VBZ) w3=0.0;
	       if(E->NODE[level][node] & VBY) w2=0.0;

	       for(j=1;j<=ends;j++) { /* j is the node we are receiving from */
	         node1=E->IEN[level][element].node[j];

                        /* only for half of the matrix ,because of the symmetry */
                 if (node1<=node)  {

		    ww1=ww2=ww3=1.0;
		    qq=(j-1)*dims;
		    eqn1=E->ID[level][node1].doff[1];
		    eqn2=E->ID[level][node1].doff[2];
		    eqn3=E->ID[level][node1].doff[3];

		    if(E->NODE[level][node1] & VBX) ww1=0.0;
		    if(E->NODE[level][node1] & VBZ) ww3=0.0;
		    if(E->NODE[level][node1] & VBY) ww2=0.0;

		    /* search for direction 1*/

		    found=0;
		    for(k=0;k<max_eqn;k++)
		      if(E->Node_map[level][CPPR][loc0+k] == eqn1) { /* found, index next equation */
			    index=k;
			    found++;
			    break;
			}

		    assert(found /* direction 1 */);

		    E->Eqn_k1[level][CPPR][loc0+index] +=  w1*ww1*elt_K[pp*lms+qq]; /* direction 1 */
		    E->Eqn_k2[level][CPPR][loc0+index] +=  w2*ww1*elt_K[(pp+1)*lms+qq]; /* direction 1 */
		    E->Eqn_k3[level][CPPR][loc0+index] +=  w3*ww1*elt_K[(pp+2)*lms+qq]; /* direction 1 */

		     /* search for direction 2*/

		    found=0;
		    for(k=0;k<max_eqn;k++)
			if(E->Node_map[level][CPPR][loc0+k] == eqn2) { /* found, index next equation */
			    index=k;
			    found++;
			    break;
			}

		    assert(found /* direction 2 */);

		    E->Eqn_k1[level][CPPR][loc0+index] += w1*ww2*elt_K[pp*lms+qq+1]; /* direction 1 */
		    E->Eqn_k2[level][CPPR][loc0+index] += w2*ww2*elt_K[(pp+1)*lms+qq+1]; /* direction 2 */
		    E->Eqn_k3[level][CPPR][loc0+index] += w3*ww2*elt_K[(pp+2)*lms+qq+1]; /* direction 3 */

		    /* search for direction 3*/

                    found=0;
		    for(k=0;k<max_eqn;k++)
		    if(E->Node_map[level][CPPR][loc0+k] == eqn3) { /* found, index next equation */
			index=k;
			found++;
			break;
		        }

                    assert(found /* direction 3 */);

		    E->Eqn_k1[level][CPPR][loc0+index] += w1*ww3*elt_K[pp*lms+qq+2]; /* direction 1 */
        E->Eqn_k2[level][CPPR][loc0+index] += w2*ww3*elt_K[(pp+1)*lms+qq+2]; /* direction 2 */
		    E->Eqn_k3[level][CPPR][loc0+index] += w3*ww3*elt_K[(pp+2)*lms+qq+2]; /* direction 3 */

		    }   /* end for j */
		  }   /* end for node1<= node */
		}      /* end for i */
	    }            /* end for element */

     (E->solver.exchange_id_d)(E, E->BI[level], level);

        neq=E->lmesh.NEQ[level];

        for(j=0;j<neq;j++)                 {
            if(E->BI[level][j] ==0.0)  fprintf(stderr,"me= %d level %d, equation %d/%d has zero diagonal term\n",E->parallel.me,level,j,neq);
	    assert( E->BI[level][j] != 0 /* diagonal of matrix = 0, not acceptable */);
            E->BI[level][j]  = (double) 1.0/E->BI[level][j];
	    }


    }     /* end for level */
}

void rebuild_BI_on_boundary(E)
     struct All_variables *E;
{
    int m,level,i,j;
    int eqn1,eqn2,eqn3;

    higher_precision *B1,*B2,*B3;
    int *C;

    const int dims=E->mesh.nsd,dofs=E->mesh.dof;

    const int max_eqn = dims*14;

   for(level=E->mesh.gridmax;level>=E->mesh.gridmin;level--)   {
        for(j=0;j<=E->lmesh.NEQ[level];j++)
            E->temp[j]=0.0;

        for(i=1;i<=E->lmesh.NNO[level];i++)  {
            eqn1=E->ID[level][i].doff[1];
            eqn2=E->ID[level][i].doff[2];
            eqn3=E->ID[level][i].doff[3];

            C=E->Node_map[level][CPPR] + (i-1)*max_eqn;
            B1=E->Eqn_k1[level][CPPR]+(i-1)*max_eqn;
            B2=E->Eqn_k2[level][CPPR]+(i-1)*max_eqn;
            B3=E->Eqn_k3[level][CPPR]+(i-1)*max_eqn;

            for(j=3;j<max_eqn;j++) {
                E->temp[eqn1] += fabs(B1[j]);
                E->temp[eqn2] += fabs(B2[j]);
                E->temp[eqn3] += fabs(B3[j]);
                }

            for(j=0;j<max_eqn;j++)
                E->temp[C[j]] += fabs(B1[j]) + fabs(B2[j]) + fabs(B3[j]);

            }

     (E->solver.exchange_id_d)(E, E->temp, level);

        for(i=0;i<E->lmesh.NEQ[level];i++)  {
            E->temp[i] = E->temp[i] - 1.0/E->BI[level][i];
        }
        for(i=1;i<=E->lmesh.NNO[level];i++)
          if (E->NODE[level][i] & OFFSIDE)   {
            eqn1=E->ID[level][i].doff[1];
            eqn2=E->ID[level][i].doff[2];
            eqn3=E->ID[level][i].doff[3];
            E->BI[level][eqn1] = (double) 1.0/E->temp[eqn1];
            E->BI[level][eqn2] = (double) 1.0/E->temp[eqn2];
            E->BI[level][eqn3] = (double) 1.0/E->temp[eqn3];
          }


    }     /* end for level */
}


/* ============================================
   Function to set up the boundary condition
   masks and other indicators.
   ============================================  */

void construct_masks(E)		/* Add lid/edge masks/nodal weightings */
     struct All_variables *E;
{
  int i,j,k,l,node,el,elt;
  int lev,elx,elz,ely,nno,nox,noz,noy;

  for(lev=E->mesh.gridmax;lev>=E->mesh.gridmin;lev--) {
      elz = E->lmesh.ELZ[lev];
      ely = E->lmesh.ELY[lev];
      noy = E->lmesh.NOY[lev];
      noz = E->lmesh.NOZ[lev];
      nno = E->lmesh.NNO[lev];

        if (E->parallel.me_loc[3]==0 )
          for (i=1;i<=E->parallel.NUM_NNO[lev].bound[5];i++)   {
            node = E->parallel.NODE[lev][i].bound[5];
 	    E->NODE[lev][node] = E->NODE[lev][node] | TZEDGE;
	    }
        if ( E->parallel.me_loc[3]==E->parallel.nprocz-1 )
          for (i=1;i<=E->parallel.NUM_NNO[lev].bound[6];i++)   {
  	    node = E->parallel.NODE[lev][i].bound[6];
	    E->NODE[lev][node] = E->NODE[lev][node] | TZEDGE;
	    }
  }
}


/*   ==========================================
     build the sub-element reference matrices
     ==========================================   */

void construct_sub_element(E)
     struct All_variables *E;

{    int i,j,k,l,m;
     int lev,nox,noy,noz,nnn,elx,elz,ely,elzu,elxu,elt,eltu;


  for(lev=E->mesh.levmax-1;lev>=E->mesh.levmin;lev--) {
          elx = E->lmesh.ELX[lev];
	  elz = E->lmesh.ELZ[lev];
	  ely = E->lmesh.ELY[lev];
          nox = E->lmesh.NOX[lev];
          noy = E->lmesh.NOY[lev];
          noz = E->lmesh.NOZ[lev];
	  elz = E->lmesh.ELZ[lev];
	  ely = E->lmesh.ELY[lev];
	  elxu = 2 * elx;
	  elzu = 2 * elz;
          if (!E->control.NMULTIGRID)  {
             elzu = 1;
             if (lev == E->mesh.levmax-1)
                 elzu = E->lmesh.ELZ[E->mesh.levmax];
             }

	  for(i=1;i<=elx;i++)
	    for(j=1;j<=elz;j++)
	      for(k=1;k<=ely;k++)    {
		  elt = j + (i-1)*elz +(k-1)*elz*elx;
		  eltu = (j*2-1) + elzu *2*(i-1) + elxu*elzu*2*(k-1);

		  for(l=1;l<=enodes[E->mesh.nsd];l++)   {
		      E->EL[lev][elt].sub[l] = eltu
                                 + offset[l].vector[0]
                                 + offset[l].vector[1] * elzu
                                 + offset[l].vector[2] * elzu * elxu;
		      }
		  }
  }
}


void construct_elt_ks(E)
     struct All_variables *E;
{
    int e,el,lev,j,k,ii,m;
    void get_elt_k();
    void get_aug_k();
    void build_diagonal_of_K();

    const int dims=E->mesh.nsd;
    const int n=loc_mat_size[E->mesh.nsd];

/*     if(E->parallel.me==0) */
/* 	fprintf(stderr,"storing elt k matrices\n"); */

    for(lev=E->mesh.gridmin;lev<=E->mesh.gridmax;lev++)  {


	for(el=1;el<=E->lmesh.NEL[lev];el++)    {

	    get_elt_k(E,el,E->elt_k[lev][el].k,lev,0);

	    if (E->control.augmented_Lagr)
	        get_aug_k(E,el,E->elt_k[lev][el].k,lev);

            build_diagonal_of_K(E,el,E->elt_k[lev][el].k,lev);

	    }

      (E->solver.exchange_id_d)(E, E->BI[lev], lev);    /*correct BI   */

            for(j=0;j<E->lmesh.NEQ[lev];j++) {
	       if(E->BI[lev][j] ==0.0)  fprintf(stderr,"me= %d level %d, equation %d/%d has zero diagonal term\n",E->parallel.me,lev,j,E->lmesh.NEQ[lev]);
               assert( E->BI[lev][j] != 0 /* diagonal of matrix = 0, not acceptable */);
               E->BI[lev][j]  = (double) 1.0/E->BI[lev][j];
	       }

    }       /* end for level */
}



void construct_elt_gs(E)
     struct All_variables *E;
{ int m,el,lev,a;
  void get_elt_g();

  const int dims=E->mesh.nsd,dofs=E->mesh.dof;
  const int ends=enodes[dims];

/*   if(E->control.verbose && E->parallel.me==0) */
/*       fprintf(stderr,"storing elt g matrices\n"); */

  for(lev=E->mesh.gridmin;lev<=E->mesh.gridmax;lev++)
    for(el=1;el<=E->lmesh.NEL[lev];el++)
      get_elt_g(E,el,E->elt_del[lev][el].g,lev);
}


/*==============================================
  For compressible cases, construct c matrix,
  where  c = \frac{d rho_r}{dr} / rho_r * u_r
  ==============================================*/

void construct_elt_cs(struct All_variables *E)
{
    int m, el, lev;
    void get_elt_c();

/*     if(E->control.verbose && E->parallel.me==0) */
/*         fprintf(stderr,"storing elt c matrices\n"); */

    for(lev=E->mesh.gridmin;lev<=E->mesh.gridmax;lev++)
            for(el=1;el<=E->lmesh.NEL[lev];el++) {
                get_elt_c(E,el,E->elt_c[lev][el].c,lev);
            }
}


/* ==============================================================
 routine for constructing stiffness and node_maps
 ============================================================== */

void construct_stiffness_B_matrix(E)
  struct All_variables *E;
{
  void build_diagonal_of_K();
  void build_diagonal_of_Ahat();
  void project_viscosity();
  void construct_node_maps();
  void construct_node_ks();
  void construct_elt_ks();
  void rebuild_BI_on_boundary();

  if (E->control.NMULTIGRID)
    project_viscosity(E);

  if (E->control.NMULTIGRID || E->control.NASSEMBLE) {
    construct_node_ks(E);
  }
  else {
    construct_elt_ks(E);
  }

  build_diagonal_of_Ahat(E);

  if (E->control.NMULTIGRID || (E->control.NASSEMBLE && !E->control.CONJ_GRAD))
    rebuild_BI_on_boundary(E);
}

/* took this apart to allow call from other subroutines */

/* 


determine viscosity layer number based on radial coordinate r

if E->viscosity.z... set to Earth values, and old, num_mat=4 style is
used then

1: lithosphere 2: 100-410 3: 410-660 and 4: lower mantle

if z_layer is used, the layer numbers will refer to those read in with
z_layer

*/
int layers_r(struct All_variables *E,float r)
{
  int llayers, i;
  float rl;
  /* 
     the z-values, as read in, are non-dimensionalized depth
     convert to radii

  */
  rl = r + E->sphere.ro;
  llayers = 0;
  for(i = 0;i < E->viscosity.num_mat;i++)
    if(r > (E->sphere.ro - E->viscosity.zbase_layer[i])){
      i++;
      break;
    }
  llayers = i;
  
  return (llayers);
}

/* determine layer number of node "node" of cap "m" */
int layers(struct All_variables *E,int node)
{
  return(layers_r(E,E->sx[3][node]));
}


/* ==============================================================
 construct array mat




 ============================================================== */
void construct_mat_group(E)
     struct All_variables *E;
{
  int m,i,j,k,kk,el,lev,a,nodea,els,llayer;
  void read_visc_layer_file(struct All_variables *E);

  const int dims=E->mesh.nsd,dofs=E->mesh.dof;
  const int ends=enodes[dims];

  if(E->viscosity.layer_control) {
      read_visc_layer_file(E);

      /* assign the global nz to mat group */
          for(el=1;el<=E->lmesh.nel;el++) {
              int nz;
              nz = ((el-1) % E->lmesh.elz) + 1;
              E->mat[el] = E->mesh.elz - (nz + E->lmesh.ezs) + 1;
          }
  } else {
          for(el=1;el<=E->lmesh.nel;el++) {
              E->mat[el] = 1;
              nodea = E->ien[el].node[2];
              llayer = layers(E,nodea);
              if (llayer)  {
                  E->mat[el] = llayer;
              }
          }
  }
}
