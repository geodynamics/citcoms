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
#include <mpi.h>

#include <math.h>
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"
#ifdef ALLOW_ELLIPTICAL
double theta_g(double , struct All_variables *);
#endif

void calc_cbase_at_tp(float , float , float *);
void myerror(struct All_variables *E,char *message);

/* ===============================================
   strips horizontal average from nodal field X.
   Assumes orthogonal mesh, otherwise, horizontals
   aren't & another method is required.
   =============================================== */

void remove_horiz_ave(E,X,H,store_or_not)
     struct All_variables *E;
     double **X, *H;
     int store_or_not;

{
    int m,i,j,k,n,nox,noz,noy;
    void return_horiz_ave();

    const int dims = E->mesh.nsd;

    noy = E->lmesh.noy;
    noz = E->lmesh.noz;
    nox = E->lmesh.nox;

    return_horiz_ave(E,X,H);

    for(k=1;k<=noy;k++)
      for(j=1;j<=nox;j++)
	for(i=1;i<=noz;i++) {
            n = i+(j-1)*noz+(k-1)*noz*nox;
            X[CPPR][n] -= H[i];
	}
}


void remove_horiz_ave2(struct All_variables *E, double **X)
{
    double *H;

    H = (double *)malloc( (E->lmesh.noz+1)*sizeof(double));
    remove_horiz_ave(E, X, H, 0);
    free ((void *) H);
}


void return_horiz_ave(E,X,H)
     struct All_variables *E;
     double **X, *H;
{
  const int dims = E->mesh.nsd;
  int m,i,j,k,d,nint,noz,nox,noy,el,elz,elx,ely,j1,j2,i1,i2,k1,k2,nproc;
  int top,lnode[5], sizeofH, noz2,iroot;
  double *Have,*temp,aa[5];
  struct Shape_function1 M;
  struct Shape_function1_dA dGamma;
  void get_global_1d_shape_fn();

  sizeofH = (2*E->lmesh.noz+2)*sizeof(double);

  Have = (double *)malloc(sizeofH);
  temp = (double *)malloc(sizeofH);

  noz = E->lmesh.noz;
  noy = E->lmesh.noy;
  elz = E->lmesh.elz;
  elx = E->lmesh.elx;
  ely = E->lmesh.ely;
  noz2 = 2*noz;

  for (i=1;i<=elz;i++)  {
    temp[i] = temp[i+noz] = 0.0;
    temp[i+1] = temp[i+1+noz] = 0.0;
    top = 0;
    if (i==elz) top = 1;
      for (k=1;k<=ely;k++)
        for (j=1;j<=elx;j++)     {
          el = i + (j-1)*elz + (k-1)*elx*elz;
          get_global_1d_shape_fn(E,el,&M,&dGamma,top);

          lnode[1] = E->ien[el].node[1];
          lnode[2] = E->ien[el].node[2];
          lnode[3] = E->ien[el].node[3];
          lnode[4] = E->ien[el].node[4];

          for(nint=1;nint<=onedvpoints[E->mesh.nsd];nint++)   {
            for(d=1;d<=onedvpoints[E->mesh.nsd];d++)
              temp[i] += X[CPPR][lnode[d]] * E->M.vpt[GMVINDEX(d,nint)]
                          * dGamma.vpt[GMVGAMMA(0,nint)];
            temp[i+noz] += dGamma.vpt[GMVGAMMA(0,nint)];
            }

          if (i==elz)  {
            lnode[1] = E->ien[el].node[5];
            lnode[2] = E->ien[el].node[6];
            lnode[3] = E->ien[el].node[7];
            lnode[4] = E->ien[el].node[8];

            for(nint=1;nint<=onedvpoints[E->mesh.nsd];nint++)   {
              for(d=1;d<=onedvpoints[E->mesh.nsd];d++)
                temp[i+1] += X[CPPR][lnode[d]] * E->M.vpt[GMVINDEX(d,nint)]
                          * dGamma.vpt[GMVGAMMA(1,nint)];
              temp[i+1+noz] += dGamma.vpt[GMVGAMMA(1,nint)];
              }

            }   /* end of if i==elz    */
          }   /* end of j  and k, and m  */
     }        /* Done for i */

  MPI_Allreduce(temp,Have,noz2+1,MPI_DOUBLE,MPI_SUM,E->parallel.horizontal_comm);

  for (i=1;i<=noz;i++) {
    if(Have[i+noz] != 0.0)
       H[i] = Have[i]/Have[i+noz];
    }
 /* if (E->parallel.me==0)
    for(i=1;i<=noz;i++)
      fprintf(stderr,"area %d %d %g\n",E->parallel.me,i,Have[i+noz]);
*/
  free ((void *) Have);
  free ((void *) temp);
}

void return_horiz_ave_f(E,X,H)
     struct All_variables *E;
     float **X, *H;
{
  const int dims = E->mesh.nsd;
  int m,i,j,k,d,nint,noz,nox,noy,el,elz,elx,ely,j1,j2,i1,i2,k1,k2,nproc;
  int top,lnode[5], sizeofH, noz2,iroot;
  float *Have,*temp,aa[5];
  struct Shape_function1 M;
  struct Shape_function1_dA dGamma;
  void get_global_1d_shape_fn();

  sizeofH = (2*E->lmesh.noz+2)*sizeof(float);

  Have = (float *)malloc(sizeofH);
  temp = (float *)malloc(sizeofH);

  noz = E->lmesh.noz;
  noy = E->lmesh.noy;
  elz = E->lmesh.elz;
  elx = E->lmesh.elx;
  ely = E->lmesh.ely;
  noz2 = 2*noz;

  for (i=1;i<=elz;i++)  {
    temp[i] = temp[i+noz] = 0.0;
    temp[i+1] = temp[i+1+noz] = 0.0;
    top = 0;
    if (i==elz) top = 1;
      for (k=1;k<=ely;k++)
        for (j=1;j<=elx;j++)     {
          el = i + (j-1)*elz + (k-1)*elx*elz;
          get_global_1d_shape_fn(E,el,&M,&dGamma,top);

          lnode[1] = E->ien[el].node[1];
          lnode[2] = E->ien[el].node[2];
          lnode[3] = E->ien[el].node[3];
          lnode[4] = E->ien[el].node[4];

          for(nint=1;nint<=onedvpoints[E->mesh.nsd];nint++)   {
            for(d=1;d<=onedvpoints[E->mesh.nsd];d++)
              temp[i] += X[CPPR][lnode[d]] * E->M.vpt[GMVINDEX(d,nint)]
                          * dGamma.vpt[GMVGAMMA(0,nint)];
            temp[i+noz] += dGamma.vpt[GMVGAMMA(0,nint)];
            }

          if (i==elz)  {
            lnode[1] = E->ien[el].node[5];
            lnode[2] = E->ien[el].node[6];
            lnode[3] = E->ien[el].node[7];
            lnode[4] = E->ien[el].node[8];

            for(nint=1;nint<=onedvpoints[E->mesh.nsd];nint++)   {
              for(d=1;d<=onedvpoints[E->mesh.nsd];d++)
                temp[i+1] += X[CPPR][lnode[d]] * E->M.vpt[GMVINDEX(d,nint)]
                          * dGamma.vpt[GMVGAMMA(1,nint)];
              temp[i+1+noz] += dGamma.vpt[GMVGAMMA(1,nint)];
              }

            }   /* end of if i==elz    */
          }   /* end of j  and k, and m  */
     }        /* Done for i */

  MPI_Allreduce(temp,Have,noz2+1,MPI_FLOAT,MPI_SUM,E->parallel.horizontal_comm);

  for (i=1;i<=noz;i++) {
    if(Have[i+noz] != 0.0)
       H[i] = Have[i]/Have[i+noz];
    }
 /* if (E->parallel.me==0)
    for(i=1;i<=noz;i++)
      fprintf(stderr,"area %d %d %g\n",E->parallel.me,i,Have[i+noz]);
*/
  free ((void *) Have);
  free ((void *) temp);
  return;
}


/******* RETURN ELEMENTWISE HORIZ AVE ********************************/
/*                                                                   */
/* This function is similar to return_horiz_ave in the citcom code   */
/* however here, elemental horizontal averages are given rather than */
/* nodal averages. Also note, here is average per element            */

void return_elementwise_horiz_ave(E,X,H)
     struct All_variables *E;
     double **X, *H;
{

  int m,i,j,k,d,noz,noy,el,elz,elx,ely,nproc;
  int sizeofH;
  int elz2;
  double *Have,*temp;

  sizeofH = (2*E->lmesh.elz+2)*sizeof(double);

  Have = (double *)malloc(sizeofH);
  temp = (double *)malloc(sizeofH);

  noz = E->lmesh.noz;
  noy = E->lmesh.noy;
  elz = E->lmesh.elz;
  elx = E->lmesh.elx;
  ely = E->lmesh.ely;
  elz2 = 2*elz;

  for (i=0;i<=(elz*2+1);i++)
  {
    temp[i]=0.0;
  }

  for (i=1;i<=elz;i++)
  {
    for (k=1;k<=ely;k++)
    {
      for (j=1;j<=elx;j++)
      {
        el = i + (j-1)*elz + (k-1)*elx*elz;
        temp[i] += X[CPPR][el]*E->ECO[E->mesh.levmax][CPPR][el].area;
        temp[i+elz] += E->ECO[E->mesh.levmax][CPPR][el].area;
      }
    }
  }



/* determine which processors should get the message from me for
               computing the layer averages */

  MPI_Allreduce(temp,Have,elz2+1,MPI_DOUBLE,MPI_SUM,E->parallel.horizontal_comm);

  for (i=1;i<=elz;i++) {
    if(Have[i+elz] != 0.0)
       H[i] = Have[i]/Have[i+elz];
    }


  free ((void *) Have);
  free ((void *) temp);
}

float return_bulk_value(E,Z,average)
     struct All_variables *E;
     float **Z;
     int average;

{
    int n,i,j,k,el,m;
    float volume,integral,volume1,integral1;

    const int vpts = vpoints[E->mesh.nsd];
    const int ends = enodes[E->mesh.nsd];

    volume1=0.0;
    integral1=0.0;

       for (el=1;el<=E->lmesh.nel;el++)  {

	  for(j=1;j<=vpts;j++)
	    for(i=1;i<=ends;i++) {
		n = E->ien[el].node[i];
		volume1 += E->N.vpt[GNVINDEX(i,j)] * E->gDA[CPPR][el].vpt[j];
		integral1 += Z[CPPR][n] * E->N.vpt[GNVINDEX(i,j)] * E->gDA[CPPR][el].vpt[j];
                }

          }


    MPI_Allreduce(&volume1  ,&volume  ,1,MPI_FLOAT,MPI_SUM,E->parallel.world);
    MPI_Allreduce(&integral1,&integral,1,MPI_FLOAT,MPI_SUM,E->parallel.world);

    if(average && volume != 0.0)
 	   integral /= volume;

    return((float)integral);
}

/************ RETURN BULK VALUE_D *****************************************/
/*                                                                        */
/* Same as return_bulk_value but allowing double instead of float.        */
/* I think when integer average =1, volume average is returned.           */
/*         when integer average =0, integral is returned.           */


double return_bulk_value_d(E,Z,average)
     struct All_variables *E;
     double **Z;
     int average;

{
    int n,i,j,el,m;
    double volume,integral,volume1,integral1;

    const int vpts = vpoints[E->mesh.nsd];
    const int ends = enodes[E->mesh.nsd];

    volume1=0.0;
    integral1=0.0;

       for (el=1;el<=E->lmesh.nel;el++)  {

          for(j=1;j<=vpts;j++)
            for(i=1;i<=ends;i++) {
                n = E->ien[el].node[i];
                volume1 += E->N.vpt[GNVINDEX(i,j)] * E->gDA[CPPR][el].vpt[j];
                integral1 += Z[CPPR][n] * E->N.vpt[GNVINDEX(i,j)] * E->gDA[CPPR][el].vpt[j];
            }

       }


    MPI_Allreduce(&volume1  ,&volume  ,1,MPI_DOUBLE,MPI_SUM,E->parallel.world);
    MPI_Allreduce(&integral1,&integral,1,MPI_DOUBLE,MPI_SUM,E->parallel.world);

    if(average && volume != 0.0)
           integral /= volume;

    return((double)integral);
}

/* ================================================== */
float find_max_horizontal(E,Tmax)
struct All_variables *E;
float Tmax;
{
 float ttmax;

 MPI_Allreduce(&Tmax,&ttmax,1,MPI_FLOAT,MPI_MAX,E->parallel.horizontal_comm);

 return(ttmax);
 }

/* ================================================== */
void sum_across_surface(E,data,total)
struct All_variables *E;
float *data;
int total;
{
 int j,d;
 float *temp;

 temp = (float *)malloc((total+1)*sizeof(float));
 MPI_Allreduce(data,temp,total,MPI_FLOAT,MPI_SUM,E->parallel.horizontal_comm);

 for (j=0;j<total;j++) {
   data[j] = temp[j];
 }

 free((void *)temp);

 return;
}

/* ================================================== */
/* ================================================== */

/* ================================================== */
void sum_across_surf_sph1(E,sphc,sphs)
struct All_variables *E;
float *sphc,*sphs;
{
 int jumpp,total,j,d;
 float *sphcs,*temp;

 temp = (float *) malloc((E->sphere.hindice*2)*sizeof(float));
 sphcs = (float *) malloc((E->sphere.hindice*2)*sizeof(float));

 /* pack */
 jumpp = E->sphere.hindice;
 total = E->sphere.hindice*2;
 for (j=0;j<E->sphere.hindice;j++)   {
   sphcs[j] = sphc[j];
   sphcs[j+jumpp] = sphs[j];
 }

 /* sum across processors in horizontal direction */
 MPI_Allreduce(sphcs,temp,total,MPI_FLOAT,MPI_SUM,E->parallel.horizontal_comm);

 /* unpack */
 for (j=0;j<E->sphere.hindice;j++)   {
   sphc[j] = temp[j];
   sphs[j] = temp[j+jumpp];
 }

 free((void *)temp);
 free((void *)sphcs);

 return;
}

/* ================================================== */


float global_fvdot(E,A,B,lev)
   struct All_variables *E;
   float **A,**B;
   int lev;

{
  int m,i,neq;
  float prod, temp,temp1;

  neq=E->lmesh.NEQ[lev];

  temp = 0.0;
  temp1 = 0.0;
  prod = 0.0;
    neq=E->lmesh.NEQ[lev];
    temp1 = 0.0;
    for (i=0;i<neq;i++)
      temp += A[CPPR][i]*B[CPPR][i];

    for (i=1;i<=E->parallel.Skip_neq[lev][CPPR];i++)
       temp1 += A[CPPR][E->parallel.Skip_id[lev][CPPR][i]]*B[CPPR][E->parallel.Skip_id[lev][CPPR][i]];

    temp -= temp1;


  MPI_Allreduce(&temp, &prod,1,MPI_FLOAT,MPI_SUM,E->parallel.world);

  return (prod);
}


double kineticE_radial(E,A,lev)
   struct All_variables *E;
   double **A;
   int lev;

{
  int m,i,neq;
  double prod, temp,temp1;

    temp = 0.0;
    prod = 0.0;

    neq=E->lmesh.NEQ[lev];
    temp1 = 0.0;
    for (i=0;i<neq;i++)
      if ((i+1)%3==0)
        temp += A[CPPR][i]*A[CPPR][i];

    for (i=1;i<=E->parallel.Skip_neq[lev][CPPR];i++)
      if ((E->parallel.Skip_id[lev][CPPR][i]+1)%3==0)
        temp1 += A[CPPR][E->parallel.Skip_id[lev][CPPR][i]]*A[CPPR][E->parallel.Skip_id[lev][CPPR][i]];

    temp -= temp1;


  MPI_Allreduce(&temp, &prod,1,MPI_DOUBLE,MPI_SUM,E->parallel.world);

  return (prod);
}

double global_vdot(E,A,B,lev)
   struct All_variables *E;
   double **A,**B;
   int lev;

{
  int m,i,neq;
  double prod, temp,temp1;

    temp = 0.0;
    prod = 0.0;

    neq=E->lmesh.NEQ[lev];
    temp1 = 0.0;
    for (i=0;i<neq;i++)
      temp += A[CPPR][i]*B[CPPR][i];

    for (i=1;i<=E->parallel.Skip_neq[lev][CPPR];i++)
       temp1 += A[CPPR][E->parallel.Skip_id[lev][CPPR][i]]*B[CPPR][E->parallel.Skip_id[lev][CPPR][i]];

    temp -= temp1;


  MPI_Allreduce(&temp, &prod,1,MPI_DOUBLE,MPI_SUM,E->parallel.world);

  return (prod);
}


double global_pdot(E,A,B,lev)
   struct All_variables *E;
   double **A,**B;
   int lev;

{
  int i,m,npno;
  double prod, temp;

  npno=E->lmesh.NPNO[lev];

  temp = 0.0;
  prod = 0.0;
    npno=E->lmesh.NPNO[lev];
    for (i=0;i<npno;i++)
      temp += A[CPPR][i]*B[CPPR][i];

  MPI_Allreduce(&temp, &prod,1,MPI_DOUBLE,MPI_SUM,E->parallel.world);

  return (prod);
}


/* return ||V||^2 */
double global_v_norm2(struct All_variables *E,  double **V)
{
    int i, m, d;
    int eqn1, eqn2, eqn3;
    double prod, temp;

    temp = 0.0;
    prod = 0.0;
    for (i=1; i<=E->lmesh.nno; i++) {
        eqn1 = E->id[CPPR][i].doff[1];
        eqn2 = E->id[CPPR][i].doff[2];
        eqn3 = E->id[CPPR][i].doff[3];
        /* L2 norm  */
        temp += (V[CPPR][eqn1] * V[CPPR][eqn1] +
                 V[CPPR][eqn2] * V[CPPR][eqn2] +
                 V[CPPR][eqn3] * V[CPPR][eqn3]) * E->NMass[CPPR][i];
    }

    MPI_Allreduce(&temp, &prod, 1, MPI_DOUBLE, MPI_SUM, E->parallel.world);

    return (prod/E->mesh.volume);
}


/* return ||P||^2 */
double global_p_norm2(struct All_variables *E,  double **P)
{
    int i, m;
    double prod, temp;

    temp = 0.0;
    prod = 0.0;
    for (i=0; i<E->lmesh.npno; i++) {
        /* L2 norm */ 
        /* should be E->eco[i].area after E->eco hase been made 0-based */
        temp += P[CPPR][i] * P[CPPR][i] * E->eco[i+1].area;
    }

    MPI_Allreduce(&temp, &prod, 1, MPI_DOUBLE, MPI_SUM, E->parallel.world);

    return (prod/E->mesh.volume);
}


/* return ||A||^2, where A_i is \int{div(u) d\Omega_i} */
double global_div_norm2(struct All_variables *E,  double **A)
{
    int i, m;
    double prod, temp;

    temp = 0.0;
    prod = 0.0;
    for (i=0; i<E->lmesh.npno; i++) {
        /* L2 norm of div(u) */
        /* should be E->eco[i].area after E->eco hase been made 0-based */
        temp += A[CPPR][i] * A[CPPR][i] / E->eco[i+1].area;

        /* L1 norm */
        /*temp += fabs(A[m][i]);*/
    }

    MPI_Allreduce(&temp, &prod, 1, MPI_DOUBLE, MPI_SUM, E->parallel.world);

    return (prod/E->mesh.volume);
}


double global_tdot_d(E,A,B,lev)
   struct All_variables *E;
   double **A,**B;
   int lev;

{
  int i,nno,m;
  double prod, temp;

  nno=E->lmesh.NNO[lev];

  temp = 0.0;
  prod = 0.0;
  nno=E->lmesh.NNO[lev];
  for (i=1;i<=nno;i++)
    if (!(E->NODE[lev][CPPR][i] & SKIP))
      temp += A[CPPR][i];

  MPI_Allreduce(&temp, &prod,1,MPI_DOUBLE,MPI_SUM,E->parallel.world);

  return (prod);
}

float global_tdot(E,A,B,lev)
   struct All_variables *E;
   float **A,**B;
   int lev;

{
  int i,nno,m;
  float prod, temp;


  temp = 0.0;
  prod = 0.0;
  nno=E->lmesh.NNO[lev];
  for (i=1;i<=nno;i++)
    if (!(E->NODE[lev][CPPR][i] & SKIP))
      temp += A[CPPR][i]*B[CPPR][i];

  MPI_Allreduce(&temp, &prod,1,MPI_FLOAT,MPI_SUM,E->parallel.world);

  return (prod);
}


float global_fmin(E,a)
   struct All_variables *E;
   float a;
{
  float temp;
  MPI_Allreduce(&a, &temp,1,MPI_FLOAT,MPI_MIN,E->parallel.world);
  return (temp);
  }

double global_dmax(E,a)
   struct All_variables *E;
   double a;
{
  double temp;
  MPI_Allreduce(&a, &temp,1,MPI_DOUBLE,MPI_MAX,E->parallel.world);
  return (temp);
  }


float global_fmax(E,a)
   struct All_variables *E;
   float a;
{
  float temp;
  MPI_Allreduce(&a, &temp,1,MPI_FLOAT,MPI_MAX,E->parallel.world);
  return (temp);
  }

double Tmaxd(E,T)
  struct All_variables *E;
  double **T;
{
  double global_dmax(),temp,temp1;
  int i,m;

  temp = -10.0;
  for(i=1;i<=E->lmesh.nno;i++)
    temp = max(T[CPPR][i],temp);

  temp1 = global_dmax(E,temp);
  return (temp1);
}


float Tmax(E,T)
  struct All_variables *E;
  float **T;
{
  float global_fmax(),temp,temp1;
  int i,m;

  temp = -10.0;
  for(i=1;i<=E->lmesh.nno;i++)
    temp = max(T[CPPR][i],temp);

  temp1 = global_fmax(E,temp);
  return (temp1);
}


double  vnorm_nonnewt(E,dU,U,lev)
  struct All_variables *E;
  double **dU,**U;
  int lev;
{
 double temp1,temp2,dtemp,temp;
 int a,e,i,m,node;
 const int dims = E->mesh.nsd;
 const int ends = enodes[dims];
 const int nel=E->lmesh.nel;

 dtemp=0.0;
 temp=0.0;
  for (e=1;e<=nel;e++)
   /*if (E->mat[m][e]==1)*/
     for (i=1;i<=dims;i++)
       for (a=1;a<=ends;a++) {
	 node = E->IEN[lev][CPPR][e].node[a];
         dtemp += dU[CPPR][ E->ID[lev][CPPR][node].doff[i] ]*
                  dU[CPPR][ E->ID[lev][CPPR][node].doff[i] ];
         temp += U[CPPR][ E->ID[lev][CPPR][node].doff[i] ]*
                 U[CPPR][ E->ID[lev][CPPR][node].doff[i] ];
         }


  MPI_Allreduce(&dtemp, &temp2,1,MPI_DOUBLE,MPI_SUM,E->parallel.world);
  MPI_Allreduce(&temp, &temp1,1,MPI_DOUBLE,MPI_SUM,E->parallel.world);

  temp1 = sqrt(temp2/temp1);

  return (temp1);
}


void sum_across_depth_sph1(E,sphc,sphs)
     struct All_variables *E;
     float *sphc,*sphs;
{
    int jumpp,total,j;

    float *sphcs,*temp;

    if (E->parallel.nprocz > 1)  {
	total = E->sphere.hindice*2;
	temp = (float *) malloc(total*sizeof(float));
	sphcs = (float *) malloc(total*sizeof(float));

	/* pack sphc[] and sphs[] into sphcs[] */
	jumpp = E->sphere.hindice;
	for (j=0;j<E->sphere.hindice;j++)   {
	    sphcs[j] = sphc[j];
	    sphcs[j+jumpp] = sphs[j];
	}

	/* sum across processors in z direction */
	MPI_Allreduce(sphcs, temp, total, MPI_FLOAT, MPI_SUM,
		      E->parallel.vertical_comm);

	/* unpack */
	for (j=0;j<E->sphere.hindice;j++)   {
	    sphc[j] = temp[j];
	    sphs[j] = temp[j+jumpp];
	}

	free(temp);
	free(sphcs);
    }
}


/* ================================================== */
/* ================================================== */
void broadcast_vertical(struct All_variables *E,
                        float *sphc, float *sphs,
                        int root)
{
    int jumpp, total, j;
    float *temp;

    if(E->parallel.nprocz == 1) return;

    jumpp = E->sphere.hindice;
    total = E->sphere.hindice*2;
    temp = (float *) malloc(total*sizeof(float));

    if (E->parallel.me_loc[3] == root) {
        /* pack */
        for (j=0; j<E->sphere.hindice; j++)   {
            temp[j] = sphc[j];
            temp[j+jumpp] = sphs[j];
        }
    }

    MPI_Bcast(temp, total, MPI_FLOAT, root, E->parallel.vertical_comm);

    if (E->parallel.me_loc[3] != root) {
        /* unpack */
        for (j=0; j<E->sphere.hindice; j++)   {
            sphc[j] = temp[j];
            sphs[j] = temp[j+jumpp];
        }
    }

    free((void *)temp);
}


/*
 * remove rigid body rotation or angular momentum from the velocity
 */

void remove_rigid_rot(struct All_variables *E)
{
    void velo_from_element_d();
    double myatan();
    double wx, wy, wz, v_theta, v_phi, cos_t,sin_t,sin_f, cos_f,frd;
    double vx[9], vy[9], vz[9];
    double r, t, f, efac,tg;
    float cart_base[9];
    double exyz[4], fxyz[4];

    int m, e, i, k, j, node;
    const int lev = E->mesh.levmax;
    const int nno = E->lmesh.nno;
    const int ends = ENODES3D;
    const int ppts = PPOINTS3D;
    const int vpts = VPOINTS3D;
    const int sphere_key = 1;
    double VV[4][9];
    double rot, fr, tr;
    double tmp, moment_of_inertia, rho;


    if(E->control.remove_angular_momentum) {
        moment_of_inertia = tmp = 0;
        for (i=1;i<=E->lmesh.elz;i++)
            tmp += (8.0*M_PI/15.0)*
                0.5*(E->refstate.rho[i] + E->refstate.rho[i+1])*
                (pow(E->sx[1][3][i+1],5.0) - pow(E->sx[1][3][i],5.0));

        MPI_Allreduce(&tmp, &moment_of_inertia, 1, MPI_DOUBLE,
                      MPI_SUM, E->parallel.vertical_comm);
    } else {
         /* no need to weight in rho(r) here. */
        moment_of_inertia = (8.0*M_PI/15.0)*
            (pow(E->sphere.ro,5.0) - pow(E->sphere.ri,5.0));
    }

    /* compute and add angular momentum components */
    
    exyz[1] = exyz[2] = exyz[3] = 0.0;
    fxyz[1] = fxyz[2] = fxyz[3] = 0.0;
    
    
      for (e=1;e<=E->lmesh.nel;e++) {
#ifdef ALLOW_ELLIPTICAL
	t = theta_g(E->eco[e].centre[1],E);
#else
	t = E->eco[e].centre[1];
#endif
	f = E->eco[e].centre[2];
	r = E->eco[e].centre[3];
	
	cos_t = cos(t);sin_t = sin(t);
	sin_f = sin(f);cos_f = cos(f);
	
	/* get Cartesian, element local velocities */
	velo_from_element_d(E,VV,e,sphere_key);
	for (j=1;j<=ppts;j++)   {
	  vx[j] = 0.0;vy[j] = 0.0;
	}
	for (j=1;j<=ppts;j++)   {
	  for (i=1;i<=ends;i++)   {
	    vx[j] += VV[1][i]*E->N.ppt[GNPINDEX(i,j)]; 
	    vy[j] += VV[2][i]*E->N.ppt[GNPINDEX(i,j)]; 
	  }
	}

        wx = -r*vy[1];
        wy =  r*vx[1];

        if(E->control.remove_angular_momentum) {
            int nz = (e-1) % E->lmesh.elz + 1;
            rho = 0.5 * (E->refstate.rho[nz] + E->refstate.rho[nz+1]);
        } else {
            rho = 1;
        }
	exyz[1] += (wx*cos_t*cos_f - wy*sin_f) * E->eco[e].area * rho;
	exyz[2] += (wx*cos_t*sin_f + wy*cos_f) * E->eco[e].area * rho;
	exyz[3] -= (wx*sin_t                 ) * E->eco[e].area * rho;
      }
    
    MPI_Allreduce(exyz,fxyz,4,MPI_DOUBLE,MPI_SUM,E->parallel.world);
    
    fxyz[1] = fxyz[1] / moment_of_inertia;
    fxyz[2] = fxyz[2] / moment_of_inertia;
    fxyz[3] = fxyz[3] / moment_of_inertia;
    
    rot = sqrt(fxyz[1]*fxyz[1] + fxyz[2]*fxyz[2] + fxyz[3]*fxyz[3]);
    fr = myatan(fxyz[2], fxyz[1]);
    tr = acos(fxyz[3] / rot);
    
    if (E->parallel.me==0) {
        if(E->control.remove_angular_momentum) {
            fprintf(E->fp,"Angular momentum: rot=%e tr=%e fr=%e\n",rot,tr*180/M_PI,fr*180/M_PI);
            fprintf(stderr,"Angular momentum: rot=%e tr=%e fr=%e\n",rot,tr*180/M_PI,fr*180/M_PI);
        } else {
            fprintf(E->fp,"Rigid rotation: rot=%e tr=%e fr=%e\n",rot,tr*180/M_PI,fr*180/M_PI);
            fprintf(stderr,"Rigid rotation: rot=%e tr=%e fr=%e\n",rot,tr*180/M_PI,fr*180/M_PI);
        }
    }
    /*
      remove rigid rotation 
    */
#ifdef ALLOW_ELLIPTICAL
      for (node=1;node<=nno;node++)   {
	/* cartesian velocity = omega \cross r  */
	vx[0] = fxyz[2]* E->x[CPPR][3][node] - fxyz[3]*E->x[CPPR][2][node];
	vx[1] = fxyz[3]* E->x[CPPR][1][node] - fxyz[1]*E->x[CPPR][3][node];
	vx[2] = fxyz[1]* E->x[CPPR][2][node] - fxyz[2]*E->x[CPPR][1][node];
	/* project into theta, phi */
	calc_cbase_at_node(node,cart_base,E);
	v_theta = vx[0]*cart_base[3] + vx[1]*cart_base[4] + vx[2]*cart_base[5] ;
	v_phi   = vx[0]*cart_base[6] + vx[1]*cart_base[7];
	E->sphere.cap[CPPR].V[1][node] -= v_theta;
	E->sphere.cap[CPPR].V[2][node] -= v_phi;
      }
#else
    sin_t = sin(tr) * rot;
    cos_t = cos(tr) * rot;
      for (node=1;node<=nno;node++)   {
	frd = fr - E->sx[CPPR][2][node];
	v_theta = E->sx[CPPR][3][node] * sin_t * sin(frd);
	v_phi =   E->sx[CPPR][3][node] * 
	  (  E->SinCos[lev][CPPR][0][node] * cos_t - E->SinCos[lev][CPPR][2][node]  * sin_t * cos(frd) );
	
	E->sphere.cap[CPPR].V[1][node] -= v_theta;
	E->sphere.cap[CPPR].V[2][node] -= v_phi;
      }
#endif
}
