/* Functions to assemble the element k matrices and the element f vector.
   Note that for the regular grid case the calculation of k becomes repetitive
   to the point of redundancy. */

#include <math.h>
#include "element_definitions.h"
#include "global_defs.h"



void add_force(struct All_variables *E, int e, double *elt_f, int m)
{
  const int dims=E->mesh.nsd;
  const int ends=enodes[E->mesh.nsd];
  int a, a1, a2, a3, p, node;

  for(a=1;a<=ends;a++)          {
    node = E->ien[m][e].node[a];
    p=(a-1)*dims;
    a1=E->id[m][node].doff[1];
    E->F[m][a1] += elt_f[p];
    a2=E->id[m][node].doff[2];
    E->F[m][a2] += elt_f[p+1];
    a3=E->id[m][node].doff[3];
    E->F[m][a3] += elt_f[p+2];
  }
}



/* ================================================================
   Function to assemble the global  F vector.
                     +
   Function to get the global H vector (mixed method driving terms)
   ================================================================ */

void assemble_forces(E,penalty)
     struct All_variables *E;
     int penalty;
{
  double elt_f[24];
  /*   double elt_h[1]; */
  int m,a,e;

  void get_elt_f();
  void get_elt_h();
  void get_elt_tr();
  void strip_bcs_from_residual();

  void exchange_id_d();

  void thermal_buoyancy();

  const int neq=E->lmesh.neq;
  const int nel=E->lmesh.nel;
  const int lev=E->mesh.levmax;


 thermal_buoyancy(E,E->temp);


 for(m=1;m<=E->sphere.caps_per_proc;m++)    {

   for(a=0;a<neq;a++)
     E->F[m][a] = 0.0;

  /*for(a=0;a<npno;a++)
    E->H[m][a]=0.0;*/

   for (e=1;e<=nel;e++)  {

     get_elt_f(E,e,elt_f,1,m);
     add_force(E, e, elt_f, m);

     /*get_elt_h(E,e,elt_h,m);
       E->H[m][e] = elt_h[0];   due to single pressure node per element */


     /* for traction bc */
     get_elt_tr(E, e, elt_f, m);
     /* add_force(E, e, elt_f, m); */

     }       /* end for e */
   }       /* end for m */

   exchange_id_d(E, E->F, lev);

   strip_bcs_from_residual(E,E->F,lev);
   return;
  }



/*==============================================================
  Function to supply the element k matrix for a given element e.
  ==============================================================  */

void get_elt_k(E,el,elt_k,lev,m)
     struct All_variables *E;
     int el,m;
     double elt_k[24*24];
     int lev;
{
    double bdbmu[4][4];
    double bdbl[4][4];

    void print_elt_k();

    int pn,qn,ad,bd;

    int a,b,i,j,j1,p,q,nint,d,k,n,es;
    double rtf[4][9],W[9],ra[9],si[9],ct[9];
    void get_global_shape_fn();
    void construct_c3x3matrix_el();
    struct Shape_function GN;
    struct Shape_function_dA dOmega;
    struct Shape_function_dx GNx;
    double visc[9],temp;


    double ba[9][9][4][7]; /* integration points,node,3x6 matrix */

    const int nn=loc_mat_size[E->mesh.nsd];
    const int vpts=vpoints[E->mesh.nsd];
    const int ppts=ppoints[E->mesh.nsd];
    const int ends=enodes[E->mesh.nsd];
    const int dims=E->mesh.nsd,dofs=E->mesh.dof;
    const int sphere_key = 1;

 /*   es = (el-1)/E->lmesh.ELZ[lev] + 1; */

    get_global_shape_fn(E,el,&GN,&GNx,&dOmega,0,sphere_key,rtf,lev,m);

    if ((el-1)%E->lmesh.ELZ[lev]==0)
      construct_c3x3matrix_el(E,el,&E->element_Cc,&E->element_Ccx,lev,m,0);

    /* Note N[a].gauss_pt[n] is the value of shape fn a at the nth gaussian
       quadrature point. Nx[d] is the derivative wrt x[d]. */

    for(k=1;k<=vpts;k++) {
      W[k]=g_point[k].weight[dims-1]*dOmega.vpt[k]*E->EVI[lev][m][(el-1)*vpts+k];

     ra[k] = rtf[3][k];
     si[k] = sin(rtf[1][k]);
     ct[k] = cos(rtf[1][k])/si[k];
     }


  for(k=1;k<=VPOINTS3D;k++)
    for(a=1;a<=ends;a++)
      for(n=1;n<=dims;n++)   {
        ba[a][k][n][1]=
          (GNx.vpt[GNVXINDEX(0,a,k)]*E->element_Cc.vpt[BVINDEX(1,n,a,k)]
          + E->N.vpt[GNVINDEX(a,k)]*E->element_Ccx.vpt[BVXINDEX(1,n,1,a,k)]
          + E->N.vpt[GNVINDEX(a,k)]*E->element_Cc.vpt[BVINDEX(3,n,a,k)])*ra[k];

        ba[a][k][n][2]=
          (E->N.vpt[GNVINDEX(a,k)]*E->element_Cc.vpt[BVINDEX(1,n,a,k)]*ct[k]
          + E->N.vpt[GNVINDEX(a,k)]*E->element_Cc.vpt[BVINDEX(3,n,a,k)]
          +(GNx.vpt[GNVXINDEX(1,a,k)]*E->element_Cc.vpt[BVINDEX(2,n,a,k)]
          + E->N.vpt[GNVINDEX(a,k)]*E->element_Ccx.vpt[BVXINDEX(2,n,2,a,k)])
            /si[k])*ra[k];

        ba[a][k][n][3]=
          GNx.vpt[GNVXINDEX(2,a,k)]*E->element_Cc.vpt[BVINDEX(3,n,a,k)];

        ba[a][k][n][4]=
          (GNx.vpt[GNVXINDEX(0,a,k)]*E->element_Cc.vpt[BVINDEX(2,n,a,k)]
         + E->N.vpt[GNVINDEX(a,k)]*E->element_Ccx.vpt[BVXINDEX(2,n,1,a,k)]
         - E->N.vpt[GNVINDEX(a,k)]*E->element_Cc.vpt[BVINDEX(2,n,a,k)]*ct[k]
         +(GNx.vpt[GNVXINDEX(1,a,k)]*E->element_Cc.vpt[BVINDEX(1,n,a,k)]
         + E->N.vpt[GNVINDEX(a,k)]*E->element_Ccx.vpt[BVXINDEX(1,n,2,a,k)])
          /si[k])*ra[k];

        ba[a][k][n][5]=
          GNx.vpt[GNVXINDEX(2,a,k)]*E->element_Cc.vpt[BVINDEX(1,n,a,k)]
         +(GNx.vpt[GNVXINDEX(0,a,k)]*E->element_Cc.vpt[BVINDEX(3,n,a,k)]
         + E->N.vpt[GNVINDEX(a,k)]*(E->element_Ccx.vpt[BVXINDEX(3,n,1,a,k)]
         - E->element_Cc.vpt[BVINDEX(1,n,a,k)]))*ra[k];

        ba[a][k][n][6]=
          GNx.vpt[GNVXINDEX(2,a,k)]*E->element_Cc.vpt[BVINDEX(2,n,a,k)]
         -ra[k]*E->N.vpt[GNVINDEX(a,k)]*E->element_Cc.vpt[BVINDEX(2,n,a,k)]
         +(GNx.vpt[GNVXINDEX(1,a,k)]*E->element_Cc.vpt[BVINDEX(3,n,a,k)]
         + E->N.vpt[GNVINDEX(a,k)]*E->element_Ccx.vpt[BVXINDEX(3,n,2,a,k)])
         /si[k]*ra[k];
        }

  for(a=1;a<=ends;a++)
    for(b=a;b<=ends;b++)   {
      ad=dims*(a-1);
      bd=dims*(b-1);

      bdbmu[1][1]=bdbmu[1][2]=bdbmu[1][3]=
      bdbmu[2][1]=bdbmu[2][2]=bdbmu[2][3]=
      bdbmu[3][1]=bdbmu[3][2]=bdbmu[3][3]=0.0;

      for(i=1;i<=dims;i++)
        for(j=1;j<=dims;j++)
          for(j1=1;j1<=6;j1++)
            for(k=1;k<=VPOINTS3D;k++)
              bdbmu[i][j] +=
                W[k]*((j1>3)?1.0:2.0)*ba[a][k][i][j1]*ba[b][k][j][j1];

		/**/
      pn=ad*nn+bd;
      qn=bd*nn+ad;

      elt_k[pn       ] = bdbmu[1][1] ; /* above */
      elt_k[pn+1     ] = bdbmu[1][2] ;
      elt_k[pn+2     ] = bdbmu[1][3] ;
      elt_k[pn+nn    ] = bdbmu[2][1] ;
      elt_k[pn+nn+1  ] = bdbmu[2][2] ;
      elt_k[pn+nn+2  ] = bdbmu[2][3] ;
      elt_k[pn+2*nn  ] = bdbmu[3][1] ;
      elt_k[pn+2*nn+1] = bdbmu[3][2] ;
      elt_k[pn+2*nn+2] = bdbmu[3][3] ;

      elt_k[qn       ] = bdbmu[1][1] ; /* below diag */
      elt_k[qn+1     ] = bdbmu[2][1] ;
      elt_k[qn+2     ] = bdbmu[3][1] ;
      elt_k[qn+nn    ] = bdbmu[1][2] ;
      elt_k[qn+nn+1  ] = bdbmu[2][2] ;
      elt_k[qn+nn+2  ] = bdbmu[3][2] ;
      elt_k[qn+2*nn  ] = bdbmu[1][3] ;
      elt_k[qn+2*nn+1] = bdbmu[2][3] ;
      elt_k[qn+2*nn+2] = bdbmu[3][3] ;
		/**/

      } /*  Sum over all the a,b's to obtain full  elt_k matrix */

    return;
}


/* =============================================
   General calling function for del_squared:
   according to whether it should be element by
   element or node by node.
   ============================================= */

void assemble_del2_u(E,u,Au,level,strip_bcs)
     struct All_variables *E;
     double **u,**Au;
     int level;
     int strip_bcs;
{
  void e_assemble_del2_u();
  void n_assemble_del2_u();

  if(E->control.NMULTIGRID||E->control.NASSEMBLE)
    n_assemble_del2_u(E,u,Au,level,strip_bcs);
  else
    e_assemble_del2_u(E,u,Au,level,strip_bcs);

  return;
}

/* ======================================
   Assemble del_squared_u vector el by el
   ======================================   */

void e_assemble_del2_u(E,u,Au,level,strip_bcs)
  struct All_variables *E;
  double **u,**Au;
  int level;
  int strip_bcs;

{
  int  el,e,i,a,b,a1,a2,a3,ii,m,nodeb;
  double elt_k[24*24],U[24],AU[24],alpha=1.0,beta=0.0;

  int indx[24];
  char uplo='U';

  void strip_bcs_from_residual();

  void exchange_id_d();

  void get_elt_k();

  double U1[24];

  const int n=loc_mat_size[E->mesh.nsd];
  const int ends=enodes[E->mesh.nsd];
  const int dims=E->mesh.nsd,dofs=E->mesh.dof;
  const int nel=E->lmesh.NEL[level];
  const int neq=E->lmesh.NEQ[level];

  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    for(i=0;i<neq;i++)
      Au[m][i] = 0.0;

    for(e=1;e<=nel;e++)   {
      for(a=1;a<=ends;a++) {
	ii = E->IEN[level][m][e].node[a];
	a1 = E->ID[level][m][ii].doff[1];
	a2 = E->ID[level][m][ii].doff[2];
	a3 = E->ID[level][m][ii].doff[3];
	for(b=1;b<=ends;b++) {
	        nodeb = E->IEN[level][m][e].node[b];
	        ii = (a*n+b)*dims-(dims*n+dims);
			/* i=1, j=1,2 */
	          /* i=1, j=1,2,3 */
		Au[m][a1] +=
		        E->elt_k[level][m][e].k[ii] *
			u[m][E->ID[level][m][nodeb].doff[1]]
		      + E->elt_k[level][m][e].k[ii+1] *
			u[m][E->ID[level][m][nodeb].doff[2]]
		      + E->elt_k[level][m][e].k[ii+2] *
			u[m][E->ID[level][m][nodeb].doff[3]];
		/* i=2, j=1,2,3 */
		Au[m][a2] +=
		        E->elt_k[level][m][e].k[ii+n] *
			u[m][E->ID[level][m][nodeb].doff[1]]
		      + E->elt_k[level][m][e].k[ii+n+1] *
			u[m][E->ID[level][m][nodeb].doff[2]]
		      + E->elt_k[level][m][e].k[ii+n+2] *
			u[m][E->ID[level][m][nodeb].doff[3]];
		/* i=3, j=1,2,3 */
		Au[m][a3] +=
		        E->elt_k[level][m][e].k[ii+n+n] *
			u[m][E->ID[level][m][nodeb].doff[1]]
		      + E->elt_k[level][m][e].k[ii+n+n+1] *
			u[m][E->ID[level][m][nodeb].doff[2]]
		      + E->elt_k[level][m][e].k[ii+n+n+2] *
			u[m][E->ID[level][m][nodeb].doff[3]];

 	    }         /* end for loop b */
        }             /* end for loop a */

       }          /* end for e */
     }         /* end for m  */

    exchange_id_d(E, Au, level);

  if(strip_bcs)
     strip_bcs_from_residual(E,Au,level);

  return; }


/* ======================================================
   Assemble Au using stored, nodal coefficients.
   ====================================================== */

void n_assemble_del2_u(E,u,Au,level,strip_bcs)
     struct All_variables *E;
     double **u,**Au;
     int level;
     int strip_bcs;
{
    int m,node, e,i;
    int eqn1,eqn2,eqn3;

    double UU,U1,U2,U3;
    void strip_bcs_from_residual();

    void exchange_id_d();

    int *C;
    higher_precision *B1,*B2,*B3;

    const int neq=E->lmesh.NEQ[level];
    const int nno=E->lmesh.NNO[level];
    const int dims=E->mesh.nsd,dofs=E->mesh.dof;
    const int max_eqn = dims*14;


  for (m=1;m<=E->sphere.caps_per_proc;m++)  {

     for(e=0;e<=neq;e++)
	Au[m][e]=0.0;

     u[m][neq]=0.0;

     for(e=1;e<=nno;e++)     {

       eqn1=E->ID[level][m][e].doff[1];
       eqn2=E->ID[level][m][e].doff[2];
       eqn3=E->ID[level][m][e].doff[3];

       U1 = u[m][eqn1];
       U2 = u[m][eqn2];
       U3 = u[m][eqn3];

       C=E->Node_map[level][m] + (e-1)*max_eqn;
       B1=E->Eqn_k1[level][m]+(e-1)*max_eqn;
       B2=E->Eqn_k2[level][m]+(e-1)*max_eqn;
       B3=E->Eqn_k3[level][m]+(e-1)*max_eqn;

       for(i=3;i<max_eqn;i++)  {
         if (C[i] != neq+1) {
	  UU = u[m][C[i]];
  	  Au[m][eqn1] += B1[i]*UU;
  	  Au[m][eqn2] += B2[i]*UU;
  	  Au[m][eqn3] += B3[i]*UU;
          }
       }
       for(i=0;i<max_eqn;i++)
	 if (C[i] != neq+1)
          Au[m][C[i]] += B1[i]*U1+B2[i]*U2+B3[i]*U3;

       }     /* end for e */
     }     /* end for m */

     exchange_id_d(E, Au, level);

    if (strip_bcs)
	strip_bcs_from_residual(E,Au,level);

    return;
}


void build_diagonal_of_K(E,el,elt_k,level,m)
     struct All_variables *E;
     int level,el,m;
     double elt_k[24*24];

{
    int a,a1,a2,p,node;

    const int n=loc_mat_size[E->mesh.nsd];
    const int dims=E->mesh.nsd;
    const int ends=enodes[E->mesh.nsd];

    for(a=1;a<=ends;a++) {
	    node=E->IEN[level][m][el].node[a];
	    /* dirn 1 */
	    a1 = E->ID[level][m][node].doff[1];
	    p=(a-1)*dims;
	    E->BI[level][m][a1] += elt_k[p*n+p];

	    /* dirn 2 */
	    a2 = E->ID[level][m][node].doff[2];
	    p=(a-1)*dims+1;
	    E->BI[level][m][a2] += elt_k[p*n+p];

	    /* dirn 3 */
	    a1 = E->ID[level][m][node].doff[3];
	    p=(a-1)*dims+2;
	    E->BI[level][m][a1] += elt_k[p*n+p];
            }

  return;
}

void build_diagonal_of_Ahat(E)
    struct All_variables *E;
{
    double assemble_dAhatp_entry();

    double BU;
    int m,e,npno,neq,level;
    float time,time0,CPU_time();

 for (level=E->mesh.gridmin;level<=E->mesh.gridmax;level++)

   for (m=1;m<=E->sphere.caps_per_proc;m++)    {

     npno = E->lmesh.NPNO[level];
     neq=E->lmesh.NEQ[level];

     for(e=1;e<=npno;e++)
	E->BPI[level][m][e]=1.0;

     if(!E->control.precondition)
	return;

     for(e=1;e<=npno;e++)  {
	BU=assemble_dAhatp_entry(E,e,level,m);
	if(BU != 0.0)
	    E->BPI[level][m][e] = 1.0/BU;
	else
	    E->BPI[level][m][e] = 1.0;
        }
     }

    return;
}

/* ==========================================
   Assemble a div_u vector element by element
   ==========================================  */

void assemble_div_u(E,U,divU,level)
     struct All_variables *E;
     double **U,**divU;
     int level;
{
    int e,j1,j2,j3,p,a,b,m;
    higher_precision elt_g[24][1];
    void get_elt_g();


    const int nel=E->lmesh.NEL[level];
    const int ends=enodes[E->mesh.nsd];
    const int dims=E->mesh.nsd,dofs=E->mesh.dof;
    const int npno=E->lmesh.NPNO[level];

  for(m=1;m<=E->sphere.caps_per_proc;m++)
    for(e=1;e<=npno;e++)
	divU[m][e] = 0.0;

  for(m=1;m<=E->sphere.caps_per_proc;m++)
       for(a=1;a<=ends;a++)   {
	  p = (a-1)*dims;
          for(e=1;e<=nel;e++) {
	    b = E->IEN[level][m][e].node[a];
	    j1= E->ID[level][m][b].doff[1];
            j2= E->ID[level][m][b].doff[2];
	    j3= E->ID[level][m][b].doff[3];
	    divU[m][e] += E->elt_del[level][m][e].g[p  ][0] * U[m][j1]
	                + E->elt_del[level][m][e].g[p+1][0] * U[m][j2]
	                + E->elt_del[level][m][e].g[p+2][0] * U[m][j3];
	    }
	 }


    return;
}


/* ==========================================
   Assemble a grad_P vector element by element
   ==========================================  */

void assemble_grad_p(E,P,gradP,lev)
     struct All_variables *E;
     double **P,**gradP;
     int lev;

{
  int m,el,e,i,j1,j2,j3,p,a,b,nel,neq;
  void strip_bcs_from_residual();

  void exchange_id_d();

  higher_precision elt_g[24][1];
  void get_elt_g();

  const int ends=enodes[E->mesh.nsd];
  const int dims=E->mesh.nsd,dofs=E->mesh.dof;

  for(m=1;m<=E->sphere.caps_per_proc;m++)  {

    nel=E->lmesh.NEL[lev];
    neq=E->lmesh.NEQ[lev];

    for(i=0;i<=neq;i++)
      gradP[m][i] = 0.0;

    for(e=1;e<=nel;e++) {

	if(0.0==P[m][e])
	    continue;

	for(a=1;a<=ends;a++)       {
	     p = (a-1)*dims;
	     b = E->IEN[lev][m][e].node[a];
	     j1= E->ID[lev][m][b].doff[1];
	     j2= E->ID[lev][m][b].doff[2];
	     j3= E->ID[lev][m][b].doff[3];
		        /*for(b=0;b<ploc_mat_size[E->mesh.nsd];b++)  */
             gradP[m][j1] += E->elt_del[lev][m][e].g[p  ][0] * P[m][e];
             gradP[m][j2] += E->elt_del[lev][m][e].g[p+1][0] * P[m][e];
             gradP[m][j3] += E->elt_del[lev][m][e].g[p+2][0] * P[m][e];
	     }
        }       /* end for el */
     }       /* end for m */

  exchange_id_d(E, gradP,  lev); /*  correct gradP   */


  strip_bcs_from_residual(E,gradP,lev);

return;
}

double assemble_dAhatp_entry(E,e,level,m)
     struct All_variables *E;
     int e,level,m;

{
    int i,j,p,a,b,node,ee,element,lnode,npno;
    void strip_bcs_from_residual();
    higher_precision elt_g[24][1];
    void get_elt_g();

    double gradP[81],divU;

    const int ends=enodes[E->mesh.nsd];
    const int dims=E->mesh.nsd,dofs=E->mesh.dof;

    npno=E->lmesh.NPNO[level];

    for(i=0;i<81;i++)
	gradP[i] = 0.0;

    divU=0.0;

    for(a=1;a<=ends;a++) {
      p = (a-1)*dims;
      node = E->IEN[level][m][e].node[a];
      j=E->ID[level][m][node].doff[1];
      gradP[p] += E->BI[level][m][j]*E->elt_del[level][m][e].g[p][0];

      j=E->ID[level][m][node].doff[2];
      gradP[p+1] += E->BI[level][m][j]*E->elt_del[level][m][e].g[p+1][0];

      j=E->ID[level][m][node].doff[3];
      gradP[p+2] += E->BI[level][m][j]*E->elt_del[level][m][e].g[p+2][0];
      }


    /* calculate div U from the same thing .... */

    /* only need to run over nodes with non-zero grad P, i.e. the ones in
       the element accessed above, BUT it is only necessary to update the
       value in the original element, because the diagonal is all we use at
       the end ... */

    for(b=1;b<=ends;b++) {
      p = (b-1)*dims;
      divU +=E->elt_del[level][m][e].g[p][0] * gradP[p];
      divU +=E->elt_del[level][m][e].g[p+1][0] * gradP[p+1];
      divU +=E->elt_del[level][m][e].g[p+2][0] * gradP[p+2];
      }

return(divU);  }


/*==============================================================
  Function to supply the element g matrix for a given element e.
  ==============================================================  */

void get_elt_g(E,el,elt_del,lev,m)
     struct All_variables *E;
     int el,m;
     higher_precision elt_del[24][1];
     int lev;

{  void get_global_shape_fn();
   void construct_c3x3matrix_el();
   int p,a,nint,es,d,i,j,k;
   double ra,ct,si,x[4],rtf[4][9];
   double temp;
   int lmsize;

   struct Shape_function GN;
   struct Shape_function_dA dOmega;
   struct Shape_function_dx GNx;

   const int dims=E->mesh.nsd,dofs=E->mesh.dof;
   const int ends=enodes[dims];
   const int vpts=vpoints[dims];
   const int sphere_key = 1;

   /* Special case, 4/8 node bilinear cartesian square/cube element -> 1 pressure point */

/*   es = (el-1)/E->lmesh.ELZ[lev]+1;  */

   if ((el-1)%E->lmesh.ELZ[lev]==0)
      construct_c3x3matrix_el(E,el,&E->element_Cc,&E->element_Ccx,lev,m,1);

   get_global_shape_fn(E,el,&GN,&GNx,&dOmega,2,sphere_key,rtf,lev,m);


   temp=p_point[1].weight[dims-1] * dOmega.ppt[1];

   ra = rtf[3][1];
   si = 1.0/sin(rtf[1][1]);
   ct = cos(rtf[1][1])*si;

   for(a=1;a<=ends;a++)      {
     for (i=1;i<=dims;i++)
       x[i]=GNx.ppt[GNPXINDEX(2,a,1)]*E->element_Cc.ppt[BPINDEX(3,i,a,1)]
        + 2.0*ra*E->N.ppt[GNPINDEX(a,1)]*E->element_Cc.ppt[BPINDEX(3,i,a,1)]
        + ra*(GNx.ppt[GNPXINDEX(0,a,1)]*E->element_Cc.ppt[BPINDEX(1,i,a,1)]
        +E->N.ppt[GNPINDEX(a,1)]*E->element_Ccx.ppt[BPXINDEX(1,i,1,a,1)]
        +ct*E->N.ppt[GNPINDEX(a,1)]*E->element_Cc.ppt[BPINDEX(1,i,a,1)]
        +si*(GNx.ppt[GNPXINDEX(1,a,1)]*E->element_Cc.ppt[BPINDEX(2,i,a,1)]
        +E->N.ppt[GNPINDEX(a,1)]*E->element_Ccx.ppt[BPXINDEX(2,i,2,a,1)]));

     p=dims*(a-1);
     elt_del[p  ][0] = -x[1] * temp;
     elt_del[p+1][0] = -x[2] * temp;
     elt_del[p+2][0] = -x[3] * temp;

      /* fprintf (E->fp,"B= %d %d %g %g %g %g %g\n",el,a,dOmega.ppt[1],GNx.ppt[GNPXINDEX(0,a,1)],GNx.ppt[GNPXINDEX(1,a,1)],elt_del[p][0],elt_del[p+1][0]);
      */
     }

   return;
 }

/* ===============================================================
   Function to create the element pressure-forcing vector (due
   to imposed velocity boundary conditions, mixed method).
   =============================================================== */

void get_elt_h(E,el,elt_h,m)
     struct All_variables *E;
     int el,m;
     double elt_h[1];
{
    int aid,i,p,a,b,d,j,k,q,global,got_g;
    unsigned int type;
    double elt_g[24][1];
    void get_elt_g();

    for(p=0;p<1;p++) elt_h[p] = 0.0;

    got_g = 0;

  type=VBX;
  for(i=1;i<=E->mesh.nsd;i++)
    { for(a=1;a<=enodes[E->mesh.nsd];a++)
	{ if (E->node[m][E->ien[m][el].node[a]] & type)
	    { if(!got_g)
		{  get_elt_g(E,el,elt_g,E->mesh.levmax,m);
		   got_g++;
		 }

	      p=E->mesh.nsd*(a-1) + i - 1;
	      for(b=1;b<=pnodes[E->mesh.nsd];b++)
		{ q = b-1;
		  elt_h[q] -= elt_g[p][q] * E->sphere.cap[m].VB[i][E->ien[m][el].node[a]];
		}
	    }
	}
      type *= (unsigned int) 2;
    }
   return;
}

/*=================================================================
  Function to create the element force vector (allowing for velocity b.c.'s)
  ================================================================= */

void get_elt_f(E,el,elt_f,bcs,m)
     struct All_variables *E;
     int el,m;
     double elt_f[24];
     int bcs;

{

  int i,p,a,b,j,k,q,es;
  int got_elt_k,nodea,nodeb;
  unsigned int type;
  const unsigned int vbc_flag[] = {0, VBX, VBY, VBZ};

  double force[9],force_at_gs[9],elt_k[24*24];
  double rtf[4][9];

  void get_global_shape_fn();
  void construct_c3x3matrix_el();

  struct Shape_function GN;
  struct Shape_function_dA dOmega;
  struct Shape_function_dx GNx;

  const int dims=E->mesh.nsd;
  const int n=loc_mat_size[dims];
  const int ends=enodes[dims];
  const int vpts=vpoints[dims];
  const int sphere_key=1;

  es = (el-1)/E->lmesh.elz + 1;

  get_global_shape_fn(E,el,&GN,&GNx,&dOmega,0,sphere_key,rtf,E->mesh.levmax,m);

  if ((el-1)%E->lmesh.elz==0)
      construct_c3x3matrix_el(E,el,&E->element_Cc,&E->element_Ccx,E->mesh.levmax,m,0);

  for(p=0;p<n;p++) elt_f[p] = 0.0;

  for(p=1;p<=ends;p++)
    force[p] = E->temp[m][E->ien[m][el].node[p]];

  for(j=1;j<=vpts;j++)       {   /*compute force at each int point */
    force_at_gs[j] = 0.0;
    for(k=1;k<=ends;k++)
      force_at_gs[j] += force[k] * E->N.vpt[GNVINDEX(k,j)] ;
    }

  for(i=1;i<=dims;i++)  {
    for(a=1;a<=ends;a++)  {
      nodea=E->ien[m][el].node[a];
      p= dims*(a-1)+i-1;

      for(j=1;j<=vpts;j++)     /*compute sum(Na(j)*F(j)*det(j)) */
        elt_f[p] += force_at_gs[j] * E->N.vpt[GNVINDEX(a,j)]
           *dOmega.vpt[j]*g_point[j].weight[dims-1]
           *E->element_Cc.vpt[BVINDEX(3,i,a,j)];

	  /* imposed velocity terms */

      if(bcs)  {
        got_elt_k = 0;
        for(j=1;j<=dims;j++) {
	  type=vbc_flag[j];
          for(b=1;b<=ends;b++) {
            nodeb=E->ien[m][el].node[b];
            if ((E->node[m][nodeb]&type)&&(E->sphere.cap[m].VB[j][nodeb]!=0.0)){
              if(!got_elt_k) {
                get_elt_k(E,el,elt_k,E->mesh.levmax,m);
                got_elt_k = 1;
                }
              q = dims*(b-1)+j-1;
              if(p!=q) {
                elt_f[p] -= elt_k[p*n+q] * E->sphere.cap[m].VB[j][nodeb];
                }
              }
            }  /* end for b */
          }      /* end for j */
        }      /* end if for if bcs */

      }
    } /*  Complete the loops for a,i  	*/



  return;
}


/*=================================================================
  Function to create the element force vector due to stress b.c.
  ================================================================= */

void get_elt_tr(struct All_variables *E, int e, double *elt_tr, int m)
{
#if 0
  struct Shape_function GN;
  struct Shape_function_dA dOmega;
  struct Shape_function_dx GNx;

  const int dims=E->mesh.nsd,dofs=E->mesh.dof;
  const int n=loc_mat_size[dims];
  const int ends=enodes[dims];
  const int vpts=vpoints[dims];
  const int sphere_key=1;

  type = SBX;
  for(j=1;j<=dims;j++) {
    for(b=1;b<=ends;b++) {
      nodeb=E->ien[m][el].node[b];
      if ((E->node[m][nodeb]&type)&&(E->sphere.cap[m].VB[j][nodeb]!=0.0)){
	if(!got_elt_k) {
	  get_elt_k(E,el,elt_k,E->mesh.levmax,m);
	  got_elt_k = 1;
	}
	q = dims*(b-1)+j-1;
	if(p!=q) {
	  elt_f[p] -= elt_k[p*n+q] * E->sphere.cap[m].VB[j][nodeb];
	}
      }
    }  /* end for b */
    type *= (unsigned int) 2;
  }      /* end for j */

  /* this part is copied and modified from
     element_residual() in Advection_diffusion.c  */

  struct Shape_function1 GM;
  struct Shape_function1_dA dGamma;
  struct Shape_function_side_dA dGamma;
  struct Shape_function1_dx GMx;

  type = SBX;

  for(a=1;a<=ends;a++)
    if (BCFlag[E->ien[m][el].node[a]] & type) {
      if (dim=3) get_global_1d_shape_fn(E,el,&GM,&dGamma,1,m);
      else get_global_side_1d_shape_fn(E, el, &GM, &

      nodes[1] = loc[loc[a].node_nebrs[0][0]].node_nebrs[2][0];
      nodes[2] = loc[loc[a].node_nebrs[0][1]].node_nebrs[2][0];
      nodes[4] = loc[loc[a].node_nebrs[0][0]].node_nebrs[2][1];
      nodes[3] = loc[loc[a].node_nebrs[0][1]].node_nebrs[2][1];

      for(aid=0,j=1;j<=onedvpoints[E->mesh.nsd];j++)
	if (a==nodes[j])
	  aid = j;
      if(aid==0)
	printf("%d: mixed up in pg-flux int: looking for %d\n",el,a);

      if (loc[a].plus[1] != 0)
	back_front = 0;
      else back_front = 1;

      for(j=1;j<=onedvpoints[dims];j++)
	for(k=1;k<=onedvpoints[dims];k++)
	  Eres[a] += dGamma.vpt[GMVGAMMA(back_front,j)] *
	    E->M.vpt[GMVINDEX(aid,j)] * g_1d[j].weight[dims-1] *
	    BC[2][E->ien[m][el].node[a]] * E->M.vpt[GMVINDEX(k,j)];
    }

#endif
}


/* =================================================================
 subroutine to get augmented lagrange part of stiffness matrix
================================================================== */

void get_aug_k(E,el,elt_k,level,m)
     struct All_variables *E;
     int el,m;
     double elt_k[24*24];
     int level;
{
     int i,j,k,p[9],a,b,nodea,nodeb;
     double Visc;

     const int n=loc_mat_size[E->mesh.nsd];
     const int ends=enodes[E->mesh.nsd];
     const int vpts=vpoints[E->mesh.nsd];
     const int dims=E->mesh.nsd;

     Visc = 0.0;
     for(a=1;a<=vpts;a++) {
	  p[a] = (a-1)*dims;
	  Visc += E->EVI[level][m][(el-1)*vpts+a];
       }
     Visc = Visc/vpts;

     for(a=1;a<=ends;a++) {
        nodea=E->IEN[level][m][el].node[a];
        for(b=1;b<=ends;b++) {
           nodeb=E->IEN[level][m][el].node[b];      /* for Kab dims*dims  */
	   i = (a-1)*n*dims+(b-1)*dims;
	   elt_k[i  ] += Visc*E->control.augmented*
	              E->elt_del[level][m][el].g[p[a]][0]*
		      E->elt_del[level][m][el].g[p[b]][0];   /*for 11 */
	   elt_k[i+1] += Visc*E->control.augmented*
	              E->elt_del[level][m][el].g[p[a]][0]*
		      E->elt_del[level][m][el].g[p[b]+1][0];  /* for 12 */
	   elt_k[i+n] += Visc*E->control.augmented*
	              E->elt_del[level][m][el].g[p[a]+1][0]*
		      E->elt_del[level][m][el].g[p[b]][0];    /* for 21 */
	   elt_k[i+n+1] += Visc*E->control.augmented*
	              E->elt_del[level][m][el].g[p[a]+1][0]*
		      E->elt_del[level][m][el].g[p[b]+1][0];  /* for 22 */

           if(3==dims) {
	       elt_k[i+2] += Visc*E->control.augmented*
	              E->elt_del[level][m][el].g[p[a]][0]*
		      E->elt_del[level][m][el].g[p[b]+2][0];  /* for 13 */
	       elt_k[i+n+2] += Visc*E->control.augmented*
	              E->elt_del[level][m][el].g[p[a]+1][0]*
		      E->elt_del[level][m][el].g[p[b]+2][0];  /* for 23 */
	       elt_k[i+n+n] += Visc*E->control.augmented*
	              E->elt_del[level][m][el].g[p[a]+2][0]*
		      E->elt_del[level][m][el].g[p[b]][0];    /* for 31 */
	       elt_k[i+n+n+1] += Visc*E->control.augmented*
	              E->elt_del[level][m][el].g[p[a]+2][0]*
		      E->elt_del[level][m][el].g[p[b]+1][0];  /* for 32 */
	       elt_k[i+n+n+2] += Visc*E->control.augmented*
	              E->elt_del[level][m][el].g[p[a]+2][0]*
		      E->elt_del[level][m][el].g[p[b]+2][0];  /* for 33 */
               }
           }
       }

   return;
   }



