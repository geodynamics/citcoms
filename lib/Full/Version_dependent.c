#include <math.h>

#include "global_defs.h"
#include "parallel_related.h"

// Setup global mesh parameters
//
void global_derived_values(E)
  struct All_variables *E;
{
  int d,lx,lz,ly,i,nox,noz,noy;
  char logfile[100], timeoutput[100];
  FILE *fp, *fptime;

  E->mesh.levmax = E->mesh.levels-1;
  nox = E->mesh.mgunitx * (int) pow(2.0,((double)E->mesh.levmax))*E->parallel.nprocx + 1;
  noy = E->mesh.mgunity * (int) pow(2.0,((double)E->mesh.levmax))*E->parallel.nprocy + 1;
  noz = E->mesh.mgunitz * (int) pow(2.0,((double)E->mesh.levmax))*E->parallel.nprocz + 1;

  if (E->control.NMULTIGRID||E->control.EMULTIGRID)  {
    E->mesh.levmax = E->mesh.levels-1;
    E->mesh.gridmax = E->mesh.levmax;
    E->mesh.nox = E->mesh.mgunitx * (int) pow(2.0,((double)E->mesh.levmax))*E->parallel.nprocx + 1;
    E->mesh.noy = E->mesh.mgunity * (int) pow(2.0,((double)E->mesh.levmax))*E->parallel.nprocy + 1;
    E->mesh.noz = E->mesh.mgunitz * (int) pow(2.0,((double)E->mesh.levmax))*E->parallel.nprocz + 1;
  }
  else   {
    if (nox!=E->mesh.nox || noy!=E->mesh.noy || noz!=E->mesh.noz) {
      if (E->parallel.me==0)
	fprintf(stderr,"inconsistent mesh for interpolation, quit the run\n");
      parallel_process_termination();
    }
    E->mesh.gridmax = E->mesh.levmax;
    E->mesh.gridmin = E->mesh.levmax;
  }

  if(E->mesh.nsd != 3)
    E->mesh.noy = 1;

  E->mesh.nnx[1] = E->mesh.nox;
  E->mesh.nnx[2] = E->mesh.noy;
  E->mesh.nnx[3] = E->mesh.noz;
  E->mesh.elx = E->mesh.nox-1;
  E->mesh.ely = E->mesh.noy-1;
  E->mesh.elz = E->mesh.noz-1;

  E->mesh.nno = E->sphere.caps;
  for(d=1;d<=E->mesh.nsd;d++)
    E->mesh.nno *= E->mesh.nnx[d];

  E->mesh.nel = E->sphere.caps*E->mesh.elx*E->mesh.elz*E->mesh.ely;

  E->mesh.nnov = E->mesh.nno;

  E->mesh.neq = E->mesh.nnov*E->mesh.nsd;

  E->mesh.npno = E->mesh.nel;
  E->mesh.nsf = E->mesh.nox*E->mesh.noy;

  for(i=E->mesh.levmax;i>=E->mesh.levmin;i--) {
    if (E->control.NMULTIGRID||E->control.EMULTIGRID) {
      nox = E->mesh.mgunitx * (int) pow(2.0,(double)i)*E->parallel.nprocx + 1;
      noy = E->mesh.mgunity * (int) pow(2.0,(double)i)*E->parallel.nprocy + 1;
      noz = E->mesh.mgunitz * (int) pow(2.0,(double)i)*E->parallel.nprocz + 1;
    }
    else {
      noz = E->mesh.noz;
      nox = E->mesh.nox;
      noy = E->mesh.noy;
      /*if (i<E->mesh.levmax) noz=2;*/
    }

    E->mesh.ELX[i] = nox-1;
    E->mesh.ELY[i] = noy-1;
    E->mesh.ELZ[i] = noz-1;
    E->mesh.NNO[i] = E->sphere.caps * nox * noz * noy;
    E->mesh.NEL[i] = E->sphere.caps * (nox-1) * (noz-1) * (noy-1);
    E->mesh.NPNO[i] = E->mesh.NEL[i] ;
    E->mesh.NOX[i] = nox;
    E->mesh.NOZ[i] = noz;
    E->mesh.NOY[i] = noy;

    E->mesh.NNOV[i] = E->mesh.NNO[i];
    E->mesh.NEQ[i] = E->mesh.nsd * E->mesh.NNOV[i] ;
    /*      fprintf(stderr,"level=%d nox=%d noy=%d noz=%d %d %d %d %d %d %d %d %d %d %d %d\n",i,nox,noy,noz,E->mesh.ELX[i],E->mesh.ELY[i],E->mesh.ELZ[i],E->mesh.NNO[i],E->mesh.NEL[i],E->mesh.NPNO[i],E->mesh.NOX[i],E->mesh.NOZ[i],E->mesh.NOY[i],E->mesh.NNOV[i],E->mesh.NEQ[i]); */
    /*      MPI_Barrier(E->parallel.world); */
  }

  E->sphere.elx = E->sphere.nox-1;
  E->sphere.ely = E->sphere.noy-1;
  E->sphere.snel = E->sphere.ely*E->sphere.elx;
  E->sphere.nsf = E->sphere.noy*E->sphere.nox;

  // Myr
  E->data.scalet = (E->data.layer_km*1e3*E->data.layer_km*1e3/E->data.therm_diff)/(1.e6*365.25*24*3600);
  // cm/yr
  E->data.scalev = (E->data.layer_km*1e3/E->data.therm_diff)/(100*365.25*24*3600);
  E->data.timedir = E->control.Atemp / fabs(E->control.Atemp);

  if(E->control.print_convergence && E->parallel.me==0)
    fprintf(stderr,"Problem has %d x %d x %d nodes\n",E->mesh.nox,E->mesh.noz,E->mesh.noy);

  return;
}


/* =================================================
   Standard node positions including mesh refinement

   =================================================  */

void node_locations(E)
     struct All_variables *E;
{
  int m,i,j,k,ii,d,node,lev;
  double ro,ri,dr,*rr,*RR,fo;
  float t1,f1,tt1;
  int noz,lnoz,step,nn;
  char output_file[255], a[255];
  FILE *fp1;

  const int dims = E->mesh.nsd;

  void coord_of_cap();
  void rotate_mesh ();
  void compute_angle_surf_area ();

  rr = (double *)  malloc((E->mesh.noz+1)*sizeof(double));
  RR = (double *)  malloc((E->mesh.noz+1)*sizeof(double));

  if(E->control.coor==1)    {
      sprintf(output_file,"%s",E->control.coor_file);
      fp1=fopen(output_file,"r");
	if (fp1 == NULL) {
          fprintf(E->fp,"(Nodal_mesh.c #1) Cannot open %s\n",output_file);
          exit(8);
	}
      fscanf(fp1,"%s%d",&a,&i);
      if (i != E->mesh.noz ) {
          fprintf(E->fp,"(Nodal_mesh.c #2) inconsistent file length: %s\n",output_file);
          exit(8);
      }
      for (k=1;k<=E->mesh.noz;k++)  {
	fscanf(fp1,"%d %f",&nn,&tt1);
	rr[k]=tt1;
      }

      fclose(fp1);
  }
  else {
    /* generate uniform mesh in radial direction */
    dr = (E->sphere.ro-E->sphere.ri)/(E->mesh.noz-1);

    for (k=1;k<=E->mesh.noz;k++)  {
      rr[k] = E->sphere.ri + (k-1)*dr;
    }
  }

  for (i=1;i<=E->lmesh.noz;i++)  {
      k = E->lmesh.nzs+i-1;
      RR[i] = rr[k];
      }

  for (lev=E->mesh.levmin;lev<=E->mesh.levmax;lev++) {

    if (E->control.NMULTIGRID||E->control.EMULTIGRID)
        step = (int) pow(2.0,(double)(E->mesh.levmax-lev));
    else
        step = 1;

      for (i=1;i<=E->lmesh.NOZ[lev];i++)
         E->sphere.R[lev][i] = RR[(i-1)*step+1];

    }          /* lev   */

  free ((void *) rr);
  free ((void *) RR);

  ro = -0.5*(M_PI/4.0)/E->mesh.elx;
  fo = 0.0;

  E->sphere.dircos[1][1] = cos(ro)*cos(fo);
  E->sphere.dircos[1][2] = cos(ro)*sin(fo);
  E->sphere.dircos[1][3] = -sin(ro);
  E->sphere.dircos[2][1] = -sin(fo);
  E->sphere.dircos[2][2] = cos(fo);
  E->sphere.dircos[2][3] = 0.0;
  E->sphere.dircos[3][1] = sin(ro)*cos(fo);
  E->sphere.dircos[3][2] = sin(ro)*sin(fo);
  E->sphere.dircos[3][3] = cos(ro);

  for (j=1;j<=E->sphere.caps_per_proc;j++)   {
     ii = E->sphere.capid[j];
     coord_of_cap(E,j,ii);
     }

  /* rotate the mesh to avoid two poles on mesh points */
  for (j=1;j<=E->sphere.caps_per_proc;j++)   {
     ii = E->sphere.capid[j];
     rotate_mesh(E,j,ii);
     }

  compute_angle_surf_area (E);   /* used for interpolation */

  for (lev=E->mesh.levmin;lev<=E->mesh.levmax;lev++)
    for (j=1;j<=E->sphere.caps_per_proc;j++)
      for (i=1;i<=E->lmesh.NNO[lev];i++)  {
        E->SinCos[lev][j][0][i] = sin(E->SX[lev][j][1][i]);
        E->SinCos[lev][j][1][i] = sin(E->SX[lev][j][2][i]);
        E->SinCos[lev][j][2][i] = cos(E->SX[lev][j][1][i]);
        E->SinCos[lev][j][3][i] = cos(E->SX[lev][j][2][i]);
        }

  /*
if (E->control.verbose)
  for (lev=E->mesh.levmin;lev<=E->mesh.levmax;lev++)   {
    fprintf(E->fp_out,"output_coordinates after rotation %d \n",lev);
    for (j=1;j<=E->sphere.caps_per_proc;j++)
      for (i=1;i<=E->lmesh.NNO[lev];i++)
        if(i%E->lmesh.NOZ[lev]==1)
             fprintf(E->fp_out,"%d %d %g %g %g\n",j,i,E->SX[lev][j][1][i],E->SX[lev][j][2][i],E->SX[lev][j][3][i]);
      }
  */



  return;
}



void construct_tic_from_input(struct All_variables *E)
{
  int i, j, k, kk, m, p, node;
  int nox, noy, noz, gnoz;
  double r1, f1, t1;
  int mm, ll;
  double con;
  double modified_plgndr_a(int, int, double);

  noy=E->lmesh.noy;
  nox=E->lmesh.nox;
  noz=E->lmesh.noz;
  gnoz=E->mesh.noz;

  /* set up a linear temperature profile first */
  for(m=1;m<=E->sphere.caps_per_proc;m++)
    for(i=1;i<=noy;i++)
      for(j=1;j<=nox;j++)
	for(k=1;k<=noz;k++) {
	  node=k+(j-1)*noz+(i-1)*nox*noz;
	  r1=E->sx[m][3][node];
	  E->T[m][node] = E->control.TBCbotval - (E->control.TBCtopval + E->control.TBCbotval)*(r1 - E->sphere.ri)/(E->sphere.ro - E->sphere.ri);
	}

  /* This part put a temperature anomaly at depth where the global
     node number is equal to load_depth. The horizontal pattern of
     the anomaly is given by spherical harmonic ll & mm. */

  for (p=0; p<E->convection.number_of_perturbations; p++) {
    mm = E->convection.perturb_mm[p];
    ll = E->convection.perturb_ll[p];
    con = E->convection.perturb_mag[p];
    kk = E->convection.load_depth[p];

    if ( (kk < 1) || (kk >= gnoz) ) continue;

    k = kk - E->lmesh.nzs + 1;
    if ( (k < 1) || (k >= noz) ) continue; // if layer k is not inside this proc.
    if (E->parallel.me_loc[1] == 0 && E->parallel.me_loc[2] == 0
	&& E->sphere.capid[1] == 1 )
      fprintf(stderr,"Initial temperature perturbation:  layer=%d  mag=%g  l=%d  m=%d\n", kk, con, ll, mm);

    for(m=1;m<=E->sphere.caps_per_proc;m++)
      for(i=1;i<=noy;i++)
	for(j=1;j<=nox;j++) {
	  node=k+(j-1)*noz+(i-1)*nox*noz;
	  t1=E->sx[m][1][node];
	  f1=E->sx[m][2][node];

	  E->T[m][node] += con*modified_plgndr_a(ll,mm,t1)*cos(mm*f1);
	}
  }

  temperatures_conform_bcs(E);

  return;
}

