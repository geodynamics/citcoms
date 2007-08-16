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
/*
  This file contains functions that are no longer used in this version of
  CitcomS. To reduce compilantion time and maintanance effort, these functions
  are removed from its original location to here.
*/



/* ==========================================================  */
/* from Parallel_related.c                                     */
/* =========================================================== */

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

  /*  MPI_Init(&argc,&argv); moved to main{} in Citcom.c, cpc 12/24/00 */
  MPI_Comm_rank(E->parallel.world, &(E->parallel.me) );
  MPI_Comm_size(E->parallel.world, &(E->parallel.nproc) );

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


void gather_TG_to_me0(E,TG)
  struct All_variables *E;
float *TG;
{

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
      MPI_Isend(TG,nsl,MPI_FLOAT,to_everyone,mst,E->parallel.world,&request[idb-1]);
    }
  }

  /* parallel_process_sync(E); */

  idb=0;
  for (i=1;i<=E->parallel.nprocxy;i++)  {
    from_proc = E->parallel.nprocz*(i-1) + E->parallel.me_loc[3];
    if (me!=from_proc)   {    /* me==0 receive all TG and add them up */
      mst = from_proc;
      idb++;
      MPI_Irecv(RG[idb],nsl,MPI_FLOAT,from_proc,mst,E->parallel.world,&request[idb-1]);
    }
  }

  MPI_Waitall(idb,request,status);

  for (i=1;i<E->parallel.nprocxy;i++)
    for (j=1;j<=E->sphere.nsf; j++)  {
      if (fabs(TG[j]) < e_16) TG[j] += RG[i][j];
    }

  /* parallel_process_sync(E); */

  return;
}


void sum_across_depth_sph(E,sphc,sphs,dest_proc)
  struct All_variables *E;
int dest_proc;
float *sphc,*sphs;
{

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

  parallel_process_sync(E);

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


void sum_across_surf_sph(E,TG,loc_proc)
  struct All_variables *E;
int loc_proc;
float *TG;
{

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
      MPI_Isend(TG,nsl,MPI_FLOAT,to_everyone,mst,E->parallel.world,&request[idb-1]);
    }
  }

  /* parallel_process_sync(E); */

  idb=0;
  for (i=1;i<=E->parallel.nprocxy;i++)  {
    from_proc = E->parallel.nprocz*(i-1) + loc_proc;
    if (me!=from_proc)   {    /* me==0 receive all TG and add them up */
      mst = from_proc;
      idb++;
      MPI_Irecv(RG[idb],nsl,MPI_FLOAT,from_proc,mst,E->parallel.world,&request[idb-1]);
    }
  }

  MPI_Waitall(idb,request,status);

  for (i=1;i<E->parallel.nprocxy;i++)
    for (j=0;j<nsl; j++)  {
      TG[j] += RG[i][j];
    }

  /* parallel_process_sync(E); */

  for (i=1;i<E->parallel.nprocxy;i++)
    free((void *) RG[i]);

  return;
}


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



/* ==========================================================  */
/* from Boundary_conditions.c                                  */
/* =========================================================== */


void renew_top_velocity_boundary(E)
  struct All_variables *E;
{
  int i,k,lev;
  int nox,noz,noy,nodel;
  float fxx10,fxx20,fyy1,fyy2,fxx0,fxx,fyy;
  float vxx1,vxx2,vxx,vvo,vvc;
  float fslope,vslope;
  static float fxx1,fxx2;

  FILE *fp;
  char output_file[255];
  nox=E->lmesh.nox;
  noz=E->lmesh.noz;
  noy=E->lmesh.noy;
  lev=E->mesh.levmax;

  fxx10=1.0;
  fyy1=0.76;
  fxx20=1.0;   /* (fxx1,fyy1), (fxx2,fyy2) the initial coordinates of the trench position */
  fyy2=0.81;

  vxx1=-2.*2.018e0;

  vvo=6.*2.018e0;
  vvc=-2.*2.018e0;     /* vvo--oceanic plate velocity; vvc--continental plate velocity      */

  if(E->advection.timesteps>1)  {
    fxx1=fxx1+E->advection.timestep*vxx1;
    fxx2=fxx2+E->advection.timestep*vxx1;
  }

  else  {
    fxx1=fxx10;
    fxx2=fxx20;
  }

  fprintf(stderr,"%f %f\n",fxx1,fxx2);

  if (E->parallel.me_loc[3] == E->parallel.nprocz-1 ) {
    for(k=1;k<=noy;k++)
      for(i=1;i<=nox;i++)   {
	nodel = (k-1)*nox*noz + (i-1)*noz+noz;
	fyy=E->SX[lev][1][1][nodel];
	if (fyy < fyy1 || fyy >fyy2 )   {
	  E->sphere.cap[1].VB[1][nodel]=0.0;
	  E->sphere.cap[1].VB[2][nodel]=-vvc;
	  E->sphere.cap[1].VB[3][nodel]=0.0;
	}    /* the region outside of the domain bounded by the trench length  */
	else if (fyy>=fyy1 && fyy <=fyy2)  {
	  if (E->SX[lev][1][2][nodel]>=0.00 && E->SX[lev][1][2][nodel]<= fxx1) {
	    E->sphere.cap[1].VB[1][nodel]=0.0;
	    E->sphere.cap[1].VB[2][nodel]=vvo;
	    E->sphere.cap[1].VB[3][nodel]=0.0;
	  }
	  else if ( E->SX[lev][1][2][nodel]>fxx1 && E->SX[lev][1][2][nodel]<fxx2) {
	    E->sphere.cap[1].VB[1][nodel]=0.0;
	    E->sphere.cap[1].VB[2][nodel]=vxx1;
	    E->sphere.cap[1].VB[3][nodel]=0.0;
	  }
	  else if ( E->SX[lev][1][2][nodel]>=fxx2) {
	    E->sphere.cap[1].VB[1][nodel]=0.0;
	    E->sphere.cap[1].VB[2][nodel]=vvc;
	    E->sphere.cap[1].VB[3][nodel]=0.0;
	  }
	}   /* end of else if (fyy>=fyy1 && fyy <=fyy2)  */

      }  /* end if for(i=1;i<nox;i++)  */
  }    /* end of E->parallel.me_loc[3]   */

  return;
}



/* ==========================================================  */
/* from Output.c                                               */
/* =========================================================== */

void output_stress(E,file_number,SXX,SYY,SZZ,SXY,SXZ,SZY)
  struct All_variables *E;
int file_number;
float *SXX,*SYY,*SZZ,*SXY,*SXZ,*SZY;
{
  int i,j,k,ii,m,fd,size2;
  int nox,noz,noy;
  char output_file[255];

  size2= (E->lmesh.nno+1)*sizeof(float);

  sprintf(output_file,"%s.%05d.SZZ",E->control.data_file,file_number);
  fd=open(output_file,O_RDWR | O_CREAT, 0644);
  write(fd,SZZ,size2);
  close (fd);

  return;
}


void print_field_spectral_regular(E,TG,sphc,sphs,proc_loc,filen)
  struct All_variables *E;
float *TG,*sphc,*sphs;
int proc_loc;
char * filen;
{
  FILE *fp,*fp1;
  char output_file[255];
  int i,node,j,ll,mm;
  float minx,maxx,t,f,rad;
  rad = 180.0/M_PI;

  maxx=-1.e26;
  minx=1.e26;
  if (E->parallel.me==proc_loc)  {

    sprintf(output_file,"%s.%s_intp",E->control.data_file,filen);
    fp=fopen(output_file,"w");
    if (fp == NULL) {
      fprintf(E->fp,"(Output.c #7) Cannot open %s\n",output_file);
      exit(8);
    }
    for (i=E->sphere.nox;i>=1;i--)
      for (j=1;j<=E->sphere.noy;j++)  {
        node = i + (j-1)*E->sphere.nox;
        t = 90-E->sphere.sx[1][node]*rad;
        f = E->sphere.sx[2][node]*rad;
        fprintf (fp,"%.3e %.3e %.4e\n",f,t,TG[node]);
        if(TG[node]>maxx)maxx=TG[node];
        if(TG[node]<minx)minx=TG[node];
      }
    fprintf(stderr,"lmaxx=%.4e lminx=%.4e for %s\n",maxx,minx,filen);
    fprintf(E->fp,"lmaxx=%.4e lminx=%.4e for %s\n",maxx,minx,filen);
    fclose(fp);

    sprintf(output_file,"%s.%s_sharm",E->control.data_file,filen);
    fp1=fopen(output_file,"w");
    if (fp1 == NULL) {
      fprintf(E->fp,"(Output.c #8) Cannot open %s\n",output_file);
      exit(8);
    }
    fprintf(fp1,"lmaxx=%.4e lminx=%.4e for %s\n",maxx,minx,filen);
    fprintf(fp1," ll   mm     cos      sin \n");
    for (ll=0;ll<=E->output.llmax;ll++)
      for(mm=0;mm<=ll;mm++)  {
        i = E->sphere.hindex[ll][mm];
        fprintf(fp1,"%3d %3d %.4e %.4e \n",ll,mm,sphc[i],sphs[i]);
      }

    fclose(fp1);
  }


  return;
}



void output_velo_related(E,file_number)
  struct All_variables *E;
  int file_number;
{
  int el,els,i,j,k,m,node,fd;
  int s,nox,noz,noy,size1,size2,size3;
  char output_file[255];
  FILE *fp1,*fp2;


  output_velo(E);
  output_visc(E);


  if (E->parallel.me_loc[3]==E->parallel.nprocz-1)      {
    sprintf(output_file,"%s.surf.%d.%d",E->control.data_file,E->parallel.me,cycles);
    fp2 = output_open(output_file);

    for(j=1;j<=E->sphere.caps_per_proc;j++)  {
      fprintf(fp2,"%3d %7d\n",j,E->lmesh.nsf);
      for(i=1;i<=E->lmesh.nsf;i++)   {
	s = i*E->lmesh.noz;
        fprintf(fp2,"%.4e %.4e %.4e %.4e\n",E->slice.tpg[j][i],E->slice.shflux[j][i],E->sphere.cap[j].V[1][s],E->sphere.cap[j].V[2][s]);
	}
      }
    fclose(fp2);

    }

  if (E->parallel.me_loc[3]==0)      {
    sprintf(output_file,"%s.botm.%d.%d",E->control.data_file,E->parallel.me,cycles);
    fp2 = output_open(output_file);

    for(j=1;j<=E->sphere.caps_per_proc;j++)  {
      fprintf(fp2,"%3d %7d\n",j,E->lmesh.nsf);
      for(i=1;i<=E->lmesh.nsf;i++)  {
	s = (i-1)*E->lmesh.noz + 1;
        fprintf(fp2,"%.4e %.4e %.4e %.4e\n",E->slice.tpgb[j][i],E->slice.bhflux[j][i],E->sphere.cap[j].V[1][s],E->sphere.cap[j].V[2][s]);
	}
      }
    fclose(fp2);
    }

  /* remove horizontal average output   by Tan2 Mar. 1 2002  */
/*    if (E->parallel.me<E->parallel.nprocz)  { */
/*      sprintf(output_file,"%s.ave_r.%d.%d",E->control.data_file,E->parallel.me,cycles); */
/*      fp2 = output_open(output_file); */
/*  	if (fp2 == NULL) { */
/*            fprintf(E->fp,"(Output.c #6) Cannot open %s\n",output_file); */
/*            exit(8); */
/*  	} */
/*      for(j=1;j<=E->lmesh.noz;j++)  { */
/*          fprintf(fp2,"%.4e %.4e %.4e %.4e\n",E->sx[1][3][j],E->Have.T[j],E->Have.V[1][j],E->Have.V[2][j]); */
/*  	} */
/*      fclose(fp2); */
/*      } */

  return;
  }



void output_temp(E,file_number)
  struct All_variables *E;
  int file_number;
{
  int m,nno,i,j,fd;
  char output_file[255];

  return;
}


void output_visc_prepare(struct All_variables *E, float **VE)
{
  void get_ele_visc();
  void visc_from_ele_to_gint();
  void visc_from_gint_to_nodes();

  float *EV, *VN[NCS];
  const int lev = E->mesh.levmax;
  const int nsd = E->mesh.nsd;
  const int vpts = vpoints[nsd];
  int i, m;


  // Here is a bug in the original code. EV is not allocated for each
  // E->sphere.caps_per_proc. Later, when elemental viscosity is written
  // to it (in get_ele_visc()), viscosity in high cap number will overwrite
  // that in a lower cap number.
  //
  // Since current CitcomS only support 1 cap per processor, this bug won't
  // manifest itself. So, I will leave it here.
  // by Tan2 5/22/2003
  int size2 = (E->lmesh.nel+1)*sizeof(float);
  EV = (float *) malloc (size2);

  for(m=1;m<=E->sphere.caps_per_proc;m++) {
    VN[m]=(float *)malloc((1+E->lmesh.nel*vpts)*sizeof(float));
  }

  get_ele_visc(E,EV,1);

  for(i=1;i<=E->lmesh.nel;i++)
    VE[1][i]=EV[i];

  visc_from_ele_to_gint(E, VE, VN, lev);
  visc_from_gint_to_nodes(E, VN, VE, lev);

  free((void *) EV);
  for(m=1;m<=E->sphere.caps_per_proc;m++) {
    free((void *) VN[m]);
  }

  return;
}


void output_visc(struct All_variables *E, int cycles)
{
  int i, j, m;
  char output_file[255];
  FILE *fp1;
  float *VE[NCS];

  sprintf(output_file,"%s.visc.%d.%d",E->control.data_file,E->parallel.me,cycles);
  fp1 = output_open(output_file);

  for(m=1;m<=E->sphere.caps_per_proc;m++) {
    VE[m]=(float *)malloc((1+E->lmesh.nno)*sizeof(float));
  }

  output_visc_prepare(E, VE);

  for(j=1;j<=E->sphere.caps_per_proc;j++) {
    fprintf(fp1,"%3d %7d\n",j,E->lmesh.nno);
    for(i=1;i<=E->lmesh.nno;i++)
      fprintf(fp1,"%.3e\n",VE[1][i]);
  }

  for(m=1;m<=E->sphere.caps_per_proc;m++) {
    free((void*) VE[m]);
  }

  fclose(fp1);

  return;
}

/* ==========================================================  */
/* from Process_buoyancy.c                                     */
/* =========================================================== */


void process_temp_field(E,ii)
 struct All_variables *E;
    int ii;
{
    void heat_flux();
    void output_temp();
    void process_output_field();
    int record_h;

/* This form prevented running for timesteps less than 10!!
    record_h = E->control.record_every/10;  */
    record_h = E->control.record_every;

/* changed to allow 0th time step to be outputted CPC 6/18/00 */
/*    if ( ((ii % record_h) == 0) || E->control.DIRECTII)    { */

    if ( (ii == 0) || ((ii % record_h) == 0) || E->control.DIRECTII)    {
      heat_flux(E);
      parallel_process_sync(E);
/*      output_temp(E,ii);  */
    }

/*    if ( ((ii % E->control.record_every) == 0) || E->control.DIRECTII)  { */
    if ( ((ii == 0) || ((ii % E->control.record_every) == 0))
		|| E->control.DIRECTII)     {
       process_output_field(E,ii);
    }

    return;
}


/* ==========================================================  */
/* from Process_velocity.c                                     */
/* =========================================================== */

void process_new_velocity(E,ii)
    struct All_variables *E;
    int ii;
{
    void output_velo_related();
    void get_STD_topo();
    void get_CBF_topo();

    int m,i,it;


    if ( (ii == 0) || ((ii % E->control.record_every) == 0)
		|| E->control.DIRECTII)     {
      get_STD_topo(E,E->slice.tpg,E->slice.tpgb,E->slice.divg,E->slice.vort,ii);
      parallel_process_sync(E);
      output_velo_related(E,ii);         /* also topo */
    }

    return;
}


void get_surface_velo(E, SV,m)
  struct All_variables *E;
  float *SV;
  int m;
  {

  int el,els,i,node,lev;
  char output_file[255];
  FILE *fp;

  const int dims=E->mesh.nsd;
  const int ends=enodes[dims];
  const int nno=E->lmesh.nno;

  lev = E->mesh.levmax;

  for(m=1;m<=E->sphere.caps_per_proc;m++)
    for (node=1;node<=nno;node++)
      if (node%E->lmesh.noz==0)   {
        i = node/E->lmesh.noz;
        SV[(i-1)*2+1] = E->sphere.cap[m].V[1][node];
        SV[(i-1)*2+2] = E->sphere.cap[m].V[2][node];
      }

  return;
  }



/* ==========================================================  */
/* from Global_operations.c                                    */
/* =========================================================== */

void sum_across_depth_sph1(E,sphc,sphs)
struct All_variables *E;
float *sphc,*sphs;
{
 int jumpp,total,j,d;
 float *sphcs,*temp;

 temp = (float *) malloc((E->sphere.hindice*2+3)*sizeof(float));
 sphcs = (float *) malloc((E->sphere.hindice*2+3)*sizeof(float));

 total = E->sphere.hindice*2+3;
 jumpp = E->sphere.hindice;
 for (j=0;j<E->sphere.hindice;j++)   {
   sphcs[j] = sphc[j];
   sphcs[j+jumpp] = sphs[j];
 }

 MPI_Allreduce(sphcs,temp,total,MPI_FLOAT,MPI_SUM,E->parallel.vertical_comm);

 for (j=0;j<E->sphere.hindice;j++)   {
   sphc[j] = temp[j];
   sphs[j] = temp[j+jumpp];
 }

 MPI_Allreduce(sphcs,temp,total,MPI_FLOAT,MPI_SUM,E->parallel.vertical_comm);

 free((void*) temp);
 free((void*) sphcs);

return;
}


/* ==========================================================  */
/* from                                                        */
/* =========================================================== */

