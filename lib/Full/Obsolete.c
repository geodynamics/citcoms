
/*************************************************************************/
/* from Process_buoyancy.c                                               */
/*************************************************************************/


void process_temp_field(E,ii)
 struct All_variables *E;
    int ii;
{
    void heat_flux();
    void output_temp();
    void parallel_process_sync();
    void process_output_field();
    int record_h;

    record_h = E->control.record_every;

    if ( (ii == 0) || ((ii % record_h) == 0) || E->control.DIRECTII)    {
      heat_flux(E);
      parallel_process_sync();
/*      output_temp(E,ii);  */
    }

    if ( ((ii == 0) || ((ii % E->control.record_every) == 0))
	 || E->control.DIRECTII)     {
       process_output_field(E,ii);
    }

    return;
}



/*************************************************************************/
/* from Global_operations.c                                              */
/*************************************************************************/

void sum_across_depth_sph1(E,sphc,sphs)
struct All_variables *E;
float *sphc,*sphs;
{
 int jumpp,total,j,d;

 static float *sphcs,*temp;
 static int been_here=0;
 static int *processors,nproc;

 static MPI_Comm world, horizon_p;
 static MPI_Group world_g, horizon_g;

if (been_here==0)  {
 processors = (int *)malloc((E->parallel.nprocz+2)*sizeof(int));
 temp = (float *) malloc((E->sphere.hindice*2+3)*sizeof(float));
 sphcs = (float *) malloc((E->sphere.hindice*2+3)*sizeof(float));

 nproc = 0;
 for (j=0;j<E->parallel.nprocz;j++) {
   d =E->parallel.me_sph*E->parallel.nprocz+E->parallel.nprocz-1-j; 
   processors[nproc] =  d;
   nproc ++;
   }

 if (nproc>0)  {
    world = E->parallel.world;
    MPI_Comm_group(world, &world_g);
    MPI_Group_incl(world_g, nproc, processors, &horizon_g);
    MPI_Comm_create(world, horizon_g, &horizon_p);
    }

 been_here++;
 }

 total = E->sphere.hindice*2+3;
  jumpp = E->sphere.hindice;
  for (j=0;j<E->sphere.hindice;j++)   {
      sphcs[j] = sphc[j];
      sphcs[j+jumpp] = sphs[j];
     }


 if (nproc>0)  {

    MPI_Allreduce(sphcs,temp,total,MPI_FLOAT,MPI_SUM,horizon_p);

    for (j=0;j<E->sphere.hindice;j++)   {
      sphc[j] = temp[j];
      sphs[j] = temp[j+jumpp];
     }

    }

return;
}


/*************************************************************************/
/* from Output.h                                                         */
/*************************************************************************/

void output_velo_related(E,file_number)
  struct All_variables *E;
  int file_number;
{
  int el,els,i,j,k,ii,m,node,fd;
  int s,nox,noz,noy,size1,size2,size3;

  char output_file[255];
  FILE *fp1,*fp2,*fp3,*fp4,*fp5,*fp6,*fp7,*fp8;
/*   static float *SV,*EV; */
/*   float *VE[NCS],*VIN[NCS],*VN[NCS]; */
  static int been_here=0;
  int lev = E->mesh.levmax;

  void parallel_process_sync();
  void get_surface_velo ();
  void get_ele_visc ();
  void visc_from_ele_to_gint();
  void visc_from_gint_to_nodes();
  const int nno = E->lmesh.nno;
  const int nsd = E->mesh.nsd;
  const int vpts = vpoints[nsd];


  if (been_here==0)  {
/*       ii = E->lmesh.nsf; */
/*       m = (E->parallel.me_loc[3]==0)?ii:0; */
/*       SV = (float *) malloc ((2*m+2)*sizeof(float)); */

      /* size2 = (E->lmesh.nel+1)*sizeof(float); */
      /* use the line from the original CitcomS   */

  sprintf(output_file,"%s.coord.%d",E->control.data_file,E->parallel.me);
  fp1=fopen(output_file,"w");
  if (fp1 == NULL) {
     fprintf(E->fp,"(Output.c #1) Cannot open %s\n",output_file);
     exit(8);
  }
  for(j=1;j<=E->sphere.caps_per_proc;j++)     {
    fprintf(fp1,"%3d %7d\n",j,E->lmesh.nno);
    for(i=1;i<=E->lmesh.nno;i++)
      fprintf(fp1,"%.3e %.3e %.3e\n",E->sx[j][1][i],E->sx[j][2][i],E->sx[j][3][i]);
    }
  fclose(fp1);

   been_here++;
    }


  sprintf(output_file,"%s.visc.%d.%d",E->control.data_file,E->parallel.me,file_number);
  fp1=fopen(output_file,"w");
  for(j=1;j<=E->sphere.caps_per_proc;j++)     {
    fprintf(fp1,"%3d %7d\n",j,E->lmesh.nno);
    for(i=1;i<=E->lmesh.nno;i++)
      fprintf(fp1,"%.3e\n",E->VI[lev][j][i]);

    }
  fclose(fp1);

  sprintf(output_file,"%s.velo.%d.%d",E->control.data_file,E->parallel.me,file_number);
  fp1=fopen(output_file,"w");
  fprintf(fp1,"%d %d %.5e\n",file_number,E->lmesh.nno,E->monitor.elapsed_time);
  for(j=1;j<=E->sphere.caps_per_proc;j++)     {
    fprintf(fp1,"%3d %7d\n",j,E->lmesh.nno);
     for(i=1;i<=E->lmesh.nno;i++)
       fprintf(fp1,"%.6e %.6e %.6e %.6e\n",E->sphere.cap[j].V[1][i],E->sphere.cap[j].V[2][i],E->sphere.cap[j].V[3][i],E->T[j][i]);
     /* for(i=1;i<=E->lmesh.nno;i++)
	fprintf(fp1,"%.6e\n",E->T[j][i]); */
    }

  fclose(fp1);

  if (E->parallel.me_loc[3]==E->parallel.nprocz-1)      {
    sprintf(output_file,"%s.surf.%d.%d",E->control.data_file,E->parallel.me,file_number);
    fp2=fopen(output_file,"w");
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
    sprintf(output_file,"%s.botm.%d.%d",E->control.data_file,E->parallel.me,file_number);
    fp2=fopen(output_file,"w");
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

/*   if (E->parallel.me<E->parallel.nprocz)  { */
/*     sprintf(output_file,"%s.ave_r.%d.%d",E->control.data_file,E->parallel.me,file_number); */
/*     fp2=fopen(output_file,"w"); */
/*     for(j=1;j<=E->lmesh.noz;j++)  { */
/*         fprintf(fp2,"%.4e %.4e %.4e %.4e\n",E->sx[1][3][j],E->Have.T[j],E->Have.V[1][j],E->Have.V[2][j]); */
/* 	} */
/*     fclose(fp2); */
/*     } */

  return;
  }



void output_temp(E,file_number)
  struct All_variables *E;
  int file_number;
{
  int m,nno,i,j,fd;
  static int *temp1;
  static int been_here=0;
  static int size2,size1;
  char output_file[255];
  void parallel_process_sync();

  return;
}



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
     fprintf(fp1,"lmaxx=%.4e lminx=%.4e for %s\n",maxx,minx,filen);
     fprintf(fp1," ll   mm     cos      sin \n");
     for (ll=0;ll<=E->sphere.output_llmax;ll++)
     for(mm=0;mm<=ll;mm++)  {
        i = E->sphere.hindex[ll][mm];
        fprintf(fp1,"%3d %3d %.4e %.4e \n",ll,mm,sphc[i],sphs[i]);
        }

     fclose(fp1);
     }


  return;
  }


/*************************************************************************/
/* from                                                                  */
/*************************************************************************/



/*************************************************************************/
/* from                                                                  */
/*************************************************************************/




/*************************************************************************/
/* from                                                                  */
/*************************************************************************/

