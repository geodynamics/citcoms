#include <stdio.h>
#include <stdlib.h>
#include <math.h>


/**/
struct sph_harm {
    int len;
    int* ll;
    int* mm;
    float* clm;
    float* slm;
};

typedef struct sph_harm sph_harm;


/**/
struct mesh {
    int ntheta;
    int nphi;
    double *theta;
    double *phi;
};

typedef struct mesh mesh;


/**/
struct field {
    mesh *grid;
    float **data;
};

typedef struct field field;


void allocate_field(field *);
void project_sph_harm_to_mesh(sph_harm *, field *);
void allocate_sph_harm(sph_harm *, int );
void get_sph_harm_coeff(char *, sph_harm* );
void get_mesh(mesh *, int , int );
double modified_plgndr_a(int , int , double );
void project_sph_harm_to_mesh(sph_harm *, field *);
void write_projected_field(char *, field *);


void print_help()
{
    const char msg[] = ""
        "Project the spherical harmonic coefficients to a regular mesh\n"
        "\n"
        "Usage: project_geoid infile outfile n_longitude n_latitude\n"
        "\n"
        "infile: name of the CitcomS geoid file\n"
        "outfile: name of the output file.\n"
	"         This file will contain 3 columns (longitude, latitude, geoid)\n"
        "n_longitude: # of grid points in longitude direction for the mesh\n"
        "n_latitude: # of grid points in latitudee direction for the mesh\n";

    fputs(msg, stderr);

    return;
}


void allocate_sph_harm(sph_harm *coeff, int len)
{
    coeff->ll = (int*)malloc(len * sizeof(int));
    coeff->mm = (int*)malloc(len * sizeof(int));
    coeff->clm = (float*)malloc(len * sizeof(float));
    coeff->slm = (float*)malloc(len * sizeof(float));
}


void get_sph_harm_coeff(char *filename, sph_harm* coeff)
{
    FILE *fp;
    char buffer[256];
    int junk0;
    float fjunk0, fjunk1, fjunk2, fjunk3;
    int ll_max, index;


    /* open CitcomS geoid file */
    fp = fopen(filename, "r");
    if(fp == NULL) {
        snprintf(buffer, 255, "Error: cannot open file: %s\n", filename);
        fputs(buffer, stderr);
        exit(-1);
    }


    /* read in the max sph harm degree */
    fgets(buffer, 255, fp);
    sscanf(buffer, "%d %d %f", &junk0, &ll_max, &fjunk0);
    /* fprintf(stderr, "%d %d %f\n", junk0, ll_max, fjunk0); */


    /* allocate memory */
    coeff->len = (ll_max + 1) * (ll_max + 2) / 2;
    allocate_sph_harm(coeff, coeff->len);


    /* read in the geoid coefficients */
    for(index=0; index<coeff->len; index++) {
        fgets(buffer, 255, fp);
        if(sscanf(buffer, "%d %d %f %f %f %f %f %f",
		  &(coeff->ll[index]), &(coeff->mm[index]),
		  &(coeff->clm[index]), &(coeff->slm[index]),
		  &fjunk0, &fjunk1, &fjunk2, &fjunk3) != 8){
	  fprintf(stderr,"read error l %i m %i \n",coeff->ll[index],coeff->mm[index]);
	  exit(-1);
	}
        /*
        fprintf(stderr, "%d %d %d %e %e\n", index,
                coeff->ll[index], coeff->mm[index],
                coeff->clm[index], coeff->slm[index]);
        */
    }
    
    return;
}


void get_test_coeff(sph_harm *coeff)
{
    coeff->len = 1;
    allocate_sph_harm(coeff, coeff->len);

    coeff->ll[0] = 3;
    coeff->mm[0] = 2;
    coeff->clm[0] = 1e-2;
    coeff->slm[0] = 0;

    return;
}


void get_mesh(mesh *grid, int ntheta, int nphi)
{
    int i;

    grid->ntheta = ntheta;
    grid->nphi = nphi;

    /* allocate memory */
    grid->theta = (double*)malloc(ntheta * sizeof(double));
    grid->phi = (double*)malloc(nphi * sizeof(double));


    /* create a regular mesh */
    for(i=0; i<grid->ntheta; i++)
        grid->theta[i] = i * (M_PI / (grid->ntheta-1));

    for(i=0; i<grid->nphi; i++)
        grid->phi[i] = i * (2.0 * M_PI / (grid->nphi-1));

    /*
    for(i=0; i<grid->ntheta; i++)
        fprintf(stderr, "%d, %f\n", i, grid->theta[i]);
    for(i=0; i<grid->nphi; i++)
        fprintf(stderr, "%d, %f\n", i, grid->phi[i]);
    */
    return;
}


void allocate_field(field *geoid)
{
    int i;
    mesh *grid = geoid->grid;

    /* allocate memory */
    geoid->data = (float **)malloc(grid->ntheta * sizeof(float *));
    if(!geoid->data){
      fprintf(stderr,"mem error\n");
      exit(-1);
    }
    for(i=0; i < grid->ntheta; i++){
      geoid->data[i] = (float *)calloc(grid->nphi, sizeof(float));
      if(!geoid->data[i]){
	fprintf(stderr,"mem error\n");
	exit(-1);
      }
    }
    return;
}


/* Compute fully normalized spherical harmonics coefficients
 * Copied from CitcomS lib/General_matrix_functions.c */

double modified_plgndr_a(int l, int m, double t)
{
    int i,ll;
    double x,fact1,fact2,fact,pll,pmm,pmmp1,somx2,plgndr;
    const double three=3.0;
    const double two=2.0;
    const double one=1.0;

    x = cos(t);
    pmm=one;
    if(m>0) {
        somx2=sqrt((one-x)*(one+x));
        fact1= three;
        fact2= two;
        for (i=1;i<=m;i++)   {
            fact=sqrt(fact1/fact2);
            pmm = -pmm*fact*somx2;
            fact1+=  two;
            fact2+=  two;
        }
    }

    if (l==m)
        plgndr = pmm;
    else  {
        pmmp1 = x*sqrt(two*m+three)*pmm;
        if(l==m+1)
            plgndr = pmmp1;
        else   {
            for (ll=m+2;ll<=l;ll++)  {
                fact1= sqrt((4.0*ll*ll-one)*(double)(ll-m)/(double)(ll+m));
                fact2= sqrt((2.0*ll+one)*(ll-m)*(ll+m-one)*(ll-m-one)
                            /(double)((two*ll-three)*(ll+m)));
                pll = ( x*fact1*pmmp1-fact2*pmm)/(ll-m);
                pmm = pmmp1;
                pmmp1 = pll;
            }
            plgndr = pll;
        }
    }

    plgndr /= sqrt(4.0*M_PI);

    if (m!=0) plgndr *= sqrt(two);

    return plgndr;
}


void project_sph_harm_to_mesh(sph_harm *coeff, field *geoid)
{
    int ll, mm;
    int i, j;
    int index;
    float val;
    float *cosm, *sinm;
    mesh *grid = geoid->grid;

    const int min_sph_degree_to_proj = 2;

    allocate_field(geoid);

    cosm = (float *)malloc(grid->nphi * sizeof(float));
    sinm = (float *)malloc(grid->nphi * sizeof(float));

    /* projecting */
    printf("Expanding spherical harmonics from degree %d to %d\n",
           min_sph_degree_to_proj, coeff->ll[coeff->len - 1]);

    for(index=0; index < coeff->len; index++) {
        ll = coeff->ll[index];
        mm = coeff->mm[index];

        /* skipping small ll */
        if(ll < min_sph_degree_to_proj) continue;

        for(j=0; j<grid->nphi; j++) {
            cosm[j] = cos(mm * grid->phi[j]);
            sinm[j] = sin(mm * grid->phi[j]);
        }

        for(i=0; i < grid->ntheta; i++) {
            /* val = Plm(theta) */
            val = modified_plgndr_a(ll, mm, grid->theta[i]);

            for(j=0; j < grid->nphi; j++) {
                /* data = val*(Clm*cos(m*phi) + Slm*sin(m*phi)) */
	      geoid->data[i][j] += val *
		(coeff->clm[index] * cosm[j] +
		 coeff->slm[index] * sinm[j]);
            }
        }
    }

    free(cosm);
    free(sinm);

    return;
}


void write_projected_field(char *filename, field *geoid)
{
    FILE *fp;
    char buffer[256];
    int i, j;
    float r2d = 180 / M_PI;

    /* open output file */
    fp = fopen(filename, "w");
    if(fp == NULL) {
        snprintf(buffer, 255, "Error: cannot open file: %s\n", filename);
        fputs(buffer, stderr);
    }


    /* write the field */
    /* the order is (longitude, latitude, geoid) */
    for(i=0; i<geoid->grid->ntheta; i++)
        for(j=0; j<geoid->grid->nphi; j++) {
            fprintf(fp, "%.3f %.3f %e\n",
                    geoid->grid->phi[j] * r2d,
                    90 - geoid->grid->theta[i] * r2d,
                    geoid->data[i][j]);
        }

    fclose(fp);
    return;
}


int main(int argc, char **argv)
{
    sph_harm coeff;
    mesh grid;
    field geoid_field;
    int ntheta, nphi;

    /* check the input */
    if(argc == 1) {
        print_help();
        return 1;
    }

    if(argc != 5) {
        fputs("Not enought input parameters provided!\n"
              "Run command with no argument for usage.\n", stderr);
        return 1;
    }

    /* we will use (theta, phi) in radian internally and will write the
     * result as (longitude, latitude) in degree later */
    nphi = strtol(argv[3], NULL, 10);
    ntheta = strtol(argv[4], NULL, 10);


    /* create a uniform mesh of (theta, phi) with (ntheta * nphi) points */
    /* theta in [0, pi] and phi in [0, 2*pi] */
    get_mesh(&grid, ntheta, nphi);

    /* attach the mesh to a (currently empty) field */
    geoid_field.grid = &grid;

    /* read the spherical harmonic coefficients from CitcomS geoid file */
    get_sph_harm_coeff(argv[1], &coeff);
    /* if debug, using this coeff */
    /* get_test_coeff(&coeff); */

    /* project the sph harm coefficients to the mesh */
    project_sph_harm_to_mesh(&coeff, &geoid_field);

    /* write the projected field as (longitude, latitude, field) */
    write_projected_field(argv[2], &geoid_field);

    return 0;
}


