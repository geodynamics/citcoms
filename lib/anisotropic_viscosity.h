/* 


header files for Anisotropic_viscosity.c, for use with CitcomCU and CitcomS


EDIT THE FILES in hc, not the citcom subdirectories



 */

#ifndef __CITCOM_READ_ANIVISC_HEADER__

#define CITCOM_ANIVISC_ORTHO_MODE 1
#define CITCOM_ANIVISC_TI_MODE 2

#define CITCOM_ANIVISC_ALIGN_WITH_VEL 0
#define CITCOM_ANIVISC_ALIGN_WITH_ISA 1
#define CITCOM_ANIVISC_MIXED_ALIGN 2

void get_constitutive(double [6][6], double, double, int, float,float,float,float,int,struct All_variables *);
void get_constitutive_ti_viscosity(double [6][6], double, double, double [3], int, double, double);
void get_constitutive_orthotropic_viscosity(double [6][6], double, double [3], int, double, double);
void get_constitutive_isotropic(double [6][6]);
void set_anisotropic_viscosity_at_element_level(struct All_variables *, int);



void conv_cart4x4_to_spherical(double [3][3][3][3], double, double, double [3][3][3][3]);
void conv_cart6x6_to_spherical(double [6][6], double, double, double [6][6]);
void rotate_ti6x6_to_director(double [6][6], double [3]);
void get_citcom_spherical_rot(double, double, double [3][3]);
void get_orth_delta(double [3][3][3][3], double [3]);
void align_director_with_ISA_for_element(struct All_variables *, int);
unsigned char calc_isa_from_vgm(double [3][3], double [3], int, double [3], struct All_variables *, int);
int is_pure_shear(double [3][3], double [3][3], double [3][3]);
void rot_4x4(double [3][3][3][3], double [3][3], double [3][3][3][3]);
void zero_6x6(double [6][6]);
void zero_4x4(double [3][3][3][3]);
void copy_4x4(double [3][3][3][3], double [3][3][3][3]);
void copy_6x6(double [6][6], double [6][6]);
void print_6x6_mat(FILE *, double [6][6]);
void c4fromc6(double [3][3][3][3], double [6][6]);
void c6fromc4(double [6][6], double [3][3][3][3]);
void isacalc(double [3][3], double *, double [3], struct All_variables *, int *);
void f_times_ft(double [3][3], double [3][3]);
void drex_eigen(double [3][3], double [3][3], int *);
void malmul_scaled_id(double [3][3], double [3][3], double, double);


void print_3x3_mat(FILE *, double [3][3]);

void calc_exp_matrixt(double [3][3],double ,double [3][3],
		      struct All_variables *);

void dgpadm_(int *,int *,double *,double *,int *,double *,int *,
	     int *,int *,int *,int *);
void get_vgm_p(double [4][9],struct Shape_function *,
	       struct Shape_function_dx *,
	       struct CC *, struct CCX *, double [4][9],
	       int ,int , int , int ,
	       double [3][3], double [3]);

void normalize_director_at_nodes(struct All_variables *, float *, float *, float *, int);
void normalize_director_at_gint(struct All_variables *, float *, float *, float *, int);

#define __CITCOM_READ_ANIVISC_HEADER__
#endif
