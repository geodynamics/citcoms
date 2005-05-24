struct CONVECTION { /* information controlling convection problems */

    int tic_method;
    float half_space_age;

#define PERTURB_MAX_LAYERS 255
    int number_of_perturbations;
    int perturb_ll[PERTURB_MAX_LAYERS];
    int perturb_mm[PERTURB_MAX_LAYERS];
    int load_depth[PERTURB_MAX_LAYERS];
    float perturb_mag[PERTURB_MAX_LAYERS];

  float blob_center[3];
  float blob_radius;
  float blob_dT;

    struct SOURCES {
	    int number;
	    float t_offset;
	    float Q[10];
	    float lambda[10];
	}  heat_sources;

} convection;


