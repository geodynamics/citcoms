struct CONVECTION { /* information controlling convection problems */
    char old_T_file[100];

    float half_space_age;
    int half_space_cooling;

#define PERTURB_MAX_LAYERS 255
    int number_of_perturbations;
    int perturb_ll[PERTURB_MAX_LAYERS];
    int perturb_mm[PERTURB_MAX_LAYERS];
    int load_depth[PERTURB_MAX_LAYERS];
    float perturb_mag[PERTURB_MAX_LAYERS];

    struct SOURCES {
	    int number;
	    float t_offset;
	    float Q[10];
	    float lambda[10];
	}  heat_sources;

} convection;


