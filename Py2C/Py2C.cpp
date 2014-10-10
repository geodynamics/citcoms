#include "Py2C.hpp"

namespace
{
  int pow2(int n)
  {
    if(n == 0)
      return 1;
    else
      return 2*pow2(n-1);
  }
  std::string to_string(int value)
  {
    std::ostringstream ostr;
    ostr << value;
    return ostr.str();
  }
}

Parameter::Parameter() 
  : value("INVALID"), section("INVALID"), isDefault(false), cRequired(false) 
{
}

Parameter::Parameter(const char* val, const char* sec, bool cReq, bool isDef) 
  : value(val), section(sec), isDefault(isDef),cRequired(cReq)
{
}

Py2CConverter::Py2CConverter(const char* pycfgfile, const char* ccfgfile)
  : fin(pycfgfile), fout(ccfgfile)
{
  initialize_parameters();
}
  
Py2CConverter::~Py2CConverter()
{
  fin.close();
  fout.close();
}
  
void Py2CConverter::convert(bool saveall)
{
  load();
  check_and_fix_errors();
  if(saveall)
    save_all();
  else
    save();
}

void Py2CConverter::initialize_parameters()
{
  // CitcomS
  // minstep is called steps in the Python version
  parameters["minstep"] =  Parameter("1", "CitcomS");
  parameters["maxstep"] = Parameter("1000", "CitcomS");
  parameters["maxtotstep"] = Parameter("1000000", "CitcomS");
  parameters["cpu_limits_in_seconds"] = Parameter("360000000", "CitcomS", true);

  parameters["solver"] = Parameter("regional", "CitcomS");
  
  // CitcomS.controller
  // storage_spacing is called monitoringFrequency in the Python versions
  parameters["storage_spacing"] = Parameter("100", "CitcomS.controller");
  parameters["checkpointFrequency"] = Parameter("100", "CitcomS.controller");
  
  // CitcomS.solver
  parameters["datadir"] = Parameter("\".\"", "CitcomS.solver");
  parameters["datafile"] = Parameter("\"regtest\"", "CitcomS.solver");
  parameters["datadir_old"] = Parameter("\".\"", "CitcomS.solver");
  parameters["datafile_old"] = Parameter("\"regtest\"", "CitcomS.solver");
  parameters["rayleigh"] = Parameter("100000", "CitcomS.solver",true);
  parameters["dissipation_number"] = Parameter("0.0", "CitcomS.solver");
  parameters["gruneisen"] = Parameter("0.0", "CitcomS.solver");
  parameters["surfaceT"] = Parameter("0.1", "CitcomS.solver");
  parameters["Q0"] = Parameter("0", "CitcomS.solver");
  parameters["stokes_flow_only"] = Parameter("0", "CitcomS.solver");
  parameters["verbose"] = Parameter("0", "CitcomS.solver");
  parameters["see_convergence"] = Parameter("1", "CitcomS.solver");

  // CitcomS.solver.mesher
  parameters["nproc_surf"] = Parameter("1", "CitcomS.solver.mesher");
  parameters["nprocx"] = Parameter("1","CitcomS.solver.mesher");
  parameters["nprocy"] = Parameter("1","CitcomS.solver.mesher");
  parameters["nprocz"] = Parameter("1","CitcomS.solver.mesher");
  parameters["coor"] = Parameter("0","CitcomS.solver.mesher");
  parameters["coor_file"] = Parameter("\"coor.dat\"","CitcomS.solver.mesher");
  parameters["coor_refine"] = Parameter("0.1,0.15,0.1,0.2","CitcomS.solver.mesher");
  parameters["nodex"] = Parameter("9","CitcomS.solver.mesher",true);
  parameters["nodey"] = Parameter("9","CitcomS.solver.mesher",true);
  parameters["nodez"] = Parameter("9","CitcomS.solver.mesher",true);
  parameters["levels"] = Parameter("1","CitcomS.solver.mesher");
  parameters["mgunitx"] = Parameter("8", "CitcomS.solver.mesher");
  parameters["mgunity"] = Parameter("8", "CitcomS.solver.mesher");
  parameters["mgunitz"] = Parameter("8", "CitcomS.solver.mesher");
  parameters["radius_outer"] = Parameter("1","CitcomS.solver.mesher");
  parameters["radius_inner"] = Parameter("0.55","CitcomS.solver.mesher");
  parameters["theta_min"] = Parameter("1.0708","CitcomS.solver.mesher",true);
  parameters["theta_max"] = Parameter("2.0708","CitcomS.solver.mesher",true);
  parameters["fi_min"] = Parameter("0","CitcomS.solver.mesher",true);
  parameters["fi_max"] = Parameter("1","CitcomS.solver.mesher",true);
  parameters["r_grid_layers"] = Parameter("1","CitcomS.solver.mesher");
  parameters["rr"] = Parameter("1","CitcomS.solver.mesher");
  parameters["nr"] = Parameter("1","CitcomS.solver.mesher");

  // CitcomS.solver.tsolver
  parameters["ADV"] = Parameter("1","CitcomS.solver.tsolver");
  parameters["filter_temp"] = Parameter("0","CitcomS.solver.tsolver");
  parameters["monitor_max_T"] = Parameter("1","CitcomS.solver.tsolver");
  parameters["finetunedt"] = Parameter("0.9","CitcomS.solver.tsolver");
  parameters["fixed_timestep"] = Parameter("0.0","CitcomS.solver.tsolver");
  parameters["adv_gamma"] = Parameter("0.5","CitcomS.solver.tsolver");
  parameters["adv_sub_iterations"] = Parameter("2","CitcomS.solver.tsolver");
  parameters["inputdiffusivity"] = Parameter("1","CitcomS.solver.tsolver");
  
  // CitcomS.solver.vsolver
  parameters["Solver"] = Parameter("cgrad","CitcomS.solver.vsolver");
  parameters["node_assemble"] = Parameter("1","CitcomS.solver.vsolver");
  parameters["precond"] = Parameter("1","CitcomS.solver.vsolver");
  parameters["accuracy"] = Parameter("1.0e-4","CitcomS.solver.vsolver");
  parameters["uzawa"] = Parameter("cg","CitcomS.solver.vsolver");
  parameters["compress_iter_maxstep"] = Parameter("100","CitcomS.solver.vsolver");
  parameters["mg_cycle"] = Parameter("1","CitcomS.solver.vsolver");
  parameters["down_heavy"] = Parameter("3","CitcomS.solver.vsolver");
  parameters["up_heavy"] = Parameter("3","CitcomS.solver.vsolver");
  parameters["vlowstep"] = Parameter("1000","CitcomS.solver.vsolver");
  parameters["vhighstep"] = Parameter("3","CitcomS.solver.vsolver");
  parameters["max_mg_cycles"] = Parameter("50","CitcomS.solver.vsolver");
  parameters["piterations"] = Parameter("1000","CitcomS.solver.vsolver");
  parameters["aug_lagr"] = Parameter("1","CitcomS.solver.vsolver");
  parameters["aug_number"] = Parameter("2000","CitcomS.solver.vsolver");
  parameters["remove_rigid_rotation"] = Parameter("1","CitcomS.solver.vsolver");
  parameters["remove_angular_momentum"] = Parameter("1","CitcomS.solver.vsolver");
  parameters["inner_accuracy_scale"] = Parameter("1.0","CitcomS.solver.vsolver");
  parameters["check_continuity_convergence"] = Parameter("1","CitcomS.solver.vsolver");
  parameters["check_pressure_convergence"] = Parameter("1","CitcomS.solver.vsolver");
  parameters["only_check_vel_convergence"] = Parameter("0","CitcomS.solver.vsolver");
  parameters["inner_remove_rigid_rotation"] = Parameter("0","CitcomS.solver.vsolver");

  // CitcomS.solver.bc
  parameters["side_sbcs"] = Parameter("0","CitcomS.solver.bc");
  parameters["pseudo_free_surf"] = Parameter("0","CitcomS.solver.bc");
  parameters["topvbc"] = Parameter("0","CitcomS.solver.bc");
  parameters["topvbxval"] = Parameter("0","CitcomS.solver.bc");
  parameters["topvbyval"] = Parameter("0","CitcomS.solver.bc");
  parameters["botvbc"] = Parameter("0","CitcomS.solver.bc");
  parameters["botvbxval"] = Parameter("0","CitcomS.solver.bc");
  parameters["botvbyval"] = Parameter("0","CitcomS.solver.bc");
  parameters["toptbc"] = Parameter("1","CitcomS.solver.bc");
  parameters["toptbcval"] = Parameter("0","CitcomS.solver.bc");
  parameters["bottbc"] = Parameter("1","CitcomS.solver.bc");
  parameters["bottbcval"] = Parameter("1","CitcomS.solver.bc");
  parameters["temperature_bound_adj"] = Parameter("0","CitcomS.solver.bc");
  parameters["depth_bound_adj"] = Parameter("0.157","CitcomS.solver.bc");
  parameters["width_bound_adj"] = Parameter("0.08727","CitcomS.solver.bc");

  // CitcomS.solver.const
  parameters["radius"] = Parameter("6.371e+06", "CitcomS.solver.const");
  parameters["density"] = Parameter("3340.0", "CitcomS.solver.const");
  parameters["thermdiff"] = Parameter("1e-06", "CitcomS.solver.const");
  parameters["gravacc"] = Parameter("9.81", "CitcomS.solver.const");
  parameters["thermexp"] = Parameter("3e-05", "CitcomS.solver.const");
  parameters["refvisc"] = Parameter("1e+21", "CitcomS.solver.const");
  parameters["cp"] = Parameter("1200", "CitcomS.solver.const");
  parameters["density_above"] = Parameter("1030.0", "CitcomS.solver.const");
  parameters["density_below"] = Parameter("6600.0", "CitcomS.solver.const");
  parameters["z_lith"] = Parameter("0.014", "CitcomS.solver.const");
  parameters["z_410"] = Parameter("0.06435", "CitcomS.solver.const");
  parameters["z_lmantle"] = Parameter("0.105", "CitcomS.solver.const");
  parameters["z_cmb"] = Parameter("0.439", "CitcomS.solver.const");

  // CitcomS.solver.ic
  parameters["restart"] = Parameter("0", "CitcomS.solver.ic");
  parameters["post_p"] = Parameter("0", "CitcomS.solver.ic");
  parameters["solution_cycles_init"] = Parameter("0", "CitcomS.solver.ic");
  parameters["zero_elapsed_time"] = Parameter("1", "CitcomS.solver.ic");
  parameters["tic_method"] = Parameter("0", "CitcomS.solver.ic");
  parameters["num_perturbations"] = Parameter("1", "CitcomS.solver.ic",true);
  parameters["perturbl"] = Parameter("1", "CitcomS.solver.ic",true);
  parameters["perturbm"] = Parameter("1", "CitcomS.solver.ic",true);
  parameters["perturblayer"] = Parameter("5", "CitcomS.solver.ic",true);
  parameters["perturbmag"] = Parameter("0.05", "CitcomS.solver.ic",true);
  parameters["half_space_age"] = Parameter("40", "CitcomS.solver.ic");
  parameters["mantle_temp"] = Parameter("1.0", "CitcomS.solver.ic");
  parameters["blob_center"] = Parameter("[-999,-999,-999]", "CitcomS.solver.ic");
  parameters["blob_radius"] = Parameter("0.063", "CitcomS.solver.ic");
  parameters["blob_dT"] = Parameter("0.18", "CitcomS.solver.ic");

  // CitcomS.solver.output
  parameters["output_format"] = Parameter("\"ascii\"", "CitcomS.solver.output");
  parameters["output_optional"] = Parameter("\"surf,botm,tracer\"", "CitcomS.solver.output");
  parameters["output_ll_max"] = Parameter("20", "CitcomS.solver.output");
  parameters["self_gravitation"] = Parameter("0", "CitcomS.solver.output");
  parameters["use_cbf_topo"] = Parameter("0", "CitcomS.solver.output");
  parameters["cb_block_size"] = Parameter("1048576", "CitcomS.solver.output");
  parameters["cb_buffer_size"] = Parameter("4194304", "CitcomS.solver.output");
  parameters["sieve_buf_size"] = Parameter("1048576", "CitcomS.solver.output");
  parameters["output_alignment"] = Parameter("262144", "CitcomS.solver.output");
  parameters["output_alignment_threshold"] = Parameter("524288", "CitcomS.solver.output");
  parameters["cache_mdc_nelmts"] = Parameter("10330", "CitcomS.solver.output");
  parameters["cache_rdcc_nelmts"] = Parameter("521", "CitcomS.solver.output");
  parameters["cache_rdcc_nbytes"] = Parameter("1048576", "CitcomS.solver.output");
  parameters["write_q_files"] = Parameter("0", "CitcomS.solver.output");
  parameters["vtk_format"] = Parameter("binary", "CitcomS.solver.output");
  parameters["gzdir_vtkio"] = Parameter("1", "CitcomS.solver.output");
  parameters["gzdir_rnr"] = Parameter("0", "CitcomS.solver.output");

  // CitcomS.solver.param
  parameters["reference_state"] = Parameter("1", "CitcomS.solver.param");
  parameters["refstate_file"] = Parameter("\"refstate.dat\"", "CitcomS.solver.param");
  parameters["mineral_physics_model"] = Parameter("3", "CitcomS.solver.param");
  parameters["file_vbcs"] = Parameter("0", "CitcomS.solver.param");
  parameters["vel_bound_file"] = Parameter("\"bevel.dat\"", "CitcomS.solver.param");
  parameters["mat_control"] = Parameter("0", "CitcomS.solver.param");
  parameters["mat_file"] = Parameter("\"mat.dat\"", "CitcomS.solver.param");
  parameters["lith_age"] = Parameter("0", "CitcomS.solver.param");
  parameters["lith_age_file"] = Parameter("\"age.dat\"", "CitcomS.solver.param");
  parameters["lith_age_time"] = Parameter("0", "CitcomS.solver.param");
  parameters["lith_age_depth"] = Parameter("0.0314", "CitcomS.solver.param");
  parameters["start_age"] = Parameter("40", "CitcomS.solver.param");
  parameters["reset_startage"] = Parameter("0", "CitcomS.solver.param");
  parameters["file_tbcs"] = Parameter("0", "CitcomS.solver.param");
  parameters["temp_bound_file"] = Parameter("btemp.dat", "CitcomS.solver.param");

  // CitcomS.solver.phase
  parameters["Ra_410"] = Parameter("0", "CitcomS.solver.phase");
  parameters["clapeyron410"] = Parameter("0.0235", "CitcomS.solver.phase");
  parameters["transT410"] = Parameter("0.78", "CitcomS.solver.phase");
  parameters["width410"] = Parameter("0.0058", "CitcomS.solver.phase");
  parameters["Ra_670"] = Parameter("0", "CitcomS.solver.phase");
  parameters["clapeyron670"] = Parameter("-0.0235", "CitcomS.solver.phase");
  parameters["transT670"] = Parameter("0.78", "CitcomS.solver.phase");
  parameters["width670"] = Parameter("0.0058", "CitcomS.solver.phase");
  parameters["Ra_cmb"] = Parameter("0", "CitcomS.solver.phase");
  parameters["clapeyroncmb"] = Parameter("-0.0235", "CitcomS.solver.phase");
  parameters["transTcmb"] = Parameter("0.875", "CitcomS.solver.phase");
  parameters["widthcmb"] = Parameter("0.0058", "CitcomS.solver.phase");

  // CitcomS.solver.tracer
  parameters["tracer"] = Parameter("0", "Citcoms.Solver.tracer");
  parameters["tracer_ic_method"] = Parameter("0", "Citcoms.Solver.tracer");
  parameters["tracers_per_element"] = Parameter("10", "Citcoms.Solver.tracer");
  parameters["tracer_file"] = Parameter("\"tracer.dat\"", "Citcoms.Solver.tracer");
  parameters["tracer_flavors"] = Parameter("0", "Citcoms.Solver.tracer");
  parameters["ic_method_for_flavors"] = Parameter("0", "Citcoms.Solver.tracer");
  parameters["z_interface"] = Parameter("0.7", "Citcoms.Solver.tracer");
  parameters["itracer_warnings"] = Parameter("1", "Citcoms.Solver.tracer");
  parameters["regular_grid_deltheta"] = Parameter("1.0", "Citcoms.Solver.tracer");
  parameters["regular_grid_delphi"] = Parameter("1.0", "Citcoms.Solver.tracer");
  parameters["chemical_buoyancy"] = Parameter("1", "Citcoms.Solver.tracer");
  parameters["buoy_type"] = Parameter("1", "Citcoms.Solver.tracer");
  parameters["buoyancy_ratio"] = Parameter("1.0", "Citcoms.Solver.tracer");
  parameters["tracer_enriched"] = Parameter("0", "Citcoms.Solver.tracer");
  parameters["Q0_enriched"] = Parameter("0.0", "Citcoms.Solver.tracer");

  // CitcomS.solver.visc
  parameters["Viscosity"] = Parameter("\"system\"", "CitcomS.solver.visc");
  parameters["visc_smooth_method"] = Parameter("3", "CitcomS.solver.visc");
  parameters["VISC_UPDATE"] = Parameter("1", "CitcomS.solver.visc");
  parameters["num_mat"] = Parameter("4", "CitcomS.solver.visc",true);
  parameters["visc0"] = Parameter("1,1,1,1", "CitcomS.solver.visc");
  parameters["TDEPV"] = Parameter("0", "CitcomS.solver.visc");
  parameters["rheol"] = Parameter("3", "CitcomS.solver.visc");
  parameters["viscE"] = Parameter("1,1,1,1", "CitcomS.solver.visc");
  parameters["viscT"] = Parameter("1,1,1,1", "CitcomS.solver.visc");
  parameters["viscZ"] = Parameter("1,1,1,1", "CitcomS.solver.visc");
  parameters["SDEPV"] = Parameter("0", "CitcomS.solver.visc");
  parameters["sdepv_misfit"] = Parameter("0.001", "CitcomS.solver.visc");
  parameters["sdepv_expt"] = Parameter("1,1,1,1", "CitcomS.solver.visc");
  parameters["PDEPV"] = Parameter("0", "CitcomS.solver.visc");
  parameters["pdepv_a"] = Parameter("1e20,1e20,1e20,1e20", "CitcomS.solver.visc");
  parameters["pdepv_b"] = Parameter("0,0,0,0", "CitcomS.solver.visc");
  parameters["pdepv_y"] = Parameter("1e20,1e20,1e20,1e20", "CitcomS.solver.visc");
  parameters["pdepv_eff"] = Parameter("1", "CitcomS.solver.visc");
  parameters["pdepv_offset"] = Parameter("0", "CitcomS.solver.visc");
  parameters["CDEPV"] = Parameter("0", "CitcomS.solver.visc");
  parameters["cdepv_ff"] = Parameter("1,1", "CitcomS.solver.visc");
  parameters["low_visc_channel"] = Parameter("0", "CitcomS.solver.visc");
  parameters["low_visc_wedge"] = Parameter("0", "CitcomS.solver.visc");
  parameters["lv_min_radius"] = Parameter("0.9764", "CitcomS.solver.visc");
  parameters["lv_max_radius"] = Parameter("0.9921", "CitcomS.solver.visc");
  parameters["lv_channel_thickness"] = Parameter("0.0047", "CitcomS.solver.visc");
  parameters["lv_reduction"] = Parameter("0.5", "CitcomS.solver.visc");
  parameters["VMIN"] = Parameter("0", "CitcomS.solver.visc");
  parameters["visc_min"] = Parameter("0.001", "CitcomS.solver.visc");
  parameters["VMAX"] = Parameter("0", "CitcomS.solver.visc");
  parameters["visc_max"] = Parameter("1000", "CitcomS.solver.visc");
  parameters["z_layer"] = Parameter("-999,-999,-999,-999", "CitcomS.solver.visc");
  parameters["visc_layer_control"] = Parameter("0", "CitcomS.solver.visc");
  parameters["visc_layer_file"] = Parameter("visc.dat", "CitcomS.solver.visc");
}

void Py2CConverter::load()
{
  std::string cfgline;
  while(std::getline(fin, cfgline))
  {
    if(cfgline.length() == 0)
    {
      // blank line, do nothing
    }
    else if(cfgline[0] == '#')
    {
      // comment line, do nothing
    }
    else if(cfgline[0] == '[')
    {
      size_t pos = cfgline.find_first_of(']');
      assert(pos != std::string::npos); // [ without a closing ] is malformed
      section_names.insert(cfgline.substr(1,pos-1));
    }
    else
    {
      size_t pos = cfgline.find_first_of(';');
      if(pos != std::string::npos)
	      cfgline = cfgline.substr(0,pos);
      std::string name, value;
      pos = cfgline.find_first_of('=');
      assert(pos != std::string::npos); // parameter line without = is malformed
      name = cfgline.substr(0,pos);
      value = cfgline.substr(pos+1);

      // trim any leading and trailing whitespaces from name and value
      name.erase(std::remove_if(name.begin(), name.end(), ::isspace), name.end());
      value.erase(std::remove_if(value.begin(), value.end(), ::isspace), value.end());

      // parse the name value pair, and record any unknown parameters
      if(!parse(name, value))
      {
        std::ostringstream ostr;
        ostr << "# unknown parameter " << name << " with value " << value;
        log_messages.push_back(ostr.str());
      }
    }
  }
}

bool Py2CConverter::parse(const std::string& name, const std::string& value)
{
  std::string namecopy = name;

  // Parameters that are named differently in C and Python config files
  if(namecopy == "steps")
    namecopy = "minstep";
  else if(namecopy == "monitoringFrequency")
    namecopy = "storage_spacing";

  if(parameters.find(namecopy) == parameters.end())
  {
    return false;
  }

  Parameter& prm(parameters[namecopy]);
  prm.value = value;
  prm.isDefault = false;
  return true;
}

void Py2CConverter::check_and_fix_errors()
{
  // Do the error checking/corrections that the Python version does automatically

  // for the full spherical model, nproc_surf must be 12, for regional it must be 1
  // for the full spherical model, nprocx=nprocy, mgunitx=mgunity
  if(parameters["solver"].value == "full")
  {
    if(std::atoi(parameters["nproc_surf"].value.c_str()) != 12)
    {
      log_messages.push_back("# WARNING: incorrect value for nproc_surf found; "
			     "setting it to 12");
      parameters["nproc_surf"].value = "12";
      parameters["nproc_surf"].isDefault = false;
    }
    if(parameters["nprocx"].value != parameters["nprocy"].value)
    {
      log_messages.push_back("# WARNING: for solver=full, need nprocx=nprocy. "
			     "changing value of nprocy to be equal to nprocx.");
      parameters["nprocy"].value = parameters["nprocx"].value;
      parameters["nprocx"].isDefault = false;
      parameters["nprocy"].isDefault = false;
    }
    if(parameters["mgunitx"].value != parameters["mgunity"].value)
    {
      log_messages.push_back("# WARNING: for solver=full, need mgunitx=mgunity. "
			     "changing value of mgunity to be equal to mgunitx.");
      parameters["mgunity"].value = parameters["mgunitx"].value;
      parameters["mgunitx"].isDefault = false;
      parameters["mgunity"].isDefault = false;
    }
  }
  else if(parameters["solver"].value == "regional")
  {
    if(std::atoi(parameters["nproc_surf"].value.c_str()) != 1)
    {
      log_messages.push_back("# WARNING: incorrect value for nproc_surf found; "
			     "setting it to 1");
      parameters["nproc_surf"].value = "1";
      parameters["nproc_surf"].isDefault = false;
    }
  }

  // if Solver=multigrid, then Equation 6.7 needs to be satisfied.
  // nodex = 1 + nprocx * mgunitx * 2^(levels-1)
  if(parameters["Solver"].value == "multigrid")
  {
    int nprocx, nprocy, nprocz, nodex, nodey, nodez;
    int mgunitx, mgunity, mgunitz, levels;
    nprocx = std::atoi(parameters["nprocx"].value.c_str()); 
    nprocy = std::atoi(parameters["nprocy"].value.c_str());
    nprocz = std::atoi(parameters["nprocz"].value.c_str());
    nodex = std::atoi(parameters["nodex"].value.c_str()); 
    nodey = std::atoi(parameters["nodey"].value.c_str());
    nodez = std::atoi(parameters["nodez"].value.c_str());
    mgunitx = std::atoi(parameters["mgunitx"].value.c_str()); 
    mgunity = std::atoi(parameters["mgunity"].value.c_str());
    mgunitz = std::atoi(parameters["mgunitz"].value.c_str());
    levels = std::atoi(parameters["levels"].value.c_str());

    if(nodex != (1+nprocx*mgunitx*pow2(levels-1)))
    {
      log_messages.push_back("# WARNING: changing mgunitx value to satisfy eqn 6.7");
      parameters["mgunitx"].value = to_string((nodex - 1)/(nprocx*pow2(levels-1)));
      parameters["mgunitx"].isDefault = false;
    }
    if(nodey != (1+nprocy*mgunity*pow2(levels-1)))
    {
      log_messages.push_back("# WARNING: changing mgunity value to satisfy eqn 6.7");
      parameters["mgunity"].value = to_string((nodey - 1)/(nprocy*pow2(levels-1)));
      parameters["mgunity"].isDefault = false;
    }
    if(nodez != (1+nprocz*mgunitz*pow2(levels-1)))
    {
      log_messages.push_back("# WARNING: changing mgunitz value to satisfy eqn 6.7");
      parameters["mgunitz"].value = to_string((nodez - 1)/(nprocz*pow2(levels-1)));
      parameters["mgunitz"].isDefault = false;
    }
  }
}

void Py2CConverter::save()
{
  std::list<std::string> all_sections;
  all_sections.push_back(std::string("CitcomS")); 
	all_sections.push_back(std::string("CitcomS.controller")); 
  all_sections.push_back(std::string("CitcomS.solver"));
	all_sections.push_back(std::string("CitcomS.solver.mesher")); 
  all_sections.push_back(std::string("CitcomS.solver.tsolver"));
	all_sections.push_back(std::string("CitcomS.solver.vsolver")); 
  all_sections.push_back(std::string("CitcomS.solver.bc"));
	all_sections.push_back(std::string("CitcomS.solver.const")); 
  all_sections.push_back(std::string("CitcomS.solver.ic"));
	all_sections.push_back(std::string("CitcomS.solver.output")); 
  all_sections.push_back(std::string("CitcomS.solver.param"));
	all_sections.push_back(std::string("CitcomS.solver.phase")); 
  all_sections.push_back(std::string("CitcomS.solver.tracer"));
	all_sections.push_back(std::string("CitcomS.solver.visc"));

  std::list<std::string>::const_iterator citer;
  for(citer = all_sections.begin(); citer != all_sections.end(); ++citer)
  {

    // check if atleast one parameter in this section has been changed from its
    // default value
    if(section_names.count(*citer) > 0)
    {
      fout << "# " << *citer << std::endl;
      std::map<std::string, Parameter>::const_iterator p;
      for(p = parameters.begin(); p != parameters.end(); ++p)
      {
        if((p->second.section == *citer) && ((!p->second.isDefault)||(p->second.cRequired)))
        {
          fout << p->first << "=" << p->second.value << std::endl;
        }
      }
      fout << std::endl;
    }
    else
    {
      // check if there are any cRequired parameters in this section or any 
      // parameters that have been changed from their default values
      std::vector<std::string> namevals;
      namevals.push_back(*citer);
      std::map<std::string, Parameter>::const_iterator p;
      for(p = parameters.begin(); p != parameters.end(); ++p)
      {
        if((p->second.section == *citer) && ((p->second.cRequired)||(!p->second.isDefault)))
        {
          namevals.push_back(p->first + "=" + p->second.value);
        }
      }
      if(namevals.size() > 1)
      {
        fout << "# " << namevals[0] << std::endl;
        for(size_t ndx=1; ndx < namevals.size(); ++ndx)
        {
          fout << namevals[ndx] << std::endl;
        }
        fout << std::endl;
      }
    }
  }

  // write out all the log messages at the end of the config file
  for(citer = log_messages.begin(); citer != log_messages.end(); ++citer)
    fout << (*citer) << std::endl;
}

void Py2CConverter::save_all()
{
  std::list<std::string> all_sections;
  all_sections.push_back(std::string("CitcomS")); 
	all_sections.push_back(std::string("CitcomS.controller")); 
  all_sections.push_back(std::string("CitcomS.solver"));
	all_sections.push_back(std::string("CitcomS.solver.mesher")); 
  all_sections.push_back(std::string("CitcomS.solver.tsolver"));
	all_sections.push_back(std::string("CitcomS.solver.vsolver")); 
  all_sections.push_back(std::string("CitcomS.solver.bc"));
	all_sections.push_back(std::string("CitcomS.solver.const")); 
  all_sections.push_back(std::string("CitcomS.solver.ic"));
	all_sections.push_back(std::string("CitcomS.solver.output")); 
  all_sections.push_back(std::string("CitcomS.solver.param"));
	all_sections.push_back(std::string("CitcomS.solver.phase")); 
  all_sections.push_back(std::string("CitcomS.solver.tracer"));
	all_sections.push_back(std::string("CitcomS.solver.visc"));

  std::list<std::string>::const_iterator citer;
  for(citer = all_sections.begin(); citer != all_sections.end(); ++citer)
  {

    fout << "# " << *citer << std::endl;
    std::map<std::string, Parameter>::const_iterator p;
    for(p = parameters.begin(); p != parameters.end(); ++p)
    {
      if((p->second.section == *citer))
      {
        fout << p->first << "=" << p->second.value << std::endl;
      }
    }
    fout << std::endl;
  }
  // write out all the log messages at the end of the config file
  for(citer = log_messages.begin(); citer != log_messages.end(); ++citer)
    fout << (*citer) << std::endl;
}
