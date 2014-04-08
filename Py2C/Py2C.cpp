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
  parameters["minstep"] =  {"1", "CitcomS"};
  parameters["maxstep"] = {"1000", "CitcomS"};
  parameters["maxtotstep"] = {"1000000", "CitcomS"};
  parameters["cpu_limits_in_seconds"] = {"360000000", "CitcomS", true};

  parameters["solver"] = {"regional", "CitcomS"};
  
  // CitcomS.controller
  // storage_spacing is called monitoringFrequency in the Python versions
  parameters["storage_spacing"] = {"100", "CitcomS.controller"};
  parameters["checkpointFrequency"] = {"100", "CitcomS.controller"};
  
  // CitcomS.solver
  parameters["datadir"] = {"\".\"", "CitcomS.solver"};
  parameters["datafile"] = {"\"regtest\"", "CitcomS.solver"};
  parameters["datadir_old"] = {"\".\"", "CitcomS.solver"};
  parameters["datafile_old"] = {"\"regtest\"", "CitcomS.solver"};
  parameters["rayleigh"] = {"100000", "CitcomS.solver",true};
  parameters["dissipation_number"] = {"0.0", "CitcomS.solver"};
  parameters["gruneisen"] = {"0.0", "CitcomS.solver"};
  parameters["surfaceT"] = {"0.1", "CitcomS.solver"};
  parameters["Q0"] = {"0", "CitcomS.solver"};
  parameters["stokes_flow_only"] = {"0", "CitcomS.solver"};
  parameters["verbose"] = {"0", "CitcomS.solver"};
  parameters["see_convergence"] = {"1", "CitcomS.solver"};

  // CitcomS.solver.mesher
  parameters["nproc_surf"] = {"1", "CitcomS.solver.mesher"};
  parameters["nprocx"] = {"1","CitcomS.solver.mesher"};
  parameters["nprocy"] = {"1","CitcomS.solver.mesher"};
  parameters["nprocz"] = {"1","CitcomS.solver.mesher"};
  parameters["coor"] = {"0","CitcomS.solver.mesher"};
  parameters["coor_file"] = {"\"coor.dat\"","CitcomS.solver.mesher"};
  parameters["coor_refine"] = {"0.1,0.15,0.1,0.2","CitcomS.solver.mesher"};
  parameters["nodex"] = {"9","CitcomS.solver.mesher",true};
  parameters["nodey"] = {"9","CitcomS.solver.mesher",true};
  parameters["nodez"] = {"9","CitcomS.solver.mesher",true};
  parameters["levels"] = {"1","CitcomS.solver.mesher"};
  parameters["mgunitx"] = {"8", "CitcomS.solver.mesher"};
  parameters["mgunity"] = {"8", "CitcomS.solver.mesher"};
  parameters["mgunitz"] = {"8", "CitcomS.solver.mesher"};
  parameters["radius_outer"] = {"1","CitcomS.solver.mesher"};
  parameters["radius_inner"] = {"0.55","CitcomS.solver.mesher"};
  parameters["theta_min"] = {"1.0708","CitcomS.solver.mesher",true};
  parameters["theta_max"] = {"2.0708","CitcomS.solver.mesher",true};
  parameters["fi_min"] = {"0","CitcomS.solver.mesher",true};
  parameters["fi_max"] = {"1","CitcomS.solver.mesher",true};
  parameters["r_grid_layers"] = {"1","CitcomS.solver.mesher"};
  parameters["rr"] = {"1","CitcomS.solver.mesher"};
  parameters["nr"] = {"1","CitcomS.solver.mesher"};

  // CitcomS.solver.tsolver
  parameters["ADV"] = {"1","CitcomS.solver.tsolver"};
  parameters["filter_temp"] = {"0","CitcomS.solver.tsolver"};
  parameters["monitor_max_T"] = {"1","CitcomS.solver.tsolver"};
  parameters["finetunedt"] = {"0.9","CitcomS.solver.tsolver"};
  parameters["fixed_timestep"] = {"0.0","CitcomS.solver.tsolver"};
  parameters["adv_gamma"] = {"0.5","CitcomS.solver.tsolver"};
  parameters["adv_sub_iterations"] = {"2","CitcomS.solver.tsolver"};
  parameters["inputdiffusivity"] = {"1","CitcomS.solver.tsolver"};
  
  // CitcomS.solver.vsolver
  parameters["Solver"] = {"cgrad","CitcomS.solver.vsolver"};
  parameters["node_assemble"] = {"1","CitcomS.solver.vsolver"};
  parameters["precond"] = {"1","CitcomS.solver.vsolver"};
  parameters["accuracy"] = {"1.0e-4","CitcomS.solver.vsolver"};
  parameters["uzawa"] = {"cg","CitcomS.solver.vsolver"};
  parameters["compress_iter_maxstep"] = {"100","CitcomS.solver.vsolver"};
  parameters["mg_cycle"] = {"1","CitcomS.solver.vsolver"};
  parameters["down_heavy"] = {"3","CitcomS.solver.vsolver"};
  parameters["up_heavy"] = {"3","CitcomS.solver.vsolver"};
  parameters["vlowstep"] = {"1000","CitcomS.solver.vsolver"};
  parameters["vhighstep"] = {"3","CitcomS.solver.vsolver"};
  parameters["max_mg_cycles"] = {"50","CitcomS.solver.vsolver"};
  parameters["piterations"] = {"1000","CitcomS.solver.vsolver"};
  parameters["aug_lagr"] = {"1","CitcomS.solver.vsolver"};
  parameters["aug_number"] = {"2000","CitcomS.solver.vsolver"};
  parameters["remove_rigid_rotation"] = {"1","CitcomS.solver.vsolver"};
  parameters["remove_angular_momentum"] = {"1","CitcomS.solver.vsolver"};
  parameters["inner_accuracy_scale"] = {"1.0","CitcomS.solver.vsolver"};
  parameters["check_continuity_convergence"] = {"1","CitcomS.solver.vsolver"};
  parameters["check_pressure_convergence"] = {"1","CitcomS.solver.vsolver"};
  parameters["only_check_vel_convergence"] = {"0","CitcomS.solver.vsolver"};
  parameters["inner_remove_rigid_rotation"] = {"0","CitcomS.solver.vsolver"};

  // CitcomS.solver.bc
  parameters["side_sbcs"] = {"0","CitcomS.solver.bc"};
  parameters["pseudo_free_surf"] = {"0","CitcomS.solver.bc"};
  parameters["topvbc"] = {"0","CitcomS.solver.bc"};
  parameters["topvbxval"] = {"0","CitcomS.solver.bc"};
  parameters["topvbyval"] = {"0","CitcomS.solver.bc"};
  parameters["botvbc"] = {"0","CitcomS.solver.bc"};
  parameters["botvbxval"] = {"0","CitcomS.solver.bc"};
  parameters["botvbyval"] = {"0","CitcomS.solver.bc"};
  parameters["toptbc"] = {"1","CitcomS.solver.bc"};
  parameters["toptbcval"] = {"0","CitcomS.solver.bc"};
  parameters["bottbc"] = {"1","CitcomS.solver.bc"};
  parameters["bottbcval"] = {"1","CitcomS.solver.bc"};
  parameters["temperature_bound_adj"] = {"0","CitcomS.solver.bc"};
  parameters["depth_bound_adj"] = {"0.157","CitcomS.solver.bc"};
  parameters["width_bound_adj"] = {"0.08727","CitcomS.solver.bc"};

  // CitcomS.solver.const
  parameters["radius"] = {"6.371e+06", "CitcomS.solver.const"};
  parameters["density"] = {"3340.0", "CitcomS.solver.const"};
  parameters["thermdiff"] = {"1e-06", "CitcomS.solver.const"};
  parameters["gravacc"] = {"9.81", "CitcomS.solver.const"};
  parameters["thermexp"] = {"3e-05", "CitcomS.solver.const"};
  parameters["refvisc"] = {"1e+21", "CitcomS.solver.const"};
  parameters["cp"] = {"1200", "CitcomS.solver.const"};
  parameters["density_above"] = {"1030.0", "CitcomS.solver.const"};
  parameters["density_below"] = {"6600.0", "CitcomS.solver.const"};
  parameters["z_lith"] = {"0.014", "CitcomS.solver.const"};
  parameters["z_410"] = {"0.06435", "CitcomS.solver.const"};
  parameters["z_lmantle"] = {"0.105", "CitcomS.solver.const"};
  parameters["z_cmb"] = {"0.439", "CitcomS.solver.const"};

  // CitcomS.solver.ic
  parameters["restart"] = {"0", "CitcomS.solver.ic"};
  parameters["post_p"] = {"0", "CitcomS.solver.ic"};
  parameters["solution_cycles_init"] = {"0", "CitcomS.solver.ic"};
  parameters["zero_elapsed_time"] = {"1", "CitcomS.solver.ic"};
  parameters["tic_method"] = {"0", "CitcomS.solver.ic"};
  parameters["num_perturbations"] = {"1", "CitcomS.solver.ic",true};
  parameters["perturbl"] = {"1", "CitcomS.solver.ic",true};
  parameters["perturbm"] = {"1", "CitcomS.solver.ic",true};
  parameters["perturblayer"] = {"5", "CitcomS.solver.ic",true};
  parameters["perturbmag"] = {"0.05", "CitcomS.solver.ic",true};
  parameters["half_space_age"] = {"40", "CitcomS.solver.ic"};
  parameters["mantle_temp"] = {"1.0", "CitcomS.solver.ic"};
  parameters["blob_center"] = {"[-999,-999,-999]", "CitcomS.solver.ic"};
  parameters["blob_radius"] = {"0.063", "CitcomS.solver.ic"};
  parameters["blob_dT"] = {"0.18", "CitcomS.solver.ic"};

  // CitcomS.solver.output
  parameters["output_format"] = {"\"ascii\"", "CitcomS.solver.output"};
  parameters["output_optional"] = {"\"surf,botm,tracer\"", "CitcomS.solver.output"};
  parameters["output_ll_max"] = {"20", "CitcomS.solver.output"};
  parameters["self_gravitation"] = {"0", "CitcomS.solver.output"};
  parameters["use_cbf_topo"] = {"0", "CitcomS.solver.output"};
  parameters["cb_block_size"] = {"1048576", "CitcomS.solver.output"};
  parameters["cb_buffer_size"] = {"4194304", "CitcomS.solver.output"};
  parameters["sieve_buf_size"] = {"1048576", "CitcomS.solver.output"};
  parameters["output_alignment"] = {"262144", "CitcomS.solver.output"};
  parameters["output_alignment_threshold"] = {"524288", "CitcomS.solver.output"};
  parameters["cache_mdc_nelmts"] = {"10330", "CitcomS.solver.output"};
  parameters["cache_rdcc_nelmts"] = {"521", "CitcomS.solver.output"};
  parameters["cache_rdcc_nbytes"] = {"1048576", "CitcomS.solver.output"};
  parameters["write_q_files"] = {"0", "CitcomS.solver.output"};
  parameters["vtk_format"] = {"binary", "CitcomS.solver.output"};
  parameters["gzdir_vtkio"] = {"1", "CitcomS.solver.output"};
  parameters["gzdir_rnr"] = {"0", "CitcomS.solver.output"};

  // CitcomS.solver.param
  parameters["reference_state"] = {"1", "CitcomS.solver.param"};
  parameters["refstate_file"] = {"\"refstate.dat\"", "CitcomS.solver.param"};
  parameters["mineral_physics_model"] = {"3", "CitcomS.solver.param"};
  parameters["file_vbcs"] = {"0", "CitcomS.solver.param"};
  parameters["vel_bound_file"] = {"\"bevel.dat\"", "CitcomS.solver.param"};
  parameters["mat_control"] = {"0", "CitcomS.solver.param"};
  parameters["mat_file"] = {"\"mat.dat\"", "CitcomS.solver.param"};
  parameters["lith_age"] = {"0", "CitcomS.solver.param"};
  parameters["lith_age_file"] = {"\"age.dat\"", "CitcomS.solver.param"};
  parameters["lith_age_time"] = {"0", "CitcomS.solver.param"};
  parameters["lith_age_depth"] = {"0.0314", "CitcomS.solver.param"};
  parameters["start_age"] = {"40", "CitcomS.solver.param"};
  parameters["reset_startage"] = {"0", "CitcomS.solver.param"};
  parameters["file_tbcs"] = {"0", "CitcomS.solver.param"};
  parameters["temp_bound_file"] = {"btemp.dat", "CitcomS.solver.param"};

  // CitcomS.solver.phase
  parameters["Ra_410"] = {"0", "CitcomS.solver.phase"};
  parameters["clapeyron410"] = {"0.0235", "CitcomS.solver.phase"};
  parameters["transT410"] = {"0.78", "CitcomS.solver.phase"};
  parameters["width410"] = {"0.0058", "CitcomS.solver.phase"};
  parameters["Ra_670"] = {"0", "CitcomS.solver.phase"};
  parameters["clapeyron670"] = {"-0.0235", "CitcomS.solver.phase"};
  parameters["transT670"] = {"0.78", "CitcomS.solver.phase"};
  parameters["width670"] = {"0.0058", "CitcomS.solver.phase"};
  parameters["Ra_cmb"] = {"0", "CitcomS.solver.phase"};
  parameters["clapeyroncmb"] = {"-0.0235", "CitcomS.solver.phase"};
  parameters["transTcmb"] = {"0.875", "CitcomS.solver.phase"};
  parameters["widthcmb"] = {"0.0058", "CitcomS.solver.phase"};

  // CitcomS.solver.tracer
  parameters["tracer"] = {"0", "Citcoms.Solver.tracer"};
  parameters["tracer_ic_method"] = {"0", "Citcoms.Solver.tracer"};
  parameters["tracers_per_element"] = {"10", "Citcoms.Solver.tracer"};
  parameters["tracer_file"] = {"\"tracer.dat\"", "Citcoms.Solver.tracer"};
  parameters["tracer_flavors"] = {"0", "Citcoms.Solver.tracer"};
  parameters["ic_method_for_flavors"] = {"0", "Citcoms.Solver.tracer"};
  parameters["z_interface"] = {"0.7", "Citcoms.Solver.tracer"};
  parameters["itracer_warnings"] = {"1", "Citcoms.Solver.tracer"};
  parameters["regular_grid_deltheta"] = {"1.0", "Citcoms.Solver.tracer"};
  parameters["regular_grid_delphi"] = {"1.0", "Citcoms.Solver.tracer"};
  parameters["chemical_buoyancy"] = {"1", "Citcoms.Solver.tracer"};
  parameters["buoy_type"] = {"1", "Citcoms.Solver.tracer"};
  parameters["buoyancy_ratio"] = {"1.0", "Citcoms.Solver.tracer"};
  parameters["tracer_enriched"] = {"0", "Citcoms.Solver.tracer"};
  parameters["Q0_enriched"] = {"0.0", "Citcoms.Solver.tracer"};

  // CitcomS.solver.visc
  parameters["Viscosity"] = {"\"system\"", "CitcomS.solver.visc"};
  parameters["visc_smooth_method"] = {"3", "CitcomS.solver.visc"};
  parameters["VISC_UPDATE"] = {"1", "CitcomS.solver.visc"};
  parameters["num_mat"] = {"4", "CitcomS.solver.visc",true};
  parameters["visc0"] = {"1,1,1,1", "CitcomS.solver.visc"};
  parameters["TDEPV"] = {"0", "CitcomS.solver.visc"};
  parameters["rheol"] = {"3", "CitcomS.solver.visc"};
  parameters["viscE"] = {"1,1,1,1", "CitcomS.solver.visc"};
  parameters["viscT"] = {"1,1,1,1", "CitcomS.solver.visc"};
  parameters["viscZ"] = {"1,1,1,1", "CitcomS.solver.visc"};
  parameters["SDEPV"] = {"0", "CitcomS.solver.visc"};
  parameters["sdepv_misfit"] = {"0.001", "CitcomS.solver.visc"};
  parameters["sdepv_expt"] = {"1,1,1,1", "CitcomS.solver.visc"};
  parameters["PDEPV"] = {"0", "CitcomS.solver.visc"};
  parameters["pdepv_a"] = {"1e20,1e20,1e20,1e20", "CitcomS.solver.visc"};
  parameters["pdepv_b"] = {"0,0,0,0", "CitcomS.solver.visc"};
  parameters["pdepv_y"] = {"1e20,1e20,1e20,1e20", "CitcomS.solver.visc"};
  parameters["pdepv_eff"] = {"1", "CitcomS.solver.visc"};
  parameters["pdepv_offset"] = {"0", "CitcomS.solver.visc"};
  parameters["CDEPV"] = {"0", "CitcomS.solver.visc"};
  parameters["cdepv_ff"] = {"1,1", "CitcomS.solver.visc"};
  parameters["low_visc_channel"] = {"0", "CitcomS.solver.visc"};
  parameters["low_visc_wedge"] = {"0", "CitcomS.solver.visc"};
  parameters["lv_min_radius"] = {"0.9764", "CitcomS.solver.visc"};
  parameters["lv_max_radius"] = {"0.9921", "CitcomS.solver.visc"};
  parameters["lv_channel_thickness"] = {"0.0047", "CitcomS.solver.visc"};
  parameters["lv_reduction"] = {"0.5", "CitcomS.solver.visc"};
  parameters["VMIN"] = {"0", "CitcomS.solver.visc"};
  parameters["visc_min"] = {"0.001", "CitcomS.solver.visc"};
  parameters["VMAX"] = {"0", "CitcomS.solver.visc"};
  parameters["visc_max"] = {"1000", "CitcomS.solver.visc"};
  parameters["z_layer"] = {"-999,-999,-999,-999", "CitcomS.solver.visc"};
  parameters["visc_layer_control"] = {"0", "CitcomS.solver.visc"};
  parameters["visc_layer_file"] = {"visc.dat", "CitcomS.solver.visc"};
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
      name.erase(std::remove_if(name.begin(), name.end(), ::isspace), 
		 name.end());
      value.erase(std::remove_if(value.begin(), value.end(), ::isspace), 
		  value.end());

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
    if(std::stoi(parameters["nproc_surf"].value) != 12)
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
    if(std::stoi(parameters["nproc_surf"].value) != 1)
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
    nprocx = std::stoi(parameters["nprocx"].value); 
    nprocy = std::stoi(parameters["nprocy"].value);
    nprocz = std::stoi(parameters["nprocz"].value);
    nodex = std::stoi(parameters["nodex"].value); 
    nodey = std::stoi(parameters["nodey"].value);
    nodez = std::stoi(parameters["nodez"].value);
    mgunitx = std::stoi(parameters["mgunitx"].value); 
    mgunity = std::stoi(parameters["mgunity"].value);
    mgunitz = std::stoi(parameters["mgunitz"].value);
    levels = std::stoi(parameters["levels"].value);

    if(nodex != (1+nprocx*mgunitx*pow2(levels-1)))
    {
      log_messages.push_back("# WARNING: changing mgunitx value to satisfy eqn 6.7");
      parameters["mgunitx"].value = 
	std::to_string((nodex - 1)/(nprocx*pow2(levels-1)));
      parameters["mgunitx"].isDefault = false;
    }
    if(nodey != (1+nprocy*mgunity*pow2(levels-1)))
    {
      log_messages.push_back("# WARNING: changing mgunity value to satisfy eqn 6.7");
      parameters["mgunity"].value = 
	std::to_string((nodey - 1)/(nprocy*pow2(levels-1)));
      parameters["mgunity"].isDefault = false;
    }
    if(nodez != (1+nprocz*mgunitz*pow2(levels-1)))
    {
      log_messages.push_back("# WARNING: changing mgunitz value to satisfy eqn 6.7");
      parameters["mgunitz"].value = 
	std::to_string((nodez - 1)/(nprocz*pow2(levels-1)));
      parameters["mgunitz"].isDefault = false;
    }
  }
}

void Py2CConverter::save()
{

  for(const auto& secname : {std::string("CitcomS"), 
	std::string("CitcomS.controller"), std::string("CitcomS.solver"),
	std::string("CitcomS.solver.mesher"), std::string("CitcomS.solver.tsolver"),
	std::string("CitcomS.solver.vsolver"), std::string("CitcomS.solver.bc"),
	std::string("CitcomS.solver.const"), std::string("CitcomS.solver.ic"),
	std::string("CitcomS.solver.output"), std::string("CitcomS.solver.param"),
	std::string("CitcomS.solver.phase"), std::string("CitcomS.solver.tracer"),
	std::string("CitcomS.solver.visc")})
  {

    // check if atleast one parameter in this section has been changed from its
    // default value
    if(section_names.count(secname) > 0)
    {
      fout << "# " << secname << std::endl;
      for(auto p = parameters.begin(); p != parameters.end(); ++p)
      {
	if((p->second.section == secname) && 
	   ((!p->second.isDefault)||(p->second.cRequired)))
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
      namevals.push_back(secname);
      for(auto p = parameters.begin(); p != parameters.end(); ++p)
      {
	if((p->second.section == secname) && 
	   ((p->second.cRequired)||(!p->second.isDefault)))
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
  for(const auto& msg : log_messages)
    fout << msg << std::endl;
}

void Py2CConverter::save_all()
{

  for(const auto& secname : {std::string("CitcomS"), 
	std::string("CitcomS.controller"), std::string("CitcomS.solver"),
	std::string("CitcomS.solver.mesher"), std::string("CitcomS.solver.tsolver"),
	std::string("CitcomS.solver.vsolver"), std::string("CitcomS.solver.bc"),
	std::string("CitcomS.solver.const"), std::string("CitcomS.solver.ic"),
	std::string("CitcomS.solver.output"), std::string("CitcomS.solver.param"),
	std::string("CitcomS.solver.phase"), std::string("CitcomS.solver.tracer"),
	std::string("CitcomS.solver.visc")})
  {

    fout << "# " << secname << std::endl;
    for(auto p = parameters.begin(); p != parameters.end(); ++p)
    {
      if((p->second.section == secname))
      {
	fout << p->first << "=" << p->second.value << std::endl;
      }
    }
    fout << std::endl;
  }
  // write out all the log messages at the end of the config file
  for(const auto& msg : log_messages)
    fout << msg << std::endl;
}
