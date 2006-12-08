from enthought.envisage.workbench.action.action_plugin_definition import \
     Action, Group, Location, Menu, WorkbenchActionSet

ID = "citcoms_plugins"

open_citcoms_hdf = Action(
    id          = ID + ".plugins.OpenCitcomSFILES.OpenCitcomSHDFFILE",
    class_name  = ID + ".plugins.OpenCitcomSFILES.OpenCitcomSHDFFILE",
    name        = "CitcomS &HDF5 file",
    tooltip     = "Open a CitcomS HDF5 data file",
    description = "Open a CitcomS HDF5 data file",
    locations   = [Location(path="MenuBar/FileMenu/OpenMenu/additions"),]
)

citcoms_cap_filter = Action(
    id          = ID + ".plugins.filter.CitcomSFilterActions.CitcomSshowCaps",
    class_name  = ID + ".plugins.filter.CitcomSFilterActions.CitcomSshowCaps",
    name        = "CitcomS &ShowCaps",
    tooltip     = "Display a specified range of caps",
    description = "Display a specified range of caps",
    locations   = [Location(path="MenuBar/VisualizeMenu/FiltersMenu/additions"),]
)

citcoms_reduce_filter = Action(
    id          = ID + ".plugins.filter.CitcomSFilterActions.CitcomSreduce",
    class_name  = ID + ".plugins.filter.CitcomSFilterActions.CitcomSreduce",
    name        = "CitcomS &Reduce",
    tooltip     = "Display a Reduce Grid for interpolation",
    description = "Display a Reduce Grid for interpolation",
    locations   = [Location(path="MenuBar/VisualizeMenu/FiltersMenu/additions"),]
)

action_set = WorkbenchActionSet(
    id      = ID + '.action_set',
    name    = 'CitcomsActionSet',
    actions = [open_citcoms_hdf, citcoms_cap_filter, citcoms_reduce_filter,]
)

###############################################################################

requires = []
extensions = [action_set]


