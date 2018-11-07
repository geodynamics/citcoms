import os
import time
import sys
from threading import Thread
import weakref
import wx
from CitcomSHDFUgrid import CitcomSHDFUgrid

class CitcomSHdf2UGridThread(Thread):
    
   hdf2ugrid = None
   filename = ''
   current_timestep=0
   nx_redu = 0
   ny_redu = 0
   nz_redu = 0
   
   callback_function = None
   
   def __init__ (self):
       Thread.__init__(self)
       self.hdf2ugrid = CitcomSHDFUgrid()
  
   def set_citcomsreader(self,filename,current_timestep,nx_redu,ny_redu,nz_redu,callback_function):
       self.filename = filename
       self.current_timestep = current_timestep
       self.nx_redu = nx_redu
       self.ny_redu = ny_redu
       self.nz_redu = nz_redu
       self.callback_function = callback_function
       
   def get_ref(self):
       return weakref.ref(self.hdf2ugrid)
   
   def run(self):
       hexagrid = self.hdf2ugrid.initialize(self.filename,self.current_timestep,self.nx_redu,self.ny_redu,self.nz_redu)
       vtk_viscosity = self.hdf2ugrid.get_vtk_viscosity()
       vtk_temperature = self.hdf2ugrid.get_vtk_temperature()
       self.callback_function(hexagrid,vtk_viscosity,vtk_temperature)
       
       
class CitcomSProgressBar(Thread):
    
    hdf2ugrid = None
    
    def __init__(self):
        Thread.__init__(self)
        
    def set_ref(self,progress_ref):
        #ref=citmain.get_ref_progress()
        self.hdf2ugrid = progress_ref()
        
    def run(self):
        progress_old = 0
        progress = 0
        pd = wx.ProgressDialog("Opening file...","Cap %d from %d" % (0,11),11,parent=None,style=wx.PD_AUTO_HIDE|wx.PD_APP_MODAL)
        while progress != -1:
            progress = self.hdf2ugrid.progress
            if progress > progress_old:
                print progress
                pd.Update(progress,"Cap %d of %d" % (progress,11))
                progress_old = progress
            time.sleep(0.5)
