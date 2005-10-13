<?xml version="1.0"?>
<!--
 ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 !
 !                               Michael A.G. Aivazis
 !                        California Institute of Technology
 !                        (C) 1998-2005 All Rights Reserved
 !
 ! <LicenseText>
 !
 ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-->


<!DOCTYPE pummel>

<blade>

  <application name="Blade"/>

    <window id="window.main" title="Sample window">
      <box style="vertical">
        <menubar id="menubar.main">
          <extend id="menubar.main" src="menu-main.pml"/>
        </menubar>
        <toolbar id="toolbar.edit">
          <extend src="toolbar-standard.pml" id="toolbar.edit"/>
        </toolbar>
        <box style="horizontal">
          <button label="button 1"/>
          <button label="button 2"/>
        </box>
      </box>
    </window>
  
</blade>


<!-- $Id: blade.pml,v 1.1.1.1 2005/03/08 16:13:57 aivazis Exp $ -->

<!-- End of file --> 
