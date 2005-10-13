<?xml version="1.0"?>
<!--
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!
!                             Michael A.G. Aivazis
!                      California Institute of Technology
!                      (C) 1998-2005  All Rights Reserved
!
! {LicenseText}
!
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-->


<!DOCTYPE inventory>

<inventory>

  <component name='journald-harness'>
    <property name='home'>../config</property>

    <component name='journal'>
        <property name='device'>file</property>

        <component name='file'>
          <property name='name'>../log/journal.log</property>
        </component>

    </component>

  </component>

</inventory>


<!-- version-->
<!-- $Id: journald-harness.pml,v 1.1.1.1 2005/03/14 06:15:28 aivazis Exp $-->

<!-- End of file -->
