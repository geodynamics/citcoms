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

  <component name='ipa'>

    <property name='port'>50001</property>
    <property name='timeout'>10*second</property>
    <property name='ticketDuration'>20*second</property>

    <component name='userManager'>
      <property name='passwd'>userdb.md5</property>
    </component>
  </component>

</inventory>


<!-- version-->
<!-- $Id: ipa.pml,v 1.2 2005/04/24 20:02:29 pyre Exp $-->

<!-- End of file -->
