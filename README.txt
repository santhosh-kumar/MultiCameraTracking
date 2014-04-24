Multiple Camera Tracking with Online Learning (MilBoost[1]/AdaBoost[2])

-------------------------------------------------------------------------------------------------------
This code support multiple object tracking in a multiple camera setup. 
Single object visual tracking is based on the implementation of the MilTrack algorithm [1], as well as an implementation of the Online AdaBoost tracker described in [2] (though with some modifications).  This code requires both OpenCV and Intel IPP to be installed.  It has only been developed on a machine running Windows 7, using Visual Studio 2008.  In order for the code to run, make sure you have the OpenCV bin directory, and the Intel IPP bin directory in your system path.

Copyright 2011, VRL UCSB.  Distributed under the terms of the GNU Lesser General Public License (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

Project Website: 
References:
[1] Visual Tracking with Online Multiple Instance Learning
    B. Babenko, M.H. Yang, S. Belongie
    CVPR 2009

[2] On-line Boosting and vision
    H. Grabner and H. Bischof
    CVPR 2006
    
[3] Distributed Particle Filter Tracking With Online Multiple Instance Learning In a Camera Sensor Network.
    Z.Ni, S.Sunderrajan, A.Rahimi and B.S.Manjunath
    ICIP 2010
    
[4] Particle Filter Tracking With Online Multiple Instance Learning
    Z.Ni, S.Sunderrajan, A.Rahimi and B.S.Manjunath
    ICIP 2010
    
Contact:

[1] Santhoshkumar Sunderrajan( santhoshkumar@umail.ucsb.edu )
[2] Zefeng Ni ( zefeng@ece.ucsb.edu )

-------------------------------------------------------------------------------------------------------
Usage: 
MultipleCameraTracking -d MultiCameraTracking.cfg 
(requiring all shared library, including opencv and intel IPP set up properly)

RUNNING MULTIPLE CAMERA TRACKING with all setting defined in the file "MultiCameraTracking.cfg"
Video(Image) files:
The video source can be a sequence of images: e.g. experiment/imgxxx - these sub-directories should contain the images of the video clips; they should be names img00000.png, img00001.png, etc. E.g. img001 for camera 1. Alternatively, experiment/videoxxx.avi -- original video files for each camera xxx. See the sample configuration file for details.
 
Initialization files: 
experiment/initialization/experiment_gtxxx.txt - this is the ground truth file for camera xxx;  every line in this file corresponds to a frame of the sequence, and contains the [x,y,width,height,frameid, objectid, framepos] information.  
    
-------------------------------------------------------------------------------------------------------
COMPILING CODE
Windows:

Linux: two make files inside "obj": Makefile, and MakefileD (for debug).
