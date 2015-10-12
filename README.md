A multi-camera tracking algorithm using OpenCV. It depends on OpenCv 2.3.1 (http://opencv.org/), Boost libraries (http://www.boost.org/) and Intel IPP (https://software.intel.com/en-us/intel-ipp).

Usage
------------

In order to run the tracker, use the following command:

### Windows ###

MultiCameraTracker.exe -d config.cfg

(or)

MultiCameraTracker.bat

### Linux ###

./MultiCameraTracker -d config.cfg 

(or)

./MultiCameraTracker.sh

Compilation
------------
Windows: Build the Visual studio project.

Linux: Run the make file inside the "obj" folder.

Note: Make sure Intel IPP, OpenCv 2.3.1 and Boost libraries are located in appropriate folders.

Data
------------
Here is the sample data folder: https://www.dropbox.com/s/gnheat18lotoi2k/data.tar.gz?dl=0

Note: Follow a similar folder structure for new scenarios.

### Contact ###
[1] Santhoshkumar Sunderrajan( santhosh@ece.ucsb.edu)
Website: http://vision.ece.ucsb.edu/~santhosh/

### Bibtex ###
If you use the code in any of your research works, please cite the following papers:
~~~
@inproceedings{ni2010particle,
  title={Particle filter tracking with online multiple instance learning},
  author={Ni, Zefeng and Sunderrajan, Santhoshkumar and Rahimi, Amir and Manjunath, BS},
  booktitle={Pattern Recognition (ICPR), 2010 20th International Conference on},
  pages={2616--2619},
  year={2010},
  organization={IEEE}
}

@inproceedings{ni2010distributed,
  title={Distributed particle filter tracking with online multiple instance learning in a camera sensor network},
  author={Ni, Zefeng and Sunderrajan, Santhoshkumar and Rahimi, Amir and Manjunath, BS},
  booktitle={Image Processing (ICIP), 2010 17th IEEE International Conference on},
  pages={37--40},
  year={2010},
  organization={IEEE}
}
~~~

### Disclaimer ###
I may have used some good codes from various sources, please feel free to notify me if you find a piece of code that I need to acknowledge.
