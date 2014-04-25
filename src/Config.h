#ifndef CONFIG_H
#define CONFIG_H
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include <iostream>
#include <fstream>

namespace MultipleCameraTracking
{
    #define STRING_SIZE                    1000
    #define MAX_ITEMS_TO_PARSE            10000
    #define MAX_NUMBER_OF_VIEWS            100
    #define DEFAULTCONFIGFILENAME        "config.cfg"
    
    //#define ENABLE_PARTICLE_LOG  //un-define to disable the logging of particle information

    #if defined(WIN32) || defined(WIN64)
    #pragma warning(disable : 4996)
    #endif

    #if defined(WIN32) || defined(WIN64)
    #define strcasecmp  strcmpi
    #endif

    typedef struct 
    {
    /*********** Input Information**************/
        int        m_numOfFrames;                            // Number of frames to track
        int        m_startFrameIndex;                        // Starting frame index
        char    m_cameraSetCstr[STRING_SIZE];            // The set of Cameras, e.g., "1,3,4"
        char    m_objectSetCstr[STRING_SIZE];            // The set of object (ID) being tracked, e.g. "1,2"
        char    m_inputDirectoryNameCstr[STRING_SIZE];    // Root directory address;
        char    m_dataFilesNameCstr[STRING_SIZE];        // Directory name to hold data files
        char    m_intializationDirectoryCstr[STRING_SIZE];// The local directory (inside the experiment directory)
                                                          //  to hold the initialization files;     
        int        m_loadVideoWithColor;    //  [0-default]: Load image as gray scale image; [1]: load as RGB color image
        bool    m_interactiveModeEnabled;//  [0-default]: pause after each frame.
        int        m_loadVideoFromImgs;    //    [0]: Load video as image sequences; [1-default]: load from a video file
        
    /*********** Output Information**************/        
        char    m_outputDirectoryNameCstr[STRING_SIZE];
        int        m_detailedLogging;                // [1-default]: Enable detailed log; [0]: disable
        int        m_verboseMode;                    // [1-default]: Enable verbose mode; [0]: disable
        int        m_saveOutputVideo;                // [0-default]: No; [1]: Yes; Save output video (with tracked blobs)
        int        m_displayOutputVideo;            // [0-default]: No; [1]: Yes; Display output video (with tracked blobs)
        int        m_displayTrainingSamples;        // [0-default]: No; [1]: Yes
        int        m_saveTrainingSamplesVideo;        // [0-default]: No; [1]: Yes
        int        m_displayTrainingExampCenterOnly;//[0-default]: No; [1]: Yes; Display only the center of the training Samples        
        int        m_waitBeforeFinishTracking;        // [1]: Hold the process before exit at the end of tracking; [0-default]:No.
        int        m_calculateTrackingError;        // [1]: Calculate the tracking error from groundTruth; [0-default]:No.
        int        m_trialNumber;                    // Trial number for this particular experiment

    /*********** Feature and classifier**************/        
        int        m_trackerFeatureType;            // [1-default]: Haar; [2]: 11-dim Culture Color; [3]: 512-dim MultiDimensional color 
        int        m_useHSVColor;                    // use HSV color instead of RGB for MultiDimensional color
        int        m_numofBinsColor;                // number of bins for each dimension of the color histogram
        int        m_trackerFeatureParameter;        // Number of Haar features if Tracker_Feature_Type = 1; Otherwise: ignored
        int        m_trackerStrongClassifierType;    // [1-default]: MilBoost; [2]: AdaBoost; [3]: MilEnsemble; [4]: MilBoost with AnyBoost;
        int        m_trackerWeakClassifierType;    // For MilBoost/AdaBoost, [1-default]: STUMP; [2]: Weighted STUMP; [3]: Perceptron
                                                // For MilEnsemble, this parameter is ignored, as only Perceptron is allowed
        int        m_percentageOfWeakClassifiersSelected;// percentage of weak classifier selected during boosting
        int        m_percentageOfWeakClassifiersRetained;// percentage of weak classifier kept from previous frame for Ensemble

    /*********** Tracker setting **************/        
        int        m_localTrackerType;    // [0-default]: Simple Tracker; [1]: Particle Filter Tracker
        int        m_posRadiusTrain;    // Default = 4; Value for SimpleTrackerParameters.m_posRadiusTrain, used for the selection 
                                    // of positive  training example when applicable; Typical value: 4 pixel for Mil, 
                                    // 1 to 4 for AdaBoost (as it is less robust)
        int        m_initPosRadiusTrain; //Default = 3; radius (Number of pixels); 
                                    // All samples within the radius to select positive training examples at the beginning
                                    // Negative examples  inside 2*Search_Window_Size and outside 1.5*m_initPosRadiusTrain

        int        m_numNegExamples;    // Number of negative examples for training 
        int        m_initNumNegExampes;// Initial number of negative examples for training
        int        m_searchWindowSize; // Applicable to simple tracker only
                                    // Also used by particle filter during initialization
        int        m_negSampleStrategy;// [0]: all over image; [1 - default] close to the search window        

    /***********  Particle filter tracker parameters **************/        
        int        m_numOfParticles;            // Number of particles
        double    m_PFTrackerStdDevX;            // Value for ParticleFilterTrackerParameters.m_standardDeviationX in pixels
        double    m_PFTrackerStdDevY;            // Value for ParticleFilterTrackerParameters.m_standardDeviationY in pixels
        double  m_PFTrackerStdDevScaleX;    // Value for ParticleFilterTrackerParameters.m_standardDeviationScaleX
        double  m_PFTrackerStdDevScaleY;    // Value for ParticleFilterTrackerParameters.m_standardDeviationScaleY    
        int        m_PFTrackerNumDispParticles;// Number of particle blobs displayed on the video if applicable
        int        m_PfTrackerMaxNumPositiveExamples;//  Maximum number of positive examples generative(typically less than the number of Particles) 
                                                // Actual number could be different depends on the sampling strategy.

        int        m_PFOutputTrajectoryOption;    // [0]: The particle of highest weight; [1-default]: average of all particles
        int        m_PfTrackerPositiveExampleStrategy;    // [0-default]: Generate examples in the same way as simpleTracker based on the particle average 
                                                    //        (or the particle of highest weight if PfTracker_Output_Trajectory_Option = 0)
                                                    // [1]: Use all unique particles (weight descending order) until reaching maximum
                                                    // [2]: Same as [1]. Except when When there are not enough unique particles, 
                                                    //        pick all unique particles, and sample the rest similar to [0]
                                                    // [3]: Random sampling from the re-sampled particles regardless of same sample

        int        m_PfTrackerNegativeExampleStrategy;    // [0-default]: Generate examples in the same way as simpleTracker based on the particle average 
                                                    //     (or the particle of highest weight if PfTracker_Output_Trajectory_Option = 0)
                                                    //  That is sampled from inside 1.5f*m_searchWindSize and outside (5 + m_posRadiusTrain)
                                                    //     or all over image if Negative_Sampling_Strategy = 0 (not recommended);    
                                                    //     All samples' scale decides by the average scale of all particles
                                                    // [1]: Classifier::Sample between two circles: outside circle radium 2*Search_Window_Size, 
                                                    //     inside circle radius decided by particles.

    /*********** Fusion setting **************/        
        int        m_geometricFusionType;                // [0-default]: No Ground Fusion; 1: Ground plane Fusion with GMM and Particle Reweighting
        int        m_saveGroundParticlesImage;            // [0-default]: Do not save the Ground plane Particles to a video; [1]: Yes
        int        m_displayGroundParticlesImage;        // [0-default]: Do not Display the Ground plane Particles; [1]: Yes

        int        m_saveGroundPlaneKFImage;            // [0-default]: Do not save the KF posterior distribution to a video; [1]: Yes.
        int        m_displayGroundPlaneKFImage;        // [0-default]: Do not display the KF posterior distribution; [1]: Yes.
        int        m_displayGMMCenters;                // [0-default]: Do not Display the GMM centers from plane Particles; [1]: Yes

        int        m_appearanceFusionType;                // [0-default]: No Appearance Fusion; [1]: Culture Color; [2]: MultiDimensional color 
        int        m_appearanceFusionStrongClassifierType;//[1-default]: MilBoost; [2] = AdaBoost; [3] = MIL_Ensemble
        int        m_appearanceFusionWeakClassifierType;    // For MilBoost/AdaBoost, [1-default]: STUMP; [2]: Weighted STUMP; [3]: Perceptron
                                                        // For MilEnsemble, this parameter is ignored, as only Perceptron is allowed
        int        m_AFNumberOfPositiveExamples;            // Number of positive examples for AF
        int        m_AFNumberOfNegativeExamples;            // Number of negative examples for AF
        int     m_AFRefreshRate;                        // How often to perform Appearance fusion (e.g. every 10 frames?)

        int        m_AFpercentageOfWeakClassifiersSelected;// Percentage of weak classifiers selected from the available weak classifier pool. 
        int        m_AFpercentageOfWeakClassifiersRetained; // Applicable for  MILEnsemble, should be lesser than Percentage_Of_Weak_Classifiers_Selected

        int        m_enableCrossCameraAutoInitialization;    // [0-default]: No;  1: Yes
        int        m_enableCrossCameraOcclusionHandling;    // [0-default]: No;  1: Yes    
    } InputParameters;

    typedef struct 
    {
        char    *TokenName;
        void    *Place;
        int        Type;
        double    Default;
        int        param_limits; //! 0: no limits, 1: both min and max, 2: only min (i.e. no negatives), 3: for special cases (not defined yet)
        double    min_limit;
        double    max_limit;
    } Mapping;

    extern InputParameters    g_configInput;
    extern std::ofstream    g_logFile;
    extern bool                g_detailedLog;
    extern bool                g_verboseMode;
    
    extern Mapping            Map[];

    int  Configure ( int ac, char *av[] );

    void PatchInputNoFrames( );
    int DisplayAndLogParams( );
}
#endif