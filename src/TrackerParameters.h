#ifndef TRACKER_PARAMETERS
#define TRACKER_PARAMETERS

#include "Public.h"
#include "Feature.h"
#include "ClassifierParameters.h"

namespace MultipleCameraTracking
{
    enum GeometricFusionType    
    {    
        NO_GEOMETRIC_FUSION                =    0, 
        FUSION_GROUND_PLANE_GMM_KF        =    1,
    };

    enum AppearanceFusionType    
    {
        NO_APPEARANCE_FUSION                        =    0,
        FUSION_CULTURE_COLOR_HISTOGRAM                =    1,
        FUSION_MULTI_DIMENSIONAL_COLOR_HISTOGRAM    =    2,
    };

    enum AppearanceFusionStrongClassifierType
    {
        APP_FUSION_MIL_BOOST    = 1, //    MILBOost
        APP_FUSION_ADA_BOOST    = 2, // AdaBoost
        APP_FUSION_MIL_ENSEMBLE = 3, // MILEnsemble
    };

    enum TrackerType
    {    
        SIMPLE_TRACKER = 0,            // Simple Tracker
        PARTICLE_FILTER_TRACKER = 1 
    };    

    enum PFTracker_TrajectoryOption
    {
        PARTICLE_HIGHEST_WEIGHT = 0,
        PARTICLE_AVERAGE        = 1
    };

    enum PFTracker_Positive_Sample_Strategy
    {
        SAMPLE_POS_SIMPLETRACKER    = 0,    //Generate examples in the same way as simpleTracker based on the particle average 
                                            //    (or the particle of highest weight if PfTracker_Output_Trajectory_Option = 0)
        SAMPLE_POS_GREEDY            = 1,    // Put all unique particles (with maximum defined by PfTracker_Num_Positive_Examples)
        SAMPLE_POS_SEMI_GREEDY        = 2,    // If there are enough unique particles, pick the particles with top weight;
                                            //       Otherwise, pick all unique particles, and sample the rest similar to [0]
        
        SAMPLE_POS_PARTICLE_RANDOM    = 3        // Random sampling from the re-sampled particles regardless of uniqueness
    };

    enum PFTracker_Negative_Sample_Strategy
    {
        SAMPLE_NEG_SIMPLETRACKER    = 0,    // Generate examples in the same way as simpleTracker based on the particle average 
                                            //     (or the particle of highest weight if PfTracker_Output_Trajectory_Option = 0)
                                            //   That is inside 2*Search_Window_Size and outside 1.5*Inner_Radius_For_Positive_Examples
                                            //     or all over image if Negative_Sampling_Strategy = 0 (not recommended);    
                                            //     All samples' scale decides by the average scale of all particles
        SAMPLE_NEG_PARTICLE_RANDOM    = 1        // Classifier::Sample between two circles: outside circle radium 2*Search_Window_Size, 
                                            //     inside circle radius decided by particles.
    };

    //Forward Declaration
    class CameraTrackingParameters;
    class TrackerParameters;
    class SimpleTrackerParameters;
    class ParticleFilterTrackerParameters;

    //declarations of shared ptr
    typedef boost::shared_ptr<CameraTrackingParameters>            CameraTrackingParametersPtr;

    typedef boost::shared_ptr<TrackerParameters>                TrackerParametersPtr;
    typedef boost::shared_ptr<SimpleTrackerParameters>            SimpleTrackerParametersPtr;
    typedef boost::shared_ptr<ParticleFilterTrackerParameters>    ParticleFilterTrackerParametersPtr;
    

    /****************************************************************
    CameraTrackingParameters
        Stores the tracking parameters for each camera
    ****************************************************************/
    class CameraTrackingParameters
    {
    public:
        CameraTrackingParameters( InputParameters *                                pInputConfigeration,                    
                                const TrackerType                                trackerType                            = SIMPLE_TRACKER,
                                const Classifier::StrongClassifierType            localTrackerStrongClassifierType    = Classifier::ONLINE_STOCHASTIC_BOOST_MIL,
                                const Classifier::WeakClassifierType            localTrackerWeakClassifierType        = Classifier::STUMP,
                                const Features::FeatureType                        trackerFeatureType                    = Features::HAAR_LIKE,        
                                const GeometricFusionType                        geometricFusionType                    = NO_GEOMETRIC_FUSION,
                                const int                                        numberOfParticles                    = 1,
                                const AppearanceFusionType                        appearanceFusionType                = NO_APPEARANCE_FUSION,
                                const AppearanceFusionStrongClassifierType        appearanceFusionStrongClassifierType= APP_FUSION_MIL_BOOST,
                                const Classifier::WeakClassifierType            appearanceFusionWeakClassifierType    = Classifier::STUMP
                                );                    

        //tracking settings 
        const int                                        m_trialNumber;                // which trial for the experiments (for random initialization and output naming )
        const TrackerType                                m_localObjectTrackerType;    // What kind of local object tracker, e.g. simple tracker, particle filter etc.                            
        
        //Feature settings
        const Features::FeatureType                        m_trackerFeatureType;        // Feature type used by the local object tracker
        
        //classifier settings
        const Classifier::StrongClassifierType            m_localTrackerStrongClassifierType;        // strong Classifier for the local tracker
        const Classifier::WeakClassifierType            m_localTrackerWeakClassifierType;        // weak classifier for the local tracker

        //fusion setting
        const AppearanceFusionType                        m_appearanceFusionType;                    // Appearance fusion type
        const AppearanceFusionStrongClassifierType        m_appearanceFusionStrongClassifierType;    // strong classifier for the appearance fusion 
        const Classifier::WeakClassifierType            m_appearanceFusionWeakClassifierType;   // weak classifier for the appearance fusion
        const GeometricFusionType                        m_geometricFusionType;                    // Geometric fusion type 
        const bool                                        m_saveGroundParticlesImage;                // save the Ground location to ground image
        const bool                                        m_saveGroundPlaneKFImage;                // save the KF posterior distribution
        const bool                                        m_displayGroundParticlesImage;            // display the Ground location to ground image
        const bool                                        m_displayGroundPlaneKFImage;            // display the KF posterior distribution
        const bool                                        m_displayGMMCenters;                    // display the centers of the GMM;
        
        const int                                        m_numberOfParticles;        // number of particles (needed by the geometric fusion)
        
        const bool                                        m_isCrossCameraAutoInitializationEnabled;    //enable auto initialization with help from other camera    
        const bool                                        m_isCrossCameraOcclusionHandlingEnabled;    //enable multi-camera multi-object occlusion handling         

        const bool                                        m_HSVRequired;                // HSV image is required by tracker, camera needs to do the conversion for Object/Tracker Class.

        const string                                    m_outputDirectoryString;            // output directory to save any results
        const string                                    m_inputDirectoryString;                // input directory 
        const string                                    m_dataNameString;                    //data files Name
        const string                                    m_nameInitilizationString;            //initialization folder name
        
        const bool                                        m_calculateTrackingError;                            // calculate the tracking error from ground truth file

        const int                                        m_numberOfBinsForColorHistogram;
        const bool                                        m_useHSVColorSpaceForColorHistogram;                // use HSV instead of RGB
        const int                                        m_appearanceFusionNumberOfPositiveExamples;            // Number of positive examples for AF
        const int                                        m_appearanceFusionNumberOfNegativeExamples;            // Number of negative examples for AF
        const int                                        m_appearanceFusionRefreshRate;                        // How often to perform Appearance fusion (e.g. every 10 frames?)

    };

    /****************************************************************
    TrackerParameters
        Stores the tracker parameters for the tracker (associated with each object)
    ****************************************************************/
    class TrackerParameters
    {
    public:
        TrackerParameters( );

        vectori            m_outputBoxColor;                        // for outputting video
        uint            m_outputBoxLineWidth;                    // line width 
        uint            m_numberOfNegativeTrainingSamples,m_init_negNumTrain;        // # negative samples to use during training, and init
        float            m_posRadiusTrain,m_init_posTrainRadius; // radius for gathering positive instances
        uint            m_maximumNumberOfPositiveTrainingSamples;                // max # of pos to train with
        bool            m_debugv;                        // displays response map during tracking [kinda slow, but help in debugging]
        vectorf            m_initState;                    // [x,y,size_x,size_y,frameInd] 
        bool            m_shouldNotUseSigmoid;            // Do not calculate probabilities with Sigmoid for strong classifiers (tends to work much better)
        bool            m_initializeWithFaceDetection;                    // initialize with the OpenCV tracker rather than m_initState
        bool            m_displayTrainingSampleCenterOnly;// display training examples with their centers only
        string            m_displayFigureNameStr;    // Name of figure (window) to display video
        string            m_trajSave;                // filename - save file containing the coordinates of the box (text file with [x y width height] per row)
        bool            m_isColor;                // load as color images
};

    /****************************************************************
    SimpleTrackerParameters
        Stores the simple tracker parameters.
    ****************************************************************/
    class SimpleTrackerParameters : public TrackerParameters
    {
    public:
        SimpleTrackerParameters( );

        uint            m_searchWindSize;        // size of search window
        uint            m_negSampleStrategy;    // [0] all over image [1 - default] close to the search window
    };

    /****************************************************************
    ParticleFilterTrackerParameters
        Particle Filter Tracker Parameters
    ****************************************************************/
    class ParticleFilterTrackerParameters : public SimpleTrackerParameters
    {
    public:
        ParticleFilterTrackerParameters( )    { };
        ~ParticleFilterTrackerParameters( ) { };

        int                m_numOfDisplayedParticles;    //number of particles to be displayed
        PFTracker_TrajectoryOption    m_outputTrajectoryOption;            
        int             m_numberOfParticles;                //number of particles
        float             m_standardDeviationX;     //Standard deviation for x
        float            m_standardDeviationY;    //Standard deviation for y
        float            m_standardDeviationScaleX;    //Standard deviation for scale x
        float            m_standardDeviationScaleY;    //Standard deviation for scale y
        PFTracker_Positive_Sample_Strategy      m_positiveSampleStrategy; // how to generate positive training examples 
        PFTracker_Negative_Sample_Strategy        m_negativeSampleStrategy; // how to generate negative training examples 
        int             m_maxNumPositiveExamples;            // Number of Positive training samples
    };
}
#endif
