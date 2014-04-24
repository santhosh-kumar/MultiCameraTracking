#ifndef TRACKER_PUBLIC
#define TRACKER_PUBLIC

#include "StrongClassifierBase.h"
#include "Public.h"
#include "TrackerParameters.h"
#include "Config.h"

namespace MultipleCameraTracking
{
    //Forward Declaration
    class Tracker;
        
    //declarations of shared ptr
    typedef boost::shared_ptr<Tracker>                    TrackerPtr;
    
    /****************************************************************
    Tracker
        Base class for all the trackers.
    ****************************************************************/
    class Tracker
    {
    public:        
        static bool        InitializeWithFace( TrackerParameters* params, Matrixu& frame );
        static void        ReplayTracker( vector<Matrixu>& vid, string states, string outputvid = "", uint R = 255, uint G = 0, uint B = 0 );
        static void        ReplayTrackers( vector<Matrixu>& vid, vector<string> statesfile, string outputvid, Matrixu colors );
                
        // Initializes tracking  with video information
        virtual void    InitializeTrackerWithParameters( Matrixu*                pFrameImageColor, 
                                                         Matrixu*                pFrameImageGray, 
                                                         int                    frameInd,
                                                         uint                    videoLength, 
                                                         TrackerParametersPtr    trackerParametersPtr,
                                                         Classifier::StrongClassifierParametersBasePtr    clfparamsPtr,
                                                         Matrixu*                pFrameDisplay            = NULL, 
                                                         Matrixu*                pFrameDisplayTraining    = NULL,
                                                         Matrixu*                pFrameImageHSV            = NULL,
                                                         Matrixf*                pGroundTruthMatrix        = NULL ) = 0;  

        // Track each frame, and update pFrameDisplay and pFrameDisplayTraining (if required)
        virtual void    TrackObjectAndSaveState( int        frameind, 
                                                  Matrixu*    pFrameImageColor, 
                                                 Matrixu*    pFrameImageGray, 
                                                 Matrixu*    pFrameDisplay            = NULL, 
                                                 Matrixu*    pFrameDisplayTraining    = NULL,
                                                 Matrixu*    pFrameImageHSV            = NULL ) = 0;
        // Saves the states to file after finishing tracking the object
        virtual void    SaveStates( ) = 0;
        virtual    void    CalculateTrackingErrroFromGroundTruth( ) = 0;
        virtual void    GenerateTrainingSampleSet(  Matrixu*    pFrameImageColor, 
                                                    Matrixu*    pFrameImageGray, 
                                                    Matrixu*    pFrameImageHSV = NULL )  = 0;

        virtual void    GenerateTestSampleSet(    Matrixu* pFrameImageColor,
                                                Matrixu* pFrameImageGray,                                                         
                                                Matrixu* pFrameImageHSV ) = 0;

        virtual void    GetTrainingSampleSets( Classifier::SampleSet& positiveSampleSet, Classifier::SampleSet& negativeSampleSet )  = 0;
        virtual void    DisplayTrainingSamples( Matrixu* pFrameDisplayTraining )  = 0;
        virtual void    UpdateClassifier(    Matrixu*    pFrameImageColor, 
                                            Matrixu*    pFrameImageGray, 
                                            Matrixu*    pFrameDisplayTraining = NULL,
                                            Matrixu*    pFrameImageHSV    = NULL ) = 0;

        virtual const    vectorf&    GetCurrentTrackerState( ) const  = 0;

        virtual void    DrawObjectFootPosition( Matrixu* pFrameDisplay ) const = 0;

    protected:

        virtual void    GeneratePositiveTrainingSampleSet(    Matrixu*    pFrameImageColor, 
                                                            Matrixu*    pFrameImageGray, 
                                                            Matrixu*    pFrameImageHSV = NULL )  = 0;

        virtual void    GenerateNegativeTrainingSampleSet(    Matrixu*    pFrameImageColor, 
                                                            Matrixu*    pFrameImageGray, 
                                                            Matrixu*    pFrameImageHSV = NULL )  = 0;

        static CvHaarClassifierCascade*        s_faceCascade;
        CameraTrackingParametersPtr            m_cameraTrackingParametersPtr;        
        Matrixf                                m_groundTruthMatrix;
    };
}
#endif