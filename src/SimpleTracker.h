#ifndef SIMPLE_TRACKER_PUBLIC
#define SIMPLE_TRACKER_PUBLIC

#include "Tracker.h"

namespace MultipleCameraTracking
{
    //Forward Declaration
    class SimpleTracker;

    //declarations of shared ptr
    typedef boost::shared_ptr<SimpleTracker>            SimpleTrackerPtr;

    /****************************************************************
    SimpleTracker
        Derives from Tracker.
    ****************************************************************/
    class SimpleTracker : public Tracker
    {
    public:
        // Constructor
        SimpleTracker( CvMat* pHomographyMatrix  )
            : m_pHomographyMatrix( pHomographyMatrix )
        { 
            s_faceCascade = NULL;
        }

        // Virtual destructor
        virtual ~SimpleTracker( )
        { 
        }

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
                                                         Matrixf*                pGroundTruthMatrix        = NULL ) ;  
                                                    
        // Track each frame, store the results to m_states, and update pFrameDisplay and pFrameDisplayTraining (if required)
        virtual void    TrackObjectAndSaveState( int        frameind, 
                                                  Matrixu*    pFrameImageColor, 
                                                 Matrixu*    pFrameImageGray, 
                                                 Matrixu*    pFrameDisplay            = NULL, 
                                                 Matrixu*    pFrameDisplayTraining    = NULL,
                                                 Matrixu*    pFrameImageHSV            = NULL );

        // Saves the states to file after finishing tracking the object
        virtual void    SaveStates( );
        virtual    void    CalculateTrackingErrroFromGroundTruth( );

        virtual const    vectorf&    GetCurrentTrackerState( ) const { return m_currentStateList; }

        virtual void    GenerateTrainingSampleSet(  Matrixu*    pFrameImageColor, 
                                                    Matrixu*    pFrameImageGray, 
                                                    Matrixu*    pFrameImageHSV = NULL ) ;

        virtual void    GenerateTestSampleSet(    Matrixu* pFrameImageColor,
                                                Matrixu* pFrameImageGray,                                                         
                                                Matrixu* pFrameImageHSV );

        virtual    void    GetTrainingSampleSets( Classifier::SampleSet& positiveSampleSet, Classifier::SampleSet& negativeSampleSet );
        virtual void    DisplayTrainingSamples( Matrixu* pFrameDisplayTraining );

        // With SimpleTracker, classifier is automatically updated within TrackObjectOnTheGivenFrame
        virtual void    UpdateClassifier(    Matrixu*    pFrameImageColor, 
                                            Matrixu*    pFrameImageGray, 
                                            Matrixu*    pFrameDisplayTraining = NULL,
                                            Matrixu*    pFrameImageHSV    = NULL );

        CvMat* GetHomographyMatrix( ){ return m_pHomographyMatrix; }
    
        virtual void    DrawObjectFootPosition( Matrixu* pFrameDisplay ) const { return; };

    protected:
        // Initializes tracker with first frame(s) and other parameters
        bool            InitializeTracker(  Matrixu*                pFrameImageColor, 
                                            Matrixu*                pFrameImageGray,  
                                            TrackerParametersPtr    trackerParametersPtr, 
                                            Classifier::StrongClassifierParametersBasePtr clfparamsPtr,
                                            Matrixu* pFrameDisplay            = NULL, 
                                            Matrixu* pFrameDisplayTraining    = NULL,
                                            Matrixu* pFrameImageHSV            = NULL );

        // Track object in a frame; 
        // Classifier is automatically updated and pFrameDisplayTraining is updated if given 
        virtual double  TrackObjectOnTheGivenFrame( Matrixu*    pFrameImageColor, 
                                                    Matrixu*    pFrameImageGray,
                                                    Matrixu*    pFrameDisplayTraining= NULL,
                                                    Matrixu*    pFrameImageHSV        = NULL );
        
        Classifier::StrongClassifierBasePtr            m_strongClassifierBasePtr;
        vectorf                                        m_currentStateList; //[leftX, topY, sizeX, sizeY, scaleX, scaleY]
        SimpleTrackerParametersPtr                    m_simpleTrackerParamsPtr;
        
        Matrixf                                        m_states;                        //saves entire track history
        Classifier::SampleSet                        m_positiveSampleSet;            //positive samples
        Classifier::SampleSet                        m_negativeSampleSet;            //negative samples
        Classifier::SampleSet                        m_testSampleSet;                //detect samples
        vectorf                                        m_liklihoodProbabilityList;        //probability vector
        bool                                        m_isInitialized;                //is the tracker initialized
        bool                                        m_shouldPreComputeHSV;            //save temporal copy of HSV image for each frame to be tracked
        
        CvMat*                                        m_pHomographyMatrix;
    private:

        virtual void    GeneratePositiveTrainingSampleSet(    Matrixu*    pFrameImageColor, 
                                                            Matrixu*    pFrameImageGray, 
                                                            Matrixu*    pFrameImageHSV = NULL );

        virtual void    GenerateNegativeTrainingSampleSet(    Matrixu*    pFrameImageColor, 
                                                            Matrixu*    pFrameImageGray, 
                                                            Matrixu*    pFrameImageHSV = NULL );        
    };
}
#endif