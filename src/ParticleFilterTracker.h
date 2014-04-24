#ifndef PARTICLE_TRACKER_PUBLIC
#define PARTICLE_TRACKER_PUBLIC

#include "Tracker.h"
#include "SimpleTracker.h"
#include "ParticleFilter.h"
#include "AppearanceBasedInformationFuser.h"

namespace MultipleCameraTracking
{
    //Forward Declaration
    class ParticleFilterTracker;
    //declarations of shared ptr
    typedef boost::shared_ptr<ParticleFilterTracker>    ParticleFilterTrackerPtr;

    /****************************************************************
    ParticleFilterTracker (Particle Filter Tracker) 
        Derives from SimpleTracker.
    ****************************************************************/     
    class ParticleFilterTracker : public SimpleTracker
    {
    public:
        //constructor
        ParticleFilterTracker( CvMat* pHomographyMatrix,
                               AppearanceBasedInformationFuserPtr appearanceFuserPtr,
                               bool                                  isAppearanceFusionEnabled = false )  
            : SimpleTracker( pHomographyMatrix ),
            m_meanGroundPlaneFootPositionOnImageList( 2, 0.0f ),
            m_particleFilterPtr( ),
            m_pGroundLocation( ),
            m_particleFilterTrackerParamsPtr( ),
            m_pWeightedAverageParticleMatrix( cvCreateMat( 1, 2, CV_32FC1 ) ),
            m_appearanceFuserPtr( appearanceFuserPtr ),
            m_isAppearanceFusionEnabled( isAppearanceFusionEnabled )
        {
            if ( m_isAppearanceFusionEnabled )
            {
                ASSERT_TRUE( m_appearanceFuserPtr != NULL );
            }
        }

        //destructor
        ~ParticleFilterTracker( );

        //(override SimpleTracker) initializes tracker with first frame(s) and other parameters
        virtual void    InitializeTrackerWithParameters( Matrixu*                pFrameImageColor, 
                                                         Matrixu*                pFrameImageGray, 
                                                         int                    frameInd,
                                                         uint                    videoLength, 
                                                         TrackerParametersPtr    trackerParametersPtr,
                                                         Classifier::StrongClassifierParametersBasePtr    clfparamsPtr,
                                                         Matrixu*                pFrameDisplay            = NULL, 
                                                         Matrixu*                pFrameDisplayTraining    = NULL,
                                                         Matrixu*                pFrameImageHSV            = NULL,
                                                         Matrixf*                pGroundTruthMatrix        = NULL );  

        //(override SimpleTracker) Track an object and store states (by calling StoreObjectStates)
        virtual void    TrackObjectAndSaveState( int        frameind, 
                                                  Matrixu*    pFrameImageColor, 
                                                 Matrixu*    pFrameImageGray, 
                                                 Matrixu*    pFrameDisplay            = NULL, 
                                                 Matrixu*    pFrameDisplayTraining    = NULL,
                                                 Matrixu*    pFrameImageHSV            = NULL );

        Classifier::SampleSet& GetTestSampleSet( ) { return m_testSampleSet; }

        CvMat* GetGroundLocation( bool shouldUseResampledParticles = false ){ EstimateGroundPoint( shouldUseResampledParticles ); return m_pGroundLocation; }

        CvMat* GetAverageParticleOnGroundMatrix( );

        //track an object without any saving, 
        void    TrackObjectWithoutSaveState( Matrixu*    pFrameImageColor, Matrixu*    pFrameImageGray, Matrixu*    pFrameImageHSV    = NULL ) ;

        // store the state of the object in the frame (typically followed by a call to TrackObjectWithoutSaveState
        void    StoreObjectState( int frameind, Matrixu * pFrameDisplay=NULL );

        //Update particle weights based on the PDF obtained using geometric fuser
        void UpdateParticlesWithGroundPDF(    CvMat*        pMeanMatrix,
                                            CvMat*        pCovarianceMatrix,
                                            Matrixu*    pColorImageMatrix,
                                            Matrixu*    pGrayImageMatrix,
                                            Matrixu*    pHsvImageMatrix,
                                            const bool    shouldUseAppFusionWeights );

        void UpdateParticleWeights( vectorf&    likelihoodList, 
                                    bool        shouldUpdateOnlyNonZeroWeights    = true,
                                    bool        shouldForceReset                = false,
                                    bool        shouldResample                    = true );

        //Force resampling
        void ForceParticleResampling( ){ m_particleFilterPtr->ResampleParticles(); }

        //generate training sample sets
        virtual void    GenerateTrainingSampleSet(  Matrixu*    pFrameImageColor, 
                                                    Matrixu*    pFrameImageGray, 
                                                    Matrixu*    pFrameImageHSV = NULL );

        //Generate Test Sample Set
        virtual void GenerateTestSampleSet(    Matrixu* pFrameImageColor,
                                            Matrixu* pFrameImageGray,                                                         
                                            Matrixu* pFrameImageHSV );

        // With ParticleFilterTracker, UpdateClassifier has to been called externally so that Particles can be adjusted before model updating
        virtual void    UpdateClassifier(    Matrixu*    pFrameImageColor, 
                                            Matrixu*    pFrameImageGray, 
                                            Matrixu*    pFrameDisplayTraining = NULL,
                                            Matrixu*    pFrameImageHSV    = NULL );

        /*****function to be implemented********/
        // Suspend the tracking due to various reasons (e.g. occlusion, not-inialized, out of view/boundary etc.)
        void            SuspendTracking( ) {};
        // Check the reliability of tracking results (e.g. too much appearance changes?)
        void            CheckTrackingReliability( ) {};
        // Resume the tracking 
        void            ResumeTracking( ) {};
        virtual void    DrawObjectFootPosition( Matrixu* pFrameDisplay ) const;

    protected:
        bool            InitializeTracker(  Matrixu*                pFrameImageColor, 
                                            Matrixu*                pFrameImageGray,  
                                            TrackerParametersPtr    trackerParametersPtr, 
                                            Classifier::StrongClassifierParametersBasePtr clfparamsPtr,
                                            Matrixu* pFrameDisplay            = NULL, 
                                            Matrixu* pFrameDisplayTraining    = NULL,
                                            Matrixu* pFrameImageHSV            = NULL );

        // track the object on the given frame, classifier is not updated, and  therefore pFrameDisplayTraining is not updated
        virtual double  TrackObjectOnTheGivenFrame( Matrixu*    pFrameImageColor, 
                                                    Matrixu*    pFrameImageGray,
                                                    Matrixu*    pFrameDisplayTraining= NULL,
                                                    Matrixu*    pFrameImageHSV        = NULL );    

        void UpdateParticleWeightsForRearrangedParticles(    Matrixu*    pFrameImageColor, 
                                                            Matrixu*    pFrameImageGray,
                                                            Matrixu*    pFrameImageHSV = NULL,
                                                            const bool  shouldUseAppFusionWeights = false );    

    private:

        //Find mean position on the image plane
        vectorf FindMeanFootPositionOnImagePlane( CvMat* pMeanGroundPlaneLocationMatrix );

        //generate positive sample set based on the current particle state
        virtual void    GeneratePositiveTrainingSampleSet(  Matrixu*    pFrameImageColor, 
                                                            Matrixu*    pFrameImageGray, 
                                                            Matrixu*    pFrameImageHSV = NULL );    

        //generate negative sample set based on the current particle state
        virtual void    GenerateNegativeTrainingSampleSet(  Matrixu*    pFrameImageColor, 
                                                            Matrixu*    pFrameImageGray, 
                                                            Matrixu*    pFrameImageHSV = NULL );

        // Locate object's foot position on the image plane.
        void            EstimateGroundPoint( bool shouldUseResampledParticle = false );

        // Draw predicted particles on a plane (for debugging purpose)
        void            DrawTestSamples( Classifier::SampleSet testSamples, Matrixu* pFrame );

        vectorf                                    m_meanGroundPlaneFootPositionOnImageList;
        ParticleFilterPtr                        m_particleFilterPtr;                // Pointer to the particle filter
        CvMat*                                    m_pGroundLocation;                    // object's ground location (on the image plane)
        ParticleFilterTrackerParametersPtr        m_particleFilterTrackerParamsPtr;    // ParticleFilter tracking parameters
        CvMat*                                    m_pWeightedAverageParticleMatrix;
        AppearanceBasedInformationFuserPtr        m_appearanceFuserPtr;
        bool                                    m_isAppearanceFusionEnabled;
    };
}
#endif