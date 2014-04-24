#include "Object.h"
#include "CommonMacros.h"

#define IMAGE_NAME_PREFIX "img"
#define IMAGE_TYPE          "png"

namespace MultipleCameraTracking
{
    /********************************************************************
    Constructor
        ObjectInstance holds all the information regarding an object 
        inside a camera.

        Typically created by a camera instance.

    Exceptions:
        None
    *********************************************************************/
    Object::Object( const int                                        objectId,
                    const int                                        cameraId,
                    const bool                                        isColorEnabled,
                    CameraTrackingParametersPtr                        cameraTrackingParametersPtr,
                    CvMat*                                            pHomographyMatrix )
        : m_objectID( objectId ),
        m_cameraID( objectId ),
        m_colorImage ( isColorEnabled ),
        m_appearanceFuserPtr( )    ,
        m_cameraTrackingParametersPtr( cameraTrackingParametersPtr ),
        m_pHomographyMatrix( pHomographyMatrix )
    {
    }


    /********************************************************************
    GenerateDefaultTrackerFeatureParameters
        Gets the default tracker feature parameter based on the input config

    Exceptions:
        None
    *********************************************************************/
    Features::FeatureParametersPtr Object::GenerateDefaultTrackerFeatureParameters( )
    {
        try
        {
            Features::FeatureParametersPtr featureParametersPtr;
            if ( m_cameraTrackingParametersPtr->m_trackerFeatureType == Features::HAAR_LIKE )
            {
                featureParametersPtr = Features::FeatureParametersPtr( new Features::HaarFeatureParameters( g_configInput.m_trackerFeatureParameter ) );
            }
            else if ( m_cameraTrackingParametersPtr->m_trackerFeatureType == Features::CULTURE_COLOR_HISTOGRAM )
            {                
                featureParametersPtr = Features::FeatureParametersPtr( new Features::CultureColorHistogramParameters( ) );
            }
            else if ( m_cameraTrackingParametersPtr->m_trackerFeatureType == Features::MULTI_DIMENSIONAL_COLOR_HISTOGRAM )
            {
                featureParametersPtr = Features::FeatureParametersPtr( 
                        new Features::MultiDimensionalColorHistogramParameters(
                            m_cameraTrackingParametersPtr->m_useHSVColorSpaceForColorHistogram,
                            m_cameraTrackingParametersPtr->m_numberOfBinsForColorHistogram
                        ) 
                    );
            }
            else if ( m_cameraTrackingParametersPtr->m_trackerFeatureType == Features::HAAR_COLOR_HISTOGRAM )
            {
                featureParametersPtr = Features::FeatureParametersPtr( 
                        new Features::HaarAndColorHistogramFeatureParameters( 
                            g_configInput.m_trackerFeatureParameter,
                            m_cameraTrackingParametersPtr->m_useHSVColorSpaceForColorHistogram,
                            m_cameraTrackingParametersPtr->m_numberOfBinsForColorHistogram 
                        ) 
                    );
            }
            else
            {
                abortError( __LINE__, __FILE__, "Unsupported Tracker Feature Type." );
            }

            ASSERT_TRUE( featureParametersPtr != NULL );

            return featureParametersPtr;
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to get the default tracker feature parameters" );
    }

    /********************************************************************
    GenerateDefaultAppearanceFusionFeatureParameters
        Gets the default appearance fusion feature parameter based
        on the input config

    Exceptions:
        None
    *********************************************************************/
    Features::FeatureParametersPtr Object::GenerateDefaultAppearanceFusionFeatureParameters( )
    {
        try
        {
            Features::FeatureParametersPtr featureParametersPtr;

            switch(  m_cameraTrackingParametersPtr->m_appearanceFusionType )
            {
            case NO_APPEARANCE_FUSION:
                abortError( __LINE__, __FILE__, "Appearance fusion is disabled" );
                break;
            case FUSION_CULTURE_COLOR_HISTOGRAM:
                featureParametersPtr = Features::FeatureParametersPtr( new Features::CultureColorHistogramParameters( ) );
                break;
            case FUSION_MULTI_DIMENSIONAL_COLOR_HISTOGRAM:
                featureParametersPtr = Features::FeatureParametersPtr( new Features::MultiDimensionalColorHistogramParameters( ) );
                break;
            default:
                abortError( __LINE__, __FILE__, "Unsupported Tracker Feature Type." );
                break;
            }

            ASSERT_TRUE( featureParametersPtr != NULL );

            return featureParametersPtr;
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to get the default appearance fusion feature parameters" );
    }

    /********************************************************************
    GenerateDefaultAppearanceFusionClassifierParameters
        Gets the default appearance fusion classifier parameter based
        on the input config
    Exceptions:
        None
    *********************************************************************/
    Classifier::StrongClassifierParametersBasePtr Object::GenerateDefaultAppearanceFusionClassifierParameters( )
    {
        try
        {
            Classifier::StrongClassifierParametersBasePtr classifierParametersPtr;
            Features::FeatureParametersPtr    featureParametersPtr;

            ASSERT_TRUE( m_cameraTrackingParametersPtr->m_appearanceFusionType != NO_APPEARANCE_FUSION );
            
            featureParametersPtr = GenerateDefaultAppearanceFusionFeatureParameters( );

            int totalNumberOfWeakClassifiers = featureParametersPtr->GetFeatureDimension();    

            int numberOfSelectedWeakClassifiers    = int( totalNumberOfWeakClassifiers *
                                                        g_configInput.m_AFpercentageOfWeakClassifiersSelected / 100 );

            LOG( "Appearance Fusion, totalNumberOfWeakClassifiers: " << totalNumberOfWeakClassifiers 
                << ", numberOfSelectedWeakClassifiers: "  << numberOfSelectedWeakClassifiers << endl );

            int numberOfRetainedWeakClassifiers;
            switch ( m_cameraTrackingParametersPtr->m_appearanceFusionStrongClassifierType )
            {
            case APP_FUSION_MIL_BOOST:        // MILBoost
                classifierParametersPtr = Classifier::StrongClassifierParametersBasePtr( new Classifier::MILBoostClassifierParameters(
                                                                                                    numberOfSelectedWeakClassifiers,
                                                                                                    totalNumberOfWeakClassifiers )
                                                                                        );    
                ASSERT_TRUE( classifierParametersPtr != NULL );

                classifierParametersPtr->m_weakClassifierType    = m_cameraTrackingParametersPtr->m_appearanceFusionWeakClassifierType;
                break;
            case APP_FUSION_ADA_BOOST:        // AdaBoost
                classifierParametersPtr = Classifier::StrongClassifierParametersBasePtr( new Classifier::AdaBoostClassifierParameters( 
                                                                                                    numberOfSelectedWeakClassifiers,
                                                                                                    totalNumberOfWeakClassifiers ) 
                                                                                        );    
                ASSERT_TRUE( classifierParametersPtr != NULL );
                classifierParametersPtr->m_weakClassifierType    = m_cameraTrackingParametersPtr->m_appearanceFusionWeakClassifierType;
                break;
            case APP_FUSION_MIL_ENSEMBLE:    // MILEnsemble
                numberOfRetainedWeakClassifiers = int( totalNumberOfWeakClassifiers*
                    g_configInput.m_AFpercentageOfWeakClassifiersRetained/100 );

                classifierParametersPtr = Classifier::StrongClassifierParametersBasePtr( new Classifier::MILEnsembleClassifierParameters(
                                                                                                    numberOfSelectedWeakClassifiers,
                                                                                                    totalNumberOfWeakClassifiers,
                                                                                                    numberOfRetainedWeakClassifiers
                                                                                        ) );    
                ASSERT_TRUE( classifierParametersPtr != NULL );
                classifierParametersPtr->m_weakClassifierType    = Classifier::PERCEPTRON;
                break;
            default:
                abortError(__LINE__,__FILE__,"Error: invalid classifier choice.");
                break;
            }
            ASSERT_TRUE( classifierParametersPtr != NULL );
            classifierParametersPtr->m_featureParametersPtr = featureParametersPtr;
            return classifierParametersPtr;
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to get the default tracker feature parameters for appearance fusion" );
    }

    /********************************************************************
    InitializeObjectParameters
        Initializes parameters for the object 
         with initial state and file name for saving its trajectory
    Exceptions:
        None
    *********************************************************************/
    void Object::InitializeObjectParameters( const vectorf & initialState, const string & trajSaveStrBase )
    {
        bool success = true;

        //For now particle filter should be enables for color fusion
        if ( m_cameraTrackingParametersPtr->m_appearanceFusionType != NO_APPEARANCE_FUSION && m_cameraTrackingParametersPtr->m_localObjectTrackerType == PARTICLE_FILTER_TRACKER )
        {
            //initialize the appearance fusion based on the config information
            m_appearanceFuserPtr 
                    = AppearanceBasedInformationFuserPtr( new AppearanceBasedInformationFuser(    m_objectID,
                                                                                                m_cameraID,
                                                                                                m_cameraTrackingParametersPtr->m_appearanceFusionType,
                                                                                                GenerateDefaultAppearanceFusionClassifierParameters( )
                                                                                            )
                                                        );
            ASSERT_TRUE( m_appearanceFuserPtr != NULL );
        }

        if ( m_cameraTrackingParametersPtr->m_localObjectTrackerType == PARTICLE_FILTER_TRACKER )
        {            
            m_trackerPtr                    = TrackerPtr( new ParticleFilterTracker ( m_pHomographyMatrix, m_appearanceFuserPtr,  m_cameraTrackingParametersPtr->m_appearanceFusionType != NO_APPEARANCE_FUSION ) ); 
            m_trackerParametersPtr            = TrackerParametersPtr( new ParticleFilterTrackerParameters( ) );  
        }
        else if ( m_cameraTrackingParametersPtr->m_localObjectTrackerType == SIMPLE_TRACKER )
        {
            m_trackerPtr                    = TrackerPtr(    new SimpleTracker ( m_pHomographyMatrix ) ); 
            m_trackerParametersPtr            = TrackerParametersPtr( new SimpleTrackerParameters ( ) );
        }

        Features::FeatureParametersPtr featureParametersPtr = GenerateDefaultTrackerFeatureParameters( );
        
        //determine the total number of weak classifiers
        int totalNumberOfWeakClassifiers = featureParametersPtr->GetFeatureDimension();
        int numberOfSelectedWeakClassifiers;

        ASSERT_TRUE( totalNumberOfWeakClassifiers > 0 );

        numberOfSelectedWeakClassifiers    = int (    totalNumberOfWeakClassifiers *
                                                g_configInput.m_percentageOfWeakClassifiersSelected/100 );
        // strong classifier model
        switch ( m_cameraTrackingParametersPtr->m_localTrackerStrongClassifierType )
        {
            case Classifier::ONLINE_STOCHASTIC_BOOST_MIL:        // MilBoost
                m_classifierParamPtr = Classifier::StrongClassifierParametersBasePtr( new Classifier::MILBoostClassifierParameters( 
                                                                                            numberOfSelectedWeakClassifiers,
                                                                                            totalNumberOfWeakClassifiers) 
                                                                                    );
                m_classifierParamPtr->m_weakClassifierType = m_cameraTrackingParametersPtr->m_localTrackerWeakClassifierType;
                break;
            case Classifier::ONLINE_ADABOOST:                    // AdaBoost
                m_classifierParamPtr = Classifier::StrongClassifierParametersBasePtr( new Classifier::AdaBoostClassifierParameters( 
                                                                                            numberOfSelectedWeakClassifiers,
                                                                                            totalNumberOfWeakClassifiers)
                                                                                    );                        
                m_classifierParamPtr->m_weakClassifierType = m_cameraTrackingParametersPtr->m_localTrackerWeakClassifierType;
                break;
            case Classifier::ONLINE_ENSEMBLE_BOOST_MIL:            // MILEnsemble

                m_classifierParamPtr = Classifier::StrongClassifierParametersBasePtr( 
                                            new Classifier::MILEnsembleClassifierParameters( numberOfSelectedWeakClassifiers,
                                                                                            totalNumberOfWeakClassifiers,
                                                                                            g_configInput.m_percentageOfWeakClassifiersRetained )
                                                                                    );
                m_classifierParamPtr->m_weakClassifierType  = m_cameraTrackingParametersPtr->m_localTrackerWeakClassifierType;
                //m_classifierParamPtr->m_weakClassifierType = Classifier::PERCEPTRON;    //only Perceptron
                break;
            case Classifier::ONLINE_ANY_BOOST_MIL:
                m_classifierParamPtr = Classifier::StrongClassifierParametersBasePtr( new Classifier::MILAnyBoostClassifierParameters( 
                                                                                        numberOfSelectedWeakClassifiers,
                                                                                        totalNumberOfWeakClassifiers)
                                                                                        );                        
                m_classifierParamPtr->m_weakClassifierType = m_cameraTrackingParametersPtr->m_localTrackerWeakClassifierType;
                break;
            default:
                abortError(__LINE__,__FILE__,"Error: invalid classifier choice.");
                break;
        }
        
        m_classifierParamPtr->m_featureParametersPtr    =     featureParametersPtr;

        // tracker parameters
        m_trackerParametersPtr->m_posRadiusTrain        =    static_cast<float>( g_configInput.m_posRadiusTrain );
        m_trackerParametersPtr->m_numberOfNegativeTrainingSamples            = g_configInput.m_numNegExamples;

        m_trackerParametersPtr->m_isColor                = m_colorImage;
        m_trackerParametersPtr->m_init_negNumTrain        = g_configInput.m_initNumNegExampes;
        m_trackerParametersPtr->m_init_posTrainRadius    = static_cast<float>( g_configInput.m_initPosRadiusTrain );
        m_trackerParametersPtr->m_initState                = initialState;
        
        m_trackerParametersPtr->m_initializeWithFaceDetection            = DEFAULT_TRACKER_INIT_WITH_FACE;
        m_trackerParametersPtr->m_debugv                = DEFAULT_TRACKER_DEBUGV; 
        m_trackerParametersPtr->m_displayTrainingSampleCenterOnly    = ( g_configInput.m_displayTrainingExampCenterOnly ==1 );
        m_trackerParametersPtr->m_displayFigureNameStr    = "Camera "+int2str(m_cameraID,3) + ", object "+int2str(m_objectID,3);
        m_trackerParametersPtr->m_trajSave                = trajSaveStrBase +"_Object" + int2str( m_objectID, 3 );

        
        if ( m_cameraTrackingParametersPtr->m_localObjectTrackerType == SIMPLE_TRACKER ||
            m_cameraTrackingParametersPtr->m_localObjectTrackerType == PARTICLE_FILTER_TRACKER )
        { 
            SimpleTrackerParametersPtr    simpleTrackerParametersPtr = boost::static_pointer_cast<SimpleTrackerParameters>(m_trackerParametersPtr);            

            simpleTrackerParametersPtr->m_searchWindSize   = g_configInput.m_searchWindowSize;
            simpleTrackerParametersPtr->m_negSampleStrategy= g_configInput.m_negSampleStrategy;            
        }

        // parameters for only the particle filter tracker 
        if ( m_cameraTrackingParametersPtr->m_localObjectTrackerType == PARTICLE_FILTER_TRACKER )
        {  
            ParticleFilterTrackerParametersPtr PFTrackerparamsPtr    = boost::static_pointer_cast<ParticleFilterTrackerParameters>(m_trackerParametersPtr);            
            PFTrackerparamsPtr->m_shouldNotUseSigmoid                =    DEFAULT_PFTRACKER_NOT_USE_SIGMOIDAL;    //use sigmoid function to calculate probability if =false
            PFTrackerparamsPtr->m_numberOfParticles                    =    m_cameraTrackingParametersPtr->m_numberOfParticles;
            PFTrackerparamsPtr->m_standardDeviationX                =    (float)(g_configInput.m_PFTrackerStdDevX);            
            PFTrackerparamsPtr->m_standardDeviationY                =    (float)(g_configInput.m_PFTrackerStdDevY);    
            PFTrackerparamsPtr->m_standardDeviationScaleX            =    (float)(g_configInput.m_PFTrackerStdDevScaleX);        
            PFTrackerparamsPtr->m_standardDeviationScaleY            =    (float)(g_configInput.m_PFTrackerStdDevScaleY);        
            PFTrackerparamsPtr->m_maxNumPositiveExamples            =    g_configInput.m_PfTrackerMaxNumPositiveExamples;
            PFTrackerparamsPtr->m_numOfDisplayedParticles            =    g_configInput.m_PFTrackerNumDispParticles;

            switch ( g_configInput.m_PfTrackerPositiveExampleStrategy )
            {
            case 0:
                PFTrackerparamsPtr->m_positiveSampleStrategy        = SAMPLE_POS_SIMPLETRACKER;
                break;
            case 1:
                PFTrackerparamsPtr->m_positiveSampleStrategy        = SAMPLE_POS_GREEDY;
                break;
            case 2:
                PFTrackerparamsPtr->m_positiveSampleStrategy        = SAMPLE_POS_SEMI_GREEDY;
                break;
            case 3:
                PFTrackerparamsPtr->m_positiveSampleStrategy        = SAMPLE_POS_PARTICLE_RANDOM;
                break;
            default:
                abortError( __LINE__, __FILE__, "invalid option for particle filter positive sampling strategy" );
            }
            
            switch( g_configInput.m_PfTrackerNegativeExampleStrategy )
            {
            case 0:
                PFTrackerparamsPtr->m_negativeSampleStrategy        = SAMPLE_NEG_SIMPLETRACKER;
                break;
            case 1:
                PFTrackerparamsPtr->m_negativeSampleStrategy        = SAMPLE_NEG_PARTICLE_RANDOM;
                break;
            default:
                abortError( __LINE__, __FILE__, "invalid option for particle filter negative sampling strategy" );
            }

            switch ( g_configInput.m_PFOutputTrajectoryOption )
            {
            case 0:
                PFTrackerparamsPtr->m_outputTrajectoryOption        = PARTICLE_HIGHEST_WEIGHT;
                break;
            case 1:
                PFTrackerparamsPtr->m_outputTrajectoryOption        = PARTICLE_AVERAGE;
                break;
            default:
                abortError( __LINE__, __FILE__, "invalid option for particle filter trajectory saving" );
            }                
        }
    }

    /********************************************************************
    Initialize the Tracker for the object        
    Exceptions:
        None
    *********************************************************************/
    bool Object::InitializeObjectTracker( Matrixu* pFrameImageColor, 
                                          Matrixu* pFrameImageGray, 
                                          int frameInd, 
                                          uint videoLength,
                                          Matrixu* pFrameDisplay, 
                                          Matrixu* pFrameDisplayTraining,
                                          Matrixu* pFrameImageHSV,
                                          Matrixf* pGroundTruthMatrix )
    {
        try
        {
            if ( m_trackerParametersPtr.get() == NULL )
            {
                abortError( __LINE__, __FILE__, "m_pSimpleTrackerParameters is NULL" );
            }

            if ( m_classifierParamPtr.get() == NULL )
            {
                abortError( __LINE__, __FILE__, "m_pClassifierParams is NULL" );
            }
            
            if( frameInd != m_trackerParametersPtr->m_initState[4] )
                return false; //return false, if initialization info is not available now at frameInd
            else 
            {
                m_trackerPtr->InitializeTrackerWithParameters( 
                    pFrameImageColor, pFrameImageGray, frameInd, videoLength, m_trackerParametersPtr, m_classifierParamPtr,
                    pFrameDisplay, pFrameDisplayTraining, pFrameImageHSV, pGroundTruthMatrix );
                return true;
            }
            
        }
        EXCEPTION_CATCH_AND_ABORT( "Error in InitializeTrackerParameters" )
    }

    /********************************************************************
    TrackObjectFrame
        Tracks the object in a given frame
        Note: 
        1. For simple tracker: classifier is automatically updated and tracked frame is displayed and saved if required
        2. For particle filter tracker:
            if fusion is not enabled, state are saved, classifier is automatically updated
            otherwise, just track the object without saving state or updating classifier
    Exceptions:
        None
    *********************************************************************/
    void Object::TrackObjectFrame(    int            frameIndex, 
                                    Matrixu*    pFrameImageColor,
                                    Matrixu*    pFrameImageGray, 
                                    Matrixu*    pFrameDisplay,
                                    Matrixu*    pFrameDisplayTraining, 
                                    Matrixu*    pFrameImageHSV )
    {
        try
        {        
            m_indPreviousTrackedFrame = frameIndex;

            //if no fusion at all track and save state
            if( m_cameraTrackingParametersPtr->m_appearanceFusionType == NO_APPEARANCE_FUSION && m_cameraTrackingParametersPtr->m_geometricFusionType== NO_GEOMETRIC_FUSION )
            {
                LOG( "Object "<< m_objectID << ": Tracking (including saving state and updating classifier)" << endl );

                m_trackerPtr->TrackObjectAndSaveState(  frameIndex,
                                                        pFrameImageColor,    pFrameImageGray,
                                                        pFrameDisplay,        pFrameDisplayTraining,
                                                        pFrameImageHSV );
            }
            //if fusion is required (only if it is particle filters, just tracked the object without save state
            else
            {
                LOG( "Object "<< m_objectID << ":Tracking object (without saving state or updating classifier)" << endl );

                boost::static_pointer_cast<ParticleFilterTracker>(m_trackerPtr)->TrackObjectWithoutSaveState( 
                                                                        pFrameImageColor, pFrameImageGray, pFrameImageHSV );

                //re-weight the particles based on the learned appearance model across different views.
                if ( m_cameraTrackingParametersPtr->m_appearanceFusionType != NO_APPEARANCE_FUSION && frameIndex > 2
                    &&  (frameIndex%m_cameraTrackingParametersPtr->m_appearanceFusionRefreshRate== 0) )
                {
                    LOG( "\n Updating weights based on the appearance fusion results" );
                    UpdateParticleWeightUsingMultiCameraAppearanceModel( pFrameImageColor, pFrameImageGray, pFrameImageHSV );
                }        

                //force particle resampling after appearance fusion
                boost::static_pointer_cast<ParticleFilterTracker>(m_trackerPtr)->ForceParticleResampling( );
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Error in Camera.TrackCameraFrame()" )
    }

    /********************************************************************
    UpdateParticleWeightsWithGroundPDF
        Update the particle weights geometric fusion.
    Exceptions:
        None
    *********************************************************************/
    void    Object::UpdateParticleWeightsWithGroundPDF( CvMat*        pMeanMatrix, 
                                                        CvMat*        pCovarianceMatrix,
                                                        Matrixu*    pColorImageMatrix,
                                                        Matrixu*    pGrayImageMatrix,
                                                        Matrixu*    pHsvImageMatrix )
    {
        try
        {
            ASSERT_TRUE( m_cameraTrackingParametersPtr->m_localObjectTrackerType == PARTICLE_FILTER_TRACKER );

            ASSERT_TRUE( pMeanMatrix != NULL );
            ASSERT_TRUE( pCovarianceMatrix != NULL );

            ParticleFilterTrackerPtr particleFilterTrackerPtr = boost::static_pointer_cast<ParticleFilterTracker>(m_trackerPtr);

            ASSERT_TRUE( particleFilterTrackerPtr != NULL );

            bool shouldUseAppFusionWeights = (m_indPreviousTrackedFrame%m_cameraTrackingParametersPtr->m_appearanceFusionRefreshRate== 0);

            particleFilterTrackerPtr->UpdateParticlesWithGroundPDF( pMeanMatrix, 
                                                                    pCovarianceMatrix,
                                                                    pColorImageMatrix,
                                                                    pGrayImageMatrix, 
                                                                    pHsvImageMatrix,
                                                                    shouldUseAppFusionWeights );
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to update the particle weights with ground plane pdf" );
    }

    /********************************************************************
    UpdateParticleWeightUsingMultiCameraAppearanceModel
        Update the particle weights using multi camera appearance model.
    Exceptions:
        None
    *********************************************************************/
    void    Object::UpdateParticleWeightUsingMultiCameraAppearanceModel( Matrixu* pFrameImageColor,
                                                                        Matrixu* pFrameImageGray,
                                                                        Matrixu* pFrameImageHSV )
    {
        try
        {
            ASSERT_TRUE( m_cameraTrackingParametersPtr->m_localObjectTrackerType == PARTICLE_FILTER_TRACKER );

            boost::static_pointer_cast<ParticleFilterTracker>(m_trackerPtr)->GenerateTestSampleSet( pFrameImageColor, pFrameImageGray, pFrameImageHSV );

            Classifier::SampleSet& testSampleSet = boost::static_pointer_cast<ParticleFilterTracker>(m_trackerPtr)->GetTestSampleSet( );
            if ( testSampleSet.Size() == 0 )
            {
                //no valid test sample set exist, return
                return;
            }
            
            vectorf    likelihoodProbabilityList( testSampleSet.Size(), 0.0 );

            ASSERT_TRUE( likelihoodProbabilityList.size( ) == testSampleSet.Size() )
            ASSERT_TRUE( m_appearanceFuserPtr != NULL );

            m_appearanceFuserPtr->FuseInformation( testSampleSet, likelihoodProbabilityList );


            //Update the weights after appearance fusion
            boost::static_pointer_cast<ParticleFilterTracker>(m_trackerPtr)->UpdateParticleWeights( likelihoodProbabilityList,
                                                                                                    true/*shouldUpdateNonZeroWeights*/,
                                                                                                    false );    

        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to update particle weights using the multi camera appearance model" );
    }

    /********************************************************************
    StoreParticleFilterTrackerState
        Notify the tracker to keep the particle filter tracker state as 
        the final state for the frame and update pFrameDisplay if given.
    Exceptions:
        None
    *********************************************************************/
    void Object::StoreParticleFilterTrackerState( int frameInd, Matrixu* pFrameDisplay )
    {
        try
        {
            ASSERT_TRUE( m_cameraTrackingParametersPtr->m_localObjectTrackerType == PARTICLE_FILTER_TRACKER );
            boost::static_pointer_cast<ParticleFilterTracker>(m_trackerPtr) -> StoreObjectState(frameInd, pFrameDisplay);
        }
        EXCEPTION_CATCH_AND_ABORT( "Error in Camera.StoreParticleFilterTrackerState" ); 
    }

    /********************************************************************
    UpdateParticleFilterTrackerAppearanceModel
        Update the appearance model of current particle filter tracker
            and update pFrameDisplayTraining if given
    Exceptions:
        None
    *********************************************************************/
    void Object::UpdateParticleFilterTrackerAppearanceModel(    Matrixu* pFrameImageColor, 
                                                                Matrixu* pFrameImageGray, 
                                                                Matrixu* pFrameDisplayTraining, 
                                                                Matrixu* pFrameImageHSV )
    {
        try
        {
            ASSERT_TRUE( m_cameraTrackingParametersPtr->m_localObjectTrackerType == PARTICLE_FILTER_TRACKER );            
            //assume frame is same as previous call to TrackObjectFrame
            boost::static_pointer_cast<ParticleFilterTracker>(m_trackerPtr)-> 
                UpdateClassifier( pFrameImageColor, pFrameImageGray, pFrameDisplayTraining,    pFrameImageHSV );
        }
        EXCEPTION_CATCH_AND_ABORT( "Error in Camera.UpdateParticleFilterTrackerAppearanceModel" ); 
    }

    /********************************************************************
    SaveObjectStatesAllFrames
        Notify the tracker to save the state for the object for all frames
        to trajectory file.
    Exceptions:
        None
    *********************************************************************/
    void Object::SaveObjectStatesAllFrames( )
    {
        try
        {
            if ( m_trackerParametersPtr.get() == NULL )
            {
                abortError(__LINE__,__FILE__,"m_pSimpleTrackerParameters is NULL");
            }

            m_trackerPtr->SaveStates();
        
            if( m_cameraTrackingParametersPtr->m_calculateTrackingError )
            {
                m_trackerPtr->CalculateTrackingErrroFromGroundTruth();
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Error while saving objects states on all the frames" );
        
    }

    /********************************************************************
    GetParticlesFootPositionOnImagePlaneForGeometricFusion
        Get ground plane position particles (still on the image plane)
    Exceptions:
        None
    *********************************************************************/
    CvMat* Object::GetParticlesFootPositionOnImagePlaneForGeometricFusion( )
    {
        try
        {
            ASSERT_TRUE( m_cameraTrackingParametersPtr->m_localObjectTrackerType == PARTICLE_FILTER_TRACKER );
            return boost::static_pointer_cast<ParticleFilterTracker>(m_trackerPtr)-> GetGroundLocation( true /*shouldUseResampledParticles*/ );    
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to get particles location on ground plane" );
    }

    /********************************************************************
    GetParticlesFootPositionOnImagePlaneForGeometricFusion
        Get average particle foot position on image plane
    Exceptions:
        None
    *********************************************************************/
    CvMat* Object::GetAverageParticleFootPositionOnImagePlaneForGeometricFusion( )
    {
        try
        {
            ASSERT_TRUE( m_cameraTrackingParametersPtr->m_localObjectTrackerType == PARTICLE_FILTER_TRACKER );
            return boost::static_pointer_cast<ParticleFilterTracker>(m_trackerPtr)->GetAverageParticleOnGroundMatrix( );    
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to get average particle location on ground plane" );
    }

    /********************************************************************
    GenerateTrainingSampleSetsForAppearanceFusion
        Extracts sample sets for appearance fusion based on the current
        state of the tracker
    Exceptions:
        None
    *********************************************************************/
    void Object::GenerateTrainingSampleSetsForAppearanceFusion( Classifier::SampleSet&    positiveSampleSet,
                                                                Classifier::SampleSet&    negativeSampleSet,
                                                                Matrixu*                pFrameImageColor,
                                                                Matrixu*                pFrameImageGray, 
                                                                Matrixu*                pFrameImageHSV )
    {
        try
        {
            ASSERT_TRUE( m_cameraTrackingParametersPtr->m_appearanceFusionType != NO_APPEARANCE_FUSION );
            ASSERT_TRUE( m_trackerPtr != NULL );
            
            boost::static_pointer_cast<ParticleFilterTracker>(m_trackerPtr)->GenerateTrainingSampleSet( pFrameImageColor, pFrameImageGray, pFrameImageHSV );

            boost::static_pointer_cast<ParticleFilterTracker>(m_trackerPtr)->GetTrainingSampleSets( positiveSampleSet, negativeSampleSet );
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to generate sample sets for appearance fusion.");
    }

    /********************************************************************
    LearnGlobalAppearanceModel
        Learn Multi Camera (global )Appearance Model
    Exceptions:
        None
    *********************************************************************/
    void    Object::LearnGlobalAppearanceModel( CameraNetworkBasePtr cameraNetworkBasePtr )
    {
        try
        {
            ASSERT_TRUE( m_appearanceFuserPtr != NULL );
            ASSERT_TRUE( cameraNetworkBasePtr != NULL );

            m_appearanceFuserPtr->LearnGlobalAppearanceModel( cameraNetworkBasePtr );
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to Learn MultiCameraAppearance Model" );
    }
    
    /********************************************************************
    DrawObjectFootPoint
        DrawObjectFootPoint
    Exceptions:
        None
    *********************************************************************/
    void    Object::DrawObjectFootPoint(Matrixu* pFrameDisplay)
    {
        try
        {
            m_trackerPtr->DrawObjectFootPosition( pFrameDisplay );
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to Draw Object Foot Point" );
    }

}