#include "Camera.h"
#include "CommonMacros.h"
#include "GeometryBasedInformationFuser.h"

#define IMAGE_NAME_PREFIX "img"
#define IMAGE_TYPE          "png"

namespace MultipleCameraTracking
{
    /********************************************************************
    Constructor
        CameraInstance object holds all the information regarding a camera.
    Exceptions:
        None
    *********************************************************************/
    Camera::Camera( const int                        cameraId,
                    const vectori&                    objectIdList,
                    CameraTrackingParametersPtr        cameraTrackingParametersPtr )
        : m_pHomographyMatrix( cvCreateMat( 3, 3, CV_64FC1 ) ),
        m_cameraTrackingParametersPtr ( ASSERT_PRECONDITION_PARAMETER( cameraTrackingParametersPtr != NULL, cameraTrackingParametersPtr ) ), 
        m_videoMatrix( ),
        m_frameMatrix( ),
        m_initialState( ),
        m_cameraID( cameraId )        
    {
        try
        {
            // input stuff
            m_readImages            =   ( g_configInput.m_loadVideoFromImgs==1 );    //whether to read from image sequence or from a single video file
            m_sourceIsColorImage    =    ( g_configInput.m_loadVideoWithColor==1 );    //whether to read as color image

            // output stuff
            m_saveTrackedVideo            =    ( g_configInput.m_saveOutputVideo==1    );    //whether to save video
            m_saveVideoTrainingExamples =    ( g_configInput.m_saveTrainingSamplesVideo == 1); //whether to save training examples on video
            m_outputDirectoryCstr        =    g_configInput.m_outputDirectoryNameCstr;    //output directory, default is the same as input directory name

            // displaying related stuff
            m_displayTrackedVideo        =    ( g_configInput.m_displayOutputVideo == 1);        //whether to display output video
            m_displayTrainingSamples    =    ( g_configInput.m_displayTrainingSamples == 1); //whether to display training examples

            //tracking parameters    
            if ( m_cameraTrackingParametersPtr->m_appearanceFusionType != NO_APPEARANCE_FUSION ||
                    m_cameraTrackingParametersPtr->m_trackerFeatureType == Features::CULTURE_COLOR_HISTOGRAM ||
                    m_cameraTrackingParametersPtr->m_trackerFeatureType == Features::MULTI_DIMENSIONAL_COLOR_HISTOGRAM )
            {
                ASSERT_TRUE( m_sourceIsColorImage );    // Input image must be color image if color fusion is required.
            }

            InitializeCameraParameters( objectIdList );            
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to construct Camera" );
    }

    /********************************************************************
    GetObjectPtr
        Get the objects based on the cameraId
    Exceptions:
        None
    *********************************************************************/
    ObjectPtr    Camera::GetObjectPtr( const int objectId ) const
    {
        try
        {
            ASSERT_TRUE( objectId >= 0 );
            ASSERT_TRUE( !m_objectPtrList.empty( ) );

            for ( int objectIndex = 0; objectIndex < m_objectPtrList.size(); objectIndex++ )
            {
                ASSERT_TRUE( m_objectPtrList[objectIndex] != NULL );

                if ( m_objectPtrList[objectIndex]->GetObjectID( ) == objectId )
                {
                    return m_objectPtrList[objectIndex];
                }
            }

            //we should have returned with the object
            ASSERT_TRUE( false )
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to get the object" );
    }

    /********************************************************************
    InitializeCameraParameters
        Initializes parameters for each camera.
        And read video data and tracking initialization data.
    Exceptions:
        None
    *********************************************************************/
    void Camera::InitializeCameraParameters( const vectori&    objectIdList )
    {

        bool success = true;
        randinitalize(m_cameraTrackingParametersPtr->m_trialNumber);

        //Data directory location
        string dataDir = string( m_cameraTrackingParametersPtr->m_inputDirectoryString );
        string outputDir = string( m_cameraTrackingParametersPtr->m_outputDirectoryString );
        if( dataDir[dataDir.length()-2] != '/' ) dataDir+="/";
        dataDir += m_cameraTrackingParametersPtr->m_dataNameString;
        dataDir += "/";

        if( outputDir[outputDir.length()-2] != '/' ) outputDir+="/";

        outputDir +=  m_cameraTrackingParametersPtr->m_dataNameString;
        outputDir +=  "/";
        
        // find the frames to be processed
        m_frameMatrix.Resize(1,2);
        m_frameMatrix(0)    =    g_configInput.m_startFrameIndex;
        m_frameMatrix(1)    =    g_configInput.m_startFrameIndex + g_configInput.m_numOfFrames - 1;
        
        // read in initial state (each row corresponds to an object, with format: [objectIndex, pos_x, pos_y, siz_x, size_y]
        success = m_initialState.DLMRead((dataDir + string( m_cameraTrackingParametersPtr->m_nameInitilizationString ) +"/" + m_cameraTrackingParametersPtr->m_dataNameString + "_gt" + int2str(m_cameraID,3)  + ".txt").c_str());
        if( !success ) 
        {
            abortError(__LINE__,__FILE__,"Error: gt file not found.");
        }

        //Homography Parameter
        Matrixf homoGraphy;
        success = homoGraphy.DLMRead( ( dataDir + "Homography_" + int2str( m_cameraID, 3 ) + ".txt" ).c_str( ) );
        if( !success )
        { 
            abortError(__LINE__,__FILE__,"Error: Homography not readable.");
        }

        #pragma omp parallel for
        for (int objInd=0; objInd < 9; objInd++)
        {
            m_pHomographyMatrix->data.db[objInd] = homoGraphy(objInd);
        }

        //Initialize the objects
        //create the list of objects inside the cameras
        for( int objInd = 0; objInd < objectIdList.size(); objInd++)
        {
            ObjectPtr objectPtr( new Object( objectIdList[objInd], 
                                            m_cameraID,
                                            m_sourceIsColorImage,
                                            m_cameraTrackingParametersPtr,
                                            m_pHomographyMatrix ) 
                                            );

            ASSERT_TRUE( objectPtr != NULL );

            //push back the object into the list
            m_objectPtrList.push_back( objectPtr );
            m_objectStatusList.push_back( OBJECT_TRACKING_UN_INITIALIZED );
        }

        string paramname = "";
        //parameter name to name proper output files
        switch( m_cameraTrackingParametersPtr->m_localTrackerStrongClassifierType )
        {
            case Classifier::ONLINE_STOCHASTIC_BOOST_MIL:     // MILTrack 
                paramname += "_MB"; 
                break;
            case Classifier::ONLINE_ADABOOST:                // OBA1
                paramname += "_AB";
                break;
            case Classifier::ONLINE_ENSEMBLE_BOOST_MIL:        // _MILENSEMBLE
                paramname += "_ME";
                break;
            case Classifier::ONLINE_ANY_BOOST_MIL:            //MILboost with anyboost
                paramname += "_MA";
                break;
            default: 
                abortError(__LINE__,__FILE__,"Error: invalid classifier choice.");
        }

        // Load the video into the memory
        m_videoMatrix.clear();
        if ( m_readImages )
        {
            m_videoMatrix = Matrixu::LoadVideo( (dataDir + "imgs" + int2str(m_cameraID,3) + "/").c_str(),
                                                IMAGE_NAME_PREFIX,
                                                IMAGE_TYPE,
                                                (int)m_frameMatrix(0),
                                                (int)m_frameMatrix(1),
                                                5,m_sourceIsColorImage);
        }
        else
        {
            m_videoMatrix = Matrixu::LoadVideoStream( (dataDir + "video" + int2str(m_cameraID,3) + ".avi").c_str(),
                                                        (int)m_frameMatrix(0),
                                                        (int)m_frameMatrix(1),
                                                        m_sourceIsColorImage);
        }

        // Load ground truth if necessary
        if( m_cameraTrackingParametersPtr->m_calculateTrackingError )
        {
            m_groundTruthMatrixList.resize( m_objectPtrList.size() );

            for (int objInd = 0; objInd < m_objectPtrList.size(); objInd++)
            {        
                m_groundTruthMatrixList[objInd].DLMRead(
                    (dataDir + string( m_cameraTrackingParametersPtr->m_nameInitilizationString ) +"/" + 
                        m_cameraTrackingParametersPtr->m_dataNameString + "_GT" + int2str(m_cameraID,3)
                        + "_Obj"+int2str(m_objectPtrList[objInd]->GetObjectID(),3)+ ".txt").c_str() );
            }
        }

        //initialize all output video files if required
        m_pVideoWriter = NULL;
        m_pVideoWriterTraining = NULL;
        if ( m_saveTrackedVideo )
        { //save tracked frames
            string m_videoSave    =    outputDir +  m_cameraTrackingParametersPtr->m_nameInitilizationString +"/"+
                "TR" + int2str(m_cameraTrackingParametersPtr->m_trialNumber,3) + "_C" + int2str(m_cameraID,3)+ paramname + ".avi";
            
            m_pVideoWriter = cvCreateVideoWriter( m_videoSave.c_str(), 
                CV_FOURCC('x','v','i','d'),
                15, 
                cvSize(m_videoMatrix[0].cols(), m_videoMatrix[0].rows() ),
                3 );

            if ( m_pVideoWriter==NULL ) 
            {
                abortError(__LINE__,__FILE__,"Error opening video file for output");
            }
        }
        
        // output video file
        if ( m_saveVideoTrainingExamples ) 
        { //save training examples
            string m_videoTrainingExamples    =  outputDir +  m_cameraTrackingParametersPtr->m_nameInitilizationString +"/"+ 
                    "TR" + int2str(m_cameraTrackingParametersPtr->m_trialNumber,3) + "_C" + int2str(m_cameraID,3)+ paramname + "_Training.avi";

            m_pVideoWriterTraining = cvCreateVideoWriter( m_videoTrainingExamples.c_str(), 
                CV_FOURCC('x','v','i','d'),
                15, 
                cvSize(m_videoMatrix[0].cols(), m_videoMatrix[0].rows() ),
                3 );

            if ( m_pVideoWriterTraining==NULL ) 
            {
                abortError(__LINE__,__FILE__,"Error opening video file for output");
            }
        }

        string m_trajSaveStrBase    = outputDir +  m_cameraTrackingParametersPtr->m_nameInitilizationString +"/"+ 
            "TR" + int2str(m_cameraTrackingParametersPtr->m_trialNumber,3) + "_C" + int2str(m_cameraID,3)+ paramname;

        // Initialize all objects
        vectorf objectInitialState;
        objectInitialState.resize( 5 );
        for (int objInd = 0; objInd < m_objectPtrList.size(); objInd++)
        {            
            for( int k = 0; k < 4; k++ )
            {
                objectInitialState[k] = 0;
            }

            for( int j = 0; j < m_initialState.rows(); j++)
            {
                if( m_initialState(j,4) == m_objectPtrList[objInd]->GetObjectID( ) )
                { //find the initial object state from the matrix m_initialState
                    for( int k = 0; k < 4; k++ )
                    {
                        objectInitialState[k] = m_initialState( j, k );
                    }
                    objectInitialState[4] = m_initialState(j,5) - m_frameMatrix(0);
                    break;
                }
            }
            m_objectPtrList[objInd]->InitializeObjectParameters( objectInitialState,m_trajSaveStrBase );                    
        }
    }

    /********************************************************************
    InitializeCameraTrackers
        initialize all the object trackers inside the camera
    Exceptions:
        None
    *********************************************************************/
    void Camera::InitializeCameraTrackers( )
    {
        try
        {
            int frameInd = 0;

            PrepareCurrentFrameForTracking( frameInd );
            
            for ( int i = 0; i < m_objectPtrList.size(); i++ )
            {
                ASSERT_TRUE( m_objectPtrList[i] != NULL );
                ASSERT_TRUE( m_objectStatusList[i] == OBJECT_TRACKING_UN_INITIALIZED );
                //try to initialize
                Matrixf * pGroundTruthMatrix = NULL; 
                if ( m_cameraTrackingParametersPtr-> m_calculateTrackingError ) 
                {
                    pGroundTruthMatrix = &m_groundTruthMatrixList[i];
                }
                
                if( m_objectPtrList[i]->InitializeObjectTracker( m_pCurrentFrameImageMatrixColor,
                                                                 m_pCurrentFrameImageMatrixGray,
                                                                 frameInd, 
                                                                 m_videoMatrix.size(),
                                                                 m_pFrameDisplay, 
                                                                 m_pFrameDisplayTraining,
                                                                 m_pCurrentFrameImageMatrixHSV, 
                                                                 pGroundTruthMatrix ) )
                {
                    m_objectStatusList[i] = OBJECT_TRACKING_IN_PROGRESS;
                }                            
            }
            if ( ( m_cameraTrackingParametersPtr->m_trackerFeatureType == Features::HAAR_LIKE 
                || m_cameraTrackingParametersPtr->m_trackerFeatureType == Features::HAAR_COLOR_HISTOGRAM ) 
                && m_pCurrentFrameImageMatrixGray->isInitII() )
            { 
                //free the integral image
                m_pCurrentFrameImageMatrixGray->FreeII();
            }

            DisplayAndSaveTrackedFrame();
            DisplayAndSaveTrainingSamples();
        }
        EXCEPTION_CATCH_AND_ABORT( "Error in InitializeCameraTrackers" )
    }

    /********************************************************************
    TrackCameraFrame
        Tracks a Given Camera Frame
        Note: 
        1. For simple tracker: classifier is automatically updated and tracked frame is displayed and saved if required
        2. For particle filter tracker:
            if fusion is not enabled, state are saved, classifier is automatically updated
            otherwise, just track the object without saving state or updating classifier
    Exceptions:
        None
    *********************************************************************/
    void Camera::TrackCameraFrame( int frameInd )
    {
        try
        {    
            PrepareCurrentFrameForTracking( frameInd );

            for (int objInd = 0; objInd < m_objectPtrList.size(); objInd++)
            {            
                switch( m_objectStatusList[objInd] ) 
                {
                case OBJECT_TRACKING_IN_PROGRESS:
                    m_objectPtrList[objInd]->TrackObjectFrame( frameInd, 
                                                    m_pCurrentFrameImageMatrixColor,
                                                    m_pCurrentFrameImageMatrixGray,
                                                    m_pFrameDisplay,
                                                    m_pFrameDisplayTraining,
                                                    m_pCurrentFrameImageMatrixHSV );
                    break;
                case OBJECT_TRACKING_UN_INITIALIZED:
                    {    //try to initialize
                        Matrixf * pGroundTruthMatrix = NULL; 
                        if ( m_cameraTrackingParametersPtr-> m_calculateTrackingError ) 
                        {
                            pGroundTruthMatrix = &m_groundTruthMatrixList[objInd];
                        }

                        if( m_objectPtrList[objInd]->InitializeObjectTracker(   m_pCurrentFrameImageMatrixColor,
                                                                                m_pCurrentFrameImageMatrixGray,
                                                                                frameInd, 
                                                                                m_videoMatrix.size(),
                                                                                m_pFrameDisplay, 
                                                                                m_pFrameDisplayTraining,
                                                                                m_pCurrentFrameImageMatrixHSV,
                                                                                pGroundTruthMatrix) )
                        {
                            m_objectStatusList[objInd] = OBJECT_TRACKING_IN_PROGRESS;
                        }
                    }
                    break;
                default:
                    abortError( __LINE__, __FILE__, "Unsupported tracking status");
                }                
            }

            //if no fusion at all, display/save all result if necessary
            if( m_cameraTrackingParametersPtr->m_geometricFusionType == NO_GEOMETRIC_FUSION && m_cameraTrackingParametersPtr->m_appearanceFusionType == NO_APPEARANCE_FUSION )
            {
                DisplayAndSaveTrackedFrame();
                DisplayAndSaveTrainingSamples();

                if( ( m_cameraTrackingParametersPtr->m_trackerFeatureType == Features::HAAR_LIKE 
                    || m_cameraTrackingParametersPtr->m_trackerFeatureType == Features::HAAR_COLOR_HISTOGRAM )  
                    && m_pCurrentFrameImageMatrixGray->isInitII() )
                { 
                    //free the integral image
                    m_pCurrentFrameImageMatrixGray->FreeII();
                }
            }
            //if fusion is enabled, frame display will be called inside StoreAllParticleFilterTrackerState
            //if fusion is enabled, sample display will be called inside LearnLocalAppearanceModel
        }

        EXCEPTION_CATCH_AND_ABORT( "Error in Camera.TrackCameraFrame()" )
    }
    
    /********************************************************************
    PrepareCurrentFrameForTracking
        Prepare current frame for tracking.
    Exceptions:
        None
    *********************************************************************/
    void Camera::PrepareCurrentFrameForTracking( int frameInd )
    {
        m_pFrameDisplayTraining=NULL;
        m_pFrameDisplay = NULL;
        //create a frame for drawing (for either display or video saving)
        if( m_displayTrackedVideo || m_saveTrackedVideo )
        {
            if( m_sourceIsColorImage )
            {
                ASSERT_TRUE( m_videoMatrix[frameInd].depth( ) == 3);
                m_frameDisplay = m_videoMatrix[frameInd];
            }
            else
            {
                m_videoMatrix[frameInd].conv2RGB(m_frameDisplay);
            }
            m_frameDisplay.createIpl();
            m_frameDisplay._keepIpl = true;
            m_pFrameDisplay = & m_frameDisplay;

            //draw the frame number on the image
            m_frameDisplay.drawText(("#"+int2str(frameInd,3)).c_str(),1,25,255,255,0);
        }

        //create a frame for drawing training samples (for either display or video saving)
        if( m_displayTrainingSamples || m_saveVideoTrainingExamples )
        {
            if( m_sourceIsColorImage )
            {
                ASSERT_TRUE( m_videoMatrix[frameInd].depth( ) == 3);
                m_frameDisplayTraining = m_videoMatrix[frameInd];
            }
            else
            {
                m_videoMatrix[frameInd].conv2RGB(m_frameDisplayTraining);
            }
            m_frameDisplayTraining.createIpl();
            m_frameDisplayTraining._keepIpl = true;
            m_pFrameDisplayTraining = & m_frameDisplayTraining;

            //draw the frame number on the image
            m_frameDisplayTraining.drawText(("#"+int2str(frameInd,3)).c_str(),1,25,255,255,0);
        }

        //convert the current frame to gray/color/hsv if necessary
        if( m_sourceIsColorImage )
        {
            ASSERT_TRUE(m_videoMatrix[frameInd].depth() == 3);

            m_pCurrentFrameImageMatrixColor    = & (m_videoMatrix[frameInd]);        

            if (  m_cameraTrackingParametersPtr->m_trackerFeatureType == Features::HAAR_LIKE 
                || m_cameraTrackingParametersPtr->m_trackerFeatureType == Features::HAAR_COLOR_HISTOGRAM )
            {  
                //create a temporary gray image
                m_videoMatrix[frameInd].conv2BW( m_grayScaleImageMatrix );
                m_pCurrentFrameImageMatrixGray  = & m_grayScaleImageMatrix;
            }
            else
            {    
                m_pCurrentFrameImageMatrixGray = NULL;
            }
        }
        else
        {
            m_pCurrentFrameImageMatrixGray    =  & (m_videoMatrix[frameInd]);
            m_pCurrentFrameImageMatrixColor = NULL;
        }

        if( m_cameraTrackingParametersPtr->m_HSVRequired )
        {
            ASSERT_TRUE ( m_pCurrentFrameImageMatrixColor != NULL );
            m_pCurrentFrameImageMatrixColor->conv2HSV( m_HSVImageMatrix );
            m_pCurrentFrameImageMatrixHSV = &m_HSVImageMatrix;
        }        
        else
        {
            m_pCurrentFrameImageMatrixHSV = NULL;
        }

        if (  m_cameraTrackingParametersPtr->m_trackerFeatureType == Features::HAAR_LIKE 
            || m_cameraTrackingParametersPtr->m_trackerFeatureType == Features::HAAR_COLOR_HISTOGRAM )
        {  
            //initialize integral image
            if( !m_pCurrentFrameImageMatrixGray->isInitII() )
            {
                m_pCurrentFrameImageMatrixGray->initII();    // Initialize Image Integral     
                ASSERT_TRUE( m_pCurrentFrameImageMatrixGray->isInitII() );
            }                
        }
    }

    /********************************************************************
    DisplayAndSaveTrackedFrame
        display and save the output video frame if required
    Exceptions:
        None
    *********************************************************************/
    void    Camera::DisplayAndSaveTrackedFrame( )
    {
        if( m_displayTrackedVideo )
        {    // display the tracked frame on screen
            m_frameDisplay.display( m_cameraID );
            cvWaitKey(1);
        }

        if( m_saveTrackedVideo )
        {    //save the tracked frame to video file
            Matrixu::WriteFrame( m_pVideoWriter, m_frameDisplay );
        }

        if( m_displayTrackedVideo || m_saveTrackedVideo)
        {    
            m_frameDisplay._keepIpl = false;
            m_frameDisplay.freeIpl();
        }
    }

    /********************************************************************
    DisplayAndSaveTrainingSamples
        display and save a video frame with training samples drawn if required
    Exceptions:
        None
    *********************************************************************/
    void Camera::DisplayAndSaveTrainingSamples()
    {
        if( m_displayTrainingSamples )
        {    // display the tracked frame on screen
            m_frameDisplayTraining.display(
                ("TrainingExamples"+int2str(m_cameraID,3)).c_str()
                );
            cvWaitKey(1);
        }
        if( m_saveVideoTrainingExamples )
        {
            Matrixu::WriteFrame( m_pVideoWriterTraining, m_frameDisplayTraining );
        }
        if( m_displayTrainingSamples || m_saveVideoTrainingExamples)
        {
            m_frameDisplayTraining._keepIpl = false;
            m_frameDisplayTraining.freeIpl();
        }
    }
    /********************************************************************
    StoreAllParticleFilterTrackerState
        For all objects, 
        Store the state of current particle filter tracker, updating display 
        if required
    Exceptions:
        None
    *********************************************************************/
    void Camera::StoreAllParticleFilterTrackerState( int frameInd )
    {
        try
        {
            for (int objectInd = 0; objectInd < m_objectPtrList.size(); objectInd++)
            {
                if( m_displayTrackedVideo || m_saveTrackedVideo )
                {
                    m_objectPtrList[objectInd]->StoreParticleFilterTrackerState( frameInd, &m_frameDisplay );
                    DisplayAndSaveTrackedFrame( );
                }
                else
                {
                    m_objectPtrList[objectInd]->StoreParticleFilterTrackerState( frameInd );
                }                
            }            
        }
        EXCEPTION_CATCH_AND_ABORT( "Error in Camera.StoreParticleFilterTrackerState" ); 
    }

    /********************************************************************
    UpdateParticleWeightsWithGroundPDF
        Update particle weights based on the ground plane kalman filter's pdf
    Exceptions:
        None
    *********************************************************************/
    void Camera::UpdateParticleWeightsWithGroundPDF( CvMat*  pMeanMatrix, CvMat* pCovarianceMatrix, int objectIndex )
    {
        try
        {
            ASSERT_TRUE( objectIndex < m_objectPtrList.size() );
            ASSERT_TRUE( pMeanMatrix != NULL );
            ASSERT_TRUE( pCovarianceMatrix != NULL );

            m_objectPtrList[objectIndex]->UpdateParticleWeightsWithGroundPDF(    pMeanMatrix,
                                                                                pCovarianceMatrix,
                                                                                m_pCurrentFrameImageMatrixColor,
                                                                                m_pCurrentFrameImageMatrixGray,
                                                                                m_pCurrentFrameImageMatrixHSV );
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to update particle weights with ground plane pdf" );
    }

    /********************************************************************
    LearnLocalAppearanceModel
        For all objectsUpdate the appearance model of current particle filter tracker
    Exceptions:
        None
    *********************************************************************/
    void Camera::LearnLocalAppearanceModel( int frameInd )
    {
        try
        {
            //PrepareCurrentFrameForTracking( frameInd );

            for (int objectInd = 0; objectInd < m_objectPtrList.size(); objectInd++)
            {
                m_objectPtrList[objectInd]->UpdateParticleFilterTrackerAppearanceModel(
                    m_pCurrentFrameImageMatrixColor, m_pCurrentFrameImageMatrixGray,
                    m_pFrameDisplayTraining, m_pCurrentFrameImageMatrixHSV );

                if( m_displayTrainingSamples )
                {
                    DisplayAndSaveTrainingSamples();
                }
            }

            if( ( m_cameraTrackingParametersPtr->m_trackerFeatureType == Features::HAAR_LIKE 
                || m_cameraTrackingParametersPtr->m_trackerFeatureType == Features::HAAR_COLOR_HISTOGRAM )  
                && m_pCurrentFrameImageMatrixGray->isInitII() )
            { 
                //free the integral image as they are no longer needed.
                m_pCurrentFrameImageMatrixGray->FreeII();
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Error in Camera.UpdateParticleFilterTrackerAppearanceModel" ); 
    }

    /********************************************************************
    SaveStatesAllFrames
        Saves the state.
    Exceptions:
        None
    *********************************************************************/
    void Camera::SaveStatesAllFrames( )
    {
        for(int objectInd = 0; objectInd< m_objectPtrList.size(); objectInd++)
        {
            m_objectPtrList[objectInd]->SaveObjectStatesAllFrames();
        }
        
        // clean up
        if ( m_pVideoWriter != NULL )
        {
            //release the videoList writer
            cvReleaseVideoWriter( &m_pVideoWriter );
        }

        if( m_pVideoWriterTraining != NULL)
        {
            //release the videoList writer
            cvReleaseVideoWriter( &m_pVideoWriterTraining );
        }
    }

    /********************************************************************
    GetReprojectedParticlesForGeometricFusion
        Project particles from the camera's image plane to the global 
        ground plane using the predefined homography.
    Exceptions:
        None
    *********************************************************************/
    CvMat* Camera::GetReprojectedParticlesForGeometricFusion( const int objectInd )
    {
        try
        {
            return GeometryBasedInformationFuser::TransformWithHomography( m_objectPtrList[objectInd]->GetParticlesFootPositionOnImagePlaneForGeometricFusion( ), m_pHomographyMatrix );
        }
        EXCEPTION_CATCH_AND_ABORT( "Error in reprojecting particles to the ground plane" );
    }

    /********************************************************************
    GetAverageImageParticleForGeometricFusion
        Get average reprojected particle for geometric fusion
    Exceptions:
        None
    *********************************************************************/
    CvMat* Camera::GetAverageImageParticleForGeometricFusion( const int objectInd )
    {
        try
        {
            return m_objectPtrList[objectInd]->GetAverageParticleFootPositionOnImagePlaneForGeometricFusion( );
        }
        EXCEPTION_CATCH_AND_ABORT( "Error in reprojecting particles to the ground plane" );
    }

    /********************************************************************
    LearnGlobalAppearanceModel
        Learn Multi Camera Appearance Model
    Exceptions:
        None
    *********************************************************************/
    void    Camera::LearnGlobalAppearanceModel( CameraNetworkBasePtr cameraNetworkBasePtr )
    {
        try
        {
            ASSERT_TRUE( !m_objectPtrList.empty() );
            ASSERT_TRUE( cameraNetworkBasePtr != NULL );

            for (int objInd = 0; objInd < m_objectPtrList.size(); objInd++)
            {
                ASSERT_TRUE( m_objectPtrList[objInd] != NULL );

                m_objectPtrList[objInd]->LearnGlobalAppearanceModel( cameraNetworkBasePtr );
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to Learn MultiCameraAppearance Model" );
    }

    /********************************************************************
    DrawAllObjectFootPoints
        DrawAllObjectFootPoints
    Exceptions:
        None
    *********************************************************************/
    void    Camera::DrawAllObjectFootPoints( )
    {
        try
        {
            if( m_pFrameDisplay != NULL )
            {
                for (int objInd = 0; objInd < m_objectPtrList.size(); objInd++)
                {
                    ASSERT_TRUE( m_objectPtrList[objInd] != NULL );

                    m_objectPtrList[objInd]->DrawObjectFootPoint( m_pFrameDisplay );
                }
            }            
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to Learn MultiCameraAppearance Model" );
    }
}