#include "CameraNetwork.h"

#include <math.h>

namespace MultipleCameraTracking
{
    /**********************************************************************
    Constructor
        Constructs a Camera network and initializes all the cameras.
        Also, initializes geometric fusion(if enabled).         
    Exceptions:
        None
    **********************************************************************/
    CameraNetwork::CameraNetwork(    const vectori    cameraIdList, 
                                    const vectori    objectIdList,                        
                                    CameraTrackingParametersPtr    cameraTrackingParametersPtr )
        : m_numberOfCameras( cameraIdList.size() ),
        m_numberOfObjects( objectIdList.size() ),
        m_cameraIdList( cameraIdList ),
        m_objectIdList( objectIdList ),
        m_videoWriterListKFDistribution( m_numberOfObjects, NULL ), 
        m_videoWriterListGroundParticlesAfterFusion ( m_numberOfObjects, NULL ), 
        m_videoWriterListGroundParticles( m_numberOfObjects, NULL ),
        m_cameraTrackingParametersPtr ( ASSERT_PRECONDITION_PARAMETER( cameraTrackingParametersPtr != NULL, cameraTrackingParametersPtr ) )
    {
        try
        {
            InitializeCameraNetwork( );

            //Initialize ground plane fusion if it's enabled
            if ( m_cameraTrackingParametersPtr->m_geometricFusionType == FUSION_GROUND_PLANE_GMM_KF)
            {
                InitializeGeometricFuser( );
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to Construct a Camera Network" );
    }

    /**********************************************************************
    ~CameraNetwork
    Exceptions:
        None
    **********************************************************************/
    CameraNetwork::~CameraNetwork( )
    {
        try
        {
            if ( m_cameraTrackingParametersPtr->m_geometricFusionType == FUSION_GROUND_PLANE_GMM_KF )
            { 
                //release the ground plane particle matrix
                for ( int objectInd = 0; objectInd < m_numberOfObjects; objectInd++ )
                {
                    cvReleaseMat( &(m_groundPlaneParticlesPtrList[objectInd]) );
                }

                //release the video writers
                if ( m_cameraTrackingParametersPtr->m_saveGroundParticlesImage )
                {
                    for ( int objectInd = 0; objectInd < m_numberOfObjects; objectInd++ )
                    {
                        if ( m_videoWriterListGroundParticles[objectInd] != NULL)    
                        {
                            cvReleaseVideoWriter( &(m_videoWriterListGroundParticles[objectInd]) );
                        }
                        if ( m_videoWriterListGroundParticlesAfterFusion[objectInd] != NULL )
                        {
                            cvReleaseVideoWriter( &(m_videoWriterListGroundParticlesAfterFusion[objectInd]) );
                        }
                    }
                }

                //release video writer
                if ( m_cameraTrackingParametersPtr->m_saveGroundPlaneKFImage )
                {
                    for ( int objectInd = 0; objectInd < m_numberOfObjects; objectInd++ )
                    {
                        if( m_videoWriterListKFDistribution[objectInd] != NULL )
                        {
                            cvReleaseVideoWriter( &(m_videoWriterListKFDistribution[objectInd]) );
                        }
                    }
                }
            }
        }
        EXCEPTION_CATCH_AND_LOG( "failed to destroy the CameraNetwork" )
    }

    /**********************************************************************
    FeedbackInformationFromGeometricFusion
        Feedback information from geometric fusion
    Exceptions:
        None
    **********************************************************************/
    void CameraNetwork::FeedbackInformationFromGeometricFusion( int frameIndex )
    {
        try
        {            
            // feed back to local visual tracking by re-weighting particles.
            #pragma omp parallel for
            for ( int cameraInd = 0; cameraInd < m_numberOfCameras; cameraInd++)
            {    
                vectorf cameraUpdatedWeights(m_cameraTrackingParametersPtr->m_numberOfParticles);
                for (int objectInd = 0; objectInd < m_numberOfObjects; objectInd++)
                {            
                    std::ofstream outputFileStreaam;
                    outputFileStreaam    << "Geometric fusion: Feedback fused ground plane information to camera " 
                                        << int2str( m_cameraPtrList[cameraInd]->GetCameraID()+1, 3 ) 
                                        << " for object " << m_objectIdList[objectInd] 
                                        << endl;

                    LOG( outputFileStreaam );

                    CvMat* pMeanMatrix = m_geometricInformationFuserPtrList[objectInd]->GetGroundPlaneKalmanMeanMatrix(frameIndex);
                    CvMat* pCovarianceMatrix =  m_geometricInformationFuserPtrList[objectInd]->GetGroundPlaneKalmanCovarianceMatrix();

                    m_cameraPtrList[cameraInd]->UpdateParticleWeightsWithGroundPDF( pMeanMatrix, pCovarianceMatrix, objectInd );

                    cvReleaseMat(&pMeanMatrix);     
                    cvReleaseMat(&pCovarianceMatrix);     
                }
            }

            if ( m_geometricInformationFuserPtrList[0]->GetGroundPlaneMeasurementType() == GeometryBasedInformationFuser::PRINCIPAL_AXIS_INTERSECTION )
            {
                return;
            }

            // display the re-weighted particles
            //collecting ground plane particle from each camera for each object
            for (int objectInd = 0; objectInd < m_numberOfObjects; objectInd++)
            {
                #pragma omp parallel for
                for ( int cameraInd=0; cameraInd < m_numberOfCameras; cameraInd++)
                {
                    for (int objectInd = 0; objectInd < m_numberOfObjects; objectInd++)
                    {                
                        CvMat* pGroundPlaneParticleMatrix = m_cameraPtrList[cameraInd]->GetReprojectedParticlesForGeometricFusion( objectInd );
                        GeometryBasedInformationFuser::CopyMatrix(    pGroundPlaneParticleMatrix,
                                                                    m_groundPlaneParticlesPtrList[objectInd], 
                                                                    0,         
                                                                    cameraInd * m_cameraTrackingParametersPtr->m_numberOfParticles ,
                                                                    2  );
                        cvReleaseMat( &pGroundPlaneParticleMatrix );
                    }
                }

                if( m_cameraTrackingParametersPtr->m_displayGroundParticlesImage || m_cameraTrackingParametersPtr->m_saveGroundParticlesImage ) 
                {
                    Matrixu* pMatrixGroundParticles =    m_geometricInformationFuserPtrList[objectInd]->DisplayOriginalGroundParticles( m_groundPlaneParticlesPtrList[objectInd], frameIndex );

                    if( m_cameraTrackingParametersPtr->m_displayGroundParticlesImage )
                    {
                        pMatrixGroundParticles->display( ("Ground Particles After feedback for Object"+int2str( m_objectIdList[objectInd], 3 )).c_str(), 1 );
                        cvWaitKey(1);
                    }

                    if( m_cameraTrackingParametersPtr->m_saveGroundParticlesImage )
                    {
                        Matrixu::WriteFrame( m_videoWriterListGroundParticlesAfterFusion[objectInd], *pMatrixGroundParticles );
                    }

                }
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to Feedback Information From Geometric Fusion" );
    }

    /**********************************************************************
    FuseGeometrically
        Fuse information geometrically.
    Exceptions:
        None
    **********************************************************************/
    void CameraNetwork::FuseGeometrically( int frameIndex )
    {
        try
        {
            LOG( "Geometric fusion: collecting potential ground plane Information from cameras for all object"  << endl );        
            
            //collecting ground plane particle from each camera for each object
            #pragma omp parallel for
            for ( int cameraInd=0; cameraInd < m_numberOfCameras; cameraInd++)
            {
                for (int objectInd = 0; objectInd < m_numberOfObjects; objectInd++)
                {                
                    if ( m_geometricInformationFuserPtrList[objectInd]->GetGroundPlaneMeasurementType() == GeometryBasedInformationFuser::PRINCIPAL_AXIS_INTERSECTION )
                    {
                        m_groundPlaneParticlesPtrList[objectInd] = cvCreateMat(  m_numberOfCameras, 2, CV_32FC1 );

                        for ( int cameraInd=0; cameraInd < m_numberOfCameras; cameraInd++ )
                        {
                            CvMat* pGroundPlaneParticleMatrix = m_cameraPtrList[cameraInd]->GetAverageImageParticleForGeometricFusion( objectInd );
                            MultipleCameraTracking::GeometryBasedInformationFuser::CopyMatrix(  pGroundPlaneParticleMatrix,
                                                                                                m_groundPlaneParticlesPtrList[objectInd],
                                                                                                0,
                                                                                                cameraInd,
                                                                                                2 /*option*/ );
                        }
                    }
                    else
                    {
                        m_groundPlaneParticlesPtrList[objectInd] = cvCreateMat( m_cameraTrackingParametersPtr->m_numberOfParticles *  m_numberOfCameras, 2, CV_32FC1 );

                        for ( int cameraInd=0; cameraInd < m_numberOfCameras; cameraInd++ )
                        {
                            CvMat* pGroundPlaneParticleMatrix = m_cameraPtrList[cameraInd]->GetReprojectedParticlesForGeometricFusion( objectInd );
                            MultipleCameraTracking::GeometryBasedInformationFuser::CopyMatrix(  pGroundPlaneParticleMatrix,
                                m_groundPlaneParticlesPtrList[objectInd],
                                0,
                                cameraInd * m_cameraTrackingParametersPtr->m_numberOfParticles,
                                2 /*option*/ );
                            cvReleaseMat(&pGroundPlaneParticleMatrix);
                        }
                    }
                }
            }

            //performing fusion for each object
            for (int objectInd = 0; objectInd < m_numberOfObjects; objectInd++)
            {
                LOG( "Geometric fusion: start for object " << m_objectIdList[objectInd] << endl );

                #ifdef ENABLE_PARTICLE_LOG
                for ( int i = 0; i < m_groundPlaneParticlesPtrList[objectInd]->rows; i++ )
                {
                    MultipleCameraTracking::g_logFile << "Particle before fusion: " << (i+1) << "[" 
                        << m_groundPlaneParticlesPtrList[objectInd]->data.fl[i*2] << ","
                        << m_groundPlaneParticlesPtrList[objectInd]->data.fl[i*2+1] << "]" << endl;
                }
                #endif

                //Information fuser will re-weight all the particles.
                m_geometricInformationFuserPtrList[objectInd]->FuseInformation( m_groundPlaneParticlesPtrList[objectInd] );

                if( m_cameraTrackingParametersPtr->m_displayGroundPlaneKFImage || m_cameraTrackingParametersPtr->m_saveGroundPlaneKFImage )
                {
                    CvMat* pMeanMatrix = m_geometricInformationFuserPtrList[objectInd]->GetGroundPlaneKalmanMeanMatrix(frameIndex);
                    CvMat* pCovarianceMatrix =  m_geometricInformationFuserPtrList[objectInd]->GetGroundPlaneKalmanCovarianceMatrix();

                    Matrixu* pMatrixKFImage = m_geometricInformationFuserPtrList[objectInd]->DisplayKalmanFilterPdf( pMeanMatrix, pCovarianceMatrix, frameIndex );

                    if( m_cameraTrackingParametersPtr->m_displayGroundPlaneKFImage )
                    {
                        pMatrixKFImage->display(("Kalman Filter for Object"+int2str( m_objectIdList[objectInd], 3 )).c_str(), 1 );
                        cvWaitKey(1);
                    }

                    if( m_cameraTrackingParametersPtr->m_saveGroundPlaneKFImage )
                    {
                        Matrixu::WriteFrame( m_videoWriterListKFDistribution[objectInd], *pMatrixKFImage );
                    }

                    cvReleaseMat(&pMeanMatrix);     
                    cvReleaseMat(&pCovarianceMatrix);     
                }

                if( m_cameraTrackingParametersPtr->m_displayGMMCenters )
                {
                    Matrixu* pMatrixGMMImage = m_geometricInformationFuserPtrList[objectInd]->
                        DisplayGMMGroundParticles( frameIndex, m_groundPlaneParticlesPtrList[objectInd] );

                    pMatrixGMMImage->display(("GMM Centers for Object"+int2str( m_objectIdList[objectInd], 3 )).c_str(), 1 );
                    cvWaitKey(1);
                }
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to fuse information geometrically" );
    }

    /**********************************************************************
    GenerateTrainingSampleSetsForAppearanceFusion
        Generate Training SampleSets For AppearanceFusion
    Exceptions:
        None
    **********************************************************************/
    void CameraNetwork::GenerateTrainingSampleSetsForAppearanceFusion(    const int                objectId,
                                                                        Classifier::SampleSet&    positiveSampleSet,
                                                                        Classifier::SampleSet&  negativeSampleSet )
    {
         try
         {
             for ( int cameraIndex = 0; cameraIndex < m_numberOfCameras; cameraIndex++ )
             {
                 ASSERT_TRUE( m_cameraPtrList[cameraIndex] != NULL );

                 ObjectPtr objectPtr = m_cameraPtrList[cameraIndex]->GetObjectPtr( objectId );

                 ASSERT_TRUE( objectPtr != NULL );

                 Classifier::SampleSet objectPositiveSampleSet;
                 Classifier::SampleSet objectNegativeSampleSet;

                 LOG( endl << "Generating Appearance Fusion Training Sets for Camera:" << cameraIndex << "," 
                     "objectID:" <<  objectId << endl );

                 objectPtr->GenerateTrainingSampleSetsForAppearanceFusion(     objectPositiveSampleSet,
                                                                            objectNegativeSampleSet,
                                                                            m_cameraPtrList[cameraIndex]->GetColorImageMatrix(),
                                                                            m_cameraPtrList[cameraIndex]->GetGrayImageMatrix(),
                                                                            m_cameraPtrList[cameraIndex]->GetHSVImageMatrix() );

                 int cameraId = m_cameraPtrList[cameraIndex]->GetCameraID();

                 for ( int positiveSampleIndex = 0; positiveSampleIndex < objectPositiveSampleSet.Size(); positiveSampleIndex++ )
                 {
                     objectPositiveSampleSet[positiveSampleIndex].m_cameraID = cameraId;
                     positiveSampleSet.PushBackSample( objectPositiveSampleSet[positiveSampleIndex] );
                 }

                 for ( int negativeSampleIndex = 0; negativeSampleIndex < objectNegativeSampleSet.Size(); negativeSampleIndex++ )
                 {
                     negativeSampleSet.PushBackSample( objectNegativeSampleSet[negativeSampleIndex] );
                 }
             }

             ASSERT_TRUE( positiveSampleSet.Size() > 0 );
             ASSERT_TRUE( negativeSampleSet.Size() > 0 );
         }
         EXCEPTION_CATCH_AND_ABORT( "failed to generate tarining samples for appearance fusion.");
     }

    /**********************************************************************
    GetHomographyMatrixList
        Get homography matrix list
    Exceptions:
        None
    **********************************************************************/
    vector<CvMat*> CameraNetwork::GetHomographyMatrixList( )
    {
        try
        {
            vector<CvMat*> homographyList;
            for ( int cameraIndex = 0; cameraIndex < m_cameraPtrList.size(); cameraIndex++ )
            {
                ASSERT_TRUE( m_cameraPtrList[cameraIndex] != NULL );
                homographyList.push_back( m_cameraPtrList[cameraIndex]->GetHomographyMatrix() );
            }
            ASSERT_TRUE( homographyList.size() == m_cameraPtrList.size() );

            return homographyList;
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to homography matrix list" );
    }

    /**********************************************************************
    InitializeCameraNetwork
        Initializes CameraNetwork and Setups the Cameras.
    Exceptions:
        None
    **********************************************************************/
    void CameraNetwork::InitializeCameraNetwork( )
    {
        try
        {
            m_groundPlaneParticlesPtrList.resize( m_numberOfObjects );

            m_geometricInformationFuserPtrList.resize( m_numberOfObjects );

            for ( int cameraInd = 0; cameraInd < m_numberOfCameras; cameraInd++ )
            {
                CameraPtr cameraPtr( new Camera( m_cameraIdList[cameraInd], 
                                                 m_objectIdList,
                                                 m_cameraTrackingParametersPtr ));    

                //push back the object into the list
                m_cameraPtrList.push_back( cameraPtr );

                if ( m_cameraPtrList[cameraInd] == NULL )
                {
                    abortError(__LINE__,__FILE__,"Camera Object is NULL.");
                }

                LOG( "Initializing Camera    :" << int2str( m_cameraPtrList[cameraInd]->GetCameraID( ), 3 ) << endl );
                
                //Start camera tracking session 
                m_cameraPtrList[cameraInd]->InitializeCameraTrackers( );
            }

            //the cameraList shouldn't be empty
            ASSERT_TRUE( !m_cameraPtrList.empty( ) );
        }
        EXCEPTION_CATCH_AND_ABORT( "Error while initializing the camera network" );
    }

    /**********************************************************************
    InitializeGeometricFuser
        Initializes Geometric Fusion.
    Exceptions:
        None
    **********************************************************************/
    void CameraNetwork::InitializeGeometricFuser( )
    {
        try
        {
            //particle filter must be enabled to allow ground plane fusion
            ASSERT_TRUE( m_cameraTrackingParametersPtr->m_localObjectTrackerType == PARTICLE_FILTER_TRACKER );

            GeometryBasedInformationFuser::GroundPlaneMeasurmentType groundPlaneMeasurementType = GeometryBasedInformationFuser::PRINCIPAL_AXIS_INTERSECTION;
            

            for ( int objectInd = 0; objectInd < m_numberOfObjects; objectInd++ )
            {
                if ( groundPlaneMeasurementType == GeometryBasedInformationFuser::PRINCIPAL_AXIS_INTERSECTION )
                {
                    m_groundPlaneParticlesPtrList[objectInd] = cvCreateMat(  m_numberOfCameras, 2, CV_32FC1 );

                    for ( int cameraInd=0; cameraInd < m_numberOfCameras; cameraInd++ )
                    {
                        //Get average image plane particle foot position
                        CvMat* pGroundPlaneParticleMatrix = m_cameraPtrList[cameraInd]->GetAverageImageParticleForGeometricFusion( objectInd );
                        
                        MultipleCameraTracking::GeometryBasedInformationFuser::CopyMatrix(  pGroundPlaneParticleMatrix,
                                                                                            m_groundPlaneParticlesPtrList[objectInd],
                                                                                            0,
                                                                                            cameraInd,
                                                                                            2 /*option*/ );
                    }
                }
                else
                {
                    m_groundPlaneParticlesPtrList[objectInd] = cvCreateMat( m_cameraTrackingParametersPtr->m_numberOfParticles *  m_numberOfCameras, 2, CV_32FC1 );
                    
                    for ( int cameraInd=0; cameraInd < m_numberOfCameras; cameraInd++ )
                    {
                        //Get reprojected particle foot position on ground plane
                        CvMat* pGroundPlaneParticleMatrix = m_cameraPtrList[cameraInd]->GetReprojectedParticlesForGeometricFusion( objectInd );
                        
                        MultipleCameraTracking::GeometryBasedInformationFuser::CopyMatrix(  pGroundPlaneParticleMatrix,
                                                                                            m_groundPlaneParticlesPtrList[objectInd],
                                                                                            0,
                                                                                            cameraInd * m_cameraTrackingParametersPtr->m_numberOfParticles,
                                                                                            2 /*option*/ );
                        cvReleaseMat(&pGroundPlaneParticleMatrix);
                    }
                }

                LOG( "\nInitializing GeometryBasedInformationFuser, for object " << m_objectIdList[objectInd] << endl );
                #ifdef ENABLE_PARTICLE_LOG
                    for ( int i = 0; i < m_groundPlaneParticlesPtrList[objectInd]->rows; i++ )
                    {    
                        //log all ground plane particle states for debug
                        g_logFile    << "Particle: " << (i+1) 
                                    << "[" << m_groundPlaneParticlesPtrList[objectInd]->data.fl[i*2] << ","
                                    << m_groundPlaneParticlesPtrList[objectInd]->data.fl[i*2+1] << "]" << endl;
                    }
                #endif

                // Create an information fusion object - uses ground plane particles to initialize the parameters
                // Note: numberOfClusters = numOfCameras
                m_geometricInformationFuserPtrList[objectInd] = MultipleCameraTracking::GeometryBasedInformationFuserPtr
                            ( new MultipleCameraTracking::GeometryBasedInformationFuser(    m_numberOfCameras,
                                                                                            m_groundPlaneParticlesPtrList[objectInd],
                                                                                            groundPlaneMeasurementType,
                                                                                            GetHomographyMatrixList( ) ) );    

                if ( m_cameraTrackingParametersPtr->m_displayGroundParticlesImage || m_cameraTrackingParametersPtr->m_saveGroundParticlesImage ) 
                {
                    Matrixu* pMatrixGroundParticles =    m_geometricInformationFuserPtrList[objectInd]->DisplayOriginalGroundParticles( m_groundPlaneParticlesPtrList[objectInd], 0 );

                    if( m_cameraTrackingParametersPtr->m_saveGroundParticlesImage )
                    {
                        string m_videoGeoFusionGroundParticles = string( m_cameraTrackingParametersPtr->m_outputDirectoryString );
    
                        m_videoGeoFusionGroundParticles    =    m_cameraTrackingParametersPtr->m_outputDirectoryString +
                            m_cameraTrackingParametersPtr->m_dataNameString + "/" +
                            m_cameraTrackingParametersPtr->m_nameInitilizationString +"/"+ 
                            "TR" + int2str(m_cameraTrackingParametersPtr->m_trialNumber,3) +
                            "_Obj" + int2str( m_objectIdList[objectInd], 3 )+ "_GP2.avi";
                        
                        m_videoWriterListGroundParticlesAfterFusion[objectInd] = cvCreateVideoWriter( 
                            m_videoGeoFusionGroundParticles.c_str(), 
                            CV_FOURCC('x','v','i','d'),
                            15, 
                            cvSize(pMatrixGroundParticles[0].cols(), pMatrixGroundParticles[0].rows() ),
                            3 );                
                    }
                }

                if ( m_cameraTrackingParametersPtr->m_displayGroundPlaneKFImage || m_cameraTrackingParametersPtr->m_saveGroundPlaneKFImage )
                {
                    CvMat* pMeanMatrix = m_geometricInformationFuserPtrList[objectInd]->GetGroundPlaneKalmanMeanMatrix(0);
                    CvMat* pCovarianceMatrix =  m_geometricInformationFuserPtrList[objectInd]->GetGroundPlaneKalmanCovarianceMatrix();

                    Matrixu* pMatrixKFImage = m_geometricInformationFuserPtrList[objectInd]->DisplayKalmanFilterPdf( pMeanMatrix, pCovarianceMatrix, 0 );

                    if( m_cameraTrackingParametersPtr->m_displayGroundPlaneKFImage )
                    {
                        pMatrixKFImage->display(("Kalman Filter for Object"+int2str( m_objectIdList[objectInd], 3 )).c_str(), 1 );
                        cvWaitKey(1);
                    }

                    if( m_cameraTrackingParametersPtr->m_saveGroundPlaneKFImage )
                    {
                        string m_videoGeoFusionKF = string( m_cameraTrackingParametersPtr->m_outputDirectoryString );

                        m_videoGeoFusionKF    =    m_videoGeoFusionKF +
                                                m_cameraTrackingParametersPtr->m_dataNameString + "/" +
                                                m_cameraTrackingParametersPtr->m_nameInitilizationString +"/"+ 
                                                "TR" + int2str(m_cameraTrackingParametersPtr->m_trialNumber,3) +
                                                "_Obj" + int2str( m_objectIdList[objectInd], 3 )+ "_.avi";

                        m_videoWriterListKFDistribution[objectInd] = cvCreateVideoWriter( 
                                                                    m_videoGeoFusionKF.c_str(), 
                                                                    CV_FOURCC('x','v','i','d'),
                                                                    15, 
                                                                    cvSize(pMatrixKFImage[0].cols(), pMatrixKFImage[0].rows() ),
                                                                    3 );

                        Matrixu::WriteFrame( m_videoWriterListKFDistribution[objectInd], *pMatrixKFImage );
                    }

                    cvReleaseMat(&pMeanMatrix);     
                    cvReleaseMat(&pCovarianceMatrix);     
                }

                if ( m_cameraTrackingParametersPtr->m_displayGMMCenters &&
                    m_geometricInformationFuserPtrList[objectInd]->GetGroundPlaneMeasurementType() 
                    != GeometryBasedInformationFuser::PRINCIPAL_AXIS_INTERSECTION )
                {
                    Matrixu* pMatrixGMMImage = m_geometricInformationFuserPtrList[objectInd]->
                        DisplayGMMGroundParticles( 0, m_groundPlaneParticlesPtrList[objectInd] );

                    pMatrixGMMImage->display(("GMM Centers for Object"+int2str( m_objectIdList[objectInd], 3 )).c_str(), 1 );
                    cvWaitKey(1);
                }
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to setup the geometric fuser");
    }

    /**********************************************************************
    TrackObjectsOnCurrentFrame
        Tracks all objects in different views for a given frame.
    Exceptions:
        None
    **********************************************************************/
    void CameraNetwork::TrackObjectsOnCurrentFrame( const int frameIndex )
    {
        try
        {
            ASSERT_TRUE( frameIndex >= 0 );

            // logging progress
            LOG( endl << endl << "****** Frame index: " << int2str( frameIndex, 3 ) << " ******" << endl );

            
            // first perform visual tracking at each camera
            #pragma omp parallel for
            for( int cameraInd=0; cameraInd < m_cameraPtrList.size( ); cameraInd++ )
            {
                LOG( "\n ---Visual tracking in Camera: " << int2str( m_cameraPtrList[cameraInd]->GetCameraID(), 3 ) << endl );
                
                m_cameraPtrList[cameraInd]->TrackCameraFrame( frameIndex );
            }

            // if no any fusion scheme, simply stop here
            if ( m_cameraTrackingParametersPtr->m_appearanceFusionType == NO_APPEARANCE_FUSION &&
                m_cameraTrackingParametersPtr->m_geometricFusionType == NO_GEOMETRIC_FUSION )
            { 
                return;
            }

            //if ground plane fusion is enabled, perform fusion and re-weight particle weights.
            if ( m_cameraTrackingParametersPtr->m_geometricFusionType == FUSION_GROUND_PLANE_GMM_KF )
            { 
                //Fuse information geometrically
                FuseGeometrically( frameIndex);
                
                if ( frameIndex > 0 )
                {
                    //Feed back information from geometric fusion
                    FeedbackInformationFromGeometricFusion(frameIndex);

                    //draw the foot position of all objects
                    #pragma omp parallel for
                    for ( int cameraInd=0; cameraInd < m_cameraPtrList.size( ); cameraInd++ )
                    {
                        // save camera's tracking results
                        m_cameraPtrList[cameraInd]->DrawAllObjectFootPoints(  );
                    }
                }
            } //end of ground plane geometric fusion

            //Update local tracker's appearance model
            for ( int cameraInd = 0; cameraInd < m_numberOfCameras; cameraInd++ )
            {
                // update camera's visual tracking appearance model (i.e. update discriminative classifier)
                m_cameraPtrList[cameraInd]->LearnLocalAppearanceModel( frameIndex );
            }    

            // if appearance fusion, do this
            if ( m_cameraTrackingParametersPtr->m_appearanceFusionType != NO_APPEARANCE_FUSION )
            {
                CameraNetworkBasePtr cameraNetworkBasePtr( this ); 
                if ( (frameIndex+1)%m_cameraTrackingParametersPtr->m_appearanceFusionRefreshRate == 0 )
                {
                    for ( int cameraInd = 0; cameraInd < m_numberOfCameras; cameraInd++)
                    {
                        m_cameraPtrList[cameraInd]->LearnGlobalAppearanceModel( cameraNetworkBasePtr );
                    }
                }
            }    

            //save the current status of objects
            #pragma omp parallel for
            for ( int cameraInd=0; cameraInd < m_cameraPtrList.size( ); cameraInd++ )
            {
                // save camera's tracking results
                m_cameraPtrList[cameraInd]->StoreAllParticleFilterTrackerState( frameIndex );
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Camera Network Analysis Failed.")
    }

    /**********************************************************************
    SaveCameraNetworkState
        Save Camera Network State.
    Exceptions:
        None
    **********************************************************************/
    void CameraNetwork::SaveCameraNetworkState( )
    {
        //Saving the states before winding up
        for ( int cameraInd=0; cameraInd < m_numberOfCameras; cameraInd++ )
        {
            try
            {
                ASSERT_TRUE( m_cameraPtrList[cameraInd] != NULL );
                m_cameraPtrList[cameraInd]->SaveStatesAllFrames( );
            }
            EXCEPTION_CATCH_AND_ABORT( "failed to save the states" );
        }
    }
}