#include "GeometryBasedInformationFuser.h"
#include "Config.h"

#define DIMENSION_STATE_VECTOR            4
#define DIMENSION_MEASUREMENT_VECTOR    2
#define DIMENSION_CONTROL_VECTOR        0

namespace MultipleCameraTracking
{
    #define TWO_TIMES_PI 2 * PI

    /********************************************************************
    GeometryBasedInformationFuser
        GeometryBasedInformationFuser fuses geometric information 
        from multiple cameras.
    Exceptions:
        None
    *********************************************************************/
    GeometryBasedInformationFuser::GeometryBasedInformationFuser( const unsigned int            numberOfCamera,
                                                                  CvMat*                        particles,
                                                                  GroundPlaneMeasurmentType        groundPlaneMeasurementType,
                                                                  vector<CvMat*>                homographyMatrixList )
        : m_emModel( ),
        m_pKalman( NULL ),
        m_numberOfClusters( numberOfCamera ),
        m_numberOfCameras( numberOfCamera ),
        m_emParamters( ),
        m_groundPlaneMeasurementType( groundPlaneMeasurementType ),
        m_homographyMatrixList( homographyMatrixList )
    {
        try
        {        
            //Set the Expectation Maximization Parameters
            SetDefaultEMParameters( );

            //Initialize the Global Kalman Filter
            ASSERT_TRUE( particles != NULL );

            //initialize the parameters of the Kalman filter
            InitializeKalman( particles );

            InitializeGroundplaneDisplay( );
        }
        EXCEPTION_CATCH_AND_ABORT( "Error while constructing the information fuser" );
    }

    /********************************************************************
    ~GeometryBasedInformationFuser
        Destructor
    Exceptions:
        None
    *********************************************************************/
    GeometryBasedInformationFuser::~GeometryBasedInformationFuser( )
    {
    }

    /********************************************************************
    CopyMatrix
        Copy source(src_start_row::end, :) to dest(dest_start_row::,:), 
        Assume two matrix of same number of columns; 
        caller is required to supplier matrix of proper size

         option = 0 -> copy data.db to data.db
         option = 1 -> copy data.fl to data.fl
         option = 2 -> copy data.db to data.fl( warning, might lead to error,check and use )
    Exceptions:
        None
    *********************************************************************/
    void GeometryBasedInformationFuser::CopyMatrix( CvMat*    pSourceMatrix,
                                                    CvMat*    pDestinationMatrix, 
                                                    int        sourceStartRowIndex,
                                                    int        destinationStartRowIndex,
                                                    int        option )
    {  
        ASSERT_TRUE( option <= 2 );
        ASSERT_TRUE( pSourceMatrix->cols == pDestinationMatrix->cols );
        ASSERT_TRUE( (pDestinationMatrix->rows - destinationStartRowIndex) >= (pSourceMatrix->rows-sourceStartRowIndex) );

        try
        {
            for ( int p = destinationStartRowIndex; sourceStartRowIndex < pSourceMatrix->rows; p++ )
            {
                for ( int q = 0; q < pSourceMatrix->cols; q++ )
                {
                    if ( option == 0 )
                    {
                        pDestinationMatrix->data.db[p*pSourceMatrix->cols+q]=pSourceMatrix->data.db[sourceStartRowIndex*pSourceMatrix->cols+q];
                    }
                    else if ( option ==1 )
                    {
                        pDestinationMatrix->data.fl[p*pSourceMatrix->cols+q]=pSourceMatrix->data.fl[sourceStartRowIndex*pSourceMatrix->cols+q];
                    }
                    else if ( option == 2 )
                    {
                        //might lead to data loss, use with care
                        pDestinationMatrix->data.fl[ ( p * pSourceMatrix->cols ) + q ] = static_cast<float>( pSourceMatrix->data.db[ ( sourceStartRowIndex * pSourceMatrix->cols ) + q ] );
                    }
                }
                sourceStartRowIndex++;
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to Copy Matrix" );
    }

    /********************************************************************
    MultiVariateNormalPdf
        Calculates Multi-Variate Gaussian PDF
    Exceptions:
        None
    *********************************************************************/
    float GeometryBasedInformationFuser::MultiVariateNormalPdf( CvMat* data, CvMat* mean, CvMat* covariance )
    {
        try
        {
            ASSERT_TRUE( data->rows == mean->rows && data->cols == mean->cols );
            ASSERT_TRUE( data->cols == 1 );
            ASSERT_TRUE( data->rows == covariance->rows && data->rows == covariance->cols );

            float covarianceDeterminant = static_cast<float>( cvDet( covariance ) );
        
            if ( covarianceDeterminant < 0 || covariance->data.fl[0] < 0 )
            {
                abortError( __LINE__, __FILE__, "Input a positive semi-definite matrix for evaluating mvn pdf." );
            }

            CvMat* dataMeanDifference = cvCreateMat( data->rows, 1, CV_32FC1 );
            cvSub( data, mean, dataMeanDifference );

            CvMat* dataMeanDifferenceTranspose = cvCreateMat( 1, data->rows, CV_32FC1 );
            cvTranspose( dataMeanDifference, dataMeanDifferenceTranspose );

            CvMat* covarianceInverse = cvCreateMat( data->rows, data->rows, CV_32FC1 );
            cvInv( covariance, covarianceInverse );

            CvMat* dataMeanDifferenceTransposeMultInvCov =  cvCreateMat( 1, data->rows, CV_32FC1 );
            cvMatMul( dataMeanDifferenceTranspose, covarianceInverse, dataMeanDifferenceTransposeMultInvCov );

            CvMat* dataMeanDifferenceAndCovMult = cvCreateMat( 1, 1, CV_32FC1 );
            cvMatMul( dataMeanDifferenceTransposeMultInvCov, dataMeanDifference, dataMeanDifferenceAndCovMult );

            float productValue = dataMeanDifferenceAndCovMult->data.fl[0];

            float mvnPdf = ( 1 / pow( sqrt( TWO_TIMES_PI ), data->rows ) ) * ( 1 / sqrt( covarianceDeterminant ) ) * exp( -1 * productValue );

            if ( mvnPdf < 0 )
            {
                mvnPdf = 0;
            }

            #ifdef ENABLE_PARTICLE_LOG
                LOG( "Evaluating MVN PDF for x = [ " << data->data.fl[0] << "," << data->data.fl[1] 
                    << "], mean: [" << mean->data.fl[0] << "," << mean->data.fl[1] 
                    << "], covariance : [" << covariance->data.fl[0] << ","<< covariance->data.fl[1] << ";"
                    << covariance->data.fl[2] << "," << covariance->data.fl[3] << "]" 
                    << "mvnPdf value" << mvnPdf << endl );
            #endif

            //release memory
            cvReleaseMat(&dataMeanDifference);     
            cvReleaseMat(&dataMeanDifferenceTranspose); 
            cvReleaseMat(&covarianceInverse); 
            cvReleaseMat(&dataMeanDifferenceTransposeMultInvCov); 
            cvReleaseMat(&dataMeanDifferenceAndCovMult); 

            return mvnPdf;
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to calculate Multi-Variate normal PDF" );
    }

    /********************************************************************
    TransformWithHomography
        Project particles from the camera's image plane to the global 
        ground plane using the predefined homography.
    Exceptions:
        None
    *********************************************************************/
    CvMat* GeometryBasedInformationFuser::TransformWithHomography( CvMat* particles, CvMat* pHomographyMatrix )
    {
        try
        {
            ASSERT_TRUE( pHomographyMatrix != NULL );
            ASSERT_TRUE( particles != NULL );

            CvMat* projectedParticles = cvCreateMat( particles->rows, 2, CV_64FC1 );
            CvMat*    transform        = cvCreateMat( 3, 1 , CV_64FC1 );

            #pragma omp parallel for
            for ( int i=0; i < particles->rows; i++ )
            {
                double    particleArray[]    = { particles->data.fl[i*particles->cols], particles->data.fl[i*particles->cols+1], 1.0 };
                CvMat    particle        = cvMat( 3, 1, CV_64FC1, particleArray );
                
                cvMatMul( pHomographyMatrix, &particle, transform );

                projectedParticles->data.db[i*2]    = transform->data.db[0] / transform->data.db[2] ;
                projectedParticles->data.db[i*2+1]    = transform->data.db[1] / transform->data.db[2] ;

                #ifdef ENABLE_PARTICLE_LOG
                    LOG( "\t" << (i+1) << "\t[" << projectedParticles->data.db[i*2] << "," << projectedParticles->data.db[i*2+1] << "]" 
                        <<"\t Original image coordinates ["<<particleArray[0] <<"," <<particleArray[1] << "]"<< endl );
                #endif
            }
            
            ASSERT_TRUE( projectedParticles != NULL );

            //release memory    
            cvReleaseMat(&transform);     
            return projectedParticles;
        }
        EXCEPTION_CATCH_AND_ABORT( "Error While Projecting to the Ground Plane" )
    }

    /********************************************************************
    EstimateGroundPlanePrincipleAxisIntersection
    Exceptions:
        None
    *********************************************************************/
    vectord GeometryBasedInformationFuser::EstimateGroundPlanePrincipleAxisIntersection(    CvMat*            pPixelMatrix,
                                                                                            vector<CvMat*>    homographyMatrixList )
    {
        try
        {
            ASSERT_TRUE( pPixelMatrix->rows > 1 );

            vectord        groundPlaneIntersection( 2, 0.0 );

            int        numberOfCameras            = pPixelMatrix->rows;

            for ( int i = 0; i < numberOfCameras; i++ )
            {
                LOG_CONSOLE( "Particle" << i << "["  << pPixelMatrix->data.fl[ 2 * i ] 
                << "," <<  pPixelMatrix->data.fl[ 2 * i+1 ] <<"]" <<endl );
            }

            ASSERT_TRUE( homographyMatrixList.size() == numberOfCameras );

            CvMat*    reprojectedLineMatrix    = cvCreateMat( numberOfCameras, 3 , CV_64FC1 );
            CvMat*    reprojectedLine            = cvCreateMat( 1, 3, CV_64FC1 );
            CvMat*    inputLine                = cvCreateMat( 1, 3, CV_64FC1 );
            inputLine->data.db[0] = 1.0;
            inputLine->data.db[1] = 0.0;

            CvMat*    inversehomographyMatrix = cvCreateMat( 3, 3, CV_64FC1 );

            for ( int cameraIndex = 0; cameraIndex < numberOfCameras; cameraIndex++ )
            {
                //assign the x-location
                inputLine->data.db[2] = (-1 * pPixelMatrix->data.fl[ 2 * cameraIndex ] );

                //invert the homography
                cvInv( homographyMatrixList[cameraIndex], inversehomographyMatrix );

                cvMatMul( inputLine, inversehomographyMatrix, reprojectedLine );

                reprojectedLineMatrix->data.db[cameraIndex*3] = reprojectedLine->data.db[0];
                reprojectedLineMatrix->data.db[cameraIndex*3+1] = reprojectedLine->data.db[1];
                reprojectedLineMatrix->data.db[cameraIndex*3+2] = reprojectedLine->data.db[2];
            }

            CvMat* U = cvCreateMat( numberOfCameras, numberOfCameras, CV_64FC1 );
            CvMat* V = cvCreateMat( 3, 3, CV_64FC1 );
            CvMat* W = cvCreateMat( numberOfCameras, 3, CV_64FC1 );

            cvSVD( reprojectedLineMatrix, W, U, V ); 

            groundPlaneIntersection[0] = V->data.db[2] / V->data.db[8];
            groundPlaneIntersection[1] = V->data.db[5] / V->data.db[8];

            LOG_CONSOLE( "Ground plane principal axis intersection:[" << groundPlaneIntersection[0] << ","
                << groundPlaneIntersection[1] << "]\n" );

            //deallocate memory
            cvReleaseMat(&U);
            cvReleaseMat(&V);
            cvReleaseMat(&W);
            cvReleaseMat(&reprojectedLineMatrix);
            cvReleaseMat(&reprojectedLine);
            cvReleaseMat(&inputLine);
            cvReleaseMat(&inversehomographyMatrix);

            return groundPlaneIntersection;
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to estimate ground location from principle axis" );
    }

    /********************************************************************
    SetEMParameters
        Sets Expectation Maximization Algorithm Parameters.
    Exceptions:
        None
    *********************************************************************/
    void GeometryBasedInformationFuser::SetDefaultEMParameters( )
    {
        try
        {
            m_emParamters.covs                    = NULL;
            m_emParamters.means                    = NULL;
            m_emParamters.weights                = NULL;
            m_emParamters.probs                    = NULL;
            m_emParamters.nclusters                = m_numberOfClusters;
            m_emParamters.cov_mat_type            = CvEM::COV_MAT_DIAGONAL;
            m_emParamters.start_step            = CvEM::START_AUTO_STEP;
            m_emParamters.term_crit.max_iter    = 10;
            m_emParamters.term_crit.epsilon        = 0.1;
            m_emParamters.term_crit.type        = CV_TERMCRIT_ITER|CV_TERMCRIT_EPS;
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to Set the Default EM Parameters for Geometric Fusion" );
    }
    
    /********************************************************************
    InitializeKalman
        Initializes Kalman Filter.
    Exceptions:
        None
    *********************************************************************/
    void GeometryBasedInformationFuser::InitializeKalman( CvMat* particles )
    {
        try
        {
    
            m_pKalman = cvCreateKalman( DIMENSION_STATE_VECTOR, DIMENSION_MEASUREMENT_VECTOR, DIMENSION_CONTROL_VECTOR );

            ASSERT_TRUE( m_pKalman != NULL );

            /*double Bu[] = {0,0,0,0};
            double H[] = {1,0,0,1,0,0,0,0 };*/

            //Initializing Kalman Filter
            double A[] = {    1,0,1,0,
                            0,1,0,1,
                            0,0,1,0,
                            0,0,0,1 };        //process model (linear motion)
            double Q[] = {    1,0,0,0,
                            0,1,0,0,
                            0,0,1,0, 
                            0,0,0,1 };    //process noise covariance
            double R[] = { 10, 0, 0, 10 };//measurement noise covariance
            double P[] = {    100,0,0,0,
                            0,100,0,0,
                            0,0,100,0,
                            0,0,0,100 };    //initial  posteriori error estimate covariance matrix (P(k)):
                    
            for( int i=0; i < DIMENSION_STATE_VECTOR * DIMENSION_STATE_VECTOR; i++ )
            {
                    m_pKalman->transition_matrix->data.fl[i]= (float)A[i];
            }

            /*for( int i=0; i < DP; i++ )
                    kalman->control_matrix->data.fl[i] = Bu[i];*/

            /*for( int i=0; i < DP*MP; i++)
                    kalman->measurement_matrix->data.fl[i] = H[i];*/

            cvSetIdentity(m_pKalman->measurement_matrix,cvRealScalar(1.0));

            for( int i=0; i < DIMENSION_STATE_VECTOR * DIMENSION_STATE_VECTOR; i++ )
            {
                m_pKalman->process_noise_cov->data.fl[i] = (float)Q[i];
                m_pKalman->error_cov_post->data.fl[i] = (float)P[i];
            }

            for( int i=0; i < DIMENSION_MEASUREMENT_VECTOR * DIMENSION_MEASUREMENT_VECTOR; i++)
            {
                m_pKalman->measurement_noise_cov->data.fl[i] = (float)R[i];
            }

            vectord measurement;

            if ( m_groundPlaneMeasurementType == PRINCIPAL_AXIS_INTERSECTION )
            {
                //use ground plane intersection for generating measurements
                measurement = EstimateGroundPlanePrincipleAxisIntersection( particles,
                                                                            m_homographyMatrixList );
            }
            else
            {
                //Fit the Gaussian and use the weighted mean as initial state_post
                m_emModel.train(particles, 0, m_emParamters, 0 );

                //Store the Mean, Covariances, Weights
                const CvMat* em_mean        = m_emModel.get_means( );
                //const CvMat** em_cov        = m_emModel.get_covs( );
                const CvMat* em_weights        = m_emModel.get_weights( );

                double* mean   = em_mean->data.db;
                double* weight = em_weights->data.db;

                for (int i=0; i < m_emParamters.nclusters; i++)
                {
                    LOG_CONSOLE( "weight of cluster:" << i+1 << " = " << weight[i] );

                    measurement[0] = measurement[0] + mean[ i*2 ]    * weight[i];
                    measurement[1] = measurement[1] + mean[ i*2+1 ] * weight[i];
                }
            }

            ASSERT_TRUE( measurement.size() == 2 );

            //Initial Posterior State
            m_pKalman->state_post->data.fl[0]=static_cast<float>(measurement[0]); //x
            m_pKalman->state_post->data.fl[1]=static_cast<float>(measurement[1]); //y
            
            m_pKalman->state_post->data.fl[2] = 0.0f; //velocity X
            m_pKalman->state_post->data.fl[3] = 0.0f; //velocity Y
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed initialize the Kalman Filter" )
    }

    /********************************************************************
    UpdateKalmanFilter
        Updates the Kalman Filter with the measurement.
    Exceptions:
        None
    *********************************************************************/
    void GeometryBasedInformationFuser::UpdateKalmanFilter( CvMat* measurement,  CvMat* measurementCov )
    {
        try 
        {
            //Update the measurement error covariance
            if( measurementCov != NULL )
            {
                CopyMatrix( measurementCov, m_pKalman->measurement_noise_cov, 0, 0, 1 ); //R is updated
            }
    
            //Kalman Predict
            cvKalmanPredict( m_pKalman, 0 );        
    
            //Kalman Correct
            cvKalmanCorrect( m_pKalman, measurement );    
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to Update Kalman Filter" );
    }

    
    /********************************************************************
    FuseInformationByAveragingClusters
        Fuses information.
        Input (particles): a collection of particles (of equal weight)
        Output (updatedWeights): new weight for each particle
    Exceptions:
        None
    *********************************************************************/
    void GeometryBasedInformationFuser::FuseInformation( CvMat* pParticlesMatrix )
    {
        try
        {
            switch( m_groundPlaneMeasurementType )
            {
            case HIGHEST_WEIGHT_CLUSTER:
                FuseInformationByPickingTheBestCluster( pParticlesMatrix );
                break;
            case WEIGHTED_AVERAGE_OF_CLUSTERS:
                FuseInformationByAveragingClusters( pParticlesMatrix );
                break;
            case PRINCIPAL_AXIS_INTERSECTION:
                FuseInformationByUsingPrincipalAxisIntersection( pParticlesMatrix );
                break;
            default:
                ASSERT_TRUE( false );
                break;
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to fuse information" );
    }

    
    /********************************************************************
    FuseInformationByAveragingClusters
        Fuses information.
        Input (particles): a collection of particles (of equal weight)
        Output (updatedWeights): new weight for each particle
    Exceptions:
        None
    *********************************************************************/
    void GeometryBasedInformationFuser::FuseInformationByAveragingClusters( CvMat* particles )
    {
        try
        {
            ///Fit the Gaussian
            m_emModel.train(particles, 0, m_emParamters, 0 );

            #ifdef ENABLE_PARTICLE_LOG
                LOG( "Fuse Information" << endl);
                for ( int i = 0; i < particles->rows; i++ )
                {
                    LOG( "Input Particle:" << (i+1) << "[" << particles->data.fl[i*2] << "," << particles->data.fl[i*2+1] << "]" << endl);
                }
            #endif
            //Store the Mean, Covariances, Weights
            const CvMat* em_mean        = m_emModel.get_means( );
            const CvMat** em_cov        = m_emModel.get_covs( );
            const CvMat* em_weights        = m_emModel.get_weights( );

            float weightedMean[]={0,0};

            for ( int i=0; i < m_emParamters.nclusters; i++ )
            {
                #ifdef ENABLE_PARTICLE_LOG
                    LOG( "weight" <<  em_weights->data.db[i] << endl );
                    LOG( "mean" << em_mean->data.db[ i*2 ] <<","<< em_mean->data.db[ i*2+1] << endl );
                #endif

                LOG( "weight of cluster:" << i+1 << " = " << em_weights->data.db[i] );

                weightedMean[0] = weightedMean[0] + static_cast<float>( em_mean->data.db[ i*2 ] * em_weights->data.db[i] );
                weightedMean[1] = weightedMean[1] + static_cast<float>( em_mean->data.db[ i*2+1 ] * em_weights->data.db[i] );
            }

            CvMat measurement=cvMat( 2, 1 ,CV_32FC1, &weightedMean );

            float R[]={0.0, 0.0, 0.0, 0.0};
            CvMat measurementCov = cvMat( 2, 2, CV_32FC1, &R );

            CvMat* diff                = cvCreateMat( 2, 1, CV_32FC1 );
            CvMat* diffTranspose    = cvCreateMat( 1, 2, CV_32FC1 );
            CvMat* product            = cvCreateMat( 2, 2, CV_32FC1 );
            CvMat* addCov            = cvCreateMat( 2, 2, CV_32FC1 );

            for ( int i=0; i < m_emParamters.nclusters; i++ )
            {
                double* covPtr        = (*(em_cov+i))->data.db;
                float covArray[]    = { static_cast<float>( covPtr[0] ), 
                                        static_cast<float>( covPtr[1] ), 
                                        static_cast<float>( covPtr[2] ),
                                        static_cast<float>( covPtr[3] ) };
                CvMat cov            = cvMat( 2 , 2, CV_32FC1, &covArray );
                float meanArray[]    = { static_cast<float>( em_mean->data.db[ i*2 ] ),
                                        static_cast<float>( em_mean->data.db[ i*2+1 ] ) };
                CvMat mu            = cvMat(2,1,CV_32FC1, &meanArray);
        
                //difference of mu and measurement
                cvSub( &mu, &measurement, diff);

                //Transpose the difference
                cvTranspose( diff, diffTranspose );

                //Multiply the difference with the transpose
                cvMatMul( diff, diffTranspose, product);

                //Add with the Component Covariance
                cvAdd( &cov, product, addCov );

                //weight the component covariance????
            /*    float weight = static_cast<float>( em_weights->data.db[i]);
                for ( int i = 0; i < 4; i++)
                {
                    addCov->data.fl[i]=addCov->data.fl[i]*weight;
                }*/

                //Total Sum
                cvAdd( &measurementCov ,addCov , &measurementCov );
            }

            //Update the Kalman Filter
            UpdateKalmanFilter( &measurement, &measurementCov );

            //remove memory
            cvReleaseMat(&diff);     
            cvReleaseMat(&diffTranspose);     
            cvReleaseMat(&product);     
            cvReleaseMat(&addCov);     
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to Update Kalman Filter" );
    }

    
    /********************************************************************
    FuseInformationByPickingTheBestCluster
        Fuses information.
        Input (particles): a collection of particles (of equal weight)
        Output (updatedWeights): new weight for each particle
    Exceptions:
        None
    *********************************************************************/
    void GeometryBasedInformationFuser::FuseInformationByPickingTheBestCluster( CvMat* particles )
    {
        try
        {
            ///Fit the Mixture of Gaussians
            m_emModel.train(particles, 0, m_emParamters, 0 );

            //Store the Mean, Covariances, Weights
            const CvMat* em_mean        = m_emModel.get_means( );
            const CvMat** em_cov        = m_emModel.get_covs( );
            const CvMat* em_weights        = m_emModel.get_weights( );

            vectord clusterWeightList(m_emParamters.nclusters, 0.0);

            for (int i=0; i < m_emParamters.nclusters; i++)
            {
                clusterWeightList[i] = em_weights->data.db[i];
                LOG_CONSOLE( "Weight of cluster " << (i+1) << ":" << clusterWeightList[i] << endl );
            }

            vectori orderDescending;
            sort_order_des( clusterWeightList, orderDescending );

            uint maximumIndex = orderDescending[0];            
            if ( clusterWeightList[0] < 0.4f )
            {
                LOG_CONSOLE( "############Warning###########" << endl << "largest cluster has a weight: " << clusterWeightList[0] <<endl );
            }
            
            double* covPtr        = (*(em_cov+maximumIndex))->data.db;
            float covArray[]    = { static_cast<float>( covPtr[0] ), 
                                    static_cast<float>( covPtr[1] ), 
                                    static_cast<float>( covPtr[2] ),
                                    static_cast<float>( covPtr[3] ) };
            float mean[]        =    {    static_cast<float>( em_mean->data.db[ maximumIndex*2 ] ),
                                        static_cast<float>( em_mean->data.db[ maximumIndex*2+1 ] ) 
                                    };

            CvMat measurement    = cvMat( 2, 1 ,CV_32FC1, &mean );
            CvMat cov            = cvMat( 2 , 2, CV_32FC1, &covArray );

            
            //Update the Kalman Filter
            UpdateKalmanFilter( &measurement, &cov );
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to Update Kalman Filter" );
    }

    /********************************************************************
    FuseInformationByUsingPrincipalAxisIntersection
        Fuses information.
        Input (particles): a collection of particles (of equal weight)
    Exceptions:
        None
    *********************************************************************/
    void GeometryBasedInformationFuser::FuseInformationByUsingPrincipalAxisIntersection( CvMat* particles )
    {
        try
        {
            #ifdef ENABLE_PARTICLE_LOG
                LOG( "Fuse Information" << endl );
                for ( int i = 0; i < particles->rows; i++ )
                {
                    LOG( "Input Particle:" << (i+1) 
                            << "[" << particles->data.fl[i*2] 
                            << "," 
                            << particles->data.fl[i*2+1] << "]" 
                            << endl );
                }
            #endif

            //estimate the intersection points
            vectord groundPlaneIntersectionList = EstimateGroundPlanePrincipleAxisIntersection( particles,
                                                                                                m_homographyMatrixList );

            ASSERT_TRUE( groundPlaneIntersectionList.size() == 2 );

            //store the measurement
            float position[] = { static_cast<float>(groundPlaneIntersectionList[0]), static_cast<float>(groundPlaneIntersectionList[1]) };
            CvMat measurement = cvMat( 2, 1 ,CV_32FC1, &position );

            //constant measurement covariance
            float covArray[]                = { 5, 0, 0, 5 };
            CvMat measurementCovariance        = cvMat( 2 , 2, CV_32FC1, &covArray );

            //update the Kalman Filter
            UpdateKalmanFilter( &measurement, &measurementCovariance );
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to Update Kalman Filter" );
    }


    /********************************************************************
    GetGroundPlaneKalmanMeanMatrix
        Get Ground Plane Kalman Mean Matrix
    Exceptions:
        None
    *********************************************************************/
    CvMat* GeometryBasedInformationFuser::GetGroundPlaneKalmanMeanMatrix( int index )
    {
        try
        {
            //set the mean
            CvMat* pMeanMatrix            = cvCreateMat( 2, 1, CV_32FC1 );
            
            ASSERT_TRUE( pMeanMatrix != NULL );
            pMeanMatrix->data.fl[0]    = m_pKalman->state_post->data.fl[0];
            pMeanMatrix->data.fl[1]    = m_pKalman->state_post->data.fl[1];
            
            return pMeanMatrix;
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to get ground plane kalman filter's mean" );
    }

    /********************************************************************
    GetGroundPlaneKalmanMeanMatrix
        Get Ground Plane Kalman Mean Matrix
    Exceptions:
        None
    *********************************************************************/
    CvMat* GeometryBasedInformationFuser::GetGroundPlaneKalmanCovarianceMatrix( )
    {
        try
        {
            CvMat* pCovarianceMatrix    = cvCreateMat( 2, 2, CV_32FC1 );
            ASSERT_TRUE( pCovarianceMatrix != NULL );
            
            pCovarianceMatrix->data.fl[0] = m_pKalman->error_cov_post->data.fl[0];
            pCovarianceMatrix->data.fl[1] = m_pKalman->error_cov_post->data.fl[1];
            pCovarianceMatrix->data.fl[2] = m_pKalman->error_cov_post->data.fl[4];
            pCovarianceMatrix->data.fl[3] = m_pKalman->error_cov_post->data.fl[5];

            return pCovarianceMatrix;
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to get ground plane kalman filter's mean" );
    }

    /********************************************************************
    InitializeGroundplaneDisplay
        Prepare the Ground plane image
    Exceptions:
        None
    *********************************************************************/    
    void GeometryBasedInformationFuser::InitializeGroundplaneDisplay( )
    {
        try
        {
            m_GroundPlaneDisplayMatrix.Resize(900, 400, 3);
            m_GroundPlaneDisplayMatrixKF.Resize( 900, 400, 3 );
            m_GroundPlaneDisplayMatrixGMM.Resize( 900, 400, 3 );

            m_colorMatrix.Resize( MAX_NUMBER_CAMERA_COLORS, 3);            

            //1st available color (Red Color)
            m_colorMatrix( 0, 0 ) = 255;
            m_colorMatrix( 0, 1 ) = 0;
            m_colorMatrix( 0, 2 ) = 0;

            //2nd  available color  (blue Color)
            m_colorMatrix( 1, 0 ) = 0;
            m_colorMatrix( 1, 1 ) = 0;
            m_colorMatrix( 1, 2 ) = 255;

            //3rd  available color  (green Color)
            m_colorMatrix( 2, 0 ) = 0;
            m_colorMatrix( 2, 1 ) = 255;
            m_colorMatrix( 2, 2 ) = 0;

            //4th  available color  (black Color)
            m_colorMatrix( 3, 0 ) = 0;
            m_colorMatrix( 3, 1 ) = 0;
            m_colorMatrix( 3, 2 ) = 0;

            //5th  available color  (orange Color)
            m_colorMatrix( 4, 0 ) = 255;
            m_colorMatrix( 4, 1 ) = 255;
            m_colorMatrix( 4, 2 ) = 0;        
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to initialize ground plane display" );
    }

    /********************************************************************
    DisplayOriginalGroundParticles
        Display the Ground plane image
    Exceptions:
        None
    *********************************************************************/    
    Matrixu* GeometryBasedInformationFuser::DisplayOriginalGroundParticles( CvMat* particles, int frameInd )
    {
        try
        {
            m_GroundPlaneDisplayMatrix.Set( 128 );//Gray background
            
            if( frameInd >= 0 )
            {
                //draw the frame number on the image
                m_GroundPlaneDisplayMatrix.drawText(("#"+int2str(frameInd,3)).c_str(),1,25,0,0,0);
            }

            int numberOfParticles = particles->rows/m_numberOfCameras;
            int xPosPixel = 0, yPosPixel = 0;
            float xPos, yPos;
            int particleIndex = 0;
            for ( int cameraInd = 0; cameraInd < m_numberOfCameras; cameraInd++ )
            {
                for ( int i = 0; i < numberOfParticles; i++ )
                {
                    xPos = particles->data.fl[particleIndex*2];
                    yPos = particles->data.fl[particleIndex*2+1];
                    
                    particleIndex++;

                    MetricToPixels(xPos, yPos, &xPosPixel, &yPosPixel);
                    int colorInd = cameraInd % MAX_NUMBER_CAMERA_COLORS;
                    m_GroundPlaneDisplayMatrix( yPosPixel,xPosPixel,0 ) = m_colorMatrix(colorInd,0);
                    m_GroundPlaneDisplayMatrix( yPosPixel,xPosPixel,1 ) = m_colorMatrix(colorInd,1);
                    m_GroundPlaneDisplayMatrix( yPosPixel,xPosPixel,2 ) = m_colorMatrix(colorInd,2);
                            
                }
            }
            return &m_GroundPlaneDisplayMatrix;
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to show the ground plane display" );
    }

    /********************************************************************
    DisplayOriginalGroundParticles
        DisplayKalmanFilterPdf
    Exceptions:
        None
    *********************************************************************/    
    Matrixu* GeometryBasedInformationFuser::DisplayKalmanFilterPdf( CvMat* pMeanMatrix, CvMat* pCovarianceMatrix, int frameInd )
    {
        try
        {
            m_GroundPlaneDisplayMatrixKF.Free();
            m_GroundPlaneDisplayMatrixKF.Resize(900, 400, 3);
            //m_GroundPlaneDisplayMatrixKF.Set( 0 );//Dark background
                
            CvMat* data = cvCreateMat( 2, 1, CV_32FC1 );
            ASSERT_TRUE( data != NULL );
            
            if( frameInd >= 0 )
            {
                //draw the frame number on the image
                m_GroundPlaneDisplayMatrixKF.drawText(("#"+int2str(frameInd,3)).c_str(),1,25,255,255,0);
            }

            float probability; 
            float normalizationFactor;
            int brightness;
            uchar displayBrightness;

            int centerXPixel = 0, centerYPixel=0;

            LOG( "KF post mean: x= "<< pMeanMatrix->data.fl[0] << "; y=" << pMeanMatrix->data.fl[1] <<endl );
            LOG( "KF post Cov: ["<<pCovarianceMatrix->data.fl[0] << ", " << pCovarianceMatrix->data.fl[1] <<", "
                                    <<pCovarianceMatrix->data.fl[2] << ", " << pCovarianceMatrix->data.fl[3] <<" ];"<<endl );

            MetricToPixels( pMeanMatrix->data.fl[0], pMeanMatrix->data.fl[1], &centerXPixel, &centerYPixel );

            normalizationFactor =  GeometryBasedInformationFuser::MultiVariateNormalPdf( pMeanMatrix, pMeanMatrix, pCovarianceMatrix );    

            int left    = 0; //min( max( centerXPixel - 200, 0 ), 399);
            int right    = 399;//min( max( centerXPixel + 200, 0 ), 399);
            int top        = min( max( centerYPixel + 200, 0), 899);
            int bottom    = min( max( centerYPixel - 200, 0), 899);;
            
            for ( int xPixel = left; xPixel <= right; xPixel++ )
            {
                for (int yPixel = bottom; yPixel <= top; yPixel++ )
                {
                    PixelsToMetric( data->data.fl, data->data.fl + 1, xPixel, yPixel);
                    probability = GeometryBasedInformationFuser::MultiVariateNormalPdf( data, pMeanMatrix, pCovarianceMatrix )/normalizationFactor;                    
                    brightness = cvRound( 255.0f * probability );
                    displayBrightness = static_cast<uchar>(max( min(brightness, 255), 0));
                    m_GroundPlaneDisplayMatrixKF( yPixel, xPixel, 0 ) = displayBrightness; 
                    m_GroundPlaneDisplayMatrixKF( yPixel, xPixel, 1 ) = displayBrightness; 
                    m_GroundPlaneDisplayMatrixKF( yPixel, xPixel, 2 ) = displayBrightness; 
                }
            }
            
            cvReleaseMat(&data);     
            return &m_GroundPlaneDisplayMatrixKF;
        }

        EXCEPTION_CATCH_AND_ABORT( "Failed to show the ground plane with KF" );
    }    
    
    /********************************************************************
    DisplayOriginalGroundParticles
        Display the GMM on top of particles (if given)
    Exceptions:
        None
    *********************************************************************/    
    Matrixu* GeometryBasedInformationFuser::DisplayGMMGroundParticles( int frameInd, CvMat* particles )
    {
        try
        {
            m_GroundPlaneDisplayMatrixGMM.Set( 80 );//Gray background

            if ( m_groundPlaneMeasurementType == PRINCIPAL_AXIS_INTERSECTION )
            {
                return &m_GroundPlaneDisplayMatrixGMM;
            }
            
            int xPosPixel = 0, yPosPixel = 0;
            float xPos, yPos;

            if( frameInd >= 0 )
            {
                //draw the frame number on the image
                m_GroundPlaneDisplayMatrixGMM.drawText(("#"+int2str(frameInd,3)).c_str(),1,25,130,130,130);
            }
            if( particles!=NULL )
            {    
                int numberOfParticles = particles->rows/m_numberOfCameras;

                int particleIndex = 0;
                for ( int cameraInd = 0; cameraInd < m_numberOfCameras; cameraInd++ )
                {
                    for ( int i = 0; i < numberOfParticles; i++ )
                    {
                        xPos = particles->data.fl[particleIndex*2];
                        yPos = particles->data.fl[particleIndex*2+1];

                        particleIndex++;

                        MetricToPixels(xPos, yPos, &xPosPixel, &yPosPixel);
                        int colorInd = cameraInd % MAX_NUMBER_CAMERA_COLORS;
                        m_GroundPlaneDisplayMatrixGMM( yPosPixel,xPosPixel,0 ) = m_colorMatrix(colorInd,0);
                        m_GroundPlaneDisplayMatrixGMM( yPosPixel,xPosPixel,1 ) = m_colorMatrix(colorInd,1);
                        m_GroundPlaneDisplayMatrixGMM( yPosPixel,xPosPixel,2 ) = m_colorMatrix(colorInd,2);
                    }
                }
            }
            
            if ( m_groundPlaneMeasurementType != PRINCIPAL_AXIS_INTERSECTION )
            {
                //draw the GMM
                const CvMat* em_mean        = m_emModel.get_means( );
                const CvMat** em_cov        = m_emModel.get_covs( );
                const CvMat* em_weights        = m_emModel.get_weights( );
                for ( int i=0; i < m_emParamters.nclusters; i++ )
                {    
                    //weight = em_weights->data.db[i]
                    float meanX = static_cast<float>( em_mean->data.db[ i*2 ] );
                    float meanY = static_cast<float>( em_mean->data.db[ i*2+1] );
                    
                    MetricToPixels(meanX, meanY, &xPosPixel, &yPosPixel);
                    m_GroundPlaneDisplayMatrixGMM.drawEllipse(    2, 2, xPosPixel, yPosPixel,    1, // height, width, x, lineWidth
                        255, 255, 255 );
                }
            }

            return &m_GroundPlaneDisplayMatrixGMM;
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to show the ground plane display" );
    }

    
    /********************************************************************
    MetricToPixels
        given a point on the ground plane, 
        find its corresponding position on the ground plane image
    Exceptions:
        None
    *********************************************************************/    
    void GeometryBasedInformationFuser::MetricToPixels( float xPosFeet, float yPosFeet, int * xPosPixel, int * yPosPixel )
    { //a image to represent ground plane (width 400 pixels, height = 900 pixels)
        *xPosPixel = min( max( cvRound( ((-1)*xPosFeet+8)*25 ), 0), 399);
        *yPosPixel = min( max( cvRound( (yPosFeet+150)*3), 0 ), 899 );
    }

    /********************************************************************
    PixelsToMetric
        given a pixel on the ground plane image,
        find its corresponding position on the ground plane
    Exceptions:
        None
    *********************************************************************/    
    void GeometryBasedInformationFuser::PixelsToMetric( float * xPosFeet, float * yPosFeet, int xPosPixel, int yPosPixel )
    { //a image to represent ground plane (width 400 pixels, height = 900 pixels)
        *xPosFeet = 8.0 - float(xPosPixel)/25.0;
        *yPosFeet = float(yPosPixel)/3.0 - 150;
    }
}