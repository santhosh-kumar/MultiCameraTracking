#include "StrongClassifierBase.h"
#include "StrongClassifierFactory.h"
#include "ParticleFilterTracker.h"
#include "GeometryBasedInformationFuser.h"

#include "Public.h"
#include "Sample.h"
#include "CommonMacros.h"

#include <numeric>

#define HAAR_CASCADE_FILE_NAME "haarcascade_frontalface_alt_tree.xml"

namespace MultipleCameraTracking
{
    /**************************************************************************
    ~ParticleFilterTracker
        Destructor. 
    Exceptions:
        None
    **************************************************************************/
    ParticleFilterTracker::~ParticleFilterTracker()
    {
        try
        {
            //Release the CvMat* explicitly
            cvReleaseMat(&m_pGroundLocation);         
            cvReleaseMat(&m_pWeightedAverageParticleMatrix );
        }
        EXCEPTION_CATCH_AND_LOG( "Error in destruction of Particle Filter Tracker" );
    }

    /**************************************************************************
    InitializeTrackerWithParameters
        Initialize the tracking, including video information, tracking parameters etc. 
        Open the video reader and writers-> call InitializeTracker()
    Exceptions:
        None
    **************************************************************************/
    void    ParticleFilterTracker::InitializeTrackerWithParameters( Matrixu*                pFrameImageColor, 
                                                                    Matrixu*                pFrameImageGray, 
                                                                    int                        frameInd,
                                                                    uint                    videoLength, 
                                                                    TrackerParametersPtr    trackerParametersPtr,
                                                                    Classifier::StrongClassifierParametersBasePtr    classifierParameterPtr,
                                                                    Matrixu*                pFrameDisplay, 
                                                                    Matrixu*                pFrameDisplayTraining,
                                                                    Matrixu*                pFrameImageHSV,
                                                                    Matrixf*                pGroundTruthMatrix )
    {        
        try
        {
            m_states.Resize( videoLength, 4 );
            
            // InitializeTracker with face
            if ( trackerParametersPtr->m_initializeWithFaceDetection )
            {  
                fprintf(stderr,"Searching for face...\n");
                abortError( __LINE__, __FILE__, "Initialization with face has not implemented with particle filter");    
            } 
            // InitializeTracker with params
            else
            {
                //m_initState = [x,y,w,h,startingFrame,objectId]
                ASSERT_TRUE( frameInd == trackerParametersPtr->m_initState[4] );

                //assign width and height for feature parameter's training
                classifierParameterPtr->m_featureParametersPtr->m_width    = (uint)trackerParametersPtr->m_initState[2];
                classifierParameterPtr->m_featureParametersPtr->m_height    = (uint)trackerParametersPtr->m_initState[3];
                
                bool success = InitializeTracker( pFrameImageColor, pFrameImageGray, trackerParametersPtr, classifierParameterPtr, pFrameDisplay, pFrameDisplayTraining, pFrameImageHSV );; 
                ASSERT_TRUE( success );
                
                //m_states --> [leftX LeftY scaledWidth scaledHeight]
                for (int i = 0; i < frameInd; i++)
                {
                    m_states(i,0) = 0; m_states(i,1) = 0; m_states(i,2) = 0;m_states(i,3) = 0;
                }
                m_states(frameInd,0)    = m_currentStateList[0];                                // left_x
                m_states(frameInd,1)    = m_currentStateList[1];                                // left_y
                m_states(frameInd,2)    = ( m_currentStateList[2] * m_currentStateList[4]);        // width * scale
                m_states(frameInd,3)    = ( m_currentStateList[3] * m_currentStateList[5] );    // height * scale
            }

            if ( pGroundTruthMatrix != NULL )
            {
                m_groundTruthMatrix = *pGroundTruthMatrix;
                ASSERT_TRUE( m_groundTruthMatrix.rows() >= videoLength );
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to initialize Particle Filter Tracker" );
    }

    /**************************************************************************
    InitializeTracker
        Initialize from first frame information, including learning 
        initial classifier model
    Exceptions:
        None
    **************************************************************************/
    bool    ParticleFilterTracker::InitializeTracker(    Matrixu*                pFrameImageColor, 
                                                        Matrixu*                pFrameImageGray,  
                                                        TrackerParametersPtr    trackerParametersPtr, 
                                                        Classifier::StrongClassifierParametersBasePtr classifierParametersPtr,
                                                        Matrixu*                pFrameDisplay, 
                                                        Matrixu*                pFrameDisplayTraining,
                                                        Matrixu*                pFrameImageHSV    )
    {        
        try
        {
            m_particleFilterTrackerParamsPtr = boost::static_pointer_cast<ParticleFilterTrackerParameters>(trackerParametersPtr); //particle filter parameters
            m_simpleTrackerParamsPtr = boost::static_pointer_cast<SimpleTrackerParameters>(trackerParametersPtr); //

             //initialize integral image for HAAR_LIKE features
            if ( classifierParametersPtr->m_featureParametersPtr->GetFeatureType() == Features::HAAR_LIKE
                || classifierParametersPtr->m_featureParametersPtr->GetFeatureType() == Features::HAAR_COLOR_HISTOGRAM )
            {
                ASSERT_TRUE( pFrameImageGray!=NULL && pFrameImageGray->isInitII() );
            }

            m_strongClassifierBasePtr = Classifier::StrongClassifierFactory::CreateAndInitializeClassifier( classifierParametersPtr );    // create strong classifier
            m_currentStateList.resize(6);                    // current object state (position, Size)

            /* m_currentStateList: [leftX, topY, sizeX, sizeY, scaleX, scaleY] */
            for ( int i=0; i<4; i++ ) 
            {
                m_currentStateList[i] = m_simpleTrackerParamsPtr->m_initState[i];
            }

            //initialize scale to 1.0 at the start
            m_currentStateList[4] = 1;    //i.e., scaleX = 1.0
            m_currentStateList[5] = 1;    //i.e., scaleY = 1.0

            // draw a colored box around object for display 
            if ( pFrameDisplay != NULL )
            {
                pFrameDisplay->drawRect(    m_currentStateList[2],
                                            m_currentStateList[3],
                                            m_currentStateList[0],
                                            m_currentStateList[1],
                                            1,
                                            0,
                                            m_simpleTrackerParamsPtr->m_outputBoxLineWidth,
                                            m_simpleTrackerParamsPtr->m_outputBoxColor[0],
                                            m_simpleTrackerParamsPtr->m_outputBoxColor[1],
                                            m_simpleTrackerParamsPtr->m_outputBoxColor[2] );                
            }

            LOG( "Initializing Particle Filtering Tracker..."<<endl );        
            LOG( "Initial object state: [" <<m_currentStateList[0] << " " <<m_currentStateList[1] << " " <<m_currentStateList[2] << " " << m_currentStateList[3] << "].\n");

            // sample positives and negatives from first frame
            //generate positive samples from the circle of radius m_simpleTrackerParamsPtr->m_init_posTrainRadius with (x,y) as center
            
            m_positiveSampleSet.SampleImage(    pFrameImageGray,            //pGrayImageMatrix
                                            (uint)m_currentStateList[0],//x
                                            (uint)m_currentStateList[1],//y
                                            (uint)m_currentStateList[2],//width
                                            (uint)m_currentStateList[3],//height
                                            m_simpleTrackerParamsPtr->m_init_posTrainRadius, //outerCircleRadius
                                            0,                            //innerCircleRadius
                                            1000000,                    //maximumNumberOfSamples
                                            pFrameImageColor,            //pRGBImageMatrix
                                            pFrameImageHSV );            //pHSVImageMatrix

            //generate samples from a ring that is far way from the center
            m_negativeSampleSet.SampleImage(    pFrameImageGray,                //pGrayImageMatrix
                                            (uint)m_currentStateList[0],    //x
                                            (uint)m_currentStateList[1],    //y
                                            (uint)m_currentStateList[2],    //width
                                            (uint)m_currentStateList[3],    //height
                                            2.0f * m_simpleTrackerParamsPtr->m_searchWindSize,        //outerCircleRadius
                                            1.5f * m_simpleTrackerParamsPtr->m_init_posTrainRadius, //innerCircleRadius
                                            m_simpleTrackerParamsPtr->m_init_negNumTrain,            //maximumNumberOfSamples
                                            pFrameImageColor,                //pRGBImageMatrix
                                            pFrameImageHSV );                //pHSVImageMatrix

            if ( m_positiveSampleSet.Size() < 1 || m_negativeSampleSet.Size() < 1 )
            {
                return false;
            }

            // train the classifier
            m_strongClassifierBasePtr->Update( m_positiveSampleSet, m_negativeSampleSet );

            DisplayTrainingSamples( pFrameDisplayTraining );

            m_positiveSampleSet.Clear();
            m_negativeSampleSet.Clear();

            int frameWidth, frameHeight;
            if ( pFrameImageGray!= NULL )
            {    
                frameWidth = pFrameImageGray->cols();
                frameHeight = pFrameImageGray->rows();
            }
            else
            {
                frameWidth = pFrameImageColor->cols();
                frameHeight = pFrameImageColor->rows();
            }
            
            m_particleFilterPtr = ParticleFilterPtr( new ParticleFilter( frameWidth, frameHeight ) );

            if ( m_particleFilterPtr.get()==NULL )
            {
                abortError(__LINE__,__FILE__,"m_particleFilterPtr is NULL");
            }

            ASSERT_TRUE( m_particleFilterTrackerParamsPtr->m_numOfDisplayedParticles >= 0 && 
                m_particleFilterTrackerParamsPtr->m_numOfDisplayedParticles < m_particleFilterTrackerParamsPtr->m_numberOfParticles );
            
            //initialize particle sets
            m_particleFilterPtr->Initialize( m_particleFilterTrackerParamsPtr->m_numberOfParticles, 
                                            m_currentStateList[0] + m_currentStateList[2]/2,    //centerX
                                            m_currentStateList[1] + m_currentStateList[3]/2,    //centerY
                                            m_currentStateList[4],                                //scaleX
                                            m_currentStateList[5] );                            //scaleY

            //initialize ground positions
            m_pGroundLocation = cvCreateMat( m_particleFilterTrackerParamsPtr->m_numberOfParticles, 2, CV_32FC1 );
            
            EstimateGroundPoint();

            m_isInitialized = true;

            //success
              return true;
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to initialize tracker parameters")
    }

    /**************************************************************************
    TrackObjectAndSaveState
        Track object and store the result. 
    Exceptions:
        None
    **************************************************************************/
    void ParticleFilterTracker::TrackObjectAndSaveState( int        frameInd, 
                                                         Matrixu*    pFrameImageColor, 
                                                          Matrixu*    pFrameImageGray, 
                                                         Matrixu*    pFrameDisplay, 
                                                          Matrixu*    pFrameDisplayTraining,
                                                         Matrixu*    pFrameImageHSV )
    {
        try
        {
            //track object on the given frame 
            TrackObjectOnTheGivenFrame( pFrameImageColor, pFrameImageGray, pFrameDisplayTraining, pFrameImageHSV );
            
            //save the state                    
            StoreObjectState( frameInd, pFrameDisplay );            

            //update the classifier
            UpdateClassifier( pFrameImageColor, pFrameImageGray, pFrameDisplayTraining,    pFrameImageHSV );
        }
        EXCEPTION_CATCH_AND_ABORT("Error while tracking and saving the object state" )
    }

    /********************************************************************
    TrackObjectWithoutSaveState
        TrackObjectWithoutSaveState
    Exceptions:
        None
    ********************************************************************/
    void    ParticleFilterTracker::TrackObjectWithoutSaveState( Matrixu*    pFrameImageColor,
                                                                Matrixu*    pFrameImageGray,
                                                                Matrixu*    pFrameImageHSV ) 
    { 
        try
        {
            TrackObjectOnTheGivenFrame( pFrameImageColor, pFrameImageGray,  NULL,    pFrameImageHSV );
        }
        EXCEPTION_CATCH_AND_ABORT( "Error while tracking "  );
    }


    /********************************************************************
    StoreObjectState
        Store the state of the object in the current frame 
        (store to matrix m_states and tempFrameDisplay)
        (i.e., the frame from the previous call to TrackObjectAndSaveState)
    Exceptions:
        None
    ********************************************************************/
    void ParticleFilterTracker::StoreObjectState(int frameind, Matrixu* pFrameDisplay )
    {
        try
        {            
            vectorf aParticle;        
            float highestWeight = 1, weight = 1; 
            //original width and height: m_currentStateList[2], m_currentStateList[3] does not change 
            if ( m_particleFilterTrackerParamsPtr->m_outputTrajectoryOption == PARTICLE_HIGHEST_WEIGHT )
            {
                vectorf previousState;
                previousState.push_back( ( m_currentStateList[0] + m_currentStateList[2] * m_currentStateList[4] / 2) );
                previousState.push_back( ( m_currentStateList[1] + m_currentStateList[3] * m_currentStateList[5] / 2) );

                m_particleFilterPtr->GetHighestOrderedUniqueParticleCloseToTheGivenState( aParticle, previousState );

                m_currentStateList[0] = aParticle[0] - m_currentStateList[2] * aParticle[2] / 2;    //leftX
                m_currentStateList[1] = aParticle[1] - m_currentStateList[3] * aParticle[3] / 2;    //topY            
                m_currentStateList[4] = aParticle[2];                                                //scaleX
                m_currentStateList[5] = aParticle[3];                                                //scaleY
                highestWeight          = aParticle[4];

                if ( g_detailedLog )
                {
                    LOG( "Highest weighted particle state: ["
                                << m_currentStateList[0] << " " << m_currentStateList[1] << " "
                                << m_currentStateList[2] << " " << m_currentStateList[3] << " "
                                << m_currentStateList[4] << " " << m_currentStateList[5] 
                                << "]. Weight: " << (float)aParticle[4]/m_particleFilterTrackerParamsPtr->m_numberOfParticles << endl );     
                }                
            }
            else
            {
                m_particleFilterPtr->GetAverageofAllParticles(aParticle);
                //aParticle: averageCenterX, averageCenterY, averageScaleX, averageScaleY
                m_currentStateList[0] = aParticle[0] - m_currentStateList[2] * aParticle[2] / 2;    //leftX
                m_currentStateList[1] = aParticle[1] - m_currentStateList[3] * aParticle[3] / 2;    //topY
                m_currentStateList[4] = aParticle[2];    //average scaleX
                m_currentStateList[5] = aParticle[3];    //average scaleY

                if ( g_detailedLog )
                {
                    LOG( "Particle filter tracker: Update state to: [" 
                        << m_currentStateList[0] << " " << m_currentStateList[1] << " "
                        << m_currentStateList[2] << " " << m_currentStateList[3] << " "
                        << m_currentStateList[4] << " " << m_currentStateList[5] <<"]."<< endl );     
                }
            }

            int numOfDisplay = min( m_particleFilterTrackerParamsPtr->m_numOfDisplayedParticles, m_particleFilterPtr->GetNumberOfOrderedUniqueParticles() );
            
            if( g_detailedLog )
            {
                LOG( "Unique number of particles after re-sampling is" << m_particleFilterPtr->GetNumberOfOrderedUniqueParticles()<<endl );
            }

            // draw a few more blue-colored box around largest particles etc.
            int displayParticleStart;

            if( m_particleFilterTrackerParamsPtr->m_outputTrajectoryOption == PARTICLE_HIGHEST_WEIGHT )
            {
                displayParticleStart = 1;
            }
            else
            {
                displayParticleStart = 0;
            }
            //weight the display 
            for (int p = displayParticleStart; p < numOfDisplay; p++) 
            {
                m_particleFilterPtr->GetOrderedUniqueParticles(p,aParticle);
                
                if( p == 0 )
                {
                    highestWeight = aParticle[4];
                }

                weight = aParticle[4]/highestWeight;

                if( pFrameDisplay != NULL ) 
                    pFrameDisplay->drawRect( aParticle[2] * m_currentStateList[2], //width
                                             aParticle[3] * m_currentStateList[3], //height
                                             aParticle[0] - m_currentStateList[2] * aParticle[2] / 2, //leftX
                                             aParticle[1] - m_currentStateList[3] * aParticle[3] / 2, //topY
                                             1, 0, 1, 0, 0, cvRound(weight*255.0f) );        
            }

            // draw a default-colored box around largest particle (or average particle depending on the choices)
            if( pFrameDisplay != NULL)
            {
                pFrameDisplay->drawRect(    m_currentStateList[2] * m_currentStateList[4], 
                    m_currentStateList[3] * m_currentStateList[5], 
                    m_currentStateList[0], 
                    m_currentStateList[1], 1, 0,
                    m_simpleTrackerParamsPtr->m_outputBoxLineWidth, 
                    m_simpleTrackerParamsPtr->m_outputBoxColor[0], 
                    m_simpleTrackerParamsPtr->m_outputBoxColor[1], 
                    m_simpleTrackerParamsPtr->m_outputBoxColor[2]    );
            }

            //save to the trajectory matrix
            m_states(frameind,0) = cvRound( m_currentStateList[0] );
            m_states(frameind,1) = cvRound( m_currentStateList[1] );
            m_states(frameind,2) = cvRound( m_currentStateList[2] * m_currentStateList[4] );
            m_states(frameind,3) = cvRound( m_currentStateList[3] * m_currentStateList[5] );            
        }

        EXCEPTION_CATCH_AND_ABORT("Error while storing the object state" )
    }

    /**************************************************************************
    DrawTestSamples
        Draw predicted particles (non-zeros weight) on the frame except for zero weight particles.        
        Note:
        1.  Not all of them if there are too many of them
        2.     Tracking is suspended until a key is pressed
    Exceptions:
        None
    **************************************************************************/
    void ParticleFilterTracker::DrawTestSamples( Classifier::SampleSet testSamples, Matrixu* pFrame )
    {
        // prepare the frame to be drawn on
        Matrixu tempFrameDisplay;
        if(m_particleFilterTrackerParamsPtr->m_isColor) 
        {
            ASSERT_TRUE(pFrame->depth()==3);
            tempFrameDisplay = *pFrame;
        }
        else
        {
            pFrame->conv2RGB(tempFrameDisplay);
        }

        tempFrameDisplay.createIpl();
        tempFrameDisplay._keepIpl = true;
        //draw negative training examples.
        for ( int j=0; j< testSamples.Size(); j++ )
        {
            //draw  small red ellipses around center
            if( m_simpleTrackerParamsPtr->m_displayTrainingSampleCenterOnly )
            {
                tempFrameDisplay.drawEllipse(    1,
                    1,
                    float( testSamples[j].m_col + float(testSamples[j].m_width)/2 ),
                    float( testSamples[j].m_row + float(testSamples[j].m_height)/2 ),
                    1,
                    255,
                    0,
                    0 );
            }
            //Or draw the actual blob
            else
            {
                tempFrameDisplay.drawEllipse( float( testSamples[j].m_height )/2,
                    float( testSamples[j].m_width )/2,
                    float( testSamples[j].m_col + float(testSamples[j].m_width)/2 ),
                    float( testSamples[j].m_row + float(testSamples[j].m_height)/2 ),
                    1, //line width
                    255,
                    0,
                    0 ); 
            }
        }
        
        tempFrameDisplay.drawText("Prediction (press to continue)",1,25,255,255,0);
        
        tempFrameDisplay.display(
            ("predicted particles"+m_simpleTrackerParamsPtr->m_displayFigureNameStr).c_str(), 2
            );
        cvWaitKey(0);        

        tempFrameDisplay._keepIpl = false;
        tempFrameDisplay.freeIpl();
    }

    /**************************************************************************
    TrackObjectOnTheGivenFrame
        Track individual frame 
        (PF Prediction->PF re-weighting (with re-sampling)->Update ground plane position)
        Classifier is not updated, and therefore pFrameDisplayTraining  is not updated even if it is given
    Exceptions:
        None
    **************************************************************************/
    double    ParticleFilterTracker::TrackObjectOnTheGivenFrame(  Matrixu*    pFrameImageColor, 
                                                                Matrixu*    pFrameImageGray,
                                                                Matrixu*    pFrameDisplayTraining,
                                                                Matrixu*    pFrameImageHSV )
    {
        try
        {
            int frameWidth, frameHeight;             
            if ( pFrameImageGray!= NULL )
            {    
                frameWidth = pFrameImageGray->cols();
                frameHeight = pFrameImageGray->rows();
            }
            else
            {
                frameWidth = pFrameImageColor->cols();
                frameHeight = pFrameImageColor->rows();
            }

            vectorf aParticle;        
            int leftX, topY, width, height;
            float widthFloat, heightFloat;

            //tracker should have been initialized before calling this
            ASSERT_TRUE( m_isInitialized );
            ASSERT_TRUE( pFrameImageColor != NULL || pFrameImageGray!= NULL );

            //particle filter Prediction. 
            m_particleFilterPtr->PredictWithBrownianMotion(  m_particleFilterTrackerParamsPtr->m_standardDeviationX,
                                                            m_particleFilterTrackerParamsPtr->m_standardDeviationY, 
                                                            m_particleFilterTrackerParamsPtr->m_standardDeviationScaleX,
                                                            m_particleFilterTrackerParamsPtr->m_standardDeviationScaleY,
                                                            m_currentStateList[2],
                                                            m_currentStateList[3] );

            //Generate Test Sample Set
            GenerateTestSampleSet( pFrameImageColor, pFrameImageGray, pFrameImageHSV );

            if ( m_testSampleSet.Size() == 0 )
            {
                m_particleFilterPtr->ForceParticleFilterRefinement( m_currentStateList[2], m_currentStateList[3] );

                GenerateTestSampleSet( pFrameImageColor, pFrameImageGray, pFrameImageHSV );

                ASSERT_TRUE( m_testSampleSet.Size( ) != 0 );
            }
            
            #if defined(_DEBUG)
                if ( m_simpleTrackerParamsPtr->m_debugv )
                {
                    if(pFrameImageColor != NULL)
                        DrawTestSamples( m_testSampleSet, pFrameImageColor );
                    else 
                        DrawTestSamples( m_testSampleSet, pFrameImageGray );
                }
            #endif
            //test with the classifier
            m_liklihoodProbabilityList = m_strongClassifierBasePtr->Classify( m_testSampleSet, m_simpleTrackerParamsPtr->m_shouldNotUseSigmoid );

            //draw H(x) results for each candidate blob
            if ( m_simpleTrackerParamsPtr->m_debugv )
            {
                Matrixf probimg( frameHeight, frameWidth );

                for( uint k=0; k<(uint)m_testSampleSet.Size(); k++ )
                {
                    probimg(m_testSampleSet[k].m_row, m_testSampleSet[k].m_col) = m_liklihoodProbabilityList[k];
                }
                probimg.convert2img().display(    ("Probability map, "+ m_simpleTrackerParamsPtr->m_displayFigureNameStr).c_str(),
                                                2 );
                cvWaitKey(1);
            }

            int bestind, worstind;
            double maximumResponse = 0.0, minimumResponse = 0.0;    
            bestind        = max_idx( m_liklihoodProbabilityList ); // find best location
            worstind    = min_idx( m_liklihoodProbabilityList ); 
            if ( !m_liklihoodProbabilityList.empty() )
            {
                maximumResponse    = m_liklihoodProbabilityList[bestind];
                minimumResponse = m_liklihoodProbabilityList[worstind];
            }

            double range = maximumResponse - minimumResponse;

//            ASSERT_TRUE( range != 0 );

            for ( int index = 0; index < m_liklihoodProbabilityList.size(); index++ )
            {
                if ( range != 0 )
                {
                    m_liklihoodProbabilityList[index] = float ( pow( (m_liklihoodProbabilityList[index] - minimumResponse) / range, 20 ) ); 
                    //m_liklihoodProbabilityList[index] = sigmoid( m_liklihoodProbabilityList[index]  );
                }
                else
                {
                    m_liklihoodProbabilityList[index] = 1;
                }
            }
            //::normalizeVec(m_liklihoodProbabilityList);

            //draw likelihood ( based on H(x)) results for each candidate blob
            if ( m_simpleTrackerParamsPtr->m_debugv )
            {
                Matrixf probimg( frameHeight, frameWidth );

                for( uint k=0; k<(uint)m_testSampleSet.Size(); k++ )
                {//in terms of sample (topY, leftX) instead of center as in the particleFilter
                    probimg(m_testSampleSet[k].m_row, m_testSampleSet[k].m_col) = m_liklihoodProbabilityList[k];
                }
                probimg.convert2img().display( ("Probability map2, "+ m_simpleTrackerParamsPtr->m_displayFigureNameStr).c_str(), 2 );
                cvWaitKey(1);
            }

            //draw the weight of all unique particles
            if ( m_simpleTrackerParamsPtr->m_debugv )
            {
                Matrixf probimg( frameHeight, frameWidth );
                
                for( int p = 0; p < m_particleFilterTrackerParamsPtr->m_numberOfParticles; p++ )
                {
                    m_particleFilterPtr->GetParticle( p, aParticle);
                    widthFloat    = aParticle[2] * m_currentStateList[2]; 
                    heightFloat    = aParticle[3] * m_currentStateList[3]; 
                    leftX    = cvRound( aParticle[0] - widthFloat / 2 );
                    topY    = cvRound( aParticle[1] - heightFloat / 2 );
                    width    = cvRound( widthFloat );
                    height  = cvRound( heightFloat );

                    probimg(topY, leftX) = aParticle[4];
                }

                probimg.convert2img().display( ("Particle Weight Before Update:"+ m_simpleTrackerParamsPtr->m_displayFigureNameStr).c_str(), 2 );
                cvWaitKey(1);
            }

            //Update particle weight 
            UpdateParticleWeights( m_liklihoodProbabilityList );

            //Draw the weight of all unique particles
            if ( m_simpleTrackerParamsPtr->m_debugv )
            {
                Matrixf probimg( frameHeight, frameWidth );
                vectorf aParticle; //stores a particle

                int leftX, topY, width, height;

                for( int p = 0; p < m_particleFilterTrackerParamsPtr->m_numberOfParticles; p++ )
                {
                    m_particleFilterPtr->GetParticle( p, aParticle);
                    
                    widthFloat    = aParticle[2] * m_currentStateList[2]; 
                    heightFloat    = aParticle[3] * m_currentStateList[3]; 
                    leftX    = cvRound( aParticle[0] - widthFloat / 2 );
                    topY    = cvRound( aParticle[1] - heightFloat / 2 );
                    width    = cvRound( widthFloat );
                    height  = cvRound( heightFloat );

                    probimg(topY, leftX) = aParticle[4];
                }

                probimg.convert2img().display( ("Particle Weight after update:"+ m_simpleTrackerParamsPtr->m_displayFigureNameStr).c_str(), 2 );
                cvWaitKey(1);
            }

            EstimateGroundPoint();

            return maximumResponse; //return maximum m_liklihoodProbabilityList    
        }

        EXCEPTION_CATCH_AND_ABORT("Error while tracking the object on a given frame" )        
    }

    /**************************************************************************
    ParticleFilterTracker::GenerateTrainingSampleSet
        Generate training sample sets for the particle filter tracker
    Exceptions:
        None
    **************************************************************************/
    void ParticleFilterTracker::GenerateTrainingSampleSet(    Matrixu*    pFrameImageColor,
                                                            Matrixu*    pFrameImageGray,
                                                            Matrixu*    pFrameImageHSV )
    {
        try
        {
            // train classifier (m_negativeSampleSet are randomly selected from image, m_positiveSampleSet is just the current tracker location)  
            vectorf aParticle;
            if( m_particleFilterTrackerParamsPtr->m_outputTrajectoryOption == PARTICLE_HIGHEST_WEIGHT )
            {
                m_particleFilterPtr->GetOrderedUniqueParticles( 0, aParticle);    
            }
            else
            {
                m_particleFilterPtr->GetAverageofAllParticles( aParticle );
            }

            //set m_currentStateList as the particle of highest weight
            float widthFloat    = aParticle[2] * m_currentStateList[2]; 
            float heightFloat    = aParticle[3] * m_currentStateList[3]; 
            
            m_currentStateList[0] = max( aParticle[0] - widthFloat / 2, 0.0f );
            m_currentStateList[1] = max( aParticle[1] - heightFloat / 2, 0.0f );
            //m_currentStateList[2], m_currentStateList[3] does not change 
            m_currentStateList[4] = aParticle[2];    
            m_currentStateList[5] = aParticle[3];

            LOG( endl << "m_currentStateList[0]:" << m_currentStateList[0] << "," <<
                 "m_currentStateList[1]:" << m_currentStateList[1] << "," << 
                 "m_currentStateList[2]:" << m_currentStateList[2] << "," << 
                 "m_currentStateList[3]:" << m_currentStateList[3] << endl );


            //generate positive sample set
            GeneratePositiveTrainingSampleSet( pFrameImageColor, pFrameImageGray, pFrameImageHSV );

            //generate negative sample set
            GenerateNegativeTrainingSampleSet( pFrameImageColor, pFrameImageGray, pFrameImageHSV );

        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to Generate Training Classifier::Sample Sets for Particle Filter");
    }


    /**************************************************************************
    ParticleFilterTracker::GeneratePositiveTrainingSampleSet
        Generate positive sample set
    Exceptions:
        None
    **************************************************************************/
    void    ParticleFilterTracker::GeneratePositiveTrainingSampleSet(    Matrixu*    pFrameImageColor, 
                                                                        Matrixu*    pFrameImageGray, 
                                                                        Matrixu*    pFrameImageHSV )
    {
        try
        {
            vectorf aParticle; //stores a particle
            int leftX, topY, width, height;
            float widthFloat, heightFloat; 

            int numOfSamples;
            int frameWidth, frameHeight; 

            if ( pFrameImageGray!= NULL )
            {    
                frameWidth = pFrameImageGray->cols();
                frameHeight = pFrameImageGray->rows();
            }
            else
            {
                frameWidth = pFrameImageColor->cols();
                frameHeight = pFrameImageColor->rows();
            }

            m_positiveSampleSet.Clear();
            ASSERT_TRUE( !m_positiveSampleSet.IsFeatureComputed() );

            // Generate positive examples
            switch ( m_particleFilterTrackerParamsPtr->m_positiveSampleStrategy )
            {
                case SAMPLE_POS_SIMPLETRACKER:

                    //similar to SimpleTracker, with m_currentStateList (average of all particle by default, 
                    m_positiveSampleSet.SampleImage( pFrameImageGray,                    //pGrayImageMatrix
                                                    (int)m_currentStateList[0],            //x - left
                                                    (int)m_currentStateList[1],            //y - top
                                                    (int)m_currentStateList[2],            //width
                                                    (int)m_currentStateList[3],            //height
                                                    m_simpleTrackerParamsPtr->m_posRadiusTrain, //outerCircleRadius
                                                    0,                                    //innerCircleRadius
                                                    m_simpleTrackerParamsPtr->m_maximumNumberOfPositiveTrainingSamples,    //maximumNumberOfSamples
                                                    pFrameImageColor,                    //pRGBImageMatrix
                                                    pFrameImageHSV,                        //pHSVImageMatrix
                                                    m_currentStateList[4],                //scaleX
                                                    m_currentStateList[5] );            //scaleY
                break; //SAMPLE_POS_SIMPLETRACKER

                case SAMPLE_POS_GREEDY:
                    LOG( "USING POS_GREEDY");
                    // Pick the particle based on weights (decreasing order) until reaching the maximum number of positive examples allowed.
                    // If there are not enough unique particles, simply use all unique particles.
                    numOfSamples = min( m_particleFilterPtr->GetNumberOfOrderedUniqueParticles(), m_particleFilterTrackerParamsPtr->m_maxNumPositiveExamples );

                    for    ( int particleIndex=0; particleIndex < numOfSamples; particleIndex++ )
                    {
                        m_particleFilterPtr->GetOrderedUniqueParticles( particleIndex, aParticle );

                        
                        widthFloat    = aParticle[2] * m_currentStateList[2]; 
                        heightFloat    = aParticle[3] * m_currentStateList[3]; 
                        leftX        = cvRound( aParticle[0] - widthFloat / 2 );
                        topY        = cvRound( aParticle[1] - heightFloat / 2 );
                        width        = cvRound( widthFloat );
                        height        = cvRound( heightFloat );

                    
                        if ( aParticle[4] != 0 
                            && leftX > 0 && topY > 0  
                            && leftX + width < frameWidth - 1
                            && topY + height < frameHeight -1 )
                        {    
                            //within images range,
                            // use each of the unique particle as a sample
                            m_positiveSampleSet.PushBackSample( pFrameImageGray,
                                                                leftX,
                                                                topY ,
                                                                cvRound( m_currentStateList[2] ),
                                                                cvRound( m_currentStateList[3] ),
                                                                1,
                                                                pFrameImageColor,
                                                                pFrameImageHSV,
                                                                aParticle[2],
                                                                aParticle[3] ); 
                        }                            
                    }

                    if ( m_positiveSampleSet.Size( ) == 0 )
                    {
                        LOG( "Failed to generate samples with original strategy, trying with GREEDY " );
                        m_positiveSampleSet.SampleImage( pFrameImageGray,                    //pGrayImageMatrix
                            (int)m_currentStateList[0],            //x - left
                            (int)m_currentStateList[1],            //y - top
                            (int)m_currentStateList[2],            //width
                            (int)m_currentStateList[3],            //height
                            m_simpleTrackerParamsPtr->m_posRadiusTrain, //outerCircleRadius
                            0,                                    //innerCircleRadius
                            m_simpleTrackerParamsPtr->m_maximumNumberOfPositiveTrainingSamples,    //maximumNumberOfSamples
                            pFrameImageColor,                    //pRGBImageMatrix
                            pFrameImageHSV,                        //pHSVImageMatrix
                            m_currentStateList[4],                //scaleX
                            m_currentStateList[5] );            //scaleY
                    }    

                    ASSERT_TRUE( m_positiveSampleSet.Size() > 0 );
                break; //SAMPLE_POS_GREEDY

                case SAMPLE_POS_SEMI_GREEDY:
                {
                    LOG( "USING POS_SEMI_GREEDY");
                    //Pick particles that are close to the average center that are ordered based on weight.
                    numOfSamples = min( m_particleFilterPtr->GetNumberOfOrderedUniqueParticles(), m_particleFilterTrackerParamsPtr->m_maxNumPositiveExamples );

                    float currentStateCenterX = m_currentStateList[0] +  m_currentStateList[2] * m_currentStateList[4] / 2;
                    float currentStateCenterY = m_currentStateList[1] +  m_currentStateList[3] * m_currentStateList[5] / 2;

                    for ( int particleIndex=0; particleIndex < numOfSamples; particleIndex++ )
                    {
                        m_particleFilterPtr->GetOrderedUniqueParticles( particleIndex, aParticle );

                        widthFloat    = aParticle[2] * m_currentStateList[2]; 
                        heightFloat    = aParticle[3] * m_currentStateList[3]; 
                        leftX        = cvRound( aParticle[0] - widthFloat / 2 );
                        topY        = cvRound( aParticle[1] - heightFloat / 2 );
                        width        = cvRound( widthFloat );
                        height        = cvRound( heightFloat );


                        float distanceFromTheCenter = sqrt( pow( ( currentStateCenterX - aParticle[0] ) , 2  ) + pow( ( currentStateCenterY - aParticle[1] ), 2 ) );
                        if (    aParticle[4] != 0 
                                && leftX > 0 && topY > 0  
                                && leftX + width < frameWidth - 1
                                && topY + height < frameHeight -1 )
                        {    //within images range,
                            if ( distanceFromTheCenter < 8 )
                            {
                                m_positiveSampleSet.PushBackSample( pFrameImageGray,
                                                                    leftX,
                                                                    topY , 
                                                                    (int)m_currentStateList[2], 
                                                                    (int)m_currentStateList[3],
                                                                    1,
                                                                    pFrameImageColor, 
                                                                    pFrameImageHSV,
                                                                    aParticle[2],
                                                                    aParticle[3] );   
                            }
                        }
                    }
                }
                break;//SAMPLE_POS_SEMI_GREEDY

                case SAMPLE_POS_PARTICLE_RANDOM:
                {
                    LOG( "USING SAMPLE_POS_PARTICLE_RANDOM");
                    float prob = ((float)( m_particleFilterTrackerParamsPtr->m_maxNumPositiveExamples )) 
                        / m_particleFilterPtr->GetNumberOfParticles();

                    int i = 0;

                    m_positiveSampleSet.Resize( m_particleFilterPtr->GetNumberOfParticles() ); 

                    for ( int particleIndex = 0; particleIndex < m_particleFilterPtr->GetNumberOfParticles(); particleIndex++ )
                    {
                        m_particleFilterPtr->GetResampledParticle( particleIndex, aParticle );
                        
                        widthFloat    = aParticle[2] * m_currentStateList[2]; 
                        heightFloat    = aParticle[3] * m_currentStateList[3]; 
                        leftX        = cvRound( aParticle[0] - widthFloat / 2 );
                        topY        = cvRound( aParticle[1] - heightFloat / 2 );
                        width        = cvRound( widthFloat );
                        height        = cvRound( heightFloat );

                        if ( leftX > 0 && topY > 0  
                            && leftX + width < frameWidth - 1
                            && topY + height < frameHeight -1 )
                        {
                            if ( randfloat( ) < prob )
                            {    
                                // keep this particle as a positive training sample
                                m_positiveSampleSet[i].m_pImgGray    = pFrameImageGray;
                                m_positiveSampleSet[i].m_pImgColor    = pFrameImageColor;
                                m_positiveSampleSet[i].m_pImgHSV    = pFrameImageHSV;
                                m_positiveSampleSet[i].m_col        = leftX;
                                m_positiveSampleSet[i].m_row        = topY;
                                m_positiveSampleSet[i].m_height        = cvRound( m_currentStateList[3] );
                                m_positiveSampleSet[i].m_width        = cvRound( m_currentStateList[2] );
                                m_positiveSampleSet[i].m_scaleX        = aParticle[2];
                                m_positiveSampleSet[i].m_scaleY        = aParticle[3];

                                i++;
                            }
                        }
                    }
                    m_positiveSampleSet.Resize( min( i, m_particleFilterTrackerParamsPtr->m_maxNumPositiveExamples ) );
                }
                break; //SAMPLE_POS_PARTICLE_RANDOM

                default:
                    abortError( __LINE__, __FILE__, "Unsupported particle filter positive sampling strategy" );
                break;
            }

            if ( m_positiveSampleSet.Size() == 0 )
            {
                m_particleFilterPtr->ForceParticleFilterRefinement( m_currentStateList[2], m_currentStateList[3] );
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to generate positive sample set" );
    }

    /**************************************************************************
    ParticleFilterTracker::GenerateNegativeTrainingSampleSet
        generate negative sample set based on the current particle state
    Exceptions:
        None
    **************************************************************************/
    void    ParticleFilterTracker::GenerateNegativeTrainingSampleSet(    Matrixu*    pFrameImageColor, 
                                                                        Matrixu*    pFrameImageGray, 
                                                                        Matrixu*    pFrameImageHSV  )
    {
        try
        {
            switch( m_particleFilterTrackerParamsPtr->m_negativeSampleStrategy ) 
            {
                case SAMPLE_NEG_SIMPLETRACKER: 
                {
                    //similar to the SimpleTracker around average (or most weighted) particle with average scale
                    m_negativeSampleSet.SampleImage( pFrameImageGray,            //pGrayImageMatrix
                                                    cvRound( m_currentStateList[0] ),    //leftX
                                                    cvRound( m_currentStateList[1] ),    //topY
                                                    cvRound( m_currentStateList[2] ),    //width
                                                    cvRound( m_currentStateList[3] ),   //height
                                                    1.5f * m_simpleTrackerParamsPtr->m_searchWindSize,    //outerCircleRadius
                                                    m_simpleTrackerParamsPtr->m_posRadiusTrain + 15,        //innerCircleRadius
                                                    m_simpleTrackerParamsPtr->m_numberOfNegativeTrainingSamples,            //maximumNumberOfSamples
                                                    pFrameImageColor,            //pRGBImageMatrix
                                                    pFrameImageHSV,                //pHSVImageMatrix
                                                    std::min( m_currentStateList[4], m_particleFilterPtr->GetMaxScale() ),        //scaleX
                                                    std::min( m_currentStateList[5], m_particleFilterPtr->GetMaxScale() ) );    //scaleY
                }
                break; //SAMPLE_NEG_SIMPLETRACKER

                case SAMPLE_NEG_PARTICLE_RANDOM://around mean weighted particles
                {
                    float maximumDistanceX, maximumDistanceY;
                    float minimumDistanceX = 0.0f, minimumDistanceY = 0.0f;


                    float currentStateCenterX = m_currentStateList[0] +  m_currentStateList[2] * m_currentStateList[4] / 2;
                    float currentStateCenterY = m_currentStateList[1] +  m_currentStateList[3] * m_currentStateList[5] / 2;

                    for ( int particleIndex=0; particleIndex < m_particleFilterPtr->GetNumberOfOrderedUniqueParticles(); particleIndex++ )
                    { 
                        vectorf aParticle;

                        // find the rectangle boundary for all particle from the current state
                        m_particleFilterPtr->GetOrderedUniqueParticles( particleIndex, aParticle );

                        //update minimum distance based on current particle
                        if ( abs( aParticle[0] - currentStateCenterX ) > minimumDistanceX )    
                        {
                            minimumDistanceX =  abs( aParticle[0] - currentStateCenterX );
                        }
                        if ( abs( aParticle[1] - currentStateCenterY ) > minimumDistanceY )                        
                        {
                            minimumDistanceY = abs( aParticle[1] - currentStateCenterY );
                        }
                    }

                    ASSERT_TRUE( minimumDistanceX >= 0 && minimumDistanceY >= 0 );

                    //set the minimum distances based so that negative window region is away from the positive region
                    minimumDistanceX = max( minimumDistanceX, (float)m_particleFilterTrackerParamsPtr->m_posRadiusTrain+5);
                    minimumDistanceY = max( minimumDistanceY, (float)m_particleFilterTrackerParamsPtr->m_posRadiusTrain+5);

                    //also ensure that it's not too far way from the object
                    minimumDistanceX = min( minimumDistanceX, 15.0f );
                    minimumDistanceY = min( minimumDistanceY, 15.0f );

                    //set the maximum distance away from minimum distance
                    maximumDistanceX = (1.5f*m_simpleTrackerParamsPtr->m_searchWindSize);
                    maximumDistanceY = (1.5f*m_simpleTrackerParamsPtr->m_searchWindSize);

                    m_negativeSampleSet.SampleImage( pFrameImageGray,            //pGrayImageMatrix
                                                    cvRound( m_currentStateList[0] ),    //leftX
                                                    cvRound( m_currentStateList[1] ),    //topY
                                                    cvRound( m_currentStateList[2] ),    //width
                                                    cvRound( m_currentStateList[3] ),   //height
                                                    maximumDistanceX,            //maximum x distance away from the center
                                                    maximumDistanceY,            //maximum y distance away from the center
                                                    minimumDistanceX,            //minimum x distance away from the center
                                                    minimumDistanceY,            //minimum y distance away from the center
                                                    m_simpleTrackerParamsPtr->m_numberOfNegativeTrainingSamples,//maximumNumberOfSamples
                                                    pFrameImageColor,            //pRGBImageMatrix
                                                    pFrameImageHSV,                //pHSVImageMatrix
                                                    std::min( m_currentStateList[4], m_particleFilterPtr->GetMaxScale() ),        //scaleX
                                                    std::min( m_currentStateList[5], m_particleFilterPtr->GetMaxScale() ) );    //scaleY
                }
                break;

                default: 
                    abortError( __LINE__, __FILE__, "Unsupported particle filter negative sampling strategy" );
                    break;
                }
            ASSERT_TRUE( m_negativeSampleSet.Size() > 0 );            
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to generate negative sample set" );
    }
                                                     
    /**************************************************************************
    UpdateClassifier
        Update classifier with the current particle state
        Extract examples->classifier Update        
    Exceptions:
        None
    **************************************************************************/
    void    ParticleFilterTracker::UpdateClassifier( Matrixu*    pFrameImageColor, 
                                                      Matrixu*    pFrameImageGray, 
                                                     Matrixu*    pFrameDisplayTraining,
                                                     Matrixu*    pFrameImageHSV ) 
    {
        try
        {
            ASSERT_TRUE( pFrameImageColor != NULL || pFrameImageGray!= NULL );
    
            GenerateTrainingSampleSet( pFrameImageColor, pFrameImageGray, pFrameImageHSV );

            m_strongClassifierBasePtr->Update(m_positiveSampleSet,m_negativeSampleSet);
            
            DisplayTrainingSamples( pFrameDisplayTraining );
            
            m_positiveSampleSet.Clear();
            m_negativeSampleSet.Clear();            
        }
        EXCEPTION_CATCH_AND_ABORT("Error while updating the classifier" )        
    }

    /**************************************************************************
    UpdateParticlesWithGroundPDF
        Update Particle Weights With Ground PDF
    Exceptions:
        None
    **************************************************************************/
    void ParticleFilterTracker::UpdateParticlesWithGroundPDF(    CvMat*        pMeanMatrix,
                                                                CvMat*        pCovarianceMatrix,
                                                                Matrixu*    pColorImageMatrix,
                                                                Matrixu*    pGrayImageMatrix,
                                                                Matrixu*    pHsvImageMatrix,
                                                                const bool    shouldUseAppFusionWeights)
    {
        try
        {
            ASSERT_TRUE( pMeanMatrix != NULL );
            ASSERT_TRUE( pCovarianceMatrix != NULL );


            //estimate the ground plane location of the current particles
            EstimateGroundPoint( false /*shouldUseResampledParticles*/ );

            CvMat* pHomographyMatrix = GetHomographyMatrix();
            CvMat* pGroundPlaneParticles = GeometryBasedInformationFuser::TransformWithHomography( m_pGroundLocation, GetHomographyMatrix() );

            CvMat* data = cvCreateMat( 2, 1, CV_32FC1 );
            ASSERT_TRUE( data != NULL );

            vectorf updatedWeights( m_particleFilterTrackerParamsPtr->m_numberOfParticles );

            if ( static_cast<double>( cvDet( pCovarianceMatrix ) ) >= 0.0   && pCovarianceMatrix->data.fl[0]> 0 )
            { 
                #pragma omp parallel for
                for ( int i=0; i < m_particleFilterTrackerParamsPtr->m_numberOfParticles; i++ )
                {
                    data->data.fl[0]    = static_cast<float>(pGroundPlaneParticles->data.db[i*2]);
                    data->data.fl[1]    = static_cast<float>(pGroundPlaneParticles->data.db[i*2+1]);

                    updatedWeights[i] = MultipleCameraTracking::GeometryBasedInformationFuser::MultiVariateNormalPdf( data, pMeanMatrix, pCovarianceMatrix );
                }
            }
            else //if Kalman filter has weird results, assign equal weight to all particle
            {
                updatedWeights.assign( m_particleFilterTrackerParamsPtr->m_numberOfParticles, 1.0 );
            }
            

            float totalWeight = 0.0f;
            for ( int i=0; i < m_particleFilterTrackerParamsPtr->m_numberOfParticles; i++ )
            {
                totalWeight += updatedWeights[i];
            }

            LOG( "\nTotalWeight:" << totalWeight );

            if ( totalWeight > 0.0 )
            {
                LOG( "\n############Updating the weights after fusion###########" );

                //Update particle weight 
                //m_particleFilterPtr->UpdateAllParticlesWeight( updatedWeights, false/*shouldUpdateNonZeroWeights*/ );

                FindMeanFootPositionOnImagePlane( pMeanMatrix );
                vectorf particle(5);
                m_particleFilterPtr->GetAverageofAllParticles(particle);
                vectorf averageParticleFootPosition(2);
                averageParticleFootPosition[0] = particle[0];
                averageParticleFootPosition[1] = particle[1] + m_currentStateList[3] * particle[3] ;


                double distance = sqrt( pow( m_meanGroundPlaneFootPositionOnImageList[0] - averageParticleFootPosition[0],2) + pow(averageParticleFootPosition[1]-m_meanGroundPlaneFootPositionOnImageList[1],2) );
                if ( distance < 20 )
                {
                    LOG( "Not rearranging the particles.");
                    return;
                }

                vectorf covarianceGroundPlaneFootPositionImageList( 2, 10.0 );

                m_particleFilterPtr->RearrangeParticlesBasedOnGroundLocation(    m_meanGroundPlaneFootPositionOnImageList, //m_meanGroundPlaneFootPositionOnImageList
                                                                                covarianceGroundPlaneFootPositionImageList,
                                                                                true,                    //shouldChangeTheScaleAccordingToOffset
                                                                                m_currentStateList[2],    //width
                                                                                m_currentStateList[3]    //height
                                                                            );

                UpdateParticleWeightsForRearrangedParticles( pColorImageMatrix, pGrayImageMatrix, pHsvImageMatrix, shouldUseAppFusionWeights );            
            }
            cvReleaseMat(&data);         
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to update the particle weights based on ground plane pdf" );
    }


    /**************************************************************************
    EstimateGroundPoint
        Find the ground plane point on the tracked blob.
    Exceptions:
        None
    **************************************************************************/
    void ParticleFilterTracker::EstimateGroundPoint( bool shouldUseResampledParticle )
    {
        try
        {
            if ( shouldUseResampledParticle )
            {
                m_particleFilterPtr->ResampleParticles( false /*shouldForceFilterStateChange*/ );
            }

            vectorf particle; 
            for (int i=0; i < m_particleFilterTrackerParamsPtr->m_numberOfParticles; i++)
            {   
                if ( shouldUseResampledParticle )
                {
                    m_particleFilterPtr->GetResampledParticle( i, particle );
                }
                else
                {
                    m_particleFilterPtr->GetParticle( i, particle );
                }
                m_pGroundLocation->data.fl[i*2]        = particle[0];    //centerX    
                m_pGroundLocation->data.fl[i*2 + 1] = particle[1] + particle[3] * m_currentStateList[3]/2; //y
            }
        }
        EXCEPTION_CATCH_AND_ABORT("Error while estimating object's ground location on the image" )        
    }

    /**************************************************************************
    FindMeanFootPositionOnImagePlane
        Find particle offset from the mean ground plane location
    Exceptions:
        None
    **************************************************************************/
    vectorf ParticleFilterTracker::FindMeanFootPositionOnImagePlane( CvMat* pMeanGroundPlaneLocationMatrix )
    {
        try
        {
            CvMat* pHomographyMatrix = GetHomographyMatrix();
            CvMat* pInverseHomographyMatrix = cvCreateMat( 3, 3, CV_64FC1 );
            
            cvInv( pHomographyMatrix, pInverseHomographyMatrix );

            CvMat* pMeanGroundParticlesFromFusion = GeometryBasedInformationFuser::TransformWithHomography( pMeanGroundPlaneLocationMatrix, pInverseHomographyMatrix );
        
            m_meanGroundPlaneFootPositionOnImageList[0] = static_cast<float>(pMeanGroundParticlesFromFusion->data.db[0]);
            m_meanGroundPlaneFootPositionOnImageList[1] = static_cast<float>(pMeanGroundParticlesFromFusion->data.db[1]);

            ASSERT_TRUE( m_meanGroundPlaneFootPositionOnImageList.size() == 2 );

            LOG( "Mean ground plane foot position on image:[" << m_meanGroundPlaneFootPositionOnImageList[0] <<
                "," << m_meanGroundPlaneFootPositionOnImageList[1] << "]" );
            
            cvReleaseMat( &pMeanGroundParticlesFromFusion );
            return m_meanGroundPlaneFootPositionOnImageList;
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to Mean ground plane foot position on image plane" );        
    }

    /********************************************************************************************************
    DrawObjectFootPosition
        Draw the mean ground plane location on a given image
    Exceptions:
        None
    ********************************************************************************************************/
    void    ParticleFilterTracker::DrawObjectFootPosition( Matrixu* pFrameDisplay ) const
    {
        try
        {
            if ( pFrameDisplay != NULL )
            {
                pFrameDisplay->drawEllipse(    5,
                                            5,
                                            m_meanGroundPlaneFootPositionOnImageList[0],
                                            m_meanGroundPlaneFootPositionOnImageList[1],
                                            1,
                                            255,
                                            0,
                                            0 );
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "ParticleFilterTracker:: Failed to draw mean ground plane foot position on image plane" );        
    }

    
    /**************************************************************************
    GetAverageParticleOnGroundMatrix
    Exceptions:
        None
    **************************************************************************/
    CvMat* ParticleFilterTracker::GetAverageParticleOnGroundMatrix( )
    { 
        try
        {
            vectorf particle(4);
            //m_particleFilterPtr->GetAverageofAllParticles(particle);

            m_particleFilterPtr->GetHighestWeightParticle(particle);

            m_pWeightedAverageParticleMatrix->data.db[0] = particle[0];
            m_pWeightedAverageParticleMatrix->data.db[1] = particle[1] + particle[3] * m_currentStateList[3]/2; //y

            return m_pWeightedAverageParticleMatrix; 
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to calculate average particle matrix" );
    }

    
    /**************************************************************************
    GenerateTestSampleSet
        Generate Test Sample Set.
    Exceptions:
        None
    **************************************************************************/
    void ParticleFilterTracker::GenerateTestSampleSet(    Matrixu* pFrameImageColor,
                                                        Matrixu* pFrameImageGray,                                                         
                                                        Matrixu* pFrameImageHSV )
    {
        try 
        {
            int frameWidth, frameHeight;             
            if ( pFrameImageGray!= NULL )
            {    
                frameWidth = pFrameImageGray->cols();
                frameHeight = pFrameImageGray->rows();
            }
            else
            {
                frameWidth = pFrameImageColor->cols();
                frameHeight = pFrameImageColor->rows();
            }
    
            //clear the test sample set explicitly
            m_testSampleSet.Clear();
            ASSERT_TRUE( !m_testSampleSet.IsFeatureComputed() );

            //check blob dimension for predicated particles and add them as candidates for checking by classifier
            for( int p = 0; p < m_particleFilterTrackerParamsPtr->m_numberOfParticles; p++ )
            {
                vectorf particle;
                m_particleFilterPtr->GetParticle( p, particle);
    
                float widthFloat    = particle[2] * m_currentStateList[2]; 
                float heightFloat    = particle[3] * m_currentStateList[3]; 
                float leftX            = cvRound( particle[0] - widthFloat / 2 );
                float topY            = cvRound( particle[1] - heightFloat / 2 );
                float width            = cvRound( widthFloat );
                float height        = cvRound( heightFloat );
    
                //Update the weight of invalid particles to zero
                if ( leftX <= 0 || leftX + width >= frameWidth - 1 || 
                    topY <= 0  || topY + height >= frameHeight -1       )
                {    
                    //outside images range, assign zero weight
                    m_particleFilterPtr->UpdateParticleWeight(p, 0);
                }
                else if ( particle[4] != 0 ) 
                {
                    m_testSampleSet.PushBackSample( pFrameImageGray, 
                                                    leftX,
                                                    topY,
                                                    (int)m_currentStateList[2],
                                                    (int)m_currentStateList[3],
                                                    1, 
                                                    pFrameImageColor,
                                                    pFrameImageHSV,
                                                    particle[2],
                                                    particle[3] );
                } 
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to generate Test Sample Set" );
    }

    /**************************************************************************
    UpdateParticleWeights
        Update Particle Weights
    Exceptions:
        None
    **************************************************************************/
    void ParticleFilterTracker::UpdateParticleWeights(  vectorf&    likelihoodList, 
                                                        bool        shouldUpdateOnlyNonZeroWeights,
                                                        bool        shouldForceReset,
                                                        bool        shouldResample )
    { 
        try
        {
            m_particleFilterPtr->UpdateAllParticlesWeight( likelihoodList,
                                                           shouldUpdateOnlyNonZeroWeights, 
                                                           shouldForceReset,
                                                           shouldResample );
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to update particle weights" );
    }

    /**************************************************************************
    UpdateParticleWeightsForRearrangedParticles
        Update Particle Weights For Rearranged Particles
    Exceptions:
        None
    **************************************************************************/
    void ParticleFilterTracker::UpdateParticleWeightsForRearrangedParticles( Matrixu*    pFrameImageColor, 
                                                                             Matrixu*    pFrameImageGray,
                                                                             Matrixu*    pFrameImageHSV,
                                                                             const bool shouldUseAppFusionWeights )
    {
        try
        {
            int frameWidth, frameHeight;             
            if ( pFrameImageGray!= NULL )
            {    
                frameWidth = pFrameImageGray->cols();
                frameHeight = pFrameImageGray->rows();
            }
            else
            {
                frameWidth = pFrameImageColor->cols();
                frameHeight = pFrameImageColor->rows();
            }

            vectorf aParticle;        
            int leftX, topY, width, height;
            float widthFloat, heightFloat;

            //tracker should have been initialized before calling this
            ASSERT_TRUE( m_isInitialized );
            ASSERT_TRUE( pFrameImageColor != NULL || pFrameImageGray!= NULL );

            //Generate Test Sample Set
            GenerateTestSampleSet( pFrameImageColor, pFrameImageGray, pFrameImageHSV );

            if ( m_testSampleSet.Size() == 0 )
            {
                m_particleFilterPtr->ForceParticleFilterRefinement( m_currentStateList[2], m_currentStateList[3] );

                GenerateTestSampleSet( pFrameImageColor, pFrameImageGray, pFrameImageHSV );

                ASSERT_TRUE( m_testSampleSet.Size( ) != 0 );
            }

            //test with the classifier
            m_liklihoodProbabilityList = m_strongClassifierBasePtr->Classify( m_testSampleSet, m_simpleTrackerParamsPtr->m_shouldNotUseSigmoid );

            //draw H(x) results for each candidate blob
            if ( m_simpleTrackerParamsPtr->m_debugv )
            {
                Matrixf probimg( frameHeight, frameWidth );

                for( uint k=0; k<(uint)m_testSampleSet.Size(); k++ )
                {
                    probimg(m_testSampleSet[k].m_row, m_testSampleSet[k].m_col) = m_liklihoodProbabilityList[k];
                }
                probimg.convert2img().display(    ("Probability map, "+ m_simpleTrackerParamsPtr->m_displayFigureNameStr).c_str(),
                    2 );
                cvWaitKey(1);
            }

            int bestind, worstind;
            double maximumResponse = 0.0, minimumResponse = 0.0;    
            bestind        = max_idx( m_liklihoodProbabilityList ); // find best location
            worstind    = min_idx( m_liklihoodProbabilityList ); 
            if ( !m_liklihoodProbabilityList.empty() )
            {
                maximumResponse    = m_liklihoodProbabilityList[bestind];
                minimumResponse = m_liklihoodProbabilityList[worstind];
            }

            double range = maximumResponse - minimumResponse;

            //ASSERT_TRUE( range != 0 );

            for ( int index = 0; index < m_liklihoodProbabilityList.size(); index++ )
            {
                if(range != 0)
                    m_liklihoodProbabilityList[index] = float ( pow( (m_liklihoodProbabilityList[index] - minimumResponse) / range,1) ); 
                //m_liklihoodProbabilityList[index] = sigmoid( m_liklihoodProbabilityList[index]  );
                else
                {
                    m_liklihoodProbabilityList[index] = 1;
                }
            }

            //draw likelihood ( based on H(x)) results for each candidate blob
            if ( m_simpleTrackerParamsPtr->m_debugv )
            {
                Matrixf probimg( frameHeight, frameWidth );

                for( uint k=0; k<(uint)m_testSampleSet.Size(); k++ )
                {//in terms of sample (topY, leftX) instead of center as in the particleFilter
                    probimg(m_testSampleSet[k].m_row, m_testSampleSet[k].m_col) = m_liklihoodProbabilityList[k];
                }
                probimg.convert2img().display( ("Probability map2, "+ m_simpleTrackerParamsPtr->m_displayFigureNameStr).c_str(), 2 );
                cvWaitKey(1);
            }

            //draw the weight of all unique particles
            if ( m_simpleTrackerParamsPtr->m_debugv )
            {
                Matrixf probimg( frameHeight, frameWidth );

                for( int p = 0; p < m_particleFilterTrackerParamsPtr->m_numberOfParticles; p++ )
                {
                    m_particleFilterPtr->GetParticle( p, aParticle);
                    widthFloat    = aParticle[2] * m_currentStateList[2]; 
                    heightFloat    = aParticle[3] * m_currentStateList[3]; 
                    leftX    = cvRound( aParticle[0] - widthFloat / 2 );
                    topY    = cvRound( aParticle[1] - heightFloat / 2 );
                    width    = cvRound( widthFloat );
                    height  = cvRound( heightFloat );

                    probimg(topY, leftX) = aParticle[4];
                }

                probimg.convert2img().display( ("Particle Weight Before Update:"+ m_simpleTrackerParamsPtr->m_displayFigureNameStr).c_str(), 2 );
                cvWaitKey(1);
            }

            //Update particle weight 
            UpdateParticleWeights( m_liklihoodProbabilityList, true/*shouldUpdateOnlyNonZeroWeights*/, true/*shouldForceReset*/, true/*shouldResample*/ );

            //Draw the weight of all unique particles
            if ( m_simpleTrackerParamsPtr->m_debugv )
            {
                Matrixf probimg( frameHeight, frameWidth );
                vectorf aParticle; //stores a particle

                int leftX, topY, width, height;

                for( int p = 0; p < m_particleFilterTrackerParamsPtr->m_numberOfParticles; p++ )
                {
                    m_particleFilterPtr->GetParticle( p, aParticle);

                    widthFloat    = aParticle[2] * m_currentStateList[2]; 
                    heightFloat    = aParticle[3] * m_currentStateList[3]; 
                    leftX    = cvRound( aParticle[0] - widthFloat / 2 );
                    topY    = cvRound( aParticle[1] - heightFloat / 2 );
                    width    = cvRound( widthFloat );
                    height  = cvRound( heightFloat );

                    probimg(topY, leftX) = aParticle[4];
                }

                probimg.convert2img().display( ("Particle Weight after update:"+ m_simpleTrackerParamsPtr->m_displayFigureNameStr).c_str(), 2 );
                cvWaitKey(1);
            }

            EstimateGroundPoint();        
        }
        EXCEPTION_CATCH_AND_ABORT( "Update Particle Weights For Rearranged Particles" );
    }
}
