#include "StrongClassifierBase.h"
#include "StrongClassifierFactory.h"
#include "SimpleTracker.h"
#include "Public.h"
#include "Sample.h"
#include "CommonMacros.h"

#define HAAR_CASCADE_FILE_NAME "haarcascade_frontalface_alt_tree.xml"

namespace MultipleCameraTracking
{
    /****************************************************************
    SimpleTracker::InitializeTracker
        Initializes a simple tracker.
    Exceptions:
        None
    ****************************************************************/
    bool    SimpleTracker::InitializeTracker( Matrixu*                pFrameImageColor, 
                                              Matrixu*                pFrameImageGray,  
                                              TrackerParametersPtr  trackerParametersPtr, 
                                              Classifier::StrongClassifierParametersBasePtr clfparamsPtr,
                                              Matrixu*                pFrameDisplay, 
                                              Matrixu*                pFrameDisplayTraining,
                                              Matrixu*                pFrameImageHSV )
    {
        try
        {
            //Get the frame and initialize the integral image
            m_simpleTrackerParamsPtr    =    boost::static_pointer_cast<SimpleTrackerParameters> (trackerParametersPtr);

            //create a classifier from the classifier parameters
            m_strongClassifierBasePtr = Classifier::StrongClassifierFactory::CreateAndInitializeClassifier( clfparamsPtr );

            //initialize the current state of the object with the initialization(this could be got from a detector)
            m_currentStateList.resize(4);
            for ( int i=0; i < 4; i++ ) 
            {
                m_currentStateList[i] = m_simpleTrackerParamsPtr->m_initState[i];
            }

            if ( clfparamsPtr->m_featureParametersPtr->GetFeatureType() == Features::HAAR_LIKE
                || clfparamsPtr->m_featureParametersPtr->GetFeatureType() == Features::HAAR_COLOR_HISTOGRAM )
            {
                ASSERT_TRUE( pFrameImageGray!=NULL && pFrameImageGray->isInitII() );
            }

            //create positive and negative sample sets
            Classifier::SampleSet positiveSampleSet, negativeSampleSet;

            LOG( "Initializing Simple Tracker..." << endl );
            LOG( "Initial object state: ["
                        << m_currentStateList[0] <<" "<<m_currentStateList[1] << " "
                        << m_currentStateList[2] <<" "<< m_currentStateList[3] << "]." <<endl);

            if( g_verboseMode )
            {
                cout    << "Initializing Simple Tracker..." << endl;
                cout    << "Initial object state: ["
                        << m_currentStateList[0] <<" "<<m_currentStateList[1] << " "
                        << m_currentStateList[2] <<" "<< m_currentStateList[3] << "]." <<endl;;
            }
            // draw a colored box around object for display 
            if ( pFrameDisplay != NULL )
            {
                pFrameDisplay->drawRect( m_currentStateList[2],
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


            // sample positives and negatives for the first frame using the given initialization
            positiveSampleSet.SampleImage( pFrameImageGray,
                                          (uint)m_currentStateList[0],
                                          (uint)m_currentStateList[1], 
                                          (uint)m_currentStateList[2], 
                                          (uint)m_currentStateList[3],
                                          m_simpleTrackerParamsPtr->m_init_posTrainRadius,
                                          0,1000000,
                                          pFrameImageColor,
                                          pFrameImageHSV );

            negativeSampleSet.SampleImage(    pFrameImageGray,
                                            (uint)m_currentStateList[0],
                                            (uint)m_currentStateList[1],
                                            (uint)m_currentStateList[2],
                                            (uint)m_currentStateList[3],
                                            2.0f*m_simpleTrackerParamsPtr->m_searchWindSize,
                                            (1.5f*m_simpleTrackerParamsPtr->m_init_posTrainRadius),
                                            m_simpleTrackerParamsPtr->m_init_negNumTrain,
                                            pFrameImageColor,
                                            pFrameImageHSV );
            
            //should have at least one sample
            if ( positiveSampleSet.Size() < 1 || negativeSampleSet.Size() < 1 )
            {
                return false;
            }

            // train
            m_strongClassifierBasePtr->Update( positiveSampleSet, negativeSampleSet );

            DisplayTrainingSamples( pFrameDisplayTraining );

            //free the memory
            negativeSampleSet.Clear();
            positiveSampleSet.Clear();
        
            m_isInitialized = true;

            return true;    //Successfully initialized
        }
        EXCEPTION_CATCH_AND_ABORT("Error While Initializing the tracker.")
    }

    /********************************************************************
    TrackObjectAndSaveState
        Track object and store the result. 
    Exceptions:
        None
    *********************************************************************/
    void SimpleTracker::TrackObjectAndSaveState( int        frameind, 
                                                  Matrixu*    pFrameImageColor, 
                                                 Matrixu*    pFrameImageGray, 
                                                 Matrixu*    pFrameDisplay, 
                                                 Matrixu*    pFrameDisplayTraining,
                                                 Matrixu*    pFrameImageHSV )
    {
        try
        {
            //track object on the given frame 
            TrackObjectOnTheGivenFrame( pFrameImageColor,pFrameImageGray, pFrameDisplayTraining, pFrameImageHSV );

            // draw a colored box around object for display or video saving
            if ( pFrameDisplay != NULL )
            {
                pFrameDisplay->drawRect( m_currentStateList[2],
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

            //save the state to states matrix
            for( int s=0; s<4; s++ ) 
            {
                m_states(frameind,s) = m_currentStateList[s];
            }
        }
        EXCEPTION_CATCH_AND_ABORT("Error while tracking and saving the object state" )
    }


    /********************************************************************
    DisplayTrainingSamples
        Display Training Samples
    Exceptions:
        None
    *********************************************************************/
    void SimpleTracker::DisplayTrainingSamples( Matrixu* pFrameDisplayTraining )
    {
        try
        {
            //Save/Display the training samples 
            if ( pFrameDisplayTraining!= NULL )
            {
                //draw negative training examples.
                for ( int j=0; j<m_negativeSampleSet.Size(); j++ )
                {
                    //draw  small red ellipses around center
                    if ( m_simpleTrackerParamsPtr->m_displayTrainingSampleCenterOnly )
                    {
                        pFrameDisplayTraining->drawEllipse(    1,
                                                            1,
                                                            float( m_negativeSampleSet[j].m_col + float(m_negativeSampleSet[j].m_width * m_negativeSampleSet[j].m_scaleX ) / 2 ),
                                                            float( m_negativeSampleSet[j].m_row + float(m_negativeSampleSet[j].m_height * m_negativeSampleSet[j].m_scaleY )/ 2 ),
                                                            1,
                                                            255,
                                                            0,
                                                            0 );
                    }
                    //Or draw the actual blob
                    else
                    {
                        pFrameDisplayTraining->drawEllipse( float( m_negativeSampleSet[j].m_height * m_negativeSampleSet[j].m_scaleY )/2,
                                                            float( m_negativeSampleSet[j].m_width * m_negativeSampleSet[j].m_scaleX )/2,
                                                            float( m_negativeSampleSet[j].m_col + float(m_negativeSampleSet[j].m_width * m_negativeSampleSet[j].m_scaleX )/2 ),
                                                            float( m_negativeSampleSet[j].m_row + float(m_negativeSampleSet[j].m_height * m_negativeSampleSet[j].m_scaleY )/2 ),
                                                            1, //line width
                                                            255,
                                                            0,
                                                            0 ); 
                    }                    
                }

                // draw positive examples
                for ( int j=0; j<m_positiveSampleSet.Size(); j++ )
                {
                    //draw small blue ellipses around the center
                    if ( m_simpleTrackerParamsPtr->m_displayTrainingSampleCenterOnly )
                    {
                        pFrameDisplayTraining->drawEllipse( 1,
                                                            1,
                                                            float( m_positiveSampleSet[j].m_col + float(m_positiveSampleSet[j].m_width * m_positiveSampleSet[j].m_scaleX )/2 ),
                                                            float( m_positiveSampleSet[j].m_row + float(m_positiveSampleSet[j].m_height * m_positiveSampleSet[j].m_scaleY)/2 ),
                                                            1,
                                                            0,
                                                            255,
                                                            0 );
                    }
                    //Or draw the actual blob
                    else
                    {
                        pFrameDisplayTraining->drawRect( (float)m_positiveSampleSet[j].m_width * m_positiveSampleSet[j].m_scaleX,
                                                        (float)m_positiveSampleSet[j].m_height * m_positiveSampleSet[j].m_scaleY,
                                                        (float)m_positiveSampleSet[j].m_col,
                                                        (float)m_positiveSampleSet[j].m_row,
                                                        1,
                                                        0,
                                                        1,
                                                        0,
                                                        255,
                                                        0 ); 
                    }                    
                } 
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to Display training samples" );
    }
    
    /****************************************************************
    SimpleTracker::UpdateClassifier
        Update the classifier.
    Exceptions:
        None
    ****************************************************************/
    void    SimpleTracker::UpdateClassifier( Matrixu*    pFrameImageColor, 
                                            Matrixu*    pFrameImageGray, 
                                            Matrixu*    pFrameDisplayTraining,
                                            Matrixu*    pFrameImageHSV )
    {
        try
        {
            ASSERT_TRUE( pFrameImageColor != NULL || pFrameImageGray!= NULL );

            //generate the training sample sets
            GenerateTrainingSampleSet( pFrameImageColor, pFrameImageGray, pFrameImageHSV );

            //Update the classifier
            m_strongClassifierBasePtr->Update(m_positiveSampleSet,m_negativeSampleSet);

            DisplayTrainingSamples( pFrameDisplayTraining );

            // clean up
            m_positiveSampleSet.Clear(); 
            m_negativeSampleSet.Clear(); 
        }

        EXCEPTION_CATCH_AND_ABORT( "Failed to update the simple tracker classifier" );
    }


    /****************************************************************
    SimpleTracker::TrackObjectOnTheGivenFrame
        Tracks an object on a frame with the trained classifiers.
    Exceptions:
        None
    ****************************************************************/
    double    SimpleTracker::TrackObjectOnTheGivenFrame( Matrixu*    pFrameImageColor, 
                                                    Matrixu*    pFrameImageGray,
                                                    Matrixu*    pFrameDisplayTraining,
                                                    Matrixu*    pFrameImageHSV )
    {
        try
        {
            //tracker should have been initialized before calling this
            ASSERT_TRUE( m_isInitialized ==  true );
            ASSERT_TRUE( pFrameImageColor != NULL || pFrameImageGray!= NULL );

            GenerateTestSampleSet( pFrameImageColor, pFrameImageGray, pFrameImageHSV );
    
            //Classify the samples(detections)
            m_liklihoodProbabilityList = m_strongClassifierBasePtr->Classify( m_testSampleSet, m_simpleTrackerParamsPtr->m_shouldNotUseSigmoid );
 
            //display the actual probability map for debug mode - makes it slower
            if ( m_simpleTrackerParamsPtr->m_debugv )
            {
                //initialize the probability image
                Matrixf probimg;
                if( pFrameImageGray != NULL )
                {
                    probimg.Resize( pFrameImageGray->rows(), pFrameImageGray->cols() );
                }
                else
                {
                    probimg.Resize( pFrameImageColor->rows(), pFrameImageColor->cols() );
                }
                
                for( uint k=0; k < (uint)m_testSampleSet.Size(); k++ )
                {
                    probimg( m_testSampleSet[k].m_row, m_testSampleSet[k].m_col ) = m_liklihoodProbabilityList[k];
                }    

                probimg.convert2img().display(
                    ("Probability map, "+m_simpleTrackerParamsPtr->m_displayFigureNameStr).c_str(),
                    2 );                
                cvWaitKey(1);
            }

            // find the best location - maximum probability
            int bestind = max_idx( m_liklihoodProbabilityList );
            double resp = m_liklihoodProbabilityList[bestind];

            //set the y and x positions
            
            m_currentStateList[1] = (float)m_testSampleSet[bestind].m_row;
            m_currentStateList[0] = (float)m_testSampleSet[bestind].m_col;

            LOG( "SimpleTracker: Update object state to: ["<<m_currentStateList[0] <<" "<<m_currentStateList[1] 
            << " "<<m_currentStateList[2] <<" "<< m_currentStateList[3] << "]."<<endl );

            
            m_testSampleSet.Clear();

            //Update the classifier based on the current state
            UpdateClassifier( pFrameImageColor, pFrameImageGray, pFrameDisplayTraining,    pFrameImageHSV );            

            return resp;
        }
        EXCEPTION_CATCH_AND_ABORT("Unexpected Error While Tracking the object on a given frame" );
    }

    /********************************************************************
    SimpleTracker::GenerateTrainingSampleSet
        Generate training sample sets for a simple tracker
    Exceptions:
        None
    *********************************************************************/
    void SimpleTracker::GenerateTrainingSampleSet( Matrixu*    pFrameImageColor, Matrixu*    pFrameImageGray, Matrixu*    pFrameImageHSV )
    {
        try
        {
            //generate positive sample set
            GeneratePositiveTrainingSampleSet( pFrameImageColor, pFrameImageGray, pFrameImageHSV );

            //generate negative sample set
            GenerateNegativeTrainingSampleSet( pFrameImageColor, pFrameImageGray, pFrameImageHSV );
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to generate training sample sets" );
    }

    /********************************************************************
    GeneratePositiveTrainingSampleSet
        GeneratePositiveTrainingSampleSet in a greedy manner.
    Exceptions:
        None
    *********************************************************************/
    void    SimpleTracker::GeneratePositiveTrainingSampleSet(    Matrixu*    pFrameImageColor, 
                                                                Matrixu*    pFrameImageGray, 
                                                                Matrixu*    pFrameImageHSV  )
    {
        try
        {
            //always clear before using
            m_positiveSampleSet.Clear();

            //Generate positive sample sets
            if ( m_simpleTrackerParamsPtr->m_posRadiusTrain == 1 )
            {    
                m_positiveSampleSet.PushBackSample( pFrameImageGray, 
                                                    (int)m_currentStateList[0], 
                                                    (int)m_currentStateList[1], 
                                                    (int)m_currentStateList[2], 
                                                    (int)m_currentStateList[3],
                                                    1.0f,
                                                    pFrameImageColor, 
                                                    pFrameImageHSV );
            }
            else
            {
                m_positiveSampleSet.SampleImage( pFrameImageGray,
                                                (int)m_currentStateList[0],
                                                (int)m_currentStateList[1],
                                                (int)m_currentStateList[2], 
                                                (int)m_currentStateList[3],
                                                m_simpleTrackerParamsPtr->m_posRadiusTrain,
                                                0,
                                                m_simpleTrackerParamsPtr->m_maximumNumberOfPositiveTrainingSamples,
                                                pFrameImageColor,
                                                pFrameImageHSV );
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to generate positive sample set for simple tracker" );
    }

    /********************************************************************
    GenerateNegativeTrainingSampleSet
        GenerateNegativeTrainingSampleSet in a greedy manner.
    Exceptions:
        None
    *********************************************************************/
    void    SimpleTracker::GenerateNegativeTrainingSampleSet(    Matrixu*    pFrameImageColor, 
                                                                Matrixu*    pFrameImageGray, 
                                                                Matrixu*    pFrameImageHSV )
    {
        try
        {
            //Clear the sample set
            m_negativeSampleSet.Clear();

            //Generate negative sample sets
            // train location classifier (m_negativeSampleSet are randomly selected from image, m_positiveSampleSet is just the current tracker location )
            if ( m_simpleTrackerParamsPtr->m_negSampleStrategy == 0 )
            {
                m_negativeSampleSet.SampleImage( pFrameImageGray, 
                    m_simpleTrackerParamsPtr->m_numberOfNegativeTrainingSamples,
                    (int)m_currentStateList[2],
                    (int)m_currentStateList[3], pFrameImageColor, pFrameImageHSV );
            }
            else
            {
                m_negativeSampleSet.SampleImage( pFrameImageGray, 
                    (int)m_currentStateList[0],
                    (int)m_currentStateList[1],
                    (int)m_currentStateList[2],
                    (int)m_currentStateList[3],
                    (1.5f*m_simpleTrackerParamsPtr->m_searchWindSize),
                    m_simpleTrackerParamsPtr->m_posRadiusTrain+5, 
                    m_simpleTrackerParamsPtr->m_numberOfNegativeTrainingSamples,
                    pFrameImageColor, pFrameImageHSV );
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to generate positive sample set for simple tracker" );
    }

    /********************************************************************
    GetTrainingSampleSets
        Get the training sample sets
    Exceptions:
        None
    *********************************************************************/
    void SimpleTracker::GetTrainingSampleSets( Classifier::SampleSet& positiveSampleSet, Classifier::SampleSet& negativeSampleSet )
    {
        try
        {
            ASSERT_TRUE( m_positiveSampleSet.Size() > 0 );
            ASSERT_TRUE( m_negativeSampleSet.Size() > 0 );

            positiveSampleSet = m_positiveSampleSet;
            negativeSampleSet = m_negativeSampleSet;
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to get the training sample sets" );
    }

        /********************************************************************************************************
    GenerateTestSampleSet
        Generate Test Sample Set.
    Exceptions:
        None
    ********************************************************************************************************/
    void SimpleTracker::GenerateTestSampleSet(    Matrixu* pFrameImageColor,
                                                Matrixu* pFrameImageGray,                                                         
                                                Matrixu* pFrameImageHSV )
    {
        try 
        {
            // Clear the test sample before using
            m_testSampleSet.Clear();

            // run current classifier on search window
            m_testSampleSet.SampleImage( pFrameImageGray,            //image
                                        (int)m_currentStateList[0],    //x - position
                                        (int)m_currentStateList[1],    //y - position
                                        (int)m_currentStateList[2],    //w - width
                                        (int)m_currentStateList[3],    //h -height
                                        (float)m_simpleTrackerParamsPtr->m_searchWindSize,
                                        0, 1000000,
                                        pFrameImageColor,
                                        pFrameImageHSV );// changed uint to int for GCC compatibility
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to generate Test Sample Set" );
    }

    
    /********************************************************************
    SaveStates
        Saves States and clean up all video output
    Exceptions:
        None
    *********************************************************************/
    void    SimpleTracker::SaveStates( )
    {
        try
        {
            // save states
            if( !m_simpleTrackerParamsPtr->m_trajSave.empty() )
            {
                //write into the specified text file
                bool scs = m_states.DLMWrite( (m_simpleTrackerParamsPtr->m_trajSave+".txt").c_str() );
                if( !scs )
                {
                    abortError(__LINE__,__FILE__,"error saving states to trajectory file");
                }
            }            
        }
        EXCEPTION_CATCH_AND_ABORT("Error While Saving the States to trajectory file.");
    }

    /********************************************************************
    InitializeTrackerWithParameters
        Initializes the tracker with parameters, videoList information etc. 
    Exceptions:
        None
    *********************************************************************/
    void    SimpleTracker::InitializeTrackerWithParameters( Matrixu*                pFrameImageColor, 
                                                            Matrixu*                pFrameImageGray, 
                                                            int                        frameInd,
                                                            uint                    videoLength, 
                                                            TrackerParametersPtr    simpleTrackerParametersPtr,
                                                            Classifier::StrongClassifierParametersBasePtr    classifierParametersPtr,
                                                            Matrixu*                pFrameDisplay, 
                                                            Matrixu*                pFrameDisplayTraining,
                                                            Matrixu*                pFrameImageHSV,
                                                            Matrixf*                pGroundTruthMatrix ) 
    {    
        // Resize the state matrix to correct size
        m_states.Resize( videoLength, 4 );
        
        // initialization
        if ( simpleTrackerParametersPtr->m_initializeWithFaceDetection )
        { //***********needs to be corrected 
            // InitializeTracker with face 
            fprintf(stderr,"Searching for face...\n");
            abortError( __LINE__, __FILE__, "initialize with face has not been supported");
        } 
        else
        {
            ASSERT_TRUE( frameInd == simpleTrackerParametersPtr->m_initState[4] ); //initialize at the right frame

            //set the classifier parameters
            classifierParametersPtr->m_featureParametersPtr->m_width    = (uint)simpleTrackerParametersPtr->m_initState[2];
            classifierParametersPtr->m_featureParametersPtr->m_height    = (uint)simpleTrackerParametersPtr->m_initState[3];

            InitializeTracker( pFrameImageColor, pFrameImageGray, simpleTrackerParametersPtr, classifierParametersPtr, pFrameDisplay, pFrameDisplayTraining, pFrameImageHSV );

            for (int i = 0; i < frameInd; i++)
            {
                m_states(i,0) = 0; m_states(i,1) = 0; m_states(i,2) = 0;m_states(i,3) = 0;
            }
            //store the state to state matrix
            m_states(frameInd,0) = m_currentStateList[0];
            m_states(frameInd,1) = m_currentStateList[1];
            m_states(frameInd,2) = m_currentStateList[2];
            m_states(frameInd,3) = m_currentStateList[3];
        }

        if( pGroundTruthMatrix != NULL )
        {
            m_groundTruthMatrix = *pGroundTruthMatrix;
            ASSERT_TRUE( m_groundTruthMatrix.rows() == videoLength );
        }
    }

    /********************************************************************
    CalculateTrackingErrroFromGroundTruth
        Calculate the tracking error based on m_GroundTruthMatrix
    Exceptions:
        None
    *********************************************************************/
    void SimpleTracker::CalculateTrackingErrroFromGroundTruth( )
    {

        try
        {
            if( m_groundTruthMatrix.rows( ) == 0 )
                return;
            
            int videoLength = m_states.rows();

            Matrixf m_statesError( videoLength, 2 );
            //first column is the mean square error (pixels) of center position
            //first column: mean square error of center pixel
            //second column:??????
            for( int frameInd=0; frameInd < videoLength; frameInd++ )
            {
                float centerTrackedX = m_states( frameInd, 0 ) + m_states(frameInd, 2) / 2;
                float centerTrackedY = m_states( frameInd, 1 ) + m_states(frameInd, 3) / 2;
                float centerX = m_groundTruthMatrix( frameInd, 1 ) + m_groundTruthMatrix(frameInd, 3) / 2;
                float centerY = m_groundTruthMatrix( frameInd, 2 ) + m_groundTruthMatrix(frameInd, 4) / 2;
                
                m_statesError( frameInd, 0 ) = sqrt( (centerX - centerTrackedX)*(centerX - centerTrackedX) +
                                                     (centerY - centerTrackedY)*(centerY - centerTrackedY) );

                //calculate overlapping area
                float area = m_groundTruthMatrix(frameInd, 3)* m_groundTruthMatrix(frameInd, 4);
                float area_tracked =  m_states(frameInd, 2) * m_states(frameInd, 3);
                
                float xleft = max( m_states( frameInd, 0 ), m_groundTruthMatrix( frameInd, 1 ) );
                float xright = min( m_states( frameInd, 0 ) + m_states( frameInd, 2 ), 
                                     m_groundTruthMatrix( frameInd, 1 ) + m_groundTruthMatrix( frameInd, 3 ) );

                float width = xright - xleft;  

                float yTop = max( m_states( frameInd, 1), m_groundTruthMatrix( frameInd, 2 ) );
                float yBottom = min( m_states( frameInd, 1 ) + m_states( frameInd, 3 ), 
                                    m_groundTruthMatrix( frameInd, 2 ) + m_groundTruthMatrix(frameInd, 4) );

                float height = yBottom - yTop;
                
                float overLappingArea = 0; 
                if( width > 0 && height > 0 )
                {
                    overLappingArea = width*height;
                }

                float precision = overLappingArea/area_tracked;
                float recall = overLappingArea/area;
                //m_statesError( frameInd, 1 ) = ( 2 * overLappingArea - area_tracked )/area ;
                m_statesError( frameInd, 1 ) = 2*precision*recall/(precision+recall); //f-measure
            }

            // save states
            if( !m_simpleTrackerParamsPtr->m_trajSave.empty() )
            {
                //write into the specified text file
                bool scs = m_statesError.DLMWrite( ( m_simpleTrackerParamsPtr->m_trajSave+"_Error.txt" ).c_str( ) );
                if( !scs )
                {
                    abortError(__LINE__,__FILE__,"error saving trajectory error file");
                }
            }            
        }
        EXCEPTION_CATCH_AND_ABORT("Error while calculating and saving trajectory error file.");
    }

}
