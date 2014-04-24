#include "ParticleFilter.h"
#include "CommonMacros.h"

#define TIME_INTERVAL_FOR_RESAMPLING                    100
#define PERCENTAGE_PARTICLES_FOR_THRESHOLD                0.01//large value -> more frequent(Vice-versa), should be less than 0.7, found empirically.

namespace MultipleCameraTracking
{
    /****************************************************************
    Initialize
        Initialize parameters of the particle filter.
    Exceptions:
        None        
    ****************************************************************/
    void    ParticleFilter::Initialize( int        numberOfParticles,
                                        float    centerX,
                                        float    centerY,
                                        float    scaleX,
                                        float    scaleY,
                                        float    maximumScaleChangeBetweenFrame, 
                                        float   maxScale,
                                        float    minScale)
    {
        try
        {        
            ASSERT_TRUE( m_filterState == UNINTIALIZED );        

            ASSERT_TRUE( numberOfParticles > 0 );
            m_numberOfParticles = numberOfParticles;

            ASSERT_TRUE( maximumScaleChangeBetweenFrame >= 0 );
            m_maximumScaleChange = maximumScaleChangeBetweenFrame; 

            //maxScale specifies maximum allowed scale change for the object
            ASSERT_TRUE( maxScale > 0 && maxScale > scaleX && maxScale > scaleY);
            m_maxScale = maxScale;

            ASSERT_TRUE( minScale > 0 && minScale < scaleX && minScale < scaleY);
            m_minScale = minScale;

            //[centerX, centerY, scaleX, scaleY, weight] --> +1 is weight
            m_particlesResampledMatrix.Resize( m_numberOfParticles, (m_particleStateDimension+1) ); 
            
            if ( m_initializationRelaxation == 0 )
            {
                for ( int row = 0; row < m_numberOfParticles; row++ )
                {
                    m_particlesResampledMatrix( row, 0 ) = centerX;
                    m_particlesResampledMatrix( row, 1 ) = centerY;
                    m_particlesResampledMatrix( row, 2 ) = scaleX;
                    m_particlesResampledMatrix( row, 3 ) = scaleY;
                    m_particlesResampledMatrix( row, m_particleStateDimension ) = 1; //all equal weight, without normalization
                }
            }
            else
            {
                for ( int row = 0; row < m_numberOfParticles; row++ )
                { 
                    //randomly disturb the initialization state
                    m_particlesResampledMatrix( row, 0 ) = centerX + m_initializationRelaxation * ( 2 * randfloat() - 1 );
                    m_particlesResampledMatrix( row, 1 ) = centerY + m_initializationRelaxation * ( 2 * randfloat() - 1 );
                    m_particlesResampledMatrix( row, 2 ) = scaleX;
                    m_particlesResampledMatrix( row, 3 ) = scaleY;
                    m_particlesResampledMatrix( row, m_particleStateDimension ) = 1; //all equal weight, without normalization
                }
            }
            
            //initialize m_particlesStateMatrix by copying m_particlesResampledMatrix
            m_particlesStateMatrix = m_particlesResampledMatrix;    
            
            //set the filter state to INITIALIZED
            m_filterState = INITIALIZED;
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to construct a particle filter" );
    }

    /****************************************************************
    CheckForParticleRefinement
        CheckForParticleRefinement
    Exceptions:
        None        
    ****************************************************************/
    void    ParticleFilter::CheckForParticleRefinement( float        width, 
                                                        float        height )
    {
        try
        {
            int validParticle = 0;
            for ( int row = 0; row < m_numberOfParticles; row++ )
            {
                //randomly disturb the initialization state
                float leftX = m_particlesResampledMatrix( row, 0 ) - m_particlesResampledMatrix( row, 2 ) * width/2;
                float topY =  m_particlesResampledMatrix( row, 1 ) - m_particlesResampledMatrix( row, 3 ) * height/2;

                float footPositionX = m_particlesResampledMatrix( row, 0 );
                float footPositionY =  m_particlesResampledMatrix( row, 1 ) + m_particlesResampledMatrix( row, 3 ) * height/2;

                if ( leftX < 0 || leftX >= m_imageWidth || topY < 0 || topY >= m_imageHeight )
                {
                    continue;
                }

                validParticle++;
            }

            if ( validParticle == 0 )
            {
                ForceParticleFilterRefinement( width, height );
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Check for particle refinement" );
    }


    /****************************************************************
    RearrangeParticlesBasedOnGroundLocation
    Exceptions:
        None
    ****************************************************************/
    void ParticleFilter::RearrangeParticlesBasedOnGroundLocation( vectorf&    meanGroundPlaneFootPositionOnImageList,
                                                                  vectorf&     varianceGroundPlaneFootPositionOnImageList,
                                                                  bool        shouldChangeTheScaleAccordingToOffset, 
                                                                  float        width, 
                                                                  float        height )
    {
        try
        {
            ASSERT_TRUE( meanGroundPlaneFootPositionOnImageList.size() == 2 );

            if ( shouldChangeTheScaleAccordingToOffset )
            {
                ASSERT_TRUE( width > 0.0f );
                ASSERT_TRUE( height > 0.0f );

                for ( int row = 0; row < m_numberOfParticles; row++ )
                {
                    float xRand = randgaus( 0, varianceGroundPlaneFootPositionOnImageList[0] );
                    float yRand = randgaus( 0, varianceGroundPlaneFootPositionOnImageList[1] );

                    float centerX = m_particlesStateMatrix( row, 0 );
                    float centerY = m_particlesStateMatrix( row, 1 );

                    float scaleX = m_particlesStateMatrix( row, 2 );
                    float scaleY = m_particlesStateMatrix( row, 3 );

                    float xPosition = std::max( 0.0f, m_particlesStateMatrix( row, 0 ) - (width/2) * scaleX );
                    float yPosition = std::max( 0.0f, m_particlesStateMatrix( row, 1 ) - (height/2) * scaleY );

                    float footPositionX = centerX;
                    float footPositionY = centerY + height/2 * scaleY;

                    float offsetX = meanGroundPlaneFootPositionOnImageList[0] - footPositionX + xRand;
                    float offsetY = meanGroundPlaneFootPositionOnImageList[1] - footPositionY + yRand;

                    //scaleX' = (2/width) * offsetX + scaleX 
                    m_particlesStateMatrix( row, 2 ) = std::min( std::max( (float)(2.0/width) * offsetX + scaleX, (float)m_minScale ), (float) m_maxScale );

                    //scaleY' = (1/height) * offsetY + scaleY
                    m_particlesStateMatrix( row, 3 ) = std::min( std::max( (float)(1.0/height) * offsetY + scaleY, (float)m_minScale ), (float) m_maxScale );

                    //update the centers according to the new scales
                    m_particlesStateMatrix( row, 0 )    = xPosition + (width/2) * m_particlesStateMatrix( row, 2 );
                    m_particlesStateMatrix( row, 1 )    = yPosition + (height/2) * m_particlesStateMatrix( row, 3 );
                }                
            }
            else
            {
                /*for ( int row = 0; row < m_numberOfParticles; row++ )
                {
                    m_particlesStateMatrix( row, 0 ) = m_particlesStateMatrix( row, 0 ) + particlOffsetList[0] ;
                    m_particlesStateMatrix( row, 1 ) = m_particlesStateMatrix( row, 1 ) + particlOffsetList[1];
                }*/
            }

            CheckForParticleRefinement( width, height );

        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to rearrange particles" );
    }

    /****************************************************************
    ForceParticleFilterRefinement
        Force particle refinement.
    Exceptions:
        None
    ****************************************************************/
    void ParticleFilter::ForceParticleFilterRefinement( float        width, 
                                                        float        height )
    {
        try
        {
            for ( int row = 0; row < m_numberOfParticles; row++ )
            { 
                //randomly disturb the initialization state
                float leftX = m_particlesResampledMatrix( row, 0 ) - m_particlesResampledMatrix( row, 2 ) * width/2;
                float topY =  m_particlesResampledMatrix( row, 1 ) - m_particlesResampledMatrix( row, 3 ) * height/2;

                float footPositionX = m_particlesResampledMatrix( row, 0 );
                float footPositionY =  m_particlesResampledMatrix( row, 1 ) + m_particlesResampledMatrix( row, 3 ) * height/2;

                if ( leftX < 0 )
                {
                    m_particlesResampledMatrix( row, 0 ) = max( footPositionX, 0.0f );
                }

                if ( leftX >= m_imageWidth )
                {
                    m_particlesResampledMatrix( row, 0 ) = min( footPositionX, m_imageWidth-1 );
                }

                if ( topY < 0 )
                {
                    m_particlesResampledMatrix( row, 1 ) = max( footPositionY -  m_particlesResampledMatrix( row, 2 ) * height/2, 0.0f );
                }

                if ( topY >= m_imageHeight )
                {
                    m_particlesResampledMatrix( row, 1 ) = min( footPositionY -  m_particlesResampledMatrix( row, 2 ) * height/2, m_imageHeight-1 );
                }


                m_particlesResampledMatrix( row, m_particleStateDimension ) = 1; //all equal weight, without normalization
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to force particle filter refinement" );
    }

    /****************************************************************
    PredictWithBrownianMotion
        Prediction from m_particlesResampledMatrix based on Brownian motion
        Need to specify the standard deviations according to the current state.
        Note: the motion is with respect to the center of the blob;
    Exceptions:
        None
    ****************************************************************/
    void    ParticleFilter::PredictWithBrownianMotion( float standardDeviationX,
                                                       float standardDeviationY,
                                                       float standardDeviationScaleX,
                                                       float standardDeviationScaleY,
                                                       float width,
                                                       float height )
    {
        try
        {
            //should call this method multiple times
            ASSERT_TRUE( m_filterState != UNINTIALIZED );

            cv::Mat mtx( 1, m_numberOfParticles, CV_32F );
            
            //disturb x
            cv::randn( mtx, cv::Scalar(0), cv::Scalar( standardDeviationX ) );

            for( int p = 0; p < m_numberOfParticles; p++ )
            {
                m_particlesStateMatrix(p,0) = static_cast<float>( 
                        cvRound( m_particlesStateMatrix(p,0) +  m_particlesStateMatrix(p,2) * mtx.at<float>(0,p) ) 
                    );
            }
            
            //disturb y
            cv::randn( mtx, cv::Scalar(0), cv::Scalar(  standardDeviationY ) ); 

            for( int p = 0; p < m_numberOfParticles; p++ )
            {
                m_particlesStateMatrix(p,1) = static_cast<float>( 
                        cvRound( m_particlesStateMatrix( p, 1 ) + m_particlesStateMatrix( p,3 ) * mtx.at<float>(0,p)) 
                    );
            }
            
            //disturb scaleX
            cv::randn( mtx, cv::Scalar(0), cv::Scalar(standardDeviationScaleX) ); 

            #pragma omp parallel for
            for( int p = 0; p < m_numberOfParticles; p++ ) //disturb scale with m_maximumScaleChange cap
            {
                if( m_maximumScaleChange > abs(mtx.at<float>(0,p)) )    
                {
                    m_particlesStateMatrix(p,2) = m_particlesStateMatrix(p,2) + mtx.at<float>(0,p);
                }
                else if( mtx.at<float>(0,p) < 0 )
                {
                    m_particlesStateMatrix(p,2) = m_particlesStateMatrix(p,2) - m_maximumScaleChange;
                }
                else
                {
                    m_particlesStateMatrix(p,2) = m_particlesStateMatrix(p,2) + m_maximumScaleChange;
                }

                if ( m_particlesStateMatrix(p,2) > m_maxScale )
                {
                    m_particlesStateMatrix(p,2) = m_maxScale;
                }

                if (m_particlesStateMatrix(p,2) < m_minScale )
                {
                    m_particlesStateMatrix(p,2) = m_minScale;
                }
            }
        
            //disturb scaleY
            cv::randn( mtx, cv::Scalar(0), cv::Scalar(standardDeviationScaleY) ); 
            #pragma omp parallel for
            for( int p = 0; p < m_numberOfParticles; p++ ) //disturb scale with m_maximumScaleChange cap
            {
                if ( m_maximumScaleChange > abs( mtx.at<float>(0,p) ) )  
                {
                    m_particlesStateMatrix(p,3) = m_particlesStateMatrix(p,3) + mtx.at<float>(0,p);
                }
                else if ( mtx.at<float>(0,p) < 0 )
                {
                    m_particlesStateMatrix(p,3) = m_particlesStateMatrix(p,3) - m_maximumScaleChange;
                }
                else
                {
                    m_particlesStateMatrix(p,3) = m_particlesStateMatrix(p,3) + m_maximumScaleChange;
                }

                if ( m_particlesStateMatrix(p,3) > m_maxScale )
                {
                    m_particlesStateMatrix(p,3) = m_maxScale;
                }

                if (m_particlesStateMatrix(p,3) < m_minScale )
                {
                    m_particlesStateMatrix(p,3) = m_minScale;
                }
            }

            //Set the filter state
            m_filterState = PREDICTED;

            //reset the uniqueness switch
            m_isOrderedUniqueParticlesFound = false;

            //Increment the timeIndex
            m_timeIndex++;

            CheckForParticleRefinement( width, height );
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to predict particles with Brownian motion" );
    }

       /****************************************************************
    PredictWithUniformMotion
        Prediction from m_particlesResampledMatrix based on uniform motion.
        Need to specify the standard deviations according to the current state.
        Note: the motion is with respect to the center of the blob;
    Exceptions:
        None
    ****************************************************************/
    void    ParticleFilter::PredictWithUniformMotion( float standardDeviationX,
                                                      float standardDeviationY,
                                                      float standardDeviationScaleX,
                                                      float standardDeviationScaleY )
    {
        try
        {
            //should call this method multiple times
            ASSERT_TRUE( m_filterState != UNINTIALIZED );

            cv::Mat mtx( 1, m_numberOfParticles, CV_32F );

            //disturb x
            cv::randu( mtx, cv::Scalar(-standardDeviationX), cv::Scalar( standardDeviationX ) );

            for( int p = 0; p < m_numberOfParticles; p++ )
            {
                m_particlesStateMatrix(p,0) = static_cast<float>( 
                    cvRound( m_particlesStateMatrix(p,0) +  m_particlesStateMatrix(p,2) * mtx.at<float>(0,p) ) 
                    );
            }

            //disturb y
            cv::randu( mtx, cv::Scalar(-standardDeviationY), cv::Scalar(  standardDeviationY ) ); 

            for( int p = 0; p < m_numberOfParticles; p++ )
            {
                m_particlesStateMatrix(p,1) = static_cast<float>( 
                    cvRound( m_particlesStateMatrix( p, 1 ) + m_particlesStateMatrix( p,3 ) * mtx.at<float>(0,p)) 
                    );
            }

            //disturb scaleX
            cv::randn( mtx, cv::Scalar(0), cv::Scalar(standardDeviationScaleX) ); 

            #pragma omp parallel for
            for( int p = 0; p < m_numberOfParticles; p++ ) //disturb scale with m_maximumScaleChange cap
            {
                if( m_maximumScaleChange > abs(mtx.at<float>(0,p)) )    
                {
                    m_particlesStateMatrix(p,2) = m_particlesStateMatrix(p,2) + mtx.at<float>(0,p);
                }
                else if( mtx.at<float>(0,p) < 0 )
                {
                    m_particlesStateMatrix(p,2) = m_particlesStateMatrix(p,2) - m_maximumScaleChange;
                }
                else
                {
                    m_particlesStateMatrix(p,2) = m_particlesStateMatrix(p,2) + m_maximumScaleChange;
                }

                if ( m_particlesStateMatrix(p,2) > m_maxScale )
                {
                    m_particlesStateMatrix(p,2) = m_maxScale;
                }

                if (m_particlesStateMatrix(p,2) < m_minScale )
                {
                    m_particlesStateMatrix(p,2) = m_minScale;
                }
            }
        
            //disturb scaleY
            cv::randn( mtx, cv::Scalar(0), cv::Scalar(standardDeviationScaleY) ); 
            #pragma omp parallel for
            for( int p = 0; p < m_numberOfParticles; p++ ) //disturb scale with m_maximumScaleChange cap
            {
                if ( m_maximumScaleChange > abs( mtx.at<float>(0,p) ) )  
                {
                    m_particlesStateMatrix(p,3) = m_particlesStateMatrix(p,3) + mtx.at<float>(0,p);
                }
                else if ( mtx.at<float>(0,p) < 0 )
                {
                    m_particlesStateMatrix(p,3) = m_particlesStateMatrix(p,3) - m_maximumScaleChange;
                }
                else
                {
                    m_particlesStateMatrix(p,3) = m_particlesStateMatrix(p,3) + m_maximumScaleChange;
                }

                if ( m_particlesStateMatrix(p,3) > m_maxScale )
                {
                    m_particlesStateMatrix(p,3) = m_maxScale;
                }

                if (m_particlesStateMatrix(p,3) < m_minScale )
                {
                    m_particlesStateMatrix(p,3) = m_minScale;
                }
            }

            //Set the filter state
            m_filterState = PREDICTED;

            //reset the uniqueness switch
            m_isOrderedUniqueParticlesFound = false;

            //Increment the timeIndex
            m_timeIndex++;
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to predict particles with Brownian motion" );
    }


    /****************************************************************
    GetParticle
        Get predicted particle.
        particle - contains a particle state.
    Exceptions:
        None        
    ****************************************************************/
    void ParticleFilter::GetParticle( int index, vectorf& particle )
    {
        try
        {
            ASSERT_TRUE( index >= 0 && index < m_particlesStateMatrix.rows( ) );
            ASSERT_TRUE( m_filterState >= INITIALIZED );

            particle.resize(m_particleStateDimension+1);
            for ( int i = 0; i < (m_particleStateDimension+1); i++ )
            {
                particle[i] = m_particlesStateMatrix(index,i);
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to get predicted particle" );
    }
    
    /****************************************************************
    GetResampledParticle
        Get re-sampled particle.
    Exceptions:
        None        
    ****************************************************************/
    void ParticleFilter::GetResampledParticle( int particleIndex, vectorf& particle )
    {
        try
        {
            ASSERT_TRUE( particleIndex >= 0 && particleIndex <=  m_particlesResampledMatrix.rows( ) );

            particle.resize((m_particleStateDimension+1));
            for ( int i = 0; i < (m_particleStateDimension+1); i++ )
            {
                particle[i] = m_particlesResampledMatrix(particleIndex,i);
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed resamples particles." );
    }
    
    /****************************************************************
    GetOrderedUniqueParticles
        Get unique particle.
    Exceptions:
        None        
    ****************************************************************/
    void ParticleFilter::GetOrderedUniqueParticles( int index, vectorf& particle )
    {
        try
        {
            FindOrderedUniqueParticles( );

            ASSERT_TRUE( index >= 0 && index < m_particlesOrderedUniqueMatrix.rows() );

            particle.resize((m_particleStateDimension+1));

            for ( int i = 0; i < (m_particleStateDimension+1); i++ )
            {
                particle[i] = m_particlesOrderedUniqueMatrix( index, i );
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to get unique particles." );
    }

    /****************************************************************
    GetHighestWeightParticle
        GetHighestWeightParticle
    Exceptions:
        None        
    ****************************************************************/
    void    ParticleFilter::GetHighestWeightParticle( vectorf& particle )
    {
        try
        {
            vectorf particleWeights(m_numberOfParticles);
            for ( int i =0; i < m_numberOfParticles; i++ )
            {
                particleWeights[i] = m_particlesStateMatrix( i, m_particleStateDimension );
            }

            vectori orderDescending;
            sort_order_des( particleWeights, orderDescending );

            particle.resize((m_particleStateDimension+1));
            for ( int i = 0; i < m_particleStateDimension; i++ )
            {
                particle[i] = m_particlesStateMatrix( orderDescending[0], i );
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to get the highest weight particle" );        
    }

    /****************************************************************
    GetHighestOrderedUniqueParticleCloseToTheGivenState
        GetHighestOrderedUniqueParticleCloseToTheGivenState
    Exceptions:
        None        
    ****************************************************************/
    void    ParticleFilter::GetHighestOrderedUniqueParticleCloseToTheGivenState( vectorf& particle, vectorf& closestState )
    {
        try
        {
            FindOrderedUniqueParticles( );

            particle.resize((m_particleStateDimension+1));
            
            float highestWeight        = m_particlesOrderedUniqueMatrix( 0, m_particleStateDimension );
            double closestDistance    = 1000000;

            for ( int j = 0; j < m_numberOfParticles; j++ )
            {
                if ( highestWeight != m_particlesOrderedUniqueMatrix(j,m_particleStateDimension) )
                {
                    break;
                }

                vectorf candidateParticle;
                candidateParticle.resize((m_particleStateDimension+1));

                for ( int i = 0; i < (m_particleStateDimension+1); i++ )
                {
                    candidateParticle[i] = m_particlesOrderedUniqueMatrix( j, i );
                }

                if ( sqrt( pow( (candidateParticle[0] - closestState[0]) ,2) + pow( (candidateParticle[1] - closestState[1]),2)) < closestDistance )
                {
                    for ( int i = 0; i < (m_particleStateDimension+1); i++ )
                    {
                        particle[i] = candidateParticle[i];
                    }                    
                }
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to get unique particles." );
    }

    /****************************************************************
    GetAverageofAllParticles
        Get the average of all particles.
    Exceptions:
        None        
    ****************************************************************/
    void ParticleFilter::GetAverageofAllParticles(vectorf& particle) const
    {
        try
        {
            particle.resize( (m_particleStateDimension) );

            for ( int i = 0; i < m_particleStateDimension; i++ )
            {
                particle[i] = 0;
            }

            float totalWeight = 0.0f;
            for ( int index = 0; index < m_numberOfParticles; index++ )
            {
                float particleWeight = m_particlesStateMatrix(index, m_particleStateDimension );

                for ( int i = 0; i < m_particleStateDimension; i++ )
                {
                    particle[i] = particle[i] + m_particlesStateMatrix(index,i) * particleWeight;
                    
                }
                totalWeight = totalWeight + particleWeight; 
            }

            ASSERT_TRUE( totalWeight > 0 );

            for ( int i = 0; i < m_particleStateDimension; i++ )
            {
                particle[i] = particle[i] / totalWeight;
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Get Average of all the partices." );
    }

    /****************************************************************
    UpdateParticleWeight
        Set the weights of the predicted particle
    Exceptions:
        None        
    ****************************************************************/
    void ParticleFilter::UpdateParticleWeight( int index, float probability, bool shouldForceReset)
    {
        try
        {    
            ASSERT_TRUE( index >= 0 );
            ASSERT_TRUE( probability >= 0 );

            if ( !shouldForceReset )
            {
                m_particlesStateMatrix( index, m_particleStateDimension ) = probability * m_particlesStateMatrix( index, m_particleStateDimension );
            }
            else
            {
                m_particlesStateMatrix( index, m_particleStateDimension ) = probability;
            }
                    
            //enforce scale ratio
            if( m_particlesStateMatrix( index, 2 ) / m_particlesStateMatrix( index, 3 ) > 1.2 )
            {
                m_particlesStateMatrix( index, 3 ) = 0.9*m_particlesStateMatrix( index, 2 ); //prefer the bigger scale
            }
            else if ( m_particlesStateMatrix( index, 3 ) / m_particlesStateMatrix( index, 2 ) > 1.2 )
            {
                m_particlesStateMatrix( index, 2 ) = 0.9*m_particlesStateMatrix( index, 3 );
            }

            //turn off the switch
            m_isOrderedUniqueParticlesFound = false;
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to set predicted particle weight" );
    }

    /****************************************************************
    UpdateAllParticlesWeight
        Update the weights of non-zero-weight predicated particle 
        followed by re sampling to m_particlesResampledMatrix
    Exceptions:
        None        
    ****************************************************************/
    void ParticleFilter::UpdateAllParticlesWeight(  vectorf&    prob,
                                                    bool        shouldUpdateOnlyNonZeroWeights,
                                                    bool        shouldForceReset,
                                                    bool        shouldResample )
    {
        try
        {
            //find the weight of the particle and sum the weights
            if ( shouldUpdateOnlyNonZeroWeights == false )
            {
                for( int particleIndex=0; particleIndex < m_numberOfParticles; particleIndex++ )
                {
                    UpdateParticleWeight( particleIndex,  prob[particleIndex++], shouldForceReset );
                }
            }
            else
            {
                int p2 = 0;
                for ( int particleIndex=0; particleIndex < m_numberOfParticles; particleIndex++ )
                {
                    if ( m_particlesStateMatrix( particleIndex, m_particleStateDimension )!=0 )
                    {
                        UpdateParticleWeight( particleIndex,   prob[p2++], shouldForceReset );
                    }
                }
            }

            //normalize the weights after update
            NormalizeParticleWeights( );

            //Also, resample if requried
            if ( shouldResample )
            {
                ResampleIfRequired( );
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to update predicted particle weight" );
    }

    /****************************************************************
    FindOrderedUniqueParticles
        Find the unique particle within ParticleResampled and 
        descendingOrderIndex them in terms of weight

        Note: Particles should be re-sampled before calling this function
    Exceptions:
        None        
    ****************************************************************/
    void ParticleFilter::FindOrderedUniqueParticles()
    {
        try
        {
            //do not repeat the calculation
            if ( m_isOrderedUniqueParticlesFound )
            {
                return;
            }

            if ( m_filterState == PREDICTED )
            {
                vectorf particleWeights;
                for ( uint i = 0; i < m_numberOfParticles; i++ )
                {
                    particleWeights.push_back(  m_particlesStateMatrix( i, m_particleStateDimension ) );
                }

                vectori descendingOrderIndex;
            
                sort_order_des( particleWeights, descendingOrderIndex );
                
                //resize
                m_particlesOrderedUniqueMatrix.Resize( m_numberOfParticles, (m_particleStateDimension+1) );

                for ( uint i = 0; i < m_numberOfParticles; i++ )
                {
                    for ( uint j = 0; j <= m_particleStateDimension; j++ )
                    {
                        m_particlesOrderedUniqueMatrix( i, j ) = m_particlesStateMatrix( descendingOrderIndex[i], j );
                    }
                }
            }
            else if ( m_filterState == RESAMPLED )
            {
                int particleIndex;
                vectori order, index;

                index = m_resampleIndexList;
                sort_order( index, order );

                // the first is always unique
                vectori uniqueParticleIndices;
                uniqueParticleIndices.push_back( order[0] );
                
                vectorf uniqueParticleWeight;
                uniqueParticleWeight.push_back( m_particlesStateMatrix(order[0],m_particleStateDimension) );
                
                int uniqueParticleIter=0;
                for( particleIndex = 1; particleIndex < m_numberOfParticles; particleIndex++ )
                {     
                    // check whether this is the same as the previous particle,
                    if ( index[particleIndex]  == index[particleIndex-1] )
                    { 
                        uniqueParticleWeight[uniqueParticleIter] +=  m_particlesStateMatrix( order[particleIndex], m_particleStateDimension ); 
                    }
                    else  //if no, then another unique particle is found
                    {
                        uniqueParticleIndices.push_back( order[particleIndex] );
                        uniqueParticleWeight.push_back( m_particlesStateMatrix( order[particleIndex], m_particleStateDimension ) );
                        uniqueParticleIter++;
                    }
                }
                // the total number unique particle is then uniqueParticleIter+1
                m_particlesOrderedUniqueMatrix.Resize( uniqueParticleIter+1, (m_particleStateDimension+1) );

                // sort unique particle according to their weights (decreasing order)
                sort_order_des( uniqueParticleWeight, order );

                for( particleIndex = 0; particleIndex <= uniqueParticleIter; particleIndex++ )
                {
                    m_particlesOrderedUniqueMatrix(particleIndex,0) = m_particlesStateMatrix(uniqueParticleIndices[particleIndex],0);
                    m_particlesOrderedUniqueMatrix(particleIndex,1) = m_particlesStateMatrix(uniqueParticleIndices[particleIndex],1);
                    m_particlesOrderedUniqueMatrix(particleIndex,2) = m_particlesStateMatrix(uniqueParticleIndices[particleIndex],2);
                    m_particlesOrderedUniqueMatrix(particleIndex,3) = m_particlesStateMatrix(uniqueParticleIndices[particleIndex],3);
                    m_particlesOrderedUniqueMatrix(particleIndex,4) = uniqueParticleWeight[particleIndex];
                }
            }
            else
            {
                abortError( __LINE__, __FILE__, "Not a valid state for finding ordered uniqueness" );
            }

            //turn on the switch
            m_isOrderedUniqueParticlesFound = true;
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to find unique particles" );
    }    

    /****************************************************************
    GetNumberOfOrderedUniqueParticles
        Get number of unique particle.
    Exceptions:
        None        
    ****************************************************************/
    int ParticleFilter::GetNumberOfOrderedUniqueParticles()
    {
        try
        {
            if ( !m_isOrderedUniqueParticlesFound )
            {
                FindOrderedUniqueParticles();
            }

            ASSERT_TRUE( m_particlesOrderedUniqueMatrix.rows() > 0 );

            return m_particlesOrderedUniqueMatrix.rows();
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to get number of unique particles" );
    }

    /****************************************************************
    ResampleIfRequired
        Resample If Required
    Exceptions:
        None        
    ****************************************************************/
    void ParticleFilter::ResampleIfRequired( )
    {
        try
        {
            bool needResampling = false;

            switch(m_resamplingStrategy)
            {
            case TIME_INTERVAL_BASED :
            {
                needResampling = ( ( m_timeIndex % TIME_INTERVAL_FOR_RESAMPLING ) == 0 );
            }
            break;
            case WEIGHT_BASED:
            {
                float nEffectiveInverse = 0;
                float nThreshold = m_numberOfParticles * PERCENTAGE_PARTICLES_FOR_THRESHOLD;

                for ( int particleIndex = 0; particleIndex < m_numberOfParticles; particleIndex++ )
                {
                    nEffectiveInverse = nEffectiveInverse + pow( m_particlesStateMatrix( particleIndex, m_particleStateDimension ), 2 );
                }

                if ( nEffectiveInverse == 0 )
                {
                    return;
                }

                LOG( endl << " N-Effective: " << ( 1 / nEffectiveInverse ) << endl );

                needResampling = ( ( 1 / nEffectiveInverse ) < nThreshold );
            }
            break;
            case WEIGHT_OR_TIME_INTERVAL_BASED:
            {
                float nEffectiveInverse = 0;
                float nThreshold = m_numberOfParticles * PERCENTAGE_PARTICLES_FOR_THRESHOLD;

                for ( int particleIndex = 0; particleIndex < m_numberOfParticles; particleIndex++ )
                {
                    nEffectiveInverse = nEffectiveInverse + pow( m_particlesStateMatrix( particleIndex, m_particleStateDimension ), 2 );
                }

                if ( nEffectiveInverse == 0 )
                {
                    return;
                }

                LOG( endl << " N-Effective: " << ( 1 / nEffectiveInverse ) << endl );

                needResampling = ( ( m_timeIndex % TIME_INTERVAL_FOR_RESAMPLING ) == 0 ) || ( ( 1 / nEffectiveInverse ) < nThreshold );
            }
            break;
            default:
                abortError(__LINE__,__FILE__, "InValid Resampling Strategy");
            break;
            }

            if ( needResampling )
            {
                LOG( "Resampling Particle" << endl );
                ResampleParticles( true/*shouldForceFilterStateChange*/ );
            }
        }
        EXCEPTION_CATCH_AND_LOG( "Failed to Check whether resampling is required or not" );
    }

    /****************************************************************
    ResampleParticles
        Re-sample Particles
        Reference: Arulampalam's Particle Filter Tutorial

    Exceptions:
        None        
    ****************************************************************/
    void ParticleFilter::ResampleParticles( bool shouldForceFilterStateChange )
    {
        try
        {
            NormalizeParticleWeights( );

            //cummulative distribution function
            vectorf cdf;
            cdf.push_back( 0.0f );

            for ( int particleIndex = 1; particleIndex < m_numberOfParticles; particleIndex++ )
            {
                cdf.push_back( cdf[particleIndex-1] + m_particlesStateMatrix( particleIndex, m_particleStateDimension ) );
            }

            ASSERT_TRUE( cdf[m_numberOfParticles-1] <= 1.1f );

            //Set current random threshold for cumulative weight.
            float  seedThreshold = ( 1 / (float)m_numberOfParticles ) * randfloat();

            //resize the re sampled index list 
            m_resampleIndexList.resize( m_numberOfParticles ); 
            
            int resampledIndex = 0;

            //re sample particle
            for( int particleIndex = 0; particleIndex < m_numberOfParticles; particleIndex++ )
            {
                float threshold = seedThreshold + ( 1 / (float)m_numberOfParticles ) * ( particleIndex );

                while ( resampledIndex < m_numberOfParticles-1  && threshold > cdf[resampledIndex] )
                {
                    resampledIndex++;

                    if ( (resampledIndex+1) == m_numberOfParticles )
                    {
                        break;
                    }
                }

                m_resampleIndexList[particleIndex] = resampledIndex;

                for ( int i = 0; i < m_particleStateDimension; i++ )
                {
                    m_particlesResampledMatrix(particleIndex,i) = m_particlesStateMatrix(resampledIndex,i);    
                }

                m_particlesResampledMatrix( particleIndex, m_particleStateDimension ) = 1;
            }

            //record that the particle have been re-sampled and the unique particle have to be found
            if ( shouldForceFilterStateChange )
            {
                m_particlesStateMatrix    = m_particlesResampledMatrix;
                m_filterState            = RESAMPLED;
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Resampling failed" );
    }

    
    /****************************************************************
    NormalizeParticleWeights
        NormalizeParticleWeights
    Exceptions:
        None        
    ****************************************************************/
    void ParticleFilter::NormalizeParticleWeights( )
    {
        try
        {
            float totalWeight = 0.0f;
            for( int particleIndex=0; particleIndex < m_numberOfParticles; particleIndex++ )
            {
                totalWeight += m_particlesStateMatrix( particleIndex, m_particleStateDimension );
            }

            for ( int particleIndex=0; particleIndex < m_numberOfParticles; particleIndex++ )
            {
                if ( totalWeight != 0 )
                {
                    m_particlesStateMatrix( particleIndex, m_particleStateDimension ) = m_particlesStateMatrix( particleIndex, m_particleStateDimension ) / totalWeight;
                }
                else
                {
                    //assign equal weights if total weight is zero
                    m_particlesStateMatrix( particleIndex, m_particleStateDimension ) = 1e-2;
                }
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to normalize particle filter." );
    }

    /****************************************************************
    NormalizeParticleWeights
        GetACopyOfVectorFormat
    Exceptions:
        None        
    ****************************************************************/
    void ParticleFilter::GetACopyOfVectorFormat( vector< vector<float> >& particlesVector )
    {
        particlesVector.clear();

        vector<float> tempParticle( m_particleStateDimension, 0 );
        for( int pInd = 0; pInd <= m_numberOfParticles; pInd++ )
        {
            for( int i = 0 ; i <= m_particleStateDimension; i++ )
            {
                tempParticle[i]=m_particlesStateMatrix( pInd, i );
            }
            particlesVector.push_back(tempParticle);
        }
    }
}