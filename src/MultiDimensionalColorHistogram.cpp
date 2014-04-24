#include "MultiDimensionalColorHistogram.h"

namespace Features
{
/****************************************************************
    MultiDimensionalColorHistogram::Compute
        Generate the multi-dimensional color histogram (multi-part)
    Exception:
        None
    ****************************************************************/
    void MultiDimensionalColorHistogram::Generate( FeatureParametersPtr featureParametersPtr ) 
    {     
        try
        {
            MultiDimensionalColorHistogramParametersPtr temp = boost::dynamic_pointer_cast<MultiDimensionalColorHistogramParameters>(featureParametersPtr);
            
            m_numberOfBins = temp->m_numberOfBins;
            m_numberOfParts    = temp->m_numberOfParts;
            m_partPercentageVertical = temp->m_partPercentageVertical;
            m_useHSVColorSpace = temp->m_useHSVColorSpace;
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to Generate Multi-dimensional color histrogram" );
    }

    /****************************************************************
    MultiDimensionalColorHistogram::Compute
        Compute the multi-dimensional color histogram and store it in the
        given vector.
    Exception:
        None
    ****************************************************************/
    void MultiDimensionalColorHistogram::Compute( const Classifier::Sample& sample, vectorf& featureValueList ) const
    {
        try
        {
            ASSERT_TRUE( !featureValueList.empty( ) );

            // Calculate the size for each part.
            vectori partRowList;
            int accum = 0, partEnd;

            float numberOfRows = cvRound( static_cast<float>(sample.m_height) * sample.m_scaleY );

            partRowList.push_back( 0 );
            for (int i = 0; i < m_numberOfParts; i++)
            {    
                accum += m_partPercentageVertical[i];  
                partEnd = cvRound( numberOfRows * accum/100 ); 
                ASSERT_TRUE ( partEnd > partRowList[i] );
                partRowList.push_back( partEnd );
            }
            
            Matrixu* pImageMatrix = NULL;
            
            if ( m_useHSVColorSpace )
            {
                pImageMatrix = sample.GetHSVImage();
            }
            else
            {
                pImageMatrix = sample.GetColorImage();
            }
            
            ASSERT_TRUE ( pImageMatrix != NULL );
        
            float numberOfColumns =cvRound( static_cast<float>(sample.m_width) * sample.m_scaleX );

            float varianceX = pow( (numberOfColumns / 2), 2);
            float sampleCenterX = sample.m_col + numberOfColumns/2;

            ASSERT_TRUE( pImageMatrix->depth( ) == 3 );
            ASSERT_TRUE( m_numberOfBins >  0 );

            float binWidth = 256 / m_numberOfBins;    

            int partFeatureDimension = m_numberOfBins*m_numberOfBins*m_numberOfBins;
            
            vectorf partFeatureValueList;

            partFeatureValueList.resize( partFeatureDimension );

        
            for( int partIndex = 0; partIndex < m_numberOfParts; partIndex++ ) 
            {
                #pragma omp parallel for

                float partNumberOfRows = partRowList[partIndex+1] - partRowList[partIndex];
                float varianceY = pow( (partNumberOfRows / 2), 2);
                float sampleCenterY = sample.m_row + partRowList[partIndex] + partNumberOfRows/2;
                
                for ( uint rowIndex = sample.m_row+partRowList[partIndex]; rowIndex < (sample.m_row+partRowList[partIndex+1]); rowIndex++ )
                {
                    for ( uint columnIndex = sample.m_col; columnIndex < (sample.m_col+numberOfColumns); columnIndex++ )
                    {
                        uint rPixel = (*pImageMatrix)( rowIndex, columnIndex, 0 /*depth*/ );
                        uint gPixel = (*pImageMatrix)( rowIndex, columnIndex, 1 /*depth*/ );
                        uint bPixel = (*pImageMatrix)( rowIndex, columnIndex, 2 /*depth*/ );

                        uint rBin = floor(  rPixel / binWidth );
                        uint gBin = floor(  gPixel / binWidth );
                        uint bBin = floor(  bPixel / binWidth );

                        uint featureIndex = rBin + m_numberOfBins * gBin + m_numberOfBins * m_numberOfBins * bBin;

                        float featureWeight = 1.0f;

                        if ( m_shouldWeightFromCenter )
                        {
                            float weightedDistanceFromCenter =    pow( ( sampleCenterX - columnIndex ), 2 ) /  (2.0f * varianceX) +
                                pow( ( sampleCenterY - rowIndex ), 2 )  / ( 2.0f * varianceY );

                            featureWeight = exp( -1 * weightedDistanceFromCenter );
                        }

                        partFeatureValueList[ featureIndex ] = partFeatureValueList[ featureIndex ] + featureWeight;
                    }
                }
                ::normalizeVec( partFeatureValueList );

                for ( int i = 0; i < partFeatureDimension; i++ )
                {
                    featureValueList[partIndex*partFeatureDimension + i]= partFeatureValueList[i];
                }             
            }    //to next part
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to Compute Multi-Dimensional Color Histogram" )
    }
}
