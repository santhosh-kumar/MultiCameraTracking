#include "MultiDimensionalColorHistogramFeatureVector.h"
#include "MultiDimensionalColorHistogram.h"

namespace Features
{
    /****************************************************************
    MultiDimensionalColorHistogramFeatureVector::Generate
        Generates FeatureVector based on the specified feature type.    
        Feature stores the multi dimensional color histogram.
    Note: parameter "numberOfColorFeatures" is not used as the size of feature vector
        is pre-determined.
    Exception:
        None
    ****************************************************************/
    void    MultiDimensionalColorHistogramFeatureVector::Generate( FeatureParametersPtr featureParametersPtr )
    {
        try
        {
            ASSERT_TRUE( featureParametersPtr != NULL );

            m_featureParametersPtr = featureParametersPtr;

            m_featurePtr = boost::shared_ptr<MultiDimensionalColorHistogram>( new MultiDimensionalColorHistogram() );

            ASSERT_TRUE( m_featurePtr != NULL );

            m_numberOfColorFeatures = m_featureParametersPtr->GetColorFeatureDimension( );

            m_featurePtr->Generate( m_featureParametersPtr );

            //Update the feature generated flag
            m_isFeatureGenerated = true;

            ///feature index starts from this location
            m_startingIndexForFeatureMatrix = m_featureParametersPtr->GetHaarFeatureDimension( );

            ASSERT_TRUE( m_startingIndexForFeatureMatrix >= 0 );
        }
        EXCEPTION_CATCH_AND_ABORT( "Error While Generating MultiDimensionalColorHistogram Feature Vector" );
    }


    /****************************************************************
    MultiDimensionalColorHistogramFeatureVector::Compute
        Iterate over each sample in the given Classifier::SampleSet
        and compute the featurePtr. Store the computed featurePtr value 
        in the featurePtr vector.
        Classifier calls this method with a set of samples for each type.
    Exception:
        None
    ****************************************************************/
    void    MultiDimensionalColorHistogramFeatureVector::Compute( Classifier::SampleSet& sampleSet, bool shouldResizeFeatureMatrix )
    {
        try
        {
            size_t numberOfSamples    = sampleSet.Size( );

            ASSERT_TRUE( m_isFeatureGenerated == true );
            ASSERT_TRUE( m_numberOfColorFeatures > 0 );

            //return if no sample is available
            if ( numberOfSamples == 0 )
            {
                return;
            }

            //resize the feature matrix size to the number of features
            if ( shouldResizeFeatureMatrix )
            {
                sampleSet.ResizeFeatures( m_numberOfColorFeatures );
            }

            #pragma omp parallel for
            for ( int sampleIndex = 0; sampleIndex < numberOfSamples; sampleIndex++ )
            {
                vectorf colorHistogramVector( m_numberOfColorFeatures, 0.0f );
                m_featurePtr->Compute( sampleSet[sampleIndex], colorHistogramVector );

                ASSERT_TRUE( m_featurePtr != NULL );
                ASSERT_TRUE( colorHistogramVector.size( ) == m_numberOfColorFeatures );
                
                for ( uint featureIndex = m_startingIndexForFeatureMatrix; featureIndex < (m_startingIndexForFeatureMatrix+m_numberOfColorFeatures); featureIndex++ )
                {
                    //store the feature value in the feature matrix
                    sampleSet.GetFeatureValue( sampleIndex, featureIndex ) = colorHistogramVector[featureIndex-m_startingIndexForFeatureMatrix];
                }
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to Multi-Dimensional Color Histogram Feature Vector" );
    }
}