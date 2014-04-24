#include "CultureColorHistogramFeatureVector.h"
#include "CultureColorHistogram.h"

namespace Features
{
    /****************************************************************
    CultureColorHistogramFeatureVector::Generate
        Generates FeatureVector based on the specified feature type.    
        Feature stores the culture color histogram.
    Note:parameter "numberOfFeatures" is not used as the size of feature vector
        is pre-determined.
    Exception:
        None
    ****************************************************************/
    void    CultureColorHistogramFeatureVector::Generate( FeatureParametersPtr featureParametersPtr )
    {
        try
        {
            ASSERT_TRUE( featureParametersPtr != NULL );

            m_featureParametersPtr = featureParametersPtr;

            m_featurePtr = boost::shared_ptr<CultureColorHistogram>( new CultureColorHistogram() );

            ASSERT_TRUE( m_featurePtr != NULL );

            m_numberOfCultureColorFeatures = m_featureParametersPtr->GetColorFeatureDimension( );

            m_featurePtr->Generate( m_featureParametersPtr );

            //Update the feature generated flag
            m_isFeatureGenerated = true;

            m_startingIndexForFeatureMatrix = m_featureParametersPtr->GetHaarFeatureDimension( );
        }
        EXCEPTION_CATCH_AND_ABORT( "Error While Generating CultureColorHistogram Feature Vector" );
    }

        /****************************************************************
    CultureColorHistogramFeatureVector::Compute
        Iterate over each sample in the given Classifier::SampleSet
        and compute the featurePtr. Store the computed featurePtr value 
        in the featurePtr vector.
        Classifier calls this method with a set of samples for each type.
    Exception:
        None
    ****************************************************************/
    void    CultureColorHistogramFeatureVector::Compute( Classifier::SampleSet& sampleSet, bool shouldResizeFeatureMatrix )
    {
        size_t numberOfSamples    = sampleSet.Size( );

        ASSERT_TRUE( m_isFeatureGenerated == true );
        ASSERT_TRUE( m_numberOfCultureColorFeatures > 0 );

        //return if no sample is available
        if ( numberOfSamples == 0 )
        {
            return;
        }

        if ( shouldResizeFeatureMatrix )
        {
            //resize the feature matrix size to the number of features
            sampleSet.ResizeFeatures( m_numberOfCultureColorFeatures );
        }

        #pragma omp parallel for
        for ( int sampleIndex = 0; sampleIndex < numberOfSamples; sampleIndex++ )
        {
            vectorf histVector( m_numberOfCultureColorFeatures, 0.0f );
            m_featurePtr->Compute( sampleSet[sampleIndex], histVector );

            ASSERT_TRUE( histVector.size( ) == m_numberOfCultureColorFeatures );
            
            for ( uint featureIndex = m_startingIndexForFeatureMatrix; featureIndex < (m_startingIndexForFeatureMatrix+m_numberOfCultureColorFeatures); featureIndex++ )
            {
                //store the feature value in the feature matrix
                sampleSet.GetFeatureValue( sampleIndex, featureIndex ) = histVector[featureIndex];
            }
        }
    }
}