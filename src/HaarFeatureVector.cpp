#include "HaarFeatureVector.h"
#include "HaarFeature.h"

namespace Features
{
    /****************************************************************
    HaarFeatureVector::Generate
        Generates FeatureVector based on the specified feature type.    
        FeatureList stores a list of haar features.
        Call to Compute actually calculates the feature values.
    Exception:
        None
    ****************************************************************/
    void    HaarFeatureVector::Generate( FeatureParametersPtr featureParametersPtr )
    {
        try
        {
            ASSERT_TRUE( featureParametersPtr != NULL );

            m_featureParametersPtr        = featureParametersPtr;
            m_numberOfHaarFeatures        = featureParametersPtr->GetHaarFeatureDimension();

            m_featureList.resize( m_numberOfHaarFeatures );

            for ( uint featureIndex = 0; featureIndex < m_numberOfHaarFeatures; featureIndex++ )
            {
                m_featureList[featureIndex] = boost::shared_ptr<HaarFeature>( new HaarFeature() );
                ASSERT_TRUE( m_featureList[featureIndex] != 0 );
                m_featureList[featureIndex]->Generate( m_featureParametersPtr );
            }

            //Update the feature generated flag
            m_isFeatureGenerated = true;
        }
        EXCEPTION_CATCH_AND_ABORT( "Error While Generating Haar Feature" );
    }


    /****************************************************************
    HaarFeatureVector::Compute
        Iterate over each sample in the given Classifier::SampleSet
        and compute the featurePtr. Store the computed featurePtr value 
        in the featurePtr vector.
        Classifier calls this method with a set of samples for each type.
    Exception:
        None
    ****************************************************************/
    void    HaarFeatureVector::Compute( Classifier::SampleSet& sampleSet, bool shouldResizeFeatureMatrix )
    {
        size_t numberOfSamples        = sampleSet.Size( );

        ASSERT_TRUE( m_isFeatureGenerated == true );
        ASSERT_TRUE( m_numberOfHaarFeatures > 0 );

        //return if no sample is available
        if ( numberOfSamples == 0 )
        {
            return;
        }

        //resize the feature matrix size to the number of features
        if ( shouldResizeFeatureMatrix )
        {
            sampleSet.ResizeFeatures( m_numberOfHaarFeatures );
        }

        #pragma omp parallel for
        for ( uint featureIndex = 0; featureIndex < m_numberOfHaarFeatures; featureIndex++ )
        {
            //#pragma omp parallel for
            for ( int sampleIndex = 0; sampleIndex < numberOfSamples; sampleIndex++ )
            {
                ASSERT_TRUE( m_featureList[featureIndex] != NULL );

                //store the feature value in the feature matrix
                sampleSet.GetFeatureValue( sampleIndex, featureIndex ) = m_featureList[featureIndex]->Compute( sampleSet[sampleIndex] );
            }
        }
    }

    /****************************************************************
    HaarFeatureVector::SaveVisualizedFeatureVector
        Save the visualized haar features to a given directory
    Exception:
        None
    ****************************************************************/
    void    HaarFeatureVector::SaveVisualizedFeatureVector(const char* dirName)
    {    
        char fname[1024];
        Matrixu img;
        for( uint featureIndex = 0; featureIndex < m_featureList.size(); featureIndex++ )
        {
            sprintf_s( fname, "%s/HaarFeature%05d.png", dirName, featureIndex );
            img = m_featureList[featureIndex]->ToVisualize( );
            img.SaveImage(fname);
        }
    }
}