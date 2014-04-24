#include "HaarAndColorHistogramFeatureVector.h"
#include "HaarFeature.h"
#include "MultiDimensionalColorHistogram.h"

namespace Features
{
    /****************************************************************
    HaarAndColorHistogramFeatureVector::Generate
    Exception:
        None
    ****************************************************************/
    void    HaarAndColorHistogramFeatureVector::Generate( FeatureParametersPtr featureParametersPtr )
    {
        try
        {
            //Call Generate Features of the HaarFeatureVector
            HaarFeatureVector::Generate( featureParametersPtr );
            MultiDimensionalColorHistogramFeatureVector::Generate( featureParametersPtr );

            m_numberOfHaarColorFeatures = m_numberOfHaarFeatures + m_numberOfColorFeatures;

            ASSERT_TRUE( m_numberOfHaarColorFeatures > 0 );
        }
        EXCEPTION_CATCH_AND_ABORT( "Error While Generating MultiDimensionalColorHistogram Feature Vector" );
    }

    /****************************************************************
    HaarAndColorHistogramFeatureVector::Compute

    Exception:
        None
    ****************************************************************/
    void    HaarAndColorHistogramFeatureVector::Compute( Classifier::SampleSet& sampleSet, bool /*shouldResizeFeatureMatrix*/ )
    {
        try
        {
            //resize the feature matrix for efficiency
            sampleSet.ResizeFeatures( m_numberOfHaarColorFeatures );

            HaarFeatureVector::Compute( sampleSet, false/*shouldResizeFeatureMatrix*/ );
            MultiDimensionalColorHistogramFeatureVector::Compute( sampleSet, false/*shouldResizeFeatureMatrix*/ );
        }
        EXCEPTION_CATCH_AND_ABORT( "Error While Generating MultiDimensionalColorHistogram Feature Vector" );
    }
}