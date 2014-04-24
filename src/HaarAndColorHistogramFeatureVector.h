#ifndef HAAR_COLOR_FEATURE_VECTOR_H
#define HAAR_COLOR_FEATURE_VECTOR_H

#include "FeatureVector.h"
#include "HaarFeatureVector.h"
#include "MultiDimensionalColorHistogramFeatureVector.h"

namespace Features
{
    /****************************************************************
    HaarAndColorHistogramFeatureVector
        This is a wrapper class for HaarAndColorHistogramFeatureVector.
    ****************************************************************/
    class HaarAndColorHistogramFeatureVector :    public HaarFeatureVector,
                                                public MultiDimensionalColorHistogramFeatureVector
    {
    public:
        
        virtual void    Generate( FeatureParametersPtr featureParametersPtr );
        virtual void    Compute( Classifier::SampleSet& sampleSet, bool shouldResizeFeatureMatrix = true );
        virtual void    SaveVisualizedFeatureVector( const char *dirName ){}

        virtual const uint GetNumberOfFeatures( ) const
        { 
            ASSERT_TRUE( m_numberOfHaarColorFeatures !=  0 ); return m_numberOfHaarColorFeatures;
        }

    private:
        uint m_numberOfHaarColorFeatures;
    };
}
#endif