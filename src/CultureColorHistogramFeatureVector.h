#ifndef CULTURE_COLOR_FEATURE_VECTOR_H
#define CULTURE_COLOR_FEATURE_VECTOR_H

#include "FeatureVector.h"

namespace Features
{
    /****************************************************************
    CultureColorHistogramFeatureVector
        This is a wrapper class for CultureColorHistogram.
    ****************************************************************/
    class CultureColorHistogramFeatureVector :virtual public FeatureVector
    {
    public:
        virtual void    Generate( FeatureParametersPtr featureParametersPtr );
        virtual void    Compute( Classifier::SampleSet& sampleSet, bool shouldResizeFeatureMatrix = true );
        virtual void    SaveVisualizedFeatureVector( const char *dirName ){}


        virtual const uint GetNumberOfFeatures( ) const { return m_numberOfCultureColorFeatures; }

    private:
        FeaturePtr                m_featurePtr;
        uint                    m_numberOfCultureColorFeatures;
        uint                    m_startingIndexForFeatureMatrix;
    };
}
#endif