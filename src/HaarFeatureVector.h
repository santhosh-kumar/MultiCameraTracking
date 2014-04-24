#ifndef HAAR_FEATURE_VECTOR_H
#define HAAR_FEATURE_VECTOR_H

#include "FeatureVector.h"

namespace Features
{
    /****************************************************************
     HaarFeatureVector
        This is a wrapper class for HaarFeature.
    ****************************************************************/
    class HaarFeatureVector : virtual public FeatureVector
    {
    public:
        HaarFeatureVector( ){}

        virtual void    Generate( FeatureParametersPtr featureParametersPtr );
        virtual void    Compute( Classifier::SampleSet& sampleSet, bool shouldResizeFeatureMatrix = true );

        virtual void    SaveVisualizedFeatureVector( const char * dirname );

        virtual const uint GetNumberOfFeatures( ) const { return m_numberOfHaarFeatures; }

    protected:
        FeatureList                m_featureList;
        uint                    m_numberOfHaarFeatures;
    };
}
#endif
