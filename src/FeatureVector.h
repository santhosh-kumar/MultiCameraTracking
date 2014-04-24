#ifndef FEATURE_VECTOR_H
#define FEATURE_VECTOR_H

#include "FeatureParameters.h"
#include "Feature.h"
#include "SampleSet.h"
#include "CommonMacros.h"

namespace Features
{
    //Forward Declarations
    class FeatureVector;
    class HaarFeatureVector;
    class MultiDimensionalColorHistogramFeatureVector;
    class CultureColorHistogramFeatureVector;
    class HaarAndColorHistogramFeatureVector;

    typedef boost::shared_ptr<FeatureVector>        FeatureVectorPtr;
    typedef boost::shared_ptr<HaarFeatureVector>    HaarFeatureVectorPtr;
    typedef boost::shared_ptr<MultiDimensionalColorHistogramFeatureVector>    MultiDimensionalColorHistogramFeatureVectorPtr;
    typedef    boost::shared_ptr<CultureColorHistogramFeatureVector>            CultureColorHistogramFeatureVectorPtr;
    typedef    boost::shared_ptr<HaarAndColorHistogramFeatureVector>            HaarAndColorHistogramFeatureVectorPtr;

    /****************************************************************
     FeatureVector
        This is a wrapper class for features.
    ****************************************************************/
    class FeatureVector
    {
    public:
        FeatureVector( )
            : m_featureParametersPtr( ),
            m_isFeatureGenerated( false )
        {
        }

        //Pure virtual functions
        virtual void        Generate( FeatureParametersPtr featureParametersPtr ) = 0;
        virtual void        Compute( Classifier::SampleSet& sampleSet, bool shouldResizeFeatureMatrix = true ) = 0;
        virtual void        SaveVisualizedFeatureVector( const char *dirName ) = 0;

        virtual const uint GetNumberOfFeatures( ) const = 0;
        
    protected:
        FeatureParametersPtr    m_featureParametersPtr;
        bool                    m_isFeatureGenerated;
    };
}
#endif