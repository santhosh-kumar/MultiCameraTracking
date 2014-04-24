#ifndef COLOR_FEATURE_VECTOR_H
#define COLOR_FEATURE_VECTOR_H

#include "FeatureVector.h"
namespace Features
{
    /****************************************************************
    MultiDimensionalColorHistogramFeatureVector
        This is a wrapper class for MultiDimensionalColorHistogram.
    ****************************************************************/
    class MultiDimensionalColorHistogramFeatureVector : virtual public FeatureVector
    {
    public:
        //Generate features
        virtual void        Generate( FeatureParametersPtr featureParametersPtr );

        //Computes features for the given sample set
        virtual void        Compute( Classifier::SampleSet& sampleSet, bool shouldResizeFeatureMatrix = true  );

        //Save the visualized feature vector - Unused
        virtual void        SaveVisualizedFeatureVector( const char *dirName ){}

        //Get Number of Features
        virtual const uint    GetNumberOfFeatures( ) const { return m_numberOfColorFeatures; }

    protected:

        FeaturePtr                m_featurePtr;
        uint                    m_numberOfColorFeatures;
        uint                    m_startingIndexForFeatureMatrix; //Used while Color Feature is concatenated to the Other features.
    };
}
#endif