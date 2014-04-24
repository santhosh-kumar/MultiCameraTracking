#ifndef MULTI_COLOR_HISTOGRAM
#define MULTI_COLOR_HISTOGRAM

#include "Feature.h"

namespace Features
{
    /****************************************************************
    MultiDimensionalColorHistogram
        A multi dimensional color histogram obtained by binning the 
        pixels in R,G and B spaces separately.
    ****************************************************************/
    class MultiDimensionalColorHistogram : public Feature
    {
    public:
        MultiDimensionalColorHistogram( )
         : m_shouldWeightFromCenter( true )
        { 
        }

        virtual const FeatureType GetFeatureType( ) const { return MULTI_DIMENSIONAL_COLOR_HISTOGRAM; };
 
        //MultiDimensionalColorHistogram has more than one dim
        virtual float    Compute( const Classifier::Sample& sample ) const 
        {
            abortError( __LINE__, __FILE__, "Error: MultiDimensionalColorHistogram has only one Dimension" ); return 0.0f; 
        }

        virtual void    Compute( const Classifier::Sample& sample, vectorf& featureValueList ) const;
        
        //initialize feature instance (unlike the haar feature) 
        virtual void            Generate( FeatureParametersPtr featureParametersPtr ) ;

    private:
        vectori    m_partPercentageVertical;    // a list of percentage (of total height) for each part, must add up to 100
        uint    m_numberOfBins;
        uint    m_numberOfParts;            // number of parts 
        bool    m_useHSVColorSpace;            // use HSV instead of RGB
        bool    m_shouldWeightFromCenter;
    };
}
#endif