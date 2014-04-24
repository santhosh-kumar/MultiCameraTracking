#ifndef CULTURE_COLOR
#define CULTURE_COLOR

#include "Feature.h"

namespace Features
{
    /****************************************************************
    CultureColorHistogram
        A multi dimensional culture color histogram obtained by dividing
        the blob into horizontal slices (parts) and concatenates the culture 
        color histogram of each part
    ****************************************************************/
    class CultureColorHistogram : public Feature
    {
    public:
        CultureColorHistogram( ) { };

        virtual const FeatureType    GetFeatureType() const { return CULTURE_COLOR_HISTOGRAM; };

        virtual float            Compute( const Classifier::Sample& sample ) const 
                                    { abortError( __LINE__, __FILE__, "CultureCoor is of more than one dimension" );; return 0.0f; }
        
        virtual void            Compute( const Classifier::Sample& sample, vectorf& featureValueList ) const;
        
        virtual void            Generate( FeatureParametersPtr featureParametersPtr ); 

    private:
        vectori        m_partPercentageVertical;    // a list of percentage (of total height) for each part, must add up to 100
        uint        m_numberOfParts;            // number of parts     
        void        ComputeFeature( const Classifier::Sample& sample, cv::Mat& sampleImgHSV, vectorf& featureValueList ) const;
    };
}
#endif