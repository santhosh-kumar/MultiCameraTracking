#ifndef HAAR_FEATURE
#define HAAR_FEATURE

#include "Feature.h"

namespace Features
{
    /****************************************************************
    HaarFeature
        Haar type features. Obtained using wavelet based mother basis.
    ****************************************************************/
    class HaarFeature : public Feature
    {
    public:
        HaarFeature( );

        virtual const FeatureType    GetFeatureType() const { return HAAR_LIKE; };
        
        //randomly generate a feature instance (this) based on feature parameters
        virtual void            Generate( FeatureParametersPtr featureParametersPtr );
        
        //copy from another haar feature instance
        HaarFeature&            operator= ( const HaarFeature& aHaarFeature );

        virtual Matrixu            ToVisualize( int featureIndex = -1 );

        float                    GetExpectedValue() const;

        // Haar-like feature is of one dim
        virtual float            Compute( const Classifier::Sample& sample ) const;    
        virtual void            Compute( const Classifier::Sample& sample, vectorf& featureValueList ) const
                                { abortError( __LINE__, __FILE__, "Error: HaarFeature has only one Dimension" );}
                
        //member variables
        static StopWatch        m_sw;

    private:
        uint                    m_width;
        uint                    m_height;
        uint                    m_channel;
        vectorf                    m_weights;
        vector<IppiRect>        m_rects;
        vectorf                    m_rsums;
        double                    m_maxSum;        
    };
}
#endif