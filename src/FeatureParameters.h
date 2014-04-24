#ifndef H_FEATURE_PARAMETERS
#define H_FEATURE_PARAMETERS

#define  DEFAULT_CULTURE_COLOR_DIM    11
#define  COLOR_NUM_PARTS 1 //1, 2, etc.
#include "Public.h"
#include "CommonMacros.h"
#include <boost/shared_ptr.hpp>

namespace Features
{
    enum FeatureType    {    HAAR_LIKE = 0,
                            CULTURE_COLOR_HISTOGRAM = 1,
                            MULTI_DIMENSIONAL_COLOR_HISTOGRAM = 2,
                            HAAR_COLOR_HISTOGRAM = 3,
                        };

    //Forward Declaration
    class FeatureParameters;
    class HaarFeatureParameters;
    class MultiDimensionalColorHistogramParameters;
    class CultureColorHistogramParameters;
    class HaarAndColorHistogramFeatureParameters;

    //typedef
    typedef boost::shared_ptr<FeatureParameters>                        FeatureParametersPtr;
    typedef boost::shared_ptr<HaarFeatureParameters>                    HaarFeatureParametersPtr;
    typedef boost::shared_ptr<MultiDimensionalColorHistogramParameters> MultiDimensionalColorHistogramParametersPtr;
    typedef boost::shared_ptr<CultureColorHistogramParameters>            CultureColorHistogramParametersPtr;
    typedef boost::shared_ptr<HaarAndColorHistogramFeatureParameters>    HaarAndColorHistogramFeatureParametersPtr;

    /*****************************************************************
    FeatureParameters
        Abstract class for feature parameters.
    ****************************************************************/
    class FeatureParameters
    {
    public:
        virtual FeatureType    GetFeatureType( )        const            = 0;
        virtual uint        GetFeatureDimension( )    const            = 0;

        virtual uint GetColorFeatureDimension( )    const            = 0;
        virtual uint GetHaarFeatureDimension( )        const            = 0;
        
        uint                m_width;        // original width of the rectangular blob on which the feature is calculated
        uint                m_height;        // original height of the rectangular blob on which the feature is calculated
    };

    /****************************************************************
    HaarFeatureParameters
        General Haar Feature Parameters
    ****************************************************************/
    class HaarFeatureParameters : virtual public FeatureParameters
    {
    public:
        HaarFeatureParameters( uint featureDimensionHaar );

        virtual FeatureType    GetFeatureType() const { return HAAR_LIKE; }

        virtual uint GetFeatureDimension( )     const { return m_featureDimensionHaar; } 
        
        virtual uint GetColorFeatureDimension( )    const    { return 0;}
        virtual uint GetHaarFeatureDimension( )        const    { return m_featureDimensionHaar; }

        friend class HaarFeature;

    protected:        
        uint                m_maximumNumberOfRectangles;
        uint                m_minimumNumberOfRectangles;
        int                    m_useChannels[1024];    // >=0: used; <0:not used
        int                    m_numberOfChannels;        // number of channels used for the computation of feature
        uint                m_featureDimensionHaar;
    };

    /****************************************************************
    MultiDimensionalColorHistogramParameters
        Multi Dimensional Color Histogram Parameters
    ****************************************************************/
    class MultiDimensionalColorHistogramParameters : virtual public FeatureParameters
    {
    public:
        MultiDimensionalColorHistogramParameters( bool useHSVColor = false, uint numberOfBins    = 8 )
            : m_numberOfBins (numberOfBins),
            m_numberOfParts( COLOR_NUM_PARTS ),
            m_partPercentageVertical( COLOR_NUM_PARTS, 100/COLOR_NUM_PARTS ),
            m_featureDimensionColor( numberOfBins * numberOfBins * numberOfBins * m_numberOfParts ),
            m_useHSVColorSpace( useHSVColor )
        {         
            
        }
        
        virtual FeatureType    GetFeatureType( )    const { return MULTI_DIMENSIONAL_COLOR_HISTOGRAM; };
        virtual uint GetFeatureDimension( )        const { return m_featureDimensionColor; }

        virtual uint GetColorFeatureDimension( )    const    { return m_featureDimensionColor;}
        virtual uint GetHaarFeatureDimension( )        const    { return 0; }

        friend class MultiDimensionalColorHistogram;

    protected:
        const uint        m_numberOfBins;        //number of bins for each color 
        const vectori    m_partPercentageVertical;    // a list of percentage (of total height) for each part, must add up to 100
        const uint        m_numberOfParts;            // number of parts 
        const uint        m_featureDimensionColor;
        const bool        m_useHSVColorSpace;            // use HSV instead of RGB
    };

    /****************************************************************
    CultureColorHistogramParameters
        Multi-part Culture Color Histogram Parameters
    ****************************************************************/
    class CultureColorHistogramParameters : virtual public FeatureParameters
    {
    public:
        CultureColorHistogramParameters( );
        CultureColorHistogramParameters( vectori partPercentage ); 
        virtual FeatureType    GetFeatureType() const { return CULTURE_COLOR_HISTOGRAM; }
        virtual uint GetFeatureDimension( ) const { return m_featureDimensionForCultureColor; }

        virtual uint GetColorFeatureDimension( )    const    { return m_featureDimensionForCultureColor;}
        virtual uint GetHaarFeatureDimension( )        const    { return 0; }
                
    friend class CultureColorHistogram; 

    private:
        const vectori    m_partPercentageVertical;                // a list of percentage (of total height) for each part, must add up to 100
        const uint        m_numberOfParts;                        // number of parts 
        const uint        m_featureDimensionForCultureColor;         // total number of feature dimension
    };

    /****************************************************************
    HaarAndColorHistogramFeatureParameters
        HaarAndColorHistogramFeatureParameters
    ****************************************************************/
    class HaarAndColorHistogramFeatureParameters :    public HaarFeatureParameters, 
                                                    public MultiDimensionalColorHistogramParameters
    {
    public:
        HaarAndColorHistogramFeatureParameters( uint featureDimensionHaar, bool useHSVColor = false, uint numberOfBins    = 8 )
            : HaarFeatureParameters( featureDimensionHaar ),
            MultiDimensionalColorHistogramParameters( useHSVColor, numberOfBins )            
        {}

        virtual FeatureType    GetFeatureType( )    const { return HAAR_COLOR_HISTOGRAM; };
        virtual uint GetFeatureDimension( )        const { return (m_featureDimensionHaar+m_featureDimensionColor); }
        virtual uint GetColorFeatureDimension( )    const    { return m_featureDimensionColor; }
        virtual uint GetHaarFeatureDimension( )        const    { return m_featureDimensionHaar;  }
    };
}
#endif