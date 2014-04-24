#include "FeatureParameters.h"
#include "DefaultParameters.h"

namespace Features
{
    /****************************************************************
    HaarFeatureParameters: Haar-Like Feature Parameters
        Default Constructor
    ****************************************************************/
    HaarFeatureParameters::HaarFeatureParameters( uint featureDimensionHaar )
        : m_minimumNumberOfRectangles ( DEFAULT_HAAR_MINIMUM_NUMBER_OF_RECTANGLES ),
          m_maximumNumberOfRectangles ( DEFAULT_HAAR_MAXIMUM_NUMBER_OF_RECTANGLES ),
          m_numberOfChannels( DEFAULT_HAAR_NUMBER_OF_CHANNELS ),
          m_featureDimensionHaar( featureDimensionHaar )
    {
        for ( int channelIndex = 0; channelIndex < 1024; channelIndex++ )
        {
            m_useChannels[channelIndex] = -1;
        }
        m_useChannels[0]    = 0;
    }

    /****************************************************************
    CultureColorHistogramParameters
        Default Constructor (single part)
    ****************************************************************/
    CultureColorHistogramParameters::CultureColorHistogramParameters(  )
        : m_numberOfParts(1),
        m_featureDimensionForCultureColor( DEFAULT_CULTURE_COLOR_DIM ),
        m_partPercentageVertical( 1, 100 )
    {
    
    }

    /****************************************************************
    CultureColorHistogramParameters
        Constructor with exact part information ( a list of percentages) 
    ****************************************************************/
    CultureColorHistogramParameters::CultureColorHistogramParameters( vectori partPercentageVertical )
        : m_numberOfParts( partPercentageVertical.size() ),
        m_featureDimensionForCultureColor( partPercentageVertical.size()*DEFAULT_CULTURE_COLOR_DIM ),
        m_partPercentageVertical( partPercentageVertical )    
    {
        int count = 0;
        for (int i=0; i < partPercentageVertical.size(); i++)
        {
            ASSERT_TRUE(partPercentageVertical[i] > 0);
            count += partPercentageVertical[i];
        }

        ASSERT_TRUE(count == 100);        
    }
}