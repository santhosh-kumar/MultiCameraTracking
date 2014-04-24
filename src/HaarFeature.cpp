#include "HaarFeature.h"

namespace Features
{
    /****************************************************************
    HaarFeature::HaarFeature
        Constructor
    Exception:
        None
    ****************************************************************/
    HaarFeature::HaarFeature( )
        : m_channel( 0 )
    {
        m_width        = 0;
        m_height    = 0;
    }

    /****************************************************************
    HaarFeature::Generate
        Generate a Haar Feature based on the haar feature parameters
        (the rectangles and their weights are randomly generated)
    Exception:
        None
    ****************************************************************/
    void    HaarFeature::Generate( FeatureParametersPtr featureParametersPtr )
    {
        ASSERT_TRUE( featureParametersPtr != NULL );

        //type cast to haar featurePtr
        HaarFeatureParametersPtr haarFeatureParametersPtr = boost::dynamic_pointer_cast<HaarFeatureParameters>( featureParametersPtr );

        m_width         = haarFeatureParametersPtr->m_width;
        m_height        = haarFeatureParametersPtr->m_height;
        m_maxSum        = 0.0f;

        //choose the random number of rectangles for Haar Feature Generation
        int numberOfRectangles    = randint(haarFeatureParametersPtr->m_minimumNumberOfRectangles, haarFeatureParametersPtr->m_maximumNumberOfRectangles );
        m_rects.resize( numberOfRectangles );
        m_weights.resize( numberOfRectangles );
        m_rsums.resize( numberOfRectangles );         

        for ( int rectangleIndex=0; rectangleIndex < numberOfRectangles; rectangleIndex++ )
        {
            m_weights[rectangleIndex]        = randfloat( )*2 - 1;
            m_rects[rectangleIndex].x        = randint( 0,(uint)(haarFeatureParametersPtr->m_width-3) );
            m_rects[rectangleIndex].y        = randint( 0,(uint)(haarFeatureParametersPtr->m_height-3) );
            m_rects[rectangleIndex].width    = randint(1,(haarFeatureParametersPtr->m_width-m_rects[rectangleIndex].x-2));
            m_rects[rectangleIndex].height    = randint(1 ,(haarFeatureParametersPtr->m_height-m_rects[rectangleIndex].y-2));
            m_rsums[rectangleIndex]            = abs( m_weights[rectangleIndex] * (m_rects[rectangleIndex].width+1)*(m_rects[rectangleIndex].height+1)*255);
        
            ASSERT_TRUE(    m_rects[rectangleIndex].x >=0 && 
                            m_rects[rectangleIndex].y>= 0 && 
                            m_rects[rectangleIndex].width > 0 &&
                            m_rects[rectangleIndex].height > 0 
                        )
        }

        if ( haarFeatureParametersPtr->m_numberOfChannels < 0 )
        {
            haarFeatureParametersPtr->m_numberOfChannels = 0;

            for( int k=0; k < 1024; k++ )
            {
                haarFeatureParametersPtr->m_numberOfChannels += haarFeatureParametersPtr->m_useChannels[k]>=0;
            }
        }

        m_channel = haarFeatureParametersPtr->m_useChannels[randint(0,haarFeatureParametersPtr->m_numberOfChannels-1)];
    }

    /****************************************************************
        Feature::ToVisualize
            Visualize the Haar-Like feature
            Return an image. 
        Exception:
            None
    *******************************************************************/
    Matrixu    HaarFeature::ToVisualize( int featureIndex )
    {
        Matrixu v(m_height,m_width,3);
        v.Set(0);
        v._keepIpl = true;

        for( uint k=0; k<m_rects.size(); k++ )
        {
            if( m_weights[k] < 0 )
                //v.drawRect(m_rects[k],1,(int)(255*max(-1*m_weights[k],0.5)),0,0);
                v.drawRect(m_rects[k],1,(int)(255*max(-1*m_weights[k],(float) 0.5)),0,0);//[Zefeng Ni]: Corrected for GCC compatible
            else
                //v.drawRect(m_rects[k],1,0,(int)(255*max(m_weights[k],0.5)),(int)(255*max(m_weights[k],0.5)));
                v.drawRect(m_rects[k],1,0,(int)(255*max(m_weights[k],(float)0.5)),(int)(255*max(m_weights[k],(float)0.5))); //[Zefeng Ni]: Corrected for GCC compatible
        }

        if ( featureIndex >= 0 )
        {
            v.display(  ("HaarFeature-" +  int2str(featureIndex,3)).c_str(), 2 );
            cvWaitKey(1);            
        }

        v._keepIpl = false;

        return v;
    }

    /****************************************************************
    HaarFeature::Compute
        Computes Haar-Like features and stores it in the sampleSet.
    Exception:
        None
    ****************************************************************/
    inline float HaarFeature::Compute( const Classifier::Sample& sample ) const
    {
        //Integral image should be initialized
        if ( !sample.m_pImgGray->isInitII() ) 
        {
            abortError(__LINE__,__FILE__,"Integral image not initialized before called Compute()");
        }

        IppiRect r;
        float sum = 0.0f;

        for ( int k=0; k<(int)m_rects.size(); k++ )
        {
            r    = m_rects[k];        // the rectangle in the original scale
            //find the scaled rectangle
            r.x = cvRound( float(r.x) * sample.m_scaleX ) + sample.m_col;//r.x += sample.m_col; 
            r.y = cvRound( float(r.y) * sample.m_scaleY ) + sample.m_row;//r.y += sample.m_row;
            r.height = cvRound( float(r.height) * sample.m_scaleY );
            r.width     = cvRound( float(r.width) * sample.m_scaleX );

            sum += m_weights[k] * sample.m_pImgGray->sumRect( r, m_channel );
        }

        /* r.x        = sample.m_col;
        r.y            = sample.m_row;
        r.width        = (int)sample.m_weight;
        r.height    = (int)sample.m_height; */

        //return (float)(sum);
        return (float)(sum/(sample.m_scaleX*sample.m_scaleY)); //return the Haar feature as if the sample is of original scale (1.0)
    }

    /****************************************************************
    HaarFeature::=
        Assignment Operator - deep copy
    Exception:
        None
    ****************************************************************/
    inline HaarFeature&    HaarFeature::operator= ( const HaarFeature &a )
    {
        m_width        = a.m_width;
        m_height    = a.m_height;
        m_channel    = a.m_channel;
        m_weights    = a.m_weights;
        m_rects        = a.m_rects;
        m_maxSum    = a.m_maxSum;
        return (*this);
    }

    /****************************************************************
    HaarFeature::GetExpectedValue
        Get Expected value
    Exception:
        None
    ****************************************************************/
    inline float    HaarFeature::GetExpectedValue() const
    {
        float sum = 0.0f;
        for ( int k=0; k<(int)m_rects.size(); k++ )
        {
            sum += m_weights[k] * m_rects[k].height * m_rects[k].width * 125;
        }
        return sum;
    }
}