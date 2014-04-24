#include "CultureColorHistogram.h"

namespace Features
{
    /****************************************************************
    CultureColorHistogram
        Initialize the feature
    Exception:
        None.    
    ****************************************************************/
    void CultureColorHistogram::Generate( FeatureParametersPtr featureParametersPtr )
    { 
        CultureColorHistogramParametersPtr temp = boost::dynamic_pointer_cast<CultureColorHistogramParameters>(featureParametersPtr);
        m_partPercentageVertical    = temp->m_partPercentageVertical;
        m_numberOfParts                = temp->m_numberOfParts;
    }


    /****************************************************************
    CultureColorHistogram
        Compute the feature for the sample
    Exception:
        None.    
    ****************************************************************/
    void CultureColorHistogram::Compute( const Classifier::Sample& sample, vectorf& featureValueList ) const
    {        
        uint scaled_height = cvRound( static_cast<float>(sample.m_height) * sample.m_scaleY );
        uint scaled_width =cvRound( static_cast<float>(sample.m_width) * sample.m_scaleX );

        if( sample.GetHSVImage() == NULL )
        {
            cv::Mat sampleImgHSV;
            // create HSV from sample image 
            sample.GetColorImage()->createIpl(true);
            cv::Mat entireImg = sample.GetColorImage()->getIpl();
            cv::Rect roi( sample.m_col, sample.m_row, scaled_width, scaled_height );
            cv::Mat sampleImg = entireImg(roi);
            //cv::imshow(sampleImg, "testCultureColor");        
            cv::cvtColor( sampleImg, sampleImgHSV, CV_BGR2HSV );
            ComputeFeature( sample,  sampleImgHSV, featureValueList );    
        }
        else
        {
            Matrixu * pImgHSV =  sample.GetHSVImage();
            
            // Calculate the size for each part.
            vectori partRowList;
            int accum = 0, partEnd;
            
            
            partRowList.push_back( 0 );
            for (int i = 0; i < m_numberOfParts; i++)
            {    
                accum += m_partPercentageVertical[i];  
                partEnd = scaled_height*accum/100; 
                ASSERT_TRUE ( partEnd > partRowList[i] );
                partRowList.push_back( partEnd );
            }

            // HSV Centroids:
            int numBins=11;
            //           Red    Pink    purple    blue    green    yellow    orange    brown    bk        gray    white
            int HC[] = {5,    206,    181,    160,    81,        46,        28,        20,        110,    121,    133};
            //int SC[] = {206,168,    186,    215,    188,    219,    211,    137,    14,        8,        15};  
            //int VC[] = {107,120,    117,    129,    104,    122,    137,    97,        2,        101,    237};
            int i, j, byte, H=0, S=0, V=0, DistanceToCentroids[11], minimumDist, CC_Histogram[11], indicator;

            // calculate histogram for each part
            for( int partIndex = 0; partIndex < m_numberOfParts; partIndex++ ) 
            {
                //initialization
                for( j = 1; j < numBins; j++)
                {
                    CC_Histogram[j]=0;
                    DistanceToCentroids[j] = 0;
                }
                
                for ( i = partRowList[partIndex]; i <  partRowList[partIndex+1]; i++)
                {
                    for ( j = 0; j < scaled_width; j++ )
                    {
                        minimumDist = 999999;
                        H = (*pImgHSV)( i+sample.m_row, j+sample.m_col, 0 );
                        //S = (*pImgHSV)( i+sample.m_row, j+sample.m_col, 1 );
                        //V = (*pImgHSV)( i+sample.m_row, j+sample.m_col, 2 );
                        for ( int c = 0; c < numBins; c++ )
                        {
                            DistanceToCentroids[c]=sqrt(
                                (double)(H-HC[0])*(double)(H-HC[0])                         
                            );     //hue channel only    
                                //+ long double(S-SC[0])*long double(S-SC[0]) 
                                //+ long double(V-VC[0])*long double(V-VC[0]) 

                            if( minimumDist > DistanceToCentroids[c] )
                            {
                                minimumDist = DistanceToCentroids[c];
                                indicator = c;
                            }
                        }
                        CC_Histogram[indicator]++;
                        }
                    }
                for( int c = 0; c < numBins; c++ )
                {
                    featureValueList[partIndex*numBins + c]= CC_Histogram[c];
                }             
            }    //for each part    

                
        }        
    }
    
    /****************************************************************
    CultureColorHistogram
    Compute the feature from sample HSV image of format cv::Mat
    Exception:
        None.    
    ****************************************************************/
    void CultureColorHistogram::ComputeFeature( const Classifier::Sample& sample,  cv::Mat& sampleImgHSV, vectorf& featureValueList ) const
    {
        // Calculate the size for each part.
        vectori partRowList;
        int accum = 0, partEnd;

        partRowList.push_back( 0 );
        for (int i = 0; i < m_numberOfParts; i++)
        {    
            accum += m_partPercentageVertical[i];  
            partEnd = sample.m_height*accum/100; 
            ASSERT_TRUE ( partEnd > partRowList[i] );
            partRowList.push_back( partEnd );
        }

        // HSV Centroids:
        int numBins=11;
        //           Red    Pink    purple    blue    green    yellow    orange    brown    bk        gray    white
        int HC[] = {5,    206,    181,    160,    81,        46,        28,        20,        110,    121,    133};
        //int SC[] = {206,168,    186,    215,    188,    219,    211,    137,    14,        8,        15};  
        //int VC[] = {107,120,    117,    129,    104,    122,    137,    97,        2,        101,    237};
        int i, j, byte, H=0, S=0, V=0, DistanceToCentroids[11], MinimumDist, CC_Histogram[11], indicator;

        // calculate histogram for each part
        for( int partIndex = 0; partIndex < m_numberOfParts; partIndex++ ) 
        {
            //initialization
            for( j = 1; j < numBins; j++)
            {
                CC_Histogram[j]=0;
                DistanceToCentroids[j] = 0;
            }
            
            for ( i = partRowList[partIndex]; i <  partRowList[partIndex+1]; i++)
            {
                for ( j = 0; j < sample.m_width; j++ )
                {
                    MinimumDist =999999;
                    H = sampleImgHSV.at<cv::Vec3b>( i, j)[0];
                    //S = sampleImgHSV.at<cv::Vec3b>( i, j)[1];
                    //V = sampleImgHSV.at<cv::Vec3b>( i, j)[2];
                    for ( int c = 0; c < numBins; c++ )
                    {
                        DistanceToCentroids[c]=sqrt(
                            (double)(H-HC[0])*(double)(H-HC[0])                         
                        );     //hue channel only    
                            //+ long double(S-SC[0])*long double(S-SC[0]) 
                            //+ long double(V-VC[0])*long double(V-VC[0]) 

                        if( MinimumDist > DistanceToCentroids[c] )
                        {
                            MinimumDist = DistanceToCentroids[c];
                            indicator = c;
                        }
                    }
                    CC_Histogram[indicator]++;
                    }
                }
            for( int c = 0; c < numBins; c++ )
            {
                featureValueList[partIndex*numBins + c]= CC_Histogram[c];
            }             
        }    //for each part    

    }

}
