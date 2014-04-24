#ifndef H_SAMPLE_SET
#define H_SAMPLE_SET

#include "Sample.h"

namespace Classifier
{
    /****************************************************************
    SampleSet
        List of Samples. Takes care resizing samples,
        calculating features etc.
    ****************************************************************/
    class SampleSet
    {
    public:

        SampleSet( );
        SampleSet( const Sample& s );

        //sample list related
        const size_t        Size() const { return m_sampleList.size(); };
        //Careful while using Resize, its a partial clearing.
        void                Resize( size_t newSize ) { m_sampleList.resize(newSize); };
        void                Clear() { m_featureMatrix.clear(); m_sampleList.clear(); };
        Classifier::Sample &            operator[] (const int sampleIndex)  { return m_sampleList[sampleIndex]; };

        void                PushBackSample( const Classifier::Sample &s ) { m_sampleList.push_back(s); };
        void                PushBackSample(    Matrixu*    pGrayImageMatrix,
                                            int            x, 
                                            int            y, 
                                            int            width  = 0,
                                            int            height = 0,
                                            float        weight = 1.0f,
                                            Matrixu*    pRGBImageMatrix = NULL,
                                            Matrixu*    pHSVImageMatrix = NULL, 
                                            float        scaleX = 1.0,
                                            float        scaleY = 1.0 );

        //feature matrix related
        void                ResizeFeatures( size_t newSize );    
        float &                GetFeatureValue( int sample, int ftr) { return m_featureMatrix[ftr](sample); };
        float                GetFeatureValue( int sample, int ftr) const { return m_featureMatrix[ftr](sample); };
        Matrixf                FeatureValues(int ftr) const { return m_featureMatrix[ftr]; };
        bool                IsFeatureComputed( ) const { return !m_featureMatrix.empty() && !m_sampleList.empty() && m_featureMatrix[0].size()>0; };
        
        //Classifier::Sample images in the given ring of interest
        void                 SampleImage(    Matrixu*    pGrayImageMatrix,
                                            int            x, 
                                            int            y,
                                            int            width,
                                            int            height,
                                            float        outerCircleRadius,
                                            float        innerCircleRadius        =    0 ,
                                            int            maximumNumberOfSamples    =    1000000, 
                                            Matrixu*    pRGBImageMatrix            =    NULL, 
                                            Matrixu*    pHSVImageMatrix            =    NULL,
                                            float        scaleX                    =   1, 
                                            float        scaleY                    =    1 );

        //randomly sample "numberOfSamples" samples in the given grayImage
        void                SampleImage(    Matrixu*    pGrayImageMatrix,
                                            uint        numberOfSamples, 
                                            int            w,
                                            int            h,
                                            Matrixu*    pRGBImageMatrix =    NULL,
                                            Matrixu*    pHSVImageMatrix =    NULL,
                                            float        scaleX            =    1, 
                                            float        scaleY            =    1 );
        

        //sample image between two rectangles
        void                SampleImage(    Matrixu*    pGrayImageMatrix,
                                            int            x, 
                                            int            y,
                                            int            width,
                                            int            height,
                                            float        maximumDistanceX, 
                                            float        maximumDistanceY, 
                                            float        minimumDistanceX, 
                                            float        minimumDistanceY,
                                            int            maximumNumberOfSamples    = 1000000,
                                            Matrixu*    pRGBImageMatrix            = NULL, 
                                            Matrixu*    pHSVImageMatrix            = NULL,
                                            float        scaleX                    = 1, 
                                            float        scaleY                    = 1 );


    private:

        void SelectSamplesUniformlyFromLargerSet( int maximumNumberOfSamples );

        vector<Classifier::Sample>        m_sampleList;
        vector<Matrixf>                    m_featureMatrix;
    };
}
#endif