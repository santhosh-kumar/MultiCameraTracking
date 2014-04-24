#include "SampleSet.h"

namespace Classifier
{
    /****************************************************************
    Classifier::SampleSet::Classifier::SampleSet
        C'tor
    Exceptions:
        None
    ****************************************************************/
    SampleSet::SampleSet( )
        : m_sampleList( ),
        m_featureMatrix( )
    {
    }

    /****************************************************************
    Classifier::SampleSet::Classifier::SampleSet
        C'tor
    Exceptions:
        None
    ****************************************************************/
    SampleSet::SampleSet( const Sample& sample )
    {
        m_sampleList.push_back(sample); 
    }


    /****************************************************************
    Classifier::SampleSet::ResizeFeatures
        resize the feature matrix to new size
    Exceptions:
        None
    ****************************************************************/
    void    SampleSet::ResizeFeatures( size_t newSize )
    {
        m_featureMatrix.resize(newSize);

        size_t numberOfSamples = m_sampleList.size();

        if ( numberOfSamples > 0 )
        {
            for ( int k=0; k < newSize; k++ )
            {
                m_featureMatrix[k].Resize( 1, numberOfSamples );
            }
        }
    }

    /****************************************************************
    Classifier::SampleSet::PushBackSample
        Pushes the sample into the list.
    Exceptions:
        None
    ****************************************************************/
    void    SampleSet::PushBackSample(    Matrixu*    pGrayImageMatrix,
                                        int            x, 
                                        int            y, 
                                        int            width,
                                        int            height,
                                        float        weight,
                                        Matrixu*    pRGBImageMatrix,
                                        Matrixu*    pHSVImageMatrix, 
                                        float        scaleX,
                                        float        scaleY )
    { 
        Classifier::Sample sample( pGrayImageMatrix, y ,x, width ,height, weight, pRGBImageMatrix, pHSVImageMatrix, scaleX, scaleY ); 
        PushBackSample(sample); 
    }

    /****************************************************************
    SampleImage
        Samples the image (constraint by two concentric circles) and stores the sample in a list.
        Takes the starting (x,y) and width,height. 
        Within the ranged defined by the two circles 
        [large circle: in-radius (not including) and smaller circle: out-radius (including)],
        it randomly samples the image (with an uniform distribution)
        when innerCircleRadius=0 (default), then just samples points inside sample circle 
    Exceptions:
        None
    ****************************************************************/
    void    SampleSet::SampleImage( Matrixu*    pGrayImageMatrix,
                                    int            x, 
                                    int            y,
                                    int            width,
                                    int            height,
                                    float        outerCircleRadius, //larger circle
                                    float        innerCircleRadius,//smaller circle
                                    int            maximumNumberOfSamples,
                                    Matrixu*    pRGBImageMatrix, 
                                    Matrixu*    pHSVImageMatrix,
                                    float        scaleX, 
                                    float        scaleY )
    {
        try
        {
            ASSERT_TRUE( outerCircleRadius > innerCircleRadius );
            ASSERT_TRUE( pGrayImageMatrix != NULL || pRGBImageMatrix != NULL || pHSVImageMatrix    != NULL );
        

            int scaledWidth        = cvRound( float(width) * scaleX );
            int scaledHeight    = cvRound( float(height)* scaleY );
            
            int numberOfRows;    
            int numberOfColumns;
            if ( pGrayImageMatrix != NULL ) 
            {        
                numberOfRows = pGrayImageMatrix->rows() - scaledHeight - 1;
                numberOfColumns = pGrayImageMatrix->cols() - scaledWidth - 1;
            }
            else if( pRGBImageMatrix != NULL )
            {
                numberOfRows = pRGBImageMatrix->rows() - scaledHeight - 1;
                numberOfColumns = pRGBImageMatrix->cols() - scaledWidth - 1;
            }
            else
            {
                numberOfRows = pHSVImageMatrix->rows() - scaledHeight - 1;
                numberOfColumns = pHSVImageMatrix->cols() - scaledWidth - 1;
            }
    

            float outerCircleRadiusSquare = outerCircleRadius * outerCircleRadius;
            float innerCircleRadiusSquare = innerCircleRadius * innerCircleRadius;

            int distance;

            uint minrow = max( 0, (int)y - (int)outerCircleRadius );
            uint maxrow = min( (int)numberOfRows - 1, (int)y + (int)outerCircleRadius );
            uint mincol = max( 0, (int)x - (int)outerCircleRadius );
            uint maxcol = min( (int)numberOfColumns - 1, (int)x + (int)outerCircleRadius);

            //resample based on the area of the ring of interest
            m_sampleList.resize( (maxrow-minrow+1)*(maxcol-mincol+1) );
            
            int validSampleIndex = 0;
            
            //find the total number of samples within the range
            for ( int r = minrow; r <= (int)maxrow; r++ )
            {
                for ( int c = mincol; c <= (int)maxcol; c++ )
                {
                    distance = (y-r)*(y-r) + (x-c)*(x-c);

                    //should be a valid sample
                    if ( distance < outerCircleRadiusSquare && distance >= innerCircleRadiusSquare ) 
                    {
                        m_sampleList[validSampleIndex].m_pImgGray    = pGrayImageMatrix;
                        m_sampleList[validSampleIndex].m_col        = c;
                        m_sampleList[validSampleIndex].m_row        = r;
                        m_sampleList[validSampleIndex].m_height        = height;
                        m_sampleList[validSampleIndex].m_width        = width;
                        m_sampleList[validSampleIndex].m_pImgColor    = pRGBImageMatrix;
                        m_sampleList[validSampleIndex].m_pImgHSV    = pHSVImageMatrix;
                        m_sampleList[validSampleIndex].m_scaleX        = scaleX;
                        m_sampleList[validSampleIndex].m_scaleY        = scaleY;
                        validSampleIndex++;
                    }
                }
            }

            //resize to valid sample index
            m_sampleList.resize(validSampleIndex);

            SelectSamplesUniformlyFromLargerSet( maximumNumberOfSamples );
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to sample images in the given ring of interest" );
    }

    /****************************************************************
    SampleImage
        Samples the image (constraint by two rectangles) and stores the sample in a list.
        Takes the starting (x,y) and width,height. Within the range within 
        a rectangle box [maximumDistanceX, maximumDistanceY] (excluding) 
        and outside a smaller rectangle box [minimumDistanceX minimumDistanceY] (including)],
        it randomly samples within the range (with an uniform distribution)
    Exceptions:
        None
    ****************************************************************/
    void    SampleSet::SampleImage(    Matrixu*    pGrayImageMatrix,
                                    int            x, 
                                    int            y,
                                    int            width,
                                    int            height,
                                    float        maximumDistanceX, 
                                    float        maximumDistanceY, 
                                    float        minimumDistanceX, 
                                    float        minimumDistanceY,
                                    int            maximumNumberOfSamples,
                                    Matrixu*    pRGBImageMatrix, 
                                    Matrixu*    pHSVImageMatrix,
                                    float        scaleX, 
                                    float        scaleY )
    {
        try
        {
            ASSERT_TRUE( maximumDistanceY > minimumDistanceY );
            ASSERT_TRUE( maximumDistanceX > minimumDistanceX );

            ASSERT_TRUE( pGrayImageMatrix != NULL || pRGBImageMatrix != NULL || pHSVImageMatrix    != NULL );

            int scaledWidth = cvRound( float(width)*scaleX );
            int scaledHeight = cvRound( float(height)*scaleY );

            int numberOfRows;    
            int numberOfColumns;
            if ( pGrayImageMatrix != NULL ) 
            {        
                numberOfRows = pGrayImageMatrix->rows() - scaledHeight - 1;
                numberOfColumns = pGrayImageMatrix->cols() - scaledWidth - 1;
            }
            else if( pRGBImageMatrix != NULL )
            {
                numberOfRows = pRGBImageMatrix->rows() - scaledHeight - 1;
                numberOfColumns = pRGBImageMatrix->cols() - scaledWidth - 1;
            }
            else
            {
                numberOfRows = pHSVImageMatrix->rows() - scaledHeight - 1;
                numberOfColumns = pHSVImageMatrix->cols() - scaledWidth - 1;
            }

            uint minrow = max( 0, (int)y - (int)maximumDistanceY );
            uint maxrow = min( (int)numberOfRows-1, (int)y + (int)maximumDistanceY );
            uint mincol = max( 0, (int)x - (int)maximumDistanceX );
            uint maxcol = min( (int)numberOfColumns-1, (int)x + (int)maximumDistanceX );

            m_sampleList.resize( (maxrow-minrow+1) * (maxcol-mincol+1) );
            
            int validSampleIndex = 0;
            int distanceX, distanceY;

            for ( int r = minrow; r <= (int)maxrow; r++ )
            {
                for ( int c = mincol; c <= (int)maxcol; c++ )
                {
                    distanceX = abs( x-c ); distanceY = abs( y-r );
                    if ( distanceX >= minimumDistanceX || distanceY >= minimumDistanceY ) 
                    {
                        m_sampleList[validSampleIndex].m_pImgGray    = pGrayImageMatrix;
                        m_sampleList[validSampleIndex].m_col        = c;
                        m_sampleList[validSampleIndex].m_row        = r;
                        m_sampleList[validSampleIndex].m_height        = height;
                        m_sampleList[validSampleIndex].m_width        = width;
                        m_sampleList[validSampleIndex].m_pImgColor    = pRGBImageMatrix;
                        m_sampleList[validSampleIndex].m_pImgHSV    = pHSVImageMatrix;
                        m_sampleList[validSampleIndex].m_scaleX        = scaleX;
                        m_sampleList[validSampleIndex].m_scaleY        = scaleY;
                        validSampleIndex++;
                    }
                }
            }

            //resize to valid sample index
            m_sampleList.resize(validSampleIndex);

            SelectSamplesUniformlyFromLargerSet( maximumNumberOfSamples );
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to sample image inbetween two ROI with different radius" );
    }

    /****************************************************************
    SampleImage
        Samples the entire image and stores the sample in a list.
        Based on width,height it randomly samples the entire image.
    Exceptions:
        None
    ****************************************************************/
    void    SampleSet::SampleImage(Matrixu*    pGrayImageMatrix,
                                   uint        numberOfSamples, 
                                   int        w,
                                   int        h,
                                   Matrixu*    pRGBImageMatrix,
                                   Matrixu*    pHSVImageMatrix,
                                   float    scaleX, 
                                   float    scaleY )
    {
        try
        {
            ASSERT_TRUE( pGrayImageMatrix != NULL || pRGBImageMatrix != NULL || pHSVImageMatrix    != NULL );
            //find the width and height based on the current scale
            int scaledWidth        = cvRound( float(w) * scaleX );
            int scaledHeight    = cvRound( float(h) * scaleY );

            int numberOfRows;    
            int numberOfColumns;
            if ( pGrayImageMatrix != NULL ) 
            {        
                numberOfRows = pGrayImageMatrix->rows() - scaledHeight - 1;
                numberOfColumns = pGrayImageMatrix->cols() - scaledWidth - 1;
            }
            else if( pRGBImageMatrix != NULL )
            {
                numberOfRows = pRGBImageMatrix->rows() - scaledHeight - 1;
                numberOfColumns = pRGBImageMatrix->cols() - scaledWidth - 1;
            }
            else
            {
                numberOfRows = pHSVImageMatrix->rows() - scaledHeight - 1;
                numberOfColumns = pHSVImageMatrix->cols() - scaledWidth - 1;
            }

            ASSERT_TRUE( numberOfSamples <= ( numberOfRows * numberOfColumns ) );

            //resize the sample list to the required number of samples
            m_sampleList.resize( numberOfSamples );

            #pragma omp for
            for ( int i = 0; i < (int)numberOfSamples; i++ )
            {
                m_sampleList[i].m_pImgGray    = pGrayImageMatrix;
                m_sampleList[i].m_col        = randint( 0, numberOfColumns );
                m_sampleList[i].m_row        = randint( 0, numberOfRows );
                m_sampleList[i].m_height    = h;
                m_sampleList[i].m_width        = w;
                m_sampleList[i].m_pImgColor = pRGBImageMatrix;
                m_sampleList[i].m_pImgHSV    = pHSVImageMatrix;
                m_sampleList[i].m_scaleX    = scaleX;
                m_sampleList[i].m_scaleY    = scaleY;
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to random sample entire image for the number of samples" );
    }

    /****************************************************************
    SelectSamplesUniformlyFromLargerSet
        Select 
    Exceptions:
        None
    ****************************************************************/
    void SampleSet::SelectSamplesUniformlyFromLargerSet( int maximumNumberOfSamples )
    {
        try
        {
            uint numberOfSamples = Size();

            //random sample according to uniform distribution 
            float probability = ( static_cast<float>(maximumNumberOfSamples+1) ) / numberOfSamples;

            int i = 0;
            if ( probability < 1 )
            {    
                // random pick maximumNumberOfSamples from all available samples
                while( i <  m_sampleList.size() )
                {
                    if ( randfloat() >  probability  )
                    { 
                        //remove this sample
                        m_sampleList.erase( m_sampleList.begin() + i );
                    }
                    else
                    { 
                        //keep this sample and move on the next one
                        i++;
                    }
                }
                m_sampleList.resize( min( i, maximumNumberOfSamples ) );
            }            
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to select samples uniformly from larger set." );
    }
}