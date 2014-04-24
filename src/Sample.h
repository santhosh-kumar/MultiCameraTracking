#ifndef H_SAMPLE
#define H_SAMPLE

#include "Matrix.h"
#include "Public.h"
#include "CommonMacros.h"

namespace Classifier
{
    /****************************************************************
    Sample
        Wrapper for image samples.
    ****************************************************************/
    class Sample
    {
    public:
        //default constructor
        Sample( );

        //overloaded constructor
        Sample( Matrixu*    imgGray,
                int            row,
                int            col, 
                int            width        = 0,
                int            height        = 0,
                float        weight        = 1.0,
                Matrixu*    imgColor    = NULL,
                Matrixu*    imgHSV        = NULL,
                float        scaleX        = 1.0,
                float        scaleY        = 1.0 );

        //assignment operator
        Sample&    operator = ( const Sample& sample );

        Matrixu* GetColorImage( ) const { return m_pImgColor; }
        Matrixu* GetGrayImage( ) const    { return m_pImgGray; }
        Matrixu* GetHSVImage( ) const    { return m_pImgHSV;}

    public:

        Matrixu*            m_pImgGray;
        Matrixu*            m_pImgColor;//RGB color
        Matrixu*            m_pImgHSV;
        int                    m_row;
        int                    m_col;
        int                    m_width;
        int                    m_height;
        float                m_weight;
        float                m_scaleX;
        float                m_scaleY;
        int                    m_cameraID; //which camera the sample originates
    };
}
#endif