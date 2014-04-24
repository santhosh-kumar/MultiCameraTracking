#include "Sample.h"

namespace Classifier
{
    /****************************************************************
    Sample
        Constructor
    ****************************************************************/
    Sample::Sample( )
        : m_pImgGray( NULL ),
        m_col( 0 ),
        m_row( 0 ),
        m_height( 0 ),
        m_width( 0 ),
        m_weight( 1.0f ),
        m_pImgColor( NULL ),
        m_pImgHSV( NULL ),
        m_scaleX( 1 ),
        m_scaleY( 1 ),
        m_cameraID( 0 )
    {
    }

    /****************************************************************
    Sample
        Constructor
    Exceptions:
        None
    ****************************************************************/
    Sample::Sample( Matrixu*    imgGray,
                   int            row,
                   int            col, 
                   int            width,
                   int            height,
                   float        weight,
                   Matrixu*        imgColor,
                   Matrixu*        imgHSV,
                   float        scaleX,
                   float        scaleY )
        : m_pImgGray( imgGray ),
        m_col( col ),
        m_row( row ),
        m_height( height ),
        m_width( width ),
        m_weight( weight ),
        m_pImgColor ( imgColor ),
        m_pImgHSV( imgHSV ),
        m_scaleX( scaleX ),
        m_scaleY( scaleY ),
        m_cameraID( 0 )
    {
        ASSERT_TRUE( m_row >= 0 && m_col >= 0 && m_height >= 0 && m_width >= 0 && m_scaleY > 0 && m_scaleX );
    }

    /****************************************************************
    Sample::operator=
        Assignment operator.
        Everything copied except the weight
    Exceptions:
        None
    ****************************************************************/
    Sample&    Sample::operator= ( const Sample& sample )
    {
        m_pImgGray    = sample.m_pImgGray;
        m_pImgColor    = sample.m_pImgColor;
        m_pImgHSV    = sample.m_pImgHSV;

        m_row        = sample.m_row;
        m_col        = sample.m_col;
        m_width        = sample.m_width;
        m_height    = sample.m_height;
        m_scaleX    = sample.m_scaleX;
        m_scaleY    = sample.m_scaleY;
        
        return (*this);
    }
}
