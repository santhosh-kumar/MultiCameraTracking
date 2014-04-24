#ifndef H_MATRIX
#define H_MATRIX

#include "Public.h"

template<class T> class Matrix;
typedef Matrix<float>    Matrixf;
typedef Matrix<uchar>    Matrixu;

#ifndef WIN32
#include <typeinfo> //[Zefeng Ni] for gcc compatibility
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
// This is an IPP based matrix class.  It can be used for both matrix math and for multi channel
// image manipulation.

template<class T> class Matrix
{

public:
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // members
    int                _rows,_cols,_depth;
    vector<void*>    _data;
private:
    // image specific
    int                _dataStep;
    IplImage        *_iplimg;
    // integral images
    vector<Ipp32f*>    _iidata;
    int                _iidataStep;
    int                _iipixStep;
    bool            _ii_init;

    IppiSize        _roi; //whole image roi (needed for some functions)
    IppiRect        _roirect;

public:
    bool            _keepIpl;  // if set to true, calling freeIpl() will have no effect;  this is for speed up only...

public:
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // constructors
                Matrix();
                Matrix( int rows, int cols );
                Matrix( int rows, int cols, int depth );
                Matrix( const Matrix<T>& x );
                Matrix( const vector<T>& v );
                ~Matrix();
    static        Matrix<T>    Eye( int sz );
    void        Resize( uint rows, uint cols, uint depth=1 );
    void        Resize( uint depth );
    void        Free();
    void        Set(T val);
    void        Set(T val, int channel);

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // access
    T&            operator() ( const int k ) const;
    T&            operator() ( const int row, const int col ) const;
    T&            operator() ( const int row, const int col, const int depth ) const;
    vector<T>    operator() ( const vectori rows, const vectori cols );
    vector<T>    operator() ( const vectori rows, const vectori cols, const vectori depths );
    float        ii ( const int row, const int col, const int depth ) const;
    Matrix<T>    getCh(uint ch);
    IplImage*    getIpl() { return _iplimg; };

    int            rows() const { return _rows; };
    int            cols() const { return _cols; };
    int            depth() const { return _depth; };
    uint        size() const { return _cols*_rows; };
    int            length() const { return max(_cols,_rows); };

    //Matrix<T>& operator= ( const vector<T> &x )
    //vector<T>    toVec();

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // matrix operations
    Matrix<T>&    operator=  ( const Matrix<T> &x );
    Matrix<T>&    operator=  ( const vector<T> &x );
    Matrix<T>    operator+ ( const Matrix<T> &b ) const;
    Matrix<T>    operator+ ( const T &a) const;
    Matrix<T>    operator- ( const Matrix<T> &b ) const;
    Matrix<T>    operator- ( const T &a) const;
    Matrix<T>    operator* ( const T &a) const;
    Matrix<T>    operator& ( const Matrix<T> &b) const;
    Matrixu        operator< ( const T &a) const;
    Matrixu        operator> ( const T &a) const;
    Matrix<T>    normalize() const;
    Matrix<T>    Sqr() const;
    Matrix<T>    Exp() const;
    void        Trans(Matrix<T> &res);
    T            Max(uint channel=0) const;
    T            Min(uint channel=0) const;
    double        Sum(uint channel=0) const;
    void        Max(T &val, uint &row, uint &col, uint channel=0) const;
    void        Min(T &val, uint &row, uint &col, uint channel=0) const;
    float        Mean(uint channel=0) const;
    float        Var(uint channel=0) const;
    float        VarW(const Matrixf &w, T *mu=NULL) const;
    float        MeanW(const vectorf &w)  const;
    float        MeanW(const Matrixf &w)  const;
    float        Dot(const Matrixf &x);


    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // image operations
    //        Note: many of these functions use OpenCV.  To do this, createIpl() is called to create an Ipl version of the image (OpenCV format).
    //        At the end of these functions freeIpl() is called to erase the Ipl version.  For speed up purposes, you can see Matrix._keepIpl=true
    //        to prevent these functions from erasing the Ipl image.  However, care must be taken to erase the image and create a new one when the
    //        Matrix changes or gets updated somehow (otherwise the Matrix will change, but the Ipl will stay the same).

    void        initII();
    bool        isInitII() const { return _ii_init; };
    void        FreeII();
    float        sumRect(const IppiRect &rect, int channel) const;
    void        drawRect(IppiRect rect, int lineWidth=3, int R=255, int G=0, int B=0);
    void        drawRect(float width, float height, float x,float y, float sc, float th, int lineWidth=3, int R=255, int G=0, int B=0);
    void        drawEllipse(float height, float width, float x,float y, int lineWidth=3, int R=255, int G=0, int B=0);
    void        drawEllipse(float height, float width, float x,float y, float startang, float endang, int lineWidth=3, int R=255, int G=0, int B=0);
    void        drawText(const char* txt, float x, float y, int R=255, int G=255, int B=0);
    void        warp(Matrixu &res, uint rows, uint cols, float x, float y, float sc=1.0f, float th=0.0f, float sr=1.0f, float phi=0.0f);
    void        warpAll(uint rows, uint cols, vector<vectorf> params, vector<Matrixu> &res);
    void        computeGradChannels();
    Matrixu        imResize(float p, float x=-1);
    void        conv2RGB(Matrixu &res);
    void        conv2HSV(Matrixu &res);
    void        conv2BW(Matrixu &res);
    float        dii_dx(uint x, uint y, uint channel=0);
    float        dii_dy(uint x, uint y, uint channel=0);

    void        createIpl(bool force=false);
    void        freeIpl();

    void        LoadImage(const char *filename, bool color=false);
    void        SaveImage(const char *filename);
    static void    SaveImages(vector<Matrixu> imgs, const char *dirname, float resize=1.0f);
    static vector<Matrixu> LoadVideo(const char *dirname, const char *basename, const char *ext, int start, int end, int digits, bool color=false);
    static vector<Matrixu> LoadVideoStream(const char *aviFileName, int start, int end, bool color=false);
    static vector<Matrixu> LoadVideo(const char *fname, bool color=false, int maxframes=10000);
    static void    PlayVideo( vector<Matrixu> &vid, int wait=1 );
    static void    SaveVideo( vector<Matrixu> &vid, const char* fname, int fps=15 );
    static bool    CaptureImage(CvCapture* capture, Matrixu &res, int color=0);
    static bool WriteFrame(CvVideoWriter* w, Matrixu &img);
    static void    PlayCam(int color=0, const char* fname=NULL);
    static void PlayCamOpenCV();

    static Matrix<T>        vecMat2Mat(const vector<Matrix<T> > &x);
    static vector<Matrix<T> >    vecMatTranspose(const vector<Matrix<T> > &x);

    bool        DLMRead( const char *fname, char *delim="," ); // compatible with Matlab dlmread & dlmwrite
    bool        DLMWrite( const char *fname, char *delim="," );

    void        display(int fignum, float p=1.0f);
    void        display(const char* figName, float p=1.0f);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // misc
    Matrixu        convert2img(float min=0.0f, float max=0.0f);


    void        IplImage2Matrix(IplImage *img);
    void        GrayIplImage2Matrix(IplImage *img);


};

template<class T> ostream&            operator<< ( ostream& os, const Matrix<T>& x );

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// constructors
template<class T>                    Matrix<T>::Matrix()
{
    _rows        = 0;
    _cols        = 0;
    _depth        = 0;
    _iplimg        = NULL;
    _keepIpl    = false;
}

template<class T>                    Matrix<T>::Matrix(int rows, int cols)
{
    _rows        = 0;
    _cols        = 0;
    _depth        = 0;
    _iplimg        = NULL;
    _keepIpl    = false;
    _ii_init    = false;
    Resize(rows,cols,1);
}

template<class T>                    Matrix<T>::Matrix(int rows, int cols, int depth)
{
    _rows        = 0;
    _cols        = 0;
    _depth        = 0;
    _iplimg        = NULL;
    _keepIpl    = false;
    _ii_init    = false;
    Resize(rows,cols,depth);
}

template<class T>                    Matrix<T>::Matrix(const Matrix<T> &a)
{
    _rows        = 0;
    _cols        = 0;
    _depth        = 0;
    _iplimg        = NULL;
    _keepIpl    = (typeid(T) == typeid(uchar)) && a._keepIpl;
    _ii_init    = false;
    Resize(a._rows, a._cols, a._depth);
    if( typeid(T) == typeid(uchar) )
            for( uint k=0; k<_data.size(); k++ )
                ippiCopy_8u_C1R((Ipp8u*)a._data[k], a._dataStep, (Ipp8u*)_data[k], _dataStep, _roi );
        else
            for( uint k=0; k<_data.size(); k++ )
                ippiCopy_32f_C1R((Ipp32f*)a._data[k], a._dataStep, (Ipp32f*)_data[k], _dataStep, _roi );
    if( a._ii_init ){
        _iidata.resize(a._iidata.size());

        for( uint k=0; k<_iidata.size(); k++ ){
            if( _iidata[k] != NULL ) ippiFree(_iidata[k]);
            _iidata[k] = ippiMalloc_32f_C1(_cols+1,_rows+1,&(_iidataStep));
            _iipixStep = _iidataStep/sizeof(Ipp32f);
            ippiCopy_32f_C1R((Ipp32f*)a._iidata[k], a._iidataStep, (Ipp32f*)_iidata[k], _iidataStep, _roi );
        }
        _ii_init = true;
    }

    if( a._iplimg != NULL && typeid(T) == typeid(uchar))
    {
        ((Matrixu*)this)->createIpl();
        cvCopy(a._iplimg, _iplimg);
    }
}

template<class T> Matrix<T>            Matrix<T>::Eye( int sz )
{
    Matrix<T> res(sz,sz);
    for( int k=0; k<sz; k++ )
        res(k,k) = 1;
    return res;
}
template<class T> void                Matrix<T>::Resize(uint rows, uint cols, uint depth)
{
    if( rows<0 || cols<0 )
        abortError(__LINE__, __FILE__,"NEGATIVE MATRIX SIZE");

    if( _rows == rows && _cols == cols && _depth == depth ) return;
    bool err = false;
    Free();
    _rows = rows;
    _cols = cols;
    _depth = depth;

    _data.resize(depth);

    for( uint k=0; k<_data.size(); k++ ){
        if( typeid(T) == typeid(uchar) ){
            _data[k] = (void*)ippiMalloc_8u_C1(cols,rows,&(_dataStep));
        }
        else{
            _data[k] = (void*)ippiMalloc_32f_C1(cols,rows,&(_dataStep));//malloc((uint)rows*cols*sizeof(Ipp32f));
        }
        err = err || _data[k] == NULL;
    }

    _roi.width = cols;
    _roi.height = rows;
    _roirect.width = cols;
    _roirect.height = rows;
    _roirect.x = 0;
    _roirect.y = 0;
    Set(0);

    //free ipl
    if( _iplimg != NULL )
        cvReleaseImage(&_iplimg);

    if( err )
        abortError(__LINE__, __FILE__,"OUT OF MEMORY");
}

template<class T> void                Matrix<T>::Resize(uint depth)
{

    if( _depth == depth ) return;
    bool err=false;


    _data.resize(depth);

    for( uint k=_depth; k<depth; k++ ){
        if( typeid(T) == typeid(uchar) ){
            _data[k] = (void*)ippiMalloc_8u_C1(_cols,_rows,&(_dataStep));
        }
        else{
            _data[k] = (void*)ippiMalloc_32f_C1(_cols,_rows,&(_dataStep));//malloc((uint)rows*cols*sizeof(Ipp32f));
        }
        err = err || _data[k] == NULL;
        Set(0,k);
    }
    _depth = depth;


    if( err )
        abortError(__LINE__, __FILE__,"OUT OF MEMORY");
}

template<class T> void                Matrix<T>::Free()
{
    if( _ii_init ) FreeII();
    if( _iplimg != NULL ) cvReleaseImage(&_iplimg);
    _ii_init = false;

    for( uint k=0;  k<_data.size(); k++ )
        if( _data[k] != NULL )
            if( typeid(T) == typeid(uchar) ){
                ippiFree((Ipp8u*)_data[k]);
            }
            else{
                ippiFree((Ipp32f*)_data[k]);
            }

    _rows = 0;
    _cols = 0;
    _depth = 0;
    _data.resize(0);
}

template<class T> void                Matrix<T>::Set(T val)
{
    for( uint k=0; k<_data.size(); k++ )
        if( typeid(T) == typeid(uchar) ){
            ippiSet_8u_C1R((Ipp8u)val,(Ipp8u*)_data[k], _dataStep,_roi);
        }
        else{
            ippiSet_32f_C1R((Ipp32f)val,(Ipp32f*)_data[k], _dataStep,_roi);
        }
            //for( uint j=0; j<(uint)_rows*_dataStep; j++ )
            //    ((Ipp32f*)_data[k])[j] = val;
}
template<class T> void                Matrix<T>::Set(T val, int k)
{
    if( typeid(T) == typeid(uchar) ){
        ippiSet_8u_C1R((Ipp8u)val,(Ipp8u*)_data[k], _dataStep,_roi);
    }
    else{
        ippiSet_32f_C1R((Ipp32f)val,(Ipp32f*)_data[k], _dataStep,_roi);
    }
}


template<class T> void                Matrix<T>::FreeII()
{
    for( uint k=0;  k<_iidata.size(); k++ )
        ippiFree(_iidata[k]);
    _iidata.resize(0);
    _ii_init = false;
}

template<class T>                    Matrix<T>::~Matrix()
{
    Free();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
// operators

template<class T> inline T&            Matrix<T>::operator() ( const int row, const int col, const int depth ) const
{
    if( typeid(T) == typeid(uchar) )
        return (T&) ((Ipp8u*)_data[depth])[row*(_dataStep) + col];
    else
        return (T&) ((Ipp32f*)_data[depth])[row*(_dataStep/sizeof(Ipp32f)) + col];
}

template<class T> inline T&            Matrix<T>::operator() ( const int row, const int col ) const
{
    return (*this)(row,col,0);
}
template<class T> inline T&            Matrix<T>::operator() ( const int k ) const
{
    return (*this)(k/_cols,k%_cols,0);
}
template<class T> inline vector<T>    Matrix<T>::operator() ( const vectori rows, const vectori cols )
{
    assert(rows.size() == cols.size());
    vector<T> res;
    res.resize(rows.size());
    for( uint k=0; k<rows.size(); k++ )
        res[k] = (*this)(rows[k],cols[k]);
    return res;
}

template<class T> inline vector<T>    Matrix<T>::operator() ( const vectori rows, const vectori cols, const vectori depths )
{
    assert(rows.size() == cols.size() && cols.size() == depths.size());
    vector<T> res;
    res.resize(rows.size());
    for( uint k=0; k<rows.size(); k++ )
        res[k] = (*this)(rows[k],cols[k],depths[k]);
    return res;
}

template<class T> inline Matrix<T>    Matrix<T>::getCh(uint ch)
{
    Matrix<T> a(_rows, _cols, 1);
    if( typeid(T) == typeid(uchar) )
            ippiCopy_8u_C1R((Ipp8u*)_data[ch], _dataStep, (Ipp8u*)a._data[0], a._dataStep, _roi );
        else
            ippiCopy_32f_C1R((Ipp32f*)_data[ch], _dataStep, (Ipp32f*)a._data[0], a._dataStep, _roi );
    return a;
}
template<class T> Matrix<T>&        Matrix<T>::operator= ( const Matrix<T> &a )
{
    if( this != &a ){
        Resize(a._rows, a._cols, a._depth);
        if( typeid(T) == typeid(uchar) )
            for( uint k=0; k<_data.size(); k++ )
                ippiCopy_8u_C1R((Ipp8u*)a._data[k], a._dataStep, (Ipp8u*)_data[k], _dataStep, _roi );
        else
            for( uint k=0; k<_data.size(); k++ )
                ippiCopy_32f_C1R((Ipp32f*)a._data[k], a._dataStep, (Ipp32f*)_data[k], _dataStep, _roi );
                //ippmCopy_va_32f_SS((Ipp32f*)a._data[k],sizeof(Ipp32f)*_cols,sizeof(Ipp32f),(Ipp32f*)_data[k],sizeof(Ipp32f)*_cols,sizeof(Ipp32f),_cols,_rows);
        if( a._ii_init ){
            _iidata.resize(a._iidata.size());

            for( uint k=0; k<_iidata.size(); k++ ){
                if( _iidata[k] != NULL ) ippiFree(_iidata[k]);
                _iidata[k] = ippiMalloc_32f_C1(_cols+1,_rows+1,&(_iidataStep));
                _iipixStep = _iidataStep/sizeof(Ipp32f);
                ippiCopy_32f_C1R((Ipp32f*)a._iidata[k], a._iidataStep, (Ipp32f*)_iidata[k], _iidataStep, _roi );
            }
            _ii_init = true;
        }

    }
    return (*this);
}





template<class T> Matrix<T>&        Matrix<T>::operator= ( const vector<T> &a )
{
    Resize(1,a.size(),1);
    for( uint k=0; k<a.size(); k++ )
        (*this)(k) = a[k];

    return (*this);
}





template<class T> Matrix<T>            Matrix<T>::operator+ ( const Matrix<T> &a ) const
{
    Matrix<T> res(rows(),cols());
    assert(rows()==a.rows() && cols()==a.cols());

    if( typeid(T) == typeid(uchar) )
        for( uint k=0; k<_data.size(); k++ )
            ippiAdd_8u_C1RSfs((Ipp8u*)a._data[k], a._dataStep, (Ipp8u*)_data[k], _dataStep,(Ipp8u*)res._data[k], res._dataStep,
                _roi, 0);
    else
        for( uint k=0; k<_data.size(); k++ )
            ippiAdd_32f_C1R((Ipp32f*)a._data[k], a._dataStep, (Ipp32f*)_data[k], _dataStep,(Ipp32f*)res._data[k], res._dataStep,
                _roi);

    return res;
}

template<class T> Matrix<T>            Matrix<T>::operator+ ( const T &a ) const
{
    Matrix<T> res;
    res.Resize(rows(),cols(),depth());

    if( typeid(T) == typeid(uchar) )
        for( uint k=0; k<_data.size(); k++ )
            ippiAddC_8u_C1RSfs((Ipp8u*)_data[k], _dataStep,(Ipp8u)a,(Ipp8u*)res._data[k], res._dataStep,
                _roi, 0);
    else
        for( uint k=0; k<_data.size(); k++ )
            ippiAddC_32f_C1R((Ipp32f*)_data[k], _dataStep,(Ipp32f)a,(Ipp32f*)res._data[k], res._dataStep,
                _roi);

    return res;
}
template<class T> Matrix<T>            Matrix<T>::operator- ( const Matrix<T> &a ) const
{
    return (*this) + (a*-1);
}

template<class T> Matrix<T>            Matrix<T>::operator- ( const T &a ) const
{
    return (*this) + (a*-1);
}
template<class T> Matrix<T>            Matrix<T>::operator* ( const T &a ) const
{
    Matrix<T> res(rows(),cols());

    if( typeid(T) == typeid(uchar) )
        for( uint k=0; k<_data.size(); k++ )
            ippiMulC_8u_C1RSfs((Ipp8u*)_data[k], _dataStep,(Ipp8u)a,(Ipp8u*)res._data[k], res._dataStep,
                _roi, 0);
    else
        for( uint k=0; k<_data.size(); k++ )
            ippiMulC_32f_C1R((Ipp32f*)_data[k], _dataStep,(Ipp32f)a,(Ipp32f*)res._data[k], res._dataStep,
                _roi);

    return res;
}
template<class T> Matrix<T>            Matrix<T>::operator& ( const Matrix<T> &b) const
{
    Matrix<T> res(rows(),cols());
    assert(rows()==b.rows() && cols()==b.cols() && depth()==b.depth());

    if( typeid(T) == typeid(uchar) )
        for( uint k=0; k<_data.size(); k++ )
            ippiMul_8u_C1RSfs((Ipp8u*)_data[k], _dataStep,(Ipp8u*)b._data[k], b._dataStep,(Ipp8u*)res._data[k], res._dataStep,
                _roi, 1);
    else
        for( uint k=0; k<_data.size(); k++ )
            ippiMul_32f_C1R((Ipp32f*)_data[k], _dataStep,(Ipp32f*)b._data[k], b._dataStep,(Ipp32f*)res._data[k], res._dataStep,
                _roi);

    return res;
}

template<class T> Matrixu            Matrix<T>::operator< ( const T &b) const
{
    Matrixu res(rows(),cols());

    for( uint i=0; i<size(); i++ )
        res(i) = (uint) ((*this)(i) < b);

    return res;
}

template<class T> Matrixu            Matrix<T>::operator> ( const T &b) const
{
    Matrixu res(rows(),cols());

    for( uint i=0; i<size(); i++ )
        res(i) = (uint) ((*this)(i) > b);

    return res;
}
template<class T> Matrix<T>            Matrix<T>::normalize() const
{
    double sum = this->Sum();
    return (*this) * (T)(1.0/(sum+1e-6));
}
template<class T> Matrix<T>            Matrix<T>::Sqr ( ) const
{
    Matrix<T> res(rows(),cols());

    if( typeid(T) == typeid(uchar) )
        for( uint k=0; k<_data.size(); k++ )
            ippiSqr_8u_C1RSfs((Ipp8u*)_data[k], _dataStep,(Ipp8u*)res._data[k], res._dataStep,
                _roi, 0);
    else
        for( uint k=0; k<_data.size(); k++ )
            ippiSqr_32f_C1R((Ipp32f*)_data[k], _dataStep,(Ipp32f*)res._data[k], res._dataStep,
                _roi);

    return res;
}
template<class T> Matrix<T>            Matrix<T>::Exp ( ) const
{
    Matrix<T> res(rows(),cols());

    if( typeid(T) == typeid(uchar) )
        for( uint k=0; k<_data.size(); k++ )
            ippiExp_8u_C1RSfs((Ipp8u*)_data[k], _dataStep,(Ipp8u*)res._data[k], res._dataStep,
                _roi, 0);
    else
        for( uint k=0; k<_data.size(); k++ )
            ippiExp_32f_C1R((Ipp32f*)_data[k], _dataStep,(Ipp32f*)res._data[k], res._dataStep,
                _roi);

    return res;
}
template<class T> ostream&            operator<<(ostream& os, const Matrix<T>& x)
{  //display matrix
    os << "[ ";
    char tmp[1024];
    for (int j=0; j<x.rows(); j++) {
        if( j>0 ) os << "  ";
        for (int i=0; i<x.cols(); i++) {
            if( typeid(T) == typeid(uchar) )
                sprintf_s(tmp,"%3d",(int)x(j,i));
            else
                sprintf_s(tmp,"%02.2f",(float)x(j,i));
            os << tmp << " ";
        }
        if( j!=x.rows()-1 )
            os << "\n";
    }
    os << "]";
    return os;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class T> void                Matrix<T>::Trans(Matrix<T> &res)
{
    res.Resize(_cols,_rows,_depth);
    for( uint k=0; k<res._data.size(); k++ )
        if( typeid(T) == typeid(uchar) )
            ippiTranspose_8u_C1R((Ipp8u*)_data[k], _dataStep,(Ipp8u*)res._data[k], res._dataStep, _roi);
        else
            abortError(__LINE__,__FILE__,"Trans not implemented for floats");
            //ippmTranspose_m_32f((Ipp32f*)_data[k], sizeof(Ipp32f)*_cols, sizeof(Ipp32f), _rows, _cols, (Ipp32f*)res._data[k], sizeof(Ipp32f)*_rows, sizeof(Ipp32f));

}

template<class T> T                    Matrix<T>::Max(uint channel) const
{
    T max;
    if( typeid(T) == typeid(uchar) )
        ippiMax_8u_C1R((Ipp8u*)_data[channel], _dataStep, _roi, (Ipp8u*)&max);
    else{
        ippiMax_32f_C1R((Ipp32f*)_data[channel], _dataStep, _roi, (Ipp32f*)&max);
    }

    return max;
}

template<class T> T                    Matrix<T>::Min(uint channel)  const
{
    T min;
    if( typeid(T) == typeid(uchar) )
        ippiMin_8u_C1R((Ipp8u*)_data[channel], _dataStep, _roi, (Ipp8u*)&min);
    else
        ippiMin_32f_C1R((Ipp32f*)_data[channel], _dataStep, _roi, (Ipp32f*)&min);

    return min;
}




template<class T> void                Matrix<T>::Max(T &val, uint &row, uint &col, uint channel) const
{
    if( typeid(T) == typeid(uchar) )
        ippiMaxIndx_8u_C1R((Ipp8u*)_data[channel], _dataStep, _roi, (Ipp8u*)&val, (int*)&col, (int*)&row);
    else{
        ippiMaxIndx_32f_C1R((Ipp32f*)_data[channel], _dataStep, _roi, (Ipp32f*)&val, (int*)&col, (int*)&row);
    }

}

template<class T> void                Matrix<T>::Min(T &val, uint &row, uint &col, uint channel)  const
{
    if( typeid(T) == typeid(uchar) )
        ippiMinIndx_8u_C1R((Ipp8u*)_data[channel], _dataStep, _roi, (Ipp8u*)&val, (int*)&col, (int*)&row);
    else
        ippiMinIndx_32f_C1R((Ipp32f*)_data[channel], _dataStep, _roi, (Ipp32f*)&val, (int*)&col, (int*)&row);

}




template<class T> float                Matrix<T>::Mean(uint channel)  const
{
    double mean;
    if( typeid(T) == typeid(uchar) )
        ippiMean_8u_C1R((Ipp8u*)_data[channel], _dataStep, _roi, &mean );
    else
        ippiMean_32f_C1R((Ipp32f*)_data[channel], _dataStep, _roi, &mean, ippAlgHintFast);

    return (float)mean;
}

template<class T> float                Matrix<T>::Var(uint channel)  const
{
    double mean,var;
    if( typeid(T) == typeid(uchar) )
        ippiMean_StdDev_8u_C1R((Ipp8u*)_data[channel], _dataStep, _roi, &mean, &var);
    else
        ippiMean_StdDev_32f_C1R((Ipp32f*)_data[channel], _dataStep, _roi, &mean, &var);

    return (float)(var*var);
}

template<class T> float                Matrix<T>::VarW(const Matrixf &w, T *mu)  const
{
    T mm;
    if( mu == NULL )
        mm = (*this).MeanW(w);
    else
        mm = *mu;
    return ((*this)-mm).Sqr().MeanW(w);
}

template<class T> float                Matrix<T>::MeanW(const vectorf &w)  const
{
    float mean=0.0f;
    assert(w.size() == this->size());
    for( uint k=0; k<w.size(); k++ )
        mean += w[k]*(*this)(k);

    return mean;
}

template<class T> float                Matrix<T>::MeanW(const Matrixf &w)  const
{
    return (float)((*this)&w).Sum();
}

template<class T> double            Matrix<T>::Sum(uint channel)  const
{
    double sum;
    if( typeid(T) == typeid(uchar) )
        ippiSum_8u_C1R((Ipp8u*)_data[channel], _dataStep, _roi, &sum);
    else
        ippiSum_32f_C1R((Ipp32f*)_data[channel], _dataStep, _roi, &sum, ippAlgHintFast);

    return sum;
}


template<class T> Matrixu            Matrix<T>::convert2img(float min, float max)
{
    if( max==min ){
        max = Max();
        min = Min();
    }

    Matrixu res(rows(),cols());
    // scale to 0 to 255
    Matrix<T> tmp;
    tmp = (*this);
    tmp = (tmp-(T)min)*(255/((T)max-(T)min));

    //#pragma omp parallel for
    for( int d=0; d<depth(); d++ )
        for( int row=0; row<rows(); row++ )
            for( int col=0; col<cols(); col++ )
                res(row,col) = (uchar)tmp(row,col);

    return res;

}

template<class T> vector<Matrixu>    Matrix<T>::LoadVideoStream(const char *aviFileName, int start, int end, bool color)
{
    vector<Matrixu> res(end-start+1);

    string inputFile(aviFileName);

    cv::VideoCapture cap(inputFile);

    if(!cap.isOpened())
    {
        abortError( __LINE__, __FILE__, "Fail to load input video video stream" );
    }
    
    cv::Mat frame;
    cv::Mat frameBW; 
    IplImage img, imgBW;
    
    int fr; 

    for (fr=1;  fr < start; fr++)
    {
        cap.grab();
    }
    for (fr=start; cap.grab() && fr<= end; fr++)
    {        
        cap.retrieve(frame); //read the first frame to get the proper Size
        if( color )
        {
            img = frame; //convert to IplImage        
            res[fr-start].Resize(img.height, img.width, img.nChannels);        
            res[fr-start].IplImage2Matrix(&img);
        }
        else{
            cv::cvtColor(frame,frameBW,CV_RGB2GRAY);//convert img to gray scale 
            imgBW = frameBW;
            res[fr-start].Resize(imgBW.height, imgBW.width, imgBW.nChannels);
            res[fr-start].GrayIplImage2Matrix(&imgBW);
        }
    }
    
    if( fr < end )
    {
        abortError( __LINE__, __FILE__, "Fail to load input video video stream" );
    }
    cap.release();
    return res;
}


template<class T> vector<Matrixu>    Matrix<T>::LoadVideo(const char *dirname, const char *basename, const char *ext, int start, int end, int digits, bool color)
{
    vector<Matrixu> res(end-start+1);

    char format[1024];
    char fname[1024];
    for( int k=start; k<=end; k++ ){
        sprintf_s(format,"%s/%s%%0%ii.%s",dirname,basename,digits,ext);
        sprintf_s(fname,format,k);
        res[k-start].LoadImage(fname, color);
    }

    return res;
}
template<class T> vector<Matrixu>    Matrix<T>::LoadVideo(const char *fname, bool color, int maxframes)
{
    CvCapture* capture = cvCaptureFromFile( fname );
    if( capture == NULL )
        abortError(__LINE__,__FILE__,"Error reading in video file");

    vector<Matrixu> vid;
    Matrixu tmp;
    bool c=true;
    while(c && vid.size()<(uint)maxframes){
        c=CaptureImage(capture, tmp, color);
        vid.push_back(tmp);
        fprintf(stderr,"%sLoading video: %d frames",ERASELINE,vid.size());
    }

    fprintf(stderr, "\n");
    return vid;
}
//template<class T> void                Matrix<T>::PlayVideo( vector<Matrixu> &vid, int wait=1 )
template<class T> void                Matrix<T>::PlayVideo( vector<Matrixu> &vid, int wait) //[Zefeng Ni] for gcc compatibility
{
    for( uint k=0; k<vid.size(); k++ ){
        vid[k].display(1);
        cvWaitKey(wait);
    }
}
template<class T> void                Matrix<T>::SaveVideo( vector<Matrixu> &vid, const char* fname, int fps )
{
    CvVideoWriter* w = cvCreateVideoWriter( fname, CV_FOURCC('x','v','i','d'), fps, cvSize(vid[0].cols(),vid[0].rows()), 3 );
    if( w == NULL ) abortError(__LINE__,__FILE__,"video file cannot be opened");
    for( uint k=0; k<vid.size(); k++ ){
        vid[k].createIpl();
        cvWriteFrame( w, vid[k].getIpl() );
    }
    cvReleaseVideoWriter( &w );
}

template<class T> Matrix<T>            Matrix<T>::vecMat2Mat(const vector<Matrix<T> > &x)
{
    Matrix<T> t(x.size(),x[0].size());

    #pragma omp parallel for
    for( int k=0; k<(int)t.rows(); k++ )
        for( int j=0; j<t.cols(); j++ )
            t(k,j) = x[k](j);

    return t;
}
template<class T> vector<Matrix<T> >    Matrix<T>::vecMatTranspose(const vector<Matrix<T> > &x)
{
    vector<Matrix<T> > t(x[0].size());

    #pragma omp parallel for
    for( int k=0; k<(int)t.size(); k++ )
        t[k].Resize(1,x.size());

    #pragma omp parallel for
    for( int k=0; k<(int)t.size(); k++ )
        for( uint j=0; j<x.size(); j++ )
            t[k](j) = x[j](k);


    return t;
}
template<class T> bool                Matrix<T>::DLMWrite( const char *fname, char *delim )
{
    remove( fname );
    ofstream strm; strm.open(fname, std::ios::out);
    if (strm.fail()) { abortError(  __LINE__, __FILE__,"unable to write" ); return false; }

    for( int r=0; r<rows(); r++ ) {
        for( int c=0; c<cols(); c++ ) {
            strm << (float)(*this)(r,c);
            if( c<(cols()-1)) strm << delim;
        }
        strm << endl;
    }

    strm.close();
    return true;
}
template<class T> bool                Matrix<T>::DLMRead( const char *fname, char *delim )
{
    ifstream strm; 

    strm.open(fname, std::ios::in);
    if( strm.fail() ) return false;
    char * tline = new char[40000000];

    // get number of cols
    strm.getline( tline, 40000000 );
    int ncols = ( strtok(tline," ,")==NULL ) ? 0 : 1;
    while( strtok(NULL," ,")!=NULL ) ncols++;

    // read in each row
    strm.seekg( 0, ios::beg );
    Matrix<T> *rowVec; vector<Matrix<T>*> allRowVecs;
    while(!strm.eof() && strm.peek()>=0) {
        strm.getline( tline, 40000000 );
        rowVec = new Matrix<T>(1,ncols);
        (*rowVec)(0,0) = (T) atof( strtok(tline,delim) );
        for( int col=1; col<ncols; col++ )
            (*rowVec)(0,col) = (T) atof( strtok(NULL,delim) );
        allRowVecs.push_back( rowVec );
    }
    int mrows = allRowVecs.size();

    // finally create matrix
    Resize(mrows,ncols);
    for( int row=0; row<mrows; row++ ) {
        rowVec = allRowVecs[row];
        for( int col=0; col<ncols; col++ )
            (*this)(row,col) = (*rowVec)(0,col);
        delete rowVec;
    }
    allRowVecs.clear();
    delete [] tline;
    strm.close();
    return true;
}

#endif

