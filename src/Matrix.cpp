// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#include "Matrix.h"

template<> float                Matrixu::ii ( const int row, const int col, const int depth ) const
{
    return (float) ((Ipp32f*)_iidata[depth])[row*_iipixStep + col];
}
template<> float                Matrixu::dii_dx(uint x, uint y, uint channel)
{
    if( !isInitII() ) abortError(__LINE__,__FILE__,"cannot take dii/dx, ii is not init");

    if( (x+1) > (uint)cols() || x < 1 ) return 0.0f;
    //0.5*(GET3(ii,y,(x+1),bin,rows,cols) - GET3(ii,y,(x-1),bin,rows,cols));

    return 0.5f * ( ii(y,(x+1),channel) - ii(y,(x-1),channel) );
}

template<> float                Matrixu::dii_dy(uint x, uint y, uint channel)
{
    if( !isInitII() ) abortError(__LINE__,__FILE__,"cannot take dii/dx, ii is not init");

    if( (y+1) > (uint)rows() || y < 1 ) return 0.0f;
    //0.5*(GET3(ii,y,(x+1),bin,rows,cols) - GET3(ii,y,(x-1),bin,rows,cols));

    return 0.5f * ( ii((y+1),x,channel) - ii((y-1),x,channel) );
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

template<> void                    Matrixu::initII()
{
    bool err=false;
    _iidata.resize(_depth);
    for( uint k=0; k<_data.size(); k++ ){
        if( _iidata[k] == NULL )
            _iidata[k] = ippiMalloc_32f_C1(_cols+1,_rows+1,&(_iidataStep));
        if( _iidata[k] == NULL ) abortError(__LINE__,__FILE__,"OUT OF MEMORY!");
        _iipixStep = _iidataStep/sizeof(Ipp32f);
        IppStatus is = ippiIntegral_8u32f_C1R((Ipp8u*)_data[k], _dataStep, (Ipp32f*)_iidata[k], _iidataStep, _roi, 0);
        assert( is == ippStsNoErr );
        err = err || _data[k] == NULL;
    }
    _ii_init = true;
}

template<> float                Matrixu::sumRect(const IppiRect &rect, int channel) const
{
    // debug checks
    assert(_ii_init);
    assert( rect.x >= 0 && rect.y >= 0 && (rect.y+rect.height) <= _rows
        && (rect.x+rect.width) <= _cols && channel < _depth);
    int maxy = (rect.y+rect.height)*_iipixStep;
    int maxx = rect.x+rect.width;
    int y = rect.y*_iipixStep;

    float tl = ((Ipp32f*)_iidata[channel])[y + rect.x];
    float tr = ((Ipp32f*)_iidata[channel])[y + maxx];
    float br = ((Ipp32f*)_iidata[channel])[maxy + maxx];
    float bl = ((Ipp32f*)_iidata[channel])[maxy + rect.x];

    return br + tl - tr - bl;
    //return ii(maxy,maxx,channel) + ii(rect.y,rect.x,channel)
    //    - ii(rect.y,maxx,channel) - ii(maxy,rect.x,channel);
}


template<> void                    Matrixu::IplImage2Matrix(IplImage *img)
{
    //Resize(img->height, img->width, img->nChannels);
    bool origin = img->origin==1;

    if( _depth == 1 )
        for( int row=0; row<_rows; row++ )
            for( int k=0; k<_cols*3; k+=3 )
                if( origin )
                    ((Ipp8u*)_data[0])[(_rows - row - 1)*_dataStep+k/3] = img->imageData[row*img->widthStep+k];
                else
                    ((Ipp8u*)_data[0])[row*_dataStep+k/3] = img->imageData[row*img->widthStep+k];
    else
        #pragma omp parallel for
        for( int row=0; row<_rows; row++ )
            for( int k=0; k<_cols*3; k+=3 ){
                if( origin ){
                    ((Ipp8u*)_data[0])[(_rows - row - 1)*_dataStep+k/3] = img->imageData[row*img->widthStep+k+2];
                    ((Ipp8u*)_data[1])[(_rows - row - 1)*_dataStep+k/3] = img->imageData[row*img->widthStep+k+1];
                    ((Ipp8u*)_data[2])[(_rows - row - 1)*_dataStep+k/3] = img->imageData[row*img->widthStep+k];
                }
                else{
                    ((Ipp8u*)_data[0])[row*_dataStep+k/3] = img->imageData[row*img->widthStep+k+2];
                    ((Ipp8u*)_data[1])[row*_dataStep+k/3] = img->imageData[row*img->widthStep+k+1];
                    ((Ipp8u*)_data[2])[row*_dataStep+k/3] = img->imageData[row*img->widthStep+k];
                }
            }

        if( _keepIpl )
            _iplimg = img;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template<> void                    Matrixu::LoadImage(const char *filename, bool color)
{
    IplImage *img;
    img = cvLoadImage(filename,(int)color);
    if( img == NULL )
        abortError(__LINE__,__FILE__,"Error loading file");
    Resize(img->height, img->width, img->nChannels);

    if( color )
        IplImage2Matrix(img);
    else{
        for( int row=0; row<_rows; row++ )
            for( int k=0; k<_cols; k++ ){
                ((Ipp8u*)_data[0])[row*_dataStep+k] = img->imageData[row*img->widthStep+k];
            }

    }
    cvReleaseImage(&img);
}

template<> void                    Matrixu::createIpl(bool force)
{
    if( _iplimg != NULL && !force) return;
    if( _iplimg != NULL ) cvReleaseImage(&_iplimg);
    CvSize sz; sz.width = _cols; sz.height = _rows;

    int depth = 3;
    _iplimg = cvCreateImageHeader( sz, IPL_DEPTH_8U, depth );

    //_iplimg->align = 32;
    //_iplimg->widthStep = (((_iplimg->width * _iplimg->nChannels *
    //     (_iplimg->depth & ~IPL_DEPTH_SIGN) + 7)/8)+ _iplimg->align - 1) & (~(_iplimg->align - 1));
    //_iplimg->widthStep = _dataStep*depth;
    //_iplimg->imageSize = _iplimg->height*_iplimg->widthStep;
    cvCreateData(_iplimg);

    //cvInitImageHeader( _iplimg, sz, IPL_DEPTH_8U, _depth, IPL_ORIGIN_TL, 16 );
    //IplImage *_iplimg = cvCreateImage( sz, IPL_DEPTH_8U, _depth );

    //assert( _depth==1 ? _iplimg->widthStep == _dataStep : _iplimg->widthStep/3 == _dataStep ); // should always be the same (multiple of 32)
    if( _depth == 1 )
        for( int row=0; row<_rows; row++ )
            for( int k=0; k<_cols*3; k+=3 ){
                _iplimg->imageData[row*_iplimg->widthStep+k+2]=((Ipp8u*)_data[0])[row*_dataStep+k/3];
                _iplimg->imageData[row*_iplimg->widthStep+k+1]=((Ipp8u*)_data[0])[row*_dataStep+k/3];
                _iplimg->imageData[row*_iplimg->widthStep+k  ]=((Ipp8u*)_data[0])[row*_dataStep+k/3];
            }
    else
        //for( int k=0; k<_rows*_dataStep*3; k+=3 ){
        //    _iplimg->imageData[k+2] = ((Ipp8u*)_data[0])[k/3]; // B
        //    _iplimg->imageData[k+1] = ((Ipp8u*)_data[1])[k/3]; // G
        //    _iplimg->imageData[k  ] = ((Ipp8u*)_data[2])[k/3]; // R
        //}
        for( int row=0; row<_rows; row++ )
            for( int k=0; k<_cols*3; k+=3 ){
                _iplimg->imageData[row*_iplimg->widthStep+k+2]=((Ipp8u*)_data[0])[row*_dataStep+k/3];
                _iplimg->imageData[row*_iplimg->widthStep+k+1]=((Ipp8u*)_data[1])[row*_dataStep+k/3];
                _iplimg->imageData[row*_iplimg->widthStep+k  ]=((Ipp8u*)_data[2])[row*_dataStep+k/3];
            }

}

template<> void                    Matrixu::freeIpl()
{
    if( !_keepIpl && _iplimg != NULL) cvReleaseImage(&_iplimg);
}

template<> void                    Matrixu::SaveImage(const char *filename)
{
    createIpl();
    int success = cvSaveImage( filename, _iplimg );
    freeIpl();
}

template<> void                    Matrixu::GrayIplImage2Matrix(IplImage *img)
{
    //Resize(img->height, img->width, img->nChannels);
    bool origin = img->origin==1;

    if( _depth == 1 )
        for( int row=0; row<_rows; row++ )
            for( int k=0; k<_cols; k++ )
                if( origin )
                    ((Ipp8u*)_data[0])[(_rows - row - 1)*_dataStep+k] = img->imageData[row*img->widthStep+k];
                else
                    ((Ipp8u*)_data[0])[row*_dataStep+k] = img->imageData[row*img->widthStep+k];

}

template<> bool                    Matrixu::CaptureImage(CvCapture* capture, Matrixu &res, int color)
{
    IplImage *img;
    if( capture == NULL ) return false;
    img = cvQueryFrame( capture );
    if( img == NULL ) return false;
    res.Resize(img->height, img->width, 1+2*color);
    if( color ){
        //res.Resize(img->height, img->width, 3);
        res.IplImage2Matrix(img);
    }
    else{
        static IplImage *img2;
        if( img2 == NULL ) img2 = cvCreateImage( cvSize(res._cols, res._rows), IPL_DEPTH_8U, 1 );
        cvCvtColor( img, img2, CV_BGR2GRAY );
        img2->origin = 0;
        res.GrayIplImage2Matrix(img2);
    }
    img = NULL;
    return true;
}
template<> bool                    Matrixu::WriteFrame(CvVideoWriter* w, Matrixu &img)
{
    img.createIpl();
    if( w != NULL ){
        IplImage* iplimg = img.getIpl();
        iplimg->origin = 1;
        cvWriteFrame( w, iplimg );
        return true;
    }else
        return false;
}


template<> void                    Matrixu::display(int fignum, float p)
{
    assert(size() > 0);
    createIpl();
    char name[1024];
    sprintf_s(name,"Figure %d",fignum);
    cvNamedWindow( name, 0/*CV_WINDOW_AUTOSIZE*/ );
    cvShowImage( name, _iplimg );
    //cvResizeWindow( name, max((int)(_cols*p),200), (int)max((int)(_rows*p),_rows*(200.0f/_cols)) );
    cvResizeWindow( name, max((int)(_cols*p),(int)200), (int)max((int)(_rows*p),(int)(_rows*(200.0f/_cols))) );//[Zefeng Ni] for gcc compatibility
    //cvWaitKey(0);//DEBUG
    freeIpl();
}

template<> void                    Matrixu::display(const char* figName, float p)
{
    assert( figName!=NULL );
    createIpl();
    cvNamedWindow( figName, 0/*CV_WINDOW_AUTOSIZE*/ );
    cvShowImage( figName, _iplimg );
    cvResizeWindow( figName, max((int)(_cols*p),(int)200), (int)max((int)(_rows*p),(int)(_rows*(200.0f/_cols))) );//[Zefeng Ni] for gcc compatibility
    //cvWaitKey(0);//DEBUG
    freeIpl();
}

template<> void                    Matrixu::PlayCam(int color, const char* fname)
{
    CvCapture* capture = cvCreateCameraCapture( 0 );
    if( capture==NULL ) abortError(__LINE__,__FILE__,"camera not found!");
    CvVideoWriter* w = NULL;


    Matrixu frame;
    frame._keepIpl = false;
    cout << "Press q to quit" << endl;

    StopWatch sw(true);
    double ttime=0.0;
    for( int cnt=0; true; cnt++ )
    {
        CaptureImage(capture,frame,color);

        // initialize video output
        if( fname != NULL && w == NULL)
            w = cvCreateVideoWriter( fname, CV_FOURCC('x','v','i','d'), 10, cvSize(frame.cols(), frame.rows()), 3 );

        // output (both screen and possibly to file)
        frame._keepIpl=true;
        frame.display(1); char q = cvWaitKey(1);
        WriteFrame(w, frame);
        frame._keepIpl=false; frame.freeIpl();

        // check key input
        if( q=='q' ) break;

        // timing
        ttime = sw.Elapsed(true);
        fprintf(stderr,"%s%d Frames/%f sec = %f FPS",ERASELINE,cnt,ttime,((double)cnt)/ttime);
    }

    cvReleaseCapture( &capture );
    if( w != NULL )
        cvReleaseVideoWriter( &w );

}

template<> void                    Matrixu::PlayCamOpenCV()
{
    CvCapture* capture = cvCaptureFromCAM( -1 );
    if( capture == NULL )
        abortError(__LINE__,__FILE__,"Error finding cam");

    cout << "Press q to quit" << endl;
    IplImage *img;
    cvNamedWindow( "Cam", 0/*CV_WINDOW_AUTOSIZE*/ );

    StopWatch sw(true);
    double ttime=0.0;
    for( int cnt=0; true; cnt++ )
    {
        img = cvQueryFrame( capture );
        cvShowImage( "Cam", img );
        ttime = sw.Elapsed(true);
        fprintf(stderr,"%s%d Frames/%f sec = %f FPS",ERASELINE,cnt,ttime,((double)cnt)/ttime);
        char q = cvWaitKey(1);
        if( q=='q' ) break;
    }

    cvReleaseCapture( &capture );
    cout << endl << "Ending PlayCam" << endl;
}

template<> void                    Matrixu::drawRect(IppiRect rect, int lineWidth, int R, int G, int B )
{
    createIpl();
    CvPoint p1, p2;
    p1 = cvPoint(rect.x, rect.y);
    p2 = cvPoint(rect.x+rect.width,rect.y+rect.height);
    cvDrawRect(_iplimg, p1, p2, CV_RGB(R, G, B), lineWidth);
    IplImage2Matrix(_iplimg);
    freeIpl();
}

template<> void                    Matrixu::drawRect(float width, float height, float x,float y, float sc, float th, int lineWidth, int R, int G, int B)
{

    sc = 1.0f/sc;
    th = -th;

    double cth = cos(th)*sc;
    double sth = sin(th)*sc;

    CvPoint p1, p2, p3, p4;

    p1.x = (int)(-cth*width/2 + sth*height/2 + width/2 + x);
    p1.y = (int)(-sth*width/2 - cth*height/2 + height/2 + y);

    p2.x = (int)(cth*width/2 + sth*height/2 + width/2 + x);
    p2.y = (int)(sth*width/2 - cth*height/2 + height/2 + y);

    p3.x = (int)(cth*width/2 - sth*height/2 + width/2 + x);
    p3.y = (int)(sth*width/2 + cth*height/2 + height/2 + y);

    p4.x = (int)(-cth*width/2 - sth*height/2 + width/2 + x);
    p4.y = (int)(-sth*width/2 + cth*height/2 + height/2 + y);

    //cout << p1.x << " " << p1.y << endl;
    //cout << p2.x << " " << p2.y << endl;
    //cout << p3.x << " " << p3.y << endl;
    //cout << p4.x << " " << p4.y << endl;

    createIpl();
    cvLine( _iplimg, p1, p2, CV_RGB( R, G, B), lineWidth, CV_AA );
    cvLine( _iplimg, p2, p3, CV_RGB( R, G, B), lineWidth, CV_AA );
    cvLine( _iplimg, p3, p4, CV_RGB( R, G, B), lineWidth, CV_AA );
    cvLine( _iplimg, p4, p1, CV_RGB( R, G, B), lineWidth, CV_AA );
    IplImage2Matrix(_iplimg);
    freeIpl();
}



template<> void                    Matrixu::drawEllipse(float height, float width, float x,float y, int lineWidth, int R, int G, int B)
{
    createIpl();
    CvPoint p = cvPoint((int)x,(int)y);
    CvSize s = cvSize((int)width, (int)height);
    cvEllipse( _iplimg, p, s, 0, 0, 365, CV_RGB( R, G, B), lineWidth );
    IplImage2Matrix(_iplimg);
    freeIpl();
}
template<> void                    Matrixu::drawEllipse(float height, float width, float x,float y, float startang, float endang, int lineWidth, int R, int G, int B)
{
    createIpl();
    CvPoint p = cvPoint((int)x,(int)y);
    CvSize s = cvSize((int)width, (int)height);
    cvEllipse( _iplimg, p, s, 0, startang, endang, CV_RGB( R, G, B), lineWidth );
    IplImage2Matrix(_iplimg);
    freeIpl();
}
template<> void                    Matrixu::drawText(const char* txt, float x, float y, int R, int G, int B)
{
    createIpl();
    CvFont font;
    cvInitFont( &font, CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 2, 8 );
    CvPoint p = cvPoint((int)x,(int)y);
    cvPutText( _iplimg, txt, p, &font, CV_RGB( R, G, B) );
    IplImage2Matrix(_iplimg);
    freeIpl();
}
template<> void                    Matrixu::warp(Matrixu &res,uint rows, uint cols, float x, float y, float sc, float th, float sr, float phi)
{
    res.Resize(rows,cols,_depth);

    double coeffs[2][3];
    double quad[4][2];

    double cth = cos(th)*sc;
    double sth = sin(th)*sc;

    quad[0][0] = -cth*cols/2 + sth*rows/2 + cols/2;
    quad[0][1] = -sth*cols/2 - cth*rows/2 + rows/2;

    quad[1][0] = cth*cols/2 + sth*rows/2 + cols/2;
    quad[1][1] = sth*cols/2 - cth*rows/2 + rows/2;

    quad[2][0] = cth*cols/2 - sth*rows/2 + cols/2;
    quad[2][1] = sth*cols/2 + cth*rows/2 + rows/2;

    quad[3][0] = -cth*cols/2 - sth*rows/2 + cols/2;
    quad[3][1] = -sth*cols/2 + cth*rows/2 + rows/2;

    //cout << quad[0][0]+x << " " << quad[0][1]+y << endl;
    //cout << quad[1][0]+x << " " << quad[1][1]+y << endl;
    //cout << quad[2][0]+x << " " << quad[2][1]+y << endl;
    //cout << quad[3][0]+x << " " << quad[3][1]+y << endl << endl;

    IppiRect r;
    r.x = (int)x;
    r.y = (int)y;
    r.width = cols;
    r.height = rows;

    IppStatus ii = ippiGetAffineTransform(r, quad, coeffs);

    //#pragma omp parallel for
    for( int k=0; k<_depth; k++ )
        ippiWarpAffine_8u_C1R((Ipp8u*)_data[k],_roi, _dataStep, _roirect, (Ipp8u*)res._data[k],res._dataStep, res._roirect, coeffs, IPPI_INTER_LINEAR);

}

template<> void                    Matrixu::warpAll(uint rows, uint cols, vector<vectorf> params, vector<Matrixu> &res)
{
    res.resize(params[0].size());

    #pragma omp parallel for
    for( int k=0; k<(int)params[0].size(); k++ )
        warp(res[k],rows,cols,params[0][k],params[1][k],params[2][k],params[3][k]);
}
template<> void                    Matrixu::computeGradChannels()
{
    Ipp32s kernel[3] = {-1, 0, 1};

    IppiSize r = _roi;
    r.width-=3;
    r.height-=3;

    ippiFilterRow_8u_C1R((Ipp8u*)_data[0], _dataStep, (Ipp8u*)_data[_depth-2], _dataStep, r, kernel, 3, 2, -1);
    ippiFilterColumn_8u_C1R((Ipp8u*)_data[0], _dataStep, (Ipp8u*)_data[_depth-1], _dataStep, r, kernel, 3, 2, -1);
}

template<> Matrixu                Matrixu::imResize(float r, float c)
{
    float pr, pc; int nr, nc;
    if( c<0 ){
        pr = r; pc = r;
        nr = (int)(r*_rows);
        nc = (int)(r*_cols);
    }else{
        pr = r/_rows; pc = c/_cols;
        nr = (int)r;
        nc = (int)c;
    }

    Matrixu res((int)(nr), (int)(nc), _depth);
    IppStatus ippst;
    int bufSize;
#ifdef IPP_61
    for( int k=0; k<_depth; k++ )
        ippst = ippiResize_8u_C1R((Ipp8u*)_data[k],_roi, _dataStep, _roirect, (Ipp8u*)res._data[k], res._dataStep, res._roi, pc, pr, IPPI_INTER_LINEAR);
#else
    Ipp8u* pBuffer = NULL;
    ippiResizeGetBufSize(_roirect, res._roirect, 1, IPPI_INTER_LINEAR, &bufSize);
    
    for( int k=0; k<_depth; k++ )
        ippiResizeSqrPixel_8u_C1R((Ipp8u*)_data[k], _roi, _dataStep,
                          _roirect, (Ipp8u*)res._data[k], res._dataStep, res._roirect,
                          pc, pr, 0.0, 0.0, IPPI_INTER_LINEAR,
                          pBuffer);
    
    ippsFree(pBuffer);
#endif

    return res;
}

template<> void                    Matrixu::SaveImages(std::vector<Matrixu> imgs, const char *dirname, float resize)
{
    char fname[1024];

    for( uint k=0; k<imgs.size(); k++ ){
        sprintf_s(fname,"%s/img%05d.png",dirname,k);
        if( resize == 1.0f )
            imgs[k].SaveImage(fname);
        else{
            imgs[k].imResize(resize).SaveImage(fname);
        }
    }
}

template<> void                    Matrixu::conv2RGB(Matrixu &res)
{
    res.Resize(_rows,_cols,3);
    for( int k=0; k<_rows*_dataStep; k++ )
    {
        ((Ipp8u*)res._data[0])[k] = ((Ipp8u*)_data[0])[k];
        ((Ipp8u*)res._data[1])[k] = ((Ipp8u*)_data[0])[k];
        ((Ipp8u*)res._data[2])[k] = ((Ipp8u*)_data[0])[k];
    }
}
template<> void                    Matrixu::conv2HSV(Matrixu &res)
{
    res.Resize(_rows,_cols,3);
    if( res.isInitII() )
    {
        res.FreeII();
    }
    
    CvSize sz; sz.width = _cols; sz.height = _rows;

    IplImage* pImgHSV = cvCreateImage( sz, IPL_DEPTH_8U, 3);
    createIpl();
    cvCvtColor(_iplimg, pImgHSV, CV_BGR2HSV);
    
    bool origin = pImgHSV->origin==1;

    #pragma omp parallel for
    for( int row=0; row<_rows; row++ )
        for( int k=0; k<_cols*3; k+=3 ){
            if( origin ){
                ((Ipp8u*)_data[0])[(_rows - row - 1)*_dataStep+k/3] = pImgHSV->imageData[row*pImgHSV->widthStep+k];
                ((Ipp8u*)_data[1])[(_rows - row - 1)*_dataStep+k/3] = pImgHSV->imageData[row*pImgHSV->widthStep+k+1];
                ((Ipp8u*)_data[2])[(_rows - row - 1)*_dataStep+k/3] = pImgHSV->imageData[row*pImgHSV->widthStep+k+2];
            }
            else{
                ((Ipp8u*)_data[0])[row*_dataStep+k/3] = pImgHSV->imageData[row*pImgHSV->widthStep+k];
                ((Ipp8u*)_data[1])[row*_dataStep+k/3] = pImgHSV->imageData[row*pImgHSV->widthStep+k+1];
                ((Ipp8u*)_data[2])[row*_dataStep+k/3] = pImgHSV->imageData[row*pImgHSV->widthStep+k+2];
            }
        }
    cvReleaseImage(& pImgHSV);
    freeIpl();
}

template<> void                    Matrixu::conv2BW(Matrixu &res)
{
    res.Resize(_rows,_cols,1);

    if( res.isInitII() )
    {
        res.FreeII();
    }
    /*
    double t;
    for( int k=0; k<(int)size(); k++ )
    {
        t = (double) ((Ipp8u*)_data[0])[k];
        t+= (double) ((Ipp8u*)_data[1])[k];
        t+= (double) ((Ipp8u*)_data[2])[k];
        ((Ipp8u*)res._data[0])[k] = (Ipp8u) (t/3.0);
    }

    if( res._keepIpl ) res.freeIpl();*/

    CvSize sz; sz.width = _cols; sz.height = _rows;

    IplImage* pImgBW = cvCreateImage( sz, IPL_DEPTH_8U, 1);
    createIpl();
    cvCvtColor(_iplimg,pImgBW,CV_BGR2GRAY);
    res.GrayIplImage2Matrix(pImgBW);
    cvReleaseImage(& pImgBW);
    freeIpl();
}

template<> float                Matrixf::Dot(const Matrixf &x)
{
    assert( this->size() == x.size() );
    float sum = 0.0f;
    #pragma omp parallel for reduction(+: sum)
    for( int i=0; i<(int)size(); i++ )
        sum += (*this)(i)*x(i);

    return sum;
}
