#include "StrongClassifierBase.h"
#include "Tracker.h"
#include "Public.h"
#include "Sample.h"
#include "CommonMacros.h"

#define HAAR_CASCADE_FILE_NAME "haarcascade_frontalface_alt_tree.xml"

namespace MultipleCameraTracking
{
    CvHaarClassifierCascade* Tracker::s_faceCascade = NULL;

    /****************************************************************
    Tracker::InitializeWithFace
        Initialize the tracker with opencv's face detector.
    Exceptions:
        None
    ****************************************************************/
    bool    Tracker::InitializeWithFace( TrackerParameters* params, Matrixu& frame )
    {
        const int minsz = 20;

        //Get the name of the haar-cascade .xml file
        const char* cascade_name = HAAR_CASCADE_FILE_NAME;
        ASSERT_TRUE( cascade_name != NULL );

        //Load the cascade
        if ( Tracker::s_faceCascade == NULL )
        {
            Tracker::s_faceCascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 );
        }

        frame.createIpl();
        IplImage* img    = frame.getIpl();
        ASSERT_TRUE( img != NULL );

        //convert to grayscale
        IplImage* gray    = cvCreateImage( cvSize(img->width, img->height), IPL_DEPTH_8U, 1 );
        ASSERT_TRUE( gray != NULL );
        cvCvtColor(img, gray, CV_BGR2GRAY );

        frame.freeIpl();

        //histogram equalization
        cvEqualizeHist( gray, gray );

        //memory storage
        CvMemStorage* storage = cvCreateMemStorage(0);
        cvClearMemStorage(storage);

        //call opencv's haar feature based detector
        CvSeq* faces = cvHaarDetectObjects( gray,
                                            Tracker::s_faceCascade,
                                            storage,
                                            1.05,
                                            3,
                                            CV_HAAR_DO_CANNY_PRUNING, 
                                            cvSize( minsz, minsz ) );

        int index = faces->total-1;
        CvRect* r = (CvRect*)cvGetSeqElem( faces, index );
        if ( r == NULL )
        {
            return false;
        }

        while ( r && (r->width<minsz || r->height < minsz || (r->y+r->height+10)>frame.rows() || (r->x+r->width)>frame.cols() ||
            r->y<0 || r->x<0) )
        {
            r = (CvRect*)cvGetSeqElem( faces, --index );
        }

        //set the params
        params->m_initState.resize(4);
        params->m_initState[0]    = (float)r->x;
        params->m_initState[1]    = (float)r->y;
        params->m_initState[2]    = (float)r->width;
        params->m_initState[3]    = (float)r->height+10;

        return true;
    }

    /****************************************************************
    Tracker::ReplayTracker
        Replays the tracker based on the saved output.
    Exceptions:
        None
    ****************************************************************/
    void    Tracker::ReplayTracker( vector<Matrixu>& vid, string statesfile, string outputvid, uint R, uint G, uint B)
    {
        Matrixf states;
        states.DLMRead(statesfile.c_str());

        Matrixu colorframe;
        // save videoList file
        CvVideoWriter* w = NULL;
        if ( ! outputvid.empty() )
        {
            w = cvCreateVideoWriter( outputvid.c_str(), CV_FOURCC('x','v','i','d'), 15, cvSize(vid[0].cols(), vid[0].rows()), 3 );

            if( w==NULL ) 
            {
                abortError(__LINE__,__FILE__,"Error opening videoList file for output");
            }
        }

        for( uint k=0; k < vid.size(); k++ )
        {
            vid[k].conv2RGB(colorframe);
            colorframe.drawRect(states(k,2),states(k,3),states(k,0),states(k,1),1,0,2,R,G,B);
            colorframe.drawText(("#"+int2str(k,3)).c_str(),1,25,255,255,0);
            colorframe._keepIpl=true;
            colorframe.display(0,2);
            cvWaitKey(1);
            if( w != NULL )
            {
                cvWriteFrame( w, colorframe.getIpl() );
            }
            colorframe._keepIpl=false; colorframe.freeIpl();
        }

        // clean up
        if( w != NULL )
        {
            cvReleaseVideoWriter( &w );
        }
    }

    /****************************************************************
    Tracker::ReplaysTracker
        Replays the tracker based on the saved output.
    Exceptions:
        None
    ****************************************************************/
    void    Tracker::ReplayTrackers(vector<Matrixu> &vid, vector<string> statesfile, string outputvid, Matrixu colors)
    {
        Matrixu states;

        vector<Matrixu> resvid(vid.size());
        Matrixu colorframe;

        // save videoList file
        CvVideoWriter* w = NULL;
        if ( ! outputvid.empty() )
        {
            w = cvCreateVideoWriter( outputvid.c_str(), CV_FOURCC('x','v','i','d'), 15, cvSize(vid[0].cols(), vid[0].rows()), 3 );

            if ( w==NULL ) 
            {
                abortError(__LINE__,__FILE__,"Error opening videoList file for output");
            }
        }

        for ( uint k=0; k<vid.size(); k++ )
        {
            vid[k].conv2RGB(resvid[k]);
            resvid[k].drawText(("#"+int2str(k,3)).c_str(),1,25,255,255,0);
        }

        for ( uint j=0; j < statesfile.size(); j++ )
        {
            states.DLMRead(statesfile[j].c_str());
            for( uint k=0; k<vid.size(); k++ )
            {
                resvid[k].drawRect(states(k,3),states(k,2),states(k,0),states(k,1),1,0,3,colors(j,0),colors(j,1),colors(j,2));
            }
        }

        for ( uint k=0; k<vid.size(); k++ )
        {
            resvid[k]._keepIpl=true;
            resvid[k].display(0,2);
            cvWaitKey(1);
            if( w!=NULL && k<vid.size()-1)
            {            
                Matrixu::WriteFrame(w, resvid[k]);
            }
            resvid[k]._keepIpl=false; resvid[k].freeIpl();
        }

        // clean up
        if( w != NULL )
        {
            cvReleaseVideoWriter( &w );
        }
    }
}
