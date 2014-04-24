#ifndef CAMERA_HEADER
#define CAMERA_HEADER

#include "Matrix.h"
#include "Feature.h"
#include "Tracker.h"
#include "Public.h"
#include "Object.h"
#include "CameraNetworkBase.h"
#include "DefaultParameters.h"
#include "Config.h"

#include <boost/shared_ptr.hpp>

using namespace boost;

namespace MultipleCameraTracking
{
    enum ObjectTrackingStatus 
    {
        OBJECT_TRACKING_UN_INITIALIZED    = 0, 
        OBJECT_TRACKING_IN_PROGRESS        = 2,
        OBJECT_TRACKING_SUSPENDED        = 3
    };

    /****************************************************************
    Camera
        This class is the heart of the tracking system. 
        It holds all the information about cameras, various parameters.
        Objects are stored in a list. Objects across camera views are
        associated by their objectId.
    ****************************************************************/
    class Camera
    {
    public:
        //Constructor
        Camera( const int                        cameraId,
                const vectori&                    objectIdList,
                CameraTrackingParametersPtr        cameraTrackingParametersPtr );

        ~Camera( ){ }

        int            GetCameraID( ) const { return m_cameraID; };
        ObjectPtr    GetObjectPtr( const int objectId ) const;

        Matrixu*    GetColorImageMatrix( ){ return m_pCurrentFrameImageMatrixColor; }
        Matrixu*    GetGrayImageMatrix( ){ return m_pCurrentFrameImageMatrixGray; }
        Matrixu*    GetHSVImageMatrix( ){ return &m_HSVImageMatrix; }

        //Learn global appearance model using the appearance fuser
        void LearnGlobalAppearanceModel( CameraNetworkBasePtr cameraNetworkBasePtr );

        //Get reprojected particles from the ground plane
        CvMat* GetReprojectedParticlesForGeometricFusion( const int objectInd );
        CvMat* GetAverageImageParticleForGeometricFusion( const int objectInd );
        
        //Initializes the parameters for the camera
        void InitializeCameraParameters( const vectori&    objectIdList );

        //initialize all object trackers inside the camera
        void InitializeCameraTrackers( );

        //Track all objects inside a Camera Frame
        void TrackCameraFrame( int frameInd );

        //Store the particle filter tracker state
        void StoreAllParticleFilterTrackerState( int frameInd ); 

        //Update the particle filter tracker appearance
        void LearnLocalAppearanceModel( int frameInd);
            
        //Save the state on the output trajectory files
        void SaveStatesAllFrames( );
    
        //Update particle weights based on the ground plane kalman filter's pdf
        void UpdateParticleWeightsWithGroundPDF( CvMat*  pMeanMatrix, CvMat* pCovarianceMatrix, int objectId );

        //Get homography matrix
        CvMat* GetHomographyMatrix( ){ return m_pHomographyMatrix; }

        //draw object's predicted foot points on the image
        void    DrawAllObjectFootPoints( );

    private:

        DISALLOW_IMPLICIT_CONSTRUCTORS( Camera );

        //update display for tracked frame
        void    DisplayAndSaveTrackedFrame( ); 

        //update display for training samples
        void    DisplayAndSaveTrainingSamples();

        //auto-initialize object, occlusion detection, handling, and resumption    
        void    MultipleObjectInteraction( ) { /*implement this*/ };     

        // create temporary frame image for displaying, tracking, initialization integral image etc. when necessary
        void    PrepareCurrentFrameForTracking( int frameInd );

        //camera property
        CvMat*                        m_pHomographyMatrix;        //Stores the Homography
        int                            m_cameraID;                    //Camera ID
        vector<ObjectPtr>            m_objectPtrList;            //List of objects being tracked
        vector<ObjectTrackingStatus> m_objectStatusList;        //List of object status
        

        //input data parameters
        vector<Matrixu>                m_videoMatrix;                //Input video Sequence for tracking
        bool                        m_sourceIsColorImage;        //read input video as color 
        bool                        m_readImages;                //read images or video stream
        Matrixf                        m_frameMatrix;                //Stores starting and ending frame numbers
        Matrixf                        m_initialState;                //Initial states for all objects
    
        Matrixu*                    m_pCurrentFrameImageMatrixGray;    //a pointer to the currently tracked frame (Gray image)
        Matrixu*                    m_pCurrentFrameImageMatrixHSV;    //a pointer to the currently tracked frame (HSV image)
        Matrixu*                    m_pCurrentFrameImageMatrixColor;//a pointer to the currently tracked frame (Color image)
        Matrixu                        m_grayScaleImageMatrix;            //a temporary black and white image for Haar feature calculation if raw video is color
        Matrixu                        m_HSVImageMatrix;                //a temporal HSV image matrix for HSV related feature calculation if hsv is needed.

        //tracking parameters                
        //the following parameters "typically" passed from the owner CameraNetwork class
        CameraTrackingParametersPtr    m_cameraTrackingParametersPtr;        
        
        //output results (including displaying)
        const char*                    m_outputDirectoryCstr;  //Directory for output result
        bool                        m_displayTrackedVideo;    // display video with tracker state (colored box)
        bool                        m_saveTrackedVideo;        // save video with tracking box
        CvVideoWriter*                m_pVideoWriter;            // if m_saveTrackedVideo == true
        Matrixu                        m_frameDisplay;            //for displaying and saving tracked frame
        Matrixu*                    m_pFrameDisplay;

        bool                        m_displayTrainingSamples;    // display video with training examples
        bool                        m_saveVideoTrainingExamples;//save the training examples    
        CvVideoWriter*                m_pVideoWriterTraining;        //for saving training example on frame
        Matrixu                        m_frameDisplayTraining;        //for displaying and saving training examples on frame        
        Matrixu*                    m_pFrameDisplayTraining;

        vector<Matrixf>                m_groundTruthMatrixList;    //hold the ground truth for each object in the camera
    };

    //typedef for camera pointer
    typedef boost::shared_ptr<Camera>    CameraPtr;
    typedef std::vector<CameraPtr>        CameraPtrList;
}
#endif