#ifndef CAMERA_NETWORK_HEADER
#define CAMERA_NETWORK_HEADER

#include "CameraNetworkBase.h"
#include "Camera.h"
#include "GeometryBasedInformationFuser.h"
#include <boost/shared_ptr.hpp>

using namespace boost;

namespace MultipleCameraTracking
{
    /****************************************************************
    CameraNetwork
        Camera Network encapsulates all the functions of a network.
        It holds a list of cameras, all the messages to each of the
        camera is passed through this class.
    ****************************************************************/
    class CameraNetwork : public CameraNetworkBase
    {
    public :

        //Constructor
        CameraNetwork(    const vectori                    cameraIdList, 
                        const vectori                    objectIdList,                        
                        CameraTrackingParametersPtr        cameraTrackingParametersPtr );

        //Destructor
        ~CameraNetwork( );

        //Generates training example for appearance fusion
        virtual void GenerateTrainingSampleSetsForAppearanceFusion( const int                objectId, 
                                                                    Classifier::SampleSet&    positiveSampleSet,
                                                                    Classifier::SampleSet&    negativeSampleSet );

        //Track Objects on a given frame across different views
        virtual void TrackObjectsOnCurrentFrame( const int frameInd );

        //Save Camera Network state in a file(if enabled)
        virtual void SaveCameraNetworkState( );
    
    private :

        DISALLOW_IMPLICIT_CONSTRUCTORS( CameraNetwork );

        void                FeedbackInformationFromGeometricFusion( int frameIndex );
        void                FuseGeometrically( int frameIndex );
        vector<CvMat*>        GetHomographyMatrixList( );

        void                InitializeCameraNetwork( );
        void                InitializeGeometricFuser( );

        
        const int                                        m_numberOfCameras;
        const int                                        m_numberOfObjects;
        const vectori                                    m_cameraIdList;
        const vectori                                    m_objectIdList;
        CameraPtrList                                    m_cameraPtrList;
        CameraTrackingParametersPtr                        m_cameraTrackingParametersPtr;        
        vector<CvMat*>                                    m_groundPlaneParticlesPtrList;                            
        GeometryBasedInformationFuserPtrList            m_geometricInformationFuserPtrList;
        vector<CvVideoWriter*>                            m_videoWriterListGroundParticles;            
        vector<CvVideoWriter*>                            m_videoWriterListKFDistribution;            
        vector<CvVideoWriter*>                            m_videoWriterListGroundParticlesAfterFusion;
    };

    //typedef for CameraNetwork pointer
    typedef boost::shared_ptr<CameraNetwork> CameraNetworkPtr;
}
#endif