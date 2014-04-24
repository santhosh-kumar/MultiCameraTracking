#ifndef CAMERA_OBJECT_HEADER
#define CAMERA_OBJECT_HEADER

#include "Matrix.h"
#include "Feature.h"
#include "Tracker.h"
#include "SimpleTracker.h"
#include "ParticleFilterTracker.h"
#include "Public.h"
#include "CameraNetworkBase.h"
#include "AppearanceBasedInformationFuser.h"
#include "DefaultParameters.h"
#include "Config.h"

namespace MultipleCameraTracking
{
    /****************************************************************
    Object
        This class holds all the information about an 
        object(target of interest) inside a particular camera,
        i.e. an object instance belong to a camera
        instance. 
    ****************************************************************/
    class Object
    {
    public:
        //Constructor
        Object(    const    int                                            objectId,
                const    int                                            cameraId, 
                const    bool                                        isColorEnabled,    
                CameraTrackingParametersPtr                            cameraTrackingParametersPtr,
                CvMat*                                                pHomographyMatrix );

        int    GetObjectID( ) const { return m_objectID; };
        
        //Initializes the object parameters with initial state and file name for saving its trajectory
        void        InitializeObjectParameters( const vectorf& initialState,  const string& trajSaveStrBase );
        
        /*****    Basic Object Tracker    *****/
        //    Initialize the tracker for the object
        bool        InitializeObjectTracker( Matrixu* pFrameImageColor, 
                                             Matrixu* pFrameImageGray,
                                             int frameInd,     
                                             uint videoLength,
                                             Matrixu* pFrameDisplay            = NULL, 
                                             Matrixu* pFrameDisplayTraining    = NULL,
                                             Matrixu* pFrameImageHSV        = NULL,
                                             Matrixf* pGroundTruthMatrix    = NULL );

        //    Track the object in a given frame
        void        TrackObjectFrame(    int frameind, 
                                        Matrixu* pFrameImageColor, 
                                        Matrixu* pFrameImageGray, 
                                        Matrixu* pFrameDisplay=NULL, 
                                        Matrixu* pFrameDisplayTraining=NULL, 
                                        Matrixu* pFrameImageHSV=NULL );

        //    Notify the tracker to keep the particle filter tracker state as the final state for the frame, 
        void        StoreParticleFilterTrackerState( int frameInd,  Matrixu* pFrameDisplay = NULL ); 

        //    Update the particle filter tracker appearance model
        void        UpdateParticleFilterTrackerAppearanceModel( Matrixu* pFrameImageColor, 
                                                                Matrixu* pFrameImageGray, 
                                                                Matrixu* pFrameDisplayTraining    = NULL, 
                                                                Matrixu* pFrameImageHSV            = NULL );

        //    Save the state on the object's trajectory file
        void        SaveObjectStatesAllFrames( );

        /*****    For Ground Plane Fusion        *****/
        //    Get ground plane position particles (still on the image plane)
        CvMat*        GetParticlesFootPositionOnImagePlaneForGeometricFusion( );
        CvMat*        GetAverageParticleFootPositionOnImagePlaneForGeometricFusion( );

        /*****        For Color Appearance Fusion        *****/
        void        GenerateTrainingSampleSetsForAppearanceFusion(    Classifier::SampleSet&    positiveSampleSet,
                                                                    Classifier::SampleSet&    negativeSampleSet,
                                                                    Matrixu*                pFrameImageColor,
                                                                    Matrixu*                pFrameImageGray, 
                                                                    Matrixu*                pFrameImageHSV );

        void        LearnGlobalAppearanceModel( CameraNetworkBasePtr cameraNetworkBasePtr );

        /*****        For Occlusion Handling        *****/
        void        AutoInitialize( Matrixf AppearanceModel )    { };
        void        SuspendTracking( )    {};
        void        ResumeTracking( Matrixf AppearanceModel)    { };

        void        UpdateParticleWeightUsingMultiCameraAppearanceModel( Matrixu* pFrameImageColor,
                                                                        Matrixu* pFrameImageGray,
                                                                        Matrixu* pFrameImageHSV );

        void        UpdateParticleWeightsWithGroundPDF( CvMat*        pMeanMatrix,
                                                        CvMat*        pCovarianceMatrix,
                                                        Matrixu*    pColorImageMatrix,
                                                        Matrixu*    pGrayImageMatrix,
                                                        Matrixu*    pHsvImageMatrix );

        void        DrawObjectFootPoint(Matrixu* pFrameDisplay);
    private:
        DISALLOW_IMPLICIT_CONSTRUCTORS( Object );

        Features::FeatureParametersPtr GenerateDefaultTrackerFeatureParameters( );
        Features::FeatureParametersPtr GenerateDefaultAppearanceFusionFeatureParameters( );
        Classifier::StrongClassifierParametersBasePtr GenerateDefaultAppearanceFusionClassifierParameters( );

        CameraTrackingParametersPtr                            m_cameraTrackingParametersPtr;
        CvMat*                                                m_pHomographyMatrix;

        Classifier::StrongClassifierParametersBasePtr        m_classifierParamPtr;            //Strong Classifier Parameters
        TrackerParametersPtr                                m_trackerParametersPtr;            //Simple Tracker Parameters
        TrackerPtr                                            m_trackerPtr;                    //Object for Tracking Module
        AppearanceBasedInformationFuserPtr                    m_appearanceFuserPtr;
        const int                                            m_objectID;                        //Object ID
        const int                                            m_cameraID;                        //which camera does this object belongs to
        
        const bool                                            m_colorImage;                    //whether input raw video is color 
        int                                                    m_indPreviousTrackedFrame;        //the frame index in the most recent call to TrackObjectFrame        
        
    };

    typedef boost::shared_ptr<Object> ObjectPtr;    
}
#endif