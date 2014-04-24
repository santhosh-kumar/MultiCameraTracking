#include "TrackerParameters.h"

namespace MultipleCameraTracking
{
    /****************************************************************
    CameraTrackingParameters
        Stores the tracking parameters for each camera
    ****************************************************************/
    CameraTrackingParameters::CameraTrackingParameters( InputParameters *                               pInputConfigeration,
                                                        const TrackerType                               trackerType,
                                                        const Classifier::StrongClassifierType          localTrackerStrongClassifierType,
                                                        const Classifier::WeakClassifierType            localTrackerWeakClassifierType,
                                                        const Features::FeatureType                     trackerFeatureType,
                                                        const GeometricFusionType                       geometricFusionType,
                                                        const int                                       numberOfParticles,
                                                        const AppearanceFusionType                      appearanceFusionType,
                                                        const AppearanceFusionStrongClassifierType      appearanceFusionStrongClassifierType,
                                                        const Classifier::WeakClassifierType            appearanceFusionWeakClassifierType )
        : m_trialNumber( pInputConfigeration->m_trialNumber ),
        m_localObjectTrackerType( trackerType),
        m_localTrackerStrongClassifierType( localTrackerStrongClassifierType ), 
        m_localTrackerWeakClassifierType( localTrackerWeakClassifierType ),
        m_trackerFeatureType( trackerFeatureType ),
        m_geometricFusionType( geometricFusionType ),
        m_saveGroundParticlesImage( pInputConfigeration->m_saveGroundParticlesImage == 1 ),
        m_displayGroundParticlesImage( pInputConfigeration->m_displayGroundParticlesImage == 1 ),
        m_saveGroundPlaneKFImage( pInputConfigeration->m_saveGroundPlaneKFImage == 1 ),
        m_displayGroundPlaneKFImage( pInputConfigeration->m_displayGroundPlaneKFImage == 1 ),
        m_numberOfParticles( numberOfParticles ),
        m_appearanceFusionType( appearanceFusionType ),
        m_appearanceFusionStrongClassifierType( appearanceFusionStrongClassifierType ),
        m_appearanceFusionWeakClassifierType( appearanceFusionWeakClassifierType ),
        m_isCrossCameraAutoInitializationEnabled( pInputConfigeration->m_enableCrossCameraAutoInitialization == 1 ),
        m_isCrossCameraOcclusionHandlingEnabled( pInputConfigeration->m_enableCrossCameraOcclusionHandling == 1 ),
        m_displayGMMCenters ( pInputConfigeration->m_displayGMMCenters == 1),
        m_outputDirectoryString( pInputConfigeration->m_outputDirectoryNameCstr ),
        m_inputDirectoryString( pInputConfigeration->m_inputDirectoryNameCstr ),
        m_dataNameString( pInputConfigeration->m_dataFilesNameCstr ),
        m_nameInitilizationString( pInputConfigeration->m_intializationDirectoryCstr ),
        m_calculateTrackingError ( pInputConfigeration->m_calculateTrackingError == 1 ),
        m_numberOfBinsForColorHistogram ( static_cast<uint>(pInputConfigeration->m_numofBinsColor) ),
        m_useHSVColorSpaceForColorHistogram ( pInputConfigeration->m_useHSVColor == 1 ),
        m_HSVRequired ( pInputConfigeration->m_useHSVColor == 1 || appearanceFusionType ==  FUSION_CULTURE_COLOR_HISTOGRAM ),
        m_appearanceFusionNumberOfPositiveExamples( pInputConfigeration->m_AFNumberOfPositiveExamples ),
        m_appearanceFusionNumberOfNegativeExamples( pInputConfigeration->m_AFNumberOfNegativeExamples ),
        m_appearanceFusionRefreshRate( pInputConfigeration->m_AFRefreshRate )
    {
    };

    /****************************************************************
    TrackerParams
            Stores the tracker parameters for the tracker (associated with each object)
    ****************************************************************/
    TrackerParameters::TrackerParameters()
    {
        m_outputBoxColor.resize(3);
        //default box color similar to Magenta.
        m_outputBoxColor[0]        = 0;
        m_outputBoxColor[1]        = 255;
        m_outputBoxColor[2]        = 0;
        m_outputBoxLineWidth    = 2;
        m_numberOfNegativeTrainingSamples            = 15;
        m_posRadiusTrain        = 1;
        m_maximumNumberOfPositiveTrainingSamples        = 100000;
        m_init_negNumTrain        = 1000;
        m_init_posTrainRadius    = 3;
        m_initState.resize(4);
        m_debugv                = false;
        m_shouldNotUseSigmoid                = true;
        m_initializeWithFaceDetection            = true;
        m_isColor                = true;
        m_trajSave                = "";
    }

    /****************************************************************
    SimpleTrackerParams
    ****************************************************************/
    SimpleTrackerParameters::SimpleTrackerParameters()
    {
        m_searchWindSize        = 30;
        m_negSampleStrategy     = 1;
    }
}