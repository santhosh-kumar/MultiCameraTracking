#ifndef APPERANCE_INFORMATION_FUSER
#define APPERANCE_INFORMATION_FUSER

#include "StrongClassifierBase.h"
#include "CameraNetworkBase.h"
#include "TrackerParameters.h"
#include "CommonMacros.h"
#include "Matrix.h"
#include "Config.h"

#include <boost/shared_ptr.hpp>

namespace MultipleCameraTracking
{
    /****************************************************************
    AppearanceBasedInformationFuser
        AppearanceBasedInformationFuser fuses appearance
        information from multiple cameras. 
    ****************************************************************/
    class AppearanceBasedInformationFuser
    {
    public:
        // Constructor
        AppearanceBasedInformationFuser( const int                                          objectId,
                                        const int                                           cameraId,
                                        const AppearanceFusionType                          appearanceFusionType,
                                        Classifier::StrongClassifierParametersBasePtr       strongClassifierParametersBasePtr );

        // Learn Global Appearance Model
        void LearnGlobalAppearanceModel( CameraNetworkBasePtr cameraNetworkBasePtr  );

        // Compute Likelihoods
        void FuseInformation( Classifier::SampleSet& testSampleSet,
                              vectorf&               likelihoodProbabilities );

    private:

        DISALLOW_IMPLICIT_CONSTRUCTORS( AppearanceBasedInformationFuser );

        const int                                           m_objectId;
        const int                                           m_cameraId;
        const AppearanceFusionType                          m_appearanceFusionType;
        Classifier::StrongClassifierBasePtr                 m_strongClassifierBasePtr;
        Classifier::StrongClassifierParametersBasePtr       m_strongClassifierParametersBasePtr;
    };

    //typedef for boost pointer
    typedef boost::shared_ptr<AppearanceBasedInformationFuser> AppearanceBasedInformationFuserPtr;
}
#endif