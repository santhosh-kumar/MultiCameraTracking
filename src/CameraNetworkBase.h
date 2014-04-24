#ifndef CAMERA_NETWORK_BASE
#define CAMERA_NETWORK_BASE

#include "CommonMacros.h"
#include "SampleSet.h"

#include <boost/shared_ptr.hpp>

namespace MultipleCameraTracking
{
    /****************************************************************
    CameraNetworkBase
        A Base class for CameraNetwork. This is an abstract interface.
        Mainly for message passing across different cameras.
    ****************************************************************/
    class CameraNetworkBase
    {
    public:
        virtual void GenerateTrainingSampleSetsForAppearanceFusion( const int objectId, Classifier::SampleSet& positiveSampleSet, Classifier::SampleSet& negativeSampleSet ) = 0;

        //Save Camera Network state in a file(if enabled)
        virtual void SaveCameraNetworkState( ) = 0;

        //Track Objects on a given frame across different views
        virtual void TrackObjectsOnCurrentFrame( const int frameIndex ) = 0;
    };

    //typedef for CameraNetworkBase pointer - do not use shared_ptr, it will delete the local ptr when it goes out of scope
    typedef CameraNetworkBase* CameraNetworkBasePtr;
};
#endif
