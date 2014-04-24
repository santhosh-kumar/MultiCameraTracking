#include "CameraNetwork.h"
#include "Config.h"
#include "DefaultParameters.h"

//Configure the system
bool ConfigureSystem( int argc, char* argv[] )
{
    /* Read configuration file*/
    if ( MultipleCameraTracking::Configure( argc, argv ) < 0 )   
    {
        //failure
        return false;
    }

    //Initialize logFile
    std::string logFilePath = std::string(    MultipleCameraTracking::g_configInput.m_outputDirectoryNameCstr ) + "/"
                                                + MultipleCameraTracking::g_configInput.m_dataFilesNameCstr + "/"
                                                + MultipleCameraTracking::g_configInput.m_intializationDirectoryCstr + '/'
                                                + "TR" + int2str( MultipleCameraTracking::g_configInput.m_trialNumber, 3 )+ ".log";

    cout << "Creating the log file: " << logFilePath << endl;

    MultipleCameraTracking::g_logFile.open( logFilePath.c_str( ), ios_base::out );

    ASSERT_TRUE( MultipleCameraTracking::g_logFile.is_open( ) );
    
    MultipleCameraTracking::g_detailedLog = ( MultipleCameraTracking::g_configInput.m_detailedLogging == 1);
    MultipleCameraTracking::g_verboseMode = ( MultipleCameraTracking::g_configInput.m_verboseMode == 1);
    //success

    MultipleCameraTracking::DisplayAndLogParams();
    return true;
}

void    MultiCameraTracking( int argc, char* argv[] )
{
    //configure the system
    ASSERT_TRUE( ConfigureSystem( argc, argv ) );

    //get the number of cameras and objects from the config file
    int numberOfCameras = 0;
    int numberOfObjects = 0;

    vectori cameraIdList;
    vectori objectIdList;

    // find the ids for the camera set
    char* pch= strtok ( MultipleCameraTracking::g_configInput.m_cameraSetCstr, " ," ); //strtok - string to token conversion
    while( pch != NULL )
    {
        numberOfCameras++;
        cameraIdList.push_back( atoi(pch) );
        pch = strtok( NULL, " ," );
    }

    //there should be at least one camera
    if ( numberOfCameras < 1 )
    {
        abortError( __LINE__, __FILE__, "Number of Cameras should be >= 1.");
    }

    // find the ids for the object set 
    char* pch2= strtok ( MultipleCameraTracking::g_configInput.m_objectSetCstr, " ," );
    while( pch2 != NULL )
    {
        numberOfObjects++;
        objectIdList.push_back( atoi(pch2) );
        pch2 = strtok( NULL, " ," );
    }

    if ( numberOfObjects < 1 )
    {
        abortError( __LINE__, __FILE__, "Number of Objects should be >= 1.");
    }


    //Set up the camera network
    MultipleCameraTracking::GeometricFusionType theGeometricFusinType = MultipleCameraTracking::NO_GEOMETRIC_FUSION; 
    switch ( MultipleCameraTracking::g_configInput.m_geometricFusionType )
    {
        case 0:
            theGeometricFusinType = MultipleCameraTracking::NO_GEOMETRIC_FUSION;
            break;
        case 1:
            theGeometricFusinType = MultipleCameraTracking::FUSION_GROUND_PLANE_GMM_KF;
            break;
        default:
            abortError( __LINE__, __FILE__, "Invalid Geometric fusion parameter");
            break;
    }

    MultipleCameraTracking::AppearanceFusionType theAppearanceFusinType = MultipleCameraTracking::NO_APPEARANCE_FUSION;
    switch ( MultipleCameraTracking::g_configInput.m_appearanceFusionType )
    {
    case 0:
        theAppearanceFusinType = MultipleCameraTracking::NO_APPEARANCE_FUSION;
        break;
    case 1:
        theAppearanceFusinType = MultipleCameraTracking::FUSION_CULTURE_COLOR_HISTOGRAM;
        break;
    case 2:
        theAppearanceFusinType = MultipleCameraTracking::FUSION_MULTI_DIMENSIONAL_COLOR_HISTOGRAM;
        break;
    default:
        abortError( __LINE__, __FILE__, "Invalid Appearance fusion parameter");
        break;
    }

    Features::FeatureType        theTrackerFeatureType = Features::HAAR_LIKE;    
    switch( MultipleCameraTracking::g_configInput.m_trackerFeatureType )
    {
    case 1: 
        theTrackerFeatureType = Features::HAAR_LIKE;
        break;
    case 2:
        theTrackerFeatureType = Features::CULTURE_COLOR_HISTOGRAM;
        break;
    case 3:
        theTrackerFeatureType = Features::MULTI_DIMENSIONAL_COLOR_HISTOGRAM;
        break;
    case 4:
        theTrackerFeatureType = Features::HAAR_COLOR_HISTOGRAM;
        break;
    default:
        abortError( __LINE__, __FILE__, "Invalid feature parameter for tracker");
        break;
    }


    Classifier::StrongClassifierType theLocalTrackerStrongClassifierType = Classifier::ONLINE_STOCHASTIC_BOOST_MIL;
    switch( MultipleCameraTracking::g_configInput.m_trackerStrongClassifierType )
    {
    case 1:     // MilBoost 
        theLocalTrackerStrongClassifierType = Classifier::ONLINE_STOCHASTIC_BOOST_MIL;
        break;
    case 2:        // AdaBoost
        theLocalTrackerStrongClassifierType = Classifier::ONLINE_ADABOOST;
        break;
    case 3:        //MilEnsemble
        theLocalTrackerStrongClassifierType = Classifier::ONLINE_ENSEMBLE_BOOST_MIL;
        break;
    case 4:
        theLocalTrackerStrongClassifierType = Classifier::ONLINE_ANY_BOOST_MIL;
        break;
    default:
        abortError(__LINE__,__FILE__,"Error: invalid strong classifier choice for local tracker");
        break;
    }

    Classifier::WeakClassifierType     theLocalTrackerWeakClassifierType = Classifier::STUMP;
    switch( MultipleCameraTracking::g_configInput.m_trackerWeakClassifierType )
    {
    case 1:
        theLocalTrackerWeakClassifierType = Classifier::STUMP;
        break;
    case 2: 
        theLocalTrackerWeakClassifierType = Classifier::WEIGHTED_STUMP;
        break;
    case 3:
        theLocalTrackerWeakClassifierType = Classifier::PERCEPTRON;
        break;
    default:
        abortError(__LINE__,__FILE__,"Error: invalid weak classifier choice for local tracker");
        break;
    }

    MultipleCameraTracking::AppearanceFusionStrongClassifierType theAppearanceFusionStrongClassifierType = MultipleCameraTracking::APP_FUSION_MIL_BOOST;
    switch( MultipleCameraTracking::g_configInput.m_appearanceFusionStrongClassifierType )
    {
    case 1:     // MilBoost 
        theAppearanceFusionStrongClassifierType = MultipleCameraTracking::APP_FUSION_MIL_BOOST;
        break;
    case 2:        // AdaBoost
        theAppearanceFusionStrongClassifierType = MultipleCameraTracking::APP_FUSION_ADA_BOOST;
        break;
    case 3:        //MilEnsemble
        theAppearanceFusionStrongClassifierType = MultipleCameraTracking::APP_FUSION_MIL_ENSEMBLE;
        break;
    default:
        abortError(__LINE__,__FILE__,"Error: invalid strong classifier choice for appearance fusion");
        break;
    }
    
    Classifier::WeakClassifierType    theAppearanceFusionWeakClassifierType = Classifier::STUMP;
    switch( MultipleCameraTracking::g_configInput.m_appearanceFusionWeakClassifierType )
    {
    case 1: 
        theAppearanceFusionWeakClassifierType    = Classifier::STUMP;
        break;
    case 2:
        theAppearanceFusionWeakClassifierType    = Classifier::WEIGHTED_STUMP;
        break;
    case 3: 
        theAppearanceFusionWeakClassifierType    = Classifier::PERCEPTRON;
        break;
    default:
        abortError(__LINE__,__FILE__,"Error: invalid weak classifier choice for appearance fusion");
        break;
    }

    MultipleCameraTracking::TrackerType theLocalTrackerType = MultipleCameraTracking::SIMPLE_TRACKER ;
    switch( MultipleCameraTracking::g_configInput.m_localTrackerType)
    {
    case 0:
        theLocalTrackerType = MultipleCameraTracking::SIMPLE_TRACKER;
        break;
    case 1:
        theLocalTrackerType = MultipleCameraTracking::PARTICLE_FILTER_TRACKER;
        break;
    default:
        abortError(__LINE__,__FILE__,"Error: invalid local object tracker choice");
        break;
    }

    MultipleCameraTracking::CameraTrackingParametersPtr cameraTrackingParametersPtr(
        new MultipleCameraTracking::CameraTrackingParameters(
                                                            &(MultipleCameraTracking::g_configInput),
                                                            theLocalTrackerType,
                                                            theLocalTrackerStrongClassifierType,
                                                            theLocalTrackerWeakClassifierType,
                                                            theTrackerFeatureType,                                                            
                                                            theGeometricFusinType,
                                                            MultipleCameraTracking::g_configInput.m_numOfParticles,
                                                            theAppearanceFusinType,
                                                            theAppearanceFusionStrongClassifierType,
                                                            theAppearanceFusionWeakClassifierType
                                                            )
        );

    //Create a Camera Network and Start with Initialization
    MultipleCameraTracking::CameraNetworkPtr cameraNetworkPtr(
            new  MultipleCameraTracking::CameraNetwork ( cameraIdList, objectIdList,cameraTrackingParametersPtr    )
                                                    );
 

    ASSERT_TRUE( cameraNetworkPtr != NULL );

    //Tracking frame by frame in the Camera network, main for loop
    //frameind=0 is used for initialization
    for ( int frameind = 1; frameind < MultipleCameraTracking::g_configInput.m_numOfFrames; frameind++ )
    {
        cameraNetworkPtr->TrackObjectsOnCurrentFrame( frameind );
        if( MultipleCameraTracking::g_configInput.m_interactiveModeEnabled )
        {
            cout << "Press enter to continue to next frame" << endl;
            if ( getchar() == 'x')
            {
                break;
            }
        }
    } //end main for loop

    cameraNetworkPtr->SaveCameraNetworkState( );

    //close the logs
    MultipleCameraTracking::g_logFile.close();
}

//main function for the executable
int        main( int argc, char* argv[] )
{

    cout << "STARTING MULTICAMERA TRACKING" << endl;

    MultiCameraTracking( argc, argv );

    if ( MultipleCameraTracking::g_configInput.m_waitBeforeFinishTracking )
    {
        cout << "END OF TRACKING (Press enter to exit)" << endl;
        //getchar();
    }
    else
    {
        cout << "END OF TRACKING" << endl;
    }
    
    return 0;
}
