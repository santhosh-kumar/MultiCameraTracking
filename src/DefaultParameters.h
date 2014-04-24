#ifndef H_DEFAULT_PARAMS
#define H_DEFAULT_PARAMS

namespace MultipleCameraTracking
{
    //default classifier parameters
//    #define DEFAULT_TRACKER_INIT_NEG_NUM_TRAIN                                65        //default value for SimpleTrackerParameters.m_init_negNumTrain 
//    #define DEFAULT_TRACKER_INIT_POS_RADIUS_TRAIN                            3.0F    //default value for SimpleTrackerParameters.m_init_posTrainRadius 
    #define DEFAULT_TRACKER_INIT_WITH_FACE                                    false    //default value for SimpleTrackerParameters.m_initWithFace
    #define DEFAULT_TRACKER_DEBUGV                                            false    //default value for SimpleTrackerParameters.m_debugv
                                                                
    #define DEFAULT_PFTRACKER_NOT_USE_SIGMOIDAL                                true    //default TrackerParameters.m_shouldNotUseSigmoid for PFTracker, i.e. use sigmoid function to calculate probability

    #define DEFAULT_HAAR_MINIMUM_NUMBER_OF_RECTANGLES                        2        
    #define DEFAULT_HAAR_MAXIMUM_NUMBER_OF_RECTANGLES                        6
    #define DEFAULT_HAAR_NUMBER_OF_CHANNELS                                    -1

    #define DEFAULT_GAUSSIAN_WEAK_CLASSIFIER_LEARNING_RATE                    0.85f
    #define DEFAULT_STRONG_CLASSIFIER_STORE_FEATURE_HISTORY                    true    
}
#endif