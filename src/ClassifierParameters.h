#ifndef CLASSIFIER_PARAMETERS_H
#define CLASSIFIER_PARAMETERS_H

#include "Public.h"
#include "Feature.h"
#include "WeakClassifierBase.h"
#include "DefaultParameters.h"

namespace Classifier
{
    enum StrongClassifierType
    {    
        ONLINE_ADABOOST              = 0,        // [0] Online AdaBoost (Oza/Grabner)
        //ONLINE_STOCHASTIC_BOOST_LR = 1,        // [1] Online StochBoost_LR
        ONLINE_STOCHASTIC_BOOST_MIL  = 2,        // [2] Online StochBoost_MIL
        ONLINE_ENSEMBLE_BOOST_MIL    = 3,        // [3] Online Ensemble_MIL
        ONLINE_ANY_BOOST_MIL         = 4,        // [4] ONLINE STOCHBOOST_MIL With anyboost
    };    

    //Forward Declaration
    class StrongClassifierParametersBase;
    class AdaBoostClassifierParameters;
    class MILBoostClassifierParameters;
    class MILEnsembleClassifierParameters;
    class MILAnyBoostClassifierParameters;

    //declarations of shared ptr
    typedef boost::shared_ptr<StrongClassifierParametersBase>       StrongClassifierParametersBasePtr;
    typedef boost::shared_ptr<AdaBoostClassifierParameters>         AdaBoostClassifierParametersPtr;
    typedef boost::shared_ptr<MILBoostClassifierParameters>         MILBoostClassifierParametersPtr;
    typedef boost::shared_ptr<MILEnsembleClassifierParameters>      MILEnsembleClassifierParametersPtr;
    typedef boost::shared_ptr<MILAnyBoostClassifierParameters>      MILAnyBoostClassifierParametersPtr;

    /****************************************************************
    StrongClassifierParametersBase
        Base class for all the strong classifier parameters.
    ****************************************************************/
    class StrongClassifierParametersBase
    {
    public:
        StrongClassifierParametersBase( const int       numberOfSelectedWeakClassifiers,
                                        const int       totalNumberOfWeakClassifiers,
                                        const float     percentageOfRetainedWeakClassifiers = 0,
                                        const float     learningRate                        = DEFAULT_GAUSSIAN_WEAK_CLASSIFIER_LEARNING_RATE, 
                                        const bool      storeFeatureHistory                 = DEFAULT_STRONG_CLASSIFIER_STORE_FEATURE_HISTORY )
            : m_featureParametersPtr( ),
            m_weakClassifierType( STUMP ),
            m_learningRate( learningRate ),
            m_storeFeatureHistory( storeFeatureHistory ),
            m_numberOfSelectedWeakClassifiers( numberOfSelectedWeakClassifiers ),
            m_totalNumberOfWeakClassifiers( totalNumberOfWeakClassifiers ),
            m_percentageOfRetainedWeakClassifiers( percentageOfRetainedWeakClassifiers/100.0f )
        {
        }

        virtual StrongClassifierType GetClassifierType( ) const = 0; 
        
        Features::FeatureParametersPtr          m_featureParametersPtr;
        WeakClassifierType                      m_weakClassifierType;
        float                                   m_learningRate;
        bool                                    m_storeFeatureHistory;
        int                                     m_numberOfSelectedWeakClassifiers;
        int                                     m_totalNumberOfWeakClassifiers;
        float                                   m_percentageOfRetainedWeakClassifiers;
    };

    /****************************************************************
    AdaBoostClassifierParameters
        Ada boost classifier parameters.
    ****************************************************************/
    class AdaBoostClassifierParameters : public StrongClassifierParametersBase
    {
    public:
        AdaBoostClassifierParameters(   const int numberOfSelectedWeakClassifiers,
                                        const int totalNumberOfWeakClassifiers )
            : StrongClassifierParametersBase( numberOfSelectedWeakClassifiers,
                                               totalNumberOfWeakClassifiers )
        {
        }

        virtual StrongClassifierType    GetClassifierType( ) const { return ONLINE_ADABOOST; };
    };

    /****************************************************************
    MILBoostClassifierParameters
        MIL boost classifier parameters.
    ****************************************************************/
    class MILBoostClassifierParameters : public StrongClassifierParametersBase
    {
    public:
        MILBoostClassifierParameters(    const int numberOfSelectedWeakClassifiers,
                                        const int totalNumberOfWeakClassifiers )
            : StrongClassifierParametersBase( numberOfSelectedWeakClassifiers,
                                            totalNumberOfWeakClassifiers )
        {
        }

        virtual StrongClassifierType    GetClassifierType( )const { return ONLINE_STOCHASTIC_BOOST_MIL; };
    };

    /****************************************************************
    MILEnsembleClassifierParameters
        MIL Ensemble classifier parameters.
    ****************************************************************/
    class MILEnsembleClassifierParameters : public StrongClassifierParametersBase
    {
    public:
        MILEnsembleClassifierParameters(    const int numberOfSelectedWeakClassifiers,
                                            const int totalNumberOfWeakClassifiers,
                                            const float percentageOfRetainedWeakClassifiers ) 
            : StrongClassifierParametersBase( numberOfSelectedWeakClassifiers,
                                            totalNumberOfWeakClassifiers,
                                            percentageOfRetainedWeakClassifiers )
        {        
        }
        virtual StrongClassifierType    GetClassifierType( ) const { return ONLINE_ENSEMBLE_BOOST_MIL; };
    };

    /****************************************************************
    MILAnyBoostClassifierParameters
        MIL Boost (AnyBoost) classifier parameters.
    ****************************************************************/
    class MILAnyBoostClassifierParameters : public StrongClassifierParametersBase
    {
    public:
        MILAnyBoostClassifierParameters(    const int numberOfSelectedWeakClassifiers,
                                            const int totalNumberOfWeakClassifiers ) 
            : StrongClassifierParametersBase( numberOfSelectedWeakClassifiers,
            totalNumberOfWeakClassifiers )
        {                    
        }

        virtual StrongClassifierType    GetClassifierType( ) const { return ONLINE_ANY_BOOST_MIL; };
    };
}
#endif