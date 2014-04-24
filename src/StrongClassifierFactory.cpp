#include "StrongClassifierFactory.h"
#include "AdaBoostClassifier.h"
#include "MILBoostClassifier.h"
#include "MILEnsembleClassifier.h"
#include "MILAnyBoostClassifier.h"

namespace Classifier
{
    /****************************************************************
    CreateAndInitializeClassifier
        Creates a classifier according to the type specified in the
        parameters. Initializes appropriate classifier.
    ****************************************************************/
    StrongClassifierBasePtr    StrongClassifierFactory::CreateAndInitializeClassifier( StrongClassifierParametersBasePtr strongClassifierParametersBasePtr )
    {
        try
        {
            StrongClassifierBasePtr strongClassifierBasePtr;

            switch ( strongClassifierParametersBasePtr->GetClassifierType( ) )
            {
                case ONLINE_ADABOOST:
                    strongClassifierBasePtr = StrongClassifierBasePtr( new AdaBoostClassifier( strongClassifierParametersBasePtr ) );
                    break;
                case ONLINE_STOCHASTIC_BOOST_MIL:
                    strongClassifierBasePtr = StrongClassifierBasePtr( new MILBoostClassifier( strongClassifierParametersBasePtr ) );
                    break;
                case ONLINE_ENSEMBLE_BOOST_MIL:
                    strongClassifierBasePtr = StrongClassifierBasePtr( new MILEnsembleClassifier( strongClassifierParametersBasePtr ) );
                    break;
                case ONLINE_ANY_BOOST_MIL:
                    strongClassifierBasePtr    = StrongClassifierBasePtr( new MILAnyBoostClassifier( strongClassifierParametersBasePtr ) );
                    break;
                default:
                    abortError(__LINE__,__FILE__,"Incorrect pStrongClassifierBase type!");
            }

            ASSERT_TRUE( strongClassifierBasePtr.get( ) != NULL );

            return strongClassifierBasePtr;
        }
        EXCEPTION_CATCH_AND_ABORT("Failed to create and initialize the classifier" )
    }
}