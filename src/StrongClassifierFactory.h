#include "StrongClassifierBase.h"

namespace Classifier
{
    class StrongClassifierFactory
    {
    public:
        //static functions
        static StrongClassifierBasePtr    CreateAndInitializeClassifier( Classifier::StrongClassifierParametersBasePtr clfParamsPtr );
    };
}