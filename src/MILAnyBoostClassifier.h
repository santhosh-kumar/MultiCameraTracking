#ifndef MIL_ANYBOOST_H
#define MIL_ANYBOOST_H

#include "StrongClassifierBase.h"

namespace Classifier
{
    /****************************************************************
    MILAnyBoostClassifier
        The class that implements MIL Anyboost based classifier.
    ****************************************************************/
    class MILAnyBoostClassifier : public StrongClassifierBase
    {
    public:
        MILAnyBoostClassifier( Classifier::StrongClassifierParametersBasePtr strongClassifierParametersBasePtr )
            : StrongClassifierBase( strongClassifierParametersBasePtr )
        {
            try
            {
                m_milAnyBoostClassifierParametersPtr    = boost::static_pointer_cast<MILAnyBoostClassifierParameters>( strongClassifierParametersBasePtr );
            }
            EXCEPTION_CATCH_AND_ABORT( "Failed to Construct MILAnyBoost" );
        }
        
        //update the classifier
        virtual void        Update( Classifier::SampleSet& positiveSampleSet, Classifier::SampleSet& negativeSampleSet );
        
        //classify the set of samples
        virtual vectorf        Classify( Classifier::SampleSet& sampleSet, bool isLogRatioEnabled = true );

    private:
        MILAnyBoostClassifierParametersPtr        m_milAnyBoostClassifierParametersPtr;
    };
}
#endif
