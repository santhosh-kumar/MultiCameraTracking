#ifndef MILBOOST_H
#define MILBOOST_H

#include "StrongClassifierBase.h"

namespace Classifier
{
    /****************************************************************
    MILBoostClassifier
        The class that implements MIL boost based classifier.
    ****************************************************************/
    class MILBoostClassifier : public StrongClassifierBase
    {
    public:
        
        MILBoostClassifier( Classifier::StrongClassifierParametersBasePtr strongClassifierParametersBasePtr )
            : StrongClassifierBase( strongClassifierParametersBasePtr )
        {
            m_MILBoostClassifierParametersPtr   =   
                boost::static_pointer_cast<MILBoostClassifierParameters>( strongClassifierParametersBasePtr );
        }

        virtual void        Update( Classifier::SampleSet& positiveSampleSet,
                                    Classifier::SampleSet& negativeSampleSet );

        virtual vectorf     Classify(   Classifier::SampleSet& sampleSet, 
                                        bool isLogRatioEnabled = true );

        //multiple positive bags
        virtual void        Update( Classifier::SampleSet& positiveSampleSet,
                                    Classifier::SampleSet& negativeSampleSet, 
                                    int                    numPositiveBags );

    private:
        MILBoostClassifierParametersPtr        m_MILBoostClassifierParametersPtr;
    };
}
#endif
