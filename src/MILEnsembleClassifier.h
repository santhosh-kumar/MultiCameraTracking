#ifndef MILENSEMBLE_H
#define MILENSEMBLE_H

#include "StrongClassifierBase.h"

namespace Classifier
{
    /****************************************************************
    MILEnsembleClassifier
        The class that implements MIL Ensemble boosting based classifier.
    ****************************************************************/
    class MILEnsembleClassifier : public StrongClassifierBase
    {
    public:
        
        MILEnsembleClassifier( Classifier::StrongClassifierParametersBasePtr strongClassifierParametersBasePtr )
            : StrongClassifierBase( strongClassifierParametersBasePtr )
        {
            m_MILEnsembleClassifierParametersPtr    = boost::static_pointer_cast<MILEnsembleClassifierParameters>( strongClassifierParametersBasePtr );
        }

        virtual void        Update( Classifier::SampleSet& positiveSampleSet, Classifier::SampleSet& negativeSampleSet );
        virtual vectorf        Classify( Classifier::SampleSet& sampleSet, bool isLogRatioEnabled = true );

    private:

        void                RetainBestPerformingWeakClassifiers( Classifier::SampleSet& positiveSampleSet,
                                                                Classifier::SampleSet&    negativeSampleSet,
                                                                vectorf&                positiveHypothesis, 
                                                                vectorf&                negativeHypothesis );

        void                RetainBestPerformingWeakClassifiersWithAdaptiveWeighting( Classifier::SampleSet& positiveSampleSet,
                                                                Classifier::SampleSet&    negativeSampleSet,
                                                                vectorf&                positiveHypothesis, 
                                                                vectorf&                negativeHypothesis );

        void                UpdateWithAdaptiveWeighting( Classifier::SampleSet& positiveSampleSet, Classifier::SampleSet& negativeSampleSet );

        MILEnsembleClassifierParametersPtr        m_MILEnsembleClassifierParametersPtr;
    };
}
#endif
