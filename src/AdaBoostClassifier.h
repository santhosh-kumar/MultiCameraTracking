#ifndef ADABOOST_H
#define ADABOOST_H

#include "StrongClassifierBase.h"

namespace Classifier
{
    /****************************************************************
    AdaBoostClassifier
        Implements AdaBoost based Strong Classifier.
    ****************************************************************/
    class AdaBoostClassifier : public StrongClassifierBase
    {
    public:

        //Constructor
        AdaBoostClassifier( Classifier::StrongClassifierParametersBasePtr strongClassifierParametersBasePtr );

        //Update the strong classifier
        virtual void    Update( Classifier::SampleSet& positiveSampleSet,
                                Classifier::SampleSet& negativeSampleSet );

        //Classify the sample set
        virtual vectorf Classify(   Classifier::SampleSet&  sampleSet,
                                    bool                    isLogRatioEnabled = true );

    private:

        //Initialize the classifier
        void    Initialize( Classifier::StrongClassifierParametersBasePtr strongClassifierParametersBasePtr );

        AdaBoostClassifierParametersPtr        m_adaBoostClassifierParametersPtr;
        vectorf                                m_alphaList;
        float                                  m_sumOfAlphas;
        vector<vectorf>                        m_countFPv;
        vector<vectorf>                        m_countFNv;
        vector<vectorf>                        m_countTPv;
        vector<vectorf>                        m_countTNv; //[selector][feature]
    };
}
#endif
