#ifndef ONLINE_STUMP_WEAK_CLASSIFIER_H
#define ONLINE_STUMP_WEAK_CLASSIFIER_H

#include "WeakClassifierBase.h"

namespace Classifier
{
    /****************************************************************
    OnlineStumpsWeakClassifier
        Online stumps weak classifier.
    ****************************************************************/
    class OnlineStumpsWeakClassifier : public WeakClassifierBase
    {
    public:
        OnlineStumpsWeakClassifier( );
        OnlineStumpsWeakClassifier( const int featureId );

        virtual WeakClassifierType        GetClassifierType( ) { return STUMP; }

        virtual void        Update( const Classifier::SampleSet&    positiveSampleSet,
                                    const Classifier::SampleSet&    negativeSampleSet, 
                                    vectorf*    pPositiveSamplesWeightList, 
                                    vectorf*    pNegativeSamplesWeightList );

        virtual bool        Classify( const Classifier::SampleSet& sampleSet, const int sampleIndex );
        virtual float        ClassifyF( const Classifier::SampleSet& sampleSet, const int sampleIndex );

        virtual bool        IsValidWeakClassifier( );

    private:
        virtual void        Initialize( );

        float                m_mu0;
        float                m_mu1;
        float                m_sig0;
        float                m_sig1;
        float                m_n1;
        float                m_n0;
        float                m_e1;
        float                m_e0;
    };
}
#endif