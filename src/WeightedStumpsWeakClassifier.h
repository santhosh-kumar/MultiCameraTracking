#ifndef WEIGHTED_STUMP_WEAK_CLASSIFIER_H
#define WEIGHTED_STUMP_WEAK_CLASSIFIER_H

#include "WeakClassifierBase.h"

namespace Classifier
{
    /****************************************************************
    WeightedStumpsWeakClassifier
        Stumps weak classifier.
    ****************************************************************/
    class WeightedStumpsWeakClassifier : public WeakClassifierBase
    {
    public:
        WeightedStumpsWeakClassifier( ); 
        WeightedStumpsWeakClassifier( const int featureId );

        virtual WeakClassifierType        GetClassifierType( ) { return WEIGHTED_STUMP; }

        virtual void        Initialize( );
        virtual void        Update( const Classifier::SampleSet&    positiveSampleSet,
                                    const Classifier::SampleSet&    negativeSampleSet, 
                                    vectorf*                        pPositiveSamplesWeightList = NULL, 
                                    vectorf*                        pNegativeSamplesWeightList = NULL );
        virtual bool        Classify( const Classifier::SampleSet& sampleSet, const int sampleIndex ){ return ClassifyF( sampleSet, sampleIndex ) > 0; } 
        virtual float        ClassifyF( const Classifier::SampleSet& sampleSet, const int sampleIndex );

        virtual bool        IsValidWeakClassifier( ){ return true; }

    private:
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