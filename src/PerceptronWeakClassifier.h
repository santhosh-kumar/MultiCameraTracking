#ifndef PERCEPTRON_WEAK_CLASSIFIER_H
#define PERCEPTRON_WEAK_CLASSIFIER_H

#include "WeakClassifierBase.h"

namespace Classifier
{
    /****************************************************************
    PerceptronWeakClassifier
        Perceptron weak classifier.
    ****************************************************************/
    class PerceptronWeakClassifier : public WeakClassifierBase
    {
    public:
        PerceptronWeakClassifier( );

        PerceptronWeakClassifier( const int featureId );

        virtual WeakClassifierType        GetClassifierType( ) { return PERCEPTRON; }

        virtual void        Initialize( );

        virtual void        Update( const Classifier::SampleSet&    positiveSampleSet,
                                    const Classifier::SampleSet&    negativeSampleSet, 
                                    vectorf*                        pPositiveSamplesWeightList = NULL, 
                                    vectorf*                        pNegativeSamplesWeightList = NULL );

        virtual bool        Classify( const Classifier::SampleSet&  sampleSet,
                                      const int                     sampleIndex );

        virtual float       ClassifyF( const Classifier::SampleSet& sampleSet, const int sampleIndex );

        virtual bool        IsValidWeakClassifier( ){ return true; }

    private:
        vectorf                m_weightList;
    };
}
#endif