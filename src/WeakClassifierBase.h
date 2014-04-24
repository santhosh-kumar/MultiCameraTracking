#ifndef WEAK_CLASSIFIER_BASE_H
#define WEAK_CLASSIFIER_BASE_H

#include "Feature.h"

namespace Classifier
{
    //Forward declarations
    class OnlineStumpsWeakClassifier;
    class WeightedStumpsWeakClassifier;
    class PerceptronWeakClassifier;

    enum WeakClassifierType{    STUMP,
                                WEIGHTED_STUMP,
                                PERCEPTRON,
                            };

    /****************************************************************
    WeakClassifierBase
        Base class for all the weak classifiers.
    ****************************************************************/
    class WeakClassifierBase
    {
    public:
        WeakClassifierBase( );
        WeakClassifierBase( const int featureId );

        //pure virtual functions
        virtual void        Initialize( ) = 0;

        virtual void        Update( const Classifier::SampleSet&    positiveSampleSet,
                                    const Classifier::SampleSet&    negativeSampleSet, 
                                    vectorf*                        pPositiveSamplesWeightList = NULL, 
                                    vectorf*                        pNegativeSamplesWeightList = NULL ) = 0;

        virtual bool        Classify( const Classifier::SampleSet& sampleSet, const int sampleIndex )=0;

        virtual float        ClassifyF( const Classifier::SampleSet& sampleSet, const int sampleIndex )=0;

        virtual WeakClassifierType        GetClassifierType( ) = 0;

        virtual bool                    IsValidWeakClassifier( ) = 0;

        //member functions
        virtual vectorb        ClassifySet( const Classifier::SampleSet& sampleSet );
        virtual vectorf        ClassifySetF( const Classifier::SampleSet& sampleSet );

        float                GetFeatureValue( const Classifier::SampleSet& sampleSet, const int sampleIndex );
        void                SetLearningRate( const float learningRate ){ m_learningRate = learningRate; }

    protected:

        bool                            m_isWeakClassifierTrained;
        const int                        m_featureIndex;
        float                            m_learningRate;
    };
}
#endif
