#ifndef STRONG_CLASSIFIER_BASE_H
#define STRONG_CLASSIFIER_BASE_H

#include "ClassifierParameters.h"
#include "WeakClassifierBase.h"
#include "OnlineStumpsWeakClassifier.h"
#include "WeightedStumpsWeakClassifier.h"
#include "PerceptronWeakClassifier.h"

#include "FeatureVector.h"
#include "HaarFeatureVector.h"
#include "CultureColorHistogramFeatureVector.h"
#include "MultiDimensionalColorHistogramFeatureVector.h"
#include "HaarAndColorHistogramFeatureVector.h"

namespace Classifier
{
    //Forward Declaration
    class StrongClassifierBase;
    class AdaBoostClassifier;
    class MILBoostClassifier;
    class MILEnsembleClassifier;

    //declarations of shared ptr
    typedef boost::shared_ptr<StrongClassifierBase>        StrongClassifierBasePtr;
    typedef boost::shared_ptr<AdaBoostClassifier>        AdaBoostClassifierPtr;
    typedef boost::shared_ptr<MILBoostClassifier>        MILBoostClassifierPtr;
    typedef boost::shared_ptr<MILEnsembleClassifier>    MILEnsembleClassifierPtr;

    #define MIL_STOPPING_THRESHOLD 1e-100

    /****************************************************************
    StrongClassifierBase
        Base class for all the strong classifiers.
    ****************************************************************/
    class StrongClassifierBase
    {
    public:

        StrongClassifierBase( StrongClassifierParametersBasePtr strongClassifierParametersBasePtr );

        // pure virtual functions
        virtual void        Update( Classifier::SampleSet& positiveSampleSet, Classifier::SampleSet& negativeSampleSet ) = 0;
        virtual vectorf        Classify( Classifier::SampleSet& sampleSet, bool isLogRatioEnabled = true ) = 0;

        virtual void        Update( Classifier::SampleSet& positiveSampleSet, Classifier::SampleSet& negativeSampleSet, int numPositiveBags ) { };
        //member functions
        int    GetNumberOfFeatures( ) 
        { 
            ASSERT_TRUE( m_featureVectorPtr != NULL );
            return m_featureVectorPtr->GetNumberOfFeatures( ); 
        }

        
    private:
        void InitializeFeatureVector( );
        void InitializeWeakClassifiers( );

    protected:
        StrongClassifierParametersBasePtr                m_strongClassifierParametersBasePtr;
        Features::FeatureVectorPtr                        m_featureVectorPtr;
        StopWatch                                        m_classifierStopWatch;
        vectori                                            m_selectorList;             //List of features selected in boosting
        vector<WeakClassifierBase*>                        m_weakClassifierPtrList; //List of weak classifiers that are available.
        uint                                            m_numberOfSamples;    
        uint                                            m_counter;
    };
}
#endif