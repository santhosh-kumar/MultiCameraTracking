#include "WeightedStumpsWeakClassifier.h"

#define FEATURE_DIMENSION                            1
#define MAXIMUM_NUMBER_OF_ITERATIONS                10
#define MINIMUM_ERROR_THRESHOLD_FOR_PERCEPTRON        0.5

namespace Classifier
{
    /****************************************************************
    ClfWStump::ClfWStump( ) 
        C'tor
    ****************************************************************/
    WeightedStumpsWeakClassifier::WeightedStumpsWeakClassifier( ) 
        : WeakClassifierBase( ) 
    {
        Initialize();
    }

    /****************************************************************
    WeightedStumpsWeakClassifier::WeightedStumpsWeakClassifier( ) 
        C'tor
    ****************************************************************/
    WeightedStumpsWeakClassifier::WeightedStumpsWeakClassifier( const int featureId ) 
        : WeakClassifierBase(featureId) 
    {

        m_learningRate                = 0.85f;
        Initialize();
    }

    /****************************************************************
    ClfWStump::Initialize 
        C'tor
    ****************************************************************/
    void    WeightedStumpsWeakClassifier::Initialize( )
    {
        m_mu0                        = 0;
        m_mu1                        = 0;
        m_sig0                        = 1;
        m_sig1                        = 1;
        m_isWeakClassifierTrained    = false;
    }

    /****************************************************************
    WeightedStumpsWeakClassifier::ClassifyF 
        Classify the set of samples using the learned weak classifier.
    Exceptions:
        None
    ****************************************************************/
    float    WeightedStumpsWeakClassifier::ClassifyF( const Classifier::SampleSet& sampleSet, const int sampleIndex )
    {
        float xx    = GetFeatureValue(sampleSet,sampleIndex);
        double p0    = exp( (xx-m_mu0)*(xx-m_mu0)*m_e0 )*m_n0;
        double p1    = exp( (xx-m_mu1)*(xx-m_mu1)*m_e1 )*m_n1;
        float r        = (float)(log(1e-5+p1)-log(1e-5+p0));
        return r;
    }

    /****************************************************************
    WeightedStumpsWeakClassifier::Update 
        Update the weak classifier
    Exceptions:
        None
    ****************************************************************/
    void    WeightedStumpsWeakClassifier::Update( const Classifier::SampleSet&    positiveSampleSet,
                                                 const Classifier::SampleSet&    negativeSampleSet, 
                                                 vectorf*            pPositiveSamplesWeightList,
                                                 vectorf*            pNegativeSamplesWeightList )
    {
        Matrixf poswm, negwm, poswn, negwn;
        if( ( positiveSampleSet.Size() != pPositiveSamplesWeightList->size() ) || (negativeSampleSet.Size() != pNegativeSamplesWeightList->size()) )
        {
            abortError(__LINE__,__FILE__,"ClfWStump::Update - number of samples and number of weights mismatch");
        }

        float posmu=0.0, negmu=0.0;
        if( positiveSampleSet.Size()>0 )
        {
            poswm = *pPositiveSamplesWeightList;
            poswn = poswm.normalize();
            posmu = positiveSampleSet.FeatureValues(m_featureIndex).MeanW(poswn);
        }

        if( negativeSampleSet.Size()>0 ) 
        {
            negwm = *pNegativeSamplesWeightList;
            negwn = negwm.normalize();
            negmu = negativeSampleSet.FeatureValues(m_featureIndex).MeanW(negwn);
        }

        if( m_isWeakClassifierTrained )
        {
            if( positiveSampleSet.Size()>0 )
            {
                m_mu1    = ( m_learningRate*m_mu1  + (1-m_learningRate)*posmu );
                m_sig1    = ( m_learningRate*m_sig1  + (1-m_learningRate)*positiveSampleSet.FeatureValues(m_featureIndex).VarW(poswn,&m_mu1) );
            }

            if( negativeSampleSet.Size()>0 )
            {
                m_mu0    = ( m_learningRate*m_mu0  + (1-m_learningRate)*negmu );
                m_sig0    = ( m_learningRate*m_sig0  + (1-m_learningRate)*negativeSampleSet.FeatureValues(m_featureIndex).VarW(negwn,&m_mu0) );
            }
        }
        else
        {
            m_isWeakClassifierTrained = true;
            m_mu1 = posmu;
            m_mu0 = negmu;
            if( negativeSampleSet.Size()>0 ) m_sig0 = negativeSampleSet.FeatureValues(m_featureIndex).VarW(negwn,&negmu)+1e-9f;
            if( positiveSampleSet.Size()>0 ) m_sig1 = positiveSampleSet.FeatureValues(m_featureIndex).VarW(poswn,&posmu)+1e-9f;
        }

        m_n0 = 1.0f/pow(m_sig0,0.5f);
        m_n1 = 1.0f/pow(m_sig1,0.5f);
        m_e1 = -1.0f/(2.0f*m_sig1);
        m_e0 = -1.0f/(2.0f*m_sig0);
    }
}