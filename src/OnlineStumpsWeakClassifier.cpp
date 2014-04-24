#include "OnlineStumpsWeakClassifier.h"

#define FEATURE_DIMENSION                            1
#define MAXIMUM_NUMBER_OF_ITERATIONS                10
#define MINIMUM_ERROR_THRESHOLD_FOR_PERCEPTRON        0.5

namespace Classifier
{
    /****************************************************************
    OnlineStumpsWeakClassifier
        C'tor
    ****************************************************************/
    OnlineStumpsWeakClassifier::OnlineStumpsWeakClassifier( ) 
        : WeakClassifierBase( ) 
    { 
        Initialize();
    }

    /****************************************************************
    IsValidWeakClassifier
    ****************************************************************/
    bool OnlineStumpsWeakClassifier::IsValidWeakClassifier( )
    {
        try
        {
            if ( m_mu0 == m_mu1 )
            {
                return false;
            }
            
            return true;
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to check the weak claassifier validity" );
    }

    /****************************************************************
    OnlineStumpsWeakClassifier
        C'tor
    ****************************************************************/
    OnlineStumpsWeakClassifier::OnlineStumpsWeakClassifier( const int featureId )        
        : WeakClassifierBase( featureId )
    {
        Initialize( );
        m_learningRate    = 0.85f;
    }

    /****************************************************************
    OnlineStumpsWeakClassifier::Initialize
        Initializes the parameters
    Exceptions:
        None
    ****************************************************************/
    void OnlineStumpsWeakClassifier::Initialize()
    {
        m_mu0    = 0;
        m_mu1    = 0;
        m_sig0    = 1;
        m_sig1    = 1;
        m_isWeakClassifierTrained = false;
    }

    /****************************************************************
    OnlineStumpsWeakClassifier::Classify
        Classify 
    Exceptions:
        None
    ****************************************************************/
    bool    OnlineStumpsWeakClassifier::Classify( const Classifier::SampleSet& sampleSet, const int sampleIndex )
    {
        float xx = GetFeatureValue( sampleSet, sampleIndex );
        double p0 = exp( (xx-m_mu0)*(xx-m_mu0)*m_e0 )*m_n0;
        double p1 = exp( (xx-m_mu1)*(xx-m_mu1)*m_e1 )*m_n1;
        bool r = p1>p0;
        return r;
    }

    /****************************************************************
    OnlineStumpsWeakClassifier::ClassifyF
        ClassifyF
    Exceptions:
        None
    ****************************************************************/
    float    OnlineStumpsWeakClassifier::ClassifyF( const Classifier::SampleSet& sampleSet, const int sampleIndex )
    {
        float xx = GetFeatureValue(sampleSet,sampleIndex);
        double p0 = exp( (xx-m_mu0)*(xx-m_mu0)*m_e0 )*m_n0;
        double p1 = exp( (xx-m_mu1)*(xx-m_mu1)*m_e1 )*m_n1;
        float r = (float)(log(1e-5+p1)-log(1e-5+p0));
        return r;
    }


    /****************************************************************
    OnlineStumpsWeakClassifier::Update
        Update
    Exceptions:
        None
    ****************************************************************/
    void    OnlineStumpsWeakClassifier::Update( const Classifier::SampleSet&    positiveSampleSet,
                                               const Classifier::SampleSet&        negativeSampleSet, 
                                               vectorf*                /*pPositiveSamplesWeightList*/, 
                                               vectorf*                /*pNegativeSamplesWeightList*/ )
    {
        float positiveSampleFeatureMeanValue=0.0;
        if ( positiveSampleSet.Size() > 0 )
        {
            positiveSampleFeatureMeanValue = positiveSampleSet.FeatureValues(m_featureIndex).Mean();
        }

        float negativeSampleFeatureMeanValue = 0.0f;
        if( negativeSampleSet.Size() > 0 )
        {
            negativeSampleFeatureMeanValue = negativeSampleSet.FeatureValues(m_featureIndex).Mean();
        }

        if ( m_isWeakClassifierTrained )
        {
            if ( positiveSampleSet.Size()>0 )
            {
                m_mu1    = ( m_learningRate*m_mu1  + (1-m_learningRate) * positiveSampleFeatureMeanValue );
                m_sig1    = ( m_learningRate*m_sig1 + (1-m_learningRate)  * ( (positiveSampleSet.FeatureValues(m_featureIndex)-m_mu1).Sqr().Mean() ) );
            }

            if ( negativeSampleSet.Size()>0 )
            {
                m_mu0    = ( m_learningRate*m_mu0  + (1-m_learningRate) * negativeSampleFeatureMeanValue );
                m_sig0    = ( m_learningRate*m_sig0 + (1-m_learningRate) * ( (negativeSampleSet.FeatureValues(m_featureIndex)-m_mu0).Sqr().Mean() ) );
            }
        }
        else
        {
            m_isWeakClassifierTrained = true;
            if ( positiveSampleSet.Size() > 0 )
            {
                m_mu1 = positiveSampleFeatureMeanValue;
                m_sig1 = positiveSampleSet.FeatureValues(m_featureIndex).Var()+1e-9f;
            }

            if ( negativeSampleSet.Size()>0 )
            {
                m_mu0 = negativeSampleFeatureMeanValue;
                m_sig0 = negativeSampleSet.FeatureValues(m_featureIndex).Var()+1e-9f;
            }
        }

        //update the factors for fast computation
        m_n0 = 1.0f / pow(m_sig0,0.5f);
        m_n1 = 1.0f / pow(m_sig1,0.5f);
        m_e1 = -1.0f/(2.0f*m_sig1+1e-99f);
        m_e0 = -1.0f/(2.0f*m_sig0+1e-99f);
    }
}