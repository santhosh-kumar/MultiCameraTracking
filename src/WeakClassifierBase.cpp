#include "WeakClassifierBase.h"

namespace Classifier
{
    /****************************************************************
    WeakClassifierBase
        Constructor
    Exceptions:
        None
    ****************************************************************/
    WeakClassifierBase::WeakClassifierBase( )
        : m_isWeakClassifierTrained( false ),
        m_featureIndex( -1 )
    {
    }

    /****************************************************************
    WeakClassifierBase
        Constructor
    Exceptions:
        None
    ****************************************************************/
    WeakClassifierBase::WeakClassifierBase( const int featureId )
        :m_isWeakClassifierTrained( false ),
        m_featureIndex( featureId )
    {
    }

    /****************************************************************
    WeakClassifierBase::ClassifySet
        Classifies the given set of samples.
    Exceptions:
        None
    ****************************************************************/
    vectorb    WeakClassifierBase::ClassifySet( const Classifier::SampleSet& sampleSet )
    {
        vectorb responseList( sampleSet.Size() );
        
        #pragma omp parallel for
        for ( int sampleIndex = 0; sampleIndex < sampleSet.Size() ; sampleIndex++ )
        {
            responseList[sampleIndex] = Classify( sampleSet, sampleIndex );
        }
        return responseList;
    }

    /****************************************************************
    WeakClassifierBase::ClassifySetF
        Classifies the given set of samples.
    Exceptions:
        None
    ****************************************************************/
    vectorf    WeakClassifierBase::ClassifySetF( const Classifier::SampleSet& sampleSet )
    {
        vectorf responseList( sampleSet.Size( ) );

        #pragma omp parallel for
        for ( int sampleIndex = 0; sampleIndex < sampleSet.Size(); sampleIndex++ )
        {
            responseList[sampleIndex] = ClassifyF( sampleSet, sampleIndex );
        }
        return responseList;
    }

    /****************************************************************
    WeakClassifierBase::ClassifySetF
        Classifies the given set of samples.
    Exceptions:
        None
    ****************************************************************/
    float    WeakClassifierBase::GetFeatureValue( const Classifier::SampleSet& sampleSet, const int sampleIndex )
    { 
        ASSERT_TRUE( sampleSet.IsFeatureComputed() ); 
        return sampleSet.GetFeatureValue( sampleIndex, m_featureIndex );
    }
}