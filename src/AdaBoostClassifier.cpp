#include "AdaBoostClassifier.h"
#include "CommonMacros.h"

#include <algorithm>
#include <numeric>

namespace Classifier
{
    AdaBoostClassifier::AdaBoostClassifier( Classifier::StrongClassifierParametersBasePtr strongClassifierParametersBasePtr )
    : StrongClassifierBase( strongClassifierParametersBasePtr )
    {
        Initialize(strongClassifierParametersBasePtr);
    }

    /****************************************************************
    AdaBoostClassifier::Initialize
    Exceptions:
        None
    ****************************************************************/
    void    AdaBoostClassifier::Initialize( StrongClassifierParametersBasePtr strongClassifierParametersBasePtr )
    {
        try
        {
            m_adaBoostClassifierParametersPtr = boost::static_pointer_cast<Classifier::AdaBoostClassifierParameters>( strongClassifierParametersBasePtr );

            ASSERT_TRUE( m_adaBoostClassifierParametersPtr != NULL );

            resizeVec( m_countFPv, m_adaBoostClassifierParametersPtr->m_numberOfSelectedWeakClassifiers, 
                                m_adaBoostClassifierParametersPtr->m_totalNumberOfWeakClassifiers, 1.0f );
            resizeVec( m_countTPv, m_adaBoostClassifierParametersPtr->m_numberOfSelectedWeakClassifiers, 
                                m_adaBoostClassifierParametersPtr->m_totalNumberOfWeakClassifiers, 1.0f );
            
            resizeVec( m_countFNv, m_adaBoostClassifierParametersPtr->m_numberOfSelectedWeakClassifiers,
                                m_adaBoostClassifierParametersPtr->m_totalNumberOfWeakClassifiers, 1.0f );
            resizeVec( m_countTNv, m_adaBoostClassifierParametersPtr->m_numberOfSelectedWeakClassifiers,
                                m_adaBoostClassifierParametersPtr->m_totalNumberOfWeakClassifiers, 1.0f );

            m_alphaList.resize( m_adaBoostClassifierParametersPtr->m_numberOfSelectedWeakClassifiers, 0 );
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to initialize AdaBoostClassifier" );
    }

    /****************************************************************
    AdaBoostClassifier::Update
    Exceptions:
        None
    ****************************************************************/
    void    AdaBoostClassifier::Update( Classifier::SampleSet& positiveSampleSet, Classifier::SampleSet& negativeSampleSet )
    {
        try
        {
            m_classifierStopWatch.Start();
            size_t totalNumberOfSamples = positiveSampleSet.Size() + negativeSampleSet.Size();

            // Compute features
            if ( !positiveSampleSet.IsFeatureComputed() ) 
            {
                m_featureVectorPtr->Compute( positiveSampleSet );
            }

            if ( !negativeSampleSet.IsFeatureComputed() )
            {
                m_featureVectorPtr->Compute( negativeSampleSet );
            }

            vectorf poslam(positiveSampleSet.Size(),.5f/positiveSampleSet.Size()), neglam(negativeSampleSet.Size(),.5f/negativeSampleSet.Size());
            vector<vectorb> pospred(GetNumberOfFeatures()), negpred(GetNumberOfFeatures());
            vectorf errs(GetNumberOfFeatures());
            vectori order(GetNumberOfFeatures());

            m_sumOfAlphas=0.0f;
            m_selectorList.clear();

            // Update all weak classifiers and get predicted labels
            #pragma omp parallel for
            for ( int k=0; k<GetNumberOfFeatures(); k++ )
            {
                m_weakClassifierPtrList[k]->Update(positiveSampleSet,negativeSampleSet);
                pospred[k] = m_weakClassifierPtrList[k]->ClassifySet(positiveSampleSet);
                negpred[k] = m_weakClassifierPtrList[k]->ClassifySet(negativeSampleSet);
            }

            ASSERT_TRUE( m_adaBoostClassifierParametersPtr != NULL );

            vectori worstinds;

            // loop over selectors
            for( int selectedFeatureIndex=0; selectedFeatureIndex<m_adaBoostClassifierParametersPtr->m_numberOfSelectedWeakClassifiers; selectedFeatureIndex++ )
            {
                #pragma omp parallel for
                for( int weakClassifierIndex=0; weakClassifierIndex<m_adaBoostClassifierParametersPtr->m_totalNumberOfWeakClassifiers; weakClassifierIndex++ )
                {
                    for( int positiveSampleIndex=0; positiveSampleIndex<(int)poslam.size(); positiveSampleIndex++ )
                    {
                        //if( poslam[positiveSampleIndex] > 1e-5 )
                        (pospred[weakClassifierIndex][positiveSampleIndex])? m_countTPv[selectedFeatureIndex][weakClassifierIndex] += poslam[positiveSampleIndex] : m_countFPv[selectedFeatureIndex][weakClassifierIndex] += poslam[positiveSampleIndex];
                    }
                }
            #pragma omp parallel for
                for( int weakClassifierIndex=0; weakClassifierIndex<m_adaBoostClassifierParametersPtr->m_totalNumberOfWeakClassifiers; weakClassifierIndex++ ){
                    for( int j=0; j<(int)neglam.size(); j++ ){
                        //if( neglam[j] > 1e-5 )
                        (!negpred[weakClassifierIndex][j])? m_countTNv[selectedFeatureIndex][weakClassifierIndex] += neglam[j] : m_countFNv[selectedFeatureIndex][weakClassifierIndex] += neglam[j];
                    }
                }
                #pragma omp parallel for
                for( int weakClassifierIndex=0; weakClassifierIndex<m_adaBoostClassifierParametersPtr->m_totalNumberOfWeakClassifiers; weakClassifierIndex++ ){
                    errs[weakClassifierIndex] = ( m_countFPv[selectedFeatureIndex][weakClassifierIndex]    
                                                + m_countFNv[selectedFeatureIndex][weakClassifierIndex] )
                                                    /(  m_countFPv[selectedFeatureIndex][weakClassifierIndex]    +    
                                                        m_countFNv[selectedFeatureIndex][weakClassifierIndex]    + 
                                                        m_countTPv[selectedFeatureIndex][weakClassifierIndex]    +
                                                        m_countTNv[selectedFeatureIndex][weakClassifierIndex]    );
                }

                // pick the best weak pStrongClassifierBase and update m_selectorList and _selectedFtrs
                float minerr=0;
                uint bestind=0;

                sort_order(errs,order); // //ascending order (minimize the error rate)

                // find best in that isn'selectedFeatureIndex already included
                for( uint k=0; k<order.size(); k++ )
                {
                    if( count( m_selectorList.begin(), m_selectorList.end(), order[k])==0 )
                    {
                        m_selectorList.push_back(order[k]);
                        minerr = errs[k];
                        bestind = order[k];
                        break;
                    }
                }
                    // find worst ind
                worstinds.push_back(order[order.size()-1]);

                // Update alpha
                //m_alphaList[selectedFeatureIndex] = max(0,min(0.5f*log((1-minerr)/(minerr+0.00001f)),10));
                m_alphaList[selectedFeatureIndex] = max((float)0,min(0.5f*log((1-minerr)/(minerr+0.00001f)),(float)10));//[zefeng Ni] FOR gcc compatibility

                m_sumOfAlphas += m_alphaList[selectedFeatureIndex];

                // Update sample weights
                float corw = 1/(2-2*minerr);//reduce weight
                float incorw = 1/(2*minerr);//increase weight
                #pragma omp parallel for
                for( int j=0; j<(int)poslam.size(); j++ )
                {
                    poslam[j] *= (pospred[bestind][j]==1)? corw : incorw;
                }
                #pragma omp parallel for
                for( int j=0; j<(int)neglam.size(); j++ )
                {
                    neglam[j] *= (negpred[bestind][j]==0)? corw : incorw;
                }
            }

            m_numberOfSamples += static_cast<uint>(totalNumberOfSamples);
            m_classifierStopWatch.Stop();

            return;
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to Update the classifier with Adaboost" );
    }


    /****************************************************************
    AdaBoostClassifier::Classify
        Classify the set of samples provided and store the response in a
        List.
    Exceptions:
        None
    ****************************************************************/
    vectorf    AdaBoostClassifier::Classify( Classifier::SampleSet& sampleSet, bool isLogRatioEnabled )
    {
        try
        {
            //get the number of samples
            size_t numberOfSamples = sampleSet.Size( );

            // Compute features
            if ( !sampleSet.IsFeatureComputed( ) ) 
            {
                m_featureVectorPtr->Compute( sampleSet );
            }

            //response list to store the response for each sample
            vectorf responseList(numberOfSamples);

            // for each selector, accumulate in the responseList
            for ( int selectedFeatureIndex = 0; selectedFeatureIndex < (int)m_selectorList.size(); selectedFeatureIndex++ )
            {
                vectorb weaklyClassifiedLabelList;
                weaklyClassifiedLabelList = m_weakClassifierPtrList[m_selectorList[selectedFeatureIndex]]->ClassifySet( sampleSet );

                #pragma omp parallel for
                for( int sampleIndex=0; sampleIndex<numberOfSamples; sampleIndex++ )
                {
                    responseList[sampleIndex] += weaklyClassifiedLabelList[sampleIndex] ?  m_alphaList[selectedFeatureIndex] : -m_alphaList[selectedFeatureIndex];
                }
            }

            // return probabilities or log odds ratio
            if ( !isLogRatioEnabled )
            {
                #pragma omp parallel for
                for( int sampleIndex=0; sampleIndex<(int)responseList.size(); sampleIndex++ )
                {
                    responseList[sampleIndex] = sigmoid( 2 * responseList[sampleIndex] );
                }
            }

            //return the responseList corresponding to each of the samples
            return responseList;
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to Classify the given sample set" );
    }
}