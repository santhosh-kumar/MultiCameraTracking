#include "MILAnyBoostClassifier.h"
#include "CommonMacros.h"

#include <algorithm>
#include <numeric>

namespace Classifier
{
    /****************************************************************
    MILBoostClassifier::Update
    Exceptions:
        None
    ****************************************************************/
    void    MILAnyBoostClassifier::Update( Classifier::SampleSet& positiveSampleSet, Classifier::SampleSet& negativeSampleSet)
    {
        //start the stop watch
        m_classifierStopWatch.Start();

        //get the number of positive and negative samples
        size_t numberOfNegativeSamples = negativeSampleSet.Size();
        size_t numberOfPositiveSamples = positiveSampleSet.Size();

        // Compute features if it'selectedFeatureIndex not already computed
        if ( !positiveSampleSet.IsFeatureComputed( ) )
        {
            m_featureVectorPtr->Compute(positiveSampleSet);
        }

        if ( !negativeSampleSet.IsFeatureComputed( ) )
        {
            m_featureVectorPtr->Compute( negativeSampleSet );
        }

        // initialize H - hypothesis
        vectorf positiveHypothesis;
        vectorf negativeHypothesis;
        positiveHypothesis.clear(); 
        negativeHypothesis.clear();
        positiveHypothesis.resize( positiveSampleSet.Size( ), 0.0f );
        negativeHypothesis.resize( negativeSampleSet.Size( ), 0.0f );

        //clear the selected feature list
        m_selectorList.clear();

    
        vector<vectorf> positiveSamplePredictionWithDifferentWeakClassifiers( m_weakClassifierPtrList.size( ) );
        vector<vectorf> negativeSamplePredictionWithDifferentWeakClassifiers( m_weakClassifierPtrList.size( ) );

        // train all weak classifiers without weights and find the prediction for positive and negative samples
        #pragma omp parallel for
        for ( int featureIndex = 0; featureIndex < m_milAnyBoostClassifierParametersPtr->m_totalNumberOfWeakClassifiers; featureIndex++ )
        {
            m_weakClassifierPtrList[featureIndex]->Update( positiveSampleSet, negativeSampleSet );
            positiveSamplePredictionWithDifferentWeakClassifiers[featureIndex] = m_weakClassifierPtrList[featureIndex]->ClassifySetF( positiveSampleSet );
            negativeSamplePredictionWithDifferentWeakClassifiers[featureIndex] = m_weakClassifierPtrList[featureIndex]->ClassifySetF( negativeSampleSet );
        }

        //used for termination
        double previousNegLoglikehood = 1000000;
        
        vectorf positiveInstanceWeight( numberOfPositiveSamples );
        vectorf negativeInstanceWeight( numberOfNegativeSamples );

        vectorf positiveInstanceProbability;
        float    positiveBagProbability;
        vectorf negativeInstanceProbability;
        //vectorf negativeBagProbability; 

        positiveInstanceProbability.resize( numberOfPositiveSamples, 0);
        negativeInstanceProbability.resize( numberOfNegativeSamples, 0 );
        //negativeBagProbability.resize( numberOfNegativeSamples, 0 );
        
        float negLogLikelihood; 
        vectori order;
        uint k = 0;

        // pick the best features
        for ( int selectedFeatureIndex = 0; selectedFeatureIndex < m_milAnyBoostClassifierParametersPtr->m_numberOfSelectedWeakClassifiers; selectedFeatureIndex++ )
        {
            // Compute errors/negLogLikelihoodList for all weak classifiers
            vectorf objectFunctionList( m_weakClassifierPtrList.size( ), 0 );

            float likeliHood = 1.0f;

            for ( int positiveSampleIndex = 0; positiveSampleIndex < numberOfPositiveSamples; positiveSampleIndex++ )
            {
                positiveInstanceProbability[positiveSampleIndex] =     sigmoid( positiveHypothesis[positiveSampleIndex] );
                likeliHood *= ( 1- positiveInstanceProbability[positiveSampleIndex] );
            }

            positiveBagProbability = 1 - pow(likeliHood, 1/(float)numberOfPositiveSamples);//with geometric mean 

            negLogLikelihood = -log( positiveBagProbability + 1e-5f );

            for( int negativeSampleIndex=0; negativeSampleIndex<numberOfNegativeSamples; negativeSampleIndex++ )
            {
                negativeInstanceProbability[negativeSampleIndex] = sigmoid( negativeHypothesis[negativeSampleIndex] );    

                negLogLikelihood += -log( 1e-5f + 1 - negativeInstanceProbability[negativeSampleIndex] )/numberOfNegativeSamples;
            } //negative bag probability = instance probability
                    
            if (  ( previousNegLoglikehood - negLogLikelihood ) < MIL_STOPPING_THRESHOLD )
            {//stop selecting if no improvement
                if( m_selectorList.size() <= 2 )
                { //special 
                    selectedFeatureIndex--;
                    #pragma omp parallel for
                    for ( int positiveSampleIndex = 0; positiveSampleIndex < positiveSampleSet.Size(); positiveSampleIndex++ )
                    {
                        positiveHypothesis[positiveSampleIndex] -= positiveSamplePredictionWithDifferentWeakClassifiers[m_selectorList[selectedFeatureIndex]][positiveSampleIndex];
                    }

                    #pragma omp parallel for
                    for ( int negativeSampleIndex = 0; negativeSampleIndex < negativeSampleSet.Size(); negativeSampleIndex++ )
                    {
                        negativeHypothesis[negativeSampleIndex] -= negativeSamplePredictionWithDifferentWeakClassifiers[m_selectorList[selectedFeatureIndex]][negativeSampleIndex];
                    }        

                    m_selectorList.pop_back();
                    k++;
                    for ( ; k < order.size(); k++ )
                    {
                        if ( count( m_selectorList.begin(), m_selectorList.end(), order[k] ) == 0 )
                        {
                            m_selectorList.push_back( order[k] );
                            break;
                        }
                    }
                    if( k == order.size() )
                    {
                        abortError( __LINE__, __FILE__, "Could not find best 2 weak classifiers" );
                    }
                    //update alpha weighting for future if necessary 
                    // Update H1 = H0 + h_1
                    #pragma omp parallel for
                    for ( int positiveSampleIndex = 0; positiveSampleIndex < positiveSampleSet.Size(); positiveSampleIndex++ )
                    {
                        positiveHypothesis[positiveSampleIndex] += positiveSamplePredictionWithDifferentWeakClassifiers[m_selectorList[selectedFeatureIndex]][positiveSampleIndex];
                    }

                    #pragma omp parallel for
                    for ( int negativeSampleIndex = 0; negativeSampleIndex < negativeSampleSet.Size(); negativeSampleIndex++ )
                    {
                        negativeHypothesis[negativeSampleIndex] += negativeSamplePredictionWithDifferentWeakClassifiers[m_selectorList[selectedFeatureIndex]][negativeSampleIndex];
                    }                    
                    continue;
                }        
                else
                {
                    m_selectorList.pop_back();
                    break;
                }
            }        
            else
            {
                previousNegLoglikehood = negLogLikelihood;
            }

            //update weight
            for ( int positiveSampleIndex = 0; positiveSampleIndex < numberOfPositiveSamples; positiveSampleIndex++ )
            {
                positiveInstanceWeight[positiveSampleIndex]    = ( 1 / (float)numberOfPositiveSamples ) *
                    ( 1 - positiveBagProbability) * positiveInstanceProbability[positiveSampleIndex]/ positiveBagProbability; 
            }

            for( int negativeSampleIndex=0; negativeSampleIndex<numberOfNegativeSamples; negativeSampleIndex++ )
            {
                negativeInstanceWeight[negativeSampleIndex]    =
                    -( 1 / (float)numberOfNegativeSamples ) * negativeInstanceProbability[negativeSampleIndex]; 
            }//negative bag probability = instance probability
            
            #pragma omp parallel for            
            for ( int weakClassifierIndex = 0; weakClassifierIndex < (int)m_weakClassifierPtrList.size(); weakClassifierIndex++ )
            {
                for ( int positiveSampleIndex = 0; positiveSampleIndex < numberOfPositiveSamples; positiveSampleIndex++ )
                {
                    objectFunctionList[weakClassifierIndex] += positiveInstanceWeight[positiveSampleIndex] 
                        * positiveSamplePredictionWithDifferentWeakClassifiers[weakClassifierIndex][positiveSampleIndex];
                }

                for( int negativeSampleIndex=0; negativeSampleIndex<numberOfNegativeSamples; negativeSampleIndex++ )
                {
                    objectFunctionList[weakClassifierIndex] += negativeInstanceWeight[negativeSampleIndex] 
                        * negativeSamplePredictionWithDifferentWeakClassifiers[weakClassifierIndex][negativeSampleIndex];
                }
            }

            // pick best weak pStrongClassifierBase    
            sort_order_des( objectFunctionList, order ); //descending order (maximize the object function)

            // find best weak classifier that isn't already included
        
            for ( k = 0; k < order.size(); k++ )
            {
                if ( count( m_selectorList.begin(), m_selectorList.end(), order[k] ) == 0 )
                {
                    m_selectorList.push_back( order[k] );
                    break;
                }
            }
            if ( k == order.size() ) //all have been selected
            {
                //stop selecting if no improvement
                break;
            }
            else
            {
                //update alpha weighting for future if necessary 
                // Update H = H + h_m
                #pragma omp parallel for
                for ( int positiveSampleIndex = 0; positiveSampleIndex < positiveSampleSet.Size(); positiveSampleIndex++ )
                {
                    positiveHypothesis[positiveSampleIndex] += positiveSamplePredictionWithDifferentWeakClassifiers[m_selectorList[selectedFeatureIndex]][positiveSampleIndex];
                }

                #pragma omp parallel for
                for ( int negativeSampleIndex = 0; negativeSampleIndex < negativeSampleSet.Size(); negativeSampleIndex++ )
                {
                    negativeHypothesis[negativeSampleIndex] += negativeSamplePredictionWithDifferentWeakClassifiers[m_selectorList[selectedFeatureIndex]][negativeSampleIndex];
                }        
            }        
        }

        if (m_selectorList.size() == 1)
            LOG_CONSOLE( endl );

        //store the selected feature index if the feature history is enabled
        if ( m_milAnyBoostClassifierParametersPtr->m_storeFeatureHistory )
        {
            if ( MultipleCameraTracking::g_verboseMode )
            {
                //store the selected feature index if the feature history is enabled
                //cvDestroyAllWindows();
                LOG_CONSOLE( endl << "Selected features[" << m_selectorList.size() <<  "] : " );
            }

            for ( uint selectedFeatureIndex = 0; selectedFeatureIndex < m_selectorList.size(); selectedFeatureIndex++  )
            {
                //visualize the features
                //m_featureVectorPtr->GetFeature(m_selectorList[selectedFeatureIndex])->ToVisualize( m_selectorList[selectedFeatureIndex] );
                if ( MultipleCameraTracking::g_verboseMode )
                {
                    LOG_CONSOLE( m_selectorList[selectedFeatureIndex] << "," );
                }
            }
        }

        m_counter++;

        //stop the stop watch
        m_classifierStopWatch.Stop();

        return;
    }

    /****************************************************************
    MILAnyBoostClassifier::Classify
    Exceptions:
        None
    ****************************************************************/
    vectorf    MILAnyBoostClassifier::Classify(Classifier::SampleSet& sampleSet, bool isLogRatioEnabled)
    {
        int numberOfSamples = sampleSet.Size();
        vectorf responseList(numberOfSamples);
        
        // Compute features
        if ( !sampleSet.IsFeatureComputed( ) ) 
        {
            m_featureVectorPtr->Compute( sampleSet );
        }

        // for each selector, accumulate in the responseList
        for    ( uint selectedFeatureIndex = 0; selectedFeatureIndex<m_selectorList.size(); selectedFeatureIndex++ )
        {
            vectorf weaklyClassifiedLabelList;
            weaklyClassifiedLabelList = m_weakClassifierPtrList[m_selectorList[selectedFeatureIndex]]->ClassifySetF( sampleSet );

            #pragma omp parallel for
            for    ( int sampleIndex=0; sampleIndex<numberOfSamples; sampleIndex++ )
            {
                responseList[sampleIndex] += weaklyClassifiedLabelList[sampleIndex];
            }
        }

        // return probabilities or log odds ratio
        if ( !isLogRatioEnabled )
        {
            #pragma omp parallel for
            for( int j=0; j<(int)responseList.size(); j++ )
            {
                responseList[j] = sigmoid(responseList[j]);
            }
        }

        return responseList;
    }
}