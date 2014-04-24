#include "MILBoostClassifier.h"
#include "CommonMacros.h"

#include <algorithm>
#include <numeric>

namespace Classifier
{
    /****************************************************************
    MILBoostClassifier::Update
    One positive bag
    Exceptions:
        None
    ****************************************************************/
    void    MILBoostClassifier::Update( Classifier::SampleSet& positiveSampleSet, Classifier::SampleSet& negativeSampleSet)
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

        vectorf posw( positiveSampleSet.Size() );
        vectorf negw( negativeSampleSet.Size() );
        vector<vectorf> positiveSamplePredictionWithDifferentWeakClassifiers( m_weakClassifierPtrList.size( ) );
        vector<vectorf> negativeSamplePredictionWithDifferentWeakClassifiers( m_weakClassifierPtrList.size( ) );

        // train all weak classifiers without weights and find the prediction for positive and negative samples
        #pragma omp parallel for
        for ( int featureIndex = 0; featureIndex < m_MILBoostClassifierParametersPtr->m_totalNumberOfWeakClassifiers; featureIndex++ )
        {
            m_weakClassifierPtrList[featureIndex]->Update( positiveSampleSet, negativeSampleSet );
            positiveSamplePredictionWithDifferentWeakClassifiers[featureIndex] = m_weakClassifierPtrList[featureIndex]->ClassifySetF( positiveSampleSet );
            negativeSamplePredictionWithDifferentWeakClassifiers[featureIndex] = m_weakClassifierPtrList[featureIndex]->ClassifySetF( negativeSampleSet );
        }

        //used for termination
        double previousNegLoglikehood = 1000000;

        // pick the best features
        for ( int selectedFeatureIndex = 0; selectedFeatureIndex < m_MILBoostClassifierParametersPtr->m_numberOfSelectedWeakClassifiers; selectedFeatureIndex++ )
        {
            // Compute errors/negLogLikelihoodList for all weak classifiers
            vectorf positiveLikelihoodList( m_weakClassifierPtrList.size( ), 1.0f );
            vectorf negativeLikelihoodList( m_weakClassifierPtrList.size( ) );
            vectorf negLogLikelihoodList( m_weakClassifierPtrList.size( ) );

            #pragma omp parallel for
            for ( int weakClassifierIndex = 0; weakClassifierIndex < (int)m_weakClassifierPtrList.size(); weakClassifierIndex++ )
            {
                float likeliHood = 1.0f;

                for ( int positiveSampleIndex = 0; positiveSampleIndex < numberOfPositiveSamples; positiveSampleIndex++ )
                {
                    likeliHood *= ( 1-sigmoid( positiveHypothesis[positiveSampleIndex] + 
                                        positiveSamplePredictionWithDifferentWeakClassifiers[weakClassifierIndex][positiveSampleIndex] ) );
                }
                positiveLikelihoodList[weakClassifierIndex] = (float)-log( 1-likeliHood + 1e-5);

                likeliHood=0.0f;
                for( int negativeSampleIndex=0; negativeSampleIndex<numberOfNegativeSamples; negativeSampleIndex++ )
                {
                    likeliHood += (float)-log(1e-5f+1-sigmoid(negativeHypothesis[negativeSampleIndex]+negativeSamplePredictionWithDifferentWeakClassifiers[weakClassifierIndex][negativeSampleIndex]));
                }
                negativeLikelihoodList[weakClassifierIndex] = likeliHood;

                negLogLikelihoodList[weakClassifierIndex] = positiveLikelihoodList[weakClassifierIndex]/numberOfPositiveSamples + negativeLikelihoodList[weakClassifierIndex]/numberOfNegativeSamples;
            }

            // pick best weak pStrongClassifierBase
            vectori order;        
            sort_order( negLogLikelihoodList, order ); //ascending order (minimize the negative log likelihood)

            // find best weak classifier that isn't already included
            uint k = 0;
            for (; k < order.size(); k++ )
            {
                if ( count( m_selectorList.begin(), m_selectorList.end(), order[k] ) == 0 && m_weakClassifierPtrList[order[k]]->IsValidWeakClassifier( ) )
                {
                    m_selectorList.push_back( order[k] );
                    break;
                }
            }
            if (  ( previousNegLoglikehood - negLogLikelihoodList[k] ) < MIL_STOPPING_THRESHOLD )
            {
                //stop selecting if no improvement
                m_selectorList.pop_back();
                break;
            }
            else
            {
                previousNegLoglikehood = negLogLikelihoodList[k];
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

        //store the selected feature index if the feature history is enabled
        if ( m_MILBoostClassifierParametersPtr->m_storeFeatureHistory )
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
                LOG_CONSOLE( m_selectorList[selectedFeatureIndex] << "," );
            }
        }

        m_counter++;

        //stop the stop watch
        m_classifierStopWatch.Stop();

        return;
    }

    
        /****************************************************************
    MILBoostClassifier::Update
    Multiple positive bags
    Exceptions:
        None
    ****************************************************************/
    void    MILBoostClassifier::Update( Classifier::SampleSet& positiveSampleSet, Classifier::SampleSet& negativeSampleSet,int numPositiveBags )
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
         
        ASSERT_TRUE( numPositiveBags > 0 );
        
        // initialize H - hypothesis
        vectorf positiveHypothesis;
        vectorf negativeHypothesis;
        positiveHypothesis.clear(); 
        negativeHypothesis.clear();
        positiveHypothesis.resize( positiveSampleSet.Size( ), 0.0f );
        negativeHypothesis.resize( negativeSampleSet.Size( ), 0.0f );

        //clear the selected feature list
        m_selectorList.clear();

        vectorf posw( positiveSampleSet.Size() );
        vectorf negw( negativeSampleSet.Size() );
        vector<vectorf> positiveSamplePredictionWithDifferentWeakClassifiers( m_weakClassifierPtrList.size( ) );
        vector<vectorf> negativeSamplePredictionWithDifferentWeakClassifiers( m_weakClassifierPtrList.size( ) );

        // train all weak classifiers without weights and find the prediction for positive and negative samples
        #pragma omp parallel for
        for ( int featureIndex = 0; featureIndex < m_MILBoostClassifierParametersPtr->m_totalNumberOfWeakClassifiers; featureIndex++ )
        {
            m_weakClassifierPtrList[featureIndex]->Update( positiveSampleSet, negativeSampleSet );
            positiveSamplePredictionWithDifferentWeakClassifiers[featureIndex] = m_weakClassifierPtrList[featureIndex]->ClassifySetF( positiveSampleSet );
            negativeSamplePredictionWithDifferentWeakClassifiers[featureIndex] = m_weakClassifierPtrList[featureIndex]->ClassifySetF( negativeSampleSet );
        }

        //used for termination
        double previousNegLoglikehood = 1000000;

        // pick the best features
        for ( int selectedFeatureIndex = 0; selectedFeatureIndex < m_MILBoostClassifierParametersPtr->m_numberOfSelectedWeakClassifiers; selectedFeatureIndex++ )
        {
            // Compute errors/negLogLikelihoodList for all weak classifiers
            vectorf positiveLikelihoodList( m_weakClassifierPtrList.size( ), 1.0f );
            vectorf negativeLikelihoodList( m_weakClassifierPtrList.size( ) );
            vectorf negLogLikelihoodList( m_weakClassifierPtrList.size( ) );

            #pragma omp parallel for
            for ( int weakClassifierIndex = 0; weakClassifierIndex < (int)m_weakClassifierPtrList.size(); weakClassifierIndex++ )
            {
                positiveLikelihoodList[weakClassifierIndex] = 0.0f;

                int positiveSampleIndex = 0;
                int cameraID = positiveSampleSet[positiveSampleIndex].m_cameraID;

                for ( int positiveBagIndex = 1; positiveBagIndex <= numPositiveBags; positiveBagIndex++)
                {
                    float likeliHood = 1.0f;
                    
                    int numberOfSamplesInABag = 0;
                    for ( ; positiveSampleIndex < numberOfPositiveSamples; positiveSampleIndex++ )
                    {
                        if( positiveSampleSet[positiveSampleIndex].m_cameraID != cameraID )
                        {
                            cameraID = positiveSampleSet[positiveSampleIndex].m_cameraID;
                            break;
                        }
                        else
                        {
                            numberOfSamplesInABag++;
                        }

                        likeliHood *= ( 1-sigmoid( positiveHypothesis[positiveSampleIndex] + 
                            positiveSamplePredictionWithDifferentWeakClassifiers[weakClassifierIndex][positiveSampleIndex] ) );

                    }

                    positiveLikelihoodList[weakClassifierIndex] = (float)-log( 1-likeliHood + 1e-5)/numberOfSamplesInABag + 
                        positiveLikelihoodList[weakClassifierIndex];
                }                                

                float likeliHood=0.0f;
                for( int negativeSampleIndex=0; negativeSampleIndex<numberOfNegativeSamples; negativeSampleIndex++ )
                {
                    likeliHood += (float)-log(1e-5f+1-sigmoid(negativeHypothesis[negativeSampleIndex]+negativeSamplePredictionWithDifferentWeakClassifiers[weakClassifierIndex][negativeSampleIndex]));
                }
                negativeLikelihoodList[weakClassifierIndex] = likeliHood;

                negLogLikelihoodList[weakClassifierIndex] = positiveLikelihoodList[weakClassifierIndex]/numPositiveBags + negativeLikelihoodList[weakClassifierIndex]/numberOfNegativeSamples;
            }

            // pick best weak pStrongClassifierBase
            vectori order;        
            sort_order( negLogLikelihoodList, order ); //ascending order (minimize the negative log likelihood)

            // find best weak classifier that isn't already included
            uint k = 0;
            for (; k < order.size(); k++ )
            {
                if ( count( m_selectorList.begin(), m_selectorList.end(), order[k] ) == 0 )
                {
                    m_selectorList.push_back( order[k] );
                    break;
                }
            }
            if (  ( previousNegLoglikehood - negLogLikelihoodList[k] ) < MIL_STOPPING_THRESHOLD )
            {
                //stop selecting if no improvement
                m_selectorList.pop_back();
                break;
            }
            else
            {
                previousNegLoglikehood = negLogLikelihoodList[k];
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

        //store the selected feature index if the feature history is enabled
        if ( m_MILBoostClassifierParametersPtr->m_storeFeatureHistory )
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
                LOG_CONSOLE( m_selectorList[selectedFeatureIndex] << "," );
            }
        }

        m_counter++;

        //stop the stop watch
        m_classifierStopWatch.Stop();

        return;
    }


    /****************************************************************
    MILBoostClassifier::Classify
    Exceptions:
        None
    ****************************************************************/
    vectorf    MILBoostClassifier::Classify(Classifier::SampleSet& sampleSet, bool isLogRatioEnabled)
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