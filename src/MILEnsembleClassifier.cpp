#include "MILEnsembleClassifier.h"
#include "CommonMacros.h"

#include <algorithm>
#include <numeric>

namespace Classifier
{
    /****************************************************************
    MILEnsembleClassifier::Update
    Exceptions:
        None
    ****************************************************************/
    void    MILEnsembleClassifier::Update( Classifier::SampleSet& positiveSampleSet, Classifier::SampleSet& negativeSampleSet)
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

        //used for termination
        double previousNegLoglikehood = 1000000;

        // initialize H - hypothesis
        vectorf positiveHypothesis;
        vectorf negativeHypothesis;
        positiveHypothesis.clear(); 
        negativeHypothesis.clear();
        positiveHypothesis.resize( positiveSampleSet.Size( ), 0.0f );
        negativeHypothesis.resize( negativeSampleSet.Size( ), 0.0f );

        //retain the best weak classifiers from the previous time instance
        RetainBestPerformingWeakClassifiers( positiveSampleSet, negativeSampleSet, positiveHypothesis, negativeHypothesis );

        vector<vectorf> positiveSamplePredictionWithDifferentWeakClassifiers( m_weakClassifierPtrList.size( ) );
        vector<vectorf> negativeSamplePredictionWithDifferentWeakClassifiers( m_weakClassifierPtrList.size( ) );

        // train all weak classifiers without weights and find the prediction for positive and negative samples
        #pragma omp parallel for
        for ( int featureIndex = 0; featureIndex < m_MILEnsembleClassifierParametersPtr->m_totalNumberOfWeakClassifiers; featureIndex++ )
        {
            //check whether the index has already been retained
            if ( count( m_selectorList.begin(), m_selectorList.end(), featureIndex ) != 0 )
            {
                positiveSamplePredictionWithDifferentWeakClassifiers[featureIndex] = m_weakClassifierPtrList[featureIndex]->ClassifySetF( positiveSampleSet );
                negativeSamplePredictionWithDifferentWeakClassifiers[featureIndex] = m_weakClassifierPtrList[featureIndex]->ClassifySetF( negativeSampleSet );

                continue;
            }

            //always clear-up the remaining weak classifier before selecting for Ensemble
            m_weakClassifierPtrList[featureIndex]->Initialize();

            m_weakClassifierPtrList[featureIndex]->Update( positiveSampleSet, negativeSampleSet );
            positiveSamplePredictionWithDifferentWeakClassifiers[featureIndex] = m_weakClassifierPtrList[featureIndex]->ClassifySetF( positiveSampleSet );
            negativeSamplePredictionWithDifferentWeakClassifiers[featureIndex] = m_weakClassifierPtrList[featureIndex]->ClassifySetF( negativeSampleSet );
        }

        // pick the best features
        #pragma omp parallel for
        for ( int selectedFeatureIndex = m_selectorList.size( ); selectedFeatureIndex < m_MILEnsembleClassifierParametersPtr->m_numberOfSelectedWeakClassifiers; selectedFeatureIndex++ )
        {
            // Compute errors/negLogLikelihoodList for all weak classifiers
            vectorf positiveLikelihoodList( m_weakClassifierPtrList.size( ), 1.0f );
            vectorf negativeLikelihoodList( m_weakClassifierPtrList.size( ) );
            vectorf negLogLikelihoodList( m_weakClassifierPtrList.size( ) );

            #pragma omp parallel for
            for ( int weakClassifierIndex = 0; weakClassifierIndex < (int)m_weakClassifierPtrList.size(); weakClassifierIndex++ )
            {
                double likeliHood = 1.0f;

                //#pragma omp parallel for reduction(*: likeliHood)
                for ( int positiveSampleIndex = 0; positiveSampleIndex < numberOfPositiveSamples; positiveSampleIndex++ )
                {
                    likeliHood *= ( 1-sigmoid( positiveHypothesis[positiveSampleIndex] + 
                                        positiveSamplePredictionWithDifferentWeakClassifiers[weakClassifierIndex][positiveSampleIndex] 
                                        )
                                  );
                }
                positiveLikelihoodList[weakClassifierIndex] = (float)-log( 1-likeliHood + 1e-5);

                likeliHood=0.0f;
                for( int j=0; j<numberOfNegativeSamples; j++ )
                {
                    likeliHood += (float)-log(1e-5f+1-sigmoid(negativeHypothesis[j]+negativeSamplePredictionWithDifferentWeakClassifiers[weakClassifierIndex][j]));
                }
                negativeLikelihoodList[weakClassifierIndex] = likeliHood;

                negLogLikelihoodList[weakClassifierIndex] = positiveLikelihoodList[weakClassifierIndex]/numberOfPositiveSamples + negativeLikelihoodList[weakClassifierIndex]/numberOfNegativeSamples;
            }

            // pick best weak pStrongClassifierBase
            vectori order;
            sort_order( negLogLikelihoodList, order );//ascending order (minimize the negative log likelihood)


            // find best weak classifier that isn't already included
            uint k = 0;
            for ( ; k < order.size(); k++ )
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
                previousNegLoglikehood = negLogLikelihoodList[ order[k] ];
                //TODO - need to find alpha by Line Search

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


        m_counter++;

        //stop the stop watch
        m_classifierStopWatch.Stop();

        return;
    }

    /****************************************************************
    MILEnsembleClassifier::Classify
    Exceptions:
        None
    ****************************************************************/
    vectorf    MILEnsembleClassifier::Classify(Classifier::SampleSet& sampleSet, bool shouldNotUseSigmoid)
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
        if ( !shouldNotUseSigmoid )
        {
            #pragma omp parallel for
            for( int j=0; j<(int)responseList.size(); j++ )
            {
                responseList[j] = sigmoid(responseList[j]);
            }
        }

        return responseList;
    }


    /****************************************************************
    MILEnsembleClassifier::RetainBestPerformingWeakClassifiers
    Exceptions:
        None
    ****************************************************************/
    void    MILEnsembleClassifier::RetainBestPerformingWeakClassifiers( Classifier::SampleSet&    positiveSampleSet,
                                                                        Classifier::SampleSet&    negativeSampleSet,
                                                                        vectorf&                positiveHypothesis, 
                                                                        vectorf&                negativeHypothesis )
    {
        try
        {
            ASSERT_TRUE( positiveSampleSet.Size() > 0 && negativeSampleSet.Size() > 0 );

            const int numberOfWeakClassifiersToRetain = m_MILEnsembleClassifierParametersPtr->m_percentageOfRetainedWeakClassifiers * m_selectorList.size();

            //if number of features to retain is 0, clear the selected list and return immediately
            if ( m_selectorList.empty( ) )
            {
                return;
            }

            if ( numberOfWeakClassifiersToRetain == 0 )
            {
                m_selectorList.clear();         
                return;
            }

            //get the number of positive and negative samples
            size_t numberOfNegativeSamples = negativeSampleSet.Size();
            size_t numberOfPositiveSamples = positiveSampleSet.Size();

            //copy the selected list to a local variable
            vectori previouslySelectedFeatureList = m_selectorList;

            //clear the selected feature list
            m_selectorList.clear( );

            ASSERT_TRUE( !m_weakClassifierPtrList.empty() );
            
            double previousNegLoglikehood = 1000000;

            vector<vectorf> positiveSamplePredictionWithDifferentWeakClassifiers( previouslySelectedFeatureList.size( ) );
            vector<vectorf> negativeSamplePredictionWithDifferentWeakClassifiers( previouslySelectedFeatureList.size( ) );

            // train all weak classifiers without weights and find the prediction for positive and negative samples
            #pragma omp parallel for
            for ( int previouslySelectedFeatureIndex = 0; previouslySelectedFeatureIndex < previouslySelectedFeatureList.size(); previouslySelectedFeatureIndex++ )
            {
                WeakClassifierBase* previousWeakClassifier = m_weakClassifierPtrList[previouslySelectedFeatureList[previouslySelectedFeatureIndex]];

                positiveSamplePredictionWithDifferentWeakClassifiers[previouslySelectedFeatureIndex] = previousWeakClassifier->ClassifySetF( positiveSampleSet );
                negativeSamplePredictionWithDifferentWeakClassifiers[previouslySelectedFeatureIndex] = previousWeakClassifier->ClassifySetF( negativeSampleSet );
            }

            // pick the best features
            #pragma omp parallel for
            for ( int selectedFeatureIndex = 0; selectedFeatureIndex < numberOfWeakClassifiersToRetain; selectedFeatureIndex++ )
            {
                // Compute errors/negLogLikelihoodList for all weak classifiers
                vectorf positiveBagNegLogLikelihoodList( previouslySelectedFeatureList.size( ), 1.0f );
                vectorf negativeBagNegLogLikelihoodList( previouslySelectedFeatureList.size( ) );
                vectorf negLogLikelihoodList( previouslySelectedFeatureList.size( ) );

                #pragma omp parallel for
                for ( int newSelectionIndex = 0; newSelectionIndex < (int)previouslySelectedFeatureList.size(); newSelectionIndex++ )
                {
                    float likeliHood = 1.0f;

                    #pragma omp parallel for
                    for ( int positiveSampleIndex = 0; positiveSampleIndex < numberOfPositiveSamples; positiveSampleIndex++ )
                    {
                        // find the likelihood with H + h_m
                        likeliHood *=    ( 1-sigmoid( positiveHypothesis[positiveSampleIndex] + 
                            positiveSamplePredictionWithDifferentWeakClassifiers[newSelectionIndex][positiveSampleIndex] 
                                                    )
                                        );
                    }
                    positiveBagNegLogLikelihoodList[newSelectionIndex] = (float)-log( 1-likeliHood + 1e-5);

                    likeliHood=0.0f;
                    for( int j=0; j < numberOfNegativeSamples; j++ )
                    {
                        likeliHood += (float)-log(1e-5f+1-sigmoid(negativeHypothesis[j]+negativeSamplePredictionWithDifferentWeakClassifiers[newSelectionIndex][j]));
                    }
                    negativeBagNegLogLikelihoodList[newSelectionIndex] = likeliHood;

                    negLogLikelihoodList[newSelectionIndex] = positiveBagNegLogLikelihoodList[newSelectionIndex]/numberOfPositiveSamples + negativeBagNegLogLikelihoodList[newSelectionIndex]/numberOfNegativeSamples;
                }

                // pick best weak pStrongClassifierBase
                vectori order;
                sort_order( negLogLikelihoodList, order ); //ascending order (minimize the negative log likelihood)

                // find best weak classifier that isn't already included
                uint k = 0;
                for (; k < order.size(); k++ )
                {
                    if ( count( m_selectorList.begin(), m_selectorList.end(), previouslySelectedFeatureList[order[k]] ) == 0 )
                    {
                        m_selectorList.push_back( previouslySelectedFeatureList[order[k]] );
                        break;
                    }
                }    

                if ( k == order.size() )
                {
                    break;
                }

                if (  ( previousNegLoglikehood - negLogLikelihoodList[k] ) < MIL_STOPPING_THRESHOLD )
                {
                    //stop selecting if no improvement
                    m_selectorList.pop_back();
                    break;
                }
                else
                {  
                    previousNegLoglikehood = negLogLikelihoodList[ order[k] ];
                    //TODO - need to find alpha by Line Search

                    // Update H = H + h_m
                    #pragma omp parallel for
                    for ( int positiveSampleIndex = 0; positiveSampleIndex < positiveSampleSet.Size(); positiveSampleIndex++ )
                    {
                        positiveHypothesis[positiveSampleIndex] += positiveSamplePredictionWithDifferentWeakClassifiers[order[k]][positiveSampleIndex];
                    }

                    #pragma omp parallel for
                    for ( int negativeSampleIndex = 0; negativeSampleIndex < negativeSampleSet.Size(); negativeSampleIndex++ )
                    {
                        negativeHypothesis[negativeSampleIndex] += negativeSamplePredictionWithDifferentWeakClassifiers[order[k]][negativeSampleIndex];
                    }    
                }
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Error While retaining best performing weak classifiers" );
    }

    /****************************************************************
    MILEnsembleClassifier::UpdateWithAdaptiveWeighting
    Update the classifier. The function does it efficiency by adaptively 
    weight samples after each selection loop to avoid repeated evaluation 
    of strong classifier hypothesis on training examples
    Exceptions:
        None
    ****************************************************************/
    void    MILEnsembleClassifier::UpdateWithAdaptiveWeighting( Classifier::SampleSet& positiveSampleSet, Classifier::SampleSet& negativeSampleSet )
    {
    

    }

    /****************************************************************
    MILEnsembleClassifier::RetainBestPerformingWeakClassifiersWithAdaptiveWeighting
    Exceptions:
        None
    ****************************************************************/
    void MILEnsembleClassifier::RetainBestPerformingWeakClassifiersWithAdaptiveWeighting( Classifier::SampleSet& positiveSampleSet,
        Classifier::SampleSet&    negativeSampleSet,
        vectorf&                positiveHypothesis, 
        vectorf&                negativeHypothesis )
    {

    }

}