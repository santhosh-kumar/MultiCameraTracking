#include "StrongClassifierBase.h"

namespace Classifier
{
    /****************************************************************
    StrongClassifierBase
    ****************************************************************/
    StrongClassifierBase::StrongClassifierBase( StrongClassifierParametersBasePtr strongClassifierParametersBasePtr )
        : m_strongClassifierParametersBasePtr( strongClassifierParametersBasePtr ),
        m_featureVectorPtr( ),
        m_classifierStopWatch( ),
        m_numberOfSamples( 0 ),
        m_counter( 0 )
    {
        try
        {
            ASSERT_TRUE( m_strongClassifierParametersBasePtr != NULL );

            //set the number of features to be selected by the weak classifier
            if ( m_strongClassifierParametersBasePtr->m_numberOfSelectedWeakClassifiers > m_strongClassifierParametersBasePtr->m_totalNumberOfWeakClassifiers  || m_strongClassifierParametersBasePtr->m_numberOfSelectedWeakClassifiers < 1 )
            {
                m_strongClassifierParametersBasePtr->m_numberOfSelectedWeakClassifiers = m_strongClassifierParametersBasePtr->m_totalNumberOfWeakClassifiers/2;
            }

            //initialize the feature vector
            InitializeFeatureVector( );

            ASSERT_TRUE( m_featureVectorPtr != NULL );

            //initialize the weak classifiers
            InitializeWeakClassifiers( );

            ASSERT_TRUE( !m_weakClassifierPtrList.empty() );

            //store the feature history
            if ( m_strongClassifierParametersBasePtr->m_storeFeatureHistory )
            {
                m_featureVectorPtr->SaveVisualizedFeatureVector( "Haar Features" );
            }

        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to Construct StrongClassifierBase" );
    }

    /****************************************************************
    InitializeFeatureVector

    Exceptions:
        None
    ****************************************************************/
    void StrongClassifierBase::InitializeFeatureVector( )
    {
        try
        {
            ASSERT_TRUE( m_strongClassifierParametersBasePtr != NULL );

            //create the feature vector according to the feature type
            if ( m_strongClassifierParametersBasePtr->m_featureParametersPtr->GetFeatureType() == Features::HAAR_LIKE )
            {
                m_featureVectorPtr = Features::FeatureVectorPtr(new Features::HaarFeatureVector( ));
            }
            else if ( m_strongClassifierParametersBasePtr->m_featureParametersPtr->GetFeatureType() == Features::MULTI_DIMENSIONAL_COLOR_HISTOGRAM )
            {
                m_featureVectorPtr = Features::FeatureVectorPtr( new Features::MultiDimensionalColorHistogramFeatureVector( ) );
            }
            else if ( m_strongClassifierParametersBasePtr->m_featureParametersPtr->GetFeatureType() == Features::CULTURE_COLOR_HISTOGRAM )
            {
                m_featureVectorPtr = Features::FeatureVectorPtr( new Features::CultureColorHistogramFeatureVector( ) );
            }
            else if ( m_strongClassifierParametersBasePtr->m_featureParametersPtr->GetFeatureType() == Features::HAAR_COLOR_HISTOGRAM )
            {
                m_featureVectorPtr = Features::FeatureVectorPtr( new Features::HaarAndColorHistogramFeatureVector( ) );
            }
            else
            {
                abortError( __LINE__, __FILE__, "Unsupported Feature Type" );
            }

            //Generate feature vectors
            ASSERT_TRUE( m_featureVectorPtr != NULL );

            //Generate Features
            m_featureVectorPtr->Generate( m_strongClassifierParametersBasePtr->m_featureParametersPtr );
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to initialize the feature vector" );
    }

    /****************************************************************
    InitializeWeakClassifiers

    Exceptions:
        None
    ****************************************************************/
    void StrongClassifierBase::InitializeWeakClassifiers( )
    {
        try
        {
            ASSERT_TRUE( m_strongClassifierParametersBasePtr != NULL );

            m_selectorList.resize( m_strongClassifierParametersBasePtr->m_numberOfSelectedWeakClassifiers, 0 );
            m_weakClassifierPtrList.resize( m_strongClassifierParametersBasePtr->m_totalNumberOfWeakClassifiers );

            for ( int featureIndex = 0; featureIndex < m_strongClassifierParametersBasePtr->m_totalNumberOfWeakClassifiers; featureIndex++ )
            {
                if ( m_strongClassifierParametersBasePtr->m_weakClassifierType == STUMP )
                {
                    m_weakClassifierPtrList[featureIndex] = new OnlineStumpsWeakClassifier(featureIndex);
                    m_weakClassifierPtrList[featureIndex]->SetLearningRate( m_strongClassifierParametersBasePtr->m_learningRate );
                }
                else if ( m_strongClassifierParametersBasePtr->m_weakClassifierType == WEIGHTED_STUMP )
                {
                    m_weakClassifierPtrList[featureIndex] = new WeightedStumpsWeakClassifier(featureIndex);
                    m_weakClassifierPtrList[featureIndex]->SetLearningRate( m_strongClassifierParametersBasePtr->m_learningRate );
                }
                else if ( m_strongClassifierParametersBasePtr->m_weakClassifierType == PERCEPTRON )
                {
                    m_weakClassifierPtrList[featureIndex] = new PerceptronWeakClassifier(featureIndex);
                    m_weakClassifierPtrList[featureIndex]->SetLearningRate( m_strongClassifierParametersBasePtr->m_learningRate );
                }
                else
                {
                    abortError(__LINE__,__FILE__,"incorrect weak pStrongClassifierBase name");
                }
            }

            ASSERT_TRUE( !m_weakClassifierPtrList.empty() );
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to initialize the weak classifiers" );
    }
}