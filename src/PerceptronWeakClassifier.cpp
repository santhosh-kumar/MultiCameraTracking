#include "PerceptronWeakClassifier.h"

#define FEATURE_DIMENSION                            1
#define MAXIMUM_NUMBER_OF_ITERATIONS                100
#define NEGATIVE_EXAMPLE_LABEL                        0
#define POSITIVE_EXAMPLE_LABEL                        1
#define PERCEPTRON_LEARNING_RATE                    0.1
#define PERCEPTRON_THRESHOLD                        0.5

namespace Classifier
{
    /****************************************************************
    PerceptronWeakClassifier
        Constructor
    ****************************************************************/
    PerceptronWeakClassifier::PerceptronWeakClassifier( ) 
        : WeakClassifierBase( ) 
    { 
        try
        {
            Initialize( );
        }
        EXCEPTION_CATCH_AND_ABORT( "Error While Creating Perceptron Based Weak Classifier" );
    }

    /****************************************************************
    PerceptronWeakClassifier
        Constructor
    ****************************************************************/
    PerceptronWeakClassifier::PerceptronWeakClassifier( const int featureId )        
        : WeakClassifierBase( featureId )
    {
        try
        {
            Initialize( );
        }
        EXCEPTION_CATCH_AND_ABORT( "Error While Creating Perceptron Based Weak Classifier" );
    }

    /****************************************************************
    PerceptronWeakClassifier::Initialize
        Initializes the parameters
    Exceptions:
        None
    ****************************************************************/
    void PerceptronWeakClassifier::Initialize()
    {
        try
        {
            m_weightList = vectorf( FEATURE_DIMENSION + 1 );
        }
        EXCEPTION_CATCH_AND_ABORT( "Error While Creating Perceptron Based Weak Classifier" );
    }

    /****************************************************************
    PerceptronWeakClassifier::Classify
        Classify 
    Exceptions:
        None
    ****************************************************************/
    bool    PerceptronWeakClassifier::Classify( const Classifier::SampleSet& sampleSet, const int sampleIndex )
    {
        try
        {
            return ( ClassifyF( sampleSet,sampleIndex ) > 0 );
        }
        EXCEPTION_CATCH_AND_ABORT( "Error while classifying perceptron based classifier" );
    }

    /****************************************************************
    PerceptronWeakClassifier::ClassifyF
        ClassifyF
    Exceptions:
        None
    ****************************************************************/
    float    PerceptronWeakClassifier::ClassifyF( const Classifier::SampleSet& sampleSet, const int sampleIndex )
    {
        try
        {
            double featureValue    = GetFeatureValue( sampleSet, sampleIndex );

            ASSERT_TRUE( m_weightList.size( ) == 2 );
            double response = featureValue * m_weightList[0] > m_weightList[1] ? 1 : -1;

            return static_cast<float>( response );
        }
        EXCEPTION_CATCH_AND_ABORT( "Error while classifying perceptron based classifier" );
    }


    /****************************************************************
    PerceptronWeakClassifier::Update
        Update
    Exceptions:
        None
    ****************************************************************/
    void    PerceptronWeakClassifier::Update(  const Classifier::SampleSet&    positiveSampleSet,
                                               const Classifier::SampleSet&    negativeSampleSet, 
                                               vectorf*            /*pPositiveSamplesWeightList*/, 
                                               vectorf*            /*pNegativeSamplesWeightList*/ )
    {
        try
        {
            ASSERT_TRUE( positiveSampleSet.Size() > 0 && negativeSampleSet.Size() > 0 );

            unsigned int numberOfIterations = 0;
            m_weightList    = vectorf( FEATURE_DIMENSION+1, 0.0f );
            m_weightList[1] = PERCEPTRON_THRESHOLD;

            unsigned int totalNumberOfSamples = positiveSampleSet.Size() + negativeSampleSet.Size();            
            int error;
            int maximumAllowedErrorCount    = min( positiveSampleSet.Size(), negativeSampleSet.Size() );
            int previousErrorCount            = maximumAllowedErrorCount;

            while ( numberOfIterations < MAXIMUM_NUMBER_OF_ITERATIONS )
            {
                unsigned int errorCount = 0;

                for ( int positiveSampleIndex = 0; positiveSampleIndex < positiveSampleSet.Size(); positiveSampleIndex++ )
                {
                    double featureValue    = GetFeatureValue( positiveSampleSet, positiveSampleIndex );
                    int result = (featureValue * m_weightList[0] ) > m_weightList[1] ? POSITIVE_EXAMPLE_LABEL : NEGATIVE_EXAMPLE_LABEL;
                    error = POSITIVE_EXAMPLE_LABEL - result;
                    if ( error != 0 )
                    {
                        m_weightList[0] = m_weightList[0] + PERCEPTRON_LEARNING_RATE * error * featureValue;
                        errorCount++;
                    }
                }

                for ( int negativeSampleIndex = 0; negativeSampleIndex < negativeSampleSet.Size(); negativeSampleIndex++ )
                {
                    double featureValue    = GetFeatureValue( negativeSampleSet, negativeSampleIndex );
                    double response = featureValue * m_weightList[0];
                    int result = (featureValue * m_weightList[0] ) > m_weightList[1] ? POSITIVE_EXAMPLE_LABEL : NEGATIVE_EXAMPLE_LABEL;
                    error = NEGATIVE_EXAMPLE_LABEL - result;
                    if ( error != 0 )
                    {
                        m_weightList[0] = m_weightList[0] + PERCEPTRON_LEARNING_RATE * error * featureValue;
                        errorCount++;
                    }
                }

                float errorRateImprovement = ( static_cast<float>( previousErrorCount - errorCount ) / totalNumberOfSamples );

                if ( errorCount < maximumAllowedErrorCount &&  errorRateImprovement < 0.01 )
                {
                    break;
                }

                previousErrorCount = errorCount;
                numberOfIterations++;
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Error while classifying perceptron based classifier" );
    }
}