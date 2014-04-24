#include "AppearanceBasedInformationFuser.h"
#include "StrongClassifierFactory.h"
#include "CameraNetworkBase.h"
#include "Config.h"

#define RANDOM_SAMPLE_SELECTION_THRESHOLD 0.5f
#define MULTIPLE_POSITIVE_BAG 0

namespace MultipleCameraTracking
{
    /********************************************************************
    AppearanceBasedInformationFuser
        AppearanceBasedInformationFuser fuses information from multiple cameras.
    Exceptions:
        None
    *********************************************************************/
    AppearanceBasedInformationFuser::AppearanceBasedInformationFuser(    const int                                            objectId,
                                                                        const int                                            cameraId,
                                                                        const AppearanceFusionType                            appearanceFusionType,
                                                                        Classifier::StrongClassifierParametersBasePtr        strongClassifierParametersBasePtr )
        : m_objectId( objectId ),
        m_cameraId( cameraId ),
        m_appearanceFusionType( appearanceFusionType ),
        m_strongClassifierParametersBasePtr( strongClassifierParametersBasePtr )
    {
        try
        {
            strongClassifierParametersBasePtr->m_learningRate = 0;

            m_strongClassifierBasePtr = Classifier::StrongClassifierFactory::CreateAndInitializeClassifier( ( strongClassifierParametersBasePtr ) );

            ASSERT_TRUE( m_strongClassifierBasePtr.get() != NULL );
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to Initialize the AppearanceBasedInformationFuser");
    }

    /********************************************************************
    LearnAppearanceModel
        Learn the appearance model using the positive and negative 
        samples provided by different cameras.
    Exceptions:
        None
    *********************************************************************/
    void AppearanceBasedInformationFuser::LearnGlobalAppearanceModel( CameraNetworkBasePtr cameraNetworkBasePtr )
    {
        try
        {
            ASSERT_TRUE( m_strongClassifierBasePtr != NULL );
            ASSERT_TRUE( cameraNetworkBasePtr != NULL );

            Classifier::SampleSet positiveSampleSet;
            Classifier::SampleSet negativeSampleSet;

            LOG( "Generating Training Sample Sets for Appearance Fusion" );

            cameraNetworkBasePtr->GenerateTrainingSampleSetsForAppearanceFusion( m_objectId,
                                                                                positiveSampleSet,
                                                                                negativeSampleSet );

            ASSERT_TRUE( positiveSampleSet.Size() > 0 );
            ASSERT_TRUE( negativeSampleSet.Size() > 0 );

            Classifier::SampleSet randomPositiveSampleSet;
            Classifier::SampleSet randomNegativeSampleSet;
            if( MULTIPLE_POSITIVE_BAG == 0 )
            {//randomly remove positive and negative samples for robustness
                for ( int sampleIndex = 0; sampleIndex < positiveSampleSet.Size(); sampleIndex++ )
                {
                    if ( randfloat() > RANDOM_SAMPLE_SELECTION_THRESHOLD )
                    {
                        randomPositiveSampleSet.PushBackSample( positiveSampleSet[sampleIndex] );
                    }
                }

                for ( int sampleIndex = 0; sampleIndex < negativeSampleSet.Size(); sampleIndex++ )
                {
                    if ( randfloat() > RANDOM_SAMPLE_SELECTION_THRESHOLD )
                    {
                        randomNegativeSampleSet.PushBackSample( negativeSampleSet[sampleIndex] );
                    }
                }
                m_strongClassifierBasePtr->Update( randomPositiveSampleSet, randomNegativeSampleSet );
            }
            else
            {
                m_strongClassifierBasePtr->Update( positiveSampleSet, negativeSampleSet,5 );
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed while learning the global appearance model" );
    }

    /********************************************************************
    FuseInformation
        Fuse information by .
    Exceptions:
        None
    *********************************************************************/
    void AppearanceBasedInformationFuser::FuseInformation( Classifier::SampleSet& testSampleSet, vectorf& likelihoodProbabilities )
    { 
        try
        {
            ASSERT_TRUE( testSampleSet.Size() > 0 );
            ASSERT_TRUE( likelihoodProbabilities.size( ) == testSampleSet.Size() );
            ASSERT_TRUE( m_strongClassifierBasePtr != NULL );

            likelihoodProbabilities = m_strongClassifierBasePtr->Classify( testSampleSet );

            //Use sigmoidal function
            for ( int index = 0; index < likelihoodProbabilities.size(); index++ )
            {
                likelihoodProbabilities[index] = pow( sigmoid( likelihoodProbabilities[index]  ), 1 );
            }

            float totalWeight = 0.0f;
            for ( int i = 0; i < testSampleSet.Size(); i++ )
            {
                totalWeight += max( likelihoodProbabilities[i], 0.0f );
            }
        }
        EXCEPTION_CATCH_AND_ABORT( "Failed to Fuse AppearanceBased Information" );
    }
}
