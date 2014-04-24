#ifndef PARTICLE_FILTER
#define PARTICLE_FILTER

#include "Exception.h"
#include "CommonMacros.h"
#include "Matrix.h"
#include "Public.h"

#include "cxcore.hpp"

#define PF_INITIALIZATION_BOUNDARY  0 

namespace MultipleCameraTracking
{
    //Forward Declaration
    class ParticleFilter;

    //Declaration of shared ptr
    typedef boost::shared_ptr<ParticleFilter>    ParticleFilterPtr;

    /****************************************************************
     Particle Filters
        This class implements a BootsStrap ParticleFilter.

        State Diagram: 

    ****************************************************************/
    class ParticleFilter
    {
    public:

        //to find whether m_particlesOrderedUniqueMatrix is unique after re sampling
        enum State {    UNINTIALIZED = 0, 
                        INITIALIZED  = 1 ,
                        PREDICTED     = 2,
                        RESAMPLED     = 3,
                    };

        enum ResamplingStrategy { WEIGHT_BASED,
                                  TIME_INTERVAL_BASED,
                                  WEIGHT_OR_TIME_INTERVAL_BASED
                                };

        //constructor
        ParticleFilter( const float                    imageWidth,
                        const float                    imageHeight,
                        const unsigned short        particleStateDimension    = 4,
                        const ResamplingStrategy    resamplingStrategy        = WEIGHT_BASED )
            : m_imageWidth( imageWidth ),
            m_imageHeight( imageHeight ),
            m_particlesStateMatrix( ),
            m_particlesResampledMatrix( ),
            m_particlesOrderedUniqueMatrix( ),
            m_resampleIndexList( ),
            m_numberOfParticles( ),
            m_maximumScaleChange( ),
            m_filterState( UNINTIALIZED ),
            m_particleStateDimension( particleStateDimension ), 
            m_initializationRelaxation ( PF_INITIALIZATION_BOUNDARY ),
            m_timeIndex( 0 ),
            m_isOrderedUniqueParticlesFound( false ),
            m_resamplingStrategy( resamplingStrategy )
        {    
        }

        // Initialize particle filters
        void    Initialize( int numberOfParticles,
                            float centerX,
                            float centerY,
                            float scaleX,
                            float scaleY,
                            float maximumScaleChangeBetweenFrame = 0.5,
                            float maxScale = 10,
                            float minScale = 0.1 );
        
        // Prediction from m_particlesResampledMatrix based on Brownian motion 
        void    PredictWithBrownianMotion( float standardDeviationX, 
                                           float standardDeviationY,
                                           float standardDeviationScaleX,
                                           float standardDeviationScaleY,
                                           float width,
                                           float height );

        //check for particle refinement
        void    CheckForParticleRefinement( float width, float height );

        //Prediction from m_particlesResampledMatrix based on Uniform motion
        void    PredictWithUniformMotion(   float standardDeviationX,
                                            float standardDeviationY,
                                            float standardDeviationScaleX,
                                            float standardDeviationScaleY );

        void    ForceParticleFilterRefinement(    float        width, 
                                                float        height );

        // access functions to retrieve individual particle
        void    GetParticle( int index, vectorf& particle );
        void    GetResampledParticle( int index, vectorf& particle );
        void    GetOrderedUniqueParticles( int index, vectorf& particle );
        void    GetHighestOrderedUniqueParticleCloseToTheGivenState( vectorf& particle, vectorf& closestState );
        void    GetAverageofAllParticles(vectorf& particle) const;
        void    GetHighestWeightParticle( vectorf& particle );

        // Number of unique particles in m_particlesResampledMatrix
        int GetNumberOfOrderedUniqueParticles( );

        float GetMaxScale( ) const { return m_maxScale; }

        // Set a particular predicted particle with a new weight 
        void UpdateParticleWeight( int index, float probability, bool shouldForceReset = false );

        // Update the weights of non-zero-weight predicated particles followed by re sampling to m_particlesResampledMatrix
        void UpdateAllParticlesWeight( vectorf& probability, bool shouldUpdateOnlyNonZeroWeights = true, bool shouldForceReset = false, bool shouldResample = true );

        //Resample the particles
        void ResampleParticles( bool shouldForceFilterStateChange = false );

        //Get Number of Unique Particles
        int     GetNumberOfParticles( ) const { return m_numberOfParticles; };

        void GetACopyOfVectorFormat( vector< vector<float> > & particlesVector );

        void RearrangeParticlesBasedOnGroundLocation(    vectorf&    meanGroundPlaneFootPositionOnImageList,
                                                        vectorf&     varianceGroundPlaneFootPositionOnImageList,
                                                        bool        shouldChangeTheScaleAccordingToOffset = false,
                                                        float        width = 0.0f,
                                                        float        height = 0.0f );
    private:

        //Find the unique particles within ParticleResampled and order them in terms of weight
        void                    FindOrderedUniqueParticles( );

        //Check for re-sampling required
        void                    ResampleIfRequired( );

        //Normalize Particle weights
        void NormalizeParticleWeights( );

        Matrixf                        m_particlesStateMatrix;            //[centerX, centerY, scaleX, scaleY, weight]; 
        Matrixf                        m_tempParticleStateMatrix; 
        Matrixf                        m_particlesResampledMatrix;
        Matrixf                        m_particlesOrderedUniqueMatrix;
        vectori                        m_resampleIndexList;
        int                            m_numberOfParticles;
        float                        m_maximumScaleChange;      // maximum scale changes between frame
        State                        m_filterState;
        const unsigned short        m_particleStateDimension; //dimension of the particle filter state (i.e. the index for the particle weight)
        float                        m_maxScale;
        float                        m_minScale;
        float                        m_imageWidth;
        float                        m_imageHeight;
        const    int                    m_initializationRelaxation;
        uint                        m_timeIndex;
        bool                        m_isOrderedUniqueParticlesFound;    //For optimization, so that repeated calculation
        const ResamplingStrategy    m_resamplingStrategy;
    };
}
#endif
