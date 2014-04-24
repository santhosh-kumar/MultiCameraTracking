#ifndef H_IMGFTR
#define H_IMGFTR

#include "FeatureParameters.h"
#include "Matrix.h"
#include "Public.h"
#include "SampleSet.h"
#include "CommonMacros.h"

#include <boost/shared_ptr.hpp>

namespace Features
{
    //Forward Declaration
    class Feature;
    class FeatureVector;

    typedef boost::shared_ptr<Feature>    FeaturePtr;
    typedef vector<FeaturePtr>            FeatureList;

    /****************************************************************
    Feature
        Base class for Feature.
    ****************************************************************/
    class Feature
    {
    public:

        //Pure Virtual function
        virtual const FeatureType        GetFeatureType( ) const = 0;

        //create (initialize) a feature instance from associated feature parameters
        virtual void            Generate( FeatureParametersPtr featureParametersPtr ) = 0;
        
        //compute the feature value for the given sample (for one-dim feature vector) 
        virtual float            Compute( const Classifier::Sample& sample ) const = 0;

        //compute the feature value for the given sample (for multi-dim feature vector)
        virtual void            Compute( const Classifier::Sample& sample, vectorf& featureValueList ) const = 0;

        //visualize the feature for debugging 
        virtual Matrixu            ToVisualize( int featureIndex = -1 ) { Matrixu empty; return empty; };    
    };
}
#endif