/*******************************************************************************
Information fusion based on Expectation Maximization Algorithm in OpenCV
http://opencv.willowgarage.com/documentation/expectation-maximization.html
********************************************************************************/
#ifndef GEOMETRIC_INFORMATION_FUSER
#define GEOMETRIC_INFORMATION_FUSER

#include "Matrix.h"
#include "Config.h"
#include "CommonMacros.h"

#include "ml.h"
#include "highgui.h"
#include "math.h"

#include <boost/shared_ptr.hpp>

#define MAX_NUMBER_CAMERA_COLORS 5

namespace MultipleCameraTracking
{
    /***************************************************************************
    GeometryBasedInformationFuser

        GeometryBasedInformationFuser fuses information from multiple cameras. 
        It uses Kalman filter to fuse information from multiple cameras.
        It fits a GMM over the shared particles and re-weights the particles
        based on GMM distribution.

        Reference:
        http://vision.ece.ucsb.edu/publications/view_abstract.cgi?357
    ****************************************************************************/
    class GeometryBasedInformationFuser
    {
    public:

        enum GroundPlaneMeasurmentType { HIGHEST_WEIGHT_CLUSTER                = 0,
                                         WEIGHTED_AVERAGE_OF_CLUSTERS        = 1,
                                         PRINCIPAL_AXIS_INTERSECTION        = 2,
                                        };

        GeometryBasedInformationFuser( const unsigned int           numberOfCamera,
                                       CvMat*                       particles,
                                       GroundPlaneMeasurmentType    groundPlaneMeasurementType,
                                       vector<CvMat*>                homographyMatrixList );
        ~GeometryBasedInformationFuser( );


        GroundPlaneMeasurmentType GetGroundPlaneMeasurementType( ){ return m_groundPlaneMeasurementType; }

        void FuseInformation( CvMat* pParticlesMatrix );

        CvMat* GetGroundPlaneKalmanMeanMatrix( int index );
        CvMat* GetGroundPlaneKalmanCovarianceMatrix( );

        Matrixu*   DisplayOriginalGroundParticles( CvMat* particles, int frameInd = -1 ) ;
        Matrixu*   DisplayKalmanFilterPdf( CvMat* pMeanMatrix, CvMat* pCovarianceMatrix, int frameInd = -1 );
        Matrixu*   DisplayGMMGroundParticles( int frameInd = -1, CvMat* particles=NULL );

        //static methods
        static void CopyMatrix( CvMat*    pSourceMatrix,
                                CvMat*    pDestinationMatrix, 
                                int        sourceStartRowIndex = 0,
                                int        destinationStartRowIndex = 0,
                                int        option = 0 );

        static float MultiVariateNormalPdf( CvMat* dataVector, CvMat* meanVector, CvMat* covarianceMatrix );

        //Project the Particles to Ground Plane using the Homography
        static CvMat*    TransformWithHomography( CvMat* particles, CvMat* pHomographyMatrix );

        //estimate ground plane positions from principle axis
        static     vectord EstimateGroundPlanePrincipleAxisIntersection(    CvMat*            pPixelMatrix,
                                                                        vector<CvMat*>    homographyMatrixList );    
    private:

        DISALLOW_IMPLICIT_CONSTRUCTORS( GeometryBasedInformationFuser );

        //set the default EM parameters
        void SetDefaultEMParameters( );

        //initialize the Kalman Filter
        void InitializeKalman( CvMat* particles );

        //Update the Kalman Filter
        void UpdateKalmanFilter( CvMat* measurement, CvMat* measurementCov =NULL );

        //Fuse information by picking the best cluster for measurement
        void FuseInformationByPickingTheBestCluster( CvMat* pParticlesMatrix );    

        //Fuse information by using weighted average for measurement
        void FuseInformationByAveragingClusters( CvMat* pParticlesMatrix );    

        //Fuse information using the principal axis intersection
        void FuseInformationByUsingPrincipalAxisIntersection( CvMat* pParticlesMatrix );

        //given a point on the ground plane, find its corresponding position on the ground plane image
        void MetricToPixels( float xPosFeet, float yPosFeet, int * xPosPixel, int * yPosPixel );
        
        //given a pixel on the ground plane image, find its corresponding position on the ground plane
        void PixelsToMetric( float * xPosFeet, float * yPosFeet, int xPosPixel, int yPosPixel );

        //initialize the ground plane display
        void InitializeGroundplaneDisplay( );

        CvEM                          m_emModel;                //EM Model
        CvEMParams                    m_emParamters;            //Parameters of the expectation maximization algorithm
        CvKalman*                     m_pKalman;                //Kalman Filter
        const unsigned int            m_numberOfClusters;       //Number of Clusters or Gaussian
        unsigned int                  m_numberOfCameras;        //Number of Cameras to be fused
        //member variables for display purpose
        Matrixu                        m_colorMatrix;           //hold the colors to use for different cameras
        Matrixu                        m_GroundPlaneDisplayMatrix;
        Matrixu                        m_GroundPlaneDisplayMatrixKF;
        Matrixu                        m_GroundPlaneDisplayMatrixGMM;
        GroundPlaneMeasurmentType      m_groundPlaneMeasurementType;
        vector<CvMat*>                 m_homographyMatrixList;
    };

    //typedef for a boost pointer
    typedef boost::shared_ptr<GeometryBasedInformationFuser> GeometryBasedInformationFuserPtr;
    typedef std::vector<GeometryBasedInformationFuserPtr>    GeometryBasedInformationFuserPtrList;
}
#endif
