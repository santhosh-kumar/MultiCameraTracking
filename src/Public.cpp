// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#include "Public.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////
// random functions

void                                randinitalize( const int init )
{
    rng_state = cvRNG(init);
}

int                                    randint( const int min, const int max )
{
    return cvRandInt( &rng_state )%(max-min+1) + min;
}

float                                randfloat( )
{
    return (float)cvRandReal( &rng_state );
}

vectori                                randintvec( const int min, const int max, const uint num )
{
    vectori v(num);
    for( uint k=0; k<num; k++ ) v[k] = randint(min,max);
    return v;
}
vectorf                                randfloatvec( const uint num )
{
    vectorf v(num);
    for( uint k=0; k<num; k++ ) v[k] = randfloat();
    return v;
}
float                                randgaus(const float mean, const float sigma)
{
  double x, y, r2;

  do{
      x = -1 + 2 * randfloat();
      y = -1 + 2 * randfloat();
      r2 = x * x + y * y;
  }
  while (r2 > 1.0 || r2 == 0);

  return (float) (sigma * y * sqrt (-2.0 * log (r2) / r2)) + mean;
}

vectorf                                randgausvec(const float mean, const float sigma, const int num)
{
    vectorf v(num);
    for( int k=0; k<num; k++ ) v[k] = randgaus(mean,sigma);
    return v;
}

vectori                                sampleDisc(const vectorf &weights, const uint num)
{
    vectori inds(num,0);
    int maxind = (int)weights.size()-1;

    // normalize weights
    vectorf nw(weights.size());

    nw[0] = weights[0];
    for( uint k=1; k<weights.size(); k++ )
        nw[k] = nw[k-1]+weights[k];

    // get uniform random numbers
    static vectorf r;
    r = randfloatvec(num);

    //#pragma omp parallel for
    for( int k=0; k<(int)num; k++ )
        for( uint j=0; j<weights.size(); j++ ){
            if( r[k] > nw[j] && inds[k]<maxind) inds[k]++;
            else break;
        }

    return inds;
    
}

string                                int2str( int i, int ndigits )
{
  ostringstream temp;
  temp << setfill('0') << setw(ndigits) << i;
  return temp.str();
}
