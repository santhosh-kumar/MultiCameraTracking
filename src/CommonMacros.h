#ifndef COMMONMACROS
#define COMMONMACROS

#include "Public.h"
#include "Config.h"

//Template function to Check if the parameter is valid
template <class T>
T CheckIfValidParameter( bool predicate, T arg, int line, char* file )
{
    if ( predicate )
    {
        return arg;
    }
    else
    {
        abortError( line, file, "Precondition Failed on Parameter" );
    }
}

#define EXCEPTION_CATCH_AND_ABORT(context)        \
catch( std::exception& ex )                        \
{                                                \
    std::cerr << "Context: " << context <<    " Error Message:"     << ex.what() << endl;    \
    abortError( __LINE__, __FILE__, context );    \
}                                                

#define EXCEPTION_CATCH_AND_LOG(context)        \
catch( std::exception& ex )                        \
{                                                \
    g_logFile << "Context: " << context <<    " Error Message:"     << ex.what() << endl;    \
}        

#if defined(WIN32) || defined(WIN64)
#define ASSERT_TRUE( predicate )                            \
if ( (predicate) == false )                                    \
{                                                            \
    DebugBreak();                                            \
    abortError( __LINE__, __FILE__, "Assertion Failed." );    \
}                                                            
#else
#define ASSERT_TRUE( predicate )                            \
if ( (predicate) == false )                                    \
{                                                            \
    abortError( __LINE__, __FILE__, "Assertion Failed." );    \
}                                                            
#endif

#define ASSERT_PRECONDITION_PARAMETER( predicate, parameter )            \
    CheckIfValidParameter( predicate, parameter, __LINE__, __FILE__ )

/*    A macro to disallow the evil copy constructor and operator= functions
    This should be used in the private: declarations for a class    */
#define DISALLOW_EVIL_CONSTRUCTORS(TypeName)    \
TypeName(const TypeName&);                        \
void operator=(const TypeName&)

/*A macro to disallow all the implicit constructors, namely the
default constructor, copy constructor and operator= functions.
This should be used in the private: declarations for a class
that wants to prevent anyone from instantiating it. This is
especially useful for classes containing only static methods.*/
#define DISALLOW_IMPLICIT_CONSTRUCTORS(TypeName) \
TypeName();                                    \
DISALLOW_EVIL_CONSTRUCTORS(TypeName)

#define LOG( context )        \
MultipleCameraTracking::g_logFile << context;        \
if ( MultipleCameraTracking::g_verboseMode )        \
    cout << context;

#define LOG_CONSOLE( context )    \
    cout << context

#endif