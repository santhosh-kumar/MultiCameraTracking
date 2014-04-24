#include <iostream>
#include <exception>

namespace Exception
{
    /****************************************************************
    AssertionFailed
    ****************************************************************/
    class AssertionFailed: public std::exception
    {
        AssertionFailed( const unsigned int lineNumber,
                         const char*        fileName,
                         const char*        errorMessage )
            : std::exception( ),
            m_lineNumber( lineNumber ),
            m_fileName( fileName ),
            m_errorMessage( errorMessage )
        {
        }

        virtual const char* what( ) const throw( )
        {
            return m_errorMessage;
        }

    private:
        const unsigned int m_lineNumber;
        const char*           m_fileName;
        const char*           m_errorMessage;
    };

    /****************************************************************
    PreconditionFailed
    ****************************************************************/
    class PreconditionFailed: public std::exception
    {
        PreconditionFailed( const unsigned int lineNumber,
                            const char*        fileName,
                            const char*        errorMessage )
            : std::exception( ),
            m_lineNumber( lineNumber ),
            m_fileName( fileName ),
            m_errorMessage( errorMessage )
        {
        }

        virtual const char* what( ) const throw( )
        {
            return m_errorMessage;
        }

    private:
        const unsigned int m_lineNumber;
        const char*           m_fileName;
        const char*           m_errorMessage;
    };

    /****************************************************************
    ConfigurationParameterNotFound
    ****************************************************************/
    class ConfigurationParameterNotFound : public std::exception
    {
        ConfigurationParameterNotFound( const char*        parameterName,
                                        const unsigned int lineNumber,
                                        const char*        fileName,
                                        const char*        errorMessage )
            : std::exception( ),
            m_parameterName( parameterName ),
            m_lineNumber( lineNumber ),
            m_fileName( fileName ),
            m_errorMessage( errorMessage )
        {
        }

        virtual const char* what( ) const throw( )
        {
            return m_errorMessage;
        }

    private:
        const unsigned int m_lineNumber;
        const char*           m_fileName;
        const char*           m_errorMessage;
        const char*           m_parameterName;
    };
}
