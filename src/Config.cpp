#include "Config.h"
#include "CommonMacros.h"

namespace MultipleCameraTracking
{
    InputParameters g_configInput;
    std::ofstream    g_logFile;
    bool            g_detailedLog, g_verboseMode;
    
    // Mapping_Map Syntax:
    // {NAMEinConfigFile,                               &g_configInput.VariableName,                                  Type,InitialValue,LimitType,MinLimit,MaxLimit}
    // Types : {0:int, 1:text, 2: double}
    // LimitType: {0:none, 1:both, 2:minimum, 3: for special case}
    // We could separate this based on types to make it more flexible and allow also defaults for text types.
    // Default only for int/double type.

    Mapping Map[] = {
        //Input Information  
        {"Input_Directory_Name",                        &g_configInput.m_inputDirectoryNameCstr,                        1,      0.0,        0,      0,      0},
        {"Initialization_Name",                         &g_configInput.m_intializationDirectoryCstr,                    1,      0.0,        0,      0,      0},
        {"Input_Data_FilesName",                        &g_configInput.m_dataFilesNameCstr,                             1,      0.0,        0,      0,      0},
        {"Camera_Set",                                  &g_configInput.m_cameraSetCstr,                                 1,      0.0,        0,      0,      0},
        {"Object_Set",                                  &g_configInput.m_objectSetCstr,                                 1,      0.0,        0,      0,      0},        
        {"Load_Video_With_Color",                       &g_configInput.m_loadVideoWithColor,                            0,      0,          1,      0,      1},
        {"Load_Video_From_Images",                      &g_configInput.m_loadVideoFromImgs,                             0,      1,          1,      0,      1},        
        {"Number_of_Frames",                            &g_configInput.m_numOfFrames,                                   0,      1,          2,      1,      0},
        {"Starting_Frame_Index",                        &g_configInput.m_startFrameIndex,                               0,      1,          2,      1,      0},
        {"Enable_Interative_Mode",                      &g_configInput.m_interactiveModeEnabled,                        0,      0,          1,      0,      1},

        //Output Information
        {"Trial_Number",                                &g_configInput.m_trialNumber,                                   0,      1,          2,      1,      0},
        {"Enable_Verbose_Mode",                         &g_configInput.m_verboseMode,                                   0,      1,          1,      0,      1},
        {"Enalbe_Detailed_Log",                         &g_configInput.m_detailedLogging,                               0,      1,          1,      0,      1},        
        {"Output_Directory_Name",                       &g_configInput.m_outputDirectoryNameCstr,                       1,      0,          0,      0,      0},
        {"Whether_Save_Output_Video",                   &g_configInput.m_saveOutputVideo,                               0,      0,          1,      0,      1},  
        {"Whether_Display_Training_Samples",            &g_configInput.m_displayTrainingSamples,                        0,      0,          1,      0,      1},  
        {"Display_Training_Center_Only",                &g_configInput.m_displayTrainingExampCenterOnly,                0,      0,          1,      0,      1},  
        {"Whether_Save_Training_Samples",               &g_configInput.m_saveTrainingSamplesVideo,                      0,      0,          1,      0,      1},  
        {"Whether_Display_Output_Video",                &g_configInput.m_displayOutputVideo,                            0,      0,          1,      0,      1},  
        {"Wait_Before_TrackingEnd",                     &g_configInput.m_waitBeforeFinishTracking,                      0,      0,          1,      0,      1},        
        {"Whether_Calculate_Tracking_Error",            &g_configInput.m_calculateTrackingError,                        0,      0,          1,      0,      1},  

        // Feature and classifier
        {"Tracker_Feature_Type",                        &g_configInput.m_trackerFeatureType,                            0,      1,          1,      1,      3.0},
        {"Use_HSV_Color",                               &g_configInput.m_useHSVColor,                                   0,      0,          1,      0,      1.0},
        {"Color_Number_Of_Bins",                        &g_configInput.m_numofBinsColor,                                0,      8,          1,      4,      16.0},
        {"Tracker_Feature_Parameter",                   &g_configInput.m_trackerFeatureParameter,                       0,      0,          0,      0,      0},  
        {"Tracker_Strong_Classifier_Type",              &g_configInput.m_trackerStrongClassifierType,                   0,      1,          2,      1,      4.0},
        {"Tracker_Weak_Classifier_Type",                &g_configInput.m_trackerWeakClassifierType,                     0,      1,          1,      1,      3.0},
        {"Percentage_Of_Weak_Classifiers_Selected",     &g_configInput.m_percentageOfWeakClassifiersSelected,           0,      20,         1,      1,      100},
        {"Percentage_Of_Weak_Classifier_Retained",      &g_configInput.m_percentageOfWeakClassifiersRetained,           0,      10,         1,      1,      100},

        //tracking setting
        {"Local_Tracker_Type",                          &g_configInput.m_localTrackerType,                              0,       0,         2,      0,      0},  
        {"Inner_Radius_For_Positive_Examples",          &g_configInput.m_posRadiusTrain,                                0,       4,         2,      1,      0},  
        {"Initial_Radius_For_Positive_Examples",        &g_configInput.m_initPosRadiusTrain,                            0,       3,         2,      1,      0},  
        {"Initial_Number_Of_Negative_Examples",         &g_configInput.m_initNumNegExampes,                             0,       65,        2,      10,     0},          
        {"Number_Of_Negative_Examples",                 &g_configInput.m_numNegExamples,                                0,       65,        2,      10,     0},  
        {"Search_Window_Size",                          &g_configInput.m_searchWindowSize,                              0,       25,        2,      1,      0},  
        {"Negative_Sampling_Strategy",                  &g_configInput.m_negSampleStrategy,                             0,       0,         1,      0,      1},  
    
        // Particle filter tracker parameters
        {"Num_Of_Particles",                            &g_configInput.m_numOfParticles,                                0,        50,       2,      1,      0}, 
        {"Particle_Filter_Std_Dev_X",                   &g_configInput.m_PFTrackerStdDevX,                              2,        5,        2,      0,      0}, 
        {"Particle_Filter_Std_Dev_Y",                   &g_configInput.m_PFTrackerStdDevY,                              2,        5,        2,      0,      0}, 
        {"Particle_Filter_Std_Dev_ScaleX",              &g_configInput.m_PFTrackerStdDevScaleX,                         2,        0,        2,      0,      0}, 
        {"Particle_Filter_Std_Dev_ScaleY",              &g_configInput.m_PFTrackerStdDevScaleY,                         2,        0,        2,      0,      0}, 
        {"PfTracker_Max_Num_Positive_Examples",         &g_configInput.m_PfTrackerMaxNumPositiveExamples,               0,        30,       2,      1,      0},         
        {"PfTracker_Num_Disp_Particles",                &g_configInput.m_PFTrackerNumDispParticles,                     0,        5,        2,      1,      0},         
        {"PfTracker_Output_Trajectory_Option",          &g_configInput.m_PFOutputTrajectoryOption,                      0,        0,        1,      0,      1}, 
        {"PfTracker_Positive_Example_Strategy",         &g_configInput.m_PfTrackerPositiveExampleStrategy,              0,        0,        2,      0,      0},                
        {"PfTracker_Negative_Example_Strategy",         &g_configInput.m_PfTrackerNegativeExampleStrategy,              0,        0,        2,      0,      0},                
        
        // Fusion setting
        {"Geometric_Fusion_Type",                       &g_configInput.m_geometricFusionType,                           0,        0,        2,      0,      0},  
        {"Save_Ground_Particles_Image",                 &g_configInput.m_saveGroundParticlesImage,                      0,        0,        1,      0,      1},  
        {"Save_Ground_Plane_KF_Image",                  &g_configInput.m_saveGroundPlaneKFImage,                        0,        0,        1,      0,      1},          
        {"Display_Ground_GMM_Centers",                  &g_configInput.m_displayGMMCenters,                             0,        0,        1,      0,      1},  
        {"Display_Ground_Particles_Image",              &g_configInput.m_displayGroundParticlesImage,                   0,        0,        1,      0,      1},  
        {"Display_Ground_Plane_KF_Image",               &g_configInput.m_displayGroundPlaneKFImage,                     0,        0,        1,      0,      1},  
        {"Appearance_Fusion_Type",                      &g_configInput.m_appearanceFusionType,                          0,        1,        2,      0,      0},
        {"Appearance_Fusion_Strong_Classifier_Type",    &g_configInput.m_appearanceFusionStrongClassifierType,          0,        1,        1,      1,      3.0},
        {"Appearance_Fusion_Weak_Classifier_Type",      &g_configInput.m_appearanceFusionWeakClassifierType,            0,        1,        1,      1,      3.0},
        {"Enable_Cross_Camera_Occlusion_Handle",        &g_configInput.m_enableCrossCameraOcclusionHandling,            0,        0,        1,      0,      1},  
        {"Enable_Cross_Camera_Auto_Initialization",     &g_configInput.m_enableCrossCameraAutoInitialization,           0,        0,        1,      0,      1},  
        {"Percentage_Of_Weak_Classifiers_Selected_AF",  &g_configInput.m_AFpercentageOfWeakClassifiersSelected,         0,        20,       1,      1,      100},
        {"Percentage_Of_Weak_Classifier_Retained_AF",   &g_configInput.m_AFpercentageOfWeakClassifiersRetained,         0,        10,       1,      1,      100},

        {"Appearance_Fusion_Num_Of_Positive_Examples",  &g_configInput.m_AFNumberOfPositiveExamples,                    0,        30,       1,      1,      100},
        {"Appearance_Fusion_Num_Of_Negative_Examples",  &g_configInput.m_AFNumberOfNegativeExamples,                    0,        50,       1,      1,      100},
        {"Appearance_Fusion_Refresh_Rate",              &g_configInput.m_AFRefreshRate,                                 0,        1,        1,      1,      100},
        {NULL,                                NULL,                                                                     -1,       0.0,      0,      0.0,    0.0}
    };

    static void Usage(void)
    {
        fprintf( stderr, "\n Usage:  MultipleCameraTracking [-h] or MultipleCameraTracking -d config.txt [-p xxx=xxx] [-p xxx=xxx]\n");
    }
    /*!
     ***********************************************************************
     * \brief
     *    Returns the index number from Map[] for a given parameter name.
     * \param s
     *    parameter name string
     * \return
     *    the index number if the string is a valid parameter name,         \n
     *    -1 for error
     ***********************************************************************
     */
    static int ParameterNameToMapIndex (char *s)
    {
      int i = 0;

      while (Map[i].TokenName != NULL)
        if (0==strcasecmp (Map[i].TokenName, s))
          return i;
        else
          i++;
      return -1;
    }

    /*!
     ***********************************************************************
     * \brief
     *    Sets initial values for parameters.
     * \return
     *    -1 for error
     ***********************************************************************
     */
    static int InitParams(void)
    {
      int i = 0;

      while (Map[i].TokenName != NULL)
      {
        if (Map[i].Type == 0)  // int
          * (int *) (Map[i].Place) = (int) Map[i].Default;
        else if (Map[i].Type == 1) // text string
          * (char*) (Map[i].Place) = '\0'; //default empty string
        else if (Map[i].Type == 2) // double
          * (double *) (Map[i].Place) = Map[i].Default;
          i++;
      }
      return -1;
    }

    /*!
     ***********************************************************************
     * \brief
     *    allocates memory buf, opens file Filename in f, reads contents into
     *    buf and returns buf
     * \param Filename
     *    name of config file
     * \return
     *    if successfull, content of config file
     *    NULL in case of error. Error message will be set in errortext
     ***********************************************************************
     */
    static char *GetConfigFileContent (char *Filename)
    {
      long FileSize;
      FILE *f;
      char *buf;

      if (NULL == (f = fopen (Filename, "r")))
      {
        printf ("Cannot open configuration file %s.\n", Filename);
        return NULL;
      }

      if (0 != fseek (f, 0, SEEK_END))
      {
        printf ("Cannot fseek in configuration file %s.\n", Filename);
        return NULL;
      }

      FileSize = ftell (f);
      if (FileSize < 0 || FileSize > 60000)
      {
        printf ("Unreasonable Filesize %ld reported by ftell for configuration file %s.\n", FileSize, Filename);
        return NULL;
      }
      if (0 != fseek (f, 0, SEEK_SET))
      {
        printf ("Cannot fseek in configuration file %s.\n", Filename);
        return NULL;
      }

      if ((buf = (char*)malloc (FileSize + 1))==NULL) 
      {
        printf("Cannot alloc mem: buf\n");
        return NULL;
      }

      // Note that ftell() gives us the file size as the file system sees it.  The actual file size,
      // as reported by fread() below will be often smaller due to CR/LF to CR conversion and/or
      // control characters after the dos EOF marker in the file.

      FileSize = (long) fread (buf, 1, FileSize, f);
      buf[FileSize] = '\0';

      fclose (f);
      return buf;
    }


    /*!
     ***********************************************************************
     * \brief
     *    Parses the character array buf and writes global variable input, which is defined in
     *    configfile.h.  This hack will continue to be necessary to facilitate the addition of
     *    new parameters through the Map[] mechanism (Need compiler-generated addresses in map[]).
     * \param buf
     *    buffer to be parsed
     * \param bufsize
     *    buffer size of buffer
     * \return
     *    0: success
     *    non-zero: fail
     ***********************************************************************
     */
    static int ParseContent (char *buf, int bufsize)
    {

      char *items[MAX_ITEMS_TO_PARSE];
      int MapIdx;
      int item = 0;
      int InString = 0, InItem = 0;
      char *p = buf;
      char *bufend = &buf[bufsize];
      int IntContent;
      double DoubleContent;
      int i;

    // Stage one: Generate an argc/argv-type list in items[], without comments and whitespace.
    // This is context insensitive and could be done most easily with lex(1).

      while (p < bufend)
      {
        switch (*p)
        {
          case 13:
            p++;
            break;
          case '#':                 // Found comment
            *p = '\0';              // Replace '#' with '\0' in case of comment immediately following integer or string
            while (*p != '\n' && p < bufend)  // Skip till EOL or EOF, whichever comes first
              p++;
            InString = 0;
            InItem = 0;
            break;
          case '\n':
            InItem = 0;
            InString = 0;
            *p++='\0';
            break;
          case ' ':
          case '\t':              // Skip whitespace, leave state unchanged
            if (InString)
              p++;
            else
            {                     // Terminate non-strings once whitespace is found
              *p++ = '\0';
              InItem = 0;
            }
            break;

          case '"':               // Begin/End of String
            *p++ = '\0';
            if (!InString)
            {
              items[item++] = p;
              InItem = ~InItem;
            }
            else
              InItem = 0;
            InString = ~InString; // Toggle
            break;

          default:
            if (!InItem)
            {
              items[item++] = p;
              InItem = ~InItem;
            }
            p++;
        }
      }

      item--;

      for (i=0; i<item; i+= 3)
      {
        if (0 > (MapIdx = ParameterNameToMapIndex (items[i])))
        {
          //snprintf (errortext, ET_SIZE, " Parsing error in config file: Parameter Name '%s' not recognized.", items[i]);
          //error (errortext, 300);
          printf ("\n\tParsing error in config file: Parameter Name '%s' not recognized.", items[i]);
          continue;
        }
        if (strcasecmp ("=", items[i+1]))
        {
          printf (" Parsing error in config file: '=' expected as the second token in each line.");
          return -1;
        }

        // Now interpret the Value, context sensitive...

        switch (Map[MapIdx].Type)
        {
          case 0:           // Numerical
            if (1 != sscanf (items[i+2], "%d", &IntContent))
            {
              printf (" Parsing error: Expected numerical value for Parameter of %s, found '%s'.", items[i], items[i+2]);
              return -1;
            }
            * (int *) (Map[MapIdx].Place) = IntContent;
            printf (".");
            break;
          case 1:
            strncpy ((char *) Map[MapIdx].Place, items [i+2], STRING_SIZE);
            printf (".");
            break;
          case 2:           // Numerical double
            if (1 != sscanf (items[i+2], "%lf", &DoubleContent))
            {
              printf (" Parsing error: Expected numerical value for Parameter of %s, found '%s'.", items[i], items[i+2]);
              return -1;
            }
            * (double *) (Map[MapIdx].Place) = DoubleContent;
            printf (".");
            break;
          default:
            printf("Unknown value type in the map definition of configfile.h");
            return -1;
        }
      }
      //memcpy (input, &g_configInput, sizeof (InputParameters));
      return 0;
    }

    /*!
     ***********************************************************************
     * \brief
     *    display and logging encoding parameters.
     * \return
     *    -1 for error
     ***********************************************************************
     */
    int DisplayAndLogParams(void)
    {
        int i = 0;
        
        LOG(    "******************************************************\n"
            <<    "*              Configuration Parameters              *\n"
            <<    "******************************************************\n" );

        while (Map[i].TokenName != NULL)
        {
            if (Map[i].Type == 0)
            {
                LOG( " Parameter " << Map[i].TokenName << " = " << * (int *) (Map[i].Place) << '\n' );
            }          
            
            else if (Map[i].Type == 1)
            {
                LOG( " Parameter " << Map[i].TokenName << " = " << (char *) (Map[i].Place) << '\n' );
            }
          
            else if (Map[i].Type == 2)
            {
                LOG( " Parameter " << Map[i].TokenName << " = " << * (double *) (Map[i].Place) << '\n' );
            }
          
          i++;
        }
        LOG( "******************************************************\n" );
      
      return 0;
    }


    /*
     * \brief
     *    Chomp space chars at the beginning and at the end of the str
     * \param str
     *    The string to be chomped.
     */
    static void ChompStr(char* str)
    {
      char* temp;
      size_t i, j, len;
      int found;
      
      len = strlen(str);
      
      if (len <= 0) return;
      
      temp = (char*) malloc(len+1);
      if (temp == NULL)
        return;
      
      memset(temp, 0, len+1);
      found = 0;
      for (i = 0, j = 0; i < len; i++)
        if (str[i] != ' ' && str[i] != '\t' && str[i] != '\n' && str[i] != 10 && str[i] != 13)
          break;
          
      for (; i < len; i++)
        temp[j++] = str[i];
        
      for (i = strlen(temp) - 1; i >= 0; i--)
        if (temp[i] == ' ' || temp[i] == '\t' || temp[i] == '\n' || temp[i] == 10 || temp[i] == 13)
          temp[i] = 0;
        else
          break;
      strcpy(str, temp);
      str[strlen(temp)] = 0;
      
      free(temp);
    }

    /*
     * \brief
     *    Get one double number from the buffer
     * \param src
     *    The source buffer
     * \param bufEnd
     *    The end of the source buffer
     * \param data
     *    The data extracted from the buffer
     * \return
     *    The start position after the number is extracted
     */
    static char* getNumber(char* src, char* bufEnd, double* data)
    {
      char* dst;
      char  temp[255];
      while ( (*src < '0' || *src > '9' ) && *src != '-' && *src != '+' && src < bufEnd )
        src++;
      dst = src;
      while ( ((*src >= '0' && *src <= '9') || *src == '.' || *src == '-' || *src == '+') && src < bufEnd )
        src++;
      strncpy(temp, dst, src-dst);
      temp[src-dst] = 0;
      *data = atof(temp);
      
      return src;
    }

    /*
     * \brief
     *    Invalidate the parameters
     * \return
     *    0: success
     *    non-zero: fail
     */
    int CheckParams()
    {

      return 0;
    }


    /*!
     ***********************************************************************
     * \brief
     *    Parse the command line parameters and read the config files.
     * \param ac
     *    number of command line parameters
     * \param av
     *    command line parameters
     * \return
     *    0: sucess
     *    non-zero: fail
     ***********************************************************************
     */
    int Configure(int ac, char*av[])
    {
        char* content; 
        int  CLcount, ContentLen, NumberParams;
        char *filename=DEFAULTCONFIGFILENAME;

        memset (&g_configInput, 0, sizeof (InputParameters)); 
      // Set default parameters.
        printf ("Setting Default Parameters...\n");
        InitParams();
     
        CLcount = 1;
        if (ac==2)
        {
            if (0 == strncmp (av[1], "-h", 2))
            {
                Usage();
                return -1;
            }
        }
        
        if (ac>=3)
        {
            if (0 == strncmp (av[1], "-d", 2))
            {
                filename=av[2];
                CLcount = 3;
            }
            if (0 == strncmp (av[1], "-h", 2))
            {
                Usage();
                return -1;
            }
        }
        printf ("Parsing Configfile %s\n", filename);
        content = GetConfigFileContent (filename);

        if (NULL == content)
           return -1;

        if ( 0 != ParseContent (content, (int)strlen(content)))
        {
            printf ("\n");
            free (content);
            return -1;
        }

        printf ("\n");
        free (content);
      // Parse the command line
      while (CLcount < ac)
      {
        if (0 == strncmp (av[CLcount], "-h", 2))
        {
          Usage();
          return -1;
        }

        if (0 == strncmp (av[CLcount], "-p", 2))  // A config change?
        {
          // Collect all data until next parameter (starting with -<x> (x is any character)),
          // put it into content, and parse content.

          CLcount++;
          ContentLen = 0;
          NumberParams = CLcount;

          // determine the necessary size for content
          while (NumberParams < ac && av[NumberParams][0] != '-')
            ContentLen += (int)strlen (av[NumberParams++]);        // Space for all the strings
          ContentLen += 1000;                     // Additional 1000 bytes for spaces and \0s


          if ((content = (char*) malloc (ContentLen))==NULL) 
          {
            printf("Mem error, Configure: content");
            return -1;
          }
          content[0] = '\0';

          // concatenate all parameters identified before

          while (CLcount < NumberParams)
          {
            char *source = &av[CLcount][0];
            char *destin = &content[strlen (content)];

            while (*source != '\0')
            {
              if (*source == '=')  // The Parser expects whitespace before and after '='
              {
                *destin++=' '; *destin++='='; *destin++=' ';  // Hence make sure we add it
              } else
                *destin++=*source;
              source++;
            }
            *destin = '\0';
            CLcount++;
          }
          printf ("Parsing command line string '%s'", content);
          if ( 0 != ParseContent (content, (int) strlen(content)))
          {
            free (content);
            printf ("\n");
            return -1;
          }
          free (content);
          printf ("\n");
        }
        else
        {
          printf ("Error in command line, ac %d, around string '%s', missing -f or -p parameters?", CLcount, av[CLcount]);
          return -1;
        }
      }
      printf ("\n");
        
      return CheckParams();
}

}
