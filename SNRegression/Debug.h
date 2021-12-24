#ifndef __Debug__h
#define __Debug__h

#include <windows.h>
#include <strstream>
#define endl "\n"


#define CATCH_LINE "Catch in file " << __FILE__ << " at line " << __LINE__ <<"." << endl

class DebugOutputEx{
   friend class DebugOutput;
protected:
   DebugOutputEx(std::strstream* ss) : ssOut(ss){}
   std::strstream *ssOut;
public:
   ~DebugOutputEx();

   template<class T>
   DebugOutputEx& operator<<(T entry){
      *ssOut << entry;
      return *this;
      }

   DebugOutputEx& operator<<(char* entry){
      if( entry ){
         *ssOut << entry;
         }
      return *this;
      }

   DebugOutputEx& operator<<(const char* entry){
      if( entry ){
         *ssOut << entry;
         }
      return *this;
      }
   };


class DebugOutput{
   DebugOutput(DebugOutput&){};
   void operator=(DebugOutput&){}
protected:
   std::strstream* ssOut;
public:
   DebugOutput() : ssOut(0){}
   ~DebugOutput(){}

// DebugOutput >> DebugOutputEx >> DebugOutputEx ... etc..
   template<class T>
   DebugOutputEx operator<<(T entry){
      ssOut = new strstream;
      *ssOut << entry; 
      return DebugOutputEx(ssOut,&csOutputSync);
      }

   DebugOutputEx operator<<(char* entry){
      ssOut = new std::strstream;
      if( entry ){
         *ssOut << entry; 
         }

      return DebugOutputEx(ssOut);
      }

   DebugOutputEx operator<<(const char* entry){
      ssOut = new std::strstream;
      if( entry ){
         *ssOut << entry; 
         }

      return DebugOutputEx(ssOut);
      }
   };


#ifdef _L1DEBUG
   #ifndef _L2DEBUG
      #define _L2DEBUG
   #endif
// NOTE: This does work.  Might want to try this in the furture.
//   #define DebugLevel1( a ) { DebugOutput db; db << a; }
   #define DebugLevel1( a ) DebugOutput() << a;
#else
   #define DebugLevel1( a ) 
#endif

#ifdef _L2DEBUG
   #define DebugLevel2( a ) DebugOutput() << a;
#else
   #define DebugLevel2( a ) 
#endif

#endif
