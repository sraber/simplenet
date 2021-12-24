#include <fstream>
#include <list>
#include <string>
#include <sstream>
#include "CSVReader.h"



std::list<MPG_Data> ReadMPGData(std::string filename)
{
   std::ifstream file(filename);
   std::list<MPG_Data> datalist;
   std::string line;
   std::getline(file,line);
   do {
      MPG_Data data;
      std::string sitem;
      std::istringstream strline(line);

      strline >> data.mpg;
      strline >> data.cylinders;
      strline >> data.displacement;
      strline >> data.horsepower;
      strline >> data.weight;
      strline >> data.acceleration;
      strline >> data.year;
      strline >> data.origin;
      //std::string temp;
      //std::string name;
      //do {
      //   strline >> temp;
      //   name = name + temp + " ";
      //} while (!strline.eof() && !strline.bad() );
      //data.name = name;
      datalist.push_back(data);

      std::getline(file, line);
   } while (!file.eof());

   return datalist;
}

