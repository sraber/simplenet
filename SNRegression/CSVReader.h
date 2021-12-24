#pragma once

struct MPG_Data {
   float mpg;
   int cylinders;
   float displacement;
   float horsepower;
   float weight;
   float acceleration;
   int year;
   int origin;
   std::string name;
};

std::list<MPG_Data> ReadMPGData(std::string filename);
