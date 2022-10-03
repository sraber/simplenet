// SNSearch.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <ctime>

#include <algorithm>
#include <map>
#include <list>
#include <vector>
#include <utility>
#include <tuple>
#include <random>

using namespace cv;
using namespace std;

#define USHORT unsigned short

typedef pair<USHORT, USHORT> vertex;
typedef vector<vertex> vertex_vector;
// Def: r1,c1,r2,c2,weight
typedef tuple<USHORT, USHORT, USHORT, USHORT, int> edge;
#define edg_r1(EDG) std::get<0>(EDG)
#define edg_c1(EDG) std::get<1>(EDG)
#define edg_r2(EDG) std::get<2>(EDG)
#define edg_c2(EDG) std::get<3>(EDG)
#define edg_w(EDG) std::get<4>(EDG)

static void help() {
    std::cout << std::endl <<
    "Usage:" << std::endl <<
    "./ssearch input_image " << std::endl;
}

struct Segmentataion {
   USHORT Rows;
   USHORT Cols;
   unsigned int SizeBias;  // K in F and H paper.
   Segmentataion(USHORT r, USHORT c, unsigned int k) : Rows(r), Cols(c), SizeBias(k) {}
};

// REVIEW:  Perhaps the way to generalize MaxInternal to say, a spectrum of a image patch,
//          would be to make a different component type.

class Component {
private:
   Segmentataion* pSeg;
   unsigned int ID;
   vertex TopLeft;
   vertex BottomRight;
   unsigned int MinBoarder;
   unsigned int MaxInternal;
   unsigned int Size;
   map<unsigned int, list<vertex>> ColDetail;
   Component() {
      // No calls to uninitialized Component.
   }
   void MergeColumnWith(list<vertex>& vcol, vertex& s) {
      for (vertex& t : vcol) {
         const USHORT a = t.first;
         const USHORT b = t.second;
         // Col range of the source component.
         // Range: [c,d]
         const USHORT c = s.first;
         const USHORT d = s.second;

         // Check for overlap in the ranges.
         // Range: [e,f]  May not exist.
         USHORT e = std::max(a, c);
         USHORT f = std::min(b, d);
         if (e <= f) {
            if (t.first > s.first) { t.first = s.first; }
            if (t.second < s.second) { t.second = s.second; }
            // There was overlap between the source range and one of the column
            // ranges.  That range was expanded to include the source range.
            // Our work is done.  Return to caller.
            return;
         }
      }

      // There was no overlap with the source range.  Add a new range to the column
      // vector.
      vcol.push_back(s);
   }
public:
   Component(unsigned int id, vertex base, Segmentataion* pseg) : 
      pSeg(pseg),
      ID(id), 
      TopLeft(base), BottomRight(base), 
      MinBoarder(0), MaxInternal(0), Size(1){
      ColDetail[base.first].push_back( vertex(base.second, base.second) );
      // Wicked fast, C++ 11, complicated syntax way to do it.
      //ColDetail.emplace( std::piecewise_construct,
      //                   std::forward_as_tuple(base.first),
      //                   std::forward_as_tuple(base.second, base.second) );
   }
   ~Component() {
      //cout << "~Component " << ID << endl;
   }
   inline unsigned int GetID() {
      return ID;
   }
   inline int GetSize() {
      return Size;
   }
   inline int InternalDif() {
      return MaxInternal + (pSeg->SizeBias / Size);
   }
   inline const vertex& GetTopLeft() {
      return TopLeft;
   }
   inline const vertex& GetBottomRight() {
      return BottomRight;
   }
   inline vertex GetRowRange() {
      return vertex(TopLeft.first, BottomRight.first);
   }
   inline const map<unsigned int, list<vertex>>& GetColDetail() {
      return ColDetail;
   }
   bool operator==(Component& cp) {
      return ID == cp.ID;
   }
   bool operator!=(Component& cp) {
      return ID != cp.ID;
   }
   void MergeWith(edge& edg, Component& cp) {
      Size += cp.Size;
      MinBoarder = edg_w(edg);
      // Maintain the internal difference value.
      unsigned int idif = std::max(MaxInternal, cp.MaxInternal);
      // The edge weight between the two components may be larger than the
      // internal difference, just not large enough to overcome the tau(k) offset.
      MaxInternal =  std::max( MinBoarder, idif);

      // Use the following range variables for clarity.
      // Row range of target component.  (The one that will undergo the merge)
      // Range: [a,b]
      const USHORT a = TopLeft.first;
      const USHORT b = BottomRight.first;
      // Row range of the source component.
      // Range: [c,d]
      const USHORT c = cp.TopLeft.first;
      const USHORT d = cp.BottomRight.first;

      // Check for overlap in the ranges.
      // Range: [e,f]  May not exist.
      USHORT e = std::max(a, c);
      USHORT f = std::min(b, d);
      if (e <= f) {
         // There is overlap.  Expand the range of the coloumns of the target
         // to include the range of the source.
         // Copy range is: [e,f]
         for (USHORT r = e; r <= f; r++) {
            list<vertex>& tcol = ColDetail[r];
            list<vertex>& scol = cp.ColDetail[r];
            for (vertex& s : scol) {
               MergeColumnWith(tcol, s);
            }
         }
      }

      // Are there rows in the source below (lower row values) the
      // overlap region.
      if (c < a) {
         // There are lower row numbers in the source that need to be
         // transfered to the target.
         // Copy all the new column spans from the source to the target.
         // Copy range is: [c,a-1]
         for (USHORT r = c; r < a; r++) {
            //ColDetail[r] = cp.ColDetail[r];

            ColDetail[r] = move( cp.ColDetail.at(r) );

            //ColDetail.emplace( std::piecewise_construct,
            //         std::forward_as_tuple(r),
            //         std::forward_as_tuple(cp.ColDetail[r]) );
         }
      }

      // Are there rows in the source above (larger row values) the
      // overlap region.
      if (b < d) {
         // There are larger row numbers in the source that need to be
         // transfered to the target.
         // Copy all the new column spans from the source to the target.
         // Copy range is: [b+1,d]
         for (USHORT r = b+1; r <= d; r++) {
            //ColDetail[r] = cp.ColDetail[r];

            ColDetail[r] = move( cp.ColDetail.at(r) );

            //ColDetail.emplace( std::piecewise_construct,
            //         std::forward_as_tuple(r),
            //         std::forward_as_tuple(cp.ColDetail[r]) );
         }
      }

      TopLeft.first = std::min(a, c);
      BottomRight.first = std::max(b, d);

      TopLeft.second = std::min(TopLeft.second, cp.TopLeft.second);
      BottomRight.second = std::max(BottomRight.second, cp.BottomRight.second);
   }
};

map<unsigned int, Component> Comps;
map<vertex, unsigned int> CompLookup;

struct EdgeWeightComparator
{
    bool operator() (const edge &a, const edge &b)
    {
        return edg_w(a) < edg_w(b);  
    }
};

void Merge(Component& pc1, Component& pc2, edge& edg, unsigned int id1, unsigned int id2) {
   vertex rr = pc2.GetRowRange();
   const map<unsigned int, list<vertex>>& cd = pc2.GetColDetail();

   // Replace pc2 in CompLookup with pc1.
   for (USHORT r = rr.first; r <= rr.second; r++) {
      const list<vertex>& lcc = cd.at(r);
      for (const vertex& cc : lcc) {
         for (USHORT c = cc.first; c <= cc.second; c++) {
            CompLookup.at(vertex(r, c)) = id1;
         }
      }
   }

   // Now merge pc2 into pc1.  pc2 will be invalid.
   pc1.MergeWith(edg, pc2);

   //cout << "merge into: " << id1 <<  " erasing " << id2;
   Comps.erase(id2);
}


int main(USHORT argc, char** argv) {

   // If image path and f/q is not passed as command
   // line arguments, quit and display help message
   if (argc < 1) {
      help();
      return -1;
   }

   vector<edge> EdgeVec;

   // read image
   Mat im = imread(argv[1],cv::IMREAD_GRAYSCALE);

   //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
   /// <summary>
   /// Settings
   /// </summary>
   /// <param name="argc"></param>
   /// <param name="argv"></param>
   /// <returns></returns>
   GaussianBlur(im, im, Size(0, 0), 0.5, 0.5);
   USHORT rows = im.rows;
   const USHORT cols = im.cols;
   const USHORT minimum = 400;
   Segmentataion seg(rows, cols, 1000);
   //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   vertex_vector FrontOffsets;
   vertex_vector MiddleOffsets;
   vertex_vector EndOffsets;
   vertex_vector BottomOffsets;
   vertex_vector& Offsets = MiddleOffsets;

   FrontOffsets.push_back(pair<USHORT, USHORT>(0, 1));
   FrontOffsets.push_back(pair<USHORT, USHORT>(1, 1));
   FrontOffsets.push_back(pair<USHORT, USHORT>(1, 0));

   MiddleOffsets.push_back(pair<USHORT, USHORT>(0, 1));
   MiddleOffsets.push_back(pair<USHORT, USHORT>(1, 1));
   MiddleOffsets.push_back(pair<USHORT, USHORT>(1, 0));
   MiddleOffsets.push_back(pair<USHORT, USHORT>(1, -1));

   EndOffsets.push_back(pair<USHORT, USHORT>(1, 0));
   EndOffsets.push_back(pair<USHORT, USHORT>(1, -1));

   BottomOffsets.push_back(pair<USHORT, USHORT>(0, 1));

   cout << "Building edges and initializing components. " << endl;

   unsigned int id = 1;
   for (USHORT r = 0; r < rows; r++) {
      for (USHORT c = 0; c < cols; c++) {
         //-------------------------------------------------
         // Create one component for each pixel.
         Comps.emplace( std::piecewise_construct,
                        std::forward_as_tuple(id),
                        std::forward_as_tuple(id, vertex(r,c), &seg) );
            
         CompLookup.emplace(std::piecewise_construct,
                              std::forward_as_tuple(r,c),
                              std::forward_as_tuple(id) );

         id++;
         //--------------------------------------------------

         if (r == rows - 1) {
            if (c == cols - 1) {
               continue;
            }
            Offsets = BottomOffsets;
         }
         else if (c == 0) {
            Offsets = FrontOffsets;
         }
         else if (c == cols - 1) {
            Offsets = EndOffsets;
         }
         else {
            Offsets = MiddleOffsets;
         }

         USHORT v1 = im.at<unsigned char>(r,c);
         for (pair<USHORT,USHORT>& pr : Offsets) {
            USHORT r2 = r + pr.first;
            USHORT c2 = c + pr.second;
            USHORT v2 = im.at<unsigned char>(r2, c2);
            unsigned int weight = abs(v1 - v2);
            // emplace_back accepts variadic list of arguments and forwards them directly
            // to the object stored in the container.  Meaning NO COPIES!!
            EdgeVec.emplace_back(r, c, r2, c2, weight);
         }
      }
   }

   cout << "Sorting the edges." << endl;

   std::sort(EdgeVec.begin(), EdgeVec.end(), EdgeWeightComparator() );

   cout << "Merging components." << endl;

   namedWindow("Output", WINDOW_KEEPRATIO | WINDOW_NORMAL);

   for (edge& edg : EdgeVec) {
      unsigned int id1 = CompLookup.at(vertex(edg_r1(edg), edg_c1(edg)));
      unsigned int id2 = CompLookup.at(vertex(edg_r2(edg), edg_c2(edg)));

      if (id1 != id2) {
         Component& pc1 = Comps.at(id1);
         Component& pc2 = Comps.at(id2);

         // REVIEW: This is the boundary test.  To make the code flexible
         //         for experimentation this part needs to be generalized.
         //         InternalDif and initialization also need to be generalized.
         if (edg_w(edg) <= std::min(pc1.InternalDif(), pc2.InternalDif())) {
            // The edge weight between the components is not great enough to
            // justify a boundry.  The components should be merged.

            // REVIEW: These if statements need to be cleaned up.  Consolidate the blocks
            //         (the merge code) into a method call or something.

            if (pc2.GetSize() <= pc1.GetSize()) {
               Merge(pc1, pc2, edg, id1, id2);
            }
            else {
               Merge(pc2, pc1, edg, id2, id1);
            }
         }
      }
   }

   //bool satiated = true;

   for (edge& edg : EdgeVec) {
      unsigned int id1 = CompLookup.at(vertex(edg_r1(edg), edg_c1(edg)));
      unsigned int id2 = CompLookup.at(vertex(edg_r2(edg), edg_c2(edg)));

      if (id1 != id2) {
         Component& pc1 = Comps.at(id1);
         Component& pc2 = Comps.at(id2);
         if (pc1.GetSize() < minimum ) {
            Merge(pc2, pc1, edg, id2, id1);
            // Merge small components.
            //if (pc2.GetSize() < minimum) {
            //   satiated = false;
            //}
         }
         else if(pc2.GetSize() < minimum ) {
            Merge(pc1, pc2, edg, id1, id2);

            // Merge small components.
            //if (pc1.GetSize() < minimum) {
            //   satiated = false;
            //}
         }
      }
   }
 
   Mat imOut(im.rows,im.cols,CV_8UC3);

   cout << "Preparing the picture." << endl;

   std::random_device rd;     // only used once to initialise (seed) engine
   std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
   std::uniform_int_distribution<unsigned short> uni(0,255); // guaranteed unbiased

   for (pair<const unsigned int,Component>& p : Comps) {
      //if (p.second.GetSize() > minimum) {
         uchar rd = static_cast<uchar>( uni(rng) );
         uchar gn = static_cast<uchar>( uni(rng) );
         uchar bl = static_cast<uchar>( uni(rng) );

         auto row = p.second.GetRowRange();
         auto cols = p.second.GetColDetail();

         for (USHORT r = row.first; r <= row.second; r++) {
            list<vertex>& lcc = cols.at(r);
            for (vertex& col : lcc) {
               for (USHORT c = col.first; c <= col.second; c++) {
                  imOut.at<Vec3b>(r, c) = Vec3b(bl, gn, rd);
               }
            }
         }
      //}
   }

   // show output
   imshow("Output", imOut);
   cv::waitKey();
   

   return 0;
}
