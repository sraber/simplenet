typedef  std::vector<Matrix> matrix_map;

void PassMapMat(matrix_map& mm)
{
   cout << mm[1];
}

void TestVectorOfMatrix()
{
   matrix_map mat_map(3);
   
   for (int i = 0; i < 3; i++) {
      mat_map[i].resize(3, 3);
   }

   for (Matrix& mm : mat_map) {
      mm.setRandom();
   }

   //ColVector x(3), y(3);
   //x.setRandom();

   //y = mat_map[1] * x;

   cout << mat_map[1] << endl << endl;
   PassMapMat(mat_map);
}

ColVector make_binary_label(ColVector multi_class_label, int positive_label)
{
   int label = GetLabel(multi_class_label);  
   ColVector y(2);
   if (label == positive_label) {
      y(0) = 1.0;
      y(1) = 0.0;   // This is the "not" column.
   }
   else {
      y(0) = 0.0;
      y(1) = 1.0;   // This is the "not" column.
   }
   return y;
}   


int train = 2;

   {
      MNISTReader::MNIST_list mr = reader.read_batch(10);
      TrasformMNISTtoMatrix(fl2.W, mr[5].x);
      ScaleToOne(fl2.W.data(), fl2.W.rows() * fl2.W.cols());
      std::cout << "init to " << GetLabel(mr[5].y);
      MakeMNISTImage("C:\\projects\\neuralnet\\simplenet\\SNCVMNIST\\init_w.bmp", fl2.W);

         y = make_binary_label(mr[5].y, train);
         TrasformMNISTtoMatrix(m, mr[5].x);
         ScaleToOne(m.data(), m.rows() * m.cols());

         Matrix convo_out = fl2.Eval(m);

         // Flatten the convo result.
         Eigen::Map<ColVector> v(convo_out.data(), convo_out.size());

      for (int n = 0; n < 1000; n++) {

         ColVector cv = v;
         for (auto lli : LayerList) {
            cv = lli->Eval(cv);
         }

         double e = loss.Eval(cv, y);

         RowVector g = loss.LossGradiant();
         for (layer_list::reverse_iterator riter = LayerList.rbegin();
            riter != LayerList.rend();
            riter++) {
            g = (*riter)->BackProp(g);
         }
         double eta = 1.0;
         for (auto lit : LayerList) {
            lit->Update(eta);
         }
      }
   }
   
   //-------------------------------------------------------------

   for (int c = 0; c < 100; c++) {
      MNISTReader::MNIST_list dl = reader.read_batch(60);
      for (int loop = 0; loop < 2; loop++) {
         for (int n = 0; n < 60; n++) {
            y = make_binary_label(dl[n].y, train);
            TrasformMNISTtoMatrix(m, dl[n].x);
            ScaleToOne(m.data(), m.rows() * m.cols());

            Matrix convo_out = fl2.Eval(m);

            // Flatten the convo result.
            Eigen::Map<ColVector> v(convo_out.data(), convo_out.size());

            ColVector cv = v;
            for (auto lli : LayerList) {
               cv = lli->Eval(cv);
            }

            double e = loss.Eval(cv, y);

            RowVector g = loss.LossGradiant();
            for (layer_list::reverse_iterator riter = LayerList.rbegin();
               riter != LayerList.rend();
               riter++) {
               g = (*riter)->BackProp(g);
            }

            // Reflate the matrix.
            Matrix fatter(28, 28);
            double* pmd = fatter.data();
            double* pmde = pmd + (fatter.cols() * fatter.rows());
            double* pv = g.data();
            for (; pmd < pmde; pmd++, pv++) {
               *pmd = *pv;
            }

            fl2.BackProp(fatter, false);
         }
         double eta = 1.0;
         fl2.Update(eta);
         for (auto lit : LayerList) {
            lit->Update(eta);
         }
      }
   }

   Matrix im(fl2.W.rows(), fl2.W.cols());
   im = fl2.W;
   ScaleToOne(im.data(), im.cols() * im.rows());
   
   MakeMNISTImage("C:\\projects\\neuralnet\\simplenet\\SNCVMNIST\\corr.bmp", im);

   loss.ResetCounters();
   MNISTReader::MNIST_list dl = reader.read_batch(60);
   for (auto dli : dl) {
         y = make_binary_label(dli.y, train);
         TrasformMNISTtoMatrix(m, dli.x);
         ScaleToOne(m.data(), m.rows() * m.cols());

         Matrix convo_out = fl2.Eval(m);

         // Flatten the convo result.
         Eigen::Map<ColVector> v(convo_out.data(), convo_out.size());

         ColVector cv = v;
         for (auto lli : LayerList) {
            cv = lli->Eval(cv);
         }

         double e = loss.Eval(cv, y);
      /*
      if (++count == 10) {
         count = 0;
         std::cout << " correct/incorrect " << loss.Correct << " , " << loss.Incorrect << endl;
      }
      */
   }
   std::cout << " correct/incorrect " << loss.Correct << " , " << loss.Incorrect << endl;