r=linspace(-2,2,100);
[xx,yy]=meshgrid(r+0.102287,r-1.67881);
dr=linspace(-10,10,100);
[dx,dy]=meshgrid(dr,dr);

ds=dlmread("C:\\projects\\neuralnet\\simplenet\\SNLogic\\ds.csv");
es1=dlmread("C:\\projects\\neuralnet\\simplenet\\SNLogic\\es.1.csv");
es2=dlmread("C:\\projects\\neuralnet\\simplenet\\SNLogic\\es.2.csv");
es3=dlmread("C:\\projects\\neuralnet\\simplenet\\SNLogic\\es.3.csv");
es4=dlmread("C:\\projects\\neuralnet\\simplenet\\SNLogic\\es.4.csv");
est = (es1 + es2 + es3 + es4)/4;

nsurf(xx,yy,est,'W row','W col', 'Avg Error Surface')
nsurf(xx,yy,es1,'W row','W col', 'P1 Error Surface')
nsurf(xx,yy,es2,'W row','W col', 'P2 Error Surface')
nsurf(xx,yy,es3,'W row','W col', 'P3 Error Surface')
nsurf(xx,yy,es4,'W row','W col', 'P4 Error Surface')

nsurf(dx,dy,ds,'X','Y', 'Decision Surface')