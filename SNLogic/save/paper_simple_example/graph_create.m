r=linspace(-2,2,99);
[xx,yy]=meshgrid(r-1,r-1);
dr=linspace(-10,10,99);
[dx,dy]=meshgrid(dr,dr);

ds=dlmread("C:\\projects\\neuralnet\\simplenet\\SNLogic\\ds.csv");
es1=dlmread("C:\\projects\\neuralnet\\simplenet\\SNLogic\\es.1.csv");
es2=dlmread("C:\\projects\\neuralnet\\simplenet\\SNLogic\\es.2.csv");
es3=dlmread("C:\\projects\\neuralnet\\simplenet\\SNLogic\\es.3.csv");

est = (es1 + es2 + es3 )/3;

nsurf(xx,yy,est,'W0','W1', 'Avg Error Surface')
nsurf(xx,yy,es1,'W0','W1', 'P1 Error Surface')
nsurf(xx,yy,es2,'W0','W1', 'P2 Error Surface')
nsurf(xx,yy,es3,'W0','W1', 'P3 Error Surface')


nsurf(dx,dy,ds,'X','Y', 'Decision Surface')