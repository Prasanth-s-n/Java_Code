import java.util.ArrayList;
import java.util.List;

import org.jblas.DoubleMatrix;
import java.io.IOException;
import com.github.sh0nk.matplotlib4j.NumpyUtils;
import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import com.github.sh0nk.matplotlib4j.builder.ContourBuilder;

public class Gradient {
	
	public List<Double> arr_with_interval(double s,double interval,double e){
		
		List<Double> fin_lis=new ArrayList<>();
		
		double tot_count=(Math.abs(s)+Math.abs(e))/Math.abs(interval);
		
		for(int i=0;i<tot_count;i++) {
			
			fin_lis.add(s+interval*i);
		
		}
		
		return fin_lis;
	}
	
	
	public double[][] gradient(DoubleMatrix A,DoubleMatrix b,DoubleMatrix xo,int iters) {
		
		double[][] xa=new double[b.length][iters];
        
		for (int i=0;i<iters;i++) {
        	
        	DoubleMatrix g= (A.mmul(xo)).sub(b);
        	
        	//g.print();
        	DoubleMatrix alpha_num =((g.transpose()).mmul(g));
        	DoubleMatrix alpha_den=(((g.transpose()).mmul(A)).mmul(g));
        	
        	double alpha = alpha_num.get(0,0)/alpha_den.get(0,0);
        	
        	DoubleMatrix xn = xo.sub(g.mmuli(alpha));
        	
        	xa[0][i]=xo.get(0,0);
        	
        	xa[1][i]=xo.get(1,0);
        	
        	xo=xn;
        	double err_norm=((A.mmul(xn)).sub(b)).norm2();
        	if (err_norm<=0.0001) {
        		break;
        	}
        }
        
		return xa;
	}
	
	
	
	
	public static void main(String[] args) throws IOException,PythonExecutionException{
		
		
		Gradient m = new Gradient();
		
		List<Double> x = m.arr_with_interval(-4, 0.25, 3);
		
		List<Double> y = m.arr_with_interval(-4, 0.25, 7);
	
		DoubleMatrix A = new DoubleMatrix(new double[][] { {4,2},{2,2} });
        DoubleMatrix b = new DoubleMatrix(new double[][] { {-1},{1}});
        DoubleMatrix xo = new DoubleMatrix(new double[][] {{-3},{7}});
        
        double[][] xa=m.gradient(A,b,xo,10);
        List<Double> xa_1= new ArrayList<Double>();
        List<Double> xa_2=new ArrayList<Double>();
        
        for (int j = 0; j < xa[0].length; j++) { 
            xa_1.add(xa[0][j]); 
            xa_2.add(xa[1][j]);
        }
        NumpyUtils.Grid<Double> grid = NumpyUtils.meshgrid(x, y);
		
        List<List<Double>> zCalced = grid.calcZ((xi, yj) ->(2*xi * xi + yj*yj+ 2*xi*yj +xi-yj));
		
        Plot plt = Plot.create();
		List<Double> l =  NumpyUtils.linspace(1, 100,20);
		ContourBuilder contour = plt.contour().add(x, y, zCalced).levels(l);
		
		plt.plot().add(xa_1,xa_2);
		plt.plot().add(xa_1,xa_2,"*");
		plt.title("Gradient Method");
		plt.show();
        
        
        
        }
}