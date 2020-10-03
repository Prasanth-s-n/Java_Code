package MIS3_Assignment;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import org.jblas.DoubleMatrix;


public class MIS3_ConjGrad_A3 {
	

	public List<Double> arr_with_interval(double s,double interval,double e){
			
			List<Double> fin_lis=new ArrayList<>();
			
			double tot_count=(Math.abs(s)+Math.abs(e))/Math.abs(interval);
			
			for(int i=0;i<tot_count;i++) {
				
				fin_lis.add(s+interval*i);
			
			}
			
			return fin_lis;
	}
	
	public void print_mat(DoubleMatrix mat) {
		//For traversing through rows
		for(int i =0;i<mat.rows;i++) {
			//For traversing through colums
			for(int j=0;j<mat.columns;j++) {
				
				System.out.print("\t"+mat.get(i,j)+"\t");
				
			}
			System.out.println();
		}
		
	}
	
	public DoubleMatrix mat_mulScal(DoubleMatrix S , double scalar) {
			
			double[][] ret_mat = new double[S.rows][S.columns];
			
			for (int i=0;i<S.rows;i++) {
				
				for(int j=0;j<S.columns;j++) {
					
					ret_mat[i][j] = S.get(i,j)*scalar;
					
				}
				
			}
			
			return new DoubleMatrix(ret_mat);
	}
	
	
	public DoubleMatrix rand_matrix(int[] range,int m,int n) {
		
		double[][] ret_mat = new double[m][n];
		
		
		for (int i=0;i<m;i++) {
			for(int j=0;j<n;j++) {
				
				ret_mat[i][j] = ThreadLocalRandom.current().nextInt(range[0],range[1]+1);
				
			}
		}
		//new DoubleMatrix(ret_mat).print();
		return new DoubleMatrix(ret_mat);
		
		
	}
	
	public three_mat conjugate_gradient(DoubleMatrix A,DoubleMatrix b,DoubleMatrix x){
		
		double[][] x_mat = new double[A.rows][b.length];
		double[][] p_mat = new double[A.rows][b.length];
		double[][] r_mat = new double[A.rows][b.length];
		DoubleMatrix r = b.sub(A.mmul(x));
		DoubleMatrix d = r;
		DoubleMatrix rold = r.transpose().mmul(r);
		double rsold = rold.get(0,0);
		for(int i=1;i<=b.length;i++) {
			
			DoubleMatrix Ap = A.mmul(d);
			double alpha = rsold/d.transpose().mmul(Ap).get(0,0);
			x = x.add(mat_mulScal(d,alpha));
			r = r.sub(mat_mulScal(Ap,alpha));
			double rnew = r.transpose().mmul(r).get(0,0);
			for (int j = 0;j<x.rows;j++) {
				x_mat[j][i-1] = x.get(j,0);
				p_mat[j][i-1] = d.get(j,0);
				r_mat[j][i-1] = r.get(j,0);
				}
			if (Math.sqrt(rnew)<=1e-10) {
				break;
			}
			d = r.add(mat_mulScal(d,(rnew/rsold)));
			rsold = rnew;
					}
		return new three_mat(x_mat,p_mat,r_mat);
	}
	
	public void prove_perpendicular(DoubleMatrix residuals,DoubleMatrix directions) {
		int rn = residuals.rows;
		for (int i=1;i<rn;i++) {
			for(int j=0;j<i;j++) {
				System.out.println("Dot product of r"+i+ " and d"+j+" is "+residuals.getColumn(i).transpose().mmul(directions.getColumn(j)) );
			}
		}
	}
	
	public static void main(String[] args){
		
		
		MIS3_ConjGrad_A3 m = new MIS3_ConjGrad_A3();
		
        int n = 4;
        int[] range1 = {-1,2};
        DoubleMatrix A =m.rand_matrix(range1,n,n);
        A  = A.transpose().mmul(A);
        int[] range2 = {-9,9};
        DoubleMatrix b = m.rand_matrix(range2, n,1);
        DoubleMatrix xo = m.rand_matrix(range2, n,1);
        
        three_mat xpr = m.conjugate_gradient(A, b, xo);
        DoubleMatrix x_co  = new DoubleMatrix(xpr.x);
        DoubleMatrix D_co = new DoubleMatrix(xpr.p);
        DoubleMatrix r_co = new DoubleMatrix(xpr.r);
        System.out.println("A*x is");
        m.print_mat(A.mmul((x_co.getColumn(n-1))));
        System.out.println();
        System.out.println("b is");
        m.print_mat(b);
        System.out.println();
        System.out.println("Dt*A*D matrix is");
        m.print_mat(D_co.transpose().mmul(A.mmul(D_co)));
        System.out.println();
        System.out.println("Resvt*Resv matrix is");
        m.print_mat(r_co.transpose().mmul(r_co));
        m.prove_perpendicular(r_co, D_co);//checking whether residuals and directions are perpendicular.
        
	}

}

class three_mat {
	
	public double[][] x;
	public double[][] p;
	public double[][] r;
	public three_mat(double[][] x,double [][]y,double [][]z) {
		
		
		this.x = x;
		this.p = y;
		this.r = z;
		
		
	}

}
 
