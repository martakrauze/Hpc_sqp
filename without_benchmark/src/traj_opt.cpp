#include <eigen3/Eigen/Dense>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <autodiff/forward/dual.hpp>
#include <iostream>

// #include <MakeRandomVector.hpp>

////// Parametry modelu: //////
// Masy, długości i momenty bezwładności członów
double l1=1;
double m1=1;
double J1=1./12. * m1 * l1 * l1;
double l2=1;
double m2=1;
double J2=1./12. * m2 * l2 * l2;
double l3=1;
double m3=1;
double J3=1./12. * m3 * l3 * l3;

// Położenie początkowe i końcowe końcówki ostatniego członu
double xp=2;
double yp=0.5;
double xk=2.5;
double yk=1;

// Czas symulacji
double T=1;

// Liczba kroków czasowych
const int n=16;

// Liczba iteracji algorytmu SQP
int max_iter=20;

////// Koniec parametrów //////

// Dyskretyzacja czasu
double h=T/double(n-1);
Eigen::Matrix<double, n, 1> t;


// Macierz masowa
autodiff::MatrixXreal M(9,9);

// Wektor długości
double l[3]={l1,l2,l3};

// Wektory od środka masy do złącz
Eigen::Matrix<double, 2, 1> P[3];
Eigen::Matrix<double, 2, 1> S[3];

Eigen::Matrix<double, 2, 2> Q;

autodiff::VectorXreal d(int part, int joint, autodiff::VectorXreal P_g[], autodiff::VectorXreal S_g[])
{
   autodiff::VectorXreal r(2);
   r << 0,0;
   for(int i=joint; i<=part; i++)
   {
      r=r-P_g[i]+S_g[i];
   }
   r=r-S_g[part];
   r=Q*r;
   return r;
}

autodiff::VectorXreal d_dot(int part, int joint, autodiff::VectorXreal P_g[], autodiff::VectorXreal S_g[], autodiff::VectorXreal& fidot)
   {
      autodiff::VectorXreal r(2);
        r << 0,0;
      for(int i=joint; i<=part; i++)
      {
         r=r+P_g[i]*fidot(i)-S_g[i]*fidot(i);
      }
      r=r+S_g[part]*fidot(part);
      return r;
   }

autodiff::MatrixXreal R(autodiff::real fi)
{
    autodiff::MatrixXreal rot(2,2);
    rot<<cos(fi), -sin(fi),
         sin(fi), cos(fi);
    return rot;
}

autodiff::VectorXreal dynamics_function(autodiff::VectorXreal x,autodiff::VectorXreal u)
{ 

   autodiff::VectorXreal g(9);
   g << 0, 0, u(0)-u(1), 0, 0, u(1)-u(2), 0, 0, u(2);
   
   autodiff::VectorXreal thetadot(3);
   thetadot << x(3), x(4), x(5);
   autodiff::VectorXreal fi(3);
   fi << x(0), x(1)+x(0),x(2)+x(1)+x(0);
   autodiff::VectorXreal fidot(3);
   fidot << x(3), x(3)+x(4), x(3)+x(4)+x(5);

   autodiff::VectorXreal P_g[3];
   autodiff::VectorXreal S_g[3];
   
   
     for(int part=0;part<3;part++)
   {
      P_g[part]=R(fi(part))*P[part];
      S_g[part]=R(fi(part))*S[part];
   }

   autodiff::MatrixXreal B(9,3);
   autodiff::VectorXreal zeros(2,1);
   zeros << 0,0;
    
   B <<d(0,0,P_g,S_g), zeros, zeros,
        1, 0, 0,
       d(1,0,P_g,S_g), d(1,1,P_g,S_g), zeros,
          1, 1, 0,
       d(2,0,P_g,S_g), d(2,1,P_g,S_g),d(2,2,P_g,S_g),
       1,1,1;  
    
   autodiff::MatrixXreal Bdot(9,3);

   Bdot<<d_dot(0,0,P_g,S_g,fidot), zeros, zeros,
         0, 0, 0,
         d_dot(1,0,P_g,S_g,fidot), d_dot(1,1,P_g,S_g,fidot), zeros,
         0, 0, 0,
         d_dot(2,0,P_g,S_g,fidot), d_dot(2,1,P_g,S_g,fidot),d_dot(2,2,P_g,S_g,fidot),
         0, 0, 0;
    
   autodiff::MatrixXreal M_new(3,3);
   autodiff::VectorXreal g_new(3);

    M_new=B.transpose() * M * B;
    
    g_new=B.transpose() * (g - M * Bdot * thetadot);

   autodiff::VectorXreal r4;
   r4=M_new.colPivHouseholderQr().solve(g_new);

   autodiff::VectorXreal r(6);
   
   r << x(3),x(4),x(5),r4;
   
   return r;
}


autodiff::real f_celu(autodiff::VectorXreal& x)
{
    autodiff::real obj = 0;
    autodiff::VectorXreal u1 = x(Eigen::seq(6*n,7*n-1));
    autodiff::VectorXreal u2 = x(Eigen::seq(7*n,8*n-1));
    autodiff::VectorXreal u3 = x(Eigen::seq(8*n,9*n-1));

    for(int i=0; i<n-1; i++)
    {
        obj = obj + 1./2. * h * (u1(i)*u1(i) + u1(i+1)*u1(i+1) + u2(i)*u2(i) + u2(i+1)*u2(i+1) + u3(i)*u3(i) + u3(i+1)*u3(i+1));
    }
    return obj;
}

autodiff::VectorXreal f_ogr(autodiff::VectorXreal& x)
{
    autodiff::VectorXreal theta1 = x(Eigen::seq(0*n,1*n-1));
    autodiff::VectorXreal theta2 = x(Eigen::seq(1*n,2*n-1));
    autodiff::VectorXreal theta3 = x(Eigen::seq(2*n,3*n-1));
    autodiff::VectorXreal thetadot1 = x(Eigen::seq(3*n,4*n-1));
    autodiff::VectorXreal thetadot2 = x(Eigen::seq(4*n,5*n-1));
    autodiff::VectorXreal thetadot3 = x(Eigen::seq(5*n,6*n-1));
    
    autodiff::MatrixXreal u(n,3);
    u(Eigen::all,0)=x(Eigen::seq(6*n,7*n-1));
    u(Eigen::all,1)=x(Eigen::seq(7*n,8*n-1));
    u(Eigen::all,2)=x(Eigen::seq(8*n,9*n-1));
    autodiff::MatrixXreal q(n,6);
    q(Eigen::all,0)=theta1;
    q(Eigen::all,1)=theta2;
    q(Eigen::all,2)=theta3;
    q(Eigen::all,3)=thetadot1;
    q(Eigen::all,4)=thetadot2;
    q(Eigen::all,5)=thetadot3;
    
    
    autodiff::VectorXreal r(10);
     r << thetadot1(0), thetadot1(n-1), thetadot2(0), thetadot2(n-1), thetadot3(0), thetadot3(n-1), l1*cos(theta1(n-1))+l2*cos(theta2(n-1)+theta1(n-1))+l3*cos(theta3(n-1)+theta2(n-1)+theta1(n-1))-xk, l1*sin(theta1(n-1))+l2*sin(theta2(n-1)+theta1(n-1))+l3*sin(theta3(n-1)+theta2(n-1)+theta1(n-1))-yk, l1*cos(theta1(0))+l2*cos(theta2(0)+theta1(0))+l3*cos(theta3(0)+theta2(0)+theta1(0))-xp, l1*sin(theta1(0))+l2*sin(theta2(0)+theta1(0))+l3*sin(theta3(0)+theta2(0)+theta1(0))-yp; // Warunki brzegowe

    autodiff::VectorXreal vec(6*(n-1)+10);
    for(int i=0; i<n-1; i++)
    {
        vec(Eigen::seq(6*i,6*(i+1)-1)) = q(i+1,Eigen::all).transpose()-q(i,Eigen::all).transpose()-1./2.*h*(dynamics_function(q(i+1,Eigen::all).transpose(),u(i+1, Eigen::all).transpose())+dynamics_function(q(i,Eigen::all).transpose(),u(i,Eigen::all).transpose()));
    }
    vec(Eigen::seq(6*(n-1),6*(n-1)+9))=r;
    
    return vec;
}

autodiff::real Lagran(autodiff::VectorXreal& x, autodiff::VectorXreal& lambda)
{
    autodiff::real ret;
    autodiff::real a;
    a=lambda.transpose()*f_ogr(x);
    ret=f_celu(x)+a;
    return ret;
}

autodiff::real skalarny(autodiff::VectorXreal x, autodiff::VectorXreal y, int len)
{
    autodiff::real ret=0;
    for(int i=0; i<len; i++)
    {
        ret=ret+x(i)*y(i);
    }
    return ret;
}

int wypisz_x(autodiff::VectorXreal x)
{
    std::ofstream out("results.m");
    out<<"# Created by traj_opt"<<"\n"<<"# name: x"<<"\n"<<"# type: matrix"<<"\n"<<"# rows: "<<9*n<<"\n"<<"# columns: 1"<<"\n";
    for(int i=0; i<9*n; i++)
    {
        out<<" "<<x(i)<<"\n";
    }
    out.close();
    return 0;
}


int main()
{
    M(0,0)=m1; M(1,1)=m1; M(2,2)=J1; M(3,3)=m2; M(4,4)=m2; M(5,5)=J2; M(6,6)=m3; M(7,7)=m3; M(8,8)=J3;

    for(int i=0;i<n;i++)
    {
        t(i)=double(i)*h;
    }

    for(int part=0;part<3;part++)
    {
        P[part] << -l[part]/2,0.;
        S[part] << +l[part]/2,0.;
    }

    Q << 0, -1,
        1, 0;

    //Przybliżenie początkowe

    Eigen::Matrix<double, n, 1> theta10;
    Eigen::Matrix<double, n, 1> theta20;
    Eigen::Matrix<double, n, 1> theta30;
    Eigen::Matrix<double, n, 1> zeros;
    Eigen::Matrix<double, n, 1> ones;
    for (int i=0; i<n; i++)
    {
        zeros(i)=0.;
        ones(i)=1.;
    }

    

    theta10=0.4151*ones+t/T*(0.4037-0.4151);
    theta20=0.6687*ones+t/T*(0.5246-(0.6687));
    theta30=-1.9897*ones+t/T*(-1.1228-(-1.9897));

    autodiff::VectorXreal x(9*n);
    x << theta10,theta20,theta30,zeros,zeros,zeros,ones,ones,ones;

    //Przybliżenie początkowe mnożników lagrange'a
    autodiff::VectorXreal lambda(6*(n-1)+10);
    for (int i=0; i<6*(n-1)+10; i++)
    {
        lambda(i)=0;
    }

    //Macierz z zerami :-)
    Eigen::MatrixXd zerosM(6*(n-1)+10,6*(n-1)+10);
    for (int i=0; i<6*(n-1)+10; i++)
    {
        for (int j=0; j<6*(n-1)+10; j++)
        {
            zerosM(i,j)=0.;
        }
    }
    
    //Deklaracja różnych rzeczy
    autodiff::VectorXreal g;
    autodiff::MatrixXreal gx;
    autodiff::real J;
    autodiff::VectorXreal Jx;
    autodiff::real L;
    autodiff::MatrixXreal Lx;
    autodiff::MatrixXreal A(6*(n-1)+10+9*n,6*(n-1)+10+9*n);
    autodiff::VectorXreal b(6*(n-1)+10+9*n);
    autodiff::VectorXreal c(6*(n-1)+10+9*n);
    autodiff::MatrixXreal Lxx(9*n,9*n);

    //Zerowe przybliżenie Hessiana macierzą jednostkową 
    for (int i=0; i<9*n; i++)
    {
        for (int j=0; j<9*n; j++)
        {
            Lxx(i,j)=0.;
        }
        Lxx(i,i)=1.;
    }

    //Zerowe obliczenie funkcji lagrange'a i jej gradientu
    Lx=autodiff::gradient(Lagran, autodiff::wrt(x), autodiff::at(x,lambda), L);

    //Iteracje algorytmu sqp
    for(int iter=0; iter<max_iter; iter++)
    {
    
    gx=autodiff::jacobian(f_ogr, autodiff::wrt(x), autodiff::at(x), g);

    
    Jx=autodiff::gradient(f_celu, autodiff::wrt(x), autodiff::at(x), J);
    
    //std::cout<<J<<"\n";

    //Tworzenie macierzy i wektora prawych stron do układu równań
    A<< Lxx, -gx.transpose(),
        gx, zerosM;
    b << -Jx, -g;

    //Rozwiązywanie układu równań
    c=A.colPivHouseholderQr().solve(b);
    
    //Nowe przybliżenie x i lambda
    auto x_old=x;
    x=x+c(Eigen::seq(0,9*n-1));
    lambda=c(Eigen::seq(9*n,6*(n-1)+9+9*n));


    auto sk=x-x_old;
    auto Lx_old=Lx;
    Lx=autodiff::gradient(Lagran, autodiff::wrt(x), autodiff::at(x,lambda), L);
    auto yk=Lx-Lx_old;
    auto Lxx_old=Lxx;
 
    //Nowe przybliżenie hessianu funkcji Lagrange'a:
    Lxx=Lxx_old+yk*(yk.transpose())/(skalarny(sk,yk,9*n))-Lxx_old*sk*(sk.transpose())*Lxx_old/(skalarny(sk,Lxx_old*sk,9*n));

    }
    //std::cout<<"Dziala";
    wypisz_x(x);
    return 0;
}

