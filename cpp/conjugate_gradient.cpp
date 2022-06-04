#include <iostream>
#include <vector>
#include <Eigen/Dense>

const double dx = 1e-3;

double f(double x)
{
	return pow(x, 4) - 4 * x;
}

double df(double x)
{
	//return 4 * pow(x, 3) - 4;
	return (f(x + dx) - f(x)) / dx;
}

double f(std::vector<double> X)
{
	return pow(X[0] - 1, 2) + pow(X[1] - 1, 4);
}

double df0(std::vector<double> X)
{
	//return 2 * (X[0] - 1);
	return (f({ X[0] + dx,X[1] }) - f(X)) / dx;
}

double df1(std::vector<double> X)
{
	//return 4 * pow(X[1] - 1, 3);
	return (f({ X[0], X[1] + dx }) - f(X)) / dx;
}

/**
 * @description: 	共轭梯度算法（一元变量）
 * @param x0		初始位置
 * @param max_iter	最大迭代次数
 * @param epsilon	相邻两次迭代的改变量
 * @return 			结束位置
 */
double conjugate_gradient_x(double x0, int max_iter, double epsilon)
{
	int i = 0;
	double alpha = 0.001;
	double beta = 0.5;
	double delta = 0.25;
	double xnew;
	while (i < max_iter)
	{
		double gk = df(x0);
		double dk = -gk;

		//xnew = x0 + alpha*dk;
		int mk = 0;
		while (mk < 10)
		{
			if (f(x0 + pow(beta, mk)*dk) < f(x0) + delta*pow(beta, mk)*gk*dk)
				break;
			++mk;
		}
		xnew = x0 + pow(beta, mk)*dk;

		double gknew = df(xnew);
		double betak = pow(gknew, 2) / pow(gk, 2);
		double dknew = -gknew + betak*dk;
		//xnew += alpha*dknew;
		xnew += pow(beta, mk)*dknew;

		++i;
		std::cout << "迭代次数：" << i << " " << x0 << std::endl;
		if (abs(f(xnew) - f(x0)) < epsilon)
			break;
		x0 = xnew;
	}
	return xnew;
}

/**
 * @description: 	共轭梯度算法（多元变量）
 * @param X0		初始位置
 * @param max_iter	最大迭代次数
 * @param epsilon	相邻两次迭代的改变量
 * @return 			结束位置
 */
std::vector<double> conjugate_gradient_x0x1(std::vector<double> X0, int max_iter, double epsilon)
{
	int i = 0;
	double alpha = 0.001;
	double beta = 0.5;
	double delta = 0.25;
	std::vector<double> Xnew;
	while (i < max_iter)
	{
		Eigen::Vector2d gk = { df0(X0), df1(X0) };
		Eigen::Vector2d dk = -gk;

		//Xnew = { X0[0] + alpha*dk(0), X0[1] + alpha*dk(1) };
		int mk = 0;
		while (mk < 10)
		{
			Xnew = { X0[0] + pow(beta, mk)*dk(0), X0[1] + pow(beta, mk)*dk(1) };
			if (f(Xnew) < f(X0) + delta*pow(beta, mk)*gk.transpose()*dk)
				break;
			++mk;
		}
		Xnew = { X0[0] + pow(beta, mk)*dk(0), X0[1] + pow(beta, mk)*dk(1) };

		Eigen::Vector2d gknew = { df0(X0), df1(X0) };
		double betak = (gknew.transpose()*gknew)(0, 0) / (gk.transpose()*gk)(0, 0);
		Eigen::Vector2d dknew = -gknew + betak*dk;
		//Xnew = { Xnew[0] + alpha*dk(0), Xnew[1] + alpha*dk(1) };
		Xnew = { Xnew[0] + pow(beta, mk)*dk(0), Xnew[1] + pow(beta, mk)*dk(1) };

		++i;
		std::cout << "迭代次数：" << i << " " << X0[0] << " " << X0[1] << std::endl;
		if (abs(f(Xnew) - f(X0)) < epsilon)
			break;
		X0 = Xnew;
	}
	return X0;
}

int main(int argc, char* argv[])
{
	double result = conjugate_gradient_x(10, 100, 1e-5);
	std::cout << "最终迭代位置：" << result << std::endl;

	std::vector<double> results = conjugate_gradient_x0x1({ 10,10 }, 100, 1e-5);
	std::cout << "最终迭代位置：" << results[0] << " " << results[1] << std::endl;

	system("pause");
	return EXIT_SUCCESS;
}

