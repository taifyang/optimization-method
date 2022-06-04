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

double d2f(double x)
{
	//return 12 * pow(x, 2);
	return (df(x + dx) - df(x)) / dx;
}

double f(std::vector<double> X)
{
	return pow(X[0] - 1, 2) + pow(X[1] - 1, 4);
}

double df0(std::vector<double> X)
{
	//return 2 * (X[0] - 1);
	return (f({ X[0] + dx, X[1] }) - f(X)) / dx;
}

double df1(std::vector<double> X)
{
	//return 4 * pow(X[1] - 1, 3);
	return (f({ X[0], X[1] + dx }) - f(X)) / dx;
}

double d2f0(std::vector<double> X)
{
	//return 2;
	return (df0({ X[0] + dx, X[1] }) - df0(X)) / dx;
}

double d2f1(std::vector<double> X)
{
	//return 12 * pow(X[1] - 1, 2);
	return (df1({ X[0] , X[1] + dx }) - df1(X)) / dx;
}

double df0df1(std::vector<double> X)
{
	//return 0;
	return (df1({ X[0] + dx, X[1] }) - df1(X)) / dx;
}

/**
 * @description: 	阻尼牛顿算法（一元变量）
 * @param x0		初始位置
 * @param max_iter	最大迭代次数
 * @param epsilon	相邻两次迭代的改变量
 * @return 			结束位置
 */
double damp_newton_x(double x0, int max_iter, double epsilon)
{
	int i = 0;
	double beta = 0.5;
	double delta = 0.25;
	double xnew;
	while (i < max_iter)
	{
		double gk = df(x0);
		double Gk = d2f(x0);
		double dk =  - gk / Gk;

		int mk = 0;
		while (mk < 10)
		{
			if (f(x0 + pow(beta, mk)*dk) < f(x0) + delta*pow(beta, mk)*gk*dk)
				break;
			++mk;
		}
		xnew = x0 + pow(beta, mk)*dk;

		++i;
		std::cout << "迭代次数：" << i << " " << x0 << std::endl;
		if (abs(f(xnew) - f(x0)) < epsilon)
			break;
		x0 = xnew;
	}
	return xnew;
}

/**
 * @description: 	阻尼牛顿算法（多元变量）
 * @param X0		初始位置
 * @param max_iter	最大迭代次数
 * @param epsilon	相邻两次迭代的改变量
 * @return 			结束位置
 */
std::vector<double> damp_newton_x0x1(std::vector<double> X0, int max_iter, double epsilon)
{
	int i = 0;
	double beta = 0.5;
	double delta = 0.25;
	std::vector<double> Xnew;
	while (i < max_iter)
	{
		Eigen::Vector2f gk;
		gk << df0(X0), df1(X0);
		Eigen::Matrix2f Gk;
		Gk << d2f0(X0), df0df1(X0), df0df1(X0), d2f1(X0);
		Eigen::Vector2f dk = -Gk.inverse()*gk;

		int mk = 0;
		while (mk < 10)
		{
			Xnew = { X0[0] + pow(beta, mk)*dk(0), X0[1] + pow(beta, mk)*dk(1) };
			if (f(Xnew) < f(X0) + delta*pow(beta, mk)*gk.transpose()*dk)
				break;
			++mk;
		}
		Xnew = { X0[0] + pow(beta, mk)*dk(0), X0[1] + pow(beta, mk)*dk(1) };

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
	double result = damp_newton_x(10, 100, 1e-5);
	std::cout << "最终迭代位置：" << result << std::endl;

	std::vector<double> results = damp_newton_x0x1({ 10,10 }, 100, 1e-5);
	std::cout << "最终迭代位置：" << results[0] << " " << results[1] << std::endl;

	system("pause");
	return EXIT_SUCCESS;
}

