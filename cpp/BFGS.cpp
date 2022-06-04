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
	return (f({X[0] + dx, X[1]}) - f(X)) / dx;
}

double df1(std::vector<double> X)
{
	//return 4 * pow(X[1] - 1, 3);
	return (f({X[0], X[1] + dx}) - f(X)) / dx;
}

/**
 * @description: 	BFGS算法（一元变量）
 * @param x0		初始位置
 * @param max_iter	最大迭代次数
 * @param epsilon	相邻两次迭代的改变量
 * @return 			结束位置
 */
double BFGS_x(double x0, int max_iter, double epsilon)
{
	int i = 0;
	double beta = 0.5;
	double delta = 0.25;
	double Bk = 1;
	double xnew;
	while (i < max_iter)
	{
		double gk = df(x0);
		double dk = -gk / Bk;
		int mk = 0;
		while (mk < 10)
		{
			if (f(x0 + pow(beta, mk) * dk) < f(x0) + delta * pow(beta, mk) * gk * dk)
				break;
			++mk;
		}
		xnew = x0 + pow(beta, mk) * dk;

		float sk = xnew - x0;
		float yk = df(xnew) - gk;

		if (sk * yk > 0)
			Bk = Bk - (Bk * sk * sk * Bk) / (sk * Bk * sk) + (yk * yk) / (yk * sk);

		++i;
		std::cout << "迭代次数：" << i << " " << x0 << std::endl;
		if (abs(f(xnew) - f(x0)) < epsilon)
			break;
		x0 = xnew;
	}
	return xnew;
}

/**
 * @description: 	修正牛顿算法（多元变量）
 * @param X0		初始位置
 * @param max_iter	最大迭代次数
 * @param epsilon	相邻两次迭代的改变量
 * @return 			结束位置
 */
std::vector<double> BFGS_x0x1(std::vector<double> X0, int max_iter, double epsilon)
{
	int i = 0;
	double beta = 0.5;
	double delta = 0.25;
	Eigen::Matrix2d Bk = Eigen::Matrix2d::Identity();
	std::vector<double> Xnew;
	while (i < max_iter)
	{
		Eigen::Vector2d gk;
		gk << df0(X0), df1(X0);
		Eigen::Vector2d dk = -Bk.inverse() * gk;

		int mk = 0;
		while (mk < 10)
		{
			Xnew = {X0[0] + pow(beta, mk) * dk(0), X0[1] + pow(beta, mk) * dk(1)};
			if (f(Xnew) < f(X0) + delta * pow(beta, mk) * gk.transpose() * dk)
				break;
			++mk;
		}
		Xnew = {X0[0] + pow(beta, mk) * dk(0), X0[1] + pow(beta, mk) * dk(1)};

		Eigen::Vector2d sk;
		sk << pow(beta, mk) * dk(0), pow(beta, mk) * dk(1);
		Eigen::Vector2d yk;
		yk << df0(Xnew), df1(Xnew);
		yk -= gk;

		if (sk.transpose() * yk > 0)
			Bk = Bk - (Bk * sk * sk.transpose() * Bk) / (sk.transpose() * Bk * sk) + (yk * yk.transpose()) / (yk.transpose() * sk);

		++i;
		std::cout << "迭代次数：" << i << " " << X0[0] << " " << X0[1] << std::endl;
		if (abs(f(Xnew) - f(X0)) < epsilon)
			break;
		X0 = Xnew;
	}
	return X0;
}

int main(int argc, char *argv[])
{
	double result = BFGS_x(10, 100, 1e-5);
	std::cout << "最终迭代位置：" << result << std::endl;

	std::vector<double> results = BFGS_x0x1({10, 10}, 100, 1e-5);
	std::cout << "最终迭代位置：" << results[0] << " " << results[1] << std::endl;

	system("pause");
	return EXIT_SUCCESS;
}
