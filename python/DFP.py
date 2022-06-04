import sympy
import numpy as np
from numpy import matlib as mb


def DFP_x(func, x0, max_iter, epsilon):
    '''
    description:    DFP算法（一元变量）
    param f         要求极值的函数
    param x0        初始位置
    param max_iter  最大迭代次数
    param epsilon   相邻两次迭代的改变量
    return          结束位置
    '''
    i = 0
    x0 = float(x0)
    d_func = sympy.diff(func, x)
    beta = 0.5
    delta = 0.25
    Hk = 1
    while i < max_iter:
        gk = d_func.subs(x, x0)
        dk = -Hk*gk

        mk = 0
        while mk < 10:
            if func.subs(x, x0+beta**mk*dk) < func.subs(x, x0) + delta*beta**mk*gk*dk:
                break
            mk += 1
        xnew = x0 + beta**mk*dk

        sk = xnew - x0
        yk = d_func.subs(x, xnew) - gk

        if sk*yk > 0:
            Hk = Hk - (Hk*yk*yk*Hk)/(yk*Hk*yk) + (sk*sk)/(sk*yk)

        i += 1
        print('迭代第%d次：%.5f' % (i, xnew))
        if abs(func.subs(x, xnew)-func.subs(x, x0)) < epsilon:
            break
        x0 = xnew
    return xnew


def DFP_x0x1(func, X0, max_iter, epsilon):
    '''
    description:    DFP算法（多元变量）
    param f         要求极值的函数
    param X0        初始位置
    param max_iter  最大迭代次数
    param epsilon   相邻两次迭代的改变量
    return          结束位置
    '''
    i = 0
    X0[0], X0[1] = float(X0[0]), float(X0[1])
    dx0_func = sympy.diff(func, x0)
    dx1_func = sympy.diff(func, x1)
    beta = 0.5
    delta = 0.25
    Hk = mb.identity(len(X0))
    while i < max_iter:
        gk = np.mat([float(dx0_func.subs([(x0, X0[0]), (x1, X0[1])])), float(dx1_func.subs([(x0, X0[0]), (x1, X0[1])]))]).T
        dk = -Hk*gk

        mk = 0
        while mk < 10:
            if func.subs([(x0, X0[0]+beta**mk*dk[0, 0]), (x1, X0[1]+beta**mk*dk[1, 0])]) < func.subs([(x0, X0[0]), (x1, X0[1])]) + delta*beta**mk*gk.T*dk:
                break
            mk += 1
        Xnew = [X0[0] + beta**mk*dk[0, 0], X0[1] + beta**mk*dk[1, 0]]

        sk = np.mat([beta**mk*dk[0, 0], beta**mk*dk[1, 0]]).T
        yk = np.mat([float(dx0_func.subs([(x0, Xnew[0]), (x1, Xnew[1])])), float(dx1_func.subs([(x0, Xnew[0]), (x1, Xnew[1])]))]).T - gk

        if sk.T*yk > 0:
            Hk = Hk - (Hk*yk*yk.T*Hk)/(yk.T*Hk*yk) + (sk*sk.T)/(sk.T*yk)

        i += 1
        print('迭代第%d次：[%.5f, %.5f]' % (i, Xnew[0], Xnew[1]))
        if abs(func.subs([(x0, Xnew[0]), (x1, Xnew[1])])-func.subs([(x0, X0[0]), (x1, X0[1])])) < epsilon:
            break
        X0 = Xnew
    return Xnew


if __name__ == '__main__':
    x = sympy.symbols("x")
    x0 = sympy.symbols("x0")
    x1 = sympy.symbols("x1")
    result = DFP_x(x**4-4*x, 10, 100, 1e-5)
    print('最终迭代位置：%.5f' % result)
    result = DFP_x0x1((x0-1)**2+(x1-1)**4, [10, 10], 100, 1e-5)
    print('最终迭代位置：[%.5f, %.5f]' % (result[0], result[1]))
