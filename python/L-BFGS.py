import sympy
import numpy as np


def twoloop(s, y, rho, gk):
    n = len(s)

    if n >= 1 and type(gk) == np.matrix:
        h0 = float((s[-1].T*y[-1])/(y[-1].T*y[-1]))
    elif n >= 1:
        h0 = (s[-1]*y[-1])/(y[-1]*y[-1])
    else:
        h0 = 1

    a = np.empty((n,))

    if type(gk) == np.matrix:
        q = gk.copy()
    else:
        q = gk

    for i in range(n - 1, -1, -1):
        if type(gk) == np.matrix:
            a[i] = rho[i] * s[i].T * q
        else:
            a[i] = rho[i] * s[i] * q
        q -= a[i] * y[i]
    z = h0*q

    for i in range(n):
        if type(gk) == np.matrix:
            b = rho[i] * y[i].T * z
        else:
            b = rho[i] * y[i] * z
        z += s[i] * (float(a[i] - b))

    return z


def LBFGS_x(f, x0, max_iter, epsilon):
    '''
    description:    LBFGS算法（一元变量）
    param f         要求极值的函数
    param x0        初始位置
    param max_iter  最大迭代次数
    param epsilon   相邻两次迭代的改变量
    return          结束位置
    '''
    i = 0
    x0 = float(x0)
    df = sympy.diff(f, x)
    beta = 0.5
    delta = 0.25
    s, y, rho = [], [], []
    while i < max_iter:
        gk = df.subs(x, x0)
        dk = -twoloop(s, y, rho, gk)

        mk = 0
        while mk < 10:
            if f.subs(x, x0+beta**mk*dk) < f.subs(x, x0) + delta*beta**mk*gk*dk:
                break
            mk += 1
        xnew = x0 + beta**mk*dk

        sk = xnew - x0
        yk = df.subs(x, xnew) - gk

        if sk*yk > 0:
            rho.append(1/(sk*yk))
            s.append(sk)
            y.append(yk)
        if len(rho) > 5:
            rho.pop(0)
            s.pop(0)
            y.pop(0)

        i += 1
        print('迭代第%d次：%.5f' % (i, xnew))
        if abs(f.subs(x, xnew)-f.subs(x, x0)) < epsilon:
            break
        x0 = xnew
    return xnew


def LBFGS_x0x1(f, X0, max_iter, epsilon):
    '''
    description:    LBFGS算法（多元变量）
    param f         要求极值的函数
    param X0        初始位置
    param max_iter  最大迭代次数
    param epsilon   相邻两次迭代的改变量
    return          结束位置
    '''
    i = 0
    X0[0], X0[1] = float(X0[0]), float(X0[1])
    df0 = sympy.diff(f, x0)
    df1 = sympy.diff(f, x1)
    beta = 0.5
    delta = 0.25
    s, y, rho = [], [], []
    while i < max_iter:
        gk = np.mat([float(df0.subs([(x0, X0[0]), (x1, X0[1])])), float(df1.subs([(x0, X0[0]), (x1, X0[1])]))]).T
        dk = -twoloop(s, y, rho, gk)

        mk = 0
        while mk < 10:
            if f.subs([(x0, X0[0]+beta**mk*dk[0, 0]), (x1, X0[1]+beta**mk*dk[1, 0])]) < f.subs([(x0, X0[0]), (x1, X0[1])]) + delta*beta**mk*gk.T*dk:
                break
            mk += 1
        Xnew = [X0[0] + beta**mk*dk[0, 0], X0[1] + beta**mk*dk[1, 0]]

        sk = np.mat([beta**mk*dk[0, 0], beta**mk*dk[1, 0]]).T
        yk = np.mat([float(df0.subs([(x0, Xnew[0]), (x1, Xnew[1])])), float(df1.subs([(x0, Xnew[0]), (x1, Xnew[1])]))]).T - gk

        if sk.T*yk > 0:
            rho.append(1/(sk.T*yk))
            s.append(sk)
            y.append(yk)
        if len(rho) > 5:
            rho.pop(0)
            s.pop(0)
            y.pop(0)

        i += 1
        print('迭代第%d次：[%.5f, %.5f]' % (i, Xnew[0], Xnew[1]))
        if abs(f.subs([(x0, Xnew[0]), (x1, Xnew[1])])-f.subs([(x0, X0[0]), (x1, X0[1])])) < epsilon:
            break
        X0 = Xnew
    return Xnew


if __name__ == '__main__':
    x = sympy.symbols("x")
    x0 = sympy.symbols("x0")
    x1 = sympy.symbols("x1")
    result = LBFGS_x(x**4-4*x, 10, 100, 1e-5)
    print('最终迭代位置：%.5f' % result)
    result = LBFGS_x0x1((x0-1)**2+(x1-1)**4, [10, 10], 100, 1e-5)
    print('最终迭代位置：[%.5f, %.5f]' % (result[0], result[1]))
