import sympy
import numpy as np
from numpy import matlib as mb


def Broyden_x(f, x0, max_iter, epsilon):
    '''
    description:    Broyden算法（一元变量）
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
    Hk = 1
    while i < max_iter:
        gk = df.subs(x, x0)
        dk = -Hk*gk

        mk = 0
        while mk < 10:
            if f.subs(x, x0+beta**mk*dk) < f.subs(x, x0) + delta*beta**mk*gk*dk:
                break
            mk += 1
        xnew = x0 + beta**mk*dk

        sk = xnew - x0
        yk = df.subs(x, xnew) - gk

        phik = 0.5
        vk = (yk*Hk*yk)**0.5*(sk/(yk*sk)-(Hk*yk)/(yk*Hk*yk))
        Hk = Hk - (Hk*yk*yk*Hk)/(yk*Hk*yk) + (sk*sk)/(sk*yk) + phik*vk*vk

        i += 1
        print('迭代第%d次：%.5f' % (i, xnew))
        if abs(f.subs(x, xnew)-f.subs(x, x0)) < epsilon:
            break
        x0 = xnew
    return xnew


def Broyden_x0x1(f, X0, max_iter, epsilon):
    '''
    description:    Broyden算法（多元变量）
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
    Hk = mb.identity(len(X0)) 
    while i < max_iter:
        gk = np.mat([float(df0.subs([(x0, X0[0]), (x1, X0[1])])), float(df1.subs([(x0, X0[0]), (x1, X0[1])]))]).T 
        dk = -Hk*gk

        mk = 0
        while mk < 10:
            if f.subs([(x0, X0[0]+beta**mk*dk[0,0]), (x1, X0[1]+beta**mk*dk[1,0])]) < f.subs([(x0, X0[0]), (x1, X0[1])]) + delta*beta**mk*gk.T*dk:
                break
            mk += 1
        Xnew = [X0[0] + beta**mk*dk[0,0], X0[1] + beta**mk*dk[1,0]]

        sk = np.mat([beta**mk*dk[0,0], beta**mk*dk[1,0]]).T
        yk = np.mat([float(df0.subs([(x0, Xnew[0]), (x1, Xnew[1])])), float(df1.subs([(x0, Xnew[0]), (x1, Xnew[1])]))]).T - gk

        phik = 0.5
        vk = float(np.sqrt(yk.T*Hk*yk))*(sk/(sk.T*yk)-(Hk*yk)/(yk.T*Hk*yk))
        Hk = Hk - (Hk*yk*yk.T*Hk)/(yk.T*Hk*yk) + (sk*sk.T)/(sk.T*yk) + phik*vk*vk.T

        i += 1
        print('迭代第%d次：[%.5f, %.5f]' %(i, Xnew[0], Xnew[1]))      
        if abs(f.subs([(x0, Xnew[0]), (x1, Xnew[1])])-f.subs([(x0, X0[0]), (x1, X0[1])])) < epsilon:
            break
        X0 = Xnew
    return Xnew


if __name__ == '__main__':      
    x = sympy.symbols("x") 
    x0 = sympy.symbols("x0")
    x1 = sympy.symbols("x1")
    result = Broyden_x(x**4-4*x, 10, 100, 1e-5)
    print('最终迭代位置：%.5f' %result)
    result = Broyden_x0x1((x0-1)**2+(x1-1)**4, [10,10], 100, 1e-5)
    print('最终迭代位置：[%.5f, %.5f]' %(result[0], result[1]))

