import sympy
import numpy as np


def gauss_newton_x(f, x0, max_iter, epsilon):
    '''
    description:    高斯牛顿算法（一元变量）
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
    while i < max_iter:
        Jk = df.subs(x, x0) 
        gk = Jk * f.subs(x, x0)
        dk = -gk/(Jk*Jk)

        mk = 0
        while mk < 10:
            if 0.5*f.subs(x, x0+beta**mk*dk)**2 < 0.5*f.subs(x,x0)**2 + delta*beta**mk*gk*dk:
                break
            mk += 1
        xnew = x0 + beta**mk*dk  

        i += 1
        print('迭代第%d次：%.5f' %(i, xnew))      
        if abs(f.subs(x, xnew)-f.subs(x, x0)) < epsilon:
            break
        x0 = xnew
    return xnew

def gauss_newton_x0x1(f, X0, max_iter, epsilon):
    '''
    description:    高斯牛顿算法（多元变量）
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
    while i < max_iter:
        Jk = np.mat([float(df0.subs([(x0, X0[0]), (x1, X0[1])])), float(df1.subs([(x0, X0[0]), (x1, X0[1])]))]).T 
        gk = Jk * f.subs([(x0, X0[0]), (x1, X0[1])])
        dk = -(Jk.T*Jk).I*gk.T

        mk = 0    
        while mk < 10:
            if 0.5*f.subs([(x0, X0[0]+beta**mk*dk[0,0]), (x1, X0[1]+beta**mk*dk[0,1])])**2 < 0.5*f.subs([(x0, X0[0]), (x1, X0[1])])**2 + delta*beta**mk*gk.T*dk.T:
                break
            mk += 1
        Xnew = [X0[0] + beta**mk*dk[0,0], X0[1] + beta**mk*dk[0,1]]
       
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
    result = gauss_newton_x(x**4-4*x, 10, 100, 1e-5)
    print('最终迭代位置：%.5f' %result)
    result = gauss_newton_x0x1((x0-1)**2+(x1-1)**4, [10,10], 100, 1e-5)
    print('最终迭代位置：[%.5f, %.5f]' %(result[0], result[1]))

