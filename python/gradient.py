import sympy
import numpy as np


def gradient_x(f, x0, max_iter, epsilon):
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
    # alpha = 0.001
    beta = 0.5
    delta = 0.25 
    while i < max_iter:
        gk = df.subs(x, x0)
        dk = -gk
        
        # xnew = x0 + alpha*dk
        mk = 0
        while mk < 10:
            if f.subs(x, x0+beta**mk*dk) < f.subs(x,x0) + delta*beta**mk*gk*dk:
                break
            mk += 1
        xnew = x0 + beta**mk*dk
        i += 1
        print('迭代第%d次：%.5f' %(i, xnew))
        if abs(df.subs(x, xnew)-df.subs(x, x0)) < epsilon:
            break
        x0 = xnew
    return xnew

def gradient_x0x1(f, X0, max_iter, epsilon):
    '''
    description:    梯度下降算法（多元变量）
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
    # alpha = 0.001
    beta = 0.5 
    delta = 0.25 
    while i < max_iter:
        gk = np.mat([float(df0.subs([(x0, X0[0]), (x1, X0[1])])), float(df1.subs([(x0, X0[0]), (x1, X0[1])]))]).T 
        dk = -gk   

        # Xnew = [X0[0] + alpha*dk[0,0], X0[1] + alpha*dk[1,0]]
        mk = 0
        while mk < 10:
            if f.subs([(x0, X0[0]+beta**mk*dk[0,0]), (x1, X0[1]+beta**mk*dk[1,0])]) < f.subs([(x0, X0[0]), (x1, X0[1])]) + delta*beta**mk*gk.T*dk:
                break
            mk += 1
        Xnew = [X0[0] + beta**mk*dk[0,0], X0[1] + beta**mk*dk[1,0]]
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
    result = gradient_x(x**4-4*x, 10, 100, 1e-5)
    print('最终迭代位置：%.5f' %result)
    result = gradient_x0x1((x0-1)**2+(x1-1)**4, [10,10], 100, 1e-5)
    print('最终迭代位置：[%.5f, %.5f]' %(result[0], result[1]))

