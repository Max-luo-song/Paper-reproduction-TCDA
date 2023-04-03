'''
    程序功能描述：实现TCDA整个算法，包含TCDA Algotithm1/2/3

    手动输入：m——MD个数
            n——ES个数
    随机生成：bi[m]——竞拍价格
            aj[n]——要求价格
            di[m]——需求数量
            sj[n]——提供数量
            I[m]——初始MD集合，从0开始，共m个
            J[n]——初始ES集合，从0开始，共n个
            yueta[m][n]——MD ES是否可通信
    程序输出：x_final代表MD的分配结果
            y_final代表ES的分配结果
            z_final代表MD和ES之间的分配数量关系
            pb代表MD花费结果
            rs代表RS花费结果
'''

from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import os

# 全局变量，赋值一次
# bi = []
# aj = []
# di = []
# sj = []
# yueta = []

# funciton：求解基础LP()
def LP(I, J):
    ## max SW(I,J) = Σx_i*b_i - Σy_j*a_j
    c1 = []
    for i in range(len(I)):
        c1.append(bi[I[i]])
    for i in range(len(J)):
        c1.append(-aj[J[i]])
    for i in range(len(I)):
        for j in range(len(J)):
            c1.append(0)
    c = np.array(c1)

    ### 初始化
    Aeq = []
    beq = [0] * (len(I)+len(J))
    init = []
    x = [0] * len(I)
    y = [0] * len(J)
    z = [[0]*len(J) for i in range(len(I))]
    ### s.t.——Σz_ij = d_i * x_i
    for i in range(len(I)+len(J)+len(I)*len(J)): #init包含所有随机变量
        init.append(0)
    for i in range(len(I)):
        temp = init.copy()
        temp[i] = -di[I[i]]
        for j in range(len(J)):
            temp[len(I)+len(J)+i*len(J)+j] = 1   #令z_ij为1
        Aeq.append(temp)
    ### s.t.——Σz_ij = y_j
    for j in range(len(J)):
        temp = init.copy()
        temp[len(I)+j] = -1
        for i in range(len(I)):
            temp[len(I)+len(J)+i*len(J)+j] = 1
        Aeq.append(temp)
    Aeq = np.array(Aeq)
    beq = np.array(beq)
    ### 下面是随机变量约束###
    for i in range(len(I)):
        x[i] = (0,1)
    for j in range(len(J)):
        y[j] = (0, sj[J[j]])
    for i in range(len(I)):
        for j in range(len(J)):
            z[i][j] = (0, sj[J[j]]*yueta[I[i]][J[j]])
    bound = []
    for i in range(len(x)):
        bound.append(x[i])
    for j in range(len(y)):
        bound.append(y[j])
    for i in range(len(I)):
        for j in range(len(J)):
            bound.append(z[i][j])
    bound = tuple(bound)
    result = optimize.linprog(-c, None, None, Aeq, beq, bounds=bound, options={'presolve': False})
    #### 下面是从result内提取最优数值
    x = result['x']
    x_result = []
    y_result = []
    z_result = []
    for i in range(len(x)):
        if i < len(I):
            x_result.append(x[i])
        elif i < len(I)+len(J):
            y_result.append(x[i])
        else:
            z_result.append(x[i])
    # print('x_result'+str(x_result))
    # print('y_result' + str(y_result))
    # print('z_result' + str(z_result))
    # print('-------------------------')
    return x_result, y_result, z_result

# function：求解接触LPQ()
def LPQ(I, J, Q):
    ## max SW(I,J) = Σx_i*b_i - Σy_j*a_j
    c1 = []
    for i in range(len(I)):
        c1.append(bi[I[i]])
    for i in range(len(J)):
        c1.append(-aj[J[i]])
    for i in range(len(I)):
        for j in range(len(J)):
            c1.append(0)
    c = np.array(c1)

    ### 初始化
    Aeq = []
    beq = [0] * (len(I)+len(J))
    for pp in range(len(Q)):
        beq[pp] = Q[pp]
    init = []
    x = [0] * len(I)
    y = [0] * len(J)
    z = [[0]*len(J) for i in range(len(I))]
    ### s.t.——Σz_ij = d_i * x_i
    for i in range(len(I)+len(J)+len(I)*len(J)): #init包含所有随机变量
        init.append(0)
    for i in range(len(I)):
        temp = init.copy()
        temp[i] = -di[I[i]]
        for j in range(len(J)):
            temp[len(I)+len(J)+i*len(J)+j] = 1   #令z_ij为1
        Aeq.append(temp)
    ### s.t.——Σz_ij = y_j
    for j in range(len(J)):
        temp = init.copy()
        temp[len(I)+J[j]] = -1
        for i in range(len(I)):
            temp[len(I)+len(J)+i*len(J)+j] = 1
        Aeq.append(temp)
    Aeq = np.array(Aeq)
    beq = np.array(beq)
    ### 下面是随机变量约束###
    for i in range(len(I)):
        x[i] = (0,1)
    for j in range(len(J)):
        y[j] = (0, sj[J[j]])
    for i in range(len(I)):
        for j in range(len(J)):
            z[i][j] = (0, sj[J[j]]*yueta[I[i]][J[j]])
    bound = []
    for i in range(len(x)):
        bound.append(x[i])
    for j in range(len(y)):
        bound.append(y[j])
    for i in range(len(I)):
        for j in range(len(J)):
            bound.append(z[i][j])
    bound = np.array(bound)
    result = optimize.linprog(-c, None, None, Aeq, beq, bounds=bound, options={'presolve': False})
    #### 下面是从result内提取最优数值
    x = result['x']
    x_result = []
    y_result = []
    z_result = []
    for i in range(len(x)):
        if i < len(I):
            x_result.append(x[i])
        elif i < len(I)+len(J):
            y_result.append(x[i])
        else:
            z_result.append(x[i])
    # print('x_result'+str(x_result))
    # print('y_result' + str(y_result))
    # print('z_result' + str(z_result))
    # print('-------------------------')
    return x_result, y_result, z_result

# function：特殊处理LP_y_final_exp的情况
def LP_y_exp(I, J, ajj, sjj):
    ## max SW(I,J) = Σx_i*b_i - Σy_j*a_j
    c1 = []
    for i in range(len(I)):
        c1.append(bi[I[i]])
    for i in range(len(J)):
        c1.append(-ajj[J[i]])
    for i in range(len(I)):
        for j in range(len(J)):
            c1.append(0)
    c = np.array(c1)

    ### 初始化
    Aeq = []
    beq = [0] * (len(I)+len(J))
    init = []
    x = [0] * len(I)
    y = [0] * len(J)
    z = [[0]*len(J) for i in range(len(I))]
    ### s.t.——Σz_ij = d_i * x_i
    for i in range(len(I)+len(J)+len(I)*len(J)): #init包含所有随机变量
        init.append(0)
    for i in range(len(I)):
        temp = init.copy()
        temp[i] = -di[I[i]]
        for j in range(len(J)):
            temp[len(I)+len(J)+i*len(J)+j] = 1   #令z_ij为1
        Aeq.append(temp)
    ### s.t.——Σz_ij = y_j
    for j in range(len(J)):
        temp = init.copy()
        temp[len(I)+j] = -1
        for i in range(len(I)):
            temp[len(I)+len(J)+i*len(J)+j] = 1
        Aeq.append(temp)
    Aeq = np.array(Aeq)
    beq = np.array(beq)
    ### 下面是随机变量约束###
    for i in range(len(I)):
        x[i] = (0,1)
    for j in range(len(J)):
        y[j] = (0, sjj[J[j]])
    for i in range(len(I)):
        for j in range(len(J)):
            z[i][j] = (0, sjj[J[j]]*yueta[I[i]][J[j]])
    bound = []
    for i in range(len(x)):
        bound.append(x[i])
    for j in range(len(y)):
        bound.append(y[j])
    for i in range(len(I)):
        for j in range(len(J)):
            bound.append(z[i][j])
    bound = tuple(bound)
    result = optimize.linprog(-c, None, None, Aeq, beq, bounds=bound, options={'presolve': False})
    #### 下面是从result内提取最优数值
    x = result['x']
    x_result = []
    y_result = []
    z_result = []
    for i in range(len(x)):
        if i < len(I):
            x_result.append(x[i])
        elif i < len(I)+len(J):
            y_result.append(x[i])
        else:
            z_result.append(x[i])
    # print('x_result'+str(x_result))
    # print('y_result' + str(y_result))
    # print('z_result' + str(z_result))
    # print('-------------------------')
    return x_result, y_result, z_result

def TCDA_Algorithm1(I, J):
    ######## Resource Allocation ##########
    xx, yy, zz = LP(I, J)
    II = []
    III = []
    for i in range(len(xx)):
        if xx[i] != 0:  # 此处进行线性规划，超过0.5认为有效
            II.append(i)
    Q = Cal_Q(I, J)
    for k in range(len(II)):
        xxx, yyy, zzz = LPQ(I, J, Q[II[k]]) #### solve LP(I,J,Qk), 得到xxx, yyy, zzz
        if xxx[II[k]] >= 0.5:
            III.append(II[k])
    xxxx, yyyy, zzzz = LP(III, J) ### solve LP(III,J), 得到xxxx,yyyy,zzzz
    x_final = [0] * m
    y_final = [0] * n
    z_final = [[0] * n for i in range(m)]

    for i in range(len(III)):
        x_final[III[i]] = 1
        for j in range(len(J)):
            z_final[III[i]][J[j]] = round(zzzz[I[i] * len(J) + J[j]])
    y_final = yyyy
    for j in range(len(y_final)):
        y_final[j] = round(y_final[j])
    return x_final, y_final, z_final, III

def TCDA_Algorithm2(I, III, J):
    global bi
    bi_temp = bi.copy()
    pCV = [0] * len(bi)
    Q = Cal_Q(I, J)
    for i in range(len(III)):
        lb = 0
        ub = bi[III[i]]
        while math.fabs(lb - ub) >= 1:
            # while ub-lb >= 1:
            temp = lb + ub
            # temp = math.ceil(temp)
            mid = temp / 2.0
            bi[III[i]] = mid
            xxx, yyy, zzz = LPQ(I, J, Q[III[i]])
            if xxx[III[i]] >= 0.5:
                ub = mid
                if math.fabs(lb - ub) <= 1:
                    # if lb == ub -1:
                    pCV[III[i]] = mid
                    break
            else:
                lb = mid
                if math.fabs(lb - ub) <= 1:
                    # if lb == ub - 1:
                    pCV[III[i]] = mid + 1
                    break
    bi = bi_temp
    return pCV

def TCDA_Algorithm3(I, III, J, x_final, y_final, pCV):
    pVCG = Cal_pVCG(I, III, J)
    pb = [0] * len(I)
    rs = [0] * len(J)
    for i in range(len(I)):
        if x_final[I[i]] == 1:
            pb[I[i]] = pCV[I[i]]
        else:
            pb[I[i]] = 0
    for j in range(len(J)):
        if y_final[J[j]] != 0:
            rs[J[j]] = pVCG[J[j]]
        else:
            rs[J[j]] = 0
    return pb, rs

# funciton：进行VCG的计算依据公式10
def Cal_pVCG(I, III, J): # _i代表input
    VCG = [0] * n
    for j in range(len(J)):
        J_exp = []
        for k in range(len(J)):
            if k != j:
                J_exp.append(J[k])
        VCG[J[j]] = y_final[J[j]]*aj[J[j]] + SW(III, J) - SW_expect(I, III, J_exp, j, J)
    return VCG

# function：计算社会福利,按照SW(I, J) = Σx_i*b_i - Σy_j*a_j
# truthfulness情况b_i == v_i, a_j == c_j
def SW(I, J):
    temp = 0.0
    for i in range(len(I)):
        temp = temp + x_final[I[i]]*bi[I[i]]
    for j in range(len(J)):
        temp = temp - y_final[J[j]]*aj[J[j]]
    return temp

# function：计算社会福利,按照SW(I, J) = Σx_i*b_i - Σy_j*a_j  J中除去except_Num
def SW_expect(I, III, J_exp, jj, J):
    y_final_exp = Get_y_final_exp(I, J, jj)
    temp = 0.0
    for i in range(len(I)):
        temp = temp + x_final[I[i]]*bi[I[i]]
    y_final_exp_temp = [0] * len(J)
    t = 0 #用于计数
    for i in range(len(J)):
        if i != jj:
            y_final_exp_temp[i] = y_final_exp[t]
            t = t + 1
    y_final_exp = y_final_exp_temp
    for j in range(len(J_exp)):
        temp = temp - y_final_exp[J_exp[j]]*aj[J_exp[j]]
    return temp

# function：计算Qk padding向量
def Cal_Q(I, J):
    Qk = []
    ### s1--method
    for i in range(len(I)):
        temp = [0] * len(I)
        max_one = 0
        for j in range(len(J)):
            if yueta[I[i]][J[j]] == 1:   #可通信
                max_one = max(max_one, sj[J[j]])
        temp[i] = max_one
        Qk.append(temp)
    return Qk

# function：计算除了jj的y_final
def Get_y_final_exp(I, J, jj):
    xx, yy, zz = LP(I, J)
    II = []
    III = []
    for i in range(len(xx)):
        if xx[i] >= 0.5:  # 此处进行线性规划，超过0.5认为有效
            II.append(i)
    Q = Cal_Q(I, J)
    for k in range(len(II)):
        #### set padding Qk
        xxx, yyy, zzz = LPQ(I, J, Q[II[k]]) #### solve LP(I,J,Qk), 得到xxx, yyy, zzz
        if xxx[II[k]] >= 0.5:
            III.append(II[k])
    J_new = []
    for j in range(len(J)-1):
        J_new.append(j)
    ajj = []
    for j in range(len(J)):
        if J[j] != jj:
            ajj.append(aj[J[j]])
    sjj = []
    for j in range(len(J)):
        if J[j] != jj:
            sjj.append(sj[J[j]])
    xxxx, yyyy, zzzz = LP_y_exp(III, J_new, ajj, sjj)
    y_final_exp = yyyy
    return y_final_exp


if __name__ == '__main__':
    print('TCDA-Algorithm1/2/3，以论文给的example为验证')
    global bi, aj, di, sj, yueta
    m = 7
    n = 4
    I = [0, 1, 2, 3, 4, 5, 6]
    J = [0, 1, 2, 3]
    bi = [12, 14, 15, 7, 11, 14, 6]
    aj = [2.9, 3.1, 3.2, 2.8]
    di = [3, 3, 4, 2, 3, 4, 2]
    sj = [7, 8, 12, 6]
    yueta = [[0]*n for i in range(m)]
    yueta[0][0] = yueta[0][3] = 1
    yueta[1][1] = 1
    yueta[2][0] = yueta[2][1] = 1
    yueta[3][1] = yueta[3][2] = 1
    yueta[4][2] = 1
    yueta[5][1] = yueta[5][2] = yueta[5][3] = 1
    yueta[6][3] = 1
    x_final, y_final, z_final, III = TCDA_Algorithm1(I, J)
    pCV = TCDA_Algorithm2(I, III, J)
    pb, rs = TCDA_Algorithm3(I, III, J, x_final, y_final, pCV)
    print('x_final is:' + str(x_final))
    print('y_final is:' + str(y_final))
    print('z_final is:' + str(z_final))
    print('pb is:' + str(pb))
    print('rs is:' + str(rs))
