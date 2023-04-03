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

# function：生成一维随机数组
def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start<= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list

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
    # ### s2--method
    # for i in range(len(I)):
    #     temp = [0] * len(I)
    #     max_one = 0
    #     for j in range(len(J)):
    #         max_one = max(max_one, sj[J[j]])
    #     temp[i] = max_one
    #     Qk.append(temp)
    ### s3--methmod
    # for i in range(len(I)):
    #     temp = [0] * len(I)
    #     max_one = 0
    #     for j in range(len(J)):
    #         max_one = max(max_one, sj[J[j]])
    #     temp[i] = 2*max_one
    #     Qk.append(temp)
    # ### s4--method
    # for i in range(len(I)):
    #     temp = [0] * len(I)
    #     max_one = 0
    #     for j in range(len(J)):
    #         max_one = max(max_one, sj[J[j]])
    #     temp[i] = 3*max_one
    #     Qk.append(temp)
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

def init(m,n):
    I = [0] * m
    J = [0] * n
    bi = [0] * m
    mdu_p = [0] * m
    di = [0] * m
    sj = [0] * n
    aj = [0] * n
    yueta = [[0] * n for i in range(m)]

    for i in range(m):
        I[i] = i
    #print('I', I)
    for j in range(n):
        J[j] = j
    #print('J', J)
    for i in range(m):
        mdu_p[i] = np.random.rand()
        di[i] = np.random.randint(300, 500)
        bi[i] = mdu_p[i] * di[i]
    #print('bi', bi)
    for j in range(n):
        aj[j] = np.random.rand()
        sj[j] = np.random.randint(200, 801)
    #print('aj', aj)
    #print('sj', sj)
    for i in range(m):
        for j in range(n):
            yueta[i][j] = np.random.randint(0, 2)
    #print(yueta)
    return I,J,bi,di,sj,aj,yueta
    ''' example
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
    '''

if __name__ == '__main__':
    print('TCDA-experiment')
    # print('ESs数量对系统性能的影响——MDs(固定500) 自变量 number of ESs（20,200,20） 因变量 Social Welfare')
    # SocialWelfare=[]
    # m = 500
    # n = 0
    # #计算SW，m固定500，n从20到200，间隔20
    # for i in range(20,201,20):
    #     n = i
    #     I = [0] * m
    #     J = [0] * n
    #     bi = [0] * m
    #     mdu_p = [0] * m
    #     di = [0] * m
    #     sj = [0] * n
    #     aj = [0] * n
    #     yueta = [[0] * n for i in range(m)]
    #     I,J,bi,di,sj,aj,yueta = init(m,n)
    #     x_final, y_final, z_final, III = TCDA_Algorithm1(I, J)
    #     SocialWelfare.append(SW(III,J))
    #     print(SocialWelfare)
    # print('SocialWelfare:'+str(SocialWelfare))
    #
    # print('ESs数量对系统性能的影响——MDs(固定500) 自变量 number of ESs（20,200,20） 因变量 NMSD')
    # #计算NMSD(Number of winning MDs)，m固定500，n从20到200，间隔20
    # NMSD = []
    # for i in range(20,201,20):
    #     count = 0 #最后获胜的MDs数量
    #     n = i
    #     I = [0] * m
    #     J = [0] * n
    #     bi = [0] * m
    #     mdu_p = [0] * m
    #     di = [0] * m
    #     sj = [0] * n
    #     aj = [0] * n
    #     yueta = [[0] * n for i in range(m)]
    #     I,J,bi,di,sj,aj,yueta = init(m,n)
    #     x_final, y_final, z_final, III = TCDA_Algorithm1(I, J)
    #     for i in range(m):
    #         if x_final[i] >= 0.5: #大于0.5默认为有效
    #             count = count + 1
    #     NMSD.append(count)
    #     print(NMSD)
    # print('NMSD:'+str(NMSD))
    #
    # print('ESs数量对系统性能的影响——MDs(固定500) 自变量 number of ESs（20,200,20） 因变量 RU')
    # # 计算RU(获胜的ESs资源分配总和占系统总资源的比例)，m固定500，n从20到200，间隔20
    # RU = []
    # for i in range(20, 201, 20):
    #     count = 0
    #     count_end = 0  # 最后获胜的MDs数量
    #     n = i
    #     I = [0] * m
    #     J = [0] * n
    #     bi = [0] * m
    #     mdu_p = [0] * m
    #     di = [0] * m
    #     sj = [0] * n
    #     aj = [0] * n
    #     yueta = [[0] * n for i in range(m)]
    #     I, J, bi, di, sj, aj, yueta = init(m, n)
    #     x_final, y_final, z_final, III = TCDA_Algorithm1(I, J)
    #     for j in range(n):
    #         count_end = count_end + y_final[j]
    #     for j in range(n):
    #         count  = count + sj[j]
    #     ratio = float(count_end)/float(count)
    #     RU.append(ratio)
    #     print(RU)
    # print('RU:' + str(RU))
    #
    # print('MDs数量对系统性能的影响 ESs(固定100) 自变量 numbers of MDs(100,1000,100) 因变量 Social Welfare')
    # # 计算SW，m从100到1000，间隔100.n固定100.
    # SocialWelfare = []
    # m = 0
    # n = 100
    # for i in range(100, 1001, 100):
    #     m = i
    #     I = [0] * m
    #     J = [0] * n
    #     bi = [0] * m
    #     mdu_p = [0] * m
    #     di = [0] * m
    #     sj = [0] * n
    #     aj = [0] * n
    #     yueta = [[0] * n for i in range(m)]
    #     I, J, bi, di, sj, aj, yueta = init(m, n)
    #     x_final, y_final, z_final, III = TCDA_Algorithm1(I, J)
    #     SocialWelfare.append(SW(III, J))
    #     print(SocialWelfare)
    # print('SocialWelfare:' + str(SocialWelfare))
    #
    # print('MDs数量对系统性能的影响 ESs(固定100) 自变量 numbers of MDs(100,1000,100) 因变量 NMSD')
    # # 计算NMSD(Number of winning MDs)，m固定500，n从20到200，间隔20
    # NMSD = []
    # for i in range(100, 1001, 100):
    #     count = 0  # 最后获胜的MDs数量
    #     m = i
    #     I = [0] * m
    #     J = [0] * n
    #     bi = [0] * m
    #     mdu_p = [0] * m
    #     di = [0] * m
    #     sj = [0] * n
    #     aj = [0] * n
    #     yueta = [[0] * n for i in range(m)]
    #     I, J, bi, di, sj, aj, yueta = init(m, n)
    #     x_final, y_final, z_final, III = TCDA_Algorithm1(I, J)
    #     for i in range(m):
    #         if x_final[i] >= 0.5:  # 大于0.5默认为有效
    #             count = count + 1
    #     NMSD.append(count)
    #     print(NMSD)
    # print('NMSD:' + str(NMSD))
    #
    # print('MDs数量对系统性能的影响 ESs(固定100) 自变量 numbers of MDs(100,1000,100) 因变量 RU')
    # # 计算RU(获胜的ESs资源分配总和占系统总资源的比例)，m固定500，n从20到200，间隔20
    # RU = []
    # for i in range(100,1001,100):
    #     count = 0
    #     count_end = 0  # 最后获胜的MDs数量
    #     m = i
    #     I = [0] * m
    #     J = [0] * n
    #     bi = [0] * m
    #     mdu_p = [0] * m
    #     di = [0] * m
    #     sj = [0] * n
    #     aj = [0] * n
    #     yueta = [[0] * n for i in range(m)]
    #     I, J, bi, di, sj, aj, yueta = init(m, n)
    #     x_final, y_final, z_final, III = TCDA_Algorithm1(I, J)
    #     for j in range(n):
    #         count_end = count_end + y_final[j]
    #     for j in range(n):
    #         count = count + sj[j]
    #     ratio = float(count_end) / float(count)
    #     RU.append(ratio)
    #     print(RU)
    # print('RU:' + str(RU))

    # print('不同填充变量Q（Paddings）对系统性能的影响 自变量 m（100,700,100)  自变量n(20,140,20) 自变量participants（m+n） 因变量 Social Welfare')
    # SocialWelfare = []
    # for i in range(1,8):
    #     m = 100 * i
    #     n = 20 * i
    #     I = [0] * m
    #     J = [0] * n
    #     bi = [0] * m
    #     mdu_p = [0] * m
    #     di = [0] * m
    #     sj = [0] * n
    #     aj = [0] * n
    #     yueta = [[0] * n for i in range(m)]
    #     I, J, bi, di, sj, aj, yueta = init(m, n)
    #     x_final, y_final, z_final, III = TCDA_Algorithm1(I, J)
    #     SocialWelfare.append(SW(III, J))
    #     print(SocialWelfare)
    # print('SocialWelfare:' + str(SocialWelfare))

    # print('性质验证：m-100 n-20 因变量是Payments & bidding price')
    # m = 100
    # n = 20
    # I = [0] * m
    # J = [0] * n
    # bi = [0] * m
    # mdu_p = [0] * m
    # di = [0] * m
    # sj = [0] * n
    # aj = [0] * n
    # yueta = [[0] * n for i in range(m)]
    # I, J, bi, di, sj, aj, yueta = init(m, n)
    # x_final, y_final, z_final, III = TCDA_Algorithm1(I, J)
    # pCV = TCDA_Algorithm2(I, III, J)
    # # os.system("pause")
    # pb, rs = TCDA_Algorithm3(I, III, J, x_final, y_final, pCV)
    # print('bi is:'+str(bi))
    # print('pb is:' + str(pb))

    # print('性质验证：m-100 n-50 因变量是Rewards(rsj) & asking  prices(aj)')
    # m = 100
    # n = 50
    # I = [0] * m
    # J = [0] * n
    # bi = [0] * m
    # mdu_p = [0] * m
    # di = [0] * m
    # sj = [0] * n
    # aj = [0] * n
    # yueta = [[0] * n for i in range(m)]
    # I, J, bi, di, sj, aj, yueta = init(m, n)
    # x_final, y_final, z_final, III = TCDA_Algorithm1(I, J)
    # print('TCDA_Algorithm1 finish')
    # # pCV = TCDA_Algorithm2(I, III, J)
    # # os.system("pause")
    # pCV = [0] * len(I)
    # pb, rs = TCDA_Algorithm3(I, III, J, x_final, y_final, pCV)
    # for i in range(len(aj)):
    #     aj[i] = aj[i] * y_final[i]
    # print('aj is:'+str(aj))
    # print('rs is:' + str(rs))

    # print('性质验证：m-100 n-(8,80,8) 因变量是total Payments & total reward')
    # m = 100
    # total_payment_array = []
    # total_rewards_array = []
    # for i in range(8, 81, 8):
    #     n = i
    #     print(i)
    #     I = [0] * m
    #     J = [0] * n
    #     bi = [0] * m
    #     mdu_p = [0] * m
    #     di = [0] * m
    #     sj = [0] * n
    #     aj = [0] * n
    #     yueta = [[0] * n for i in range(m)]
    #     I, J, bi, di, sj, aj, yueta = init(m, n)
    #     x_final, y_final, z_final, III = TCDA_Algorithm1(I, J)
    #     pCV = TCDA_Algorithm2(I, III, J)
    #     # os.system("pause")
    #     pb, rs = TCDA_Algorithm3(I, III, J, x_final, y_final, pCV)
    #     total_payment = 0
    #     for i in range(len(pb)):
    #         total_payment = total_payment + pb[i]
    #     total_payment_array.append(total_payment)
    #     total_reward = 0
    #     for i in range(len(rs)):
    #         total_reward = total_reward + rs[i]
    #     total_rewards_array.append(total_reward)
    # print('total_payment is:'+str(total_payment_array))
    # print('total_reward is:'+str(total_rewards_array))

    # print('性质验证——m=7,n=4 bi变化(6,19,1)对获胜MD（指定为MD1）的Utility影响')
    # global bi, aj, di, sj, yueta
    # m = 7
    # n = 4
    # I = [0, 1, 2, 3, 4, 5, 6]
    # J = [0, 1, 2, 3]
    # bi = [12, 14, 15, 7, 11, 14, 6]
    # aj = [2.9, 3.1, 3.2, 2.8]
    # di = [3, 3, 4, 2, 3, 4, 2]
    # sj = [7, 8, 12, 6]
    # yueta = [[0]*n for i in range(m)]
    # yueta[0][0] = yueta[0][3] = 1
    # yueta[1][1] = 1
    # yueta[2][0] = yueta[2][1] = 1
    # yueta[3][1] = yueta[3][2] = 1
    # yueta[4][2] = 1
    # yueta[5][1] = yueta[5][2] = yueta[5][3] = 1
    # yueta[6][3] = 1
    # utility = []
    # for i in range(6,19,1):
    #     bi[0] = i
    #     x_final, y_final, z_final, III = TCDA_Algorithm1(I, J)
    #     pCV = TCDA_Algorithm2(I, III, J)
    #     pb, rs = TCDA_Algorithm3(I, III, J, x_final, y_final, pCV)
    #     utility.append(x_final[0]*bi[0]-pb[0])
    # print('MD1 的utility:'+str(utility))

    # print('性质验证——m=7,n=4 bi变化(7,22,1)对失败MD（指定为MD2）的Utility影响')
    # global bi, aj, di, sj, yueta
    # m = 7
    # n = 4
    # I = [0, 1, 2, 3, 4, 5, 6]
    # J = [0, 1, 2, 3]
    # bi = [12, 14, 15, 7, 11, 14, 6]
    # aj = [2.9, 3.1, 3.2, 2.8]
    # di = [3, 3, 4, 2, 3, 4, 2]
    # sj = [7, 8, 12, 6]
    # yueta = [[0]*n for i in range(m)]
    # yueta[0][0] = yueta[0][3] = 1
    # yueta[1][1] = 1
    # yueta[2][0] = yueta[2][1] = 1
    # yueta[3][1] = yueta[3][2] = 1
    # yueta[4][2] = 1
    # yueta[5][1] = yueta[5][2] = yueta[5][3] = 1
    # yueta[6][3] = 1
    # utility = []
    # for i in range(7,22,1):
    #     bi[0] = i
    #     x_final, y_final, z_final, III = TCDA_Algorithm1(I, J)
    #     pCV = TCDA_Algorithm2(I, III, J)
    #     pb, rs = TCDA_Algorithm3(I, III, J, x_final, y_final, pCV)
    #     utility.append(x_final[1]*bi[1]-pb[1])
    # print('MD2 的utility:'+str(utility))

    # print('性质验证——m=7,n=4 aj变化(2,4.1,0.1)对成功ES（指定为ES1）的Utility影响')
    # global bi, aj, di, sj, yueta
    # m = 7
    # n = 4
    # I = [0, 1, 2, 3, 4, 5, 6]
    # J = [0, 1, 2, 3]
    # bi = [12, 14, 15, 7, 11, 14, 6]
    # aj = [2.9, 3.1, 3.2, 2.8]
    # di = [3, 3, 4, 2, 3, 4, 2]
    # sj = [7, 8, 12, 6]
    # yueta = [[0]*n for i in range(m)]
    # yueta[0][0] = yueta[0][3] = 1
    # yueta[1][1] = 1
    # yueta[2][0] = yueta[2][1] = 1
    # yueta[3][1] = yueta[3][2] = 1
    # yueta[4][2] = 1
    # yueta[5][1] = yueta[5][2] = yueta[5][3] = 1
    # yueta[6][3] = 1
    # utility = []
    # for i in np.arange(2.0,4.1,0.1):
    #     aj[0] = i
    #     x_final, y_final, z_final, III = TCDA_Algorithm1(I, J)
    #     pCV = TCDA_Algorithm2(I, III, J)
    #     pb, rs = TCDA_Algorithm3(I, III, J, x_final, y_final, pCV)
    #     utility.append(rs[0]-y_final[0]*aj[0])
    # print('ES1 的utility:'+str(utility))

    # print('性质验证——m=7,n=4 aj变化(2,4.1,0.1)对失败ES（指定为ES3）的Utility影响')
    # global bi, aj, di, sj, yueta
    # m = 7
    # n = 4
    # I = [0, 1, 2, 3, 4, 5, 6]
    # J = [0, 1, 2, 3]
    # bi = [12, 14, 15, 7, 11, 14, 6]
    # aj = [2.9, 3.1, 3.2, 2.8]
    # di = [3, 3, 4, 2, 3, 4, 2]
    # sj = [7, 8, 12, 6]
    # yueta = [[0]*n for i in range(m)]
    # yueta[0][0] = yueta[0][3] = 1
    # yueta[1][1] = 1
    # yueta[2][0] = yueta[2][1] = 1
    # yueta[3][1] = yueta[3][2] = 1
    # yueta[4][2] = 1
    # yueta[5][1] = yueta[5][2] = yueta[5][3] = 1
    # yueta[6][3] = 1
    # utility = []
    # for i in np.arange(2,4.1,0.1):
    #     aj[0] = i
    #     x_final, y_final, z_final, III = TCDA_Algorithm1(I, J)
    #     pCV = TCDA_Algorithm2(I, III, J)
    #     pb, rs = TCDA_Algorithm3(I, III, J, x_final, y_final, pCV)
    #     utility.append(rs[2]-y_final[2]*aj[2])
    # print('ES3 的utility:'+str(utility))

