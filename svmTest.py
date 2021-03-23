from numpy import  *

def loadDataSet(filename): #读取数据
    dataMat=[]
    labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=line.strip().split(' ')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat #返回数据特征和数据类别

#定义类，方便存储数据
class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # 存储各类参数
        self.X = dataMatIn  #数据特征
        self.labelMat = classLabels #数据类别
        self.C = C #软间隔参数C，参数越大，非线性拟合能力越强
        self.tol = toler #停止阀值
        self.m = shape(dataMatIn)[0] #数据行数
        self.alphas = mat(zeros((self.m,1))) # 初始化 一列数组
        self.b = 0 #初始设为0
        self.eCache = mat(zeros((self.m,2))) #缓存 E值，两列数组，第一列标志是否更新，若为0则没有更新过E值
        self.K = mat(zeros((self.m,self.m))) #核函数的计算结果
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

# 核函数
def kernelTrans(X, A, kernelTuple): # 参数X:支持向量的特征树；A：某一行特征数据；kernelTuple：('lin',k1)核函数的类型和带宽
    m,n = shape(X) # X的行列
    K = mat(zeros((m,1))) # 创建m行一列0矩阵，引用该矩阵

    if kernelTuple[0]=='lin': # 线性核函数
        K = X * A.T
    elif kernelTuple[0]=='rbf': # 径向基函数(radial bias function)
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kernelTuple[1]**2)) #返回生成的结果
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K

# 计算差值Ek（参考《统计学习方法》p127公式7.105）
def calcEk(oS, k):
    # Ei = sum(aj*yj*kij+b)-yi
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

# 计算并更新E
def updateEk(oS, k):
    Ek = calcEk(oS, k) # 计算 E
    oS.eCache[k] = [1, Ek] # 更新E

def selectJrand(i,m): #在0-m中随机选择一个不是i的整数
    j=i
    while (j==i):
        j=int(random.uniform(0,m))
    return j

# 随机选取aj，并返回其行数j及E值
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]  # 返回矩阵中的非零位置的行数,即不满足KTT的行数
    if (len(validEcacheList)) > 1:
        # 遍历间隔边界上的支持向量点，直到目标函数有足够的下降
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):  # 返回步长最大的aj
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m) #在0-m中随机选择一个不是i的 j
        Ej = calcEk(oS, j)
    return j, Ej

# 求沿着约束方向经剪辑后 alpha2_best的解
def clipAlpha(aj,H,L):  #保证a在L和H范围内（L <= a <= H）
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

# （内层循环）SMO子问题——二变量二次规划解析问题，首先检验ai是否满足KKT条件，如果不满足，随机选择aj进行优化，更新ai,aj,b值
def innerL(i, oS): #输入参数i和所有参数数据
    Ei = calcEk(oS, i) #计算E值

    # 检验这行数据是否符合KKT条件 参考《统计学习方法》p128公式7.111-113
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #随机选取aj，并返回其行数以及E值
        alphaIold = oS.alphas[i].copy() # 第i行 alpha
        alphaJold = oS.alphas[j].copy() # 第j行 alpha
        # 以下代码的公式参考《统计学习方法》p126
        # 二变量优化问题转化为单变量alpha2的优化问题，L和 H是alpha2所在对角线段端点的界
        # 若y1 != y2 则 alpha1-alpha2 = k；则 L = max(0,k) ,H = min (C,C-k)
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i]) # 若 k<0,L=-k;若 k>0,L = 0
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i]) # 若 k<0,H=C-k;若 k>0,H = C
        # 若y1 = y2 则alpha1 + alpha2 = k; 则 L = max(0,k-C) ,H = min (C,k)
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C) # 若 k<C,L=0;若 k>C,L = k-C
            H = min(oS.C, oS.alphas[j] + oS.alphas[i]) # 若 k<C,H=k;若 k>C,H = C
        if L==H:
            # print("L==H")
            return 0
        # 每次更新完alpha之后要更新E值
        # 沿着约束方向未经剪辑 alpha2_best = alpha_old - y2(E1-E2)/eta
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #参考《统计学习方法》p127公式7.107
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta # 未经剪辑时alpha的解，参考《统计学习方法》p127公式7.106
        # 沿着约束方向经剪辑后 alpha2_best的解为 alpha2_best (L<alpha2_best<H) 或 L (alpha2_best <L) 或 H (alpha2_best>H)
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L) # 经剪辑后alpha的解，参考《统计学习方法》p127公式7.108
        updateEk(oS, j) # 更新E值
        if (abs(oS.alphas[j] - alphaJold) < oS.tol): #alpha变化大小阀值（自己设定）
            # print("j not moving enough")
            return 0
        # alpha1_new = alpha1_old + y1y2(alpha2_old-alpha_new)
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#参考《统计学习方法》p127公式7.109
        updateEk(oS, i) # 更新E值
        # 两个变量优化后，要重新计算阈值 b
        #以下求解b的过程，参考《统计学习方法》p129公式7.114-7.116
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        # 如两个变量alpha同时在（0，C）之间，则 new_b1 = new_b2；若两个变量等于0或C，则选择中点作为 new_b
        if (0 < oS.alphas[i]<oS.C):
            oS.b = b1
        elif (0 < oS.alphas[j]<oS.C):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        # 变量解满足 KKT 条件
        return 1

    else:
        # 变量解不满足 KKT 条件
        return 0

#SMO函数，用于快速求解出alpha
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)): #输入参数：数据特征，数据类别，参数C，阀值toler，最大迭代次数，核函数（默认线性核）
    # 调用optStruct类，改变数据存储格式
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)): # 外层循环
        print("SMO外层循环迭代第: %d次开始" % iter)
        alphaPairsChanged = 0

        # 外层循环：若间隔边界上的支持向量点，即满足 alpha在（0，C）的样本点，若都满足KKT条件，则遍历整个训练集是否满足KKT条件

        # 若内层循环选择的 alpha2不能使得目标函数有足够的下降，则遍历间隔边界上的支持向量点，依次将其作为 alpha2使用，直到目标函数有足够的下降
        # 若找不到合适的 alpha2，则遍历训练数据集
        # 若仍然找不到 alpha2，则放弃第一个 alpha1，在通过外层循环找其他 alpha1

        if entireSet:
            for i in range(oS.m): #遍历所有间隔边界数据，即alpha在(0,C)之间 的样本点
                alphaPairsChanged += innerL(i,oS) #检验变量解是否满足KKT条件，满足为1，不满足为0
                # print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)) #显示第多少次迭代，那行特征数据使alpha发生了改变，这次改变了多少次alpha
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs: # 遍历非边界的数据，
                alphaPairsChanged += innerL(i,oS) #检验变量解是否满足KKT条件，满足为1，不满足为0
                # print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))

        iter += 1

        if entireSet: # 变量解满足KKT条件
            entireSet = False # 令循环停止
            print("变量解满足KKT条件，循环停止")
        elif (alphaPairsChanged == 0): # 变量解都不满足KKT条件
            print("变量解不满足KKT条件，循环继续")
            entireSet = True
        print("SMO外层循环迭代第: %d次结束" % iter)
    return oS.b,oS.alphas

def useRbfKernel(data_train,data_test):
    dataArr,labelArr = loadDataSet(data_train) #读取训练数据
    # SMO输入参数：数据特征，数据类别，参数C，阀值toler，最大迭代次数，核函数（类型，带宽）
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', 1.3)) #通过SMO算法得到b和alpha

    datMat=mat(dataArr)
    labelMat = mat(labelArr).transpose()

    svInd=nonzero(alphas)[0]  #选取不为0数据的行数（也就是支持向量）
    sVs=datMat[svInd] #支持向量的数据特征
    labelSV = labelMat[svInd] #支持向量的数据类别（1或-1）
    print("这是支持向量行数，共%d行" %shape(sVs)[0], svInd) #打印出支持向量行数

    m,n = shape(datMat) #训练数据的行列数
    errorCount = 0 # 初始化错误率
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', 1.3)) #将支持向量转化为核函数
        #这一行的预测结果（代码来源于《统计学习方法》p133里面最后用于预测的公式）注意最后确定的分离平面只有那些支持向量决定。
        # f(x) = sign(kernal_i*sum(alpha_i*yi)+b)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        # sign函数 -1 ,if x < 0  |   0, if x==0 |    1, if x > 0
        if sign(predict)!=sign(labelArr[i]):
            errorCount += 1
    print("训练数据错误率为: %f\n" % (float(errorCount)/m)) #打印出错误率

    dataArr_test,labelArr_test = loadDataSet(data_test) #读取测试数据

    datMat_test=mat(dataArr_test)
    labelMat = mat(labelArr_test).transpose()
    m,n = shape(datMat_test)
    errorCount_test = 0

    for i in range(m): #在测试数据上检验错误率
        kernelEval = kernelTrans(sVs,datMat_test[i,:],('rbf', 1.3))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelMat[i]):
            errorCount_test += 1
    print("测试数据错误率为: %f" % (float(errorCount_test)/m))

    # # SMO输入参数：数据特征，数据类别，参数C，阀值toler，最大迭代次数，核函数（类型，带宽）
    # b_test,alphas_test = smoP(dataArr_test, labelArr_test, 200, 0.0001, 10000, ('rbf', 1.3)) #通过SMO算法得到b和alpha
    # datMat_test=mat(dataArr_test)
    # labelMat_test = mat(labelArr_test).transpose()
    #
    # svInd_test=nonzero(alphas_test)[0]  #选取不为0数据的行数（也就是支持向量）
    # sVs=datMat[svInd_test] #支持向量的数据特征
    # labelSV_test = labelMat_test[svInd_test] #支持向量的数据类别（1或-1）
    #
    # m_test,m_test = shape(datMat_test)
    # errorCount_test = 0
    #
    # for i in range(m_test): #在测试数据上检验错误率
    #     kernelEval_test = kernelTrans(sVs,datMat_test[i,:],('rbf', 1.3))
    #     predict_test=kernelEval_test.T * multiply(labelSV_test,alphas_test[svInd_test]) + b_test
    #     if sign(predict_test)!=sign(labelMat_test[i]):
    #         errorCount_test += 1
    # print("测试数据错误率为: %f" % (float(errorCount_test)/m_test))

#主程序
def main():
    filename_traindata='./train_data.txt'
    filename_testdata='./train_data.txt'
    useRbfKernel(filename_traindata,filename_testdata)

if __name__=='__main__':
    main()

# https://blog.csdn.net/csqazwsxedc/article/details/71513197
# https://www.jiqizhixin.com/articles/2018-10-17-20