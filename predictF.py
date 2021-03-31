# 预测股票涨跌

import pandas as pd
from sklearn import svm,preprocessing
import matplotlib.pyplot as plt

# 读取数据文件
origDf = pd.read_csv('./SP500.csv',encoding='gbk')

df = origDf[['Close','High','Low','Open','Volume','Date']]
# diff列表示 本日和上日收盘价 的差
df['diff'] = df['Close']-df['Close'].shift(1)
df['diff'].fillna(0, inplace = True)

# up列表示 本日是否上涨 1涨，0跌
df['up'] = df['diff']
df['up'][df['diff']>0] = 1
df['up'][df['diff']<=0] = 0

# 预测值暂且初始化为 0
df['predictForUp'] = 0

# 目标值是真实的涨跌情况
target = df['up']
length = len(df)
trainNum = int(length*0.8)
predictNum = length-trainNum

# 选择指定列作为特征列
feature = df[['Close','High','Low','Open','Volume']]
# 标准化处理特征值
feature = preprocessing.scale(feature)

featureTrain = feature[1:trainNum-1]
targetTrain = target[1:trainNum - 1]

svmTool = svm.SVC(kernel='linear')
svmTool.fit(featureTrain,targetTrain)
predictedIndex = trainNum
feature[predictedIndex:predictedIndex+1]
while predictedIndex<length:
    testFeature = feature[predictedIndex:predictedIndex+1]
    predictForUp = svmTool.predict(testFeature)
    df.loc[predictedIndex,'predictForUp']=predictForUp
    predictedIndex = predictedIndex+1

dfWithPredicted = df[trainNum:length]
# 切少量数据显示
dfWithPredicted = dfWithPredicted.loc[13489:13499]

figure = plt.figure()
(axClose,axUpOrDown) = figure.subplots(2,sharex=True)
dfWithPredicted['Close'].plot(ax=axClose)
dfWithPredicted['predictForUp'].plot(ax=axUpOrDown,color='red',label='Predicted Data')
dfWithPredicted['up'].plot(ax=axUpOrDown,color='blue',label='Real Data')
plt.legend(loc='best') # 绘制图例
# 设置x轴坐标标签和旋转角度
major_index = dfWithPredicted.index[dfWithPredicted.index%2==0]
major_xtics = dfWithPredicted['Date'][dfWithPredicted.index%2==0]
plt.xticks(major_index,major_xtics)
plt.setp(plt.gca().get_xticklabels(),rotation=30)
plt.title('svm predict')
plt.rcParams['font.sans-serif']=['SimHei']
plt.show()

# 以预测股票涨跌案例入门基于SVM的机器学习 https://blog.csdn.net/sxeric/article/details/99620687