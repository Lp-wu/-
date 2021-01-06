"""
#homework_1:linear regression
"""
import pandas as pd
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes

if __name__ == '__main__':
    #load train.csv
    data = pd.read_csv('../data/train.csv',encoding = 'big5')   #big5格式编码，使用于繁体字地区
    print(data)

    #preprocessing
    print(type(data))
    data = data.iloc[:, 3:] #用整数指定位置选择所有行,从第四列开始选
    print(data)
    print(data=='NR')   #为了布尔索引操作，data中等于NR的地方为True，不等于NR的地方为False
    print(data[data=='NR'])
    data[data == 'NR'] = 0  #使用布尔向量来过滤数据，可以操作得到的数据



    raw_data = data.to_numpy()  #行表示一年中的第多少天
    print(raw_data,type(raw_data),raw_data.shape)#18x20x12

    #20*18*12=4320
    #將原始 4320 * 18 的资料依照每个月分组成 12 个 18 (features) * 480 (hours) 的資料,依每个特征分组。
    month_data = {}
    for month in range(12):
        sample = np.empty([18,480]) #20x24=480 ; 每次用sample只记录1个月的特征数据; 是一个表
        for day in range(20):   #提取每个月每天的相同特征为一组
            #将每天24h的18个特征装进去
            #在day循环中，每次每天的18个特征同时更新
            sample[:,day*24:(day+1)*24] = raw_data[18*(20*month+day):18*(20*month+day+1),:]

        month_data[month] = sample  #用字典将每个月的特征数据存储起来

    #每10小时为一组data，步进值为1h，所以一个月共有（480-10）+1 = 471组data
    x = np.empty([12*471,18*9], dtype=float) #0到9共有10个; 一行表示一组data; 18*9=162
    y = np.empty([12*471,1], dtype=float)
    for month in range(12):
        for day in range(20):
            for hour in range(24):

                if day==19 and hour > 14:
                    #0<=hour<=23
                    #15+9==24;每月的最后一天时,注意每月的最后一笔data的截止小时点。因为可能超出
                    continue

                # 一个月有471笔data，例如第一个月从0每次加一到14+19*24=470，类似于不同的进制
                x[month*471+day*24+hour , :] = month_data[month][: , day*24+hour+0 : day*24+hour+9].reshape(1,-1)  #存储每一组data
                y[month*471+day*24+hour , 0] = month_data[month][9 , day*24+hour+9]  #每组data的第10个PM2.5值
    print(x)
    print(y)

    #normolize
    #标准化也叫标准差标准化，经过处理的数据符合标准正态分布
    #不同笔data的相同特征分量的标准化
    mean_x = np.mean(x, axis = 0)    #18*9 纵向自上而下求平均
    print(mean_x, mean_x.shape)     #(162,)
    std_x = np.std(x, axis = 0)     #18*9
    print(std_x, std_x.shape)       #(162,)
    print(len(x))

    for i in range(len(x)): #12*471=5652
        for j in range(len(x[0])):  #18*9=162
            if std_x[j] != 0:   #如果某纵向的标准差不为0
                x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
    print(x)

    #split training data and validation set
    x_train_set = x[: math.floor(len(x) * 0.8), :]  #对浮点数向下取整
    y_train_set = y[: math.floor(len(y) * 0.8), :]
    x_validation = x[math.floor(len(x)*0.8): , : ]
    y_validation = y[math.floor(len(x)*0.8): , : ]
    print(x_train_set)
    print(y_train_set)
    print(x_validation)
    print(y_validation)
    print(len(x_train_set))
    print(len(y_train_set))
    print(len(x_validation))
    print(len(y_validation))

    #training
    dim = 18*9 + 1  #因为常数项的存在，所以 dimension (dim) 需要多加一行
    w = np.zeros([dim,1])  #每一组data的权值和偏移量
    x = np.concatenate((np.ones([12*471,1]), x), axis = 1).astype(float)    #拼接上常数项，为了bias项
    learning_rate = 0.01
    iter_time = 1000
    adagrad = np.zeros([dim, 1])
    eps = 0.0000001  #esp项避免adagrad的分母为0，而加的极小数值
    loss_list = []
    for t in range(iter_time):
        #先np.power((np.dot(x, w) - y), 2)
        #再每笔data的loss的求和
        #最后所有data的loss和的开根号
        #根均方差
        #所以对于每笔data来说，每次梯度下降为，所有笔data的gradient用一个列向量表示：2 * np.dot(x.transpose(), np.dot(x, w) - y)
        loss = np.sqrt(np.sum(np.power((np.dot(x,w) - y), 2)) / (12 * 471))  #root mean square error
        loss_list.append(loss)
        if(t % 100 ==0):
            print(str(t) + ":" + str(loss))
        gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)  #mse's gradient; dim*1
        #print(gradient.shape)
        adagrad += gradient**2  #dim*1
        w = w - learning_rate * gradient / np.sqrt(adagrad + eps)

    print(loss_list)
    loss_list = np.array(loss_list)

    '''
    #绘制的图形在一个默认的figure中
    plt.plot(range(iter_time), loss_list, 'o-', label='loss value')
    plt.xlabel('iter_num')
    plt.ylabel('loss value')
    plt.title('loss value at every iteration')
    plt.legend(loc='best')

    tx0 = iter_time-100
    tx1 = iter_time
    ty0 = 22.5
    ty1 = 25
    sx = [tx0, tx1, tx1,  tx0, tx0]
    sy = [ty0, ty0, ty1,  ty1, ty0]
    plt.plot(sx, sy, 'purple')

    plt.show()
    '''

    #创建figure，控制更多的参数
    fig = plt.figure(figsize=(8,4))

    #在主表上绘制图像
    plt.plot(range(iter_time), loss_list, 'o-', label='loss value')
    plt.xlabel('iter_num')
    plt.ylabel('loss value')
    plt.title('loss value at every iteration')
    plt.legend(loc='best')

    #在主表上添加子表
    #只使用plt
    #子图画的是figure的百分比,分别从figure left和bottom的53%和30%的位置开始绘制, 宽高分别是figure的35%和40%
    left, bottom, width, height = 0.53, 0.3, 0.35, 0.4
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.plot(range(iter_time-25,iter_time), loss_list[len(loss_list)-25:])
    #ax2.plot(range(iter_time-25,iter_time), loss_2)
    ax2.set_title('sub img')

    '''
    #使用plt和inset_axes
    #from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax1 = fig.add_subplot(1, 1, 1)
    tx0 = iter_time-50
    tx1 = iter_time+25
    ty0 = 23
    ty1 = 23.5
    sx = [tx0, tx1, tx1,  tx0, tx0]
    sy = [ty0, ty0, ty1,  ty1, ty0]
    plt.plot(sx, sy, 'purple')

    axins = inset_axes(ax1, width=1.5, height=1.5, loc='right')
    axins.plot(range(iter_time), loss_list)
    axins.axis([iter_time-25, iter_time, 23.134, 23.15])
    '''

    plt.savefig('compare_loss.png')
    plt.show()

    np.save('bias_and_weight.npy', w)
    print(w)

    #testing data pre-processing
    test_data = pd.read_csv('../data/test.csv', header = None, encoding= 'big5')
    print(test_data)
    test_data = test_data.iloc[:, 2: ]
    print(test_data)
    test_data[test_data == 'NR'] = 0
    test_data = test_data.to_numpy()
    test_x = np.empty([240, 18*9], dtype = float)
    mean_x = np.mean(test_x, axis=0)  # 18*9 纵向自上而下求平均
    std_x = np.std(test_x, axis=0)  # 18*9
    for i in range(240):
        test_x[i, :] = test_data[18*i : 18*(i+1), :].reshape(1, -1)
    for i in range(len(test_x)):
        for j in range(len(test_x[0])):
            if std_x[j] != 0:
                test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
    test_x = np.concatenate((np.ones([240,1]),test_x), axis = 1).astype(float)
    print(test_x)

    #prediction
    w = np.load('bias_and_weight.npy')
    ans_y = np.dot(test_x, w)

    #save prediction to csv file
    with open('submit.csv', mode='w', newline='') as submit_file_flag:
        csv_writer = csv.writer(submit_file_flag)
        colume_header = ['id', 'value']
        print(colume_header)
        csv_writer.writerow(colume_header)
        for i in range(len(ans_y)):
            row = ['id' + str(i), ans_y[i][0]]
            csv_writer.writerow(row)
            print(row)








    












