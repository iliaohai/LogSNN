import pandas as pd
import transformers as tfs
from torch import nn
from torch import optim
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
from transformers import AdamW, BertModel, BertTokenizer

from t5_snn import T5ClassificationModelSNN
from util.logutil import Logger
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差

# 记录日志
filename = "output/t5-small/run_bgl"
time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
filepath = filename + "_" + time_stamp + ".log"
log = Logger(filepath, level='debug')

'''
    0.Create model
'''
bert_classifier_model = T5ClassificationModelSNN().cuda()

'''
    1.Load dateset
'''
file_name = 'dataset/BGL/BGL_sequence.csv'

# 加载数据集
train_sets = pd.read_csv(file_name, header=0)  # 训练集
# 数据集大小分割，分别取10%，20%，50%
train_sets = train_sets[0: int(len(train_sets) * 1.0)]
x = train_sets['sequence'].values
y = train_sets['label'].values
# 训练和测试集分配
train_inputs, dev_inputs, train_targets, dev_targets = train_test_split(x, y, test_size=0.20, random_state=32)

'''
    2.Set batch
'''
batch_size = 160
batch_count = int(len(train_inputs) / batch_size)
batch_train_inputs, batch_train_targets = [], []
for i in range(batch_count):
    j = i * batch_size
    batch_sentences = []
    while j < (i+1) * batch_size:
        # 每一列CLS
        # print("第", j, "列")
        # print(train_inputs[j])
        sens = train_inputs[j][1: len(train_inputs[j])-2]
        sens = sens.split(",")
        cls = ""
        for e in sens:
            e = e.replace(' ', '')
            e = e.replace("'", '')
            cls += e + " "
        
        # 长度截断
        if len(cls) > 510:
            cls = cls[0: 510]
        cls = "[CLS]" + cls + "[SEB]"
        # print(cls)
        batch_sentences.append(cls)
        j = j + 1

    batch_train_inputs.append(batch_sentences)  # 带batch的cls
    batch_train_targets.append(train_targets[i*batch_size: (i+1)*batch_size])  # 带batch的label

'''
    3.train
'''
epochs = 320
# lr = 0.01
print_every_batch = 1
optimizer = AdamW(bert_classifier_model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# 记录最好结果
max_acc, max_f1, max_p, max_r, best_epoch = 0.0, 0.0, 0.0, 0.0, 0
for epoch in range(epochs):
    print_avg_loss = 0
    bert_classifier_model.train()
    for i in range(batch_count):
        # 正向传播
        inputs = batch_train_inputs[i]  # inputs中有64条训练数据
        labels = batch_train_targets[i]  # labels中有64条标签
        labels = torch.tensor(labels).cuda()
        optimizer.zero_grad()
        outputs = bert_classifier_model(inputs)
        loss = criterion(outputs, labels)
        print_avg_loss += loss.item()

        # 反向梯度信息
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bert_classifier_model.parameters(), 1.0)
        optimizer.step()

        if i % print_every_batch == (print_every_batch - 1):
            log.logger.info("epoch:%d" % epoch)
            log.logger.info("Batch: %d, Loss: %.4f" % ((i + 1), print_avg_loss / print_every_batch))
            print_avg_loss = 0

    '''
        4.evl
    '''
    bert_classifier_model.eval()
    total = len(dev_targets)
    y_pred = []
    y_true = []
    with torch.no_grad():
        for j in range(total):
            dev_input = dev_inputs[j]
            sens = dev_input[1: len(dev_input) - 2]
            sens = sens.split(",")
            dev_input = ""
            for e in sens:
                e = e.replace(' ', '')
                e = e.replace("'", '')
                dev_input += e + " "
            
            # 长度截断
            if len(dev_input) > 510:
                dev_input = dev_input[0: 510]
            dev_input = "[CLS]" + dev_input + "[SEB]"
            outputs = bert_classifier_model([dev_input])  # 此处需要传数组，所以加一个[]
            logits = outputs
            y_pred.append(torch.argmax(logits, axis=1)[0].item())
            y_true.append(dev_targets[j])

    print(y_true)
    print(y_pred)
    f1score = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')  # 0.2222222222222222
    recall = recall_score(y_true, y_pred, average='macro')
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    log.logger.info("***************** Result ******************")
    log.logger.info("epoch:%d" % epoch)
    log.logger.info("Accuracy:%s" % acc)
    log.logger.info("F1:%s" % f1score)
    log.logger.info("precision:%s" % precision)
    log.logger.info("recall:%s" % recall)
    log.logger.info("mse:%s" % mse)
    log.logger.info("mae:%s" % mae)
    if (acc + f1score)/2 > (max_acc + max_f1) / 2:
        max_acc, max_f1, max_p, max_r, best_epoch = acc, f1score, precision, recall, epoch
    log.logger.info("max_accuracy:%s" % max_acc)
    log.logger.info("max_f1:%s" % max_f1)
    log.logger.info("max_p:%s" % max_p)
    log.logger.info("max_r:%s" % max_r)
    log.logger.info("best_epoch:%d" % best_epoch)
    log.logger.info("******************************************")

    # # 绘制roc图
    # Roc_figure.plot_ROC(y_true, y_pred, "(a) ROC based on HDFS dataset")

    # 导出预测结果
    # result = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    # result.to_csv('../../../Dataset/HDFS/result_BERT-Log.csv', index=False)
