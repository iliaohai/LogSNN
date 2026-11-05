import numpy
import transformers as tfs
from torch import nn
import torch
import snntorch.surrogate as surrogate
import snntorch as snn

class BertaClassificationModelSNN(nn.Module):
        def __init__(self):
            super(BertaClassificationModelSNN, self).__init__()
            model_class, tokenizer_class, pretrained_weights = (tfs.AutoModel, tfs.AutoTokenizer, 'pre-models/bert-base-uncased')
            self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
            self.bert = model_class.from_pretrained(pretrained_weights, num_labels=2)
            self.dense = nn.Linear(768, 2)  # bert默认的隐藏单元数是768， 输出单元是2，表示二分类
            
            # 脉冲神经元
            self.lif1 = snn.Leaky(beta=1.0, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True, threshold=1.0, output=True)

        def forward(self, batch_sentences):
            self.lif1.reset_hidden()  # 显式重置隐藏状态

            batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True, max_length=512,
                                                               padding='longest')  # tokenize、add special token、pad
            input_ids = torch.tensor(batch_tokenized['input_ids']).cuda()
            attention_mask = torch.tensor(batch_tokenized['attention_mask']).cuda()
            bert_output = self.bert(input_ids, attention_mask=attention_mask)
            bert_cls_hidden_state = bert_output[0][:, 0, :].cuda()  # 提取[CLS]对应的隐藏状态
            linear_output = self.dense(bert_cls_hidden_state).cuda()
            
            # 脉冲神经元层
            spks, mem = self.lif1(linear_output)

            return mem