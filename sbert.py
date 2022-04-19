#data function
def read_chatbot_csv(chatbot_csv):
    samples = []
    with open(chatbot_csv, 'r') as f:
        for line in f:
            line = line.strip('\n')
            line_s = line.split("\t")
            if len(line_s) != 3:
                continue
            query, key, value = line_s
            if value not in ['0', '1']:
                continue
            value = float(value)
            samples.append(InputExample(texts=[query, key], label=value))
    return samples

#load data
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

path = ""
train_csv = path + 'train.tsv'
dev_csv = path + 'dev.tsv'
test_csv = path + 'test.tsv'
train_samples = []
dev_samples = []
test_samples = []
train_samples = read_chatbot_csv(train_csv)
dev_samples = read_chatbot_csv(dev_csv)
test_samples = read_chatbot_csv(test_csv)
print(len(train_samples))
print(len(dev_samples))
print(len(test_samples))

#build model
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from datetime import datetime
from torch import nn
import math

model_save_path = 'sbert_model/biencoder_'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
num_epochs = 5
word_embedding_model = models.Transformer('hfl/chinese-roberta-wwm-ext', max_seq_length=32)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

train_loss = losses.CosineSimilarityLoss(model=model)
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='dev')

#train model
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)
warmup_steps = warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
train_loss = losses.CosineSimilarityLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)


