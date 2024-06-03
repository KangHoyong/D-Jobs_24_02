
"""
- 단어 백터 차원 = 100
- 문장 길이 = 500
- 배치 크기 32
- 데이터 개수 2만
- LSTM 은닉층 크기 128
- 분류하고자 하는 클래스 개수 2 
"""
import torch.nn as nn

class TextClassifier(nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        # vocab_size 100 /  embedding_dim 500 / hidden_dim 128 / output_dim 2
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x) :
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        last_hidden = hidden.squeeze(0)
        logits = self.fc(last_hidden)

        return logits
