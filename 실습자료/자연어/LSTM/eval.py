import torch
from mecab import MeCab

def predict_test(text, model, word_to_index, index_to_tag, checkpoint_path) :

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.eval()

    test_input = text

    # Tokenize the input text

    # 불용어 정의
    stopwords = [
        '도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고',
        '을', '를', '안', '듯', '과', '와', '네', '틀', '듯', '지', '임', '게'
    ]

    mecab = MeCab()
    tokens = mecab.morphs(test_input)
    tokens = [word for word in tokens if not word in stopwords]
    token_indices = [word_to_index.get(token, 1) for token in tokens]

    # Convert tokens to tensor
    input_tensor = torch.tensor([token_indices], dtype=torch.long).to(device)

    with torch.no_grad() :
        out_pred = model(input_tensor)

        predicted_index = torch.argmax(out_pred, dim=1)
        index_to_tag = index_to_tag[predicted_index.item()]

        return index_to_tag