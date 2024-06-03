
def texts_to_sequences(tokenized_x_data, word_to_index) :
    encoded_x_data = []
    for sent in tokenized_x_data :
        # sent -> ['졸라', '잼', '있', '음', 'ㅋㅋ']
        index_sequences = []
        for word in sent :
            # word -> '졸라', '잼', '있', '음', 'ㅋㅋ'
            try :
                index_sequences.append(word_to_index[word])

            except KeyError :
                # 없으면 태깅
                index_sequences.append(word_to_index['<UNK>'])
                pass
        encoded_x_data.append(index_sequences)

    return encoded_x_data

