import numpy as np
from gensim.models import word2vec
from keras.layers import Conv1D
from keras.models import Model
from keras.layers import Input, LSTM, Dense
data_path="./poetry.txt"
title_texts=[]
content_texts=[]
max_number=8000
max_content_line=8
max_content_list=1000
max_title_line=10
max_title_list=200
latent_dim=256
input_characters = set()
target_characters = set()
w2v_model = word2vec.Word2Vec.load('word2vec.txt')

title_data = np.zeros((max_number, max_title_line, max_title_list), dtype=np.float32)
content_data=np.zeros((max_number,max_content_line,max_content_list),dtype=np.float32)
target_data=np.zeros((max_number, max_title_line, max_title_list), dtype=np.float32)
max=int(0)
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().encode('utf-8').decode('utf-8-sig').split('\n')
for line in lines[:min(max_number,len(lines)-1)]:
    title_text,content_text=line.split(":")
    title_texts.append(title_text)
    content_texts.append(content_text)
    for char in content_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in title_text:
        if char not in target_characters:
            target_characters.add(char)
# num_encoder_tokens = len(input_characters)
# num_decoder_tokens = len(target_characters)
# print(num_encoder_tokens)
# print(num_decoder_tokens)
# print(title_texts)
# input_token_index = dict( [(char, i)for i, char in enumerate(input_characters)] )
# target_token_index = dict( [(char, i) for i, char in enumerate(target_characters)] )
# for i, (input_text, target_text) in enumerate(zip(content_texts, title_texts)):
#     for line in re.split("[，。]",input_text):
#         t=0
#         n=0
#         d=[]
#         c = np.zeros(4573)
#         for X in line:
#             c[input_token_index[X]]=1.0
#             if n==4:
#                 content_data[i, t] =c
#                 break
#             n +=1
#         t +=1
    # for t, char in enumerate(input_text):
    #     c=np.zeros(4573)
    #     c[input_token_index[char]]=1.0
    #     d = d.extend(c)
    #     if t==4:
    #         content_data[i, t] =d
    #         break
    # for t, char in enumerate(target_text):
    #     title_data[i, t, target_token_index[char]] = 1.0
    #     if t > 0:
    #         target_data[i, t - 1, target_token_index[char]] = 1.0
# max_title_line=0
for txt in title_texts:
    if len(txt)>max_title_line:
        max_title_line=len(txt)
print(max_title_line)
print("max_title_line：",max_title_line)
print("max_number:",len(title_texts))
p=0
for i, (input_text, target_text) in enumerate(zip(title_texts, content_texts)):
    for t, char in enumerate(input_text):
        try:
            c = w2v_model[char]
        except KeyError:
            p +=1
            c = np.zeros(200)
        for r in range(0, 200):
            title_data[i, t, r] = c[r]
        if t>0:
            try:
                c = w2v_model[char]
            except KeyError:
                p += 1
                c = np.zeros(200)
            for r in range(0, 200):
                title_data[i, t-1, r] = c[r]

    d = []
    t=0
    for char in target_text:
        x=0
        try:
            c = w2v_model[char]
        except KeyError:
            c = np.zeros(200)
            print(char)
        if t==0:
            for r in range(0,200):
                content_data[i,x,r]=c[r]
        if  t==1:
            for r in range(200, 400):
                content_data[i, x, r] = c[r-200]
        if  t==2:
            for r in range(400, 600):
                content_data[i, x, r] = c[r-400]
        if  t==3:
            for r in range(600, 800):
                content_data[i, x, r] = c[r-600]
        if t==4:
            for r in range(800, 1000):
                content_data[i,x, r] = c[r-800]
            t=0
            x +=1
            break
        t+=1
print(content_data[0,0,3])
print(p)

X_input = Input(shape=(None,max_content_list))
X = Conv1D(1, 4, strides=1, name='conv0',padding="same")(X_input)
# encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outpus, state_h, state_c = encoder(X)
encoder_state = [state_h, state_c]

decoder_inputs = Input(shape=(None, max_title_list))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_state)
decoder_dense = Dense(max_title_list,activation="relu")
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([X_input, decoder_inputs], decoder_outputs)
model.compile(optimizer="rmsprop", loss='mse')
model.fit([content_data, title_data], target_data,
          batch_size=64,
          epochs =50,
          validation_split=0.2)
model.save('s2s_2.h5')

def vector_to_word(y):

    word=w2v_model.most_similar(positive=y,topn=1)

    return word

encoder_model = Model(X_input, encoder_state)
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs]+decoder_states)

def decode_sequence(input_seq):

    states_value = encoder_model.predict(input_seq)
    target_input = np.zeros((1, 1, max_title_list), dtype=np.float32)
    # target_input = w2v_model["S"].reshape(-1,max_title_list)
    c= w2v_model["S"]
    for r in range(0,200):
        target_input[0,0,r]=c[r]
    # target_input=c
    # target_input=np.expand_dims(w2v_model["S"], axis=0)

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_input] + states_value)
        re_output=output_tokens[0,0,]
        re_output=re_output.reshape(-1,max_title_list)
        # sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = vector_to_word(re_output)

        decoded_sentence += sampled_char[0][0]

        if sampled_char == 'E' or len(decoded_sentence) > max_title_line :
            stop_condition = True
        target_input = np.zeros((1, 1, max_title_list), dtype=np.float32)
        output_tokens=output_tokens[0,0,]
        for r in range(0, 200):
            target_input[0, 0, r] = output_tokens[r]
        # target_input=output_tokens


        states_value = [h, c]

    return decoded_sentence


for seq_index in range(2000, 2100):

    input_seq = content_data[seq_index:seq_index+1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', title_data[seq_index])
    print('Decoded sentence:', decoded_sentence)
    f=open("result.txt","a+",encoding="utf-8")
    f.write("result:"+decoded_sentence+"\n")
    f.close()
