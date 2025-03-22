import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import subprocess

# Đọc dữ liệu từ file CSV
# Sử dụng tham số 'on_bad_lines' thay thế 'error_bad_lines'
data = pd.read_csv('AI.csv', header=None, on_bad_lines='skip')

questions = data[0].tolist()
answers = ["<start> " + ans + " <end>" for ans in data[1].tolist()]

# Tạo tokenizer mã hóa 
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)

# Biến đổi các câu thành chuỗi các số
question_sequences = tokenizer.texts_to_sequences(questions)
answer_sequences = tokenizer.texts_to_sequences(answers)

# Padding các chuỗi để có cùng chiều dài
max_question_len = max(len(seq) for seq in question_sequences)
max_answer_len = max(len(seq) for seq in answer_sequences)
question_sequences = pad_sequences(question_sequences, maxlen=max_question_len, padding='post')
answer_sequences = pad_sequences(answer_sequences, maxlen=max_answer_len, padding='post')

# Tạo từ điển ngược để giải mã
word_index = tokenizer.word_index
index_word = {v: k for k, v in word_index.items()}
vocab_size = len(word_index) + 1

# Seq2Seq
embedding_dim = 50
latent_dim = 256

# mã hóa
encoder_inputs = Input(shape=(max_question_len,))
encoder_embedding = Embedding(vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# giải mã
decoder_inputs = Input(shape=(max_answer_len,))
decoder_embedding = Embedding(vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Seq2Seq
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Chuẩn bị dữ liệu đầu vào và đầu ra cho bộ giải mã
decoder_target_sequences = np.zeros((len(answers), max_answer_len), dtype="int32")
for i, seq in enumerate(answer_sequences):
    if len(seq) > 1:
        decoder_target_sequences[i, :len(seq)-1] = seq[1:]

# Tạo mô hình dự đoán cho bộ mã hóa và giải mã
encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

# dự đoán câu trả lời
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word_index['start']
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = index_word.get(sampled_token_index, '')
        decoded_sentence += ' ' + sampled_word
        if sampled_word == '<end>' or len(decoded_sentence.split()) > max_answer_len:
            stop_condition = True
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    return decoded_sentence.replace('<end>', '').strip()

# tiền xử lý
def preprocess_input(text):
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=max_question_len, padding='post')
    return padded_seq


# Hàm xử lý phản hồi
def chatbot_response(user_input, use_ollama=True):
    if use_ollama:
        ollama_response = ollama_generate(user_input)
        if ollama_response:
            return ollama_response
        print("Ollama không trả về kết quả. Sử dụng Seq2Seq.")
    input_seq = preprocess_input(user_input)
    return decode_sequence(input_seq)

user_input = "how are you"
response = chatbot_response(user_input, use_ollama=True)
print("Bot:", response)
