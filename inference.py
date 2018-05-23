import nltk
import os
import string
import tensorflow as tf

from tokenizedata import TokenizeData
from modelcreator import seq2seqModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class SessionData:
    def __init__(self):
        self.session_dict = {}

    def add_session(self):
        items = self.session_dict.items()
        if items:
            last_id = max(k for k, v in items)
        else:
            last_id = 0
        new_id = last_id + 1

        self.session_dict[new_id] = ChatSession(new_id)
        return new_id

    def get_session(self, session_id):
        return self.session_dict[session_id]

class ChatSession:
    def __init__(self, session_id):
        """
        Args:
            session_id: The integer ID of the chat session.
        """
        self.session_id = session_id

        self.last_question = None
        self.last_answer = None
        self.update_pair = True

    def after_prediction(self, new_question, new_answer):
        """
        Last pair is updated after each prediction except in a few cases.
        """
        if self.update_pair:
            self.last_question = new_question
            self.last_answer = new_answer

class BotPredictor(object):
    def __init__(self, session, corpus_dir, result_dir, result_file):

        self.session = session

        # Prepare data and hyper parameters
        print("# Prepare dataset placeholder and hyper parameters ...")
        tokenize_data = TokenizeData(corpus_dir=corpus_dir, training=False)

        self.session_data = SessionData()

        self.hparams = tokenize_data.hparams
        self.src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        src_dataset = tf.data.Dataset.from_tensor_slices(self.src_placeholder)
        self.infer_batch = tokenize_data.get_inference_batch(src_dataset)

        # Create model
        print("# Creating inference model ...")
        self.model = seq2seqModel(training=False, tokenize_data=tokenize_data,
                                  batch_input=self.infer_batch)
        # Restore model weights
        print("# Restoring model weights ...")
        self.model.saver.restore(session, os.path.join(result_dir, result_file))

        self.session.run(tf.tables_initializer())

    def predict(self, session_id, question, html_format=False):
        chat_session = self.session_data.get_session(session_id)
        #chat_session.before_prediction()  # Reset before each prediction

        if question.strip() == '':
            answer = "Don't you want to say something to me?"
            chat_session.after_prediction(question, answer)
            return answer

        new_sentence = question
        para_list = []

        for pre_time in range(2):
            tokens = nltk.word_tokenize(new_sentence.lower())
            tmp_sentence = [' '.join(tokens[:]).strip()]

            self.session.run(self.infer_batch.initializer,
                             feed_dict={self.src_placeholder: tmp_sentence})

            outputs, _ = self.model.infer(self.session)

            outputs = outputs[0]

            eos_token = self.hparams.eos_token.encode("utf-8")
            outputs = outputs.tolist()[0]

            if eos_token in outputs:
            	outputs = outputs[:outputs.index(eos_token)]

            out_sentence = self._get_final_output(outputs, chat_session, html_format=html_format)
            #
            chat_session.after_prediction(question, out_sentence)
            return out_sentence

    def _get_final_output(self, sentence, chat_session, para_list=None, html_format=False):
        #print(sentence)#testing/debug
        sentence = b' '.join(sentence).decode('utf-8')
        if sentence == '':
            return "I don't know what to say.", False

        last_word = None
        word_list = []
        for word in sentence.split(' '):
            word = word.strip()
            if not word:
                continue

            if (last_word is None or last_word in ['.', '!', '?']) and not word[0].isupper():
                word = word.capitalize()

            if not word.startswith('\'') and word != 'n\'t' \
                and (word[0] not in string.punctuation or word in ['(', '[', '{', '``', '$']) \
                and last_word not in ['(', '[', '{', '``', '$']:
                word = ' ' + word

            word_list.append(word)
            last_word = word

        return ''.join(word_list).strip()
        