from transformers import TFAutoModelForSequenceClassification
import tensorflow as tf


model_name = "textattack/bert-base-uncased-SST-2"
save_model_path = f'saved_model/{model_name}/1'
MAX_SEQ_LEN = 100


model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True)
callable = tf.function(model.call)
concrete_function = callable.get_concrete_function([tf.TensorSpec([None, MAX_SEQ_LEN], tf.int32, name="input_ids"), tf.TensorSpec([None, MAX_SEQ_LEN], tf.int32, name="attention_mask")])
model.save(save_model_path, signatures=concrete_function)
