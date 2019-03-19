import tensorflow as tf
import bert
import bert_data_helper
import os
import warnings
import numpy as np
from tqdm import tqdm
import argparse
import math

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


parser = argparse.ArgumentParser(description='tensorboard path')
parser.add_argument('-data_path', help="data_path", required=True)
parser.add_argument('-voca_path', help="voca_path", required=True)
parser.add_argument('-tensorboard_path', required=True)
parser.add_argument('-save_path', help="save_path", required=True)
parser.add_argument('-restore', default=0, type=int)
args = parser.parse_args()

#save_path = './saver/'
#tensorboard_path = './tensorboard/'

#data_path='./DATA/bpe_out/BPE_kowiki'
#voca_path='./DATA/bpe_out/BPE_voca'

data_helper = bert_data_helper.data_helper(args.data_path, args.voca_path)


def get_lr(step_num, init_lr=1e-4, warmup_step=10000, num_train_step=100001, power=1):
	end_learning_rate = 0.0

	if step_num < warmup_step:
		warmup_percent_done = step_num / warmup_step
		lr = init_lr * warmup_percent_done
	else:
		# https://www.tensorflow.org/api_docs/python/tf/train/polynomial_decay
		# https://github.com/google-research/bert/blob/master/optimization.py (create optimizer func)
		lr = (init_lr - end_learning_rate) * ((1 - step_num / num_train_step) ** (power)) + end_learning_rate
	
	return lr

'''
import matplotlib.pyplot as plt
data = []
for i in range(1,100001):
	data.append(get_lr(i, power=1))
print(data[0])
print(data[-1])
plt.plot(range(1, 100001), data)
plt.show()
'''

def train(model, epoch, data):
	loss = 0

	total_iter = len(data)
	for i in tqdm(range( total_iter ), ncols=50):
		step_num = ((epoch-1)*total_iter)+(i+1)
		lr = get_lr(step_num=step_num, warmup_step=40000 ,num_train_step=400001 ,init_lr=1e-4) # epoch: [1, @], i:[0, total_iter)
		batch_dataset, batch_boolean_mask, batch_is_next_target, batch_A_B_boundary, batch_masked_LM_target = data[i]
		
		#print('current bucket size', len(batch_dataset[0]))
		train_loss, _ = sess.run([model.train_cost, model.minimize], 
				{
					model.lr:lr,
					model.encoder_input:batch_dataset, # [batch_size, token_length]
					model.boolean_mask:batch_boolean_mask, # [batch_size, token_length]
					model.is_next_target:batch_is_next_target, # [batch_size]
					model.A_B_boundary_length:batch_A_B_boundary, # [batch_size]
					model.masked_LM_target:batch_masked_LM_target, # [# mask]
					model.keep_prob:0.9 # dropout rate = 0.1		
				}
			)
		if math.isnan(train_loss):
			print('nan', len(batch_masked_LM_target))
			print('target_max', batch_masked_LM_target.max())
			print('target_min', batch_masked_LM_target.min())
			print('mask_max', batch_dataset[batch_boolean_mask].max())
			print('mask_min', batch_dataset[batch_boolean_mask].min(), '\n')

		loss += train_loss
	print('lr', lr)
	return loss/total_iter


def test(model, data):
	batch_size = 64
	total_masked_LM_data = 0
	total_is_next_data = 0

	masked_LM_accuracy = 0
	is_next_accuracy = 0

	total_iter = len(data)
	for i in tqdm(range( total_iter ), ncols=50):
		batch_dataset, batch_boolean_mask, batch_is_next_target, batch_A_B_boundary, batch_masked_LM_target = data[i]
		total_masked_LM_data += len(batch_masked_LM_target)
		total_is_next_data += len(batch_is_next_target)

		masked_LM_correct, is_next_correct = sess.run([model.masked_LM_correct, model.is_next_correct], 
				{
					model.encoder_input:batch_dataset, # [batch_size, token_length]
					model.boolean_mask:batch_boolean_mask, # [batch_size, token_length]
					model.is_next_target:batch_is_next_target, # [batch_size]
					model.A_B_boundary_length:batch_A_B_boundary, # [batch_size]
					model.masked_LM_target:batch_masked_LM_target, # [# mask]
					model.keep_prob:1 # dropout rate = 0		
				}
			)
		masked_LM_accuracy += masked_LM_correct
		is_next_accuracy += is_next_correct

	return masked_LM_accuracy/total_masked_LM_data, is_next_accuracy/total_is_next_data



def run(model, data_helper, batch_size, token_length, bucket, dataset_shuffle=True, restore=0):
	if restore != 0:
		print('restore:', restore)
		saver.restore(sess, os.path.join(args.save_path, str(restore)+".ckpt"))
	

	with tf.name_scope("tensorboard"):
		train_loss_tensorboard = tf.placeholder(tf.float32, name='train_loss_tensorboard')
		masked_LM_accuracy_tensorboard = tf.placeholder(tf.float32, name='masked_LM_accuracy_tensorboard')
		is_next_accuracy_tensorboard = tf.placeholder(tf.float32, name='is_next_accuracy_tensorboard')

		train_summary = tf.summary.scalar("train_loss", train_loss_tensorboard)
		masked_LM_summary = tf.summary.scalar("masked_LM_accuracy", masked_LM_accuracy_tensorboard)
		is_next_summary = tf.summary.scalar("is_next_accuracy", is_next_accuracy_tensorboard)
				
		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter(args.tensorboard_path, sess.graph)

	if not os.path.exists(args.save_path):
		print("create save directory")
		os.makedirs(args.save_path)
	

	for epoch in range(restore+1, 200+1):
	# 결과 보고 for문 안으로 집어넣고 테스트하자.
		data = data_helper.get_batch_dataset(
				bucket_size=[i*bucket for i in range(1, token_length//bucket +1)], 
				token_length=token_length,#512, 
				min_first_sentence_length=3, 
				min_second_sentence_length=5, 
				batch_size=batch_size, 
				dataset_shuffle=dataset_shuffle
			)



		print("epoch:", epoch)

		#train 
		train_loss = train(model, epoch, data)
		print('train_loss:', train_loss) 
		
		#save
		saver.save(sess, os.path.join(args.save_path, str(epoch)+".ckpt"))
		
		masked_LM_accuracy, is_next_accuracy = test(model, data)
		
		print('masked_LM_accuracy:', masked_LM_accuracy) 
		print('is_next_accuracy:', is_next_accuracy)
		print()

		#tensorboard
		summary = sess.run(
				merged, 
				{
					train_loss_tensorboard:train_loss, 
					masked_LM_accuracy_tensorboard:masked_LM_accuracy,
					is_next_accuracy_tensorboard:is_next_accuracy, 
				}
			)		
		writer.add_summary(summary, epoch)
		


bpe2idx, idx2bpe = data_helper.bpe2idx, data_helper.idx2bpe
warmup_steps = 10000#4000 * 8 # paper warmup_steps: 4000(with 8-gpus), so warmup_steps of single gpu: 4000*8
embedding_size = 512#16#512
is_embedding_scale = True
stack = 6#1#6
multihead_num = 8#2#8
pad_idx = bpe2idx['</PAD>']
max_sequence_length = 512#512,
l2_weight_decay = 0.01
label_smoothing = 0.1


print('voca_size:', len(bpe2idx))
print('warmup_steps:', warmup_steps)
print('embedding_size:', embedding_size)
print('stack:', stack)
print('multihead_num:', multihead_num)
print('max_sequence_length:', max_sequence_length)
print('l2_weight_decay:', l2_weight_decay)
print('label_smoothing:', label_smoothing)
print()

config = tf.ConfigProto()
config.gpu_options.allow_growth=True # gpu 필요한만큼만 할당.
sess = tf.Session(config=config)

model = bert.Bert(
		voca_size = len(bpe2idx), 
		embedding_size = embedding_size, 
		is_embedding_scale = True, 
		stack = stack,
		multihead_num = multihead_num,
		pad_idx = bpe2idx['</PAD>'],
		max_sequence_length = max_sequence_length,
		l2_weight_decay = l2_weight_decay,
		label_smoothing=label_smoothing
	)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=10000)

print('run')
run(
		model, 
		data_helper, 
		batch_size = 80, 
		token_length = 128, 
		bucket = 32, 
		dataset_shuffle=True, 
		restore=args.restore
	)


