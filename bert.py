#https://arxiv.org/abs/1706.03762 Attention Is All You Need(Transformer)
#https://arxiv.org/abs/1607.06450 Layer Normalization
#https://arxiv.org/abs/1512.00567 Label Smoothing

import tensorflow as tf #version 1.4
import numpy as np
import os
#tf.set_random_seed(787)

class Bert:
	def __init__(self, voca_size, embedding_size, is_embedding_scale,	stack, 
			multihead_num, pad_idx, max_sequence_length, l2_weight_decay, label_smoothing):
		
		self.voca_size = voca_size
		self.embedding_size = embedding_size
		self.is_embedding_scale = is_embedding_scale # True or False
		self.stack = stack
		self.multihead_num = multihead_num
		self.pad_idx = pad_idx
		self.max_sequence_length = max_sequence_length
		self.l2_weight_decay = l2_weight_decay
		self.label_smoothing = label_smoothing # if 1.0, then one-hot encooding

		
		with tf.name_scope("placeholder"):
			self.lr = tf.placeholder(tf.float32)
			self.encoder_input = tf.placeholder(tf.int32, [None, None], name='encoder_input') 
			self.encoder_input_length = tf.shape(self.encoder_input)[1]
			self.A_B_boundary_length = tf.placeholder(tf.int32, [None], name='A_B_boundary_length')
			self.boolean_mask = tf.placeholder(tf.bool, [None, None], name='boolean_mask')
			self.is_next_target = tf.placeholder(tf.int32, [None], name='is_next_target')
			self.masked_LM_target = tf.placeholder(tf.int32, [None], name='masked_LM_target')
			self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
				# dropout (each sublayers before add and norm)  and  (sums of the embeddings and the PE) and (attention)
	

		with tf.name_scope("embedding_table"):
			with tf.device('/cpu:0'):
				zero = tf.zeros([1, self.embedding_size], dtype=tf.float32) # for padding 
				embedding_table = tf.get_variable( # https://github.com/tensorflow/models/blob/master/official/transformer/model/embedding_layer.py
						'embedding_table', 
						[self.voca_size-1, self.embedding_size], 
						initializer=tf.truncated_normal_initializer(stddev=0.02)
					) # https://github.com/google-research/bert/blob/master/modeling.py
				front, end = tf.split(embedding_table, [self.pad_idx, self.voca_size-1-self.pad_idx])
				self.embedding_table = tf.concat(
						(front, zero, end), 
						axis=0
					) # [self.voca_size, self.embedding_size]

				self.position_embedding_table = tf.get_variable(
						'position_embedding_table',
						[self.max_sequence_length, self.embedding_size],
						initializer=tf.truncated_normal_initializer(stddev=0.02) # https://github.com/google-research/bert/blob/master/modeling.py
					)# [self.max_sequence_length, self.embedding_size]

				self.segment_embedding_table = tf.get_variable(
						'segment_embedding_table',
						[2, self.embedding_size],
						initializer=tf.truncated_normal_initializer(stddev=0.02)
					) # [2, self.embedding_size]


		with tf.name_scope('encoder'):
			encoder_input_embedding, encoder_input_mask = self.embedding_and_PE_SE(
					self.encoder_input, # [N, ?]
					self.A_B_boundary_length # [N]
				)
			self.encoder_embedding = self.encoder(
					encoder_input_embedding, # [N, self.encoder_input_length, self.embedding_size]
					encoder_input_mask # [N, self.encoder_input_length, 1]
				) # [N, self.encoder_input_length, self.embedding_size] * stack
			self.last_encoder_embedding = self.encoder_embedding[-1] # [N, self.encoder_input_length, self.embedding_size]


		with tf.name_scope('pred'):
			#self.boolean_mask: [N, self.encoder_input_length]
			self.masked_position = tf.boolean_mask(self.last_encoder_embedding, self.boolean_mask) # [np.sum(boolean_mask), self.embedding_size]
			self.masked_LM_pred = tf.matmul(self.masked_position, tf.transpose(self.embedding_table)) # [np.sum(boolean_mask), self.voca_size]
			self.is_next_embedding = self.last_encoder_embedding[:, 0, :] # [N, self.embedding_size]
			self.is_next_pred = tf.layers.dense(self.is_next_embedding, units=2, activation=None) # [N, 2]

		with tf.name_scope('train_cost'): 
			# make smoothing target one hot vector
			self.masked_LM_target_one_hot = tf.one_hot(
					self.masked_LM_target, 
					depth=self.voca_size,
					on_value = (1.0-self.label_smoothing) + (self.label_smoothing / self.voca_size), # tf.float32
					off_value = (self.label_smoothing / self.voca_size), # tf.float32
					dtype= tf.float32
				) # [np.sum(boolean_mask), self.voca_size]
			
			self.is_next_target_one_hot = tf.one_hot(
					self.is_next_target, 
					depth=2,
					on_value = (1.0-self.label_smoothing) + (self.label_smoothing / 2), # tf.float32
					off_value = (self.label_smoothing / 2), # tf.float32
					dtype= tf.float32
				) # [N, 2]			

			# calc train_cost
			self.masked_LM_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
					labels = self.masked_LM_target_one_hot, 
					logits = self.masked_LM_pred
				)) # [np.sum(boolean_mask)]
			self.is_next_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
					labels = self.is_next_target_one_hot, 
					logits = self.is_next_pred
				)) # [N]


			# l2 norm
			variables = tf.trainable_variables()
			l2_norm = self.l2_weight_decay * tf.reduce_sum(
					[tf.nn.l2_loss(i) for i in variables if ('LayerNorm' not in i.name and 'bias' not in i.name and 'embedding' not in i.name)]
				)

			# sum of the mean masked_LM_cost and is_next_cost
			self.train_cost = self.masked_LM_cost + self.is_next_cost


		with tf.name_scope('optimizer'):
			optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9, beta2=0.999, epsilon=1e-6) 
			grads_and_vars = optimizer.compute_gradients(self.train_cost + l2_norm)
			#https://www.tensorflow.org/api_docs/python/tf/clip_by_norm
			clip_grads_and_vars = [(tf.clip_by_norm(gv[0], 1.0), gv[1]) for gv in grads_and_vars]
			#self.minimize = optimizer.apply_gradients(clip_grads_and_vars)
			self.minimize = optimizer.minimize(self.train_cost + l2_norm)



		with tf.name_scope('train_metric'):
			self.masked_LM_pred_argmax = tf.argmax(self.masked_LM_pred, 1, output_type=tf.int32) # [np.sum(boolean_mask)]
			self.masked_LM_correct = tf.reduce_sum(
					tf.cast(tf.equal(self.masked_LM_pred_argmax, self.masked_LM_target), tf.int32)
				)
			self.is_next_pred_argmax = tf.argmax(self.is_next_pred, 1, output_type=tf.int32) # [N]
			self.is_next_correct = tf.reduce_sum(
					tf.cast(tf.equal(self.is_next_pred_argmax, self.is_next_target), tf.int32)
				)

		
		'''
		for i in variables:
			if 'LayerNorm' not in i.name and 'bias' not in i.name  and 'embedding' not in i.name:
				print(i)
		'''

	def embedding_and_PE_SE(self, data, A_B_boundary_length):
		# data: [N, self.encoder_input_length]
		# A_B_boundary_length: [N]

		with tf.device('/cpu:0'):
			embedding = tf.nn.embedding_lookup(
					self.embedding_table, 
					data
				) # [N, self.encoder_input_length, self.embedding_size]
			PE = tf.expand_dims(
					self.position_embedding_table[:self.encoder_input_length, :], # [self.encoder_input_length, self.embedding_size]
					axis=0
				) # [1, self.encoder_input_length, self.embedding_size], will be broadcast
			SE = tf.nn.embedding_lookup(
					self.segment_embedding_table,
					tf.cast(~tf.sequence_mask(A_B_boundary_length, self.encoder_input_length), dtype=tf.int32) # [N, self.encoder_input_length], A:0, B:1
				) # [N, self.encoder_input_length, self.embedding_size] 

		if self.is_embedding_scale is True:
			embedding *= self.embedding_size ** 0.5

		# Add Position embedding and Segement embedding
		embedding += (PE + SE) # [N, self.encoder_input_length, self.embedding_size]

		# make pad mask (value of pad position must be 0)
		embedding_mask = tf.expand_dims(
				tf.cast(tf.not_equal(data, self.pad_idx), dtype=tf.float32), # [N, self.encoder_input_length]
				axis=-1
			) # [N, self.encoder_input_length, 1] 

		# pad masking (set 0 pad position)
		embedding *= embedding_mask # [N, self.encoder_input_length, self.embedding_size]

		# Layer Normalization
		embedding = tf.contrib.layers.layer_norm(embedding,	begin_norm_axis=2)

		# Drop out
		embedding = tf.nn.dropout(embedding, keep_prob=self.keep_prob)
		return embedding, embedding_mask



	def encoder(self, encoder_input_embedding, encoder_input_mask):
		# encoder_input_embedding: [N, self.encoder_input_length, self.embedding_size]
		# encoder_input_mask: [N, self.encoder_input_length, 1]

		encoder_embedding = []

		# make mask
		encoder_self_attention_mask = tf.tile(
				tf.matmul(encoder_input_mask, tf.transpose(encoder_input_mask, [0, 2, 1])), # [N, encoder_input_length, encoder_input_length] 
				[self.multihead_num, 1, 1]
			) # [self.multihead_num*N, encoder_input_length, encoder_input_length]


		for i in range(self.stack): #6
			# Multi-Head Attention
			Multihead_add_norm = self.multi_head_attention_add_norm(
					query=encoder_input_embedding,
					key_value=encoder_input_embedding,
					score_mask=encoder_self_attention_mask,
					output_mask=encoder_input_mask,
					activation=None,
					name='encoder'+str(i)
				) # [N, self.encoder_input_length, self.embedding_size]
		
			# Feed Forward
			encoder_input_embedding = self.dense_add_norm(
					Multihead_add_norm, 
					output_mask=encoder_input_mask, # set 0 bias-added pad position
					activation=self.gelu,
					name='encoder_dense'+str(i)
				) # [N, self.encoder_input_length, self.embedding_size]
			encoder_embedding.append(encoder_input_embedding)

		return encoder_embedding # [N, self.encoder_input_length, self.embedding_size] * stack




	def multi_head_attention_add_norm(self, query, key_value, score_mask=None, output_mask=None, activation=None, name=None):
		# score_mask: [self.multihead_num*N, encoder_input_length, encoder_input_length]

		# Sharing Variables
		with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
			# for문으로 self.multihead_num번 돌릴 필요 없이 embedding_size 만큼 만들고 self.multihead_num등분해서 연산하면 됨.	
			V = tf.layers.dense( # layers dense는 배치(N)별로 동일하게 연산됨.	
					key_value, 
					units=self.embedding_size, 
					activation=activation, 
					use_bias=False,
					name='V'
				) # [N, key_value_sequence_length, self.embedding_size]
			K = tf.layers.dense(
					key_value, 
					units=self.embedding_size, 
					activation=activation, 
					use_bias=False,
					name='K'
				) # [N, key_value_sequence_length, self.embedding_size]
			Q = tf.layers.dense(
					query, 
					units=self.embedding_size, 
					activation=activation, 
					use_bias=False,
					name='Q'
				) # [N, query_sequence_length, self.embedding_size]

			# linear 결과를 self.multihead_num등분하고 연산에 지장을 주지 않도록 batch화 시킴.
			# https://github.com/Kyubyong/transformer 참고.
			# split: [N, key_value_sequence_length, self.embedding_size/self.multihead_num]이 self.multihead_num개 존재 
			V = tf.concat(tf.split(V, self.multihead_num, axis=-1), axis=0) # [self.multihead_num*N, key_value_sequence_length, self.embedding_size/self.multihead_num]
			K = tf.concat(tf.split(K, self.multihead_num, axis=-1), axis=0) # [self.multihead_num*N, key_value_sequence_length, self.embedding_size/self.multihead_num]
			Q = tf.concat(tf.split(Q, self.multihead_num, axis=-1), axis=0) # [self.multihead_num*N, query_sequence_length, self.embedding_size/self.multihead_num]
	
			
			# Q * (K.T) and scaling ,  [self.multihead_num*N, query_sequence_length, key_value_sequence_length]
			score = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(self.embedding_size/self.multihead_num) 

			# masking
			if score_mask is not None:
				score *= score_mask # zero mask
				score += ((score_mask-1) * 1e+9) # -inf mask
				# encoder_self_attention
				# if encoder_input_data: i like </pad>
				# 1 1 0
				# 1 1 0
				# 0 0 0 형태로 마스킹
			
			softmax = tf.nn.softmax(score, dim=2) # [self.multihead_num*N, query_sequence_length, key_value_sequence_length]

			# Attention dropout
			# https://arxiv.org/abs/1706.03762v4 => v4 paper에는 attention dropout 하라고 되어 있음. 
			softmax = tf.nn.dropout(softmax, keep_prob=self.keep_prob)
			
			# Attention weighted sum
			attention = tf.matmul(softmax, V) # [self.multihead_num*N, query_sequence_length, self.embedding_size/self.multihead_num]			

			# split: [N, query_sequence_length, self.embedding_size/self.multihead_num]이 self.multihead_num개 존재
			concat = tf.concat(tf.split(attention, self.multihead_num, axis=0), axis=-1) # [N, query_sequence_length, self.embedding_size]
			
			# Linear
			Multihead = tf.layers.dense(
					concat, 
					units=self.embedding_size, 
					activation=activation,
					use_bias=False,
					name='linear'
				) # [N, query_sequence_length, self.embedding_size]

			if output_mask is not None:
				Multihead *= output_mask

			# Residual Drop Out
			Multihead = tf.nn.dropout(Multihead, keep_prob=self.keep_prob)
			# Add
			Multihead += query
			# Layer Norm			
			Multihead = tf.contrib.layers.layer_norm(Multihead, begin_norm_axis=2)
			
			return Multihead # [N, query_sequence_length, self.embedding_size]



	def dense_add_norm(self, embedding, output_mask=None, activation=None, name=None):
		
		# Sharing Variables
		with tf.variable_scope(name, reuse=tf.AUTO_REUSE):		
			inner_layer = tf.layers.dense(
					embedding, 
					units=4*self.embedding_size, #bert paper 
					activation=activation 
				) # [N, query_sequence_length, 4*self.embedding_size]
			dense = tf.layers.dense(
					inner_layer, 
					units=self.embedding_size, 
					activation=None
				) # [N, query_sequence_length, self.embedding_size]
			
			if output_mask is not None:
				dense *= output_mask # set 0 bias-added pad position

			# Residual Drop Out
			dense = tf.nn.dropout(dense, keep_prob=self.keep_prob)
			# Add
			dense += embedding			
			# Layer Norm
			dense = tf.contrib.layers.layer_norm(dense,	begin_norm_axis=2)
	
		return dense 
	
	
	def gelu(self, x): # https://github.com/google-research/bert/blob/master/modeling.py
		"""Gaussian Error Linear Unit.
		
		This is a smoother version of the RELU.
		Original paper: https://arxiv.org/abs/1606.08415
		Args:
			x: float Tensor to perform activation.
		
		Returns:
			`x` with the GELU activation applied.
		"""
		cdf = 0.5 * (1.0 + tf.tanh(
			(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
		return x * cdf