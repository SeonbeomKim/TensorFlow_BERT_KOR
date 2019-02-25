import numpy as np
from tqdm import tqdm

class data_helper:
	def __init__(self, data_path, voca_path):
		self.bpe2idx, self.idx2bpe = self.make_voca(voca_path)
		self.data, self.total_token_num = self.data_read(data_path, self.bpe2idx)
		self.zero_sentence_paragraph_index = []
		print(self.total_token_num)

	


	# change name (sampling)
	def get_dataset(self, bucket_size=[], token_length=512, min_first_sentence_length=3, min_second_sentence_length=5):
		bucket_dict = self.init_bucket_dict(bucket_size)

		self.zero_sentence_paragraph_index = []

		len_check = [0]*513
		paragraph_num = len(self.data)
		total_data = 0
		total_token = 0
		mask_0 = 0
		mask_1 = 0
		mask_2 = 0
		#print(paragraph_num)
		count = 0
		for paragraph_index in tqdm(range(paragraph_num), ncols=50):
			first, second = self.get_sentence_from_paragraph(
					paragraph_index, 
					token_length=token_length,
					min_first_sentence_length=min_first_sentence_length,
					min_second_sentence_length=min_second_sentence_length
				)

			for sentence_index in range(len(first)):
				first_sentence = first[sentence_index]
				first_sentence_length = len(first_sentence)

				# is_next
				second_sentence, is_next = self.make_is_next_task(
						second[sentence_index], 
						first_sentence_length,
						current_paragraph_index=paragraph_index,
						total_paragraph_num=paragraph_num,
						min_second_sentence_length=min_second_sentence_length,
						token_length=token_length
					)
				second_sentence_length = len(second_sentence)
				if second_sentence_length < min_second_sentence_length:
					print('error')
					continue
				
				# maksed_LM
				complete_sentence, complete_mask, mask_target, zero, one, two = self.make_mask_task(
						first_sentence, 
						second_sentence
					)
				mask_0 += zero
				mask_1 += one
				mask_2 += two
				total_token += (len(complete_sentence)-3)
				total_data += 1
				len_check[len(complete_sentence)] += 1

				complete_sentence_length = len(complete_sentence)
				total_data += 1
				
				for bucket in bucket_size:
					if complete_sentence_length <= bucket:
						'''
						complete_sentence = np.pad(
								complete_sentence, 
								(0, bucket-complete_sentence_length), 
								'constant', 
								constant_values=(0, self.bpe2idx['</PAD>'])
							)
						complete_mask = np.pad(
								complete_mask, 
								(0, bucket-complete_sentence_length), 
								'constant', 
								constant_values=(0, False)
							)
						bucket_dict[bucket]['dataset'].append(complete_sentence)
						bucket_dict[bucket]['is_next_target'].append(is_next)
						bucket_dict[bucket]['A_B_boundary'].append(first_sentence_length+2) # A position is CLS || first_sentence || SEP
						bucket_dict[bucket]['boolean_mask'].append(complete_mask)
						bucket_dict[bucket]['masked_LM_target'].append(mask_target)
						total_data += 1
						'''
						bucket_dict[bucket]['num'] += 1
						break




		for index, i in enumerate(len_check):
			print(index, i)
		print('total_data:', total_data)
		print('total_token:', total_token, '\tmask_token(12%):', mask_0, '\tchange_token(1.5%):', mask_1, '\tkeep_token(1.5%):', mask_2)
		print('12%_of_total_token:', total_token*0.12, '\t1.5%_of_total_token:', total_token*0.015)
		for i in bucket_dict:
			print(i, bucket_dict[i]['num'])

		import matplotlib.pyplot as plt
		plt.plot(range(0, 513), len_check)
		plt.show()
	

	def init_bucket_dict(self, bucket_size):
		bucket_dict = {}
		for bucket in bucket_size:
			bucket_dict[bucket] = {
					'dataset':[], 
					'is_next_target':[], 
					'A_B_boundary':[],
					'boolean_mask':[],
					'masked_LM_target':[],
					'num':0
				}		
		return bucket_dict


	def list_flatten(self, data):
		flatten = []
		for i in data:
			flatten += i
		return flatten

	def get_sentence_from_paragraph(self, paragraph_index, token_length=512, min_first_sentence_length=3, min_second_sentence_length=5):
		first = []
		second = []

		paragraph = self.data[paragraph_index]
		paragraph_length = len(paragraph)
		
		cumulative_length = np.zeros(paragraph_length) # 각 index별로 누적된 길이 저장 # [0]: 0~0, [1]: 0~1, [K]: 0~K
		cumulative_length[0] = len(paragraph[0])
		for sentence_index in range(1, paragraph_length):
			current_sentence_length = len(paragraph[sentence_index])
			cumulative_length[sentence_index] = current_sentence_length + cumulative_length[sentence_index-1]

			use_one_sentence = np.random.randint(1, 4) # 66% 확률로 한문장만 사용
			if use_one_sentence <= 2 or sentence_index == 1:
				first_sentence = paragraph[sentence_index-1]
				first_sentence_length = len(first_sentence)
				if first_sentence_length + 3 + min_second_sentence_length > token_length or first_sentence_length < min_first_sentence_length:
					continue

				second_sentence = paragraph[sentence_index]
				second_sentence_length = len(second_sentence)
				if second_sentence_length < min_second_sentence_length:
					continue

				if first_sentence_length + second_sentence_length + 3 > token_length:
					# 5% 정도는 second_sentence를 slice해서 씀.
					is_second_sentence_slice = np.random.randint(1, 101)
					if is_second_sentence_slice <= 5: # 5% 
						second_sentence = second_sentence[:token_length-3-first_sentence_length]
					else:
						continue


			else: # 여러문장을 하나의 문장으로 취급
				if cumulative_length[sentence_index] + 3 <= token_length: # 누적 길이가 조건을 만족하는경우
					first_sentence_last_index = np.random.randint(0, sentence_index) 
					first_sentence = self.list_flatten(paragraph[:first_sentence_last_index+1]) 
					first_sentence_length = len(first_sentence)
					if first_sentence_length + 3 + min_second_sentence_length > token_length or first_sentence_length < min_first_sentence_length:
						continue

					second_sentence = self.list_flatten(paragraph[first_sentence_last_index+1:sentence_index+1])
					second_sentence_length = len(second_sentence)
					if second_sentence_length < min_second_sentence_length:
						continue


				else: # 누적 길이가 조건보다 긴 경우: 잘라야함.
					# [0, sentence_index-2] => cumulative_length[sentence_index] - cumulative_length[sentence_index-2]: len[sentence_index-1]+len[sentence_index]
					possible_index = []
					for cum_index in range(sentence_index-1): 
						length_check = cumulative_length[sentence_index] - cumulative_length[cum_index]
						if length_check + 3 <= token_length:
							possible_index.append(cum_index+1) # cum_index+1 부터 사용해야 위 조건을 만족한다는 뜻.
					possible_index_length = len(possible_index)
		
					# 만족하는것이 없는 경우 5%의 확률로 한문장씩 쓰고, second sentence는 slice해서 씀.
					if possible_index_length == 0:
						is_second_sentence_slice = np.random.randint(1, 101)
						if is_second_sentence_slice <= 5: # 5% 
							first_sentence = paragraph[sentence_index-1]
							first_sentence_length = len(first_sentence)
							if first_sentence_length + 3 + min_second_sentence_length > token_length or first_sentence_length < min_first_sentence_length:
								continue

							second_sentence = paragraph[sentence_index]
							second_sentence_length = len(second_sentence)
							if second_sentence_length < min_second_sentence_length:
								continue

							second_sentence = second_sentence[:token_length-3-first_sentence_length]
						
						else:
							continue


					else:
						first_sentence_start_index = np.random.choice(possible_index, 1)[0]
						first_sentence_last_index = np.random.randint(first_sentence_start_index, sentence_index)
						first_sentence = self.list_flatten(paragraph[first_sentence_start_index : first_sentence_last_index+1]) 
						first_sentence_length = len(first_sentence)
						if first_sentence_length + 3 + min_second_sentence_length > token_length or first_sentence_length < min_first_sentence_length:
							continue

						second_sentence = self.list_flatten(paragraph[first_sentence_last_index+1 : sentence_index+1])
						second_sentence_length = len(second_sentence)
						if second_sentence_length < min_second_sentence_length:
							continue


			if len(second_sentence) < min_second_sentence_length:
				print('second error')
			if len(first_sentence) < min_first_sentence_length:
				print('first error')

			# 이 라인에 온다는 것은 모든 조건을 만족했다는 것.
			first.append(first_sentence)
			second.append(second_sentence)
	
		return first, second
		'''
		for first_sentence, second_sentence in zip(first, second):
			print(len(first_sentence), len(second_sentence), len(first_sentence)+len(second_sentence)+3)
			# 문장들 다 1자로 펴고, 길이 합 + 3 한게 512보다 큰 상황 발생하는지 확인하자.						
			if len(first_sentence) == 0:
				print('first_sentence_length is 0')
			if len(second_sentence) == 0:
				print('second_sentence_length is 0')
			if len(first_sentence) + len(second_sentence) + 3 > 512:
				print('longer than 512', 'first', len(first_sentence), 'second', len(second_sentence), 'first+second+3', len(first_sentence) + len(second_sentence) + 3)

		'''

	def make_is_next_task(self, second_sentence, first_sentence_length, current_paragraph_index, 
				total_paragraph_num, min_second_sentence_length=5, token_length=512):

		# for is_next sentence prediction
		is_next = np.random.randint(0, 2) # 0: not next, 1: next
		if is_next == 0: # not next
			#print(token_length)
			while True:
				other_paragraph_index = self.exclusive_randint(
						min_val=0, 
						max_val=total_paragraph_num, 
						exclusive_vals=[current_paragraph_index]+self.zero_sentence_paragraph_index
					)
				other_first, other_second = self.get_sentence_from_paragraph(
						other_paragraph_index, 
						token_length=token_length,
						min_first_sentence_length=min_second_sentence_length,
						min_second_sentence_length=min_second_sentence_length
					)
				if len(other_first):
					break
				else:
					self.zero_sentence_paragraph_index.append(other_paragraph_index)
					#count+=1
					#print('hi', count)

			other_sentences = other_first + other_second
			possible_other_sentence_index = []
			for other_sentence_index, other_sentence in enumerate(other_sentences):
				if len(other_sentence) + first_sentence_length + 3 <= token_length: 
					possible_other_sentence_index.append(other_sentence_index)
		
			if len(possible_other_sentence_index): #길이를 만족하는것이 있는 경우
				second_sentence = other_sentences[np.random.choice(possible_other_sentence_index, 1)[0]]
		
			else: # 길이를 만족하는 것이 없으면 아무 문장이나 선택하고 원본 second_sentence 길이로 slice.
				random_other_sentence_index = np.random.randint(0, len(other_sentences))
				second_sentence = other_sentences[random_other_sentence_index][:len(second_sentence)] #[:token_length-3-first_sentence_length]

		return second_sentence, is_next




	def make_mask_task(self, first_sentence, second_sentence):
		mask_0 = 0
		mask_1 = 0
		mask_2 = 0
		
		# for masked language model task
		concat_sentences_for_mask = np.array(first_sentence+second_sentence, dtype=np.int32)
		concat_sentences_for_mask_length = len(concat_sentences_for_mask) 
		
		mask_position = np.random.randint(1, 101, size=len(concat_sentences_for_mask)) <= 15 # 1~15 즉 15%는 마스킹.
		mask_method = np.random.randint(1, 11, size=sum(mask_position)) # 1~8: mask token, 9: change random token, 10: keep token

		mask_target = concat_sentences_for_mask[mask_position]
		new_token_of_mask_position = np.zeros_like(mask_target)

		for mask_method_index, mask in enumerate(mask_method):
			if mask <= 8: # 1~8: mask token
				new_token_of_mask_position[mask_method_index] = self.bpe2idx['</MASK>']
				mask_0 += 1
			elif mask == 9: # 9: change random token
				new_token_of_mask_position[mask_method_index] = self.exclusive_randint(
						min_val=0, 
						max_val=len(self.bpe2idx), 
						exclusive_vals=[mask_target[mask_method_index], self.bpe2idx['</PAD>'], self.bpe2idx['</UNK>'], self.bpe2idx['</CLS>'], self.bpe2idx['</SEP>'], self.bpe2idx['</MASK>']]
					)
				mask_1 += 1
			else: # 10: keep token
				new_token_of_mask_position[mask_method_index] = mask_target[mask_method_index]
				mask_2 += 1

		# token update
		concat_sentences_for_mask[mask_position] = new_token_of_mask_position

		# make dataset
		first_sentence_length = len(first_sentence)
		masked_first_sentence = concat_sentences_for_mask[:first_sentence_length]
		mask_of_first_sentence = mask_position[:first_sentence_length]
		masked_second_sentence = concat_sentences_for_mask[first_sentence_length:]
		mask_of_second_sentence = mask_position[first_sentence_length:]

		complete_sentence = np.concatenate(
				([self.bpe2idx['</CLS>']], masked_first_sentence, [self.bpe2idx['</SEP>']], masked_second_sentence, [self.bpe2idx['</SEP>']])
			)
		complete_mask = np.concatenate(
				([False], mask_of_first_sentence, [False], mask_of_second_sentence, [False])
			)

		return complete_sentence, complete_mask, mask_target, mask_0, mask_1, mask_2















	def get_batch_data(self, token_length=512):
		'''TODO
		data는 같은 paragraph끼리 같은 라인에 있음.
		이걸 512-cls-sep-sep 즉 509개씩 끊고, 1번 문장, 2번 문장 비율을 10 주사위 던져서 나오는 비율로 끊자.(1:9, 2:8, 9:1)
		cls || 1sentence || sep || 2sentence || sep || pad  => 512가 되도록 처리.
		이렇게 만들 때, 2sentence는 50% 비율로 다른 i의 문장 중에서 고르면 되는데 
			=> 정확히 isnext True, isnext False 비율이 반반 되도록 할 방법 생각해야함.
			=> 또한 cls, sep, sep은 masking 되면 안되니까 concat하기 전에 mask도 처리하고, mask한 위치도 기록해야함.
	
		mask 비율은 전체의 15% 
			=> 이 중 80%(0.15*0.8 = 0.12, 즉 전체의 12%)는 mask토큰으로 변환. :  0 method
			=> 이 중 10%(0.15*0.1 = 0.015, 즉 전체의 1.5%)는 mask가 아닌 다른 단어로 변환 : 1 method
			=> 이 중 10%(0.15*0.1 = 0.015, 즉 전체의 1.5%)는 단어 그대로 두고 자기 자신을 맞추도록 함. : 2 method
	

		중요
		일단 데이터셋 first second 다 만들고(전체 이터레이션 완료)
		그 후에 50%의 second만 골라서 서로 셔플. => 이 때, label 생성도 해야함.
			=> 0 ~ K list 만들고 
		그 후에 masking 하자. masking 처리할 땐 모델에서 마스킹 부분만 집중할 수 있도록 loss mask도 만들어둬야함. label도 생성.
		중요


		bert github을 보면 90000 step은 128길이로 하고(학습 빠르게), 10000step은 512길이로 했다고 함.
		'''
		#print(self.total_token_num) # 86,826,280  이것의 15%는 13,023,942
		mask_0 = 0
		mask_1 = 0
		mask_2 = 0
		total_token = 0
		
		dataset = [] # token_length 길이로 끊어진 데이터들의 집합.
		masked_LM_target = [] # 이건 각각 개수가 다를 수 있으므로 배치처리할때는 다 extend해서 쓰면됨.
		is_next_target = [] # 1이면 true, 0이면 false
		boolean_mask = [] # 추후에 mask 된 부분만 prediction 할 수 있도록 하는 boolean mask
		A_B_boundary = [] # A:cls || first_sentence || sep , B: second_sentence || sep, 즉 first_sentence_length+2가 들어감.

		paragraph_num = len(self.data)
		#print(paragraph_num)

		for index in tqdm(range(paragraph_num), ncols=50):
			paragraph = self.data[index]

			for i in range(int(np.ceil(len(paragraph)/(token_length-3)))): # cls, sep, sep 3개 제외한 길이로.
				sentences = paragraph[(token_length-3) * i : (token_length-3) * (i + 1)]
				sentences_length = len(sentences)
				if sentences_length < 20:
					break


				# get first sentence
				first_sentence_ratio = np.random.randint(1, 10) / 10 # 0.1 ~ 0.9 : 10% ~ 90%
				first_sentence_length = int(sentences_length * first_sentence_ratio)
				A_B_boundary.append(first_sentence_length+2) # cls a b c sep , 길이가 3이면 +1 한 위치는 boundary index, +2는 길이.
				second_sentence_length = sentences_length - first_sentence_length
				first_sentence = sentences[:first_sentence_length]



				# get second sentence consider is_next sentence prediction
				is_next = np.random.randint(0, 2) # 0: not next, 1: next
				is_next_target.append(is_next)

				if is_next == 0: # not next
					other_paragraph_index = self.exclusive_randint(min_val=0, max_val=paragraph_num, exclusive_vals=[index])
					other_paragraph = self.data[other_paragraph_index]
					other_paragraph_length = len(other_paragraph)
					
					if other_paragraph_length <= second_sentence_length:
						second_sentence = other_paragraph[:]
						second_sentence_length = other_paragraph_length
					else: # other_paragraph_length > second_sentence_length
						'''	ex)
						other_paragraph_length가 5, second_sentence_length가 3이면
						other_paragraph의 0~2 index중에 골라서 3길이만큼 꺼내면 됨.
							=> other_paragraph: [0,1,2,3,4]
						'''	
						read_index = np.random.randint(0, other_paragraph_length-second_sentence_length+1)
						second_sentence = other_paragraph[read_index:read_index+second_sentence_length]
						# second_sentence_length 는 동일함.
				else: #is next
					second_sentence = sentences[first_sentence_length:]
		


				# masked language model task
				concat_sentences_for_mask = np.array(first_sentence+second_sentence, dtype=np.int32)
				total_token += len(concat_sentences_for_mask)

				mask_position = np.random.randint(1, 101, size=len(concat_sentences_for_mask)) <= 15 # 1~15 즉 15%는 마스킹.
				mask_method = np.random.randint(1, 11, size=sum(mask_position)) # 1~8: mask token, 9: change random token, 10: keep token

				mask_target = concat_sentences_for_mask[mask_position]
				masked_LM_target.append(mask_target)
				new_token_of_mask_position = np.zeros_like(mask_target)

				for enum, mask in enumerate(mask_method):
					if mask <= 8: # 1~8: mask token
						new_token_of_mask_position[enum] = self.bpe2idx['</MASK>']
						mask_0 += 1
					elif mask == 9: # 9: change random token
						new_token_of_mask_position[enum] = self.exclusive_randint(
								min_val=0, 
								max_val=len(self.bpe2idx), 
								exclusive_vals=[mask_target[enum], self.bpe2idx['</PAD>'], self.bpe2idx['</UNK>'], self.bpe2idx['</CLS>'], self.bpe2idx['</SEP>'], self.bpe2idx['</MASK>']]
							)
						mask_1 += 1
					else: # 10: keep token
						new_token_of_mask_position[enum] = mask_target[enum]
						mask_2 += 1


				# token update
				concat_sentences_for_mask[mask_position] = new_token_of_mask_position
	

				# make dataset
				masked_first_sentence = concat_sentences_for_mask[:first_sentence_length]
				mask_of_first_sentence = mask_position[:first_sentence_length]
				masked_second_sentence = concat_sentences_for_mask[first_sentence_length:]
				mask_of_second_sentence = mask_position[first_sentence_length:]

				complete_sentence = np.concatenate(
						([self.bpe2idx['</CLS>']], masked_first_sentence, [self.bpe2idx['</SEP>']], masked_second_sentence, [self.bpe2idx['</SEP>']])
					)
				complete_sentence = np.pad(complete_sentence, (0, token_length-len(complete_sentence)), 'constant', constant_values=(0, self.bpe2idx['</PAD>']))
				dataset.append(complete_sentence)

				complete_mask = np.concatenate(
						([False], mask_of_first_sentence, [False], mask_of_second_sentence, [False])
					)
				complete_mask = np.pad(complete_mask, (0, token_length-len(complete_mask)), 'constant', constant_values=(0, False))
				boolean_mask.append(complete_mask)

			#if len(dataset) >= 5000:
			#	print('break', len(dataset))
			#	break

		print('total_sentence:', len(is_next_target), '\tis_next_true:', np.sum(is_next_target))
		print('total_token:', total_token, '\tmask_token(12%):', mask_0, '\tchange_token(1.5%):', mask_1, '\tkeep_token(1.5%):', mask_2)
		print('12%_of_total_token:', total_token*0.12, '\t1.5%_of_total_token:', total_token*0.015)
		return np.array(dataset, dtype=np.int32), np.array(boolean_mask, dtype=np.bool), masked_LM_target, np.array(is_next_target, dtype=np.int32), np.array(A_B_boundary, dtype=np.int32)


	def exclusive_randint(self, min_val, max_val, exclusive_vals):
		while True:
			randint = np.random.randint(min_val, max_val)
			if randint not in exclusive_vals:
				break
		return randint


	def make_voca(self, voca_path):
		bpe2idx = {'</PAD>':0, '</UNK>':1, '</CLS>':2, '</SEP>':3, '</MASK>':4}
		idx2bpe = ['</PAD>', '</UNK>', '</CLS>', '</SEP>', '</MASK>']
		idx = 5

		with open(voca_path, 'r', encoding='utf-8') as f:
			for bpe_voca in f:
				bpe_voca = bpe_voca.strip()
				if bpe_voca:
					bpe_voca = bpe_voca.split()[0] # 1은 freq
					bpe2idx[bpe_voca] = idx
					idx += 1
					idx2bpe.append(bpe_voca)

		return bpe2idx, idx2bpe

	def data_read(self, data_path, bpe2idx):
		total_token_num = 0

		data = []
		paragraph = []
		with open(data_path, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.strip()
				if line:
					idx = []
					tokens = line.split()
					for token in tokens:
						total_token_num += 1
						if token in bpe2idx:
							idx.append(bpe2idx[token])
						else:
							idx.append(bpe2idx['</UNK>'])
					paragraph.append(idx)

				else: # next line is new paragraph
					data.append(paragraph)
					paragraph = []

		return data, total_token_num
