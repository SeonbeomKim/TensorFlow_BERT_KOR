import numpy as np
from tqdm import tqdm

class data_helper:
	def __init__(self, data_path, voca_path):
		self.bpe2idx, self.idx2bpe = self.make_voca(voca_path)
		self.data, self.total_token_num = self.data_read(data_path, self.bpe2idx)


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
				A_B_boundary.append(first_sentence_length+2)
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
		data = []
		total_token_num = 0
		with open(data_path, 'r', encoding='utf-8') as f:
			for paragraph in f:
				paragraph = paragraph.strip()
				if paragraph:
					idx = []
					split_token = paragraph.split()
					if len(split_token) < 20:
						continue

					for token in split_token:
						total_token_num += 1
						if token in bpe2idx:
							idx.append(bpe2idx[token])
						else:
							idx.append(bpe2idx['</UNK>'])
					data.append(idx)

		return data, total_token_num
