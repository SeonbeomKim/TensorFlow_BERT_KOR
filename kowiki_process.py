import nltk
import argparse

parser = argparse.ArgumentParser(description='file path')
parser.add_argument('-wiki_path', required=True)
parser.add_argument('-out_path', required=True)
args = parser.parse_args()
wiki_path = args.wiki_path
out_path = args.out_path

class kowiki_process:
	def __init__(self):
		try:
			nltk.tokenize.sent_tokenize('test')
		except:
			nltk.download('punkt')

	
	def find_consecutive_false_index(self, word_length, consecutive_short_line_threshold):
		# F(K 단어 미만을 보유한 문장)가 consecutive_short_line_threshold 이상인 인덱스만 체크 
		# 일단 6단어보다 작은 문장이 5줄이상 연속으로 나오면 제외
		# T: 6단어이상, F 6단어 미만 
		# TTTTTFFFFFFFFFF 이러면 F줄만 제외.
		# FFFFFFFFFFTTTTT 이러면 F줄만 제외.
		# TTFFFFFFFFFFTTT 이러면 전체 제외.
		# 
		# 3줄 이하로 구성되어있으면 제외.
	
		is_previous_false = False
		indices = []
		for i in range(len(word_length)):
			if word_length[i] == False and is_previous_false == False:
				is_previous_false = True
				start = i		
			elif word_length[i] == True and is_previous_false == True:
				is_delete = (i-1 - start + 1 >= consecutive_short_line_threshold)
				if is_delete:
					indices.append((start, i-1))
				is_previous_false = False
	
		if is_previous_false == True:
			i = len(word_length)
			is_delete = (i-1 - start + 1 >= consecutive_short_line_threshold)
			if is_delete:
				indices.append((start, i-1))
		return indices


	def process(self, wiki_path, out_path):
		short_line_threshold = 11 # short_line_threshold 이상은 T, 미만은 F
		consecutive_short_line_threshold = 5 # consecutive_short_line_threshold 이상이면 지움.
		line_threshold = 3

		with open(wiki_path, 'r', encoding='utf-8') as f, open(out_path, 'w', encoding='utf-8') as o:

			for i, line in enumerate(f):
				line = line.strip()
				if line: # 공백이 아닌 경우
					if line[:8] == '<doc id=':
						is_previous_doc = True
						paragraph = []
						word_length = []

					elif line == '</doc>':
						paragraph_len = len(paragraph)
						if paragraph_len < line_threshold:
							continue

						# 짧은 문장이 여러번 나오는지 체크
						else:
							indices = self.find_consecutive_false_index(word_length, consecutive_short_line_threshold)
							# indices: 짧은 문장이 연속적으로 threshold 이상 나타난 경우의 index 집합.
							indices_len = len(indices)
							
							if indices_len == 0:
								start, end = 0, paragraph_len-1

							elif indices_len == 1:
								if indices[0][0] == 0:
									start, end = indices[0][1]+1, paragraph_len-1
								elif indices[0][1] == paragraph_len-1:
									start, end = 0, indices[0][0]-1
								else:
									continue		

							elif indices_len == 2:
								if indices[0][0] == 0 and indices[1][1] == paragraph_len-1: 
									start, end = indices[0][1]+1, indices[1][0]-1
								else:
									continue

							else: # >=3
								continue

							# continue를 만나지 않고 이곳에 오면 start, end에 값이 있음.
							# 이제 연속된 짧은 문장들을 제외하고 새로 씀.
							new_len = end-start+1 
							if new_len >= line_threshold: 
								paragraph = ' '.join(paragraph[start:end+1])
								sentence_tokenize = nltk.tokenize.sent_tokenize(paragraph)
								for sentence in sentence_tokenize:
									o.write(sentence+'\n')
								o.write('\n')
								#o.write(new_line)
	
					else: # paragraph part
						if is_previous_doc == True: # 제목 제외.
							is_previous_doc = False
						else:
							paragraph.append(line)
							word_length.append(len(line.split())>=short_line_threshold)


wp = kowiki_process()
wp.process(wiki_path, out_path)
#wp.find_consecutive_false_index([True, True, False, False, True, False, False, True, True, False, False, False], 3)
#wp.find_consecutive_false_index([False, True, False, False, True, False, False, True, True, False, False, False], 3)
