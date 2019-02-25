import bert_data_helper

data_path='./DATA/bpe_out/BPE_kowiki'
voca_path='./DATA/bpe_out/BPE_voca'

c = bert_data_helper.data_helper(data_path, voca_path)
#print(c.bpe2idx)
c.get_dataset(bucket_size=[i*32 for i in range(1, 512//32 +1)], token_length=512)
