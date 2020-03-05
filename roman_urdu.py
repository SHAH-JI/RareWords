import re
# from gutenberg import acquire
# from gutenberg import cleanup
import numpy as np
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry

@registry.register_problem
class RomanUrdu(text_problems.Text2TextProblem):
	"""Predict next line of poetry from the last line. From Gutenberg texts."""
	@property
	def approx_vocab_size(self):
		return 20000

	@property
	def is_generate_per_split(self):
		# generate_data will shard the data into TRAIN and EVAL for us.
		return False

	@property
	def dataset_splits(self):
		"""Splits of data to produce and number of output shards for each."""
		# 10% evaluation data
		return [{
				"split": problem.DatasetSplit.TRAIN,
				"shards": 90,
		}, {
				"split": problem.DatasetSplit.EVAL,
				"shards": 10,
		}]

	@registry.register_hparams
	def transformer_roman2Urdu():
		hparams = transformer.transformer_base()
		hparams.num_hidden_layers = 4
		hparams.hidden_size = 128
		hparams.filter_size = 512
		hparams.num_heads =4
		hparams.attention_dropout = 0.5
		hparams.layer_prepostprocess_dropout = 0.5
		hparams.learning_rate = 0.01
		return hparams

	def generate_samples(self, data_dir, tmp_dir, dataset_split):
		del data_dir
		del tmp_dir
		del dataset_split
		
		roman_sentences = open(r'.\data\roman.txt', 		'r', encoding='utf-8')
		urdu_sentences  = open(r'.\data\urdu_fixed.txt', 	'r', encoding='utf-8')
		for source, target in zip(roman_sentences, urdu_sentences):
			yield {
				"inputs": source,
				"targets": target,
			}
			
		roman_sentences.close()
		urdu_sentences.close()
		