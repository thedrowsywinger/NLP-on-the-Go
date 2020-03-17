import streamlit as st

#NLP packages
from transformers import BertTokenizer

# Loading the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True)


def text_analyzer(sentences):
	# Print the original sentence.
	print(' Original: ', sentences)
	sentence = sentences

	# Print the sentence split into tokens.
	print('Tokenized: ', tokenizer.tokenize(sentences))
	split = tokenizer.tokenize(sentences)

	# Print the sentence mapped to token ids.
	print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences)))
	token_ids =  tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences))

	return sentence, split, token_ids


import os
import torch
import time
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import (
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    squad_convert_examples_to_features
)

from transformers.data.processors.squad import SquadResult, SquadV2Processor, SquadExample

from transformers.data.metrics.squad_metrics import compute_predictions_logits

use_own_model = True

if use_own_model:
  model_name_or_path = "./model_output" #Enter the path to your custom model here
else:
    model_name_or_path = "ktrapeznikov/albert-xlarge-v2-squad-v2"

output_dir = ""

# Config
n_best_size = 1
max_answer_length = 30
do_lower_case = True
null_score_diff_threshold = 0.0

def to_list(tensor):
    return tensor.detach().cpu().tolist()

# Setup model
config_class, model_class, tokenizer_class = (
    AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer)
config = config_class.from_pretrained(model_name_or_path)
tokenizer = tokenizer_class.from_pretrained(
    model_name_or_path, do_lower_case=True)
model = model_class.from_pretrained(model_name_or_path, config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

processor = SquadV2Processor()

def run_prediction(question_texts, context_text):
    """Setup function to compute predictions"""
    examples = []

    for i, question_text in enumerate(question_texts):
        example = SquadExample(
            qas_id=str(i),
            question_text=question_text,
            context_text=context_text,
            answer_text=None,
            start_position_character=None,
            title="Predict",
            is_impossible=False,
            answers=None,
        )

        examples.append(example)

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=64,
        is_training=False,
        return_dataset="pt",
        threads=1,
    )

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=10)

    all_results = []

    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            example_indices = batch[3]

            outputs = model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs]

                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)

    output_prediction_file = "predictions.json"
    output_nbest_file = "nbest_predictions.json"
    output_null_log_odds_file = "null_predictions.json"

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        False,  # verbose_logging
        True,  # version_2_with_negative
        null_score_diff_threshold,
        tokenizer,
    )

    return predictions

def qa_analyzer(context, question):
	# Run method
    quest = []
    quest.append(question)
    predictions = run_prediction(quest, context)
    return predictions
    # for key in predictions.keys():
    #     print("Answer to the Question:")
	# 	print(predictions[key])

	

#packages


def main():

	st.title("Let's make this a fine evening shall we?")
	st.subheader("Natural Language Processing on the go")

	#QA

	#Tokenization

	if st.checkbox("Exploring BERT tokens"):
		st.subheader("Tokens will be displayed here")
		message = st.text_area("Please enter your text here:", "Type Here")
		if st.button("Analyze"):
			# st.success(message.title())
			st.subheader("Analysis: ")
			sentence, split, token_ids = text_analyzer(message)
			st.subheader("The given sentence: ")
			st.success(sentence)
			st.subheader("The sentence split into tokens: ")
			st.success(split)
			st.subheader("The sentence mapped to token ids: ")
			st.success(token_ids)
	elif st.checkbox("UTTAR-DATA"):
		st.success("Welcome to UTTAR-DATA. Given a context, this model should be able to answer any question(from that context)")
		context = st.text_area("Please provide the context for your question here: ", "Type your context")
		question = st.text_area("Ask a question from the given context: ", "Write your question here: ")

		if st.button("Ask"):
			st.success("Assessing.....")
			predictions = qa_analyzer(context, question)
			for key in predictions.keys():
				print(predictions[key])
				st.success("Question: " + question + "\n " + "Answer: "+ predictions[key])


	# if st.checkbox("UTTAR-DATA"):
        
        

        
            
            
            
                
                
                



if __name__ == "__main__":
	main()	