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
	

#packages


def main():

	st.title("Let's make this a fine evening shall we?")
	st.subheader("Natural Language Processing on the go")

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



if __name__ == "__main__":
	main()	