# TASI_ERAv2_S21

## Objective

1. Train the 124M parameter GPT2 model on the provided input data such that loss is less than 0.099999
2. Upload the model to HuggingFace Spaces as a Gradio App

## Dataset
Collection of William Shakespeare plays
- tiktoken - gpt2 tokenizer is used for tokenization
- Number of total tokens - 338025

## Steps

### Initial experiment

1. Followed the [video](https://www.youtube.com/watch?v=l8pRSuU81PU) and created the [notebook](https://github.com/sayanbanerjee32/TASI_ERAv2_S21/blob/main/gpt2_experiments.ipynb) for experiment.
2. Observed that even though bfloat16 is supported in collab T4 GPU, it increases the iteration time compared to float16. Therefore used float16 instead and had to use GradientScaler along with it.
3. Used Gradient accumulation to match up to GPT2 training batch size in terms of token number of approximately 0.5 million.


### Model Training
1. Refactored code in [gpt2_training](https://github.com/sayanbanerjee32/TASI_ERAv2_S21/blob/main/gpt2_training_cusom_input.ipynb) notebook and model_gpt2.py
2. Added model code to add the text generator function.
3. Added unscaling of GradientScaler before gradient clipping to fix high values of gradient norm values
4. Added code to save model arguments and weights that would need to be used for inferencing later
5. Pushed the model_gpt2.py and saved artifacts to HuggingFace Model Hub using HuggingFace API from this [notebook](https://github.com/sayanbanerjee32/TASI_ERAv2_S21/blob/main/gpt2_training_cusom_input.ipynb)

### Results
1. model training achieved loss of 0.083899
2. training speed was around 18k tokens / second
3. It took 425 steps to reach there. However, each step includes gradient accumulation, i.e. though weight updates were done once in each step, it iterated on 64 batches in each step. Effectively, we can say it took 27,200 iterations
4. While the training log shows 28 seconds per step, considering 64 batches within each step, we can say it took 430 ms per iteration.

### Gradio App in HuggingFace Spaces
1. Created app.py that can read the model artifacts from HuggingFace Model Hub and launch the app
2. Pushed the model.py, app.py and requirements.txt to HuggingFace spaces using huggingface API from this [notebook](https://github.com/sayanbanerjee32/TASI_ERAv2_S21/blob/main/gpt2_gradio.ipynb)

## The HuggingFace Spaces Gradio App

The app is available [here](https://huggingface.co/spaces/sayanbanerjee32/nanogpt2_text_generator)  

![image](https://github.com/sayanbanerjee32/TASI_ERAv2_S21/assets/11560595/4cfc4213-7b0a-418b-bc5c-6fd828022d16)

The App takes following as input 
1. Seed Text (Prompt) - This is provided as input text to the GPT model, based on which it generates further contents. If no data is provided, the only a space (" ") is provided as input
2. Max tokens to generate - This controls the numbers of tokens it will generate. The default value is 100.
3. Temperature - This accepts values between 0 to 1. Higher value introduces more randomness in the next token generation. Default value is set to 0.7.
4. Select Top N in each step - This is an optional field. If no value is provided (or <= 0), all available tokens are considered for the next token prediction based on SoftMax probability. However, if a number is set then only that many top tokes will be considered for the next token prediction.
