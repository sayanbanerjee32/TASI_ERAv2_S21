# TASI_ERAv2_S21

## Objective

1. Train a mini GPT model following the instruction from Andrej Karpathy in this [video](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2s)
2. Upload the model to HuggingFace Apps

## Dataset - tiny shakespeare, character-level
Tiny shakespeare, of the good old char-rnn fame :) Treated on character-level.

- Tokenization performed on Character level
- Vocab size 65. Following are the unique tokens
    - `!$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz`
- Number of total tokens - 1115394
    - trained on 1,003,854 tokens (90%)
    - validation is performed on 111,540 tokens (10%)
 
## Steps

### Initial experiment

1. Followed the [video](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2s) and created the [notebook](https://github.com/sayanbanerjee32/TASI_ERAv2_S19/blob/main/gpt_dev.ipynb) for experiment.


### Model Training
1. Refactored code in [gpt2_training](https://github.com/sayanbanerjee32/TASI_ERAv2_S21/blob/main/gpt2_training.ipynb) notebok and model_gpt2.py
2. Added model code to add the text generator function.
3. Added unscaling of GradientScaler before gradient clipping to fix high values of gradient norm values
4. Added code to save model arguments and weights that would need to be used for inferencing later
5. Pushed the model_gpt2.py and saved artifacts to HuggingFace Model Hub using huggingface API from this [notebook](https://github.com/sayanbanerjee32/TASI_ERAv2_S21/blob/main/gpt2_training.ipynb)

### Results
1. model training achieved loss of 0.083899
2. training speed was around 18k tokens / second
3. It took 425 steps to reach there. However, each step includes gradient accumulation, i.e. though weight updated were done once in each step, it iterated on 64 batches in each step. Effectively, we can say it took 27,200 interations
4. While training log shows 28 seconds per step, considering 64 batches within each step, we can say it took 430 ms per iteration.

### Gradio App in HuggingFace Spaces
1. Created app.py that can read the model artefacts from HuggingFace Model Hub and launch the app
2. Pushed the model.py, app.py and requirements.txt to HuggingFace spaces using huggingface API from this [notebook](https://github.com/sayanbanerjee32/TASI_ERAv2_S21/blob/main/gpt2_gradio.ipynb)

## The Huggingface Spaces Gradio App

The app is available [here](https://huggingface.co/spaces/sayanbanerjee32/nanogpt2_text_generator)  

![image](https://github.com/sayanbanerjee32/TASI_ERAv2_S21/assets/11560595/3c796cc1-3e1a-4525-8c6d-23effaf8ac62)  

The App takes following as input 
1. Seed Text (Prompt) - This provided as input text to the GPT model, based on which it generates further contents. If no data is provided, the only a space (" ") is provided as input
2. Max tokens to generate - This controls the numbers of tokens it will generate. The default value is 100.
3. Temperature - This accepts value between 0 to 1. Higher value introduces more randomness in the next token generation. Default value is set to 0.7.
4. Select Top N in each step - This is optional field. If no value is provided (or <= 0), all available tokens are considered for next token prediction based on SoftMax probability. However, if a number is set then only that many top characters will be considered for next token prediction.
