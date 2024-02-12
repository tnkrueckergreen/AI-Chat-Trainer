AI Chat Trainer
===============

This project uses a pre-trained GPT-2 model to generate chat responses. The model is fine-tuned on a user-provided dataset using Optuna for hyperparameter optimization. The fine-tuned model can then be used to chat with the AI in a console interface.

Requirements
------------

* Python 3.7 or higher
* PyTorch
* Hugging Face Transformers
* Optuna

Installation
------------

1. Clone the repository:
```
git clone https://github.com/tnkrueckergreen/ai-chat-trainer.git
cd ai-chat-trainer
```
2. Create a virtual environment and activate it:
```
python3 -m venv venv
source venv/bin/activate
```
3. Install the required packages:
```
pip install -r requirements.txt
```

Usage
-----

1. Prepare a dataset of chat conversations in a text file, with one conversation per line. Each line should be formatted as follows:
```
User: Hello!
AI: Hi there! How can I help you today?
User: I'm looking for a good restaurant nearby.
AI: Sure, what kind of cuisine are you in the mood for?
...
```
2. Run the main script and enter the name of the GPT-2 model you want to fine-tune (e.g., 'gpt2'):
```
python main.py
```
3. Enter the path to your dataset (e.g., 'dataset.txt'):
```
/path/to/dataset.txt
```
4. The script will fine-tune the model using Optuna for hyperparameter optimization and save the fine-tuned model to a directory named 'gpt2\_finetuned\_trial\_X', where X is the trial number.
5. The script will then start a chat interface with the fine-tuned model. Type your messages and press enter to see the AI's response.

License
-------

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Acknowledgements
----------------

* This project was inspired by the [Hugging Face Transformers](https://github.com/huggingface/transformers) library and the [Optuna](https://github.com/optuna/optuna) hyperparameter optimization framework.
* Thanks to the creators of GPT-2 for making the model available to the public.
* Thanks to the contributors of the open-source projects used in this project.
