import os
import torch
import optuna
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
from transformers.trainer_utils import IntervalStrategy
from transformers.optimization import Adafactor, AdamW

# Define a callback for Optuna
class OptunaCallback(TrainerCallback):
    """A custom callback for Optuna that reports the evaluation loss to the trial."""

    def __init__(self, trial):
        self.trial = trial

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """Report the evaluation loss to the Optuna trial."""
        self.trial.report(metrics["eval_loss"], step=state.global_step)
        if state.is_hyper_param_search and self.trial.should_prune():
            raise optuna.TrialPruneError

# Define the main class for the AI Chat Trainer
class AIChatTrainer:
    """A class for fine-tuning a GPT-2 model and using it to generate chat responses."""

    def __init__(self, model_name: str, dataset_path: str):
        """Initialize the AI Chat Trainer with the given model name and dataset path."""
        print("\n🚀 Initializing the AI Chat Trainer...")
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self.dataset, self.data_collator = self._prepare_dataset_and_collator()

    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer."""
        print("\n📚 Loading the model and tokenizer...")
        try:
            model = GPT2LMHeadModel.from_pretrained(self.model_name)
            tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            tokenizer.pad_token = tokenizer.eos_token
            print("✅ Model and tokenizer loaded successfully!")
            return model, tokenizer
        except Exception as e:
            print(f"❌ Error loading model and tokenizer: {e}")
            exit(1)

    def _prepare_dataset_and_collator(self):
        """Prepare the dataset and data collator."""
        print("\n🔍 Preparing the dataset and data collator...")
        try:
            dataset = TextDataset(tokenizer=self.tokenizer, file_path=self.dataset_path, block_size=128)
            data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
            print("✅ Dataset and data collator prepared successfully!")
            return dataset, data_collator
        except Exception as e:
            print(f"❌ Error preparing dataset and data collator: {e}")
            exit(1)

    def _fine_tune_model(self, training_args, trial):
        """Fine-tune the model with the given training arguments and Optuna trial."""
        print("\n🎯 Fine-tuning the model...")
        train_size = int(0.9 * len(self.dataset))
        eval_size = len(self.dataset) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(self.dataset, [train_size, eval_size])

        try:
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=self.data_collator,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                callbacks=[OptunaCallback(trial), EarlyStoppingCallback(early_stopping_patience=3)],
            )
            self.trainer.train()
            print("✅ Model fine-tuned successfully!")
        except Exception as e:
            print(f"❌ Error fine-tuning model: {e}")
            exit(1)

    def objective(self, trial):
        """Define the objective for the Optuna trial."""
        print(f"\n🔍 Starting trial {trial.number}...")
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
        num_train_epochs = trial.suggest_int("num_train_epochs", 1, 5)
        warmup_steps = trial.suggest_int("warmup_steps", 0, 500)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)

        training_args = TrainingArguments(
            output_dir=f"./gpt2_finetuned_trial_{trial.number}",
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=2,
            save_steps=10_000,
            save_total_limit=2,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            gradient_accumulation_steps=16,
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=500,
            load_best_model_at_end=True,
            logging_steps=100,
            logging_first_step=True,
            fp16=False,
        )
        self._fine_tune_model(training_args, trial)
        return self.trainer.evaluate().get("eval_loss")

    def chat_with_model(self):
        """Start a chat with the fine-tuned model."""
        print("\n💬 Starting the chat with the AI...")
        while True:
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            input_text = f"User: {user_input}\nAI:"
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=5,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            response = self.tokenizer.decode(output[0], skip_special_tokens=True).replace("AI:", "")
            print(f"\nAI: {response.strip()}")

def main():
    """The main function to run the script."""
    print("\n🎉 Welcome to the AI Chat Trainer!")
    print("This program will help you fine-tune a GPT-2 model to create a chatbot.")
    print("Let's get started!")

    model_name = input("\n🔤 Please enter the name of the GPT-2 model you want to fine-tune (e.g., 'gpt2'): ")
    dataset_path = input("\n📁 Please enter the path to your dataset (e.g., 'dataset.txt'): ")

    ai_chat_trainer = AIChatTrainer(model_name, dataset_path)
    
    print("\n🔍 Starting hyperparameter optimization...")
    storage_name = "sqlite:///gpt2_hyperopt.db"
    study = optuna.create_study(direction="minimize", storage=storage_name, load_if_exists=True, sampler=optuna.samplers.TPESampler())
    study.optimize(ai_chat_trainer.objective, n_trials=10)
    
    best_trial = study.best_trial
    print(f"\n🏆 Best Trial: score {best_trial.value}, params {best_trial.params}")

    model_save_path = f"./gpt2_finetuned_trial_{best_trial.number}"
    ai_chat_trainer.model.save_pretrained(model_save_path)
    ai_chat_trainer.tokenizer.save_pretrained(model_save_path)

    print("\n🎉 The fine-tuning process is complete! Your chatbot is ready to chat.")
    print("To start a chat, just type your message. To end the chat, type 'exit' or 'quit'.")
    ai_chat_trainer.chat_with_model()

if __name__ == "__main__":
    main()
