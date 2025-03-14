import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LMWrapper(AutoModelForCausalLM):
    def __init__(self, model_name):
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)

        super().__init__(self.base_model.config)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, model_name, *args, **kwargs):
        return cls(model_name)


class CoHT(AutoModelForCausalLM):
    def __init__(self, model_name):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.think_token_id = self.tokenizer.convert_tokens_to_ids("<think>")
        self.end_think_token_id = self.tokenizer.convert_tokens_to_ids("</think>")

    def forward(self, inputs_embeds=None, return_hidden_states=False, **kwargs):
        """
        Forward pass: supports both discrete token input (`input_ids`) and soft token input (`inputs_embeds`).
        """
        outputs = self.base_model(
            inputs_embeds=inputs_embeds, output_hidden_states=True, **kwargs
        )
        logits = outputs.logits
        if return_hidden_states:
            return (
                logits,
                outputs.hidden_states[-2],
            )  # Extract soft embeddings from second-last layer
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_length=100, temperature=1.0):
        """Custom generation with soft token decoding."""
        output_ids = input_ids.clone()
        hidden_states_buffer = []
        in_think_mode = False
        soft_embeds = None  # Stores soft token embeddings when in think mode

        for _ in range(max_length):
            if in_think_mode:
                # Use soft embeddings instead of discrete tokens
                logits, hidden_states = self.forward(
                    inputs_embeds=soft_embeds, return_hidden_states=True
                )
            else:
                logits, hidden_states = self.forward(
                    input_ids=output_ids, return_hidden_states=True
                )

            if in_think_mode:
                # Soft token processing: use second-last layer as soft embeddings
                soft_embeds = hidden_states[:, -1, :].unsqueeze(
                    1
                )  # Shape (batch, 1, hidden_dim)
                hidden_states_buffer.append(soft_embeds)

                # Compute probability of </think>
                think_probs = F.softmax(logits[:, -1, self.end_think_token_id], dim=-1)

                if self.should_stop_thinking(think_probs):
                    in_think_mode = False  # Exit soft token mode
                    soft_embeds = None  # Reset to normal decoding
            else:
                # Normal token sampling
                next_token = self.sample_from_probs(F.softmax(logits[:, -1, :], dim=-1))
                output_ids = torch.cat([output_ids, next_token.unsqueeze(0)], dim=-1)

                # Detect <think>
                if next_token.item() == self.think_token_id:
                    in_think_mode = True  # Enter soft token decoding

        return output_ids, hidden_states_buffer

    def should_stop_thinking(self, think_probs, threshold=0.9):
        """Determine when to exit soft token decoding based on </think> probability."""
        return think_probs.item() > threshold

    def sample_from_probs(self, probs):
        """Sample a token from probability distribution."""
        return torch.multinomial(probs, 1).item()
