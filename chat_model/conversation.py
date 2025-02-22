from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ConversationModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the language model and tokenizer"""
        try:
            # Load a suitable conversation model
            model_name = "facebook/opt-350m"  # You can use a larger model if available
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.eval()
        except Exception as e:
            print(f"Error initializing conversation model: {e}")
    
    def generate_response(self, user_input):
        """Generate AI response based on user input and conversation history"""
        try:
            # Add user input to conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # Prepare conversation context
            context = self._prepare_context()
            
            # Generate response
            inputs = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up response and add to history
            response = response.replace(context, "").strip()
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Maintain reasonable history length
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return response
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble generating a response right now."
    
    def _prepare_context(self):
        """Prepare conversation context from history"""
        context = ""
        for message in self.conversation_history[-5:]:  # Use last 5 messages for context
            role = message["role"]
            content = message["content"]
            context += f"{role.capitalize()}: {content}\nAssistant: "
        return context
    
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []
