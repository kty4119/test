from transformers import AutoImageProcessor, ViTModel, AutoTokenizer, OPTModel, OPTForCausalLM, AutoModelForCausalLM
img_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
hidden_size = img_model.config.hidden_size
print(hidden_size)

text_model = OPTForCausalLM.from_pretrained("facebook/opt-6.7b")
input_embeddings = text_model.get_input_embeddings()
embedding_dim = input_embeddings.embedding_dim
print(embedding_dim)

in_dim = text_model.config.word_embed_proj_dim
print(in_dim)