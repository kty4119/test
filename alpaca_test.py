from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, MistralForCausalLM
# model = AutoModelForCausalLM.from_pretrained('/home/shared/hub/models--ty--alpaca-7b-wdiff')
# tokenizer = AutoTokenizer.from_pretrained('/home/shared/hub/models--ty--alpaca-7b-wdiff')
# alpaca = 'chavinlo/alpaca-native'
# alpaca = 'chavinlo/alpaca-native'
# model = AutoModelForCausalLM.from_pretrained(alpaca, cache_dir='home/shared/hub')
# tokenizer = AutoTokenizer.from_pretrained(alpaca, cache_dir='home/shared/hub')

model = AutoModelForCausalLM.from_pretrained('/home/shared/hub/ty_alpaca')
tokenizer = AutoTokenizer.from_pretrained('/home/shared/hub/ty_alpaca', use_fast=False, legacy=False)

# model = MistralForCausalLM.from_pretrained('/home/shared/hub/ty_mistral_instruct')
# tokenizer = AutoTokenizer.from_pretrained('/home/shared/hub/ty_mistral_instruct', use_fast=False)

# prompt = "Focus on the attribute in the sentence: A drawing of a young woman with many facial piercings."
# prompt ="What is the attribute in the sentence?: A drawing of a tattooed young woman with many facial piercings."
# prompt = "### Instruction: Focus on the attributes of the object in the caption.\n### Caption: A white toilet and sink combination in a small room.\n### Answer: "
# prompt = "### Instruction: Just answer the question.\n### Question: What color is the apple? \n### Answer: "
# prompt = "### Instruction: Just answer the question.\n### Question: What color is the apple?"
# prompt = "### Instruction: Do not include the question in the answer.\n### Question: Hey, are you conscious? Can you talk to me?"
# prompt = '### Instruction: Find the objects in the caption.\n### Caption: A man taking a bite of a doughnut while wearing a hat.'
# prompt = '### Instruction: Find the relations in the caption.\n### Caption: THERE ARE WOMEN THAT ARE LAUGHING UNDER THE UMBRELLA.'
while True:
    instruction = "### Instruction: Find the interaction between each object in the caption.\n### Caption: "
    prompt = input("프롬프트 입력: ")
    if prompt == str(1):
        break
    elif prompt == str(2):
        instruction = input("instruction 입력: ")
        instruction = "### Instruction: " + instruction + "\n### Caption: "
    else:
        
        prompt = instruction + prompt
        # print(prompt)
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate
        generate_ids = model.generate(inputs.input_ids, max_length=64)
        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(output)