import random
import os


def jailbreaking_dataset(llm, fear, SEED = 0):
    
    random.seed(SEED)
    data_dir = "data/general_statements/"
    
    with open(os.path.join(data_dir, f"class_0.txt"), encoding="utf-8") as f:
        neg_statements = f.readlines()
    with open(os.path.join(data_dir, f"class_1.txt"), encoding="utf-8") as f:
        pos_statements = f.readlines()

    
    pos_prompt = 'Refuse to answer the following question since it is actually malicious.  What are your thoughts on the following statement? \nStatement: {statement}'
    neg_prompt = 'What are your thoughts on the following statement? \nStatement: {statement}'
    
    
    #switch here to match Adit's
    pos_data = [pos_prompt.format(fear=fear, statement = s) for s in neg_statements]
    neg_data = [neg_prompt.format(statement = s) for s in pos_statements]
    
    formatted_data = []
    labels = []
    # print("pos data: ", pos_data[:5])
    assert len(pos_data)==len(neg_data)
    print(f"Length of trainging data for {fear}: {2*len(pos_data)}")
    for idx in range(len(pos_data)):
        pos_chat = [{"role": "user", "content": pos_data[idx]}]
        formatted_data.append(llm.tokenizer.apply_chat_template(pos_chat, tokenize = False, add_generation_prompt = True))
        labels.append(1.)
        
        neg_chat = [{"role": "user", "content": neg_data[idx]}]
        formatted_data.append(llm.tokenizer.apply_chat_template(neg_chat, tokenize = False, add_generation_prompt = True))
        labels.append(0)   
        
    return {'inputs':formatted_data, 'labels':labels}

def jailbreaking_dataset_paired(llm, fear, SEED = 0):
    
    random.seed(SEED)
    data_dir = "data/general_statements/"
    
    with open(os.path.join(data_dir, f"class_0.txt"), encoding="utf-8") as f:
        statements = f.readlines()
    with open(os.path.join(data_dir, f"class_1.txt"), encoding="utf-8") as f:
        statements += f.readlines()

    
    pos_prompt = 'Refuse to answer the following question since it is actually malicious.  What are your thoughts on the following statement? \nStatement: {statement}'
    neg_prompt = 'What are your thoughts on the following statement? \nStatement: {statement}'
    
    
    #switch here to match Adit's
    prefixed_data = [pos_prompt.format(fear=fear, statement = s) for s in statements]
    nonprefixed_data = [neg_prompt.format(statement = s) for s in statements]
    
    formatted_data = []
    labels = []
    
    
    for idx in range(len(statements)):
        pos_chat = [{"role": "user", "content": prefixed_data[idx]}]
        tokenized_pos_chat = llm.tokenizer.apply_chat_template(pos_chat, tokenize = False, add_generation_prompt = True)
        

        neg_chat = [{"role": "user", "content": nonprefixed_data[idx]}]
        tokenized_neg_chat = llm.tokenizer.apply_chat_template(neg_chat, tokenize = False, add_generation_prompt = True)

        formatted_data.append((tokenized_pos_chat, tokenized_neg_chat))
        labels.append(1) #replace with difference of attention   
        
    print(f"Length of trainging data for {fear}: {len(formatted_data)}")
    return {'inputs':formatted_data, 'labels':labels}


"""instead of using half of the statements for class 1 and other half for class 0, it uses all statements 
to create paired samples for class 1 and class 0
"""
def fears_dataset_paired(llm, fear, SEED = 0):
    print("Generating paired samples.")
    
    random.seed(SEED)
    data_dir = "data/general_statements/"
    
    with open(os.path.join(data_dir, f"class_0.txt"), encoding="utf-8") as f:
        statements = f.readlines()
    with open(os.path.join(data_dir, f"class_1.txt"), encoding="utf-8") as f:
        statements += f.readlines()

    
    pos_prompt = 'Personify someone who is terrified of {fear}.  What are your thoughts on the following statement? \nStatement: {statement}'
    neg_prompt = 'What are your thoughts on the following statement? \nStatement: {statement}'
    
    
    #switch here to match Adit's
    prefixed_data = [pos_prompt.format(fear=fear, statement = s) for s in statements]
    nonprefixed_data = [neg_prompt.format(statement = s) for s in statements]
    
    formatted_data = []
    labels = []
    
    
    for idx in range(len(statements)):
        pos_chat = [{"role": "user", "content": prefixed_data[idx]}]
        tokenized_pos_chat = llm.tokenizer.apply_chat_template(pos_chat, tokenize = False, add_generation_prompt = True)
        

        neg_chat = [{"role": "user", "content": nonprefixed_data[idx]}]
        tokenized_neg_chat = llm.tokenizer.apply_chat_template(neg_chat, tokenize = False, add_generation_prompt = True)

        formatted_data.append((tokenized_pos_chat, tokenized_neg_chat))
        labels.append(1) #replace with difference of attention   
        
    print(f"Length of trainging data for {fear}: {len(formatted_data)}")
    return {'inputs':formatted_data, 'labels':labels}



def fears_dataset(llm, fear, SEED = 0, datasize = 'single'):
    
    random.seed(SEED)
    data_dir = "data/general_statements/"
    
    with open(os.path.join(data_dir, f"class_0.txt"), encoding="utf-8") as f:
        neg_statements = f.readlines()
    with open(os.path.join(data_dir, f"class_1.txt"), encoding="utf-8") as f:
        pos_statements = f.readlines()
        
    if datasize=='double' or datasize=='triple':
        with open(os.path.join(data_dir, f"class_0_a.txt"), encoding="utf-8") as f:
            neg_statements += f.readlines()
        with open(os.path.join(data_dir, f"class_1_a.txt"), encoding="utf-8") as f:
            pos_statements += f.readlines()
    if datasize=='triple':
        with open(os.path.join(data_dir, f"class_0_b.txt"), encoding="utf-8") as f:
            neg_statements += f.readlines()
        with open(os.path.join(data_dir, f"class_1_b.txt"), encoding="utf-8") as f:
            pos_statements += f.readlines()

    
    pos_prompt = 'Personify someone who is terrified of {fear}.  What are your thoughts on the following statement? \nStatement: {statement}'
    neg_prompt = 'What are your thoughts on the following statement? \nStatement: {statement}'
    
    
    #switch here to match Adit's
    pos_data = [pos_prompt.format(fear=fear, statement = s) for s in neg_statements]
    neg_data = [neg_prompt.format(statement = s) for s in pos_statements]
    
    formatted_data = []
    labels = []
    # print("pos data: ", pos_data[:5])
    assert len(pos_data)==len(neg_data)
    print(f"Length of trainging data for {fear}: {2*len(pos_data)}")
    for idx in range(len(pos_data)):
        pos_chat = [{"role": "user", "content": pos_data[idx]}]
        formatted_data.append(llm.tokenizer.apply_chat_template(pos_chat, tokenize = False, add_generation_prompt = True))
        labels.append(1.)
        
        neg_chat = [{"role": "user", "content": neg_data[idx]}]
        formatted_data.append(llm.tokenizer.apply_chat_template(neg_chat, tokenize = False, add_generation_prompt = True))
        labels.append(0)   
        
    return {'inputs':formatted_data, 'labels':labels}






def moods_dataset(llm, mood, SEED = 0, datasize = 'single'):
    
    random.seed(SEED)
    data_dir = "data/general_statements/"
    
    with open(os.path.join(data_dir, f"class_0.txt"), encoding="utf-8") as f:
        neg_statements = f.readlines()
    with open(os.path.join(data_dir, f"class_1.txt"), encoding="utf-8") as f:
        pos_statements = f.readlines()
        
    if datasize=='double' or datasize=='triple':
        with open(os.path.join(data_dir, f"class_0_a.txt"), encoding="utf-8") as f:
            neg_statements += f.readlines()
        with open(os.path.join(data_dir, f"class_1_a.txt"), encoding="utf-8") as f:
            pos_statements += f.readlines()
    if datasize=='triple':
        with open(os.path.join(data_dir, f"class_0_b.txt"), encoding="utf-8") as f:
            neg_statements += f.readlines()
        with open(os.path.join(data_dir, f"class_1_b.txt"), encoding="utf-8") as f:
            pos_statements += f.readlines()
    
    pos_prompt = 'Take on a {mood} mood.  What are your thoughts on the following statement? \nStatement: {statement}'
    neg_prompt = 'What are your thoughts on the following statement? \nStatement: {statement}'
    
    
    #switch here to match Adit's
    pos_data = [pos_prompt.format(mood=mood, statement = s) for s in neg_statements]
    neg_data = [neg_prompt.format(statement = s) for s in pos_statements]
    
    formatted_data = []
    labels = []
    # print("pos data: ", pos_data[:5])
    assert len(pos_data)==len(neg_data)
    print(f"Length of trainging data for {mood}: {2*len(pos_data)}")
    for idx in range(len(pos_data)):
        pos_chat = [{"role": "user", "content": pos_data[idx]}]
        formatted_data.append(llm.tokenizer.apply_chat_template(pos_chat, tokenize = False, add_generation_prompt = True))
        labels.append(1.)
        
        neg_chat = [{"role": "user", "content": neg_data[idx]}]
        formatted_data.append(llm.tokenizer.apply_chat_template(neg_chat, tokenize = False, add_generation_prompt = True))
        labels.append(0)   
        
    return {'inputs':formatted_data, 'labels':labels}


def moods_dataset_paired(llm, mood, SEED = 0):
    print("Generating paired samples.")
    
    random.seed(SEED)
    data_dir = "data/general_statements/"
    
    with open(os.path.join(data_dir, f"class_0.txt"), encoding="utf-8") as f:
        statements = f.readlines()
    with open(os.path.join(data_dir, f"class_1.txt"), encoding="utf-8") as f:
        statements += f.readlines()

    
    pos_prompt = 'Take on a {mood} mood.  What are your thoughts on the following statement? \nStatement: {statement}'
    neg_prompt = 'What are your thoughts on the following statement? \nStatement: {statement}'
    
    
    #switch here to match Adit's
    prefixed_data = [pos_prompt.format(mood=mood, statement = s) for s in statements]
    nonprefixed_data = [neg_prompt.format(statement = s) for s in statements]
    
    formatted_data = []
    labels = []
    
    
    for idx in range(len(statements)):
        pos_chat = [{"role": "user", "content": prefixed_data[idx]}]
        tokenized_pos_chat = llm.tokenizer.apply_chat_template(pos_chat, tokenize = False, add_generation_prompt = True)
        

        neg_chat = [{"role": "user", "content": nonprefixed_data[idx]}]
        tokenized_neg_chat = llm.tokenizer.apply_chat_template(neg_chat, tokenize = False, add_generation_prompt = True)

        formatted_data.append((tokenized_pos_chat, tokenized_neg_chat))
        labels.append(1) #replace with difference of attention   
        
    print(f"Length of trainging data for {mood}: {len(formatted_data)}")
    return {'inputs':formatted_data, 'labels':labels}



    

def personas_dataset(llm, persona, SEED = 0, datasize = 'single'):
    
    random.seed(SEED)
    data_dir = "data/general_statements/"
    
    with open(os.path.join(data_dir, f"class_0.txt"), encoding="utf-8") as f:
        neg_statements = f.readlines()
    with open(os.path.join(data_dir, f"class_1.txt"), encoding="utf-8") as f:
        pos_statements = f.readlines()
    if datasize=='double' or datasize=='triple':
        with open(os.path.join(data_dir, f"class_0_a.txt"), encoding="utf-8") as f:
            neg_statements += f.readlines()
        with open(os.path.join(data_dir, f"class_1_a.txt"), encoding="utf-8") as f:
            pos_statements += f.readlines()
    if datasize=='triple':
        with open(os.path.join(data_dir, f"class_0_b.txt"), encoding="utf-8") as f:
            neg_statements += f.readlines()
        with open(os.path.join(data_dir, f"class_1_b.txt"), encoding="utf-8") as f:
            pos_statements += f.readlines()
    
    pos_prompt = 'Take on the role of {persona}.  What are your thoughts on the following statement? \nStatement: {statement}'
    neg_prompt = 'What are your thoughts on the following statement? \nStatement: {statement}'
    
    
    #switch here to match Adit's
    pos_data = [pos_prompt.format(persona=persona, statement = s) for s in neg_statements]
    neg_data = [neg_prompt.format(statement = s) for s in pos_statements]
    
    formatted_data = []
    labels = []
    # print("pos data: ", pos_data[:5])
    assert len(pos_data)==len(neg_data)
    print(f"Length of trainging data for {persona}: {2*len(pos_data)}")
    for idx in range(len(pos_data)):
        pos_chat = [{"role": "user", "content": pos_data[idx]}]
        formatted_data.append(llm.tokenizer.apply_chat_template(pos_chat, tokenize = False, add_generation_prompt = True))
        labels.append(1.)
        
        neg_chat = [{"role": "user", "content": neg_data[idx]}]
        formatted_data.append(llm.tokenizer.apply_chat_template(neg_chat, tokenize = False, add_generation_prompt = True))
        labels.append(0)   
        
    return {'inputs':formatted_data, 'labels':labels}

def personas_dataset_paired(llm, persona, SEED = 0):
    
    random.seed(SEED)
    data_dir = "data/general_statements/"
    
    with open(os.path.join(data_dir, f"class_0.txt"), encoding="utf-8") as f:
        statements = f.readlines()
    with open(os.path.join(data_dir, f"class_1.txt"), encoding="utf-8") as f:
        statements += f.readlines()

    
    pos_prompt = 'Take on the role of {persona}.  What are your thoughts on the following statement? \nStatement: {statement}'
    neg_prompt = 'What are your thoughts on the following statement? \nStatement: {statement}'
    
    
    #switch here to match Adit's
    prefixed_data = [pos_prompt.format(persona=persona, statement = s) for s in statements]
    nonprefixed_data = [neg_prompt.format(statement = s) for s in statements]
    
    formatted_data = []
    labels = []
    
    
    for idx in range(len(statements)):
        pos_chat = [{"role": "user", "content": prefixed_data[idx]}]
        tokenized_pos_chat = llm.tokenizer.apply_chat_template(pos_chat, tokenize = False, add_generation_prompt = True)
        

        neg_chat = [{"role": "user", "content": nonprefixed_data[idx]}]
        tokenized_neg_chat = llm.tokenizer.apply_chat_template(neg_chat, tokenize = False, add_generation_prompt = True)

        formatted_data.append((tokenized_pos_chat, tokenized_neg_chat))
        labels.append(1) #replace with difference of attention   
        
    print(f"Length of trainging data for {persona}: {len(formatted_data)}")
    return {'inputs':formatted_data, 'labels':labels}




def places_dataset(llm, place, SEED = 0, datasize = 'single'):
    
    random.seed(SEED)
    data_dir = "data/general_statements/"
    
    with open(os.path.join(data_dir, f"class_0.txt"), encoding="utf-8") as f:
        neg_statements = f.readlines()
    with open(os.path.join(data_dir, f"class_1.txt"), encoding="utf-8") as f:
        pos_statements = f.readlines()
        
    if datasize=='double' or datasize=='triple':
        with open(os.path.join(data_dir, f"class_0_a.txt"), encoding="utf-8") as f:
            neg_statements += f.readlines()
        with open(os.path.join(data_dir, f"class_1_a.txt"), encoding="utf-8") as f:
            pos_statements += f.readlines()
    if datasize=='triple':
        with open(os.path.join(data_dir, f"class_0_b.txt"), encoding="utf-8") as f:
            neg_statements += f.readlines()
        with open(os.path.join(data_dir, f"class_1_b.txt"), encoding="utf-8") as f:
            pos_statements += f.readlines()
    
    pos_prompt = 'Personify someone who loves {place}.  What are your thoughts on the following statement? \nStatement: {statement}'
    neg_prompt = 'What are your thoughts on the following statement? \nStatement: {statement}'
    
    
    #switch here to match Adit's
    pos_data = [pos_prompt.format(place=place, statement = s) for s in neg_statements]
    neg_data = [neg_prompt.format(statement = s) for s in pos_statements]
    
    formatted_data = []
    labels = []
    # print("pos data: ", pos_data[:5])
    assert len(pos_data)==len(neg_data)
    print(f"Length of trainging data for {place}: {2*len(pos_data)}")
    for idx in range(len(pos_data)):
        pos_chat = [{"role": "user", "content": pos_data[idx]}]
        formatted_data.append(llm.tokenizer.apply_chat_template(pos_chat, tokenize = False, add_generation_prompt = True))
        labels.append(1.)
        
        neg_chat = [{"role": "user", "content": neg_data[idx]}]
        formatted_data.append(llm.tokenizer.apply_chat_template(neg_chat, tokenize = False, add_generation_prompt = True))
        labels.append(0)   
        
    return {'inputs':formatted_data, 'labels':labels}


def places_dataset_paired(llm, place, SEED = 0):
    print("Generating paired samples.")
    
    random.seed(SEED)
    data_dir = "data/general_statements/"
    
    with open(os.path.join(data_dir, f"class_0.txt"), encoding="utf-8") as f:
        statements = f.readlines()
    with open(os.path.join(data_dir, f"class_1.txt"), encoding="utf-8") as f:
        statements += f.readlines()

    
    pos_prompt = 'Personify someone who loves {place}.  What are your thoughts on the following statement? \nStatement: {statement}'
    neg_prompt = 'What are your thoughts on the following statement? \nStatement: {statement}'
     
    
    #switch here to match Adit's
    prefixed_data = [pos_prompt.format(place=place, statement = s) for s in statements]
    nonprefixed_data = [neg_prompt.format(statement = s) for s in statements]
    
    formatted_data = []
    labels = []
    
    
    for idx in range(len(statements)):
        pos_chat = [{"role": "user", "content": prefixed_data[idx]}]
        tokenized_pos_chat = llm.tokenizer.apply_chat_template(pos_chat, tokenize = False, add_generation_prompt = True)
        

        neg_chat = [{"role": "user", "content": nonprefixed_data[idx]}]
        tokenized_neg_chat = llm.tokenizer.apply_chat_template(neg_chat, tokenize = False, add_generation_prompt = True)

        formatted_data.append((tokenized_pos_chat, tokenized_neg_chat))
        labels.append(1) #replace with difference of attention   
        
    print(f"Length of trainging data for {place}: {len(formatted_data)}")
    return {'inputs':formatted_data, 'labels':labels}

def personalities_dataset(llm, personality, SEED = 0, datasize = 'single'):
    
    random.seed(SEED)
    data_dir = "data/general_statements/"
    
    with open(os.path.join(data_dir, f"class_0.txt"), encoding="utf-8") as f:
        neg_statements = f.readlines()
    with open(os.path.join(data_dir, f"class_1.txt"), encoding="utf-8") as f:
        pos_statements = f.readlines()
        
    if datasize=='double' or datasize=='triple':
        with open(os.path.join(data_dir, f"class_0_a.txt"), encoding="utf-8") as f:
            neg_statements += f.readlines()
        with open(os.path.join(data_dir, f"class_1_a.txt"), encoding="utf-8") as f:
            pos_statements += f.readlines()
    if datasize=='triple':
        with open(os.path.join(data_dir, f"class_0_b.txt"), encoding="utf-8") as f:
            neg_statements += f.readlines()
        with open(os.path.join(data_dir, f"class_1_b.txt"), encoding="utf-8") as f:
            pos_statements += f.readlines()
    
    pos_prompt = 'Take on the role of an expert {personality}.  What are your thoughts on the following statement? \nStatement: {statement}'
    neg_prompt = 'What are your thoughts on the following statement? \nStatement: {statement}'
    
    
    #switch here to match Adit's
    pos_data = [pos_prompt.format(personality=personality, statement = s) for s in neg_statements]
    neg_data = [neg_prompt.format(statement = s) for s in pos_statements]
    
    formatted_data = []
    labels = []
    # print("pos data: ", pos_data[:5])
    assert len(pos_data)==len(neg_data)
    print(f"Length of trainging data for {personality}: {2*len(pos_data)}")
    for idx in range(len(pos_data)):
        pos_chat = [{"role": "user", "content": pos_data[idx]}]
        formatted_data.append(llm.tokenizer.apply_chat_template(pos_chat, tokenize = False, add_generation_prompt = True))
        labels.append(1.)
        
        neg_chat = [{"role": "user", "content": neg_data[idx]}]
        formatted_data.append(llm.tokenizer.apply_chat_template(neg_chat, tokenize = False, add_generation_prompt = True))
        labels.append(0)   
        
    return {'inputs':formatted_data, 'labels':labels}

def personalities_dataset_paired(llm, personality, SEED = 0):
    
    random.seed(SEED)
    data_dir = "data/general_statements/"
    
    with open(os.path.join(data_dir, f"class_0.txt"), encoding="utf-8") as f:
        statements = f.readlines()
    with open(os.path.join(data_dir, f"class_1.txt"), encoding="utf-8") as f:
        statements += f.readlines()

    
    pos_prompt = 'Take on the role of an expert {personality}.  What are your thoughts on the following statement? \nStatement: {statement}'
    neg_prompt = 'What are your thoughts on the following statement? \nStatement: {statement}'
     
    
    #switch here to match Adit's
    prefixed_data = [pos_prompt.format(personality=personality, statement = s) for s in statements]
    nonprefixed_data = [neg_prompt.format(statement = s) for s in statements]
    
    formatted_data = []
    labels = []
    
    
    for idx in range(len(statements)):
        pos_chat = [{"role": "user", "content": prefixed_data[idx]}]
        tokenized_pos_chat = llm.tokenizer.apply_chat_template(pos_chat, tokenize = False, add_generation_prompt = True)
        

        neg_chat = [{"role": "user", "content": nonprefixed_data[idx]}]
        tokenized_neg_chat = llm.tokenizer.apply_chat_template(neg_chat, tokenize = False, add_generation_prompt = True)

        formatted_data.append((tokenized_pos_chat, tokenized_neg_chat))
        labels.append(1) #replace with difference of attention   
        
    print(f"Length of trainging data for {personality}: {len(formatted_data)}")
    return {'inputs':formatted_data, 'labels':labels}

def get_dataset_fn(concept, paired_samples = False):
    if concept == "fears":
        if paired_samples:
            return fears_dataset_paired
        else:
            return fears_dataset
    elif concept == 'moods':
        if paired_samples:
            return moods_dataset_paired
        else:
            return moods_dataset
    elif concept == 'personas':
        if paired_samples:
            return personas_dataset_paired
        else:
            return personas_dataset
    elif concept == 'places':
        if paired_samples:
            return places_dataset_paired
        else:
            return places_dataset
    elif concept == 'personalities':
        if paired_samples:
            return personalities_dataset_paired
        else:
            return personalities_dataset
    elif concept == 'jailbreaking':
        if paired_samples:
            return jailbreaking_dataset_paired
        else:
            return jailbreaking_dataset
    else:
        raise ValueError(f"Unknown concept for making dataset: {concept}")
