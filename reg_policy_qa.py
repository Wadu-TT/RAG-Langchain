# rag_policy_qa.py

# Install required packages first
# !pip install transformers datasets torch faiss-cpu wget matplotlib scikit-learn

import wget
import torch
import numpy as np
import random
import faiss
from transformers import (
    DPRContextEncoder, DPRContextEncoderTokenizer,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    AutoTokenizer, AutoModelForCausalLM
)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Helper function for visualization
def tsne_plot(data):
    tsne = TSNE(n_components=3, random_state=42, perplexity=data.shape[0]-1)
    data_3d = tsne.fit_transform(data)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.tab20(np.linspace(0, 1, len(data_3d)))
    
    for idx, point in enumerate(data_3d):
        ax.scatter(point[0], point[1], point[2], color=colors[idx], label=str(idx))
    
    ax.set_xlabel('TSNE Component 1')
    ax.set_ylabel('TSNE Component 2')
    ax.set_zlabel('TSNE Component 3')
    plt.title('3D t-SNE Visualization')
    plt.legend(title='Input Order')
    plt.show()

# Data loading and preprocessing
def read_and_split_text(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return [para.strip() for para in text.split('\n') if para.strip()]

# Context encoding and indexing
def encode_contexts(text_list):
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    
    embeddings = []
    for text in text_list:
        inputs = context_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
        outputs = context_encoder(**inputs)
        embeddings.append(outputs.pooler_output.detach().numpy())
    return np.concatenate(embeddings)

# Search function
def search_relevant_contexts(question, question_tokenizer, question_encoder, index, k=5):
    question_inputs = question_tokenizer(question, return_tensors='pt')
    question_embedding = question_encoder(**question_inputs).pooler_output.detach().numpy()
    return index.search(question_embedding, k)

# Answer generation
def generate_answer(question, contexts, model, tokenizer):
    input_text = question + ' ' + ' '.join(contexts)
    inputs = tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True)
    
    summary_ids = model.generate(
        inputs['input_ids'],
        max_new_tokens=150,
        num_beams=4,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def main():
    # Download and load policy data
    filename = 'companyPolicies.txt'
    url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'
    wget.download(url, out=filename)
    print('\nFile downloaded successfully')
    
    paragraphs = read_and_split_text(filename)
    print(f"\nLoaded {len(paragraphs)} policy paragraphs")
    
    # Create FAISS index
    context_embeddings = encode_contexts(paragraphs)
    context_embeddings_np = context_embeddings.astype('float32')
    
    index = faiss.IndexFlatL2(context_embeddings.shape[1])
    index.add(context_embeddings_np)
    print("\nFAISS index created with", index.ntotal, "entries")
    
    # Initialize question encoder
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    
    # Initialize GPT-2
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    # Example queries
    queries = [
        "What is our vacation policy?",
        "How do I submit a reimbursement request?",
        "What is the mobile device policy?",
        "How does maternity leave work?"
    ]
    
    for question in queries:
        print(f"\n{'='*50}\nQuery: {question}")
        
        # Retrieve contexts
        D, I = search_relevant_contexts(question, question_tokenizer, question_encoder, index)
        top_contexts = [paragraphs[idx] for idx in I[0]]
        
        # Generate answer
        answer = generate_answer(question, top_contexts, model, tokenizer)
        print(f"\nAnswer: {answer}")
        
        # Show top 3 contexts
        print("\nTop 3 relevant contexts:")
        for i, idx in enumerate(I[0][:3]):
            print(f"{i+1}. {paragraphs[idx]}")
            print(f"Similarity score: {D[0][i]:.2f}\n")

if __name__ == "__main__":
    main()