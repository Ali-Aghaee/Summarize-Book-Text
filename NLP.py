import re
import spacy
from spacy import displacy
import json
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
import torch


nlp = spacy.load('en_core_web_sm') # load English model

def open_file():
# opening text file
  with open('./booksummaries.txt', encoding='latin-1') as f:
    text = f.read()

  # make text cleaner
  replacements = {
          '\x19': "'", '\x1c': '', '\x13': '-', '\x01': 'a', 
          '\x18': '"', '\x0e': '', '\x0f': '', '\x08': '', 
          '\x14': '-', '\x05': '', '\x02': '', '\x03': 'ă', 
          '\x07': 'ć', '\x15': ''
  }
  for key, value in replacements.items():
      text = text.replace(key, value)
  
  return text



# make DataFrame from books
def extract_info(text):
    parts = text.split('\t',4)
    book_title = parts[0] if parts[0] else "Unknown"
    author = parts[1] if parts[1] else "Unknown"
    publication_date = parts[2] if parts[2] else "Unknown"
    genres = parts[3] if parts[3] else "Unknown"
    book_text = parts[4]

    return {
        "Book Title": book_title,
        "Author": author,
        "Publication Date": publication_date,
        "Genres": genres,
        "Book Text": book_text
    }





# make Book Text cleaner
def data_cleaner(book_text):
  # book_text = row['Book Text']

  book_text = re.sub(r'\t',' ', book_text)
  book_text = re.sub(r'\n', ' ', book_text)

  book_text = re.sub(r'\[','', book_text)
  book_text = re.sub(r'\]','', book_text)
  book_text = re.sub(r'\{.*?\}','', book_text)
  book_text = re.sub(r'\([^\)]*\)','', book_text)
  book_text = re.sub(r'https?://[^\s]*', ' ', book_text) #delete links
  
  #delete non English languages
  patterns_to_remove = [
    r'\[', r'\]', r'\{.*?\}', r'\([^\)]*\)', 
    r'https?://[^\s]*', r' {2,}',
    r' ar:.*', r' bg:.*',r' bn:.*',r' ca:.*',r' cs:.*',r' cy:.*',r' de:.*',r' eo:.*',r' es:.*',
    r' et:.*',r' eu:.*',r' eu:.*',r' fa:.*',r' fi:.*',r' fr:.*',r' ga:.*',r' gl:.*',r' gv:.*',
    r' he:.*',r' hi:.*',r' hr:.*',r' hu:.*',r' ja:.*',r' ka:.*',r' ko:.*',r' lt:.*',r' ms:.*',
    r' ne:.*',r' nl:.*',r' no:.*',r' no:.*',r' pa:.*',r' pl:.*',r' pt:.*',r' ro:.*',r' ru:.*',
    r' sk:.*',r' sr:.*',r' sv:.*',r' ta:.*',r' te:.*',r' th:.*',r' tl:.*',r' tr:.*',r' uk:.*',
    r' vi:.*',r' zh:.*',r' Ad:.*',

  ]
  for pattern in patterns_to_remove:
    book_text = re.sub(pattern, ' ', book_text)
  
  book_text = re.sub(r' {2,}', ' ', book_text) #delete 2 or more spaces

  return book_text






#summarize Book Text
summarizer = pipeline("summarization", model="facebook/bart-large-cnn",device=0)#device=0 means using GPU

def split_text_by_sentences(text, chunk_size=5): #split big text into chunks
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    return chunks

def summarize_long_text(text): # handel summarizing big text
    chunks = split_text_by_sentences(text)
    max_length = 50
    min_length = 10
    summaries = [summarizer(chunk, max_length=max_length,min_length=min_length,do_sample=False)[0]['summary_text'] for chunk in chunks]
    final_summary = ' '.join(summaries)
    while len(final_summary) >= 4500:
      final_summary = summarize_long_text(final_summary)
    if len(final_summary) >= 400:
      final_summary = summarizer(final_summary, max_length=100,do_sample=False)[0]['summary_text']
    return final_summary

def summarize_text(text):
    # اطمینان از اینکه متن خالی نیست
    if not text.strip():
        return ''

    text_length = len(text)

    if text_length >=4500:
      s = summarize_long_text(text)
    else:
      s = summarizer(text, max_length =200 , do_sample=False)[0]['summary_text']  # summarize short text
    return s



# delete stop words
def remove_stop_words(text):
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if not token.is_stop]
    stop_words = ' '.join(filtered_tokens)
    return stop_words
  




#loading with GPU
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")

# make image and save them
def generate_image_from_text(row,text, output_path):
    text = row['Remove Stop_words']
    
    # make image from text
    image = pipe(text).images[0]

    output_path = f'./image/{row['Book Title']}.png'
    # save image
    image.save(output_path)
    print('image saved')




# make Book Title cleaner
def title_cleaner(title):
  title = re.sub(r'[\$\~\\]','', title)
  title = re.sub(r'.hack//','', title)
  title = re.sub(r'','P', title)
  title = re.sub(r'\n',' ', title)
  title = re.sub(r' {2,}',' ', title)
  return title




# make Genres cleaner in list type
def genres_cleaner(genres):
  if genres != 'Unknown':
    genres = json.loads(genres)
    return list(genres.values())
  else:
    return 'Unknown'



# make Author name cleaner
def author_cleaner(author):
  author = re.sub(r'\n',' ', author)
  author = re.sub(r'[!]','.', author)
  author = re.sub(r'\+','', author)
  author = re.sub(r', \d+.*','', author)
  author = re.sub(r'»','', author)
  author = re.sub(r' "\w+"','', author)
  author = re.sub(r'\x10','D', author)
  author = re.sub(r'\x0c','', author)
  author = re.sub(r'\x1f','g', author)
  author = re.sub(r'\[','s', author)
  author = re.sub(r' \([^)]*\)','', author)
  author = re.sub(r'\)','', author)
  return author





text = open_file

#split the text to books
books = re.split(r'\d+\t/m/\w+\t', text)[1:]

#make Datafram
df = pd.DataFrame([extract_info(book) for book in books])


df['Book Text'] = df['Book Text'].apply(data_cleaner)

df = df[df['Book Text'].str.len() > 200]  # Filter short texts

# add new column for summarized text
df['Summary'] = df['Book Text'].apply(summarize_text,axis =1)

 # add new column for removed stop words from summarized text 
df['Remove Stop_words'] = df['Summary'].apply(remove_stop_words,axis = 1) 

# replace clean data
df['Book Title'] = df['Book Title'].apply(title_cleaner)

df.apply(generate_image_from_text)

# replace clean data
df['Genres'] = df['Genres'].apply(genres_cleaner)

# replace clean data
df['Author'] = df['Author'].apply(author_cleaner)