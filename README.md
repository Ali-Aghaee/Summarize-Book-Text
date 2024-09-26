# Summarize-Book-Text
Book Text Summarization and Image Generation Based on Summaries

In the NLP.py file, the main program includes data cleaning, text summarization, stop word removal, and image generation.

The NLP.ipynb file contains the main code with explanations, along with data exploration and processing.

The summary.txt file contains the summaries of the book texts.

It is worth mentioning that due to hardware and time limitations, the texts of 10,000 books have been summarized, and from those, 1,000 summarized texts have been converted into images.


Here’s a breakdown of the project:

1. Data Preprocessing & Cleaning:
I tackled noisy data by cleaning and structuring the text, replacing control characters, separating books, and removing non-English content. Each book is stored in a DataFrame with key fields (Title, Author, Genre, Text). The end goal was to convert the text into a format ready for NLP.

2. NLP for Text Summarization:
I created a summarization model that condenses the key ideas from large bodies of text. This significantly reduces the size while keeping the essence intact. 

3.Text-to-Image Generation:
To push the boundaries of this project, I took the summaries and generated visual representations using a text-to-image model. Combining NLP with Computer Vision was a fun challenge, and the results were fascinating—each book now has an accompanying image to represent its content visually.
