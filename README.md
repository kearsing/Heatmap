# Flask D3.js Interactive Heatmap
Flask site to create a D3.js interactive similarity heatmap based on text data uploaded by user.
Link to production site: http://kearsing.pythonanywhere.com/

User uploads csv with label, text.
Uses Scikit-Learn TfidfVectorizer() and Cosine Similarity to create similarity scores for the matrix.
