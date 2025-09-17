"""Small helper to pre-download NLTK resources non-interactively."""
import nltk
for pkg in ("stopwords", "wordnet"):
    nltk.download(pkg, quiet=True)
print("NLTK resources downloaded.")
