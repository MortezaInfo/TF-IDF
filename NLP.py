
import requests
from bs4 import BeautifulSoup
import re
import math
import time

class TF_IDF:
    def __init__(self, url = None, text = None) -> None:
        self.url = url
        self.text = text
    
    def get_url_movie(self):
        response = requests.get(self.url, headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'})
        content = BeautifulSoup(response.text, 'html.parser')
        ul = content.find('ul', attrs={'class':'ipc-metadata-list ipc-metadata-list--dividers-between sc-a1e81754-0 eBRbsI compact-list-view ipc-metadata-list--base'})
        return ul
    
    def get_summerize_movie(self):
        summerize_movie = []
        ul = self.get_url_movie()
        for link in ul.find_all('a'): #! for limit video use: for link in ul.find_all('a')[:10]
            url_movie = f"https://www.imdb.com{link.get('href')}"
            
            response = requests.get(url_movie, headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36', 'Referer': 'https://www.imdb.com/chart/top/'})
            content = BeautifulSoup(response.text, 'html.parser')
            summerize = content.find(class_='ipc-html-content-inner-div').get_text(strip=True)
            print('url')
            time.sleep(2)
            summerize_movie.append(summerize)
        return summerize_movie
    
    def clean_summerize(self):
        data = self.get_summerize_movie() if self.url else [self.text]
        clean_texts = []
        for sentence in data:
            clean_text = re.sub('<.*?>', '', sentence)
            clean_text = re.sub(r'[^\w\s]', '', clean_text)
            clean_text = clean_text.lower()
            stop_words = ['the', 'and', 'is', 'in', 'of', 'a', 'to', 'for', 'on', 'with', 'as', 'an', 'at', 'by', 'from', 'it', 'this', 'that']
            words = clean_text.split()
            clean_words = [word for word in words if word not in stop_words]
            clean_text = ' '.join(clean_words)
            clean_texts.append(clean_text)
        return clean_texts
    
    
    def calculate_tf(self, sentence):
        words = sentence.split()
        word_count = len(words)
        tf_scores = {}
        
        for word in set(words):
            tf_scores[word] = words.count(word) / word_count
        
        return tf_scores

    def calculate_idf(self, documents):
        N = len(documents)
        idf_scores = {}
        
        all_words = set([word for doc in documents for word in doc.split()])
        
        for word in all_words:
            doc_count = sum([1 for doc in documents if word in doc])
            idf_scores[word] = math.log(N / doc_count) + 1
        
        return idf_scores
    
    def samples(self):
        data = self.clean_summerize()
        idf_score = self.calculate_idf(data)
        tf_idf_scores = []
        for sample in data:
            tf_score = self.calculate_tf(sample)
            tfidf_scores = {}
            for word in tf_score:
                tfidf_scores[word] = tf_score[word] * idf_score[word]
            tf_idf_scores.append(tfidf_scores)
        return tf_idf_scores
    
    def cosine_similarity(self, vector1, vector2):
        total = 0

        for word in vector1:
            if word in vector2:
                total += vector1[word] * vector2[word]

        magnitude1 = math.sqrt(sum([value**2 for value in vector1.values()]))
        magnitude2 = math.sqrt(sum([value**2 for value in vector2.values()]))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        else:
            return total / (magnitude1 * magnitude2)


    def find_similar_movies(self, input_vector, k):
        trained_model = self.samples()
        similarities = {}
        for index, vector in enumerate(trained_model):
            similarities[index] = self.cosine_similarity(input_vector, vector)
        
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        return [f'video {i[0] + 1}' for i in sorted_similarities[:k]]


def input_summerize(text):
    client = TF_IDF(text = text).samples()[0]
    print(client)
    output = TF_IDF(url = 'https://www.imdb.com/chart/top/').find_similar_movies(client, 3)
    print(output)

input_summerize(input('send summerize movie: '))

    
