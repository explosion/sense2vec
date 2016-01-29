'''Serve vector queries for phrases'''
from __future__ import unicode_literals
import ujson as json

import falcon
from fancyvec.vectors import VectorMap
from spacy.lemmatizer import Lemmatizer
from spacy.en import LOCAL_DATA_DIR


class Similarity(object):
    def __init__(self, vectors_dir):
        print("Load vector map")
        self.w2v = VectorMap(128)
        self.w2v.load(vectors_dir)
        self.lemmatizer = Lemmatizer.load(LOCAL_DATA_DIR)
        self.parts_of_speech = ['NOUN', 'VERB', 'ADJ', 'ORG', 'PERSON', 'FAC',
                                'PRODUCT', 'LOC', 'GPE']
        print("Serve")

    def on_get(self, req, resp, query=''):
        print(repr(query))
        if not isinstance(query, unicode):
            query = query.decode('utf8')
        resp.content_type = 'text/string'
        resp.append_header('Access-Control-Allow-Origin', "*")
        resp.status = falcon.HTTP_200
        resp.body = json.dumps(self.handle(query), indent=4)

    def handle(self, query, n=100):
        # Don't return the original
        print("Find best query")
        key = self._find_best_key(query)
        print(repr(key))
        if not key:
            return {'key': '', 'text': query, 'results': [], 'count': 0}
        text = key.rsplit('|', 1)[0].replace('_', ' ')
        results = []
        seen = set([text])
        seen.add(self._find_head(key))
        for entry, score in self.get_similar(key, n * 2):
            head = self._find_head(entry)
            freq, _ = self.w2v[entry]
            if head not in seen:
                results.append(
                    {
                        'score': score,
                         'key': entry,
                         'text': entry.split('|')[0].replace('_', ' '),
                         'count': freq,
                         'head': head
                    })
                seen.add(head)
            if len(results) >= n:
                break
        freq, _ = self.w2v[key]
        return {'text': text, 'key': key, 'results': results,
                'count': freq,
                'head': self._find_head(key)}
     
    def _find_best_key(self, query):
        query = query.replace(' ', '_')
        if '|' in query:
            text, pos = query.rsplit('|', 1)
            key = text + '|' + pos.upper()
            return key if key in self.w2v else None
      
        freqs = []
        casings = [query, query.upper(), query.title()] if query.islower() else [query]
        for text in casings:
            for pos in self.parts_of_speech:
                key = text + '|' + pos
                if key in self.w2v:
                    freqs.append((self.w2v[key][0], key))
        return max(freqs)[1] if freqs else None
     
    def _find_head(self, entry):
        if '|' not in entry:
            return entry.lower()
        text, pos = entry.rsplit('|', 1)
        head = text.split('_')[-1]
        return min(self.lemmatizer(head, pos))
 
    def get_similar(self, query, n):
        if query not in self.w2v:
            return []
        freq, query_vector = self.w2v[query]
        words, scores = self.w2v.most_similar(query_vector, n)
        return zip(words, scores)


similarity = Similarity('data/vector_map/')
app = falcon.API()
app.add_route('/api/similarity/reddit/{query}', similarity)

print(similarity.handle('natural language processing'))
