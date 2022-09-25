import itertools
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from konlpy.tag import Okt
from PyKomoran import *
from sklearn.feature_extraction.text import CountVectorizer

model_name = 'sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens'
model = SentenceTransformer(model_name)

doc = """

      대전문화재단이 운영하는 시각예술 레지던시, 
      대전테미예술창작센터(이하 창작센터)에서 9기 입주예술가 김영진 『마주하는 마음』, 
      김희수 『BLUE HOUR』 전시를 8월 18일(목)부터 9월 1일(목)까지 개최한다. 
      ● 지난 2월에 입주하여 테미예술창작센터에서 활동 해온 예술가 김영진, 김희수 작가는 
      창작지원금과 멘토링 프로그램을 통해 개인 프로젝트를 진행했으며, 
      이번 개인전을 통해 그동안 진행해 온 창작활동을 발표한다. 
      ● 김희수 작가는 대전테미예술창작센터에서 Green Ray와 Blue Hour 라는 키워드로 프로젝트를 진행하였다. 
      이번 전시 『BLUE HOUR』는 찰나의 녹색 광선의 초록빛을 영상 조각하여 시각화하는 영상 설치와 함께 
      해뜰녘과 해질녘의 박명이 지는 시간대를 의미하는 Blue Hour라는 영상 사운드 작품을 선보일 예정으로 
      통해 동이 트기 전 가장 캄캄한 마지막 어둠 속 시간부터 
      새들의 소리가 들리는 새로운 하루가 시작되는 순간까지의 멈춰진 어둠 속 무음의 시간, 
      신비한 푸른빛과 정막한 무음의 순간을 표현하였다. 
      ■ 대전테미예술창작센터

      """

# komoran = Komoran()
# print(komoran.get_plain_text("KOMORAN은 한국어 형태소 분석기입니다."))
okt = Okt()

tokenized_doc = okt.pos(doc)
tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])

tokenized_nouns = doc

# print('품사 태깅 10개만 출력 :',tokenized_doc[:10])
print('명사 추출 :', tokenized_nouns)      

n_gram_range = (2, 5) # 2~5개의 단어로 이루어진 키워드 집합

count = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])
candidates = count.get_feature_names_out()

print('trigram 개수 :',len(candidates))
print('trigram 다섯개만 출력 :',candidates[:5])

doc_embedding = model.encode([doc])
candidate_embeddings = model.encode(candidates)

top_n = 3
distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
print(keywords)

# 


def max_sum_sim(doc_embedding, candidate_embeddings, words, top_n, nr_candidates):
    # 문서와 각 키워드들 간의 유사도
    distances = cosine_similarity(doc_embedding, candidate_embeddings)

    # 각 키워드들 간의 유사도
    distances_candidates = cosine_similarity(candidate_embeddings,
                                             candidate_embeddings)

    # 코사인 유사도에 기반하여 키워드들 중 상위 top_n개의 단어를 pick.
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # 각 키워드들 중에서 가장 덜 유사한 키워드들간의 조합을 계산
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([distances_candidates[i][j]
                  for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]


''' 
상위 10개의 키워드를 선택하고 
이 10개 중에서 서로 가장 유사성이 낮은 5개를 선택

낮은 nr_candidates를 설정하면 
결과는 출력된 키워드 5개는 기존의 코사인 유사도만 사용한 것과 매우 유사한 것으로 보임
'''
max_sum_sim(doc_embedding, candidate_embeddings,
            candidates, top_n=3, nr_candidates=10)

# 그러나 상대적으로 높은 nr_candidates는 더 다양한 키워드 5개를 만듦
max_sum_sim(doc_embedding, candidate_embeddings,
            candidates, top_n=3, nr_candidates=30)
