# https://huggingface.co/docs/transformers/main/en/model_doc/t5#transformers.T5ForConditionalGeneration
import os 

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM


if(not os.path.exists("./stable/summarization/tokenizer") or not os.path.exists("./stable/summarization/model")) :
    tokenizer = AutoTokenizer.from_pretrained("psyche/KoT5-summarization")
    model = AutoModelForSeq2SeqLM.from_pretrained("psyche/KoT5-summarization")

    os.mkdir("stable/summarization")
    os.mkdir("stable/summarization/tokenizer")
    os.mkdir("stable/summarization/models")
    tokenizer.save_pretrained("stable/summarization/tokenizer")
    model.save_pretrained("stable/summarization/model")
else :
    tokenizer = AutoTokenizer.from_pretrained("stable/summarization/tokenizer")
    model = AutoModelForSeq2SeqLM.from_pretrained("stable/summarization/model")

text = ''' 

무너질 것을 알면서도 또다시 나뭇 가지 하나를 세운다. 
행복한 순간이라는 게 있을까 아무것도 아닌 것 같이 편안하게 느껴지는 순간의 하루가 나중에 저 멀리서 보면 특별한 날이고 소중한 시간들이 된다. 
멀리서 보면 대부분이 아름다워 보이는 것처럼 눈앞에 시간들에 흔들리지 말고 다시 하나하나 세워나간다. 
널브러진 나뭇가지들을 주워와 하나하나 다듬으면서 형태와 잘라진 나무테를 관찰하다보면 같은 나무에서 자라난 나뭇가지들도 
나눠지면 하나하나 저마다 다른 형태를 지니고 있는데 하물며 사람들 하나하나 자라온 환경도 
각자 표현하는 방식도 다를텐데 모두가 같은 삶을 바라 볼 수는 없다. 
나만의 나뭇 가지 하나를 찾아 나를 발견해 보세요. 
그리고 자신만의 나뭇가지 하나를 세워 보세요.
꼭 새로운 것이 아니어도 됩니다. 
내 주위를 잘 살펴보면 그 안에서 나를 나타낼 수 있는 것들을 발견하게 됩니다. 
그리고 잘 살펴보세요. 
일상의 모든 것들이 내 것으로 되는 소중한 경험을 해보세요.

'''

# https://huggingface.co/spaces/psyche/test-space/blob/main/app.py

input_ids = tokenizer(
    text, return_tensors="pt"
).input_ids  # Batch size 1
outputs = model.generate(input_ids, max_new_tokens=20)
result = tokenizer.batch_decode(
    model.generate(tokenizer([text], return_tensors="pt")[
                   'input_ids'], max_length=1000),
    skip_special_tokens=True)[0]

print(result)
