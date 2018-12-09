# -*- coding:utf-8 -*-
from urllib.request import urlopen
from urllib.request import HTTPError
from urllib.parse import urlencode
import jieba.analyse

sum1 = 0


def test(a, b, c):
    global sum1
    if c == 1 and sum1 != 0:
        sum1 = 0
        print('a')
    else:
        sum1 = a + b


test(1, 2, 2)
test(1, 2, 1)


def num_change(sentence):
    chinese = {u"零": 0, u"一": 1, u"二": 2, u"三": 3, u"四": 4, u"五": 5, u"六": 6, u"七": 7, u"八": 8, u"九": 9,
               u"壹": 1, u"贰": 2, u"叁": 3, u"肆": 4, u"伍": 5, u"陆": 6, u"柒": 7, u"捌": 8, u"玖": 9,
               u"两": 2, u"半": 0.5, u"多": 0.5}

    units = {u"十": 10, u"百": 100, u"千": 1000, u"万": 10000, u"亿": 100000000, u"元": 1, u"岁": 1, u"块": 1,
             u"k": 1000, u"w": 10000, u'K': 1000, u"W": 10000}
    result = 0
    num = 1
    num_changed = False
    unit = [100000001, ]
    num_s = ""
    for i in sentence:
        if i.isdigit():
            num_s += i
            num = int(num_s)
            num_changed = True
        elif i in chinese.keys():
            if num_changed is False:
                num = chinese[i]
            else:
                num += chinese[i]
        elif i in units.keys():
            unit.append(units[i])
            num_s = ""
            if unit[-1] < unit[-2]:
                temp = num * unit[-1]
                result += temp
                num = 0
            else:
                result += num * 1
                result *= unit[-1]
                num = 0
        else:
            continue
    return result + num


if __name__ == '__main__':
    sentence = '五十k'
    print(num_change(sentence))

import urllib
# import insuranceqa_data
import jieba.posseg as pseg
from utils.LTML import LTML
import jieba
import json

jieba.load_userdict(r'D:\PyCharm\PyCharmProjects\simba\resources\productdict.txt')
sentence = '你好，请问太平福利全佑的承保公司是哪个'
segment = pseg.cut(sentence)
segment_utf = []
for word, flag in segment:
    print(word)
    segment_utf.append((word.encode("utf-8"), flag.encode("utf-8")))

ltml = LTML()
ltml.build_from_words([word.encode("utf-8") for word in jieba.cut(sentence)])
print(ltml.prettify())
print(ltml.tostring())
xml = ltml.tostring()

uri_base = "http://api.ltp-cloud.com/analysis/?"  # from HIT
api_key = "U1d737t0u3ooUDibTtXgFxbsGo9YVI8RQoatDkqy"
text = xml
format = 'json'
pattern = 'dp'
xml_input = 'true'
url = (uri_base
       + "api_key=" + api_key + "&"
       + "text=" + text + "&"
       + "format=" + format + "&"
       + "pattern=" + pattern + "&"
       + "xml_input" + xml_input)
try:
    response = urlopen(url)
    content = response.read().strip()
    print(content)
    print(json.loads(content)[0][0])
except HTTPError as e:
    print(e.reason)

#
# uri_base = "http://api.ltp-cloud.com/analysis/?"  # from HIT
# api_key = "U1d737t0u3ooUDibTtXgFxbsGo9YVI8RQoatDkqy"
# text = "我爱北京天安门"
# # Note that if your text contain special characters such as linefeed or '&',
# # you need to use urlencode to encode your data
# text = urllib.quote(text)
# format = 'plain'
# pattern = "dp"
#
# url = (uri_base
#        + "api_key=" + api_key + "&"
#        + "text=" + text + "&"
#        + "format=" + format + "&"
#        + "pattern=" + pattern)
#
# try:
#     response = urllib2.urlopen(url)
#     content = response.read().strip()
#     print content
# except urllib2.HTTPError, e:
#     print >> sys.stderr, e.reason
uri_base = "http://api.ltp-cloud.com/analysis/?"  # from HIT
api_key = "U1d737t0u3ooUDibTtXgFxbsGo9YVI8RQoatDkqy"
url_get_base = "http://api.ltp-cloud.com/analysis/"
args = {
    'api_key': "U1d737t0u3ooUDibTtXgFxbsGo9YVI8RQoatDkqy",
    'text': '我是中国人。',
    'pattern': 'dp',
    'format': 'plain'
}
result = urlopen(url_get_base, urlencode(args))  # POST method
content = result.read().strip()
print(content)

sentence = '你好，请问太平福利全佑的承保公司是哪个'
tag_list = jieba.analyse.extract_tags(sentence, topK=2, withWeight=False, allowPOS=("nz", "n",))
tag_list[0].encode('utf-8')
