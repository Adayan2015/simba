import re

# Slot：relation、age、sex、income、health、profession
question_dict = {
    # 关系
    'relation': ['请问您为谁投保呢？(本人，父亲，母亲，爱人，儿子，女儿，朋友等)'],
    # 年龄
    'age': ['请问您%s的年龄是什么？', '请问您%s今年几岁？', '您%s贵庚？'],
    # 性别
    'sex': ['请问您%s的性别是？', '您%s是男是女？'],
    # 收入
    'income': ['请问您家庭的年总收入（税前）是多少？', '您家庭每年收入是多少？', '您家庭每年有多少收入？'],
    # 健康状况
    'health': ['您%s有什么病例史么？', '请问您%s健康状况如何，是否患过重大疾病'],
    # 职业
    'occupation': ['请问您%s是从事什么职业的？', '请问您%s的职业是什么？', '请问您%s是什么职业？']
}
# #社保
# 'social_benefit': ['请问您%s有社保么？', '您%s有社保么？'],
# #住址
# 'address': ['请问您%s的现居地是哪里？小狮妹我住在北京', '请问您%s的现居地是啥？'],
# #目的地
# 'destination': ['您%s旅游的目的地是哪里呢？'],
# #旅行时长
# 'length_of_stay': ['您%s这次出去游玩多少天？'],
# #旅行频率
# 'frequency': ['您%s每年出游的频率是多少？'],
# #婚姻状况
# 'maritial_status': ['请问您%s是单身，已婚，还是离异？', '请问您%s目前的婚姻状况是什么？'],
# }


with open('D:\\PyCharm\\PyCharmProjects\\simba\\resources\\occupation.txt', 'r', encoding='utf-8') as f:
    occupation = f.read().split('\n')
occupation_str = '|'.join(occupation)
pattern_occ = re.compile('是*(%s)' % occupation_str)
pattern_income = re.compile('(收入|月薪|年薪|工资|每月|每个月|每年)+.*?([0-9|一二三四五六七八九十两几百千万KWkw]+)[块|人民币|元|刀]*[钱]*')
pattern_relation = re.compile(
    '[帮给为替]+我*的*([父亲|爹|爸爸|本人|自己|母亲|妈妈|爱人|老婆|妻子|丈夫|老公|儿子|孩子|小孩|小孩儿|女儿|闺女|家人|亲人|亲戚|哥|姐|妹|弟|朋友]+)')
pattern_age = re.compile('([0-9|一二三四五六七八九十]+)岁*')
pattern_sex = re.compile('(男|女|他|她)')
pattern_health = re.compile('(没有|从未|未曾|否)[患过|得过|得了|重大疾病|癌]*')
pattern_unhealth = re.compile('[曾原在自]*.*?(患过|得过|得了|有过)+([^重大疾病]+)')
str1 = '我今年23岁，工资是五十万块，想给我爹买个保险,我爸是，没有得过重大疾病'
print(re.findall(pattern_relation, str1)[0])

with open('D:\\PyCharm\\PyCharmProjects\\simba\\resources\\professiondict.txt', 'r', encoding='utf-8') as f:
    occupation = f.readlines()
for i in range(len(occupation)):
    occupation[i] = occupation[i].split()
occs = []
for occ in occupation:
    occs.append(occ[0])

occs = set(occs)
with open('D:\\PyCharm\\PyCharmProjects\\simba\\resources\\occupation.txt', 'w', encoding='utf-8') as w:
    for each in occs:
        w.write(each + '\n')

with open('D:\\PyCharm\\PyCharmProjects\\simba\\resources\\profession.txt', 'r', encoding='utf-8') as f:
    a = f.readlines()
