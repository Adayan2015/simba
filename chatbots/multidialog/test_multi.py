from chatbots.profession import ProfessionBot
import re
import numpy as np


# 多轮对话机器人
class MultiDialogBot:

    def __init__(self):
        with open('D:\\PyCharm\\PyCharmProjects\\simba\\resources\\occupation.txt', 'r', encoding='utf-8') as f:
            self.occupation = f.read().split('\n')

        self.occupation_str = '|'.join(self.occupation)
        self.ask_relation = ['请问被保人与您的关系是?', '请问被保人是您的什么人呢？']
        self.ask_age = ['请问被保人的年龄是多大呢？', '请问被保人今年贵庚？']
        self.ask_sex = ['请问被保人是男是女呢？']
        self.ask_sick = ['请问被保人是否患过重大疾病呢？', '请问被保人有无重病史?']
        self.ask_occupation = ['请问被保人的职业是什么呢？', '请问被保人是做什么的呢？']
        self.ask_income = ['请问您家庭收入怎么样？', '请问您收入如何?']
        self.slot_information = {'relation': '', 'age': '', 'occupation': '', 'sex': '', 'income': '', 'sick': ''}
        self.ask_information = {'relation': self.ask_relation, 'age': self.ask_age, 'occupation': self.ask_occupation,
                                'sex': self.ask_sex, 'income': self.ask_income, 'sick': self.ask_sick}
        self.last_ask = ''
        self.pattern_occupation = re.compile('是*(%s)' % self.occupation_str)
        self.pattern_income = re.compile('(收入|月薪|年薪|工资|每月|每个月|每年|年收入)+.*?([0-9|一二三四五六七八九十两几百千万KWkw]+)[块|人民币|元|刀]*[钱]*')
        self.pattern_relation = re.compile(
            '[帮给为替]+我*的*([父亲|爹|爸爸|本人|自己|母亲|妈妈|爱人|老婆|妻子|丈夫|老公|儿子|孩子|小孩|小孩儿|女儿|闺女|家人|亲人|亲戚|哥|姐|妹|弟|朋友|姑爷|姑|叔叔|舅|姨|爷|奶]+)')
        self.pattern_age = re.compile('([0-9|一二三四五六七八九十]+)[岁啦了]*')
        self.pattern_sex = re.compile('(男|女|他|她)')
        self.pattern_health = re.compile('(没有|从未|未曾|否)[患过|得过|得了|重大疾病|癌]*')
        self.pattern_unhealth = re.compile('[曾原在自]*.*?(患过|得过|得了|有过)+([^重大疾病]+)')
        print('----------------------------------------------------')

    def fillSlotAll(self, msg):
        msg = str(msg)

        result_relation = re.findall(self.pattern_relation, msg)
        if len(result_relation) != 0:
            self.slot_information['relation'] = result_relation[0]

        result_income = re.findall(self.pattern_income, msg)
        if len(result_income) != 0:
            self.slot_information['income'] = self.num_change(result_income[0][1])

        result_age = re.findall(self.pattern_age, msg)
        if len(result_age) != 0:
            self.slot_information['age'] = self.num_change(result_age[0])

        result_occ = re.findall(self.pattern_occupation, msg)
        if len(result_occ) != 0:
            self.slot_information['occupation'] = result_occ[0]

        result_sex = self.judgeSex(msg)
        if result_sex is not None:
            self.slot_information['sex'] = result_sex

        result_sick = re.findall(self.pattern_unhealth, msg)
        if len(result_sick) != 0:
            self.slot_information['sick'] = result_sick[0][1]

    def ask(self):
        slot_keys = self.slot_information.keys()
        todels = []
        for each in slot_keys:
            print(each)
            if self.slot_information[each] != '':
                print(self.slot_information[each])
                todels.append(each)
        ret_list = list(set(slot_keys) ^ set(todels))

        if len(ret_list) != 0:
            self.last_ask = ret_list[0]
            result = np.random.choice(self.ask_information[ret_list[0]])
            return result
        else:
            return ''

    def process(self, msg):
        msg = str(msg)
        print(self.last_ask)
        if self.last_ask == '':
            self.fillSlotAll(msg)
            print('first')

        elif self.last_ask == 'relation':
            print('process realtion')
            result_relation = re.findall(self.pattern_relation, msg)
            if len(result_relation) != 0:
                self.slot_information['relation'] = result_relation[0]
        elif self.last_ask == 'age':
            print('process age')
            result_age = re.findall(self.pattern_age, msg)
            if len(result_age) != 0:
                self.slot_information['age'] = self.num_change(result_age[0])
        elif self.last_ask == 'sex':
            print('process sex')
            result_sex = self.judgeSex(msg)
            if result_sex is not None:
                self.slot_information['sex'] = result_sex
        elif self.last_ask == 'occupation':
            print('process occupation')
            result_occ = re.findall(self.pattern_occupation, msg)
            if len(result_occ) != 0:
                self.slot_information['occupation'] = result_occ[0]
        elif self.last_ask == 'income':
            print('process income')
            result_income = re.findall(self.pattern_income, msg)
            if len(result_income) != 0:
                self.slot_information['income'] = self.num_change(result_income[0][1])
        elif self.last_ask == 'sick':
            print('process sick')
            result_sick = re.findall(self.pattern_unhealth, msg)
            if len(result_sick) != 0:
                self.slot_information['sick'] = result_sick[0][1]
            else:
                self.slot_information['sick'] = '无'

        for each in self.slot_information.keys():
            print('槽位%s:%s' % (each, self.slot_information[each]))
        if '' in self.slot_information.values():
            return self.ask()
        else:
            if int(self.slot_information['age']) < 18:
                if int(self.slot_information['income']) < 100000:
                    return self.set_meal('A')
                else:
                    return self.set_meal('B')
            elif int(self.slot_information['age']) >= 18 and int(self.slot_information['age']) < 60:
                bot = ProfessionBot()
                mixContent = {"type": "Text", "content": "%s是什么类别啊" % self.slot_information['occupation']}
                print(mixContent['content'])

                category = bot.output(mixContent)

                if self.slot_information['sex'] == 'woman':
                    if category < 3:
                        return self.set_meal('C')
                    else:
                        return self.set_meal('D')
                else:
                    if category < 3:
                        return self.set_meal('E')
                    else:
                        return self.set_meal('F')
            else:
                return self.set_meal('G')

    def num_change(self, sentence):
        chinese = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
                   "壹": 1, "贰": 2, "叁": 3, "肆": 4, "伍": 5, "陆": 6, "柒": 7, "捌": 8, "玖": 9,
                   "两": 2, "半": 0.5, "多": 0.5}

        units = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000, "元": 1, "岁": 1, "块": 1,
                 "k": 1000, "w": 10000, 'K': 1000, "W": 10000}
        result = 0
        num = 1
        num_changed = False
        unit = [100000001, ]
        num_s = ""
        for i in str(sentence):
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

    def judgeSex(self, sentence):
        sentence = str(sentence)
        sex_dict = ['男', '父', '爸', '爹', '叔', '伯', '爷', '舅', '兄', '弟', '儿子', '他', '哥', '丈夫', '公', '他',
                    '母', '妈', '姑', '姨', '奶', '姐', '妹', '女儿', '她', '婶', '婆', '闺', '女性', '女']
        for each in sex_dict:
            if each in sentence:
                num = sex_dict.index(each) + 1
                if num <= 15:
                    return 'man'
                if num > 15:
                    return 'woman'

    def set_meal(self, tag):
        if tag == 'A':
            return '重疾险，保期30 年，保额30 万；\n意外险，保期1 年，保额20 万；\n医疗险，保期1 年，保额50 万；\n专项重疾险，保期1 年，保额30 万；\n教育金保险，保期30 年，保额50 万'

        if tag == 'B':
            return '重疾险，保期30 年，保额50 万；意外险，保期1 年，保额50 万；\n医疗险，保期1 年，保额50 万；\n专项重疾险，保期1 年，保额50 万；\n教育金保险，保期30 年，保额50 万'

        if tag == 'C':
            return '寿险，保期30 年；\n重疾险，保期终身；\n意外险，保期1 年，保额30 万；\n中端医疗险，保期1 年；\n专项女性险，保期1 年'

        if tag == 'D':
            return '寿险，保期30 年；\n重疾险，保期终身；\n意外险，保期1 年，保额50 万；\n中端医疗险，保期1 年；\n专项女性险，保期1 年'

        if tag == 'E':
            return '寿险，保期30 年；\n重疾险，保期终身；\n意外险，保期1 年，保额30 万；\n中端医疗险，保期1 年'
        if tag == 'F':
            return '寿险，保期30 年；\n重疾险，保期终身；\n意外险，保期1 年，保额50 万；\n中端医疗险，保期1 年'
        if tag == 'G':
            return '意外险，保期1 年，保额30 万；\n防癌险，保期终身，保额50 万'


if __name__ == '__main__':
    bot = MultiDialogBot()
    print(bot.slot_information.keys())
    result = bot.ask()
    print(result)
