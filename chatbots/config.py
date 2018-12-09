config_default = {
    'tuling_apikey': "9a1ad6a7b80747e9abfe1d8735c7e746",
    'tuling_apiurl': "http://openapi.tuling123.com/openapi/api/v2",

    # qiyu
    'qiyu_appkey': "1f2b874cb5bd2e2009a36b5a2d8d5214",
    'qiyu_appsecret': "97BAC5047DAA49DFBA6A2B38AC8232B4",
    'qiyu_apiurl': "https://qiyukf.com/openapi/message/send",
}

with open('D:\\PyCharm\\PyCharmProjects\\simba\\distinguish_words.txt', 'r', encoding='utf-8') as f:
    distinguish_words = f.readlines()
for i in range(len(distinguish_words)):
    distinguish_words[i] = (distinguish_words[i]).strip()


def get_config(bot_name=None):
    config = config_default
    if bot_name == 'tuling':
        config['tuling_apikey'] = "9a1ad6a7b80747e9abfe1d8735c7e746"
        config['tuling_apiurl'] = "http://openapi.tuling123.com/openapi/api/v2"
    else:
        config['tuling_apikey'] = "9a1ad6a7b80747e9abfe1d8735c7e746"
        config['tuling_apiurl'] = "http://openapi.tuling123.com/openapi/api/v2"

    return config


if __name__ == '__main__':
    print(get_config())
