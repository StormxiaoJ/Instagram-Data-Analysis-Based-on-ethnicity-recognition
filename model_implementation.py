import requests
import json
import numpy as np
from lxml import etree
import re
import cv2
from io import BytesIO
from keras.models import load_model

# ====================================metadata=========================================================
headers = {
    "authority": "www.instagram.com",
    "scheme": "https",
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "accept-encoding": "gzip, deflate, br",
    "accept-language": "zh-CN,zh;q=0.9,en-AU;q=0.8,en;q=0.7,zh-TW;q=0.6",
    "cache-control": "max-age=0",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36"
}
pattern = r"^https://www.instagram.com"

image_header = {
    "authority": "scontent-syd2-1.cdninstagram.com",
    "scheme": "https",
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "accept-encoding": "gzip, deflate, br",
    "accept-language": "zh-CN,zh;q=0.9,en-AU;q=0.8,en;q=0.7,zh-TW;q=0.6",
    "cache-control": "max-age=0",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36"
}
image_pattern = r'https://scontent-syd2-1.cdninstagram.com'
mean = 0.05
var = 0.05
model_name = ".h5"


def get_image_url(link):
    try:
        if not re.match(pattern, link):
            print("url is invalid. {}".format(link))
            return None
        headers["Referer"] = link
        headers["path"] = link[len(pattern):]
        res = requests.get(link, headers=headers)
        # print(res.status_code)
        html = etree.HTML(res.content.decode())
        all_a_tags = html.xpath('//script[@type="text/javascript"]/text()')
        # print(len(all_a_tags))
        for a_tag in all_a_tags:
            if a_tag.strip().startswith('window._sharedData'):
                data = a_tag[a_tag.index("= {") + 2:-1]  # get json string
                js_data = json.loads(data, encoding='utf-8')
                image_url = js_data["entry_data"]["PostPage"][0]["graphql"]["shortcode_media"]["display_url"]
                return image_url
    except Exception as e:
        print(e.args)
        return None


from sklearn.preprocessing import StandardScaler

txt_name = "output.json"
fwrite = open(txt_name, "w")
correct_num = 0
error_num = 0
normalization_scaler = StandardScaler(with_mean=mean, with_std=var)
model = load_model(model_name)
with open("test.json", "r", encoding="utf-8") as f:
    for line in f:
        try:
            dict_tmp = eval(line)
            if 'coordinates' in dict_tmp:
                image_url = get_image_url(dict_tmp['link'])
                image_header['path'] = image_url[len(image_pattern):]
                res = requests.get(image_url, headers=image_header)
                image = face_recognition.load_image_file(BytesIO(res.content))
                face_locations = face_recognition.face_locations(image)
                if len(face_locations) == 0:
                    dict_tmp['people'] = 0
                    dict_tmp['ethnic'] = []
                else:
                    people=0
                    ethnic_result = []
                    for face in face_locations:
                        top = face[0]
                        right = face[1]
                        bottom = face[2]
                        left = face[3]
                        if bottom - top < 80 or right - left < 80:
                            pass
                        resizedImage = cv2.resize(image[top:bottom + 1, left:right + 1, :], (160, 160))
                        resizedImage = normalization_scaler.transform(resizedImage)
                        result = model.predict(resizedImage, batch_size=1)
                        ethnic_result.append(result)
                        people+=1
                    dict_tmp['people'] = people
                    dict_tmp['ethnic'] = ethnic_result[:]
                correct_num += 1

        except Exception as e:
            error_num += 1

print("correct_num : ", correct_num)
print("error_num : ", error_num)
