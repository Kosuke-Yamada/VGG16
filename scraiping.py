#-*-coding:utf-8-*-

#Usage: python scraiping.py [keyword] [count] [directory]

import os
import sys
import traceback
from mimetypes import guess_extension
from time import time, sleep
from urllib.request import urlopen, Request
from urllib.parse import quote
from bs4 import BeautifulSoup

#URLから情報を取り出す
def Fetch(url):
    req = Request(url)
    try:
        with urlopen(req, timeout = 3) as file:
            content = file.read()
            mime = file.getheader('Content-Type')
    except:
        sys.stderr.write('Error in fetching {}\n'.format(url))
        sys.stderr.write(traceback.format_exc())
        return None, None
    return content, mime

#単語と画像番号を引数としてURLを取得する関数
def ImageUrlList(word, num_image, seed):
    url = "https://search.yahoo.co.jp/image/search?p={}&oq=&ei=UTF-8&xargs=1&b=".format(quote(word))+str(seed+num_image)
    content, _ = Fetch(url)
    page = BeautifulSoup(content.decode('UTF-8'),'html.parser')
    image_link_elems = page.find_all('img')
    #print(image_link_elems)
    image_urls = [e.get('src') for e in image_link_elems if e.get('src').startswith('http')]
    image_urls = list(set(image_urls))
    return image_urls

#画像を保存する関数
def SaveImage(word, max, data_dir, seed):

    num_image = 0
    max_flag = 0
    while(max_flag == 0):
        for i, image_url in enumerate(ImageUrlList(word, num_image, seed)):
            sleep(0.1)
            image, mime = Fetch(image_url)
            if not image or not mime:
                continue
            ext = guess_extension(mime.split(';')[0])
            if ext in ('.jpe', '.jpeg'):
                ext = '.jpg'
            if not ext:
                continue
            if ext!='.jpg':
                continue
            result_file = os.path.join(data_dir, str(word)+str(seed+num_image).zfill(3) + ext)
            with open(result_file, mode = 'wb') as f:
                f.write(image)
                
            #print('fetched:'+str(num_image))
            num_image += 1
            if num_image == max:
                max_flag = 1
                break
 
#メイン関数
if __name__ == '__main__':
    word = sys.argv[1] #検索キーワード
    max = int(sys.argv[2]) #取得枚数
    seed = int(sys.argv[3]) #シード
    data_dir = './Data/'+sys.argv[4]+"/"+str(word)+"/" #train / validation / test
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    SaveImage(word, max, data_dir, seed)
