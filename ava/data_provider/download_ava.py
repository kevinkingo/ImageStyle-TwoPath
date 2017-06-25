import csv
import requests
import sys, time
import threading
import os

savepath = 'images/'
if not os.path.exists(savepath):
    os.makedirs(savepath)

thread_total = 8
time_step = 30

download_lists = []

def init():
    with open('ava_num.txt') as f:
        all_num = f.read().split('\n')
        global download_lists
        download_lists = [all_num[i::thread_total] for i in range(thread_total)]

class Download(threading.Thread):
    def __init__(self, thread_num, download_list):
        threading.Thread.__init__(self)
        self.thread_num = thread_num
        self.download_list = download_list
        self.download_length = len(self.download_list)

        self.num = 0
        self.error_list = []

    def run(self):
        print "Thread " + str(self.thread_num) + " start!"

        for key in self.download_list:
            if os.path.isfile(savepath + key + ".jpg"):
                self.num += 1
                continue
            try:
                jsp = requests.get('http://www.dpchallenge.com/image.php?IMAGE_ID=' + key)
                middle_index = jsp.content.find('Copyrighted_Image_Reuse_Prohibited_' + key)
                pic_url = jsp.content[jsp.content.rfind('\"', 0, middle_index)+1:jsp.content.find('\"', middle_index)]
                #print pic_url
                r = requests.get(pic_url)
                with open(savepath + key + ".jpg", "wb") as f:
                    f.write(r.content)
            except:
                self.error_list.append(key)
            self.num += 1
                
        print "Thread " + str(self.thread_num) + " finish."

    def get_current_result(self):
        return self.num, self.download_length, self.error_list


init()
threads = {}
for i in range(0, thread_total):
    threads[i] = Download(i, download_lists[i])
    threads[i].start()

start_time = time.time()
stop_flag = 0
error_lists = {}
while stop_flag != ((1 << thread_total) - 1):
    last_time = time.time() - start_time
    if last_time % time_step == 0:
        print "************ Last time: " + str(last_time) + " ***************"
        for i in range(0, thread_total):
            cur_num, tol_num, error_list = threads[i].get_current_result()
            print "Thread " + str(i) + ": " + str(cur_num) + " / " + str(tol_num) + "\terror " + str(len(error_list))

            if cur_num == tol_num:
                error_lists[i] = error_list
                stop_flag |= (1 << i)

        print 
        sys.stdout.flush()
