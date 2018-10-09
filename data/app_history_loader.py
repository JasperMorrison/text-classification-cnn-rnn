# coding: utf-8

import sys,os,time
from collections import Counter
import datetime

import numpy as np
import tensorflow.contrib.keras as kr

base_dir = 'data/cnews_app_history'
pkgs_id_file = 'data/app_info/app_pkgs.txt'
holidy_file = 'data/app_info/holiday.csv'
global_index_file = 'data/app_info/indexs.txt'
care_actions = ["MOVE_TO_FOREGROUND"]
history_num = 5
other_num = 2
pkgs = {}
train_per = 0.8
test_per = 0.1
val_per = 0.1

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False


def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)

def insert_queue(q, l):
    q.append(l)
    if(len(q) > history_num):
        q.pop(0)

def list_all_pkgs():
    fs = os.listdir(base_dir)
    pkgs = []
    for i in fs:
        path = base_dir + "/" + i
        if os.path.isfile(path) and i.isdigit():
            with open(path, 'r+') as f:
                f.next()
                for l in f:
                    s_l = l.strip()
                    if len(s_l) > 0:
                        pkg = s_l.split('\t')[0]
                        pkgs.append(pkg)
    all = Counter(pkgs).most_common()
    pkgs,_ = list(zip(*all))
    
    return pkgs
    
def build_pkg_id(train=False):
    pkgs = []
    if train: #Every train time remove the ids file
        os.remove(pkgs_id_file)
    if not os.path.exists(pkgs_id_file):
        with open_file(pkgs_id_file, 'w') as f:
            pkgs = list_all_pkgs()
            f.write('\n'.join(pkgs) + '\n')
    else:
        with open_file(pkgs_id_file) as f:
            pkgs = [_.strip() for _ in f.readlines()]
    pkgs_id = dict(zip(pkgs, range(1, len(pkgs) + 1))) # 0 表示没有内容，所以这里不用0
    
    return pkgs_id


def action_to_id(a):
    return dict(zip(care_actions, range(len(care_actions))))[a]
    
def pkg_to_id(p):
    pkgs_id = build_pkg_id(False) # we are not sure
    return pkgs_id[p]

def parse_line(l):
    if l is None:
        return None,None,None
    line_list = l.split('\t')
    package = line_list[0]
    activity_class = line_list[1]
    time_locale = line_list[2]
    launch_date = line_list[3]
    launch_timestamp = line_list[4]
    action = line_list[5]
    if action not in care_actions:
        return None,None,None
    
    pkg_id = pkg_to_id(package)
    action_id = action_to_id(action)
    date = launch_date.split('"')[1]
    
    return pkg_id,date,action_id

def hour_in_day(date):
    return time.strptime(date, "%Y-%m-%d %H:%M:%S").tm_hour

holiday_list = []
def check_in_file(date, is_work_day):
    global holiday_list
    if os.path.exists(holidy_file):
        d_tmp = time.strptime(date, '%Y-%m-%d %H:%M:%S')
        d_str = "{0}/{1}/{2}".format(d_tmp.tm_year, d_tmp.tm_mon, d_tmp.tm_mday)
        if len(holiday_list) > 0:
            if d_str in holiday_list:
                is_work_day = False
            return is_work_day
            
        with open_file(holidy_file) as f:
            for line in f:
                line = line.strip()
                holiday_list.append(line)
                if line == d_str:
                    is_work_day = False
    return is_work_day

def is_work_day(date):
    is_work_day = False
    day = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S').weekday()
    if day in range(5):
        is_work_day = True
    else:
        is_work_day = False
        
    is_work_day = check_in_file(date, is_work_day)
    
    return is_work_day

def put_hot(apps):
    r = []
    for app in apps:
        r.append([app, 0]) # no hot data now
    return r
    
def parse_history(q, l, num_classes):
    pkg_id,date,action_id = parse_line(l)
    if pkg_id is None:
        return None,None
    pre_pkg_ids = []
    for i in q:
        pre_pkg_id, pre_time, pre_action_id = parse_line(i)
        if pre_pkg_id is None:
            pre_pkg_ids.append(0)
        else:
            pre_pkg_ids.append(pre_pkg_id)
    '''
    history = []
    history.append([hour_in_day(date),is_work_day(date)])
    history.extend(put_hot(pre_pkg_ids))
    '''
    
    history = []
    history.extend([hour_in_day(date),is_work_day(date)])
    history.extend(pre_pkg_ids)
    
    '''
    ids_pads = [0] * num_classes
    for id in pre_pkg_ids:
        ids_pads[id] += 1
    
    history = []
    history.extend([hour_in_day(date),is_work_day(date)])
    history.extend(ids_pads)
    '''
    
    return history, pkg_id

def read_files(dir, begin, end, num_classes):
    contents, labels = [], []
    queue = [None] * history_num
    list = os.listdir(dir)
    list.sort()
    begin = int(round(len(list) * begin))
    end = int(round(len(list) * end))
    list = list[begin:end]
    for i in list:
        path = dir + "/" + i
        if os.path.isfile(path) and i.isdigit():
            cs, ls = read_file(path, queue, num_classes)
            if len(cs) != 0:
                contents.extend(cs)
                labels.extend(ls)

    return contents, labels

def read_file(filename, queue, num_classes):
    """读取文件数据"""
    contents, labels = [], []

    with open_file(filename) as f:
        f.next() # do not need first line
        for line in f:
            line = line.strip()
            content, label = parse_history(queue, line, num_classes)
            if not content is None:
                contents.append(content)
                labels.append(label)
                insert_queue(queue, line)
    return contents, labels

def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')

def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id

def read_category():
    """读取分类目录，固定"""
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']

    categories = [native_content(x) for x in categories]

    # zip：转换成元组
    # dict：将元组转换成字典，类似：{'a': 0, 'c': 2, 'b': 1}
    # 这里的作用是将类别使用整数来表示
    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)

data_type = ["train", "test", "val"]
global_contents = []
global_labels = []
global_indices = []
global_xpad = []
global_ypad = []

def get_global_index(testing, pad):
    index = []
    if os.path.exists(global_index_file) and testing:
        with open_file(global_index_file) as f:
            index = eval(f.readline())
    else:
        index = np.random.permutation(np.arange(len(pad)))
        with open_file(global_index_file, 'w') as f:
            f.write('[')
            f.write(",".join(list(map(str, index))))
            f.write(']')
    return index

def process_history_file(dir, type, config):
    global data_type
    global global_contents
    global global_indices
    global global_labels
    global global_xpad
    global global_ypad
    
    if not type in data_type:
        raise ValueError('''Unknown data type''')
    
    if len(global_contents) == 0:
        global_contents, global_labels = read_files(dir, 0.0, 1.0, config.num_classes)
        global_xpad = kr.preprocessing.sequence.pad_sequences(global_contents, config.seq_length)
        global_ypad = kr.utils.to_categorical(global_labels, num_classes=config.num_classes)  # 将标签转换为one-hot表示
        global_indices = get_global_index(type == 'test', global_ypad)
        y_in = []
        for i in range(len(global_labels)):
            y = global_labels[i]
            if y in global_contents[i][-history_num:]:
                y_in.append(1)
            else:
                y_in.append(0)
        print 'In pre: %d, Not in pre: %d, In pre per: %.2f %%' % (y_in.count(1), len(y_in), y_in.count(1) * 1.0 / len(y_in) * 100)
    
    x_pad = global_xpad
    y_pad = global_ypad
    '''
    print "global xxxx ", global_contents[0:10]
    print "global yyy", global_labels[0:10]
    '''
    
    begin = 0.0
    file_per = train_per
    if type == "train":
        begin = 0.0
        file_per = train_per
    if type == "val":
        begin = train_per
        file_per = begin + val_per
    if type == "test":
        begin = train_per + val_per
        file_per = 1.0
    
    begin = int(round(len(global_indices) * begin))
    end = int(round(len(global_indices) * file_per))
    
    indices = global_indices[begin: end]
    x_shuffle = x_pad[indices]
    y_shuffle = y_pad[indices]
    
    return x_shuffle,y_shuffle

def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad

def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
