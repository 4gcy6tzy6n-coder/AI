"""
Stage 14.2.2: 扩充数据量，稳定生成输出

问题: Stage 14.2.1只有59条数据，模型死记硬背
目标: 扩充到200+条，让模型真正学到语义对应

数据配比:
- 直接回答类: ~50%
- 解释说明类: ~25%
- 步骤建议类: ~15%
- 澄清类: ~10%
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import random
from collections import Counter


SPECIAL_TOKENS = {
    '<PAD>': 0,
    '<UNK>': 1,
    '<BOS>': 2,
    '<EOS>': 3,
}

MAX_VOCAB = 5000
PAD_ID = SPECIAL_TOKENS['<PAD>']
UNK_ID = SPECIAL_TOKENS['<UNK>']
BOS_ID = SPECIAL_TOKENS['<BOS>']
EOS_ID = SPECIAL_TOKENS['<EOS>']


class SimpleBigramTokenizer:
    """简单bigram分词器"""

    def __init__(self):
        self.word2id = SPECIAL_TOKENS.copy()
        self.id2word = {v: k for k, v in SPECIAL_TOKENS.items()}
        self.next_id = len(SPECIAL_TOKENS)
        self.fitted = False

    def _tokenize(self, text: str) -> List[str]:
        tokens = []
        i = 0
        text = text.strip()
        while i < len(text):
            if i + 1 < len(text):
                bigram = text[i:i+2]
                tokens.append(bigram)
                i += 2
            else:
                tokens.append(text[i])
                i += 1
        return tokens

    def fit(self, texts: List[str]):
        word_freq = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            for t in tokens:
                if t.strip():
                    word_freq[t] += 1

        for word, freq in word_freq.most_common(MAX_VOCAB - len(SPECIAL_TOKENS)):
            self.word2id[word] = self.next_id
            self.id2word[self.next_id] = word
            self.next_id += 1

        self.fitted = True
        print(f"  词表大小: {len(self.word2id)}")

    def encode(self, text: str, max_len: int) -> List[int]:
        tokens = self._tokenize(text)
        ids = [BOS_ID]
        for t in tokens[:max_len-2]:
            ids.append(self.word2id.get(t, UNK_ID))
        ids.append(EOS_ID)

        if len(ids) < max_len:
            ids.extend([PAD_ID] * (max_len - len(ids)))

        return ids

    def decode(self, ids: List[int]) -> str:
        words = []
        for i in ids:
            if i == EOS_ID:
                break
            if i in (PAD_ID, BOS_ID):
                continue
            if i in self.id2word:
                words.append(self.id2word[i])
            else:
                words.append('<UNK>')
        return ''.join(words)


# ============ 扩充后的数据集 ============

# 直接回答类 ~100条
DIRECT_ANSWERS = [
    {"query": "Python是什么", "response": "Python是一种高级编程语言，由Guido van Rossum发明。它语法简洁、易学易用，广泛用于Web开发、数据分析、人工智能等领域。"},
    {"query": "人工智能是什么", "response": "人工智能是计算机科学的一个分支，致力于开发能够模拟人类智能的技术。包括机器学习、自然语言处理、计算机视觉等多个研究方向。"},
    {"query": "机器学习是什么", "response": "机器学习是人工智能的一个子领域，通过让计算机从数据中学习模式来进行预测和决策，而不需要明确的编程规则。"},
    {"query": "深度学习是什么", "response": "深度学习是机器学习的子集，使用多层神经网络来处理复杂数据。它在图像识别、自然语言处理等领域取得了突破性进展。"},
    {"query": "区块链是什么", "response": "区块链是一种分布式账本技术，通过加密链条将数据块连接起来。它具有去中心化、不可篡改的特点。"},
    {"query": "云计算是什么", "response": "云计算是一种通过互联网提供计算资源的服务模式。用户可以按需使用服务器、存储、数据库等资源。"},
    {"query": "什么是API", "response": "API即应用程序编程接口，是不同软件之间通信的桥梁。通过API，一个程序可以调用另一个程序的功能。"},
    {"query": "什么是HTTP协议", "response": "HTTP是超文本传输协议，是Web通信的基础。它是请求-响应协议，常见方法有GET、POST、PUT、DELETE。"},
    {"query": "什么是数据库", "response": "数据库是存储和管理数据的系统。常见的数据库类型包括关系型数据库和非关系型数据库。"},
    {"query": "什么是递归", "response": "递归是一种编程技巧，函数调用自身来解决问题。递归需要基本情况和递归情况两个条件。"},
    {"query": "你好", "response": "你好！有什么我可以帮助你的吗？"},
    {"query": "今天天气怎么样", "response": "今天天气不错，阳光明媚，适合外出活动。"},
    {"query": "再见", "response": "再见，有需要随时找我！"},
    {"query": "谢谢", "response": "不客气！很高兴能帮到你。"},
    {"query": "你是谁", "response": "我是一个AI助手，可以回答问题、提供信息、帮助编程等。"},
    {"query": "一杯咖啡多少钱", "response": "咖啡价格因地区和店铺而异，一般在20到50元之间。"},
    {"query": "今天星期几", "response": "今天是星期三。"},
    {"query": "北京在哪里", "response": "北京是中国的首都，位于华北平原北部。"},
    {"query": "太阳系有几颗行星", "response": "太阳系有八颗行星，分别是水星、金星、地球、火星、木星、土星、天王星和海王星。"},
    {"query": "水的沸点是多少", "response": "在一个标准大气压下，水的沸点是100摄氏度。"},
    {"query": "什么是操作系统", "response": "操作系统是管理计算机硬件和软件资源的程序。它为用户提供与计算机交互的界面，并管理进程、内存、文件系统等。"},
    {"query": "什么是编译器", "response": "编译器是将高级语言代码翻译成机器语言的程序。它进行词法分析、语法分析、语义分析，生成目标代码。"},
    {"query": "什么是解释器", "response": "解释器是直接执行代码的程序，不需要编译。它逐行读取源代码并立即执行。"},
    {"query": "什么是变量", "response": "变量是存储数据的容器，每个变量有名字和值。在程序中可以随时修改变量的值。"},
    {"query": "什么是函数", "response": "函数是一段可重用的代码，接受输入并返回输出。它帮助代码组织、模块化和复用。"},
    {"query": "什么是数组", "response": "数组是相同类型元素的集合，通过索引访问。数组在内存中是连续存储的。"},
    {"query": "什么是链表", "response": "链表是线性数据结构，每个节点包含数据和指向下一个节点的指针。链表适合频繁的插入删除操作。"},
    {"query": "什么是树结构", "response": "树是层次数据结构，由节点组成，每个节点有零个或多个子节点。二叉树是最常见的类型。"},
    {"query": "什么是图结构", "response": "图由顶点和边组成，用于表示对象之间的关系。图分为有向图和无向图。"},
    {"query": "什么是哈希表", "response": "哈希表使用哈希函数将键映射到值，实现快速查找。它的时间复杂度接近O(1)。"},
    {"query": "什么是排序算法", "response": "排序算法是将数据按特定顺序排列的算法。常见的有冒泡排序、快速排序、归并排序等。"},
    {"query": "什么是快速排序", "response": "快速排序采用分治策略，选择基准元素将数组分区，然后递归排序。平均时间复杂度是O(nlogn)。"},
    {"query": "什么是二分查找", "response": "二分查找是在有序数组中查找目标值的算法。每次将搜索范围缩小一半，时间复杂度是O(logn)。"},
    {"query": "什么是动态规划", "response": "动态规划是将复杂问题分解为子问题，通过存储子问题结果避免重复计算。适用于最优子结构问题。"},
    {"query": "什么是贪心算法", "response": "贪心算法在每一步选择当前最优解，期望获得全局最优。适用于局部最优能导致全局最优的问题。"},
    {"query": "什么是回溯算法", "response": "回溯算法通过尝试不同的选择来解决问题，当路径不通时回退到上一步。用于搜索和组合问题。"},
    {"query": "什么是并发", "response": "并发是指多个任务在同一时间段内执行。操作系统通过时间片轮转实现并发执行。"},
    {"query": "什么是多线程", "response": "多线程是一个进程内有多个执行流。线程共享进程资源，适合需要并行处理的任务。"},
    {"query": "什么是锁", "response": "锁是用于控制多个线程访问共享资源的同步机制。常见的有互斥锁、读写锁等。"},
    {"query": "什么是死锁", "response": "死锁是多个线程相互等待对方释放资源而无法继续执行。避免死锁需要破坏四个必要条件。"},
    {"query": "什么是缓存", "response": "缓存是存储频繁访问数据的快速存储区域。它可以显著提高数据访问速度。"},
    {"query": "什么是CDN", "response": "CDN即内容分发网络，通过在全球部署服务器来加速内容分发，减少用户访问延迟。"},
    {"query": "什么是Docker", "response": "Docker是容器化平台，将应用及其依赖打包成容器，实现一致的运行环境。"},
    {"query": "什么是Kubernetes", "response": "Kubernetes是容器编排平台，用于自动化部署、扩展和管理容器化应用。"},
    {"query": "什么是微服务", "response": "微服务是一种架构风格，将应用拆分为多个小型、独立的服务，每个服务实现特定功能。"},
    {"query": "什么是RESTful API", "response": "RESTful API是一种设计风格，使用HTTP方法对资源进行操作。它简洁、易于理解和扩展。"},
    {"query": "什么是GraphQL", "response": "GraphQL是一种API查询语言，允许客户端精确指定需要的数据，减少数据传输量。"},
    {"query": "什么是Git", "response": "Git是分布式版本控制系统，用于跟踪代码变更、协作开发和代码管理。"},
    {"query": "什么是GitHub", "response": "GitHub是基于Git的代码托管平台，提供版本控制、问题跟踪和协作功能。"},
    {"query": "什么是CI/CD", "response": "CI/CD是持续集成和持续部署的实践，自动化代码构建、测试和部署流程。"},
    {"query": "什么是单元测试", "response": "单元测试是对软件最小单元进行测试的实践，确保每个单元按预期工作。"},
    {"query": "什么是代码审查", "response": "代码审查是团队成员检查彼此代码的过程，可以发现bug、提高代码质量和分享知识。"},
    {"query": "中国首都是哪里", "response": "中国首都是北京。北京是中国的政治、文化和国际交往中心。"},
    {"query": "世界上最高的山是什么", "response": "珠穆朗玛峰是世界最高峰，海拔8848.86米，位于中国和尼泊尔边境。"},
    {"query": "光速是多少", "response": "光速是299792458米/秒，约30万公里每秒。"},
    {"query": "地球到月球的距离", "response": "地球到月球的平均距离约38.4万公里。"},
    {"query": "什么是维生素C", "response": "维生素C是人体必需的水溶性维生素，具有抗氧化作用，有助于增强免疫力和促进铁吸收。"},
    {"query": "什么导致感冒", "response": "感冒通常由鼻病毒等病毒引起，通过飞沫或接触传播。症状包括流鼻涕、咳嗽、喉咙痛等。"},
    {"query": "如何预防感冒", "response": "预防感冒的方法包括勤洗手、避免接触病人、保持充足睡眠、均衡饮食和适度运动。"},
    {"query": "每天应该喝多少水", "response": "成年人每天建议饮水约2升，具体因体重、活动量和气候而异。"},
    {"query": "什么运动最减肥", "response": "有氧运动如跑步、游泳、骑行等对减肥最有效。结合力量训练效果更好。"},
    {"query": "早餐应该吃什么", "response": "健康的早餐应包含蛋白质、碳水化合物和蔬果，如鸡蛋、全麦面包和水果。"},
    {"query": "如何缓解压力", "response": "缓解压力的方法包括运动、冥想、深呼吸、与朋友交流和保证充足睡眠。"},
    {"query": "什么是人工智能的图灵测试", "response": "图灵测试是评估机器是否具有智能的测试，如果人类无法区分机器和人类的回答，则通过测试。"},
    {"query": "什么是量子计算", "response": "量子计算利用量子力学原理进行计算，使用量子比特，可以同时表示0和1的叠加态。"},
    {"query": "什么是5G网络", "response": "5G是第五代移动通信技术，比4G速度快10-100倍，延迟更低，支持更多设备连接。"},
    {"query": "什么是物联网", "response": "物联网是将物理设备通过传感器和网络连接起来，实现数据交换和智能控制。"},
    {"query": "什么是大数据", "response": "大数据指体量大、速度快、多样性高的数据。需要特殊技术来存储、处理和分析。"},
    {"query": "什么是机器视觉", "response": "机器视觉是让机器通过摄像头和算法理解图像和视频的技术，应用于人脸识别、自动驾驶等。"},
    {"query": "什么是自然语言处理", "response": "自然语言处理是让计算机理解、生成和处理人类语言的技术，包括文本分析、机器翻译等。"},
    {"query": "什么是强化学习", "response": "强化学习是让智能体通过与环境交互学习最优策略，以最大化累积奖励。"},
    {"query": "什么是神经网络", "response": "神经网络是受人脑启发的计算模型，由相互连接的神经元组成，可以学习复杂的模式。"},
    {"query": "什么是卷积神经网络", "response": "卷积神经网络是处理图像的神经网络，使用卷积层自动提取图像特征，广泛用于图像识别。"},
    {"query": "什么是循环神经网络", "response": "循环神经网络适合处理序列数据，可以记住之前的信息，用于自然语言处理和时间序列预测。"},
    {"query": "什么是Transformer", "response": "Transformer是注意力机制模型，突破了序列建模的瓶颈，成为NLP和许多其他领域的主流架构。"},
    {"query": "什么是BERT", "response": "BERT是基于Transformer的预训练语言模型，通过双向编码器表示，在多项NLP任务上取得突破。"},
    {"query": "什么是GPT", "response": "GPT是基于Transformer的生成式预训练模型，通过大规模文本预训练实现强大的语言生成能力。"},
    {"query": "Python谁发明的", "response": "Python由Guido van Rossum于1991年发明。他也被称为Python之父。"},
    {"query": "JavaScript用于什么", "response": "JavaScript主要用于Web前端开发，也可以用于后端开发。"},
    {"query": "什么是CSS", "response": "CSS用于控制网页的样式和布局，实现HTML内容与表现的分离。"},
    {"query": "什么是HTML", "response": "HTML是超文本标记语言，用于创建和结构化网页内容。"},
    {"query": "什么是SQL", "response": "SQL是结构化查询语言，用于管理关系数据库，进行数据的增删改查操作。"},
    {"query": "什么是NoSQL", "response": "NoSQL是非关系型数据库，适合处理非结构化数据和大规模分布式数据存储。"},
    {"query": "什么是MongoDB", "response": "MongoDB是流行的NoSQL数据库，使用文档式存储，数据以JSON格式保存。"},
    {"query": "什么是Redis", "response": "Redis是内存数据库，常用作缓存和消息队列，支持多种数据结构。"},
    {"query": "什么是Elasticsearch", "response": "Elasticsearch是基于Lucene的搜索引擎，支持全文搜索和实时分析。"},
    {"query": "什么是负载均衡", "response": "负载均衡将请求分配到多个服务器，避免单点过载，提高系统可用性和响应速度。"},
    {"query": "什么是反向代理", "response": "反向代理代理客户端向服务器发起请求，隐藏真实服务器，增强安全性和性能。"},
    {"query": "什么是SSL证书", "response": "SSL证书用于加密网站与浏览器之间的通信，保护数据传输安全。"},
    {"query": "什么是防火墙", "response": "防火墙监控和控制网络流量，根据规则允许或阻止数据传输，保护网络安全。"},
    {"query": "什么是VPN", "response": "VPN即虚拟专用网络，在公共网络上建立加密通道，实现安全远程访问。"},
    {"query": "什么是IP地址", "response": "IP地址是分配给网络设备的唯一标识，用于在网络中定位设备。"},
    {"query": "什么是域名", "response": "域名是网站的易记名称，如example.com，通过DNS解析为IP地址。"},
    {"query": "什么是DNS", "response": "DNS即域名系统，将域名解析为IP地址，是互联网的基础设施。"},
]

# 解释说明类 ~50条
EXPLANATIONS = [
    {"query": "解释Python中的列表和元组的区别", "response": "列表和元组的主要区别：1.列表使用方括号，元组使用圆括号；2.列表是可变的，元组是不可变的；3.列表适合需要修改的数据，元组适合固定数据。"},
    {"query": "什么是TCP和UDP协议", "response": "TCP和UDP都是传输层协议。TCP是面向连接的可靠传输，提供流量控制和拥塞控制；UDP是无连接的不可靠传输，不保证数据到达但速度快。"},
    {"query": "深度学习为什么需要GPU", "response": "深度学习需要GPU的原因：1.GPU有大量并行计算单元，适合矩阵运算；2.神经网络训练涉及大量矩阵乘法，GPU比CPU快数十倍。"},
    {"query": "什么是过拟合，如何避免", "response": "过拟合是模型过于复杂，在训练数据上表现很好但在新数据上表现差。避免方法包括增加训练数据、使用正则化、简化模型、使用Dropout。"},
    {"query": "面向对象编程的三大特性", "response": "面向对象编程的三大特性：1.封装：将数据和操作封装在类中；2.继承：子类继承父类的属性和方法；3.多态：不同对象对同一消息有不同的响应。"},
    {"query": "为什么天空是蓝色的", "response": "天空呈蓝色是因为大气层对阳光的散射效应。蓝光波长较短，更容易被空气分子散射，所以我们看到的天空是蓝色的。"},
    {"query": "为什么冬天会下雪", "response": "冬天下雪是因为气温降低到零度以下，空气中的水蒸气直接凝结成冰晶并形成雪花飘落。"},
    {"query": "为什么电脑需要内存", "response": "内存是电脑用于临时存储数据和程序的地方，它比硬盘读写速度快得多，让CPU能快速访问正在运行的程序和数据。"},
    {"query": "为什么海水是咸的", "response": "海水是咸的是因为河流携带的矿物质溶解进入海洋，蒸发后留下盐分，经过数十亿年积累形成现在的盐度。"},
    {"query": "为什么会有四季", "response": "四季形成是因为地球自转轴倾斜23.5度，地球围绕太阳公转时，不同位置接收的阳光角度和时长不同，导致温度变化。"},
    {"query": "为什么飞机能飞", "response": "飞机能飞是因为机翼的特殊形状使气流在上下表面产生压力差，形成向上的升力。当升力大于重力时，飞机就能起飞。"},
    {"query": "为什么手机能打电话", "response": "手机通话原理：声音转化为电信号，通过天线以无线电波发射，基站接收并传输到对方手机，再还原为声音。"},
    {"query": "为什么照片会曝光", "response": "照片曝光是因为光线在感光元件上停留时间过长。光线进入过多会导致照片过亮，感光不足则过暗。"},
    {"query": "为什么冰会浮在水上", "response": "冰比水轻所以会浮在水面上。这是因为水分子在凝固时形成晶体结构，密度比液态水小约9%。"},
    {"query": "为什么磁铁能吸引铁", "response": "磁铁能吸引铁是因为铁内部有磁畴，磁铁产生的磁场使铁的磁畴排列整齐，从而产生吸引力。"},
    {"query": "为什么火焰往上跑", "response": "火焰往上是因为热空气比冷空气轻，冷空气下沉推动热空气上升，这就是热对流原理。"},
    {"query": "为什么电池有正负极", "response": "电池正负极是电子流出和流入的两个端口。电流从正极流出，经过外部电路回到负极，形成完整的电流通路。"},
    {"query": "为什么指南针能指方向", "response": "指南针内部的小磁针受地球磁场影响，磁针的北极被地球磁场的南极吸引，所以会指向北方。"},
    {"query": "为什么声音在固体中传得快", "response": "声音在固体中传播快是因为固体分子间距小，振动容易传递。声音在钢铁中比在空气中快约17倍。"},
    {"query": "为什么镜子能照出人影", "response": "镜子背面镀有金属层，能反射大部分光线。当光线遇到镜面时反射回来，进入眼睛就看到了影像。"},
    {"query": "什么是二叉搜索树", "response": "二叉搜索树是特殊的二叉树，左子树所有节点小于根节点，右子树所有节点大于根节点。这使得搜索效率是O(logn)。"},
    {"query": "什么是堆排序", "response": "堆排序使用堆这种数据结构。堆是完全二叉树，父节点总是大于或小于子节点。建堆后反复提取最值并调整堆结构完成排序。"},
    {"query": "什么是归并排序", "response": "归并排序采用分治策略，将数组递归分割成单个元素，然后按大小合并。稳定且时间复杂度总是O(nlogn)。"},
    {"query": "什么是选择排序", "response": "选择排序每趟找出剩余元素的最小值放到已排序序列末尾。简单直观但时间复杂度是O(n²)，不适合大数据集。"},
    {"query": "什么是插入排序", "response": "插入排序类似整理扑克牌，将每个元素插入到已排序序列的正确位置。小规模数据或基本有序时效率高。"},
    {"query": "什么是冒泡排序", "response": "冒泡排序重复比较相邻元素并交换位置，像气泡一样将最大元素冒到序列末端。时间复杂度O(n²)，适合初学者学习。"},
    {"query": "什么是时间复杂度", "response": "时间复杂度表示算法执行时间与输入规模的增长关系。常用大O表示法，如O(1)、O(logn)、O(n)、O(nlogn)、O(n²)等。"},
    {"query": "什么是空间复杂度", "response": "空间复杂度表示算法占用内存空间与输入规模的增长关系。包括固定部分和可变部分，如递归栈空间和动态分配的内存。"},
    {"query": "什么是算法稳定性", "response": "算法稳定性指排序后相等元素的相对顺序是否保持不变。稳定的排序算法如归并排序、不稳定的如快速排序。"},
    {"query": "什么是CAP定理", "response": "CAP定理指出分布式系统无法同时满足一致性、可用性和分区容错性。设计师需要在这三者之间权衡选择。"},
    {"query": "什么是ACID", "response": "ACID是数据库事务的四个特性：原子性、一致性、隔离性、持久性。确保数据库操作可靠执行。"},
    {"query": "什么是BASE理论", "response": "BASE理论是对CAP定理的实践延伸，包括基本可用、软状态和最终一致性。适用于分布式系统设计。"},
    {"query": "什么是拜占庭将军问题", "response": "拜占庭将军问题讨论分布式系统中存在恶意节点时如何达成共识。是区块链和分布式系统的重要理论基础。"},
    {"query": "什么是Paxos算法", "response": "Paxos是分布式共识算法，用于在存在故障的分布式系统中达成一致性决议。包含准备阶段和接受阶段。"},
    {"query": "什么是Raft算法", "response": "Raft是分布式共识算法，相比Paxos更易于理解和实现。包括leader选举、日志复制和安全性保证。"},
    {"query": "什么是两阶段提交", "response": "两阶段提交是分布式事务协议。协调者先询问参与者是否准备好提交，第一阶段投票，第二阶段执行最终提交。"},
    {"query": "什么是乐观锁", "response": "乐观锁假设并发冲突较少，更新时检查数据是否被修改。常用版本号或CAS操作实现，失败则重试。"},
    {"query": "什么是悲观锁", "response": "悲观锁假设并发冲突较多，先锁定数据再操作。常见实现是数据库的行锁或表锁，会影响并发性能。"},
    {"query": "什么是读写锁", "response": "读写锁区分读和写操作。读操作可以并发进行，写操作需要独占访问。适合读多写少场景，提高并发性能。"},
    {"query": "什么是线程池", "response": "线程池预先创建一组线程，用任务队列管理要执行的工作。避免频繁创建销毁线程，减少系统开销。"},
    {"query": "什么是生产者消费者模式", "response": "生产者消费者模式通过队列解耦生产者和消费者。生产者发送消息，消费者异步处理，实现系统间松耦合。"},
    {"query": "什么是发布订阅模式", "response": "发布订阅模式中，发布者发送消息到主题，订阅者接收该主题的所有消息。实现一对多的消息分发。"},
    {"query": "什么是观察者模式", "response": "观察者模式定义对象间一对多依赖关系，当对象状态改变时，所有依赖它的对象都会收到通知。"},
    {"query": "什么是单例模式", "response": "单例模式确保一个类只有一个实例，并提供全局访问点。常用于配置管理、连接池等场景。"},
    {"query": "什么是工厂模式", "response": "工厂模式定义创建对象的接口，让子类决定实例化哪个类。将对象创建与使用分离，提高灵活性。"},
    {"query": "什么是策略模式", "response": "策略模式定义一系列算法，将每个算法封装起来并可互换。适合需要多种算法选择的场景。"},
    {"query": "什么是适配器模式", "response": "适配器模式将不兼容接口转换为期望的接口，让原本不能合作的类可以合作。"},
    {"query": "什么是装饰器模式", "response": "装饰器模式动态给对象添加额外职责。比继承更灵活，可以随时添加或撤销功能。"},
]

# 步骤建议类 ~35条
STEPS_SUGGESTIONS = [
    {"query": "怎么学习编程", "response": "学习编程建议：1.选择一门容易入门的语言如Python；2.每天坚持编码实践；3.阅读优秀开源项目的代码；4.尝试自己实现小项目。"},
    {"query": "如何入门深度学习", "response": "深度学习入门建议：1.掌握Python基础；2.学习机器学习基本概念；3.了解神经网络原理；4.跟随教程实现简单模型。"},
    {"query": "怎么入门Python", "response": "Python入门步骤：1.安装Python环境；2.学习基本语法和数据类型；3.练习编写简单程序；4.学习函数和模块；5.做小项目巩固。"},
    {"query": "如何准备面试", "response": "面试准备建议：1.复习核心技术知识；2.练习算法题；3.准备项目经验的描述；4.模拟面试场景；5.保持良好心态。"},
    {"query": "怎么减肥", "response": "健康减肥建议：1.控制饮食，减少高热量食物；2.增加运动量，每周进行有氧运动；3.保持充足睡眠；4.坚持记录饮食和体重。"},
    {"query": "怎么学英语", "response": "英语学习建议：1.每天背单词；2.多听多说多读多写；3.看英文电影和新闻；4.用英语写日记；5.找语伴练习口语。"},
    {"query": "如何学好数学", "response": "数学学习建议：1.理解概念而非死记公式；2.多做练习题巩固；3.建立知识体系；4.学会思考解题思路；5.及时复习错题。"},
    {"query": "怎么提高记忆力", "response": "提高记忆力方法：1.保证充足睡眠；2.使用记忆宫殿等技巧；3.分散学习而非突击；4.经常复习；5.保持健康生活方式。"},
    {"query": "如何克服拖延症", "response": "克服拖延症建议：1.将大任务分解为小步骤；2.设置明确截止时间；3.使用番茄工作法；4.消除干扰源；5.给自己适当奖励。"},
    {"query": "怎么做职业规划", "response": "职业规划步骤：1.自我评估，了解兴趣和能力；2.探索职业方向；3.设定短期和长期目标；4.制定行动计划；5.定期评估调整。"},
    {"query": "如何提高工作效率", "response": "提高工作效率方法：1.使用待办事项清单；2.优先处理重要任务；3.减少不必要的会议；4.善用工具自动化；5.适当休息保持精力。"},
    {"query": "怎么培养早起习惯", "response": "培养早起习惯：1.设定固定的起床时间；2.逐步提前作息；3.起床后晒太阳；4.避免晚上使用手机；5.给自己早起安排喜欢的事。"},
    {"query": "如何学会理财", "response": "理财入门建议：1.学习基本财务知识；2.记录收入支出；3.制定预算；4.开始储蓄和投资；5.分散风险。"},
    {"query": "怎么做自我介绍", "response": "自我介绍技巧：1.简洁明了控制在一分钟内；2.突出个人背景和技能；3.说明与岗位的匹配度；4.展现个人特色；5.练习到自然流畅。"},
    {"query": "如何写好简历", "response": "写简历建议：1.突出关键成就而非职责描述；2.使用量化数据展示成果；3.保持简洁一页纸；4.针对岗位定制；5.检查拼写语法错误。"},
    {"query": "怎么戒烟", "response": "戒烟方法：1.设定戒烟日期；2.了解戒烟好处；3.使用尼古丁替代品；4.培养新习惯替代吸烟；5.寻求家人朋友支持。"},
    {"query": "如何缓解颈椎病", "response": "缓解颈椎病：1.保持正确坐姿；2.每工作1小时活动颈椎；3.做颈椎操；4.用合适的枕头；5.严重时就医治疗。"},
    {"query": "怎么保护眼睛", "response": "保护眼睛方法：1.每用眼40分钟休息10分钟；2.保持适当屏幕距离；3.在光线充足处阅读；4.多进行户外活动；5.均衡饮食补充维生素。"},
    {"query": "如何提高免疫力", "response": "提高免疫力：1.保证充足睡眠7-8小时；2.均衡饮食多吃蔬果；3.适度运动；4.减少压力；5.保持良好卫生习惯。"},
    {"query": "怎么控制情绪", "response": "控制情绪方法：1.暂停深呼吸；2.识别情绪来源；3.换位思考；4.表达而非压抑情绪；5.必要时寻求专业帮助。"},
    {"query": "如何培养自信心", "response": "培养自信：1.设定并完成小目标；2.学习新技能；3.正视缺点接受不完美；4.保持积极自我对话；5.注意仪表体态。"},
    {"query": "怎么学会冥想", "response": "冥想入门：1.选择安静环境；2.设定5-10分钟；3.关注呼吸；4.当杂念出现时温柔带回；5.每天坚持练习。"},
    {"query": "如何改善睡眠质量", "response": "改善睡眠：1.固定作息时间；2.睡前避免咖啡因；3.营造黑暗安静环境；4.睡前放松活动；5.白天适度运动。"},
    {"query": "怎么做出美味的咖啡", "response": "做咖啡建议：1.选用新鲜咖啡豆；2.研磨度适中；3.控制水温90-96度；4.水粉比例1:15；5.新鲜萃取即时饮用。"},
    {"query": "如何在家锻炼身体", "response": "居家锻炼：1.准备瑜伽垫和哑铃；2.做俯卧撑、深蹲、平板支撑；3.跟随健身视频；4.每次30分钟；5.循序渐进增加强度。"},
    {"query": "怎么学习唱歌", "response": "学唱歌建议：1.先练呼吸和气息控制；2.模仿喜欢的歌手；3.练习音阶和音准；4.录下来听改进；5.找老师指导。"},
    {"query": "如何学会游泳", "response": "游泳入门：1.克服对水的恐惧；2.先练习呼吸和漂浮；3.学习自由泳腿部动作；4.配合手臂划水；5.保持节奏和呼吸配合。"},
    {"query": "怎么制作泡菜", "response": "制作泡菜：1.白菜切块撒盐腌制；2.洗净沥干；3.制作辣椒酱调料；4.均匀涂抹白菜；5.密封发酵3-5天。"},
    {"query": "如何种植多肉植物", "response": "种植多肉：1.选用透气排水好的土壤；2.放在阳光充足处；3.浇水遵循干透浇透原则；4.注意通风；5.繁殖可叶插或分株。"},
    {"query": "怎么修理漏水的水龙头", "response": "修理水龙头：1.关闭水源阀门；2.拆开水龙头把手；3.更换磨损的垫圈或阀芯；4.重新组装；5.打开水源测试。"},
    {"query": "如何更换汽车轮胎", "response": "换轮胎：1.安全停车拉手刹；2.用千斤顶抬起车辆；3.拧松螺母；4.拆下旧胎装上新胎；5.对角拧紧螺母。"},
    {"query": "怎么清洁电脑键盘", "response": "清洁键盘：1.关机断电；2.倒置轻拍去除杂物；3.用软布酒精擦拭；4.棉签清理缝隙；5.晾干后使用。"},
]

# 澄清类 ~15条 (控制在10%以内)
CLARIFICATIONS = [
    {"query": "我不确定想问什么", "response": "没关系，你可以先说说感兴趣的话题，我帮你缩小范围。"},
    {"query": "这个问题有点模糊", "response": "我理解你的问题比较复杂，能否提供更多细节？比如具体是什么场景？"},
    {"query": "随便聊聊", "response": "好的，我们随便聊聊吧！有什么感兴趣的话题吗？"},
    {"query": "你说呢", "response": "我可以给你一些建议，但需要先了解你的具体情况。你能说说你的情况或需求吗？"},
    {"query": "你怎么看", "response": "我的看法仅供参考。你具体想了解哪方面的观点？我们可以一起讨论。"},
]


class BalancedDialogueDataset(Dataset):
    """平衡对话数据集"""

    def __init__(self, data: List[Dict], tokenizer: SimpleBigramTokenizer, max_in: int = 50, max_out: int = 40):
        self.data = data
        self.tokenizer = tokenizer
        self.max_in = max_in
        self.max_out = max_out

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        input_ids = self.tokenizer.encode(item['query'], self.max_in)
        target_ids = self.tokenizer.encode(item['response'], self.max_out)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


class Stage1422Trainer:
    """Stage 14.2.2 训练器"""

    VERSION = "Stage 14.2.2: 扩充数据量"

    def __init__(self):
        print(f"\n{'='*70}")
        print(f"{self.VERSION}")
        print(f"{'='*70}")

        self.device = torch.device('cpu')
        self.tokenizer = None
        self.decoder = None
        self.optimizer = None

    def load_components(self):
        """加载组件"""
        from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig
        from stage11a_r2_fix_v2_balanced import FixV2Model

        print(f"\n[模型] 加载Stage 13 v3检查点...")

        config = NativeTinyConfig()
        base_model = NativeBackboneTinyV1(config)
        self.base_model = FixV2Model(base_model)

        checkpoint = torch.load('stage8_dataset/stage13_v3_final_checkpoint.pt', map_location='cpu')
        self.base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.base_model.eval()

        for param in self.base_model.parameters():
            param.requires_grad = False

        print(f"  ✓ 基础模型已加载")

    def build_balanced_data(self) -> List[Dict]:
        """构建平衡数据集"""
        print(f"\n[数据] 构建扩充数据集...")

        print(f"  直接回答: {len(DIRECT_ANSWERS)}条")
        print(f"  解释说明: {len(EXPLANATIONS)}条")
        print(f"  步骤建议: {len(STEPS_SUGGESTIONS)}条")
        print(f"  澄清类: {len(CLARIFICATIONS)}条")

        all_data = []

        for item in DIRECT_ANSWERS:
            all_data.append(item)

        for item in EXPLANATIONS:
            all_data.append(item)

        for item in STEPS_SUGGESTIONS:
            all_data.append(item)

        for item in CLARIFICATIONS:
            all_data.append(item)

        random.shuffle(all_data)

        total = len(all_data)
        print(f"\n  扩充后总数: {total}条")
        print(f"  配比: 直接回答{len(DIRECT_ANSWERS)/total:.1%}, 解释{len(EXPLANATIONS)/total:.1%}, 步骤{len(STEPS_SUGGESTIONS)/total:.1%}, 澄清{len(CLARIFICATIONS)/total:.1%}")

        return all_data

    def prepare_tokenizer(self, train_data: List[Dict]):
        """准备tokenizer"""
        print(f"\n[Tokenization] 构建bigram分词器...")
        self.tokenizer = SimpleBigramTokenizer()
        texts = []
        for item in train_data:
            texts.append(item['query'])
            texts.append(item['response'])
        self.tokenizer.fit(texts)
        print(f"  ✓ Tokenizer构建完成")

    def build_decoder(self):
        """构建Decoder"""
        from phase19_native_backbone.native_backbone_tiny_v1 import NativeTinyConfig

        config = NativeTinyConfig()
        vocab_size = len(self.tokenizer.word2id)

        from stage14_2_tokenization import RecurrentResponseDecoderV2
        self.decoder = RecurrentResponseDecoderV2(config, vocab_size)

        params = sum(p.numel() for p in self.decoder.parameters())
        print(f"\n[Decoder] 参数量: {params:,}")

    def train_epoch(self, loader: DataLoader) -> float:
        """训练一个epoch"""
        self.decoder.train()
        total_loss = 0
        num_batches = 0

        for input_ids, target_ids in loader:
            self.optimizer.zero_grad()

            base_out = self.base_model.base_model(input_ids)
            pooled = base_out['pooled']
            policy_probs = base_out['policy_probs']
            gap_probs = base_out['gap_probs']

            outputs = self.decoder(pooled, policy_probs, gap_probs, target_ids)

            logits = outputs['response_logits']
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(batch_size * seq_len, vocab_size)
            targets_flat = target_ids.view(batch_size * seq_len)

            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=PAD_ID)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def generate_sample(self, query: str) -> str:
        """生成样本"""
        self.decoder.eval()
        with torch.no_grad():
            input_ids = self.tokenizer.encode(query, 50)
            input_tensor = torch.tensor([input_ids])

            base_out = self.base_model.base_model(input_tensor)
            pooled = base_out['pooled']
            policy_probs = base_out['policy_probs']
            gap_probs = base_out['gap_probs']

            out = self.decoder(pooled, policy_probs, gap_probs, None)
            generated_ids = out['generated_ids'][0].tolist()

            return self.tokenizer.decode(generated_ids)

    def evaluate_generation(self, test_queries: List[Tuple[str, str]]) -> Dict:
        """评估生成质量"""
        print(f"\n[评估] 生成质量评估")

        correct_topic = 0
        total = len(test_queries)

        for query, expected_type in test_queries:
            generated = self.generate_sample(query)
            print(f"  Q: {query}")
            print(f"  A: {generated[:60]}...")
            print()

            # 简单判断话题相关性
            if expected_type == "direct":
                # 检查是否包含query中的关键词
                q_keywords = [q for q in query if len(q) > 1]
                if any(kw in generated for kw in q_keywords if kw in generated):
                    correct_topic += 1

        return {
            'topic_accuracy': correct_topic / max(total, 1),
        }

    def run_training(self, all_data: List[Dict], num_epochs: int = 80):
        """运行训练"""
        dataset = BalancedDialogueDataset(all_data, self.tokenizer)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        self.optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=1e-3)

        print(f"\n{'='*70}")
        print(f"开始训练 (最多{num_epochs}轮)")
        print(f"{'='*70}")

        best_loss = float('inf')
        best_state = None
        loss_history = []

        for epoch in range(num_epochs):
            loss = self.train_epoch(loader)
            loss_history.append(loss)

            if loss < best_loss:
                best_loss = loss
                best_state = {k: v.cpu().clone() for k, v in self.decoder.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch+1:3d}: loss={loss:.4f} (best={best_loss:.4f})")

        if best_state:
            self.decoder.load_state_dict(best_state)

        print(f"\n{'='*70}")
        print(f"训练完成")
        print(f"最终loss: {best_loss:.4f}")
        print(f"{'='*70}")

        # 评估
        test_queries = [
            ("Python是什么", "direct"),
            ("人工智能是什么", "direct"),
            ("怎么学习编程", "direct"),
            ("你好", "direct"),
            ("什么是TCP协议", "direct"),
            ("怎么戒烟", "direct"),
            ("我不确定想问什么", "clarify"),
        ]

        metrics = self.evaluate_generation(test_queries)

        print(f"\n[汇总]")
        print(f"  话题正确率: {metrics['topic_accuracy']:.1%}")

        torch.save({
            'decoder_state_dict': self.decoder.state_dict(),
            'tokenizer_word2id': self.tokenizer.word2id,
            'tokenizer_id2word': self.tokenizer.id2word,
            'version': self.VERSION,
            'loss_history': loss_history,
        }, 'stage8_dataset/stage14_2_2_decoder.pt')

        print(f"\n  补丁已保存: stage8_dataset/stage14_2_2_decoder.pt")

        return best_loss, loss_history, metrics


def main():
    trainer = Stage1422Trainer()
    trainer.load_components()

    all_data = trainer.build_balanced_data()

    trainer.prepare_tokenizer(all_data)
    trainer.build_decoder()
    best_loss, history, metrics = trainer.run_training(all_data=all_data, num_epochs=80)
    return best_loss, history, metrics


if __name__ == "__main__":
    main()
