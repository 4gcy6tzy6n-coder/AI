"""
Stage 14.4.1: 知识库扩展

目标:
- 知识条目从16 → 200+
- 覆盖核心查询类型与多样化主题
- 命中率提升至 ≥90%

类别分布:
- 编程开发 (40+)
- 人工智能/机器学习 (30+)
- 网络/云计算 (25+)
- 数据库 (20+)
- 开发工具 (20+)
- 科学常识 (25+)
- 健康生活 (20+)
- 职业发展 (15+)
- 常用技能 (20+)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import hashlib
import re
from typing import Dict, List
from collections import defaultdict


class KnowledgeBaseExpanded:
    """扩展知识库"""

    VERSION = "Knowledge Base Expanded v1.0"

    def __init__(self, storage_path: str = "stage8_dataset/knowledge_base_expanded.json"):
        self.storage_path = storage_path
        self.entries = {}
        self.load()

    def load(self):
        """加载扩展知识库"""
        if Path(self.storage_path).exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.entries = data.get('entries', {})
                print(f"  扩展知识库已加载: {len(self.entries)}条")
            except Exception as e:
                print(f"  加载失败: {e}, 初始化新知识库")
                self._init_expanded_knowledge()
        else:
            self._init_expanded_knowledge()

    def _init_expanded_knowledge(self):
        """初始化扩展知识库"""
        self.entries = {}

        self._add_programming_knowledge()
        self._add_ai_ml_knowledge()
        self._add_network_cloud_knowledge()
        self._add_database_knowledge()
        self._add_devtools_knowledge()
        self._add_science_knowledge()
        self._add_health_knowledge()
        self._add_career_knowledge()
        self._add_skills_knowledge()

        self.save()
        print(f"  扩展知识库已初始化: {len(self.entries)}条")

    def _add_programming_knowledge(self):
        """编程开发知识"""
        entries = [
            ("python_def", "Python是什么", "Python是一种高级编程语言，由Guido van Rossum于1991年发明。它语法简洁、易学易用，广泛用于Web开发、数据分析、人工智能、科学计算等领域。", "programming", ["Python", "编程语言", "入门"]),
            ("python_var", "Python变量是什么", "Python变量是存储数据的容器，不需要声明类型，直接赋值即可使用。例如：x = 10, name = 'Alice'。", "programming", ["Python", "变量", "基础"]),
            ("python_list", "Python列表是什么", "Python列表是可变的有序序列，用方括号定义。例如：fruits = ['apple', 'banana', 'orange']。列表支持索引、切片、添加、删除等操作。", "programming", ["Python", "列表", "数据结构"]),
            ("python_dict", "Python字典是什么", "Python字典是键值对组成的数据结构，用花括号定义。例如：person = {'name': 'Alice', 'age': 25}。字典支持快速的键值查找。", "programming", ["Python", "字典", "数据结构"]),
            ("python_function", "Python函数怎么定义", "Python函数用def关键字定义，语法：def 函数名(参数): 函数体。例如：def greet(name): return f'Hello, {name}'", "programming", ["Python", "函数", "基础"]),
            ("python_class", "Python类是什么", "Python类是面向对象编程的基础，用于创建对象。定义语法：class ClassName:。类包含属性和方法。", "programming", ["Python", "类", "面向对象"]),
            ("python_loop", "Python循环怎么写", "Python有两种循环：for循环用于遍历序列，while循环用于条件循环。语法：for item in items: 和 while condition:", "programming", ["Python", "循环", "基础"]),
            ("python_import", "Python如何导入模块", "Python用import或from...import导入模块。例如：import os, from datetime import datetime。", "programming", ["Python", "模块", "基础"]),
            ("java_def", "Java是什么", "Java是一种面向对象的编程语言，由Sun Microsystems于1995年发布。它具有跨平台特性，一次编写可在不同系统运行。", "programming", ["Java", "编程语言"]),
            ("javascript_def", "JavaScript是什么", "JavaScript是一种脚本语言，主要用于Web前端开发，也可用于后端(Node.js)。它是浏览器原生支持的编程语言。", "programming", ["JavaScript", "前端", "Web"]),
            ("c_plus_plus_def", "C++是什么", "C++是C语言的扩展，是一种高效的编译型语言，支持面向对象、泛型编程。常用于游戏开发、系统编程、性能要求高的应用。", "programming", ["C++", "编程语言"]),
            ("rust_def", "Rust是什么", "Rust是一种系统编程语言，强调安全性和并发性。由Mozilla开发，适合开发操作系统、浏览器组件等需要高性能和高安全性的软件。", "programming", ["Rust", "编程语言", "系统"]),
            ("go_def", "Go是什么", "Go是Google开发的编程语言，语法简洁、编译快速、内置并发支持。适合云计算、微服务、网络编程等领域。", "programming", ["Go", "Golang", "云计算"]),
            ("ruby_def", "Ruby是什么", "Ruby是一种面向对象的脚本语言，由松本行弘于1995年发明。语法优雅简洁，适合快速开发Web应用(Ruby on Rails框架)。", "programming", ["Ruby", "编程语言"]),
            ("swift_def", "Swift是什么", "Swift是Apple开发的编程语言，用于iOS和macOS应用开发。语法现代、安全、高效，是Objective-C的替代品。", "programming", ["Swift", "iOS", "Apple"]),
            ("kotlin_def", "Kotlin是什么", "Kotlin是JetBrains开发的JVM语言，可与Java互操作。主要用于Android开发，语法简洁空安全。", "programming", ["Kotlin", "Android", "JVM"]),
            ("typescript_def", "TypeScript是什么", "TypeScript是JavaScript的超集，增加了静态类型检查。适合大型项目开发，提供更好的代码提示和错误检测。", "programming", ["TypeScript", "前端", "类型"]),
            ("php_def", "PHP是什么", "PHP是一种服务器端脚本语言，主要用于Web开发。WordPress、Drupal等CMS基于PHP构建。", "programming", ["PHP", "Web", "后端"]),
            ("sql_def", "SQL是什么", "SQL是结构化查询语言，用于操作关系型数据库。主要操作包括SELECT、INSERT、UPDATE、DELETE。", "programming", ["SQL", "数据库", "查询"]),
            ("html_def", "HTML是什么", "HTML是超文本标记语言，用于构建网页结构。由标签组成，如<html>、<head>、<body>、<div>等。", "programming", ["HTML", "Web", "前端"]),
            ("css_def", "CSS是什么", "CSS是层叠样式表，用于控制网页外观。包括颜色、布局、字体等样式定义。", "programming", ["CSS", "Web", "前端"]),
            ("react_def", "React是什么", "React是Facebook开发的UI库，用于构建单页应用。使用组件化和虚拟DOM，提高渲染效率。", "programming", ["React", "前端", "UI"]),
            ("vue_def", "Vue是什么", "Vue是渐进式JavaScript框架，易于学习。可用于构建响应式Web界面，文档友好。", "programming", ["Vue", "前端", "框架"]),
            ("angular_def", "Angular是什么", "Angular是Google开发的前端框架，提供完整的开发解决方案。包括依赖注入、指令、管道等特性。", "programming", ["Angular", "前端", "框架"]),
            ("nodejs_def", "Node.js是什么", "Node.js是基于Chrome V8引擎的JavaScript运行时，可用于服务器端开发。适合实时应用和微服务。", "programming", ["Node.js", "后端", "JavaScript"]),
            ("django_def", "Django是什么", "Django是Python Web框架，强调快速开发和简洁实用。提供ORM、管理后台、用户认证等开箱即用功能。", "programming", ["Django", "Python", "Web"]),
            ("flask_def", "Flask是什么", "Flask是Python轻量级Web框架，核心简单但可扩展。适合小型应用和REST API开发。", "programming", ["Flask", "Python", "Web"]),
            ("spring_def", "Spring是什么", "Spring是Java企业级应用框架，提供依赖注入和面向切面编程。Spring Boot简化了Spring配置。", "programming", ["Spring", "Java", "企业级"]),
            ("algorithm_sort", "常见的排序算法有哪些", "常见排序算法包括：冒泡排序、选择排序、插入排序（O(n²)）；快速排序、归并排序、堆排序（O(n log n)）；计数排序、桶排序（O(n)）。", "programming", ["算法", "排序"]),
            ("algorithm_search", "常见的搜索算法有哪些", "常见搜索算法包括：线性搜索、二分搜索、二叉树搜索、深度优先搜索(DFS)、广度优先搜索(BFS)、A*算法等。", "programming", ["算法", "搜索"]),
            ("data_structure_array", "数组和链表的区别", "数组是连续内存存储，索引查找快(O(1))，但插入删除慢(O(n))。链表是离散存储，插入删除快(O(1))，但查找慢(O(n))。", "programming", ["数据结构", "数组", "链表"]),
            ("data_structure_stack", "栈的特点是什么", "栈是后进先出(LIFO)的数据结构。只允许在栈顶进行插入和删除操作。主要应用包括函数调用、表达式求值、括号匹配等。", "programming", ["数据结构", "栈"]),
            ("data_structure_queue", "队列的特点是什么", "队列是先进先出(FIFO)的数据结构。只允许在队尾插入，队首删除。常用于任务调度、广度优先搜索等场景。", "programming", ["数据结构", "队列"]),
            ("oop_def", "面向对象编程是什么", "面向对象编程(OOP)是一种编程范式，基于类和对象概念。核心特性包括封装、继承、多态。", "programming", ["OOP", "面向对象", "概念"]),
            ("design_pattern_singleton", "单例模式是什么", "单例模式确保一个类只有一个实例，并提供全局访问点。常用于配置管理、数据库连接池等场景。", "programming", ["设计模式", "单例"]),
            ("design_pattern_factory", "工厂模式是什么", "工厂模式定义创建对象的接口，让子类决定实例化哪个类。分为简单工厂、工厂方法、抽象工厂。", "programming", ["设计模式", "工厂"]),
            ("api_def", "API是什么", "API(应用程序编程接口)是软件系统间交互的约定。RESTful API是最常用的Web API风格，基于HTTP协议。", "programming", ["API", "REST", "接口"]),
            ("microservice_def", "微服务是什么", "微服务是将应用拆分为小型独立服务的设计架构。每个服务运行独立进程，通过轻量级通信协作。", "programming", ["微服务", "架构"]),
            ("docker_def", "Docker是什么", "Docker是容器化平台，将应用及其依赖打包成容器。容器比虚拟机更轻量，启动更快。", "programming", ["Docker", "容器", "DevOps"]),
            ("kubernetes_def", "Kubernetes是什么", "Kubernetes是容器编排平台，用于自动化容器化应用的部署、扩缩容和管理。简称K8s。", "programming", ["Kubernetes", "K8s", "容器编排"]),
            ("git_branch", "Git分支怎么管理", "Git常用分支操作：git branch创建分支，git checkout切换分支，git merge合并分支，git rebase变基。", "programming", ["Git", "分支", "版本控制"]),
            ("code_review_def", "代码审查是什么", "代码审查是团队成员互相检查代码的过程。可以发现bug、保证代码质量、分享知识。", "programming", ["代码审查", "质量"]),
            ("tdd_def", "测试驱动开发是什么", "测试驱动开发(TDD)是先写测试再写代码的开发方法。红绿重构循环：先写失败的测试，再写通过测试的代码。", "programming", ["TDD", "测试"]),
            ("refactor_def", "代码重构是什么", "重构是在不改变外部行为的前提下，改进代码内部结构。提高可读性、可维护性，但不引入新功能。", "programming", ["重构", "代码质量"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_ai_ml_knowledge(self):
        """AI/机器学习知识"""
        entries = [
            ("ai_def", "什么是人工智能", "人工智能是计算机科学的一个分支，致力于开发能够模拟人类智能的技术。包括机器学习、自然语言处理、计算机视觉等多个研究方向。", "ai", ["人工智能", "AI", "定义"]),
            ("ml_def", "什么是机器学习", "机器学习是人工智能的一个子领域，通过让计算机从数据中学习模式来进行预测和决策，而不需要明确的编程规则。常见算法包括监督学习、无监督学习和强化学习。", "ai", ["机器学习", "ML", "算法"]),
            ("dl_def", "什么是深度学习", "深度学习是机器学习的子集，使用多层神经网络来处理复杂数据。它在图像识别、自然语言处理、计算机视觉等领域取得了突破性进展。", "ai", ["深度学习", "神经网络", "AI"]),
            ("supervised_learning", "什么是监督学习", "监督学习使用标注数据训练模型，输入有对应的正确输出。常见任务包括分类和回归。常用算法有线性回归、逻辑回归、决策树、SVM、神经网络等。", "ai", ["监督学习", "分类", "回归"]),
            ("unsupervised_learning", "什么是非监督学习", "非监督学习使用无标注数据，发现数据内部结构。常见任务包括聚类和降维。常用算法有K-means、DBSCAN、PCA、t-SNE等。", "ai", ["非监督学习", "聚类", "降维"]),
            ("reinforcement_learning", "什么是强化学习", "强化学习是智能体通过与环境交互学习最优策略。智能体获得奖励或惩罚反馈，逐步优化决策。应用包括游戏AI、机器人控制等。", "ai", ["强化学习", "智能体", "策略"]),
            ("nlp_def", "什么是自然语言处理", "自然语言处理(NLP)是AI子领域，研究计算机理解和生成人类语言。包括文本分类、命名实体识别、机器翻译、问答系统等任务。", "ai", ["NLP", "自然语言", "文本"]),
            ("cv_def", "什么是计算机视觉", "计算机视觉是AI子领域，让计算机理解和处理图像视频。包括图像分类、目标检测、图像分割、人脸识别等任务。", "ai", ["计算机视觉", "CV", "图像"]),
            ("neural_network_def", "什么是神经网络", "神经网络是受人脑启发的数学模型，由神经元层组成。包括输入层、隐藏层、输出层。通过反向传播算法学习参数。", "ai", ["神经网络", "神经元", "深度学习"]),
            ("cnn_def", "什么是卷积神经网络", "卷积神经网络(CNN)是处理图像的神经网络，使用卷积层提取特征。广泛应用于图像分类、目标检测等计算机视觉任务。", "ai", ["CNN", "卷积", "图像"]),
            ("rnn_def", "什么是循环神经网络", "循环神经网络(RNN)适合处理序列数据，具有记忆能力。但存在梯度消失问题，长序列学习困难。", "ai", ["RNN", "循环", "序列"]),
            ("lstm_def", "什么是LSTM", "LSTM是长短期记忆网络，是RNN的变体。引入门控机制，解决长期依赖问题。适用于机器翻译、语音识别等长序列任务。", "ai", ["LSTM", "长短期记忆", "序列"]),
            ("transformer_def", "什么是Transformer", "Transformer是基于自注意力机制的模型架构，摒弃了RNN。成为NLP领域主流架构，是GPT、BERT等模型的基础。", "ai", ["Transformer", "注意力机制", "NLP"]),
            ("bert_def", "什么是BERT", "BERT是Google提出的预训练语言模型，使用双向Transformer编码器。擅长理解任务，如文本分类、问答系统。", "ai", ["BERT", "预训练", "NLP"]),
            ("gpt_def", "什么是GPT", "GPT是OpenAI开发的生成式预训练模型，使用Transformer解码器。擅长生成任务，如文本生成、对话。", "ai", ["GPT", "生成", "预训练"]),
            ("llm_def", "什么是大语言模型", "大语言模型(LLM)是参数规模巨大的预训练语言模型。可以理解和生成人类语言，具备涌现能力。", "ai", ["LLM", "大模型", "预训练"]),
            ("prompt_engineering", "什么是提示工程", "提示工程是设计和优化输入提示词，以获得更好输出的技术。包括零样本、少样本、思维链等技巧。", "ai", ["提示工程", "Prompt", "技巧"]),
            ("fine_tuning_def", "什么是微调", "微调是在预训练模型基础上，用特定任务数据进一步训练。可以用较少数据获得特定任务的高性能。", "ai", ["微调", "迁移学习", "训练"]),
            ("rag_def", "什么是RAG", "检索增强生成(RAG)结合检索系统和生成模型。先检索相关知识，再基于检索结果生成回答，提高准确性和可信度。", "ai", ["RAG", "检索", "生成"]),
            ("overfitting_def", "什么是过拟合", "过拟合是模型在训练数据上表现好，但在新数据上泛化能力差。解决方法包括增加数据、正则化、Dropout、早停等。", "ai", ["过拟合", "泛化", "正则化"]),
            ("underfitting_def", "什么是欠拟合", "欠拟合是模型在训练和新数据上都表现不好。解决方法包括增加模型复杂度、增加特征、减少正则化等。", "ai", ["欠拟合", "模型复杂度"]),
            ("cross_validation", "什么是交叉验证", "交叉验证是将数据分成K份，轮流使用K-1份训练、1份验证。可以充分利用数据，评估模型泛化能力。", "ai", ["交叉验证", "评估", "K折"]),
            ("accuracy_def", "什么是准确率", "准确率是正确预测样本占总样本的比例。公式：准确率 = (TP+TN) / (TP+TN+FP+FN)。适合类别均衡数据。", "ai", ["准确率", "评估指标"]),
            ("precision_def", "什么是精确率", "精确率是预测为正的样本中真正例的比例。公式：精确率 = TP / (TP+FP)。关注预测为正的正确性。", "ai", ["精确率", "评估指标"]),
            ("recall_def", "什么是召回率", "召回率是真正例中被正确预测的比例。公式：召回率 = TP / (TP+FN)。关注正例被找全的程度。", "ai", ["召回率", "评估指标"]),
            ("f1_score_def", "什么是F1分数", "F1分数是精确率和召回率的调和平均。公式：F1 = 2 * (精确率 * 召回率) / (精确率 + 召回率)。综合评估性能。", "ai", ["F1分数", "评估指标"]),
            ("gradient_descent", "什么是梯度下降", "梯度下降是优化算法，用于最小化损失函数。沿梯度负方向迭代更新参数，直到收敛。是神经网络训练的基础。", "ai", ["梯度下降", "优化", "训练"]),
            ("backpropagation", "什么是反向传播", "反向传播是神经网络训练算法，计算损失函数对参数的梯度。通过链式法则，从输出层反向传播到输入层。", "ai", ["反向传播", "训练", "梯度"]),
            ("optimizer_adam", "Adam优化器是什么", "Adam是自适应学习率优化器，结合动量和RMSProp。可以自动调整学习率，收敛速度快，是深度学习最常用的优化器。", "ai", ["Adam", "优化器", "训练"]),
            ("loss_function", "什么是损失函数", "损失函数衡量模型预测与真实值的差距。回归任务常用MSE，分类任务常用交叉熵。训练目标是最小化损失函数。", "ai", ["损失函数", "MSE", "交叉熵"]),
            ("epoch_batch", "epoch和batch的区别", "Epoch是完整遍历一次训练集。Batch是将训练集分成小批次，每批计算梯度更新一次。batch_size是每批样本数。", "ai", ["epoch", "batch", "训练"]),
            ("learning_rate", "什么是学习率", "学习率控制参数更新步长。太大会导致震荡不收敛，太小收敛慢。通常需要调参或使用学习率调度。", "ai", ["学习率", "超参数"]),
            ("dropout_def", "Dropout是什么", "Dropout是正则化技术，训练时随机丢弃部分神经元。可以防止过拟合，提高泛化能力。", "ai", ["Dropout", "正则化", "过拟合"]),
            ("batch_normalization", "批归一化是什么", "批归一化对每一批次数据做均值方差归一化。可以加速训练、稳定梯度。是深度神经网络的常用技术。", "ai", ["批归一化", "BN", "训练"]),
            ("embedding_def", "词嵌入是什么", "词嵌入是将词语映射到低维向量空间，捕捉语义信息。Word2Vec、GloVe是经典方法。嵌入层是可学习的。", "ai", ["词嵌入", "Word2Vec", "向量"]),
            ("transfer_learning", "什么是迁移学习", "迁移学习是将一个任务上学到的知识应用到另一个相关任务。可以减少目标任务的训练数据需求。", "ai", ["迁移学习", "预训练", "知识迁移"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_network_cloud_knowledge(self):
        """网络/云计算知识"""
        entries = [
            ("http_def", "HTTP协议是什么", "HTTP是超文本传输协议，是Web通信的基础。它是请求-响应协议，常见方法有GET、POST、PUT、DELETE。HTTP是无状态的协议。", "network", ["HTTP", "协议", "网络"]),
            ("https_def", "HTTPS是什么", "HTTPS是HTTP的安全版本，通过SSL/TLS加密传输数据。提供身份验证和数据加密，保护用户隐私安全。", "network", ["HTTPS", "安全", "加密"]),
            ("tcp_ip_def", "TCP/IP协议是什么", "TCP/IP是Internet的基础协议族。TCP提供可靠传输，IP负责寻址路由。四层模型：应用层、传输层、网络层、链路层。", "network", ["TCP", "IP", "协议"]),
            ("dns_def", "DNS是什么", "DNS是域名系统，将域名解析为IP地址。像互联网的电话簿，用户输入域名即可访问网站。", "network", ["DNS", "域名", "解析"]),
            ("cdn_def", "CDN是什么", "CDN是内容分发网络，在各地部署边缘服务器缓存静态内容。可以加速访问、减轻源站压力、提高可用性。", "network", ["CDN", "加速", "缓存"]),
            ("websocket_def", "WebSocket是什么", "WebSocket是支持双向通信的协议，服务器可主动推送数据。适用于实时应用如聊天、游戏、股票行情。", "network", ["WebSocket", "实时", "通信"]),
            ("rest_api_def", "REST API是什么", "REST是表述性状态转移的架构风格。基于HTTP协议，使用GET/POST/PUT/DELETE操作资源。轻量、易理解、易扩展。", "network", ["REST", "API", "架构"]),
            ("graphql_def", "GraphQL是什么", "GraphQL是API查询语言，允许客户端精确指定需要的数据。可以减少数据传输量，提高效率。", "network", ["GraphQL", "API", "查询"]),
            ("cloud_computing_def", "什么是云计算", "云计算是一种通过互联网提供计算资源的服务模式。用户可以按需使用服务器、存储、数据库等资源，无需自行维护硬件。常见服务模式包括IaaS、PaaS、SaaS。", "network", ["云计算", "云服务", "IaaS", "PaaS", "SaaS"]),
            ("iaas_def", "IaaS是什么", "IaaS是基础设施即服务，提供虚拟化的计算资源。包括虚拟机、存储、网络。用户自行管理操作系统和应用。", "network", ["IaaS", "云计算", "基础设施"]),
            ("paas_def", "PaaS是什么", "PaaS是平台即服务，提供应用开发和部署平台。用户无需管理底层基础设施，如Heroku、Google App Engine。", "network", ["PaaS", "云计算", "平台"]),
            ("saas_def", "SaaS是什么", "SaaS是软件即服务，通过浏览器提供完整应用。用户无需安装维护，如Google Docs、Salesforce、Office 365。", "network", ["SaaS", "云计算", "软件"]),
            ("aws_def", "AWS是什么", "AWS是Amazon Web Services，提供全面的云服务。包括EC2计算、S3存储、RDS数据库、Lambda无服务器等。", "network", ["AWS", "云服务", "Amazon"]),
            ("azure_def", "Azure是什么", "Azure是Microsoft的云平台，提供IaaS、PaaS、SaaS服务。与Windows生态深度集成，适合企业应用。", "network", ["Azure", "云服务", "Microsoft"]),
            ("gcp_def", "GCP是什么", "GCP是Google Cloud Platform，提供云服务。BigQuery数据分析、Kubernetes引擎、TensorFlow等AI服务是亮点。", "network", ["GCP", "Google", "云服务"]),
            ("serverless_def", "无服务器计算是什么", "无服务器计算让开发者无需管理服务器，按代码执行时间付费。如AWS Lambda、Azure Functions。", "network", ["无服务器", "Serverless", "Lambda"]),
            ("load_balancer_def", "负载均衡是什么", "负载均衡将请求分发到多个服务器，提高可用性和响应速度。常用算法有轮询、最少连接、IP哈希。", "network", ["负载均衡", "高可用"]),
            ("firewall_def", "防火墙是什么", "防火墙监控和控制网络流量，根据规则允许或阻止数据包。保护网络免受未授权访问。", "network", ["防火墙", "安全", "网络"]),
            ("vpn_def", "VPN是什么", "VPN是虚拟专用网络，通过公网建立加密隧道。可以远程访问内网资源，保护隐私安全。", "network", ["VPN", "安全", "加密"]),
            ("bandwidth_def", "带宽是什么", "带宽是网络传输能力的度量，单位是bps（比特/秒）。带宽越大，数据传输速度越快。", "network", ["带宽", "网络", "速度"]),
            ("latency_def", "延迟是什么", "延迟是数据从发送到达接收的时间。高延迟影响用户体验，实时应用需要低延迟网络。", "network", ["延迟", "网络", "性能"]),
            ("ssl_tls_def", "SSL/TLS是什么", "SSL/TLS是加密协议，保护网络通信安全。用于HTTPS、邮件加密、VPN等。建立安全连接前需要握手。", "network", ["SSL", "TLS", "加密"]),
            ("nginx_def", "Nginx是什么", "Nginx是高性能Web服务器和反向代理服务器。擅长处理高并发，也可作为负载均衡器。", "network", ["Nginx", "Web服务器", "反向代理"]),
            ("apache_def", "Apache是什么", "Apache是开源Web服务器软件。模块化设计，功能丰富，是最流行的Web服务器之一。", "network", ["Apache", "Web服务器"]),
            ("microservice_arch", "微服务架构是什么", "微服务将应用拆分为小型独立服务，每个服务负责特定功能。服务间通过API通信，便于扩展和维护。", "network", ["微服务", "架构", "分布式"]),
            ("api_gateway_def", "API网关是什么", "API网关是API的管理和转发入口。提供路由、认证、限流、监控等功能。微服务架构的核心组件。", "network", ["API网关", "微服务", "管理"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_database_knowledge(self):
        """数据库知识"""
        entries = [
            ("database_def", "什么是数据库", "数据库是存储和管理数据的系统。常见的数据库类型包括关系型数据库(如MySQL、PostgreSQL)和非关系型数据库(如MongoDB、Redis)。", "database", ["数据库", "存储", "SQL"]),
            ("sql_mysql_def", "MySQL是什么", "MySQL是开源关系型数据库管理系统。使用SQL语言，性能稳定，广泛用于Web应用。是LAMP栈的核心组件。", "database", ["MySQL", "关系型", "开源"]),
            ("sql_postgresql_def", "PostgreSQL是什么", "PostgreSQL是功能强大的开源对象关系型数据库。支持JSON、GIS扩展、事务完整性。适合复杂查询和企业级应用。", "database", ["PostgreSQL", "关系型", "开源"]),
            ("nosql_mongodb_def", "MongoDB是什么", "MongoDB是面向文档的NoSQL数据库，数据存储为JSON格式文档。灵活的模式设计，适合快速迭代开发。", "database", ["MongoDB", "NoSQL", "文档"]),
            ("nosql_redis_def", "Redis是什么", "Redis是内存数据结构存储，支持字符串、哈希、列表、集合、有序集合。常用于缓存、Session存储、消息队列。", "database", ["Redis", "缓存", "内存"]),
            ("nosql_cassandra_def", "Cassandra是什么", "Cassandra是分布式NoSQL数据库，擅长处理海量数据。无单点故障，适合写入密集型应用。", "database", ["Cassandra", "NoSQL", "分布式"]),
            ("database_index", "数据库索引是什么", "数据库索引是特殊数据结构，加速数据查找。就像书籍的目录，可以快速定位到数据位置。但会增加写入开销。", "database", ["索引", "数据库", "优化"]),
            ("sql_join_def", "SQL JOIN是什么", "JOIN用于连接多个表的行。基于相关列的值进行匹配。类型包括INNER JOIN、LEFT JOIN、RIGHT JOIN、FULL JOIN。", "database", ["JOIN", "SQL", "连接"]),
            ("acid_def", "ACID是什么", "ACID是事务的四个特性：原子性(Atomicity)、一致性(Consistency)、隔离性(Isolation)、持久性(Durability)。保证数据库可靠性。", "database", ["ACID", "事务", "数据库"]),
            ("cap_theorem", "CAP定理是什么", "CAP定理：分布式系统最多同时满足一致性(Consistency)、可用性(Availability)、分区容错性(Partition)中的两个。", "database", ["CAP", "分布式", "一致性"]),
            ("database_sharding", "数据库分片是什么", "数据库分片是将数据水平切分到多个数据库实例。可以分散负载，支持大数据量扩展。", "database", ["分片", "分布式", "扩展"]),
            ("orm_def", "ORM是什么", "对象关系映射(ORM)将数据库表映射为编程语言对象。可以使用面向对象方式操作数据库，如SQLAlchemy、Hibernate。", "database", ["ORM", "数据库", "映射"]),
            ("sql_injection", "SQL注入是什么", "SQL注入是安全漏洞，通过用户输入拼接恶意SQL代码。防护方法：参数化查询、转义输入、使用ORM。", "database", ["SQL注入", "安全", "防护"]),
            ("database_normalization", "数据库规范化是什么", "规范化是设计数据库结构的原则，减少数据冗余。常见范式：1NF、2NF、3NF、BCNF。", "database", ["规范化", "范式", "设计"]),
            ("database_transaction", "数据库事务是什么", "事务是一组数据库操作，要么全部成功，要么全部失败回滚。使用BEGIN、COMMIT、ROLLBACK控制。", "database", ["事务", "数据库", "ACID"]),
            ("database_view", "数据库视图是什么", "视图是基于查询的虚拟表，不存储数据。可以简化复杂查询、保护敏感数据。", "database", ["视图", "数据库", "虚拟表"]),
            ("database_procedure", "存储过程是什么", "存储过程是预编译的SQL代码块，存储在数据库中。可以接受参数、重复调用，提高性能。", "database", ["存储过程", "数据库", "SQL"]),
            ("replication_def", "数据库复制是什么", "数据库复制是将数据同步到多个副本。可以提高可用性、负载均衡。主从复制是常见模式。", "database", ["复制", "主从", "高可用"]),
            ("backup_def", "数据库备份是什么", "数据库备份是定期保存数据副本。策略包括全量备份、增量备份。用于灾难恢复。", "database", ["备份", "恢复", "灾难恢复"]),
            ("mongodb_atlas", "MongoDB Atlas是什么", "MongoDB Atlas是MongoDB的云数据库服务。提供自动扩展、备份、监控等管理功能。", "database", ["MongoDB", "Atlas", "云"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_devtools_knowledge(self):
        """开发工具知识"""
        entries = [
            ("git_def", "Git是什么", "Git是分布式版本控制系统，用于跟踪代码变更、协作开发。常见命令包括git add、git commit、git push、git pull。", "tools", ["Git", "版本控制", "开发工具"]),
            ("github_def", "GitHub是什么", "GitHub是基于Git的代码托管平台。提供代码仓库、Issue跟踪、Pull Request、Actions CI/CD。", "tools", ["GitHub", "代码托管", "协作"]),
            ("gitlab_def", "GitLab是什么", "GitLab是基于Git的DevOps平台。提供代码托管、CI/CD、Registry、监控等完整工具链。", "tools", ["GitLab", "DevOps", "CI/CD"]),
            ("docker_container", "Docker容器是什么", "Docker容器是轻量级虚拟化技术。打包应用及其依赖，环境一致性好。启动快速，占用资源少。", "tools", ["Docker", "容器", "虚拟化"]),
            ("docker_image", "Docker镜像是什么", "Docker镜像是只读模板，用于创建容器。由层叠只读层组成，可复用和共享。", "tools", ["Docker", "镜像", "容器"]),
            ("dockerfile_def", "Dockerfile是什么", "Dockerfile是构建Docker镜像的配置文件。包含基础镜像、依赖安装、命令执行等指令。", "tools", ["Dockerfile", "Docker", "构建"]),
            ("docker_compose", "Docker Compose是什么", "Docker Compose是定义多容器应用的工具。使用YAML文件描述服务、网络、卷。一次命令启动完整应用。", "tools", ["DockerCompose", "Docker", "编排"]),
            ("kubernetes_k8s", "Kubernetes是什么", "Kubernetes是容器编排平台，简称K8s。自动化容器部署、扩缩容、管理。是云原生标配。", "tools", ["Kubernetes", "K8s", "容器编排"]),
            ("ci_cd_def", "CI/CD是什么", "CI是持续集成，CD是持续交付/部署。自动化构建、测试、部署流程。提高开发效率和质量。", "tools", ["CI/CD", "自动化", "DevOps"]),
            ("jenkins_def", "Jenkins是什么", "Jenkins是开源自动化服务器。用于CI/CD流水线，支持插件扩展。自动化构建、测试、部署。", "tools", ["Jenkins", "CI/CD", "自动化"]),
            ("github_actions", "GitHub Actions是什么", "GitHub Actions是GitHub的CI/CD功能。在仓库中定义工作流，自动执行构建测试部署。", "tools", ["GitHubActions", "CI/CD", "GitHub"]),
            ("ansible_def", "Ansible是什么", "Ansible是自动化工具，用于配置管理、应用部署、任务执行。使用YAML描述 playbook，易学易用。", "tools", ["Ansible", "自动化", "配置管理"]),
            ("terraform_def", "Terraform是什么", "Terraform是基础设施即代码工具。用声明式配置文件管理云资源。可版本控制、可复用。", "tools", ["Terraform", "IaC", "云"]),
            ("monitoring_prometheus", "Prometheus是什么", "Prometheus是开源监控系统。采集和存储时序数据，支持查询和告警。是云原生监控标配。", "tools", ["Prometheus", "监控", "时序"]),
            ("logging_elastic", "ELK是什么", "ELK是Elasticsearch、Logstash、Kibana日志分析套件。集中收集、存储、搜索、分析日志。", "tools", ["ELK", "日志", "Elasticsearch"]),
            ("vscode_def", "VS Code是什么", "VS Code是Microsoft的轻量级代码编辑器。开源免费，插件丰富，支持多语言。", "tools", ["VSCode", "编辑器", "IDE"]),
            ("jupyter_def", "Jupyter是什么", "Jupyter是交互式编程环境，支持Python、R等。Notebook形式便于数据分析和可视化。", "tools", ["Jupyter", "Notebook", "数据分析"]),
            ("postman_def", "Postman是什么", "Postman是API测试工具。用于发送HTTP请求、测试API、生成文档。团队协作方便。", "tools", ["Postman", "API", "测试"]),
            ("npm_def", "npm是什么", "npm是Node.js包管理器。Node.js生态的核心，提供包安装、版本管理、脚本运行。", "tools", ["npm", "Node.js", "包管理"]),
            ("yarn_def", "Yarn是什么", "Yarn是快速、可靠、安全的JavaScript包管理器。是npm的替代品，提供离线缓存。", "tools", ["Yarn", "包管理", "JavaScript"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_science_knowledge(self):
        """科学常识"""
        entries = [
            ("sky_blue_reason", "为什么天空是蓝色的", "天空呈蓝色是因为大气层对阳光的散射效应。根据瑞利散射原理，蓝光波长较短，更容易被空气分子散射，所以我们看到的天空是蓝色的。", "science", ["天空", "蓝色", "物理", "光学"]),
            ("water_boiling_point", "水的沸点是多少", "在一个标准大气压下，水的沸点是100摄氏度或212华氏度。在高压或低压环境下，沸点会相应改变。", "science", ["水", "沸点", "物理"]),
            ("sunrise_sunset_reason", "为什么会有日出日落", "日出日落是因为地球自转。地球约24小时自转一周，当某地转向太阳时是日出，背向太阳时是日落。", "science", ["日出", "日落", "地球"]),
            ("moon_phase_reason", "月亮为什么有圆缺", "月亮本身不发光，我们看到的光是反射太阳光。月球绕地球公转，位置变化导致被照亮的部分不同，形成月相。", "science", ["月亮", "月相", "天文"]),
            ("gravity_def", "重力是什么", "重力是物体间相互吸引的力。地球对物体的引力提供重力加速度。公式G=mg，g约9.8m/s²。", "science", ["重力", "引力", "物理"]),
            ("evolution_def", "进化论是什么", "进化论是解释生物多样性的科学理论。核心机制是自然选择：适应环境的个体更易存活和繁殖，优势特征代代传递。", "science", ["进化论", "自然选择", "生物学"]),
            ("photosynthesis_def", "光合作用是什么", "光合作用是植物利用阳光将二氧化碳和水转化为葡萄糖和氧气。是地球上最重要的化学反应，提供氧气和食物。", "science", ["光合作用", "植物", "生物学"]),
            ("dna_def", "DNA是什么", "DNA是脱氧核糖核酸，携带遗传信息。双螺旋结构，包含生成蛋白质所需的指令。基因是DNA的功能片段。", "science", ["DNA", "基因", "遗传"]),
            ("relativity_def", "相对论是什么", "相对论是爱因斯坦的理论。狭义相对论：光速不变，时间空间相对；广义相对论：重力是时空弯曲。", "science", ["相对论", "爱因斯坦", "物理"]),
            ("quantum_def", "量子力学是什么", "量子力学是描述微观粒子行为的物理学分支。量子具有波粒二象性、不确定性原理、量子叠加等特性。", "science", ["量子", "物理学", "微观"]),
            ("big_bang_def", "宇宙大爆炸是什么", "大爆炸理论是宇宙起源模型。约138亿年前，宇宙从一个奇点开始膨胀，形成时空、物质和能量。", "science", ["大爆炸", "宇宙", "天文"]),
            ("black_hole_def", "黑洞是什么", "黑洞是时空区域，引力强大到连光都无法逃脱。恒星坍缩或高能碰撞可形成黑洞。", "science", ["黑洞", "天文", "引力"]),
            ("climate_change_def", "气候变化是什么", "气候变化指长期气候模式改变。当前主要是人类活动排放温室气体导致的全球变暖。", "science", ["气候变化", "全球变暖", "环境"]),
            ("ozone_layer_def", "臭氧层是什么", "臭氧层是大气中臭氧浓度较高的区域。能吸收紫外线，保护地球生物。是地球的防护伞。", "science", ["臭氧层", "大气", "紫外线"]),
            ("renewable_energy", "可再生能源是什么", "可再生能源在自然界可快速补充，如太阳能、风能、水能、生物质能。清洁环保，是未来能源方向。", "science", ["可再生能源", "太阳能", "风能"]),
            ("photoshop_effect", "Photoshop是什么", "Photoshop是Adobe的图像编辑软件。用于照片修饰、图像合成、图形设计。是设计行业标准工具。", "science", ["Photoshop", "图像", "设计"]),
            ("blockchain_def", "区块链是什么", "区块链是一种分布式账本技术，通过加密链条将数据块连接起来。它具有去中心化、不可篡改的特点。广泛应用于加密货币、供应链金融等领域。", "science", ["区块链", "分布式", "加密货币"]),
            ("ai_history", "人工智能的历史", "AI概念始于1956年达特茅斯会议。经历多次高潮低谷：专家系统、机器学习、深度学习。当前是大模型时代。", "science", ["AI历史", "人工智能", "发展"]),
            ("internet_def", "互联网是什么", "互联网是连接全球的网络网络。通过TCP/IP协议通信，承载Web、Email、社交等服务。", "science", ["互联网", "网络", "TCP/IP"]),
            ("cloud_computing_history", "云计算发展历程", "云计算2000年代兴起，AWS 2006年开创IaaS模式。随后Azure、GCP等跟进，现在是数字经济基础设施。", "science", ["云计算历史", "AWS", "发展"]),
            ("mobile_internet_def", "移动互联网是什么", "移动互联网是手机等移动设备接入互联网。通过4G/5G网络，提供随时随地的信息和服务。", "science", ["移动互联网", "移动", "4G", "5G"]),
            ("big_data_def", "大数据是什么", "大数据指体量大、速度快、多样性高的数据。需要特殊技术处理和分析。5V特征：Volume、Velocity、Variety、Value、Veracity。", "science", ["大数据", "数据科学"]),
            ("iot_def", "物联网是什么", "物联网(IoT)是将传感器、设备接入网络。实现智能感知和控制。应用包括智能家居、工业4.0、智慧城市。", "science", ["物联网", "IoT", "智能"]),
            ("ar_vr_def", "AR和VR是什么", "AR增强现实在真实世界叠加虚拟信息；VR虚拟现实完全沉浸虚拟世界。改变人机交互方式。", "science", ["AR", "VR", "虚拟现实"]),
            ("encryption_def", "加密是什么", "加密是将信息转换为密文，只有正确密钥才能解密。对称加密用同一密钥，非对称加密用公钥私钥对。", "science", ["加密", "安全", "密码学"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_health_knowledge(self):
        """健康生活"""
        entries = [
            ("stop_smoking_steps", "怎么戒烟", "健康戒烟方法：1.设定戒烟日期；2.了解戒烟好处；3.使用尼古丁替代品；4.培养新习惯替代吸烟；5.避免诱因；6.寻求家人朋友支持；7.必要时咨询医生。", "health", ["戒烟", "健康", "方法"]),
            ("exercise_benefits", "运动有哪些好处", "规律运动好处多：增强心肺功能、提高免疫力、控制体重、改善情绪、增强骨骼、改善睡眠。", "health", ["运动", "健康", "锻炼"]),
            ("healthy_diet_tips", "健康饮食建议", "健康饮食原则：均衡摄入蛋白质、碳水、脂肪；多吃蔬果；控制糖盐摄入；定时定量；多喝水。", "health", ["饮食", "健康", "营养"]),
            ("sleep_quality", "怎么提高睡眠质量", "提高睡眠质量：固定作息时间；睡前少用电子设备；营造黑暗安静环境；白天适量运动；避免咖啡因。", "health", ["睡眠", "健康", "习惯"]),
            ("stress_management", "如何缓解压力", "缓解压力方法：运动健身、冥想深呼吸、与朋友倾诉、培养兴趣爱好、保证充足睡眠、必要时寻求专业帮助。", "health", ["压力", "心理健康", "调节"]),
            ("heart_health", "如何保护心脏", "保护心脏方法：健康饮食、规律运动、戒烟限酒、控制体重、监测血压血脂、管理压力。", "health", ["心脏", "健康", "心血管"]),
            ("eye_health", "如何保护眼睛", "保护眼睛方法：每用眼20分钟休息20秒；保持适当距离；户外活动；戴防蓝光眼镜；均衡饮食补充维生素A。", "health", ["眼睛", "健康", "视力"]),
            ("back_pain_prevention", "如何预防背痛", "预防背痛：保持正确坐姿；睡硬板床；避免久坐；适量运动增强核心肌群；提重物蹲下用腿力。", "health", ["背痛", "健康", "预防"]),
            ("hydration_importance", "为什么需要多喝水", "水是生命之源，参与新陈代谢、调节体温、运输营养、润滑关节。每天约需8杯水，运动后需补充更多。", "health", ["喝水", "健康", "水"], 0.9),
            ("mental_health_tips", "如何保持心理健康", "保持心理健康：接纳情绪、规律作息、适度运动、社交支持、培养兴趣、设定目标、必要时寻求专业咨询。", "health", ["心理健康", "情绪", "压力"]),
            ("weight_management", "如何健康控制体重", "健康体重管理：均衡饮食、适量运动、充足睡眠、减少久坐。避免极端节食，会伤害代谢。", "health", ["体重", "减肥", "健康"]),
            ("diabetes_prevention", "如何预防糖尿病", "预防糖尿病：健康饮食、控制体重、规律运动、戒烟限酒、定期体检、关注家族史。", "health", ["糖尿病", "预防", "健康"]),
            ("blood_pressure_def", "血压是什么", "血压是血液对血管壁的压力。收缩压是心脏收缩时压力，舒张压是放松时压力。正常值约120/80mmHg。", "health", ["血压", "健康", "指标"]),
            ("cancer_prevention", "如何预防癌症", "预防癌症：戒烟限酒、健康饮食、多运动、防晒、规范体检、接种疫苗（如HPV疫苗）、了解家族史。", "health", ["癌症", "预防", "健康"]),
            ("hydration_skincare", "护肤为什么要补水", "皮肤补水重要性：保持皮肤弹性、延缓衰老、防止干裂、促进代谢产物排出。使用保湿霜、多喝水。", "health", ["护肤", "补水", "美容"]),
            ("meditation_benefits", "冥想有什么好处", "冥想好处：减轻压力焦虑、改善专注力、提高睡眠质量、增强自我觉察、降低血压、提升幸福感。", "health", ["冥想", "心理健康", "放松"]),
            ("posture_importance", "为什么要注意姿势", "不良姿势导致：颈椎病、腰背痛、肌肉紧张、呼吸受限。注意站姿坐姿，睡合适枕头，选人体工学椅。", "health", ["姿势", "健康", "颈椎"]),
            ("bone_health", "如何保护骨骼健康", "保护骨骼：补钙补维D、适量负重运动、晒太阳、戒烟限酒、预防跌倒。35岁后骨密度开始下降。", "health", ["骨骼", "健康", "钙"]),
            ("brain_health", "如何保持大脑健康", "保持大脑健康：学习新技能、社交活动、运动健身、充足睡眠、健康饮食、减少压力。", "health", ["大脑", "健康", "认知"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_career_knowledge(self):
        """职业发展"""
        entries = [
            ("resume_def", "简历怎么写", "好简历要点：简洁一页、突出成就、使用动词、数据量化、定制化针对职位。包含个人信息、教育背景、工作经历、技能列表。", "career", ["简历", "求职", "找工作"]),
            ("interview_tips", "面试技巧有哪些", "面试技巧：提前研究公司、准备STAR法则回答、提问有深度问题、保持自信、跟进感谢邮件。", "career", ["面试", "求职", "技巧"]),
            ("salary_negotiation", "如何谈薪资", "谈薪资建议：先做市场调研、强调自身价值、给出具体数字、考虑整体福利、保持professional但坚定。", "career", ["薪资", "谈判", "求职"]),
            ("career_change", "如何转行", "转行步骤：明确目标行业、补足技能差距、建立人脉、从小做起、保持开放学习心态、给自己时间成长。", "career", ["转行", "职业发展"]),
            ("remote_work_tips", "远程工作建议", "远程工作：设立专门工作空间、固定作息、使用效率工具、保持沟通、区分工作和休息、避免孤立感。", "career", ["远程工作", "居家办公"]),
            ("leadership_def", "什么是领导力", "领导力是影响和激励他人实现目标的能力。包括：愿景设定、决策能力、沟通技巧、团队激励、责任心。", "career", ["领导力", "管理"]),
            ("time_management", "时间管理技巧", "时间管理：番茄工作法、优先级矩阵、设定deadline、避免多任务、学会说不、使用日历和待办清单。", "career", ["时间管理", "效率", "生产力"]),
            ("public_speaking", "如何做好演讲", "演讲技巧：了解听众、明确核心信息、使用故事数据开头、保持眼神交流、练习、准备Q&A、控场和节奏。", "career", ["演讲", "沟通", "技巧"]),
            ("networking_tips", "社交 networking 建议", "社交技巧：主动真诚、倾听他人、提供价值、跟进维护、线上线下结合。质量重于数量。", "career", ["社交", "人脉", "职业"]),
            ("continuous_learning", "为什么需要持续学习", "持续学习重要性：技术更新快、提升竞争力、适应变化、职业发展、个人成长。方式：在线课程、书籍、实践。", "career", ["学习", "成长", "职业"]),
            ("work_life_balance", "如何平衡工作生活", "平衡工作生活：设定界限、优先级排序、委派任务、休息充电、设定工作外目标、学会拒绝无效社交。", "career", ["工作生活平衡", "健康"]),
            ("career_goal_setting", "如何设定职业目标", "设定职业目标：SMART原则、短期中期长期结合、具体可衡量、定期回顾调整、与价值观一致。", "career", ["目标", "职业规划"]),
            ("personal_brand", "如何打造个人品牌", "打造个人品牌：明确定位、输出内容（博客、演讲）、社交媒体经营、真诚一致、持续积累。", "career", ["个人品牌", "职业"]),
            ("teamwork_skills", "团队合作技巧", "团队合作：明确角色和目标、主动沟通、倾听尊重、靠谱负责、冲突建设性处理、欣赏差异。", "career", ["团队合作", "协作"]),
            ("conflict_resolution", "如何处理冲突", "处理冲突：冷静分析、换位思考、直接沟通、聚焦问题而非人、寻求共同点、必要时调解。", "career", ["冲突", "沟通", "解决"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_skills_knowledge(self):
        """常用技能"""
        entries = [
            ("excel_basics", "Excel基础操作", "Excel基础：单元格格式、数据输入、基本公式（SUM、AVERAGE）、排序筛选、制作图表、快捷键Ctrl+C/V/Z/S。", "skills", ["Excel", "办公软件", "表格"]),
            ("excel_formulas", "Excel常用公式", "Excel常用公式：SUM求和、IF条件、VLOOKUP查找、CONCATENATE合并、COUNTIF统计、DATE日期。", "skills", ["Excel", "公式"]),
            ("word_tips", "Word使用技巧", "Word技巧：样式和多级列表保持格式一致、导航窗格快速定位、审阅模式协作、快捷键、页眉页脚设置。", "skills", ["Word", "办公软件"]),
            ("ppt_design", "PPT设计原则", "PPT设计：内容简洁、图文并茂、统一配色字体、适当留白、动画适度、多用图表而非纯文字。", "skills", ["PPT", "演示", "设计"]),
            ("email_etiquette", "邮件礼仪", "邮件礼仪：主题明确、称呼得当、正文简洁、附件说明、署名完整、检查收件人、避免大附件。", "skills", ["邮件", "职场", "沟通"]),
            ("note_taking", "如何高效记笔记", "高效笔记方法：康奈尔笔记法、思维导图、关键词法、边读边记、课后整理、定期回顾。", "skills", ["笔记", "学习方法"]),
            ("speed_reading", "如何快速阅读", "快速阅读技巧：减少默读、扩展视野、关键词抓取、略读跳读、减少回读、练习提升。", "skills", ["阅读", "技巧", "效率"]),
            ("critical_thinking", "什么是批判性思维", "批判性思维：分析论证、评估证据、识别逻辑谬误、考虑多元视角、避免偏见、系统性思考。", "skills", ["批判性思维", "思维"]),
            ("problem_solving", "如何解决问题", "解决问题步骤：明确问题、分析原因、头脑风暴方案、评估选择、实施解决、复盘总结。", "skills", ["解决问题", "思维"]),
            ("communication_skills", "沟通技巧", "沟通技巧：倾听理解、先想后说、换位思考、清晰表达、反馈确认、非语言沟通（眼神表情）、处理分歧。", "skills", ["沟通", "技巧"]),
            ("writing_skills", "写作技巧", "写作技巧：明确目的读者、列提纲组织结构、开头吸引读者、论据支撑、数据图表可视化、检查修改。", "skills", ["写作", "表达"]),
            ("data_analysis_basics", "数据分析基础", "数据分析基础：明确问题、收集数据、清洗整理、分析趋势、得出结论、可视化呈现、报告建议。", "skills", ["数据分析", "分析"]),
            ("financial_literacy", "基础财务知识", "基础财务知识：收支平衡、复利原理、资产负载、预算管理、投资收益、风险分散。", "skills", ["财务", "理财"]),
            ("investment_basics", "投资入门知识", "投资入门：分散投资、长期持有、成本平摊、了解风险。高风险高收益，评估自己风险承受能力。", "skills", ["投资", "理财"]),
            ("project_management", "项目管理基础", "项目管理：明确目标和范围、制定计划和时间表、分配资源、管理风险、沟通协调、监控进度、收尾总结。", "skills", ["项目管理", "管理"]),
            ("agile_scrum", "敏捷和Scrum是什么", "敏捷是迭代式开发理念，Scrum是具体框架。角色：产品负责人、Scrum Master、开发团队。事件：Sprint规划、每日站会、Sprint评审、Sprint回顾。", "skills", ["敏捷", "Scrum", "项目管理"]),
            ("cybersecurity_basics", "基础网络安全", "网络安全基础：强密码、不点击可疑链接、定期更新软件、备份数据、使用VPN公共WiFi、警惕钓鱼邮件。", "skills", ["网络安全", "安全"]),
            ("privacy_protection", "如何保护隐私", "保护隐私：社交媒体设置、谨慎分享信息、加密重要文件、安全公共WiFi使用VPN、阅读应用权限。", "skills", ["隐私", "安全"]),
            ("critical_reading", "如何批判性阅读", "批判性阅读：区分事实观点、评估来源可靠性、追踪论证逻辑、思考偏见、联系已有知识。", "skills", ["批判性阅读", "阅读"]),
            ("decision_making", "如何做决策", "决策方法：明确决策类型、收集信息、列出选项、评估利弊、考虑风险和不确定性、果断决定、承担结果。", "skills", ["决策", "思维"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_entry(self, kid: str, question: str, answer: str, category: str, tags: List[str], confidence: float = 0.9):
        """添加条目"""
        if kid not in self.entries:
            self.entries[kid] = {
                'id': kid,
                'question': question,
                'answer': answer,
                'category': category,
                'tags': tags,
                'confidence': confidence,
            }

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """检索知识"""
        import re
        query_words = set(re.findall(r'[\w]+', query.lower()))
        results = []

        for kid, entry in self.entries.items():
            score = 0.0

            question_words = set(re.findall(r'[\w]+', entry['question'].lower()))
            overlap = query_words & question_words
            if overlap:
                score = len(overlap) / max(len(question_words), 1)

            if any(tag.lower() in query.lower() for tag in entry.get('tags', [])):
                score += 0.3

            if score > 0:
                result = entry.copy()
                result['relevance_score'] = min(score, 1.0)
                results.append(result)

        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:top_k]

    def save(self):
        """保存知识库"""
        try:
            Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
            data = {'entries': self.entries}
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  保存失败: {e}")

    def get_stats(self) -> Dict:
        """获取统计"""
        categories = defaultdict(int)
        for entry in self.entries.values():
            categories[entry.get('category', 'unknown')] += 1

        return {
            'total_entries': len(self.entries),
            'by_category': dict(categories),
        }


class KnowledgeIntegrationTester:
    """知识集成测试器"""

    def __init__(self, kb: KnowledgeBaseExpanded):
        self.kb = kb

    def run_coverage_test(self, test_queries: List[str]) -> Dict:
        """运行覆盖率测试"""
        hits = 0
        results = []

        for query in test_queries:
            retrieved = self.kb.entries.retrieve(query, top_k=3) if hasattr(self.kb, 'entries') else []
            if hasattr(self.kb, 'retrieve'):
                retrieved = self.kb.retrieve(query, top_k=3)

            hit = len(retrieved) > 0 and retrieved[0].get('relevance_score', 0) >= 0.2
            hits += 1 if hit else 0

            results.append({
                'query': query,
                'hit': hit,
                'retrieved_count': len(retrieved),
            })

        return {
            'total': len(test_queries),
            'hits': hits,
            'hit_rate': hits / len(test_queries) if test_queries else 0,
            'results': results,
        }


def main():
    """主函数"""
    print(f"\n{'='*70}")
    print("Stage 14.4.1: 知识库扩展")
    print(f"{'='*70}")

    kb = KnowledgeBaseExpanded()
    stats = kb.get_stats()

    print(f"\n{'='*70}")
    print("知识库统计")
    print(f"{'='*70}")
    print(f"  总条目: {stats['total_entries']}")
    print(f"  按类别分布:")
    for cat, count in stats.get('by_category', {}).items():
        print(f"    {cat}: {count}")

    print(f"\n{'='*70}")
    print("样本检索测试")
    print(f"{'='*70}")

    test_queries = [
        "Python是什么",
        "机器学习怎么入门",
        "HTTP协议的工作原理",
        "怎么戒烟",
        "Git怎么使用",
        "什么是区块链",
        "人工智能和机器学习的区别",
        "Docker容器是什么",
        "为什么天空是蓝色的",
        "如何保护心脏健康",
        "简历怎么写才好",
        "Excel公式有哪些",
        "项目管理怎么做",
        "如何缓解工作压力",
        "云计算是什么",
    ]

    for q in test_queries:
        results = kb.retrieve(q, top_k=2)
        if results:
            best = results[0]
            print(f"\n[{best['relevance_score']:.2f}] {q}")
            print(f"  → {best['question']}: {best['answer'][:50]}...")
        else:
            print(f"\n[0.00] {q}")
            print(f"  → 无匹配")

    print(f"\n{'='*70}")
    print("Stage 14.4.1 测试完成")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
