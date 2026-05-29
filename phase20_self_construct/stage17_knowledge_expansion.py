"""
Stage 17: 知识库扩展 (目标500+条)

扩展领域:
- 金融投资 (40+)
- 房产家居 (30+)
- 教育考试 (40+)
- 职业发展 (35+)
- 健康养生 (45+)
- 科技数码 (50+)
- 法律法规 (35+)
- 社会科学 (40+)
- 文化艺术 (40+)
- 生活方式 (45+)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
from typing import Dict, List
from collections import defaultdict


class Stage17KnowledgeExpansion:
    """Stage 17 知识库扩展"""

    VERSION = "Stage 17 Knowledge Expansion v1.0"

    def __init__(self, storage_path: str = "stage8_dataset/stage17_knowledge_expanded.json"):
        self.storage_path = storage_path
        self.entries = {}
        self.load()

    def load(self):
        """加载知识库"""
        if Path(self.storage_path).exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.entries = data.get('entries', {})
                print(f"  Stage 17知识库已加载: {len(self.entries)}条")
                return
            except Exception as e:
                print(f"  加载失败: {e}")

        self._init_base_knowledge()
        self._add_finance_investment()
        self._add_real_estate_home()
        self._add_education_exam()
        self._add_career_development()
        self._add_health_wellness()
        self._add_tech_digital()
        self._add_law_regulation()
        self._add_social_science()
        self._add_culture_art()
        self._add_lifestyle()

        self.save()
        print(f"  Stage 17知识库已初始化: {len(self.entries)}条")

    def _add_entry(self, kid: str, question: str, answer: str, category: str, tags: List[str]):
        """添加条目"""
        if kid not in self.entries:
            self.entries[kid] = {
                'id': kid,
                'question': question,
                'answer': answer,
                'category': category,
                'tags': tags,
                'confidence': 0.9,
            }

    def _init_base_knowledge(self):
        """基础投资知识"""
        entries = [
            ("stock_market_basics", "股票市场基础知识",
             "股票市场是发行和交易股票的市场。主要包括主板、创业板、科创板等。投资者通过买卖股票获得收益。",
             "finance", ["股票", "投资", "市场"]),
            ("bond_basics", "债券是什么",
             "债券是发行人为筹集资金向投资者出具的债务凭证。承诺按期支付利息和到期偿还本金。风险低于股票。",
             "finance", ["债券", "固收", "理财"]),
            ("etf_introduction", "ETF是什么",
             "交易型开放式指数基金(ETF)像股票一样在交易所交易，追踪特定指数。费用低、流动性好、分散风险。",
             "finance", ["ETF", "指数基金", "基金"]),
            ("mutual_fund_structure", "公募基金结构",
             "公募基金由基金管理人管理，投资人按份额享有收益和承担风险。包括股票型、债券型、混合型等。",
             "finance", ["公募基金", "基金"]),
            ("option_basic", "期权是什么",
             "期权是买卖双方约定的权利合约。买入期权支付权利金获得按约定价格买入/卖出资产的权利。",
             "finance", ["期权", "衍生品", "权利金"]),
            ("futures_contract", "期货合约",
             "期货是标准化合约，约定未来某一时间以约定价格买卖资产。用于套期保值或投机。",
             "finance", ["期货", "套保", "标准化"]),
            ("commodity_trading", "商品交易",
             "商品包括黄金、原油、农产品等。可通过现货、期货、ETF等渠道投资。",
             "finance", ["商品", "黄金", "原油"]),
            ("forex_basics", "外汇交易基础",
             "外汇市场是全球最大金融市场，货币对交易。汇率受经济数据、央行政策、地缘政治影响。",
             "finance", ["外汇", "汇率", "货币"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_finance_investment(self):
        """金融投资"""
        entries = [
            ("value_investing", "价值投资是什么",
             "价值投资寻找被低估的优质公司，长期持有。核心是找到护城河宽、管理层优秀、价格合理的好公司。",
             "finance", ["价值投资", "长期投资"]),
            ("growth_investing", "成长投资是什么",
             "成长投资聚焦高增长公司，即使估值偏高。关注营收增长、市场空间、创新能力。",
             "finance", ["成长投资", "高增长"]),
            ("dividend_investing", "股息投资策略",
             "股息投资选择稳定分红公司，既有现金流又可能增值。适合保守投资者。",
             "finance", ["股息", "分红", "现金流"]),
            ("asset_allocation", "资产配置原则",
             "资产配置指在不同资产类别间分配资金。如股票、债券、现金、不动产等。降低单一资产风险。",
             "finance", ["资产配置", "分散投资"]),
            ("risk_management_invest", "投资风险管理",
             "风险管理包括止损、仓位控制、分散投资等。设置合理的风险承受水平和止损点。",
             "finance", ["风险", "止损", "仓位"]),
            ("fundamental_analysis", "基本面分析",
             "基本面分析研究公司财务、行业地位、管理团队、竞争优势等。评估公司内在价值。",
             "finance", ["基本面", "财报", "估值"]),
            ("technical_analysis", "技术分析",
             "技术分析通过价格图表、成交量、趋势线等预测走势。适合短期交易。",
             "finance", ["技术分析", "K线", "趋势"]),
            ("portfolio_diversification", "投资组合多元化",
             "多元化投资在不同行业、地区、资产类别间分散。可降低非系统性风险。",
             "finance", ["多元化", "组合", "分散"]),
            ("dollar_cost_averaging", "定投策略",
             "定投是定期固定金额投资，平滑成本。适合没有时间研究市场的上班族。",
             "finance", ["定投", "定期投资"]),
            ("compound_interest", "复利效应",
             "复利是利滚利，长期增长惊人。早期开始投资，即使金额小也能积累大财富。",
             "finance", ["复利", "利滚利", "长期"]),
            ("factor_investing", "因子投资",
             "因子投资如价值、动量、质量、低波动等因子暴露获取超额收益。Smart Beta策略。",
             "finance", ["因子", "Smart Beta"]),
            ("esg_criteria", "ESG投资标准",
             "ESG指环境、社会责任、公司治理。投资者用这些标准筛选符合价值观的公司。",
             "finance", ["ESG", "可持续"]),
            ("hedge_fund_strategies", "对冲基金策略",
             "对冲基金使用Long/Short、宏观、量化等策略。追求绝对收益，适合高净值投资者。",
             "finance", ["对冲基金", "绝对收益"]),
            ("private_equity", "私募股权投资",
             "PE投资非上市公司股权，通过上市或并购退出。流动性低但潜在收益高。",
             "finance", ["PE", "私募", "股权"]),
            ("reit_investment", "REITs投资",
             "房地产信托投资基金(REITs)投资商业地产，提供稳定分红。门槛低、流动性好。",
             "finance", ["REITs", "房地产", "分红"]),
            ("tax_loss_harvesting", "税收亏损收割",
             "税损收割卖出亏损持仓抵税，同时买入相似资产保持市场暴露。有效降低税负。",
             "finance", ["税务", "亏损", "抵税"]),
            ("margin_trading", "融资融券",
             "融资融券是杠杆交易。融资买入或融券卖出，放大收益也放大风险。",
             "finance", ["融资", "融券", "杠杆"]),
            ("ipo_investing", "打新股",
             "IPO是首次公开募股。打新即参与新股认购，上市后通常有上涨空间。",
             "finance", ["IPO", "打新", "新股"]),
            ("sector_rotation", "行业轮动",
             "行业轮动策略根据经济周期在不同行业间转换。经济复苏买周期，衰退买防御。",
             "finance", ["行业轮动", "周期"]),
            ("quantitative_trading", "量化交易",
             "量化交易用数学模型和计算机程序执行交易。追求纪律性和系统化。",
             "finance", ["量化", "算法", "模型"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_real_estate_home(self):
        """房产家居"""
        entries = [
            ("first_home_purchase", "首次购房指南",
             "首次购房要明确预算、评估还款能力、了解贷款政策、注意地段和配套。新手建议买现房。",
             "realestate", ["购房", "首套", "贷款"]),
            ("mortgage_types", "房贷类型",
             "房贷包括商业贷款、公积金贷款、组合贷。固定利率vs浮动利率各有优劣。",
             "realestate", ["房贷", "利率", "贷款"]),
            ("rental_market", "租房市场",
             "租房要考虑地段、交通、配套、室友。签合同前检查房屋设施，明确押金条款。",
             "realestate", ["租房", "租金", "合同"]),
            ("property_tax", "房产税",
             "房产税按房产价值征收，用于地方财政。各国税率不同，影响持有成本。",
             "realestate", ["房产税", "税费"]),
            ("home_insurance", "房屋保险",
             "房屋保险保障房产和屋内财产。包括火灾、水灾、盗窃等保障。",
             "realestate", ["保险", "保障"]),
            ("renovation_budget", "装修预算",
             "装修前制定详细预算，留10-20%应急金。按需求分清主次，避免超支。",
             "realestate", ["装修", "预算"]),
            ("smart_home", "智能家居",
             "智能家居包括灯光、窗帘、安防、家电控制。通过语音或手机远程操控。",
             "realestate", ["智能家居", "IoT"]),
            ("energy_efficiency", "节能家居",
             "节能家居降低能耗，如LED灯、隔热材料、太阳能板。减少电费又环保。",
             "realestate", ["节能", "环保"]),
            ("home_staging", "房屋美化",
             "房屋美化通过布置、装饰提升卖相。清理杂物、改善采光、增加绿植。",
             "realestate", ["房屋美化", "出售"]),
            ("landlord_rights", "房东权益",
             "房东有权按时收租、要求合理使用房屋。但不能随意进入，需给租客合理通知。",
             "realestate", ["房东", "权益"]),
            ("tenant_rights", "租客权益",
             "租客有权享有适宜居住环境、保护隐私。押金退还受法律保护。",
             "realestate", ["租客", "权益"]),
            ("commercial_real_estate", "商业地产",
             "商业地产包括写字楼、商铺、工业地产。租金回报稳定，但流动性差。",
             "realestate", ["商业地产", "商铺"]),
            ("co_working_space", "联合办公",
             "联合办公提供灵活工位和共享设施。适合自由职业者和初创公司。",
             "realestate", ["联合办公", "共享"]),
            ("elderly_housing", "养老房产",
             "养老房产考虑医疗配套、无障碍设施、社区服务。郊区vs城市各有利弊。",
             "realestate", ["养老", "老年"]),
            ("investment_property", "投资房产",
             "投资房产期望租金收入和资本增值。要考虑空置期、维修成本、税费。",
             "realestate", ["投资", "出租"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_education_exam(self):
        """教育考试"""
        entries = [
            ("college_application", "高考志愿填报",
             "志愿填报要综合考虑分数、排名、专业兴趣、就业前景。冲稳保策略合理安排。",
             "education", ["高考", "志愿", "填报"]),
            ("graduate_school", "考研攻略",
             "考研包括初试和复试。选校要评估竞争激烈度，了解导师研究方向。",
             "education", ["考研", "研究生"]),
            ("study_abroad", "留学申请",
             "留学申请需要语言成绩、GPA、文书、推荐信。提前1-2年开始准备。",
             "education", ["留学", "申请"]),
            ("online_learning", "在线学习平台",
             "Coursera、edX、慕课等平台提供顶尖大学课程。可获得证书或学位。",
             "education", ["在线学习", "慕课"]),
            ("language_exams", "语言考试",
             "英语考试如托福、雅思、GRE。出题有规律，针对性训练可提分。",
             "education", ["托福", "雅思", "语言"]),
            ("certifications", "职业认证",
             "CPA、CFA、PMP等职业认证提升竞争力。有些是入职门槛。",
             "education", ["认证", "职业资格"]),
            ("coding_bootcamp", "编程训练营",
             "编程训练营密集培训3-6个月，适合转行者。就业导向，但质量参差。",
             "education", ["编程", "训练营"]),
            ("early_childhood", "早期教育",
             "0-6岁是大脑发育关键期。游戏、阅读、社交促进认知和情感发展。",
             "education", ["早教", "儿童"]),
            ("special_education", "特殊教育",
             "特殊教育服务于有学习障碍或残疾的儿童。IEP个别化教育计划很重要。",
             "education", ["特殊教育", "IEP"]),
            ("online_degree", "在线学位",
             "在线学位越来越被认可。适合在职人员，但需要自律和时间管理。",
             "education", ["在线学位", "学历"]),
            ("tuition_plan", "教育规划",
             "教育金需要长期规划。教育储蓄账户、教育保险、定投等工具可用。",
             "education", ["教育金", "储蓄"]),
            (" standardized_test", "标准化考试技巧",
             "SAT、ACT等标准化考试有技巧。熟悉题型、控制节奏、猜题策略。",
             "education", ["SAT", "考试技巧"]),
            ("college_financial_aid", "大学助学金",
             "助学金包括奖学金、助学金、贷款、勤工俭学。了解申请截止日期。",
             "education", ["助学金", "奖学金"]),
            ("study_skills", "学习方法",
             "高效学习包括主动回忆、交错练习、间隔复习。费曼学习法很有效。",
             "education", ["学习", "方法"]),
            ("time_management", "时间管理",
             "时间管理技巧包括番茄工作法、GTD、四象限法则。提高效率减少拖延。",
             "education", ["时间管理", "效率"]),
            ("critical_thinking", "批判性思维",
             "批判性思维分析论证、识别偏见、评估证据。是大学核心技能。",
             "education", ["批判性思维", "分析"]),
            ("creative_writing", "创意写作",
             "创意写作包括小说、诗歌、剧本等。阅读经典、勤练笔、获取反馈。",
             "education", ["写作", "创意"]),
            ("public_speaking", "演讲技能",
             "演讲技能包括结构设计、肢体语言、声音控制。多练习可克服紧张。",
             "education", ["演讲", "表达"]),
            ("research_methods", "研究方法",
             "研究方法包括定量、定性、混合方法。选题、文献综述、数据分析是基础。",
             "education", ["研究", "方法"]),
            ("professional_development", "职业发展学习",
             "LinkedIn Learning、Udemy等平台提供职业技能课程。持续学习保持竞争力。",
             "education", ["职业发展", "技能"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_career_development(self):
        """职业发展"""
        entries = [
            ("resume_writing", "简历撰写",
             "简历要简洁、有针对性、量化成果。突出与岗位相关的技能和经验。",
             "career", ["简历", "求职"]),
            ("interview_prep", "面试准备",
             "面试前研究公司、了解岗位职责STAR法则回答。准备反问环节的问题。",
             "career", ["面试", "准备"]),
            ("salary_negotiation", "薪资谈判",
             "谈判前做市场调研、了解公司薪酬结构。突出自己价值，敢于开口。",
             "career", ["薪资", "谈判"]),
            ("career_transition", "转行攻略",
             "转行要评估可迁移技能、补充新领域知识、从低门槛切入。保持开放心态。",
             "career", ["转行", "转型"]),
            ("networking", "人脉经营",
             "人脉是职业发展的重要资源。参加行业活动、维护LinkedIn、真诚互助。",
             "career", ["人脉", "社交"]),
            ("leadership", "领导力培养",
             "领导力包括愿景、沟通、决策、授权。不只是职位，是影响力。",
             "career", ["领导力", "管理"]),
            ("remote_work", "远程办公",
             "远程办公需要自律、沟通工具、清晰边界。避免孤独和过度工作。",
             "career", ["远程", "办公"]),
            ("freelance", "自由职业",
             "自由职业需要客户开发、报价、时间管理能力。收入不稳定但时间自由。",
             "career", ["自由职业", " freelance"]),
            ("entrepreneurship", "创业指南",
             "创业包括找准市场痛点MVP验证、融资、组建团队。失败率很高但回报也大。",
             "career", ["创业", "企业家"]),
            ("work_life_balance", "工作生活平衡",
             "平衡需要设定边界、学会说不、预留恢复时间。健康和家庭同样重要。",
             "career", ["平衡", "健康"]),
            ("career_growth", "职业晋升",
             "晋升需要持续产出、学习新技能、建立可见度。主动争取机会。",
             "career", ["晋升", "升职"]),
            ("job_search", "求职策略",
             "求职要广撒网、内推优先、跟进进度。保持积极心态，过程需要耐心。",
             "career", ["求职", "找工作"]),
            ("skill_upgrading", "技能升级",
             "技术变革快，需要持续学习。在线课程、证书、项目经验都是途径。",
             "career", ["技能", "学习"]),
            ("workplace_communication", "职场沟通",
             "职场沟通要清晰、专业、考虑受众。书面表达和口头表达同样重要。",
             "career", ["沟通", "职场"]),
            ("conflict_resolution", "冲突处理",
             "冲突处理先冷静、了解对方立场、寻找共同点。建设性解决增进关系。",
             "career", ["冲突", "解决"]),
            ("performance_review", "绩效评估",
             "绩效评估前准备成就案例、设定新目标。主动反馈而非被动接受。",
             "career", ["绩效", "评估"]),
            ("job_benefits", "员工福利",
             "福利包括健康保险、养老金、带薪休假。有些公司提供灵活福利账户。",
             "career", ["福利", "员工"]),
            ("burnout_recovery", "职业倦怠",
             "倦怠信号包括持续疲劳、效率下降、去人格化。休息、调整期望、寻求支持。",
             "career", ["倦怠", "压力"]),
            ("career_planning", "职业规划",
             "职业规划要思考短期和长期目标、兴趣与市场结合。定期复盘调整。",
             "career", ["规划", "职业"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_health_wellness(self):
        """健康养生"""
        entries = [
            ("nutrition_basics", "营养学基础",
             "均衡饮食包括碳水、蛋白质、脂肪、维生素、矿物质。彩虹饮食法保证多样性。",
             "health", ["营养", "饮食", "均衡"]),
            ("exercise_guidelines", "运动指南",
             "成人每周150分钟中等强度有氧运动，加上力量训练。循序渐进避免伤害。",
             "health", ["运动", "锻炼", "健身"]),
            ("mental_health", "心理健康",
             "心理健康与身体健康同样重要。减压、社交、寻求帮助是维护方式。",
             "health", ["心理", "健康"]),
            ("sleep_quality", "睡眠质量",
             "成年人需要7-9小时睡眠。固定作息、黑暗环境、午后避咖啡因改善睡眠。",
             "health", ["睡眠", "休息"]),
            ("stress_management", "压力管理",
             "压力管理包括运动、呼吸练习、时间管理、寻求支持。慢性压力有害健康。",
             "health", ["压力", "管理"]),
            ("chronic_disease", "慢性病预防",
             "慢性病如心脏病、糖尿病、癌症与生活方式密切相关。预防重于治疗。",
             "health", ["慢性病", "预防"]),
            ("gut_health", "肠道健康",
             "肠道健康影响免疫、情绪、大脑功能。益生菌、纤维、发酵食品有益。",
             "health", ["肠道", "益生菌"]),
            ("immune_system", "免疫系统",
             "免疫系统保护免受感染。睡眠、运动、均衡营养、减压增强免疫力。",
             "health", ["免疫", "抵抗力"]),
            ("heart_health", "心脏健康",
             "心脏健康需要控制血压血脂、不吸烟、适度运动。心脏病是可预防的。",
             "health", ["心脏", "心血管"]),
            ("bone_health", "骨骼健康",
             "骨骼健康需要钙、维生素D承重运动。骨质疏松早期无明显症状。",
             "health", ["骨骼", "骨质疏松"]),
            ("eye_health", "眼睛健康",
             "护眼包括20-20-20法则、多户外活动、控制屏幕时间。定期检查视力。",
             "health", ["眼睛", "视力"]),
            ("dental_health", "口腔健康",
             "口腔健康影响全身。刷牙两次、牙线、洗牙半年一次。",
             "health", ["口腔", "牙齿"]),
            ("skin_care", "护肤知识",
             "护肤包括清洁、保湿、防晒。根据肤质选择产品，避免过度护肤。",
             "health", ["护肤", "皮肤"]),
            ("weight_management", "体重管理",
             "健康体重管理是长期过程。热量平衡、习惯改变比短期节食重要。",
             "health", ["体重", "减肥"]),
            ("detox_myths", "排毒迷思",
             "人体有肝肾自动排毒，无需额外排毒产品。健康生活方式才是关键。",
             "health", ["排毒", "迷思"]),
            ("supplements", "营养补充剂",
             "补充剂不能替代食物。仅在特定情况如维生素D、贫血时补充。",
             "health", ["补充剂", "维生素"]),
            ("hydration", "饮水指南",
             "每天约2升水，包括食物中的水分。运动、高温时增加摄入。",
             "health", ["饮水", "水分"]),
            ("meditation", "冥想练习",
             "冥想减少压力、改善专注。每天10-20分钟即可产生效果。",
             "health", ["冥想", "放松"]),
            ("posture_health", "姿势健康",
             "不良姿势导致颈腰痛。每小时起身活动，注意工作站 ergonomics。",
             "health", ["姿势", "颈腰痛"]),
            ("aging_health", "健康老龄化",
             "健康老龄化包括身体活动、认知刺激、社交参与。预防失能失智。",
             "health", ["老龄化", "老年"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_tech_digital(self):
        """科技数码"""
        entries = [
            ("smartphone_selection", "智能手机选购",
             "选购看处理器、屏幕、相机、续航、系统生态。按需求和预算选择。",
             "tech", ["手机", "选购"]),
            ("laptop_buying", "笔记本电脑选购",
             "办公选轻薄本高性能，游戏选游戏本，设计选工作站。CPU、内存、SSD重要。",
             "tech", ["笔记本", "电脑"]),
            ("data_backup", "数据备份",
             "3-2-1原则：3份备份、2种介质、1份异地。云备份和本地备份结合。",
             "tech", ["备份", "数据"]),
            ("cybersecurity_basics", "网络安全基础",
             "强密码、双因素认证、及时更新、不点可疑链接。预防网络攻击。",
             "tech", ["安全", "密码"]),
            ("privacy_protection", "隐私保护",
             "检查应用权限、限制社交媒体分享、使用隐私浏览模式。数据最小化原则。",
             "tech", ["隐私", "保护"]),
            ("cloud_storage", "云存储",
             "iCloud、Google Drive、Dropbox等。选择容量、价格、隐私政策。",
             "tech", ["云存储", "同步"]),
            ("wifi_security", "WiFi安全",
             "设置强密码、WPA3加密、关闭WPS、定期更换密码。公WiFi用VPN。",
             "tech", ["WiFi", "无线"]),
            ("app_permissions", "应用权限",
             "应用权限要最小化。地图需要位置，但游戏不需要通讯录。定期检查。",
             "tech", ["权限", "应用"]),
            ("battery_health", "电池健康",
             "浅充浅放、避免过充过放、远离极端温度。锂电池寿命有限。",
             "tech", ["电池", "手机"]),
            ("smartwatch_features", "智能手表功能",
             "智能手表包括健康监测、运动追踪、通知提醒、支付功能。按需选择。",
             "tech", ["手表", "可穿戴"]),
            ("tv_technology", "电视技术",
             "LED、QLED、OLED、MicroLED不同技术。屏幕尺寸、分辨率、HDR重要。",
             "tech", ["电视", "屏幕"]),
            ("gaming_console", "游戏主机",
             "PlayStation、Xbox、Switch各有所长。独占游戏、生态是选择因素。",
             "tech", ["游戏", "主机"]),
            ("noise_cancelling", "降噪耳机",
             "主动降噪通过反向声波消除噪音。适合通勤、飞机、长途旅行。",
             "tech", ["降噪", "耳机"]),
            ("smart_home_setup", "智能家居设置",
             "智能家居生态选择Matter或HomeKit。灯泡、插座、摄像头是入门。",
             "tech", ["智能家居", "IoT"]),
            ("5g_explained", "5G explained",
             "5G是第五代移动网络，更高速度更低延迟。适合AR/VR、云游戏。",
             "tech", ["5G", "网络"]),
            ("wifi_6_benefits", "WiFi 6优势",
             "WiFi 6更高速度、更多设备连接、更低延迟。需设备支持。",
             "tech", ["WiFi 6", "路由器"]),
            ("ssd_vs_hdd", "SSD vs HDD",
             "SSD比HDD快10-100倍，但贵。系统盘选SSD，仓库盘可用HDD。",
             "tech", ["SSD", "硬盘"]),
            ("graphics_card", "显卡选择",
             "游戏和创作需要好显卡。NVIDIA和AMD是主流。按预算和用途选择。",
             "tech", ["显卡", "GPU"]),
            ("programming_basics", "编程入门",
             "Python适合初学者，JavaScript做网页，Java企业应用。选一门开始。",
             "tech", ["编程", "代码"]),
            ("ai_tools", "AI工具应用",
             "ChatGPT写作、Midjourney作图、Copilot编程。AI提高效率但需验证。",
             "tech", ["AI", "工具"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_law_regulation(self):
        """法律法规"""
        entries = [
            ("contract_basics", "合同法基础",
             "合同需要要约、承诺、当事人资格、合法目的。书面合同更安全。",
             "law", ["合同", "法律"]),
            ("intellectual_property", "知识产权",
             "知识产权包括专利、商标、版权、商业秘密。保护创新成果。",
             "law", ["知识产权", "专利"]),
            ("consumer_protection", "消费者权益",
             "消费者有权安全商品、知情选择、诚实广告、损害赔偿。七日无理由退货。",
             "law", ["消费者", "权益"]),
            ("labor_law", "劳动法",
             "劳动法规定工作时间、工资支付、社会保险、解雇保护。维护雇员权益。",
             "law", ["劳动法", "劳工"]),
            ("privacy_law", "隐私法",
             "个人信息受法律保护。收集要同意、使用有目的、保存要安全。",
             "law", ["隐私", "个人信息"]),
            ("company_registration", "公司注册",
             "注册公司要选类型、核名、提交材料、领取执照。有限责任降低风险。",
             "law", ["公司", "注册"]),
            ("tax_law_basics", "税法基础",
             "个人所得税、企业所得税、增值税等。了解合法减税途径。",
             "law", ["税法", "税务"]),
            ("traffic_law", "交通法规",
             "酒驾、超速、闯红灯扣分罚款。重大事故承担刑事责任。",
             "law", ["交通", "法规"]),
            ("property_law", "物权法",
             "物权包括所有权、用益物权、担保物权。不动产需登记生效。",
             "law", ["物权", "产权"]),
            ("family_law", "婚姻法",
             "婚姻法规定结婚离婚、财产分割、子女抚养。婚前协议可约定财产。",
             "law", ["婚姻", "离婚"]),
            ("criminal_law_basics", "刑法基础",
             "犯罪包括故意和过失。正当防卫、紧急避险是免责事由。",
             "law", ["刑法", "犯罪"]),
            ("civil_lawsuit", "民事诉讼",
             "民事纠纷可起诉维权。证据、时效、管辖是要素。可申请财产保全。",
             "law", ["诉讼", "打官司"]),
            ("arbitration", "仲裁",
             "仲裁是私密、高效的争议解决方式。商事合同常约定仲裁条款。",
             "law", ["仲裁", "争议"]),
            ("compliance", "合规",
             "企业合规包括反贿赂、数据安全、劳动权益等。违规有严重后果。",
             "law", ["合规", "企业"]),
            ("inheritance_law", "继承法",
             "继承包括法定继承和遗嘱继承。遗嘱可自由处分财产，需公证。",
             "law", ["继承", "遗嘱"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_social_science(self):
        """社会科学"""
        entries = [
            ("sociology_basics", "社会学基础",
             "社会学研究社会结构、行为、变迁。揭示不平等和社会问题根源。",
             "social", ["社会学", "社会"]),
            ("psychology_intro", "心理学入门",
             "心理学研究人类行为和心理过程。分支包括发展、认知、社会、临床。",
             "social", ["心理学", "行为"]),
            ("economics_basics", "经济学基础",
             "经济学研究资源配置。微观关注个人选择，宏观关注整体经济运行。",
             "social", ["经济", "供求"]),
            ("political_science", "政治学",
             "政治学研究政府、权力、政策。民主、专制、集权是政体类型。",
             "social", ["政治", "政府"]),
            ("anthropology", "人类学",
             "人类学研究人类文化和社会。文化相对主义是核心原则。",
             "social", ["人类学", "文化"]),
            ("linguistics", "语言学",
             "语言学研究语言结构、使用、演变。方言、标准化、语言保护是议题。",
             "social", ["语言", "语言学"]),
            ("philosophy_basics", "哲学入门",
             "哲学探讨存在、知识、价值、理性。经典问题包括自由意志、伦理学。",
             "social", ["哲学", "思考"]),
            ("history_study", "历史研究",
             "历史研究过去以理解现在。史料批判、多视角是历史方法。",
             "social", ["历史", "研究"]),
            ("media_literacy", "媒体素养",
             "媒体素养指批判性分析信息源、识别偏见、核实事实。防 misinformation。",
             "social", ["媒体", "素养"]),
            ("propaganda_recognition", "识别宣传",
             "宣传特征包括情绪操控、单向信息、来源模糊。批判性思维防洗脑。",
             "social", ["宣传", "批判"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_culture_art(self):
        """文化艺术"""
        entries = [
            ("music_theory", "乐理基础",
             "乐理包括音符、节拍、音阶、和弦。理解乐理提升演奏和创作能力。",
             "culture", ["音乐", "乐理"]),
            ("painting_techniques", "绘画技法",
             "绘画技法包括素描、油画、水彩、素描。明暗、色彩、构图是要素。",
             "culture", ["绘画", "美术"]),
            ("photography_basics", "摄影基础",
             "摄影三要素：曝光、对焦、构图。光圈、快门、ISO是曝光铁三角。",
             "culture", ["摄影", "相机"]),
            ("film_appreciation", "电影鉴赏",
             "电影分析包括镜头语言、剪辑、叙事、表演。从技术到艺术理解。",
             "culture", ["电影", "导演"]),
            ("literature_analysis", "文学分析",
             "文学分析关注主题、人物、叙事技巧、象征意义。细读是方法。",
             "culture", ["文学", "分析"]),
            ("calligraphy", "书法艺术",
             "书法是东方艺术。真草隶篆各体，点画结构有讲究。",
             "culture", ["书法", "汉字"]),
            ("dance_styles", "舞蹈风格",
             "舞蹈包括古典芭蕾、现代舞、爵士、街舞。每种有独特美学和训练。",
             "culture", ["舞蹈", "舞蹋"]),
            ("architecture_appreciation", "建筑鉴赏",
             "建筑美学包括形式、功能、结构。罗马、哥特、现代主义各具特色。",
             "culture", ["建筑", "设计"]),
            ("culinary_arts", "烹饪艺术",
             "烹饪是艺术也是科学。刀工、火候、调味是基础。五味调和是目标。",
             "culture", ["烹饪", "美食"]),
            ("tea_culture", "茶文化",
             "茶文化在中日韩各有特色。品茶讲究水、温度、器具、心境。",
             "culture", ["茶", "文化"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_lifestyle(self):
        """生活方式"""
        entries = [
            ("minimalism", "极简主义",
             "极简主义减少物质拥有，专注真正重要的事。定期断舍离是方法。",
             "lifestyle", ["极简", "断舍离"]),
            ("productivity_hacks", "效率技巧",
             "效率技巧包括时间块、番茄钟、两分钟规则。找到适合自己的系统。",
             "lifestyle", ["效率", "生产力"]),
            ("travel_hacking", "旅行技巧",
             "旅行技巧包括淡季出行、常旅客计划、信用卡积分。降低旅行成本。",
             "lifestyle", ["旅行", "机票"]),
            ("gardening", "园艺",
             "园艺是放松身心的活动。了解植物习性、光照、水分、土壤。",
             "lifestyle", ["园艺", "种植"]),
            ("pet_care", "宠物养护",
             "宠物需要适当饮食、运动、医疗。领养代替购买，负责到底。",
             "lifestyle", ["宠物", "养宠物"]),
            ("coffee_brewing", "咖啡冲泡",
             "手冲咖啡讲究研磨、水温、萃取时间。精品咖啡风味层次丰富。",
             "lifestyle", ["咖啡", "冲泡"]),
            ("wine_appreciation", "葡萄酒鉴赏",
             "葡萄酒品鉴看色泽、香气、口感。品种、产区、年份是风格因素。",
             "lifestyle", ["葡萄酒", "品酒"]),
            ("yoga_practice", "瑜伽练习",
             "瑜伽结合身体姿势、呼吸控制、冥想。提升柔韧、力量、平衡。",
             "lifestyle", ["瑜伽", "冥想"]),
            ("hiking_prep", "徒步准备",
             "徒步准备包括路线规划、装备清单、天气预报。安全第一。",
             "lifestyle", ["徒步", "登山"]),
            ("photography_tips", "摄影技巧",
             "摄影技巧包括黄金分割、引导线、帧中帧。多观察多实践。",
             "lifestyle", ["摄影", "技巧"]),
            ("diy_repair", "DIY维修",
             "基本维修技能如换灯泡、通马桶、简单家具组装。省钱又实用。",
             "lifestyle", ["DIY", "维修"]),
            ("volunteering", "志愿服务",
             "志愿服务回馈社区。找到感兴趣的事业，贡献时间和技能。",
             "lifestyle", ["志愿", "公益"]),
            ("budget_travel", "穷游指南",
             "穷游选择青旅、市井美食、公共交通。体验当地文化而非奢华。",
             "lifestyle", ["穷游", "背包客"]),
            ("digital_detox", "数字排毒",
             "数字排毒定期远离电子设备。恢复专注力、改善睡眠、增进人际。",
             "lifestyle", ["数字排毒", "断网"]),
            ("hobby_development", "兴趣爱好培养",
             "培养兴趣爱好丰富生活。学习乐器、绘画、园艺、手工等。",
             "lifestyle", ["爱好", "兴趣"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def save(self):
        """保存知识库"""
        try:
            Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
            data = {'entries': self.entries}
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  保存失败: {e}")

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

    def get_stats(self) -> Dict:
        """获取统计"""
        categories = defaultdict(int)
        for entry in self.entries.values():
            categories[entry.get('category', 'unknown')] += 1
        return {
            'total_entries': len(self.entries),
            'by_category': dict(categories),
        }


def main():
    """主函数"""
    print(f"\n{'='*70}")
    print("Stage 17: 知识库扩展")
    print(f"{'='*70}")

    kb = Stage17KnowledgeExpansion()
    stats = kb.get_stats()

    print(f"\n{'='*70}")
    print("Stage 17 知识库统计")
    print(f"{'='*70}")
    print(f"  总条目: {stats['total_entries']}")
    print(f"  按类别分布:")
    for cat, count in sorted(stats.get('by_category', {}).items()):
        print(f"    {cat}: {count}")

    print(f"\n{'='*70}")
    print("新领域检索测试")
    print(f"{'='*70}")

    test_queries = [
        "如何投资股票",
        "首次购房注意什么",
        "高考志愿怎么填",
        "简历怎么写",
        "怎么保持健康",
        "手机选购指南",
        "合同法基础",
        "社会学是什么",
        "极简主义生活",
        "瑜伽练习",
    ]

    for q in test_queries:
        results = kb.retrieve(q, top_k=2)
        if results and results[0]['relevance_score'] >= 0.2:
            print(f"\n[{results[0]['relevance_score']:.2f}] {q}")
            print(f"  → {results[0]['question']}")
        else:
            print(f"\n[0.00] {q} → 无匹配")

    print(f"\n{'='*70}")
    print("Stage 17 知识扩展测试完成")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
