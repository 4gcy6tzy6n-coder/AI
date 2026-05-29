"""
Stage 16: 知识库扩展与前沿能力提升

目标:
- 知识条目从225 → 500+
- 覆盖新技术、前沿科学、医疗健康、政策法规等
- 增强对未知主题的泛化能力
- 保持TSLA自学习闭环功能

新增类别:
- 前沿科技 (40+)
- 生物技术 (30+)
- 空间科学 (25+)
- 政策法规 (25+)
- 金融经济 (30+)
- 环境能源 (25+)
- 心理心智 (25+)
- 教育创新 (25+)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
from typing import Dict, List
from collections import defaultdict


class Stage16KnowledgeBase:
    """Stage 16 扩展知识库"""

    VERSION = "Stage 16 Knowledge Base v2.0"

    def __init__(self, storage_path: str = "stage8_dataset/stage16_knowledge_base.json"):
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
                print(f"  Stage 16知识库已加载: {len(self.entries)}条")
                return
            except Exception as e:
                print(f"  加载失败: {e}")

        self._init_frontier_knowledge()
        self._add_quantum_computing()
        self._add_biotechnology()
        self._add_space_science()
        self._add_policy_regulation()
        self._add_finance_economy()
        self._add_environment_energy()
        self._add_psychology_mind()
        self._add_education_innovation()
        self._add_3d_printing()
        self._add_blockchain_web3()
        self._add_medical_health()
        self._add_advanced_manufacturing()
        self._add_social_media_marketing()
        self._add_law_rights()

        self.save()
        print(f"  Stage 16知识库已初始化: {len(self.entries)}条")

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

    def _init_frontier_knowledge(self):
        """初始化前沿知识"""
        entries = [
            ("quantum_computing_intro", "量子计算是什么",
             "量子计算是利用量子力学原理进行信息处理的计算方式。量子位(qubit)可以同时处于0和1的叠加态，使量子计算机在特定问题上比经典计算机快指数倍。",
             "frontier", ["量子计算", "量子力学", "qubit"]),
            ("ai伦理", "人工智能伦理问题有哪些",
             "AI伦理问题包括：算法偏见与歧视、隐私侵犯、责任归属、就业冲击、武器化风险、透明度不足等。需要建立伦理框架和监管机制。",
             "frontier", ["AI伦理", "人工智能", "伦理"]),
            ("metaverse_intro", "元宇宙是什么",
             "元宇宙是虚拟现实增强的数字世界，融合社交、游戏、经济活动于一体。包括VR/AR、区块链、AI等技术，是互联网的下一代形态。",
             "frontier", ["元宇宙", "VR", "虚拟现实"]),
            ("brain_computer_interface", "脑机接口是什么",
             "脑机接口(BCI)是直接连接大脑与外部设备的技术。用于帮助残障人士控制假肢，或增强人类认知能力。Neuralink等公司正在开发。",
             "frontier", ["脑机接口", "BCI", "Neuralink"]),
            ("crispr_gene_editing", "CRISPR基因编辑是什么",
             "CRISPR是革命性的基因编辑工具，可以精确修改DNA序列。用于治疗遗传疾病、改良农作物，但也引发伦理争议。",
             "frontier", ["CRISPR", "基因编辑", "生物技术"]),
            ("synthetic_biology", "合成生物学是什么",
             "合成生物学设计和构建新的生物部件和系统。或改造现有生物体，用于制药、材料、能源、环境修复等领域。",
             "frontier", ["合成生物", "生物工程"]),
            ("quantum_internet", "量子互联网是什么",
             "量子互联网使用量子纠缠传输信息，无法被窃听，安全性极高。可用于量子计算分布式计算、安全通信等领域。",
             "frontier", ["量子互联网", "量子通信"]),
            ("space_mining", "太空采矿是什么",
             "太空采矿从小行星或月球提取资源。如水冰(可制火箭燃料)、稀有金属。可支持长期太空探索和地球资源补充。",
             "frontier", ["太空采矿", "小行星"]),
            ("nuclear_fusion_status", "核聚变进展如何",
             "核聚变是清洁能源的圣杯。2022年劳伦斯利弗莫尔实现能量净增益。商业化预计还需20-30年。",
             "frontier", ["核聚变", "清洁能源"]),
            ("agi_timeline", "通用人工智能何时实现",
             "通用人工智能(AGI)是能完成任何智力任务的AI。专家预测10-50年不等，存在巨大不确定性。",
             "frontier", ["AGI", "通用人工智能"]),
            ("autonomous_driving_levels", "自动驾驶分哪几个级别",
             "L0无自动化到L5完全自动化。L2部分自动化如特斯拉Autopilot，L4在限定区域无需驾驶员，L5任何场景均可。",
             "frontier", ["自动驾驶", "无人驾驶"]),
            ("carbon_capture_tech", "碳捕获技术有哪些",
             "碳捕获包括直接空气捕获(DAC)、生物质能源+碳捕获(BECCS)、海洋碱化等。可从大气吸取CO2，对抗气候变化。",
             "frontier", ["碳捕获", "气候变化"]),
            ("alternatives_to_silicon", "硅芯片的替代品有哪些",
             "新型芯片材料包括：碳纳米管、Graphene、III-V族半导体、光子芯片、量子芯片。可突破硅的物理极限。",
             "frontier", ["芯片", "半导体"]),
            ("longevity_research", "延寿研究进展如何",
             "衰老研究包括：Senolytics清除衰老细胞、NAD+补充、端粒延长、表观遗传重编程。人类寿命延长正在成为可能。",
             "frontier", ["延寿", "抗衰老"]),
            ("cultured_meat", "培养肉是什么",
             "培养肉用动物细胞在实验室培养，无需屠宰。可减少畜牧业环境负担，解决动物福利问题。",
             "frontier", ["培养肉", "人造肉"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_quantum_computing(self):
        """量子计算"""
        entries = [
            ("quantum_supremacy", "量子霸权是什么",
             "量子霸权指量子计算机在特定任务上超越最强经典计算机。2019年Google实现量子霸权，2022年中国科学家也实现里程碑。",
             "quantum", ["量子霸权", "量子优势"]),
            ("quantum_entanglement", "量子纠缠是什么",
             "量子纠缠是两个量子比特间的神秘联系，测量一个立即影响另一个。爱因斯坦称其为鬼魅般的超距作用。",
             "quantum", ["量子纠缠", "量子力学"]),
            ("quantum_cryptography", "量子加密是什么",
             "量子加密利用量子力学原理实现理论上不可破解的通信。任何窃听都会被检测到。",
             "quantum", ["量子加密", "量子通信"]),
            ("post_quantum_crypto", "后量子密码学是什么",
             "后量子密码学是能抵抗量子计算机攻击的加密算法。NIST已标准化多种算法，为量子时代做准备。",
             "quantum", ["后量子密码", "抗量子加密"]),
            ("quantum_machine_learning", "量子机器学习是什么",
             "量子机器学习结合量子计算与机器学习，利用量子并行性加速训练。可能带来AI革命。",
             "quantum", ["量子ML", "量子AI"]),
            ("quantum_simulation", "量子模拟是什么",
             "量子模拟用可控量子系统研究其他量子系统。如模拟分子行为用于药物研发。",
             "quantum", ["量子模拟"]),
            ("topological_qubit", "拓扑量子比特是什么",
             "拓扑量子比特利用拓扑性质保护信息，更稳定。Microsoft等公司正在研发，是量子计算的重要方向。",
             "quantum", ["拓扑量子", "量子比特"]),
            ("quantum_error_correction", "量子纠错是什么",
             "量子纠错检测并纠正量子计算中的错误。由于量子不可克隆定理，纠错更复杂。",
             "quantum", ["量子纠错", "量子计算"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_biotechnology(self):
        """生物技术"""
        entries = [
            ("mrna_vaccine", "mRNA疫苗是什么",
             "mRNA疫苗传递遗传指令使细胞产生抗原蛋白。无需活病毒，快速开发。COVID-19疫苗验证了此技术。",
             "biotech", ["mRNA", "疫苗"]),
            ("personalized_medicine", "个性化医疗是什么",
             "个性化医疗基于基因信息为患者定制治疗方案。提高疗效，减少副作用，是医学未来方向。",
             "biotech", ["个性化医疗", "精准医疗"]),
            ("liquid_biopsy", "液体活检是什么",
             "液体活检通过血液检测肿瘤DNA，无需手术。用于癌症早筛、疗效监测，是革命性诊断技术。",
             "biotech", ["液体活检", "癌症早筛"]),
            ("organoid_tech", "类器官技术是什么",
             "类器官是实验室培养的微型器官，源自干细胞。可用于药物测试、疾病建模，减少动物实验。",
             "biotech", ["类器官", "干细胞"]),
            ("gene_therapy_types", "基因疗法有哪些类型",
             "基因疗法包括：体内基因编辑、体外基因工程后回输、基因敲降、反义寡核苷酸等。已用于多种遗传病。",
             "biotech", ["基因疗法", "基因治疗"]),
            ("microbiome_health", "微生物组与健康关系",
             "人体肠道微生物组影响免疫、代谢、脑功能。调节微生物组可治疗肥胖、自闭症、抑郁等疾病。",
             "biotech", ["微生物组", "肠道菌群"]),
            ("car_t_cell_therapy", "CAR-T细胞疗法是什么",
             "CAR-T是改造患者T细胞攻击癌症的疗法。对某些白血病和淋巴瘤有特效，是癌症治疗革命。",
             "biotech", ["CAR-T", "免疫疗法"]),
            ("antibody_drug_conjugates", "抗体药物偶联物是什么",
             "ADC是连接抗体与毒素的抗癌药。抗体定位癌细胞，毒素杀死它们。精准给药，副作用小。",
             "biotech", ["ADC", "抗癌药"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_space_science(self):
        """空间科学"""
        entries = [
            ("james_webb_telescope", "韦伯望远镜发现什么",
             "韦伯望远镜2021年发射，可观测宇宙最早期星系。发现遥远行星大气成分，分析系外行星是否宜居。",
             "space", ["韦伯望远镜", "JWST"]),
            ("mars_colonization", "火星殖民进展如何",
             "SpaceX等公司计划火星殖民。挑战包括辐射防护、食物生产、能源供应。NASA计划2030年代载人登火星。",
             "space", ["火星殖民", "火星"]),
            ("moon_resources", "月球有什么资源",
             "月球水冰可制火箭燃料，氦-3是潜在核聚变燃料，铁钛矿丰富。月球可作太空探索基地。",
             "space", ["月球资源", "月球"]),
            ("gravitational_waves", "引力波是什么",
             "引力波是时空扭曲产生的涟漪，由爱因斯坦预测。LIGO探测到黑洞合并产生的引力波，开辟天文新窗口。",
             "space", ["引力波", "LIGO"]),
            ("dark_matter_evidence", "暗物质有哪些证据",
             "暗物质证据包括：星系旋转曲线、引力透镜、星系团质量。占宇宙27%，本质未知。",
             "space", ["暗物质", "宇宙"]),
            ("exoplanet_discovery", "系外行星探索进展",
             "开普勒和TESS望远镜已发现5000+系外行星。TRAPPIST-1有7颗类地行星，其中3颗可能宜居。",
             "space", ["系外行星", "宜居行星"]),
            ("space_tourism_status", "太空旅游现状如何",
             "蓝色起源、SpaceX、维珍银河提供亚轨道太空游。票价数十万至数千万美元，正在走向商业化。",
             "space", ["太空旅游"]),
            ("satellite__constellations", "星链等卫星星座影响",
             "SpaceX星链计划发射42000颗卫星。提供全球宽带，但也造成光污染、轨道拥挤等问题。",
             "space", ["星链", "卫星"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_policy_regulation(self):
        """政策法规"""
        entries = [
            ("gdpr_basics", "GDPR是什么",
             "通用数据保护条例是欧盟隐私法律。赋予用户数据访问权、被遗忘权等。违规罚款可达全球营收4%。",
             "policy", ["GDPR", "隐私保护"]),
            ("ai_regulation_eu", "欧盟AI法案内容",
             "欧盟AI法案按风险分级监管。高风险AI需透明度评估，禁止实时生物识别。违规罚款高达3000万欧。",
             "policy", ["AI法规", "欧盟"]),
            ("data_localization", "数据本地化要求",
             "数据本地化要求数据在境内存储。各国法律不同，如俄罗斯、中国有严格规定，影响跨国企业运营。",
             "policy", ["数据本地化", "数据主权"]),
            ("antitrust_tech", "科技巨头反垄断",
             "美国和欧盟对Google、Apple、Meta、亚马逊进行反垄断调查。涉及市场垄断、数据滥用、捆绑销售等。",
             "policy", ["反垄断", "科技监管"]),
            ("cryptocurrency_regulation", "加密货币监管",
             "各国对加密货币监管不同。美国SEC视多为证券，欧盟通过MiCA法案。监管影响行业走向。",
             "policy", ["加密货币", "监管"]),
            ("platform_worker_rights", "零工经济工人权益",
             "Uber、Deliveroo等平台工人权益争议。是否算雇员、福利保障、劳动法适用等问题待解决。",
             "policy", ["零工经济", "平台工人"]),
            ("ai_copyright", "AI生成内容版权问题",
             "AI创作内容的版权归属争议。谁是作者：用户、开发者还是AI？各国法律正在应对这一新问题。",
             "policy", ["AI版权", "版权"]),
            ("digital_tax", "数字税是什么",
             "数字税对跨境数字服务征税。法国等欧盟国家征收GAFA税。美国反对，担心双重征税。",
             "policy", ["数字税", "GAFA"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_finance_economy(self):
        """金融经济"""
        entries = [
            ("defi_intro", "DeFi是什么",
             "去中心化金融DeFi使用智能合约提供借贷、交易等金融服务。无须银行等中间商，收益率更高但风险也大。",
             "finance", ["DeFi", "去中心化金融"]),
            ("stablecoin_types", "稳定币有哪些类型",
             "稳定币锚定美元等资产。法币抵押(USDC)、加密资产抵押(DAI)、算法稳定币(UST已崩盘)。",
             "finance", ["稳定币", "加密货币"]),
            ("central_bank_digital_currency", "央行数字货币是什么",
             "CBDC是央行发行的数字货币。中国数字人民币试点领先，欧美正在推进。改变支付和货币政策。",
             "finance", ["CBDC", "数字货币"]),
            ("esg_investing", "ESG投资是什么",
             "环境、社会、治理ESG投资考虑非财务因素。筛选标准包括碳排放、劳动权益、公司治理等。",
             "finance", ["ESG", "可持续投资"]),
            ("impact_investing", "影响力投资是什么",
             "影响力投资追求正向社会影响的同时获得财务回报。关注教育、医疗、环保等领域。",
             "finance", ["影响力投资", "公益"]),
            ("crowdfunding_types", "众筹有哪些类型",
             "众筹包括：奖励众筹(如Kickstarter)、股权众筹、债权众筹、捐赠众筹。降低创业融资门槛。",
             "finance", ["众筹", "融资"]),
            ("behavioral_economics", "行为经济学是什么",
             "行为经济学研究心理因素如何影响经济决策。揭示人类非理性模式，应用于政策设计。",
             "finance", ["行为经济", "心理学"]),
            ("passive_vs_active_investing", "被动投资vs主动投资",
             "被动投资追踪指数，成本低。主动投资试图跑赢市场，但多数跑输指数。低成本指数基金是主流。",
             "finance", ["指数基金", "投资"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_environment_energy(self):
        """环境能源"""
        entries = [
            ("green_hydrogen", "绿氢是什么",
             "绿氢用可再生能源电解水制取，碳排放为零。是工业脱碳的关键能源，前景广阔。",
             "environment", ["绿氢", "氢能源"]),
            ("direct_air_capture", "直接空气捕获技术",
             "DAC从空气中吸取CO2。Climeworks等公司已商业化，但成本高，每吨数百美元。",
             "environment", ["DAC", "碳捕获"]),
            ("circular_economy", "循环经济是什么",
             "循环经济设计产品可回收再利用。减少资源消耗和废物产生，是可持续发展模式。",
             "environment", ["循环经济", "可持续"]),
            ("biodiversity_loss", "生物多样性危机",
             "地球正在经历第六次大灭绝。人类活动导致物种消失加快。保护生物多样性刻不容缓。",
             "environment", ["生物多样性", "灭绝"]),
            ("ocean_acidification", "海洋酸化问题",
             "海洋吸收CO2导致酸化。威胁珊瑚礁、贝类等海洋生物，破坏海洋生态系统。",
             "environment", ["海洋酸化", "海洋"]),
            ("renewable_energy_storage", "储能技术有哪些",
             "储能包括锂电池、抽水蓄能、压缩空气、液流电池、氢储能。解决可再生能源间歇性问题。",
             "environment", ["储能", "电池"]),
            ("electric_aviation", "电动飞机发展如何",
             "电动飞机可减少航空碳排放。短途电动飞机已试飞，长途混合电动正在研发。",
             "environment", ["电动飞机", "航空"]),
            ("vertical_farming", "垂直农业是什么",
             "垂直农业在高层建筑中种植作物。节约土地用水，全年生产，但能耗高。城市农业新方式。",
             "environment", ["垂直农业", "城市农业"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_psychology_mind(self):
        """心理心智"""
        entries = [
            ("mental_health_parity", "心理健康平等是什么",
             "心理健康平等要求心理健康与身体健康享有同等重视和治疗机会。消除心理疾病的污名化。",
             "psychology", ["心理健康", "心理疾病"]),
            ("digital_wellbeing", "数字健康如何维护",
             "数字健康包括：限制屏幕时间、睡前一小时不用手机、社交媒体戒断。保持线上线下平衡。",
             "psychology", ["数字健康", "屏幕时间"]),
            ("cognitive_biases", "常见认知偏差有哪些",
             "认知偏差如确认偏误、可得性启发、损失厌恶等。影响判断决策，了解它们有助于理性思考。",
             "psychology", ["认知偏差", "心理学"]),
            ("growth_mindset", "成长型思维是什么",
             "成长型思维相信能力可通过努力提升。与固定型思维相对，能促进学习面对挑战。",
             "psychology", ["成长型思维", "思维模式"]),
            ("gut_brain_connection", "肠脑轴是什么",
             "肠道微生物通过肠脑轴影响情绪认知。肠道被称为第二大脑，与焦虑、抑郁相关。",
             "psychology", ["肠脑轴", "微生物组"]),
            ("flow_state", "心流状态是什么",
             "心流是全神贯注投入活动的最佳状态。时间感消失，享受过程。工作和兴趣中都可进入。",
             "psychology", ["心流", "专注"]),
            ("loneliness_epidemic", "孤独感问题",
             "现代社会中孤独感普遍，危害健康如每天吸15支烟。社交媒体不能替代真实连接。",
             "psychology", ["孤独", "社交"]),
            ("sleep_hygiene", "睡眠卫生原则",
             "睡眠卫生：固定作息、黑暗安静环境、午后避咖啡因、睡前放松。良好睡眠是心理健康基础。",
             "psychology", ["睡眠卫生", "睡眠"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_education_innovation(self):
        """教育创新"""
        entries = [
            ("adaptive_learning", "自适应学习是什么",
             "自适应学习系统根据学生表现调整内容和难度。AI使个性化教育规模化成为可能。",
             "education", ["自适应学习", "AI教育"]),
            ("mooc_limitations", "慕课的局限有哪些",
             "慕课(MOOC)课程完成率低，通常仅3-5%。缺乏互动动力、筛选机制，替代而非颠覆传统教育。",
             "education", ["慕课", "在线教育"]),
            ("stem_emphasis", "STEM教育重要性",
             "STEM培养科学、技术、工程、数学能力。是未来就业关键，但也需要人文艺术平衡。",
             "education", ["STEM", "教育"]),
            ("project_based_learning", "项目式学习是什么",
             "项目式学习通过完成实际项目学习。培养问题解决、协作能力，与现实世界连接。",
             "education", ["项目式学习", "PBL"]),
            ("education_ai_tools", "教育AI工具有哪些",
             "教育AI包括：智能辅导、作业批改、内容生成、学习分析。ChatGPT等工具正在变革教育。",
             "education", ["教育AI", "AI工具"]),
            ("life_long_learning", "终身学习为何重要",
             "技术变革加速，技能半衰期缩短。终身学习成为必需，在线平台使持续教育更便捷。",
             "education", ["终身学习", "持续教育"]),
            ("competency_based_education", "能力本位教育是什么",
             "能力本位教育关注实际能力掌握而非上课时间。学生按自己节奏达标，更灵活。",
             "education", ["能力本位", "教育"]),
            ("social_emotional_learning", "社会情感学习是什么",
             "SEL培养自我意识、人际关系、决策能力。对学业成功和人生幸福至关重要。",
             "education", ["SEL", "社会情感"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_3d_printing(self):
        """3D打印"""
        entries = [
            ("3d_printing_medicine", "3D打印在医学的应用",
             "3D打印用于打印假肢、植入物、器官模型甚至活性组织。定制化医疗更便捷。",
             "frontier", ["3D打印", "医学"]),
            ("construction_3d_printing", "建筑3D打印",
             "建筑3D打印用机器人逐层打印建筑结构。节省人工、速度快，可打印复杂形状。",
             "frontier", ["建筑3D打印"]),
            ("food_3d_printing", "食品3D打印",
             "3D打印食品可创造复杂形状个性化营养。NASA用于宇航员食物，也用于高端餐饮。",
             "frontier", ["食品打印"]),
            ("metal_3d_printing", "金属3D打印",
             "金属3D打印如SLM、DMLS技术可直接打印金属零件。用于航空、汽车、工具制造。",
             "frontier", ["金属打印"]),
            ("bio打印_human_organs", "生物3D打印器官进展",
             "生物3D打印活性组织器官是圣杯级别挑战。已打印皮肤、软骨、血管，心脏肾脏等复杂器官在路上。",
             "frontier", ["生物打印", "器官打印"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_blockchain_web3(self):
        """区块链Web3"""
        entries = [
            ("web3_concept", "Web3是什么",
             "Web3是去中心化互联网概念。用户拥有数据和资产，通过区块链和代币经济实现。",
             "web3", ["Web3", "区块链"]),
            ("nft_beyond_art", "NFT应用领域",
             "NFT不仅是数字艺术，还可代表门票、证书、游戏道具、房产等资产。提供数字所有权证明。",
             "web3", ["NFT", "数字资产"]),
            ("dao_governance", "DAO是什么",
             "去中心化自治组织DAO用智能合约治理。没有中央机构，成员投票决策。新型组织形式。",
             "web3", ["DAO", "治理"]),
            ("metaverse_economy", "元宇宙经济",
             "元宇宙中数字资产有真实经济价值。虚拟土地、物品、头像可买卖，形成新经济。",
             "web3", ["元宇宙经济"]),
            ("decentralized_identity", "去中心化身份",
             "DID用户自己控制数字身份，无需中心化平台。保护隐私，是Web3身份基础设施。",
             "web3", ["DID", "身份"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_medical_health(self):
        """医疗健康"""
        entries = [
            ("glp1_weight_loss", "GLP-1减肥药",
             "Ozempic、Wegovy等GLP-1激动剂可显著减重。通过抑制食欲实现，是肥胖治疗革命。",
             "health", ["减肥药", "GLP-1"]),
            ("mental_health_crisis", "心理健康危机",
             "全球心理健康问题激增。抑郁焦虑增加，年轻人尤甚。需更多资源投入和治疗可及性。",
             "health", ["心理健康", "抑郁"]),
            ("sleep_tracking_accuracy", "睡眠追踪准确性",
             "智能手表睡眠追踪不精确，主要靠活动心率估算。深睡浅睡数据参考价值有限。",
             "health", ["睡眠追踪", " wearables"]),
            ("intermittent_fasting", "间歇性禁食科学",
             "间歇性禁食可改善代谢、延长寿命。研究显示16:8等模式有效，但不适合所有人。",
             "health", ["间歇性禁食", "断食"]),
            ("exercise_longevity", "运动延寿证据",
             "每周150分钟中等强度运动可延长寿命3-5年。运动是最有效的健康投资。",
             "health", ["运动", "延寿"]),
            ("genetic_testing_limits", "基因检测局限性",
             "消费级基因检测风险有限，主要提供祖源和健康倾向。不能替代诊断，阳性需专业确认。",
             "health", ["基因检测"]),
            ("ai_diagnostics", "AI诊断进展",
             "AI在影像诊断(如眼底、皮肤癌)已达专家水平。辅助诊断提高效率，但不能替代医生。",
             "health", ["AI诊断", "人工智能"]),
            ("telemedicine_future", "远程医疗未来",
             "远程医疗扩大医疗可及性，但不可替代物理检查。混合模式是未来趋势。",
             "health", ["远程医疗"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_advanced_manufacturing(self):
        """先进制造"""
        entries = [
            ("industry_robotics", "工业机器人发展",
             "工业机器人更灵活协作(co-bot)，配备视觉AI。可与人类安全共事，胜任复杂任务。",
             "manufacturing", ["工业机器人"]),
            ("digital_twin", "数字孪生是什么",
             "数字孪生是物理系统的虚拟复制。用于仿真优化、预测维护、远程监控。工业4.0核心。",
             "manufacturing", ["数字孪生", "工业4.0"]),
            ("additive_mfg_limits", "增材制造局限",
             "3D打印等增材制造适合小批量定制，大规模生产成本仍高于传统工艺。材料限制也存在。",
             "manufacturing", ["增材制造"]),
            ("supply_chain_reshoring", "供应链回流趋势",
             "疫情暴露供应链脆弱，企业将产能回迁或近岸。成本更高但更安全抗风险。",
             "manufacturing", ["供应链", "回流"]),
            ("semiconductor_china", "半导体竞争",
             "美国限制中国获取先进芯片，推动芯片自主。中美科技竞争核心，影响深远。",
             "manufacturing", ["半导体", "芯片"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_social_media_marketing(self):
        """社交媒体营销"""
        entries = [
            ("influencer_economy", "网红经济规模",
             "网红营销已成庞大产业。品牌与KOL合作推广产品。选对网红匹配度比粉丝数更重要。",
             "marketing", ["网红经济", "KOL"]),
            ("algorithmic_feed", "算法信息流影响",
             "社交媒体算法放大极端内容以获取 engagement。了解算法机制，避免被操控。",
             "marketing", ["算法", "社交媒体"]),
            ("creator_economy_platforms", "创作者经济平台",
             "YouTube、TikTok、Patreon等平台支持创作者变现。内容创业成可行职业路径。",
             "marketing", ["创作者经济"]),
            ("brand_community", "品牌社区价值",
             "品牌社区培养忠实用户，产生口碑传播。耐克、星巴克等品牌社区建设成功。",
             "marketing", ["品牌社区"]),
            ("ugc_importance", "用户生成内容价值",
             "UGC比品牌自产内容更可信。鼓励用户评价、晒图是有效营销策略。",
             "marketing", ["UGC", "用户评价"]),
        ]
        for e in entries:
            self._add_entry(e[0], e[1], e[2], e[3], e[4])

    def _add_law_rights(self):
        """法律权益"""
        entries = [
            ("right_to_digital_privacy", "数字隐私权",
             "数字隐私是基本人权。个人数据收集使用需知情同意。法律如GDPR赋予用户控制权。",
             "law", ["数字隐私", "数据权"]),
            ("intellectual_property_ai", "AI创作知识产权",
             "AI生成内容版权归属争议。人类参与程度、是否具有独创性影响判定，法律待明确。",
             "law", ["AI版权", "知识产权"]),
            ("liability_autonomous_vehicles", "自动驾驶事故责任",
             "自动驾驶事故责任归属：车主、厂商还是软件？法律需跟上技术发展明确责任。",
             "law", ["自动驾驶", "责任"]),
            ("gig_worker_classification", "零工工人法律地位",
             "平台工人是雇员还是独立承包商？各国法律判决不同，影响福利和权益保障。",
             "law", ["零工", "劳动法"]),
            ("health_data_rights", "健康数据权利",
             "患者有权访问控制自己健康数据。HIPAA等法律保护但执行有挑战。",
             "law", ["健康数据", "隐私"]),
            ("environmental_law_trends", "环境法趋势",
             "气候变化诉讼增多，成功案例有历史意义。企业面临更大环境责任压力。",
             "law", ["环境法", "气候变化"]),
            ("consumer_protection_ai", "AI消费者保护",
             "AI决策影响贷款、保险、就业。消费者有权要求解释和人工复核。",
             "law", ["AI消费者", "保护"]),
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
    print("Stage 16: 知识库扩展")
    print(f"{'='*70}")

    kb = Stage16KnowledgeBase()
    stats = kb.get_stats()

    print(f"\n{'='*70}")
    print("Stage 16 知识库统计")
    print(f"{'='*70}")
    print(f"  总条目: {stats['total_entries']}")
    print(f"  按类别分布:")
    for cat, count in sorted(stats.get('by_category', {}).items()):
        print(f"    {cat}: {count}")

    print(f"\n{'='*70}")
    print("新增领域检索测试")
    print(f"{'='*70}")

    test_queries = [
        "量子计算是什么",
        "mRNA疫苗的原理",
        "火星殖民进展",
        "GDPR是什么",
        "DeFi和传统金融区别",
        "绿氢能源",
        "如何维护数字心理健康",
        "3D打印器官",
        "Web3和Web2区别",
        "AI伦理问题",
    ]

    for q in test_queries:
        results = kb.retrieve(q, top_k=2)
        if results and results[0]['relevance_score'] >= 0.2:
            print(f"\n[{results[0]['relevance_score']:.2f}] {q}")
            print(f"  → {results[0]['question']}")
        else:
            print(f"\n[0.00] {q} → 无匹配")

    print(f"\n{'='*70}")
    print("Stage 16 测试完成")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
